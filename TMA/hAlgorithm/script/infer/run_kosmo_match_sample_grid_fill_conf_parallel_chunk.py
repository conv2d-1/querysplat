import os
import sys

sys.path.append(os.getcwd())

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from run_kosmo_match_sample_edge import (
    get_fisheye_mask,
    get_normalized_grid,
    save_matches_hloc_nested_format,
    set_seed,
    tensor_to_pil,
    to_pixel_coordinates,
)
from tqdm import tqdm

from hAlgorithm.modules.utils.parallel_utils import parallel_execution
from hAlgorithm.utils import (
    config_merge_args,
    file2dict,
    instantiate_from_config,
    parse_unknown,
)

_IMAGE_DIR_CANDIDATES = ("images", "image")
_MASK_DIR_CANDIDATES = ("masks", "mask")


def _find_subdir(scene_dir: Path, candidates: tuple[str, ...]) -> Path:
    """Return the first existing subdirectory from *candidates*, or the first candidate as fallback."""
    for name in candidates:
        p = scene_dir / name
        if p.is_dir():
            print(f"Using directory: {p}")
            return p
    fallback = scene_dir / candidates[0]
    print(f"No candidate directory found, fallback to: {fallback}")
    return fallback


def load_model(cfg, args, device):
    model = instantiate_from_config(cfg["model"]).to(device)
    model.eval()
    if args.load_from is not None:
        load_from = args.load_from
        if load_from == "latest":
            load_from = os.path.join(os.path.dirname(args.config), "checkpoint/latest/ckpt.pth")
        elif load_from == "best":
            load_from = os.path.join(os.path.dirname(args.config), "checkpoint/best/ckpt.pth")
        model.load_checkpoint(ckpt_path=load_from)
    return model


def sample_grid_and_fill(
    preds: dict[str, torch.Tensor],
    grid_step: int = 10,
    overlap_threshold: float = 0.5,
    conf_threshold: float = None,
    n_sample: int = 0,
    seed=None,
):
    """Grid-sample matches, then optionally random-fill up to *n_sample*.

    Fuses the former ``sample_with_grid`` and ``random_sample_fill`` into one
    pass so that tensor extraction, grid construction and validity masks are
    computed only once.

    Returns (matches, overlaps, precision_0, precision_1, n_grid, n_fill).
    *n_fill* is 0 when filling was not needed / not requested.
    """
    warp = preds["warp_AB"][0]          # (H, W, 2)
    overlap_AB = preds["overlap_AB"][0]  # (H, W, 1)
    precision_AB = preds.get("precision_AB")
    if precision_AB is not None:
        precision_AB = precision_AB[0]

    H, W, _ = warp.shape
    device = warp.device

    # ---- build full normalised grid once ----
    xs_full = (2 * torch.arange(W, device=device).float() + 1 - W) / W
    ys_full = (2 * torch.arange(H, device=device).float() + 1 - H) / H
    gy_full, gx_full = torch.meshgrid(ys_full, xs_full, indexing="ij")
    full_grid = torch.stack([gx_full, gy_full], dim=-1)  # (H, W, 2)

    all_matches = torch.cat([full_grid, warp], dim=-1).reshape(-1, 4)  # (H*W, 4)
    overlap_flat = overlap_AB.reshape(-1)

    # ---- global validity mask (shared by grid & fill) ----
    valid = all_matches[:, 2:].abs().amax(dim=-1).le(1 - 1 / H)
    valid &= overlap_flat > overlap_threshold
    if conf_threshold is not None and precision_AB is not None:
        valid &= precision_AB[..., -1].reshape(-1) > conf_threshold

    # ---- grid indices ----
    ys = torch.arange(0, H, grid_step, device=device)
    xs = torch.arange(0, W, grid_step, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid_flat_inds = gy.reshape(-1) * W + gx.reshape(-1)

    # ---- grid sampling ----
    grid_valid_mask = valid[grid_flat_inds]
    grid_sel = grid_flat_inds[grid_valid_mask]

    matches = all_matches[grid_sel]
    overlaps = overlap_flat[grid_sel]

    precision_0 = precision_1 = None
    # prec_shape = precision_AB.shape[2:] if precision_AB is not None else None
    # if precision_AB is not None:
    #     prec = precision_AB.reshape(-1, *prec_shape)[grid_sel]
    #     if prec.ndim == 3 and prec.shape[-1] == 2 and prec.shape[-2] == 2:
    #         precision_0 = prec[:, 0]
    #         precision_1 = prec[:, 1]

    n_grid = matches.shape[0]
    n_fill = 0

    # ---- random fill ----
    if n_sample > 0 and n_grid < n_sample:
        fill_weight = overlap_flat.clone()
        fill_weight[~valid] = 0.0
        fill_weight[grid_flat_inds] = 0.0

        num_to_sample = min(n_sample - n_grid, int((fill_weight > 0).sum().item()))
        if num_to_sample > 0:
            set_seed(seed)
            fill_inds = torch.multinomial(fill_weight, num_to_sample, replacement=False)

            matches = torch.cat([matches, all_matches[fill_inds]], dim=0)
            overlaps = torch.cat([overlaps, overlap_flat[fill_inds]], dim=0)
            n_fill = num_to_sample

            # if precision_AB is not None:
            #     fill_prec = precision_AB.reshape(-1, *prec_shape)[fill_inds]
            #     if fill_prec.ndim == 3 and fill_prec.shape[-1] == 2 and fill_prec.shape[-2] == 2:
            #         p0, p1 = fill_prec[:, 0], fill_prec[:, 1]
            #         precision_0 = torch.cat([precision_0, p0], dim=0) if precision_0 is not None else p0
            #         precision_1 = torch.cat([precision_1, p1], dim=0) if precision_1 is not None else p1

    return matches, overlaps, precision_0, precision_1, n_grid, n_fill


def process_single_pair(
    model,
    device,
    model_infer_size,
    grid_step,
    overlap_threshold,
    n_sample,
    img_A_name,
    img_B_name,
    img_A,
    img_B,
    mask_A_np,
    mask_B_np,
    output_dir,
    vis,
    fisheye_meta=None,
    input_images=None,
    W_A=None,
    H_A=None,
    W_B=None,
    H_B=None,
    seed=None,
    conf_threshold=None,
):
    """Run inference + grid sampling for one pair. Returns (pair_result, image_sizes)."""
    mask_A = mask_B = None
    if mask_A_np is not None:
        mask_A = torch.from_numpy(mask_A_np).float().to(device)
        if mask_A.ndim == 3:
            mask_A = mask_A.mean(dim=-1)
        mask_A = mask_A > 0
    if mask_B_np is not None:
        mask_B = torch.from_numpy(mask_B_np).float().to(device)
        if mask_B.ndim == 3:
            mask_B = mask_B.mean(dim=-1)
        mask_B = mask_B > 0

    input_images = (input_images / 127.5 - 1).unsqueeze(0).to(device)
    meta_data = {
        "frames": [1],
        "views": [2],
        "input_width": [model_infer_size],
        "input_height": [model_infer_size],
    }

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            results = model.model(input_images, meta_data=meta_data, bidirectional=False)

    if "final" in results["match"]:
        preds = results["match"]["final"]
    elif "refiner" in results["match"]:
        preds = results["match"]["refiner"]
    else:
        preds = results["match"]["coarse"]

    confidence_AB = preds.pop("confidence_AB")
    overlap_AB = confidence_AB[..., :1].sigmoid()
    preds["overlap_AB"] = overlap_AB

    for key, val in preds.items():
        preds[key] = val.squeeze(1).float()

    if mask_A is not None and "overlap_AB" in preds:
        if mask_A.shape != preds["overlap_AB"].shape[1:3]:
            mask_A = (
                torch.nn.functional.interpolate(
                    mask_A[None, None].float(),
                    size=preds["overlap_AB"].shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                > 0
            )
        preds["overlap_AB"] = preds["overlap_AB"] * mask_A[None, ..., None].float()

    if mask_B is not None and "overlap_BA" in preds:
        if mask_B.shape != preds["overlap_BA"].shape[1:3]:
            mask_B = (
                torch.nn.functional.interpolate(
                    mask_B[None, None].float(),
                    size=preds["overlap_BA"].shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                > 0
            )
        preds["overlap_BA"] = preds["overlap_BA"] * mask_B[None, ..., None].float()
    
    preds["precision_AB"] = confidence_AB[..., 1:4].squeeze(1)
    matches, overlaps, precision_AB, precision_BA, n_grid, n_fill = sample_grid_and_fill(
        preds,
        grid_step=grid_step,
        overlap_threshold=overlap_threshold,
        conf_threshold=conf_threshold,
        n_sample=n_sample,
        seed=seed,
    )
    if n_fill > 0:
        print(f"[{img_A_name} - {img_B_name}] grid={n_grid}, random_fill={n_fill}, total={matches.shape[0]}")

    kptsA, kptsB = to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    if vis:
        _save_visualization(
            preds,
            img_A,
            img_B,
            img_A_name,
            img_B_name,
            kptsA,
            kptsB,
            H_A,
            W_A,
            output_dir,
            device,
            seed=seed,
        )

    if fisheye_meta is not None:
        offset = torch.tensor(
            [fisheye_meta["crop_left"], fisheye_meta["crop_top"]],
            device=kptsA.device,
            dtype=kptsA.dtype,
        )
        kptsA = kptsA + offset
        kptsB = kptsB + offset

    pair_result = {
        "kptsA": kptsA.cpu().numpy(),
        "kptsB": kptsB.cpu().numpy(),
    }
    if fisheye_meta is not None:
        W_orig_A, H_orig_A = fisheye_meta["orig_A"]
        W_orig_B, H_orig_B = fisheye_meta["orig_B"]
        sizes = {
            img_A_name: np.array([H_orig_A, W_orig_A]),
            img_B_name: np.array([H_orig_B, W_orig_B]),
        }
    else:
        sizes = {
            img_A_name: np.array([H_A, W_A]),
            img_B_name: np.array([H_B, W_B]),
        }
    return img_A_name, img_B_name, pair_result, sizes


def _save_visualization(
    preds,
    img_A,
    img_B,
    img_A_name,
    img_B_name,
    kptsA,
    kptsB,
    H_A,
    W_A_orig,
    output_dir,
    device,
    seed=None,
):
    img_A_name_vis = Path(img_A_name).parent.name + "-" + Path(img_A_name).name[:-4]
    img_B_name_vis = Path(img_B_name).parent.name + "-" + Path(img_B_name).name[:-4]
    save_path = f"{output_dir}/kosmo_match_vis/{img_A_name_vis}_{img_B_name_vis}.jpg"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    warp_AB = preds["warp_AB"][0]
    overlap_AB = preds["overlap_AB"][0]

    x1 = (torch.tensor(np.array(img_A)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(img_B)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(x2[None], warp_AB[None], mode="bilinear", align_corners=False)[0]
    overlap = overlap_AB.squeeze()
    white_im = torch.ones((im2_transfer_rgb.shape[-2], im2_transfer_rgb.shape[-1]), device=device)
    vis_im = overlap * im2_transfer_rgb + (1 - overlap) * white_im
    tensor_to_pil(vis_im).save(save_path)

    precision_AB_map = preds.get("precision_AB")
    if precision_AB_map is not None:
        conf = precision_AB_map[0, ..., -1].sigmoid()  # (H, W), 0~1
        red = torch.tensor([1.0, 0.0, 0.0], device=device).reshape(3, 1, 1)
        vis_im_conf = conf * vis_im + (1 - conf) * red
        conf_save_path = os.path.splitext(save_path)[0] + "_conf.jpg"
        tensor_to_pil(vis_im_conf).save(conf_save_path)

    img_A_np = np.array(img_A)
    img_B_np = np.array(img_B)
    combined_orig = np.concatenate([img_A_np, img_B_np], axis=1).copy()
    W_A_px = img_A_np.shape[1]
    num_matches = kptsA.shape[0]
    if num_matches > 0:
        num_lines = min(30, num_matches)
        set_seed(seed)
        indices = np.random.choice(num_matches, size=num_lines, replace=False)

        for idx in indices:
            x_a, y_a = kptsA[idx].cpu().numpy()
            x_b, y_b = kptsB[idx].cpu().numpy()
            pt1 = (int(round(x_a)), int(round(y_a)))
            pt2 = (int(round(x_b + W_A_px)), int(round(y_b)))
            h_a, w_a = img_A_np.shape[:2]
            h_b, w_b = img_B_np.shape[:2]
            pt1 = (np.clip(pt1[0], 0, w_a - 1), np.clip(pt1[1], 0, h_a - 1))
            pt2 = (np.clip(pt2[0], W_A_px, W_A_px + w_b - 1), np.clip(pt2[1], 0, h_b - 1))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(combined_orig, pt1, pt2, color=color, thickness=5)
        match_vis_path = os.path.splitext(save_path)[0] + "_matches.png"
        Image.fromarray(combined_orig).save(match_vis_path)


def worker_fn(args, cfg, all_pairs):

    seed = args.seed
    num_workers = args.num_workers

    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    model = load_model(cfg, args, device)

    scene_dir = Path(args.scene_dir)
    image_dir = _find_subdir(scene_dir, _IMAGE_DIR_CANDIDATES)
    mask_dir = _find_subdir(scene_dir, _MASK_DIR_CANDIDATES)
    output_dir = os.path.join(scene_dir, args.output)

    fisheye_crop = None
    fisheye_mask_bool = None
    if args.fisheye_mode:
        fisheye_crop = tuple(args.fisheye_crop)
        H_crop = fisheye_crop[3] - fisheye_crop[1]
        W_crop = fisheye_crop[2] - fisheye_crop[0]
        fisheye_mask_bool = get_fisheye_mask(
            H_crop,
            W_crop,
            args.fisheye_mask_shrink,
        ).numpy()

    target_size = (args.process_res, args.process_res)

    # Step 1: 提取所有唯一图像名
    unique_images = set()
    for pair_line in all_pairs:
        img_A_name, img_B_name = pair_line.strip().split(" ")
        unique_images.add(img_A_name)
        unique_images.add(img_B_name)
    unique_images = list(unique_images)
    print(f"{len(all_pairs)} pairs -> {len(unique_images)} unique images")

    # Step 2: 读取单张图像的函数
    def load_single_image(img_name):
        img_np = cv2.cvtColor(cv2.imread(str(image_dir / img_name)), cv2.COLOR_BGR2RGB)

        mask_path = mask_dir / (os.path.splitext(img_name)[0] + ".png")
        mask_np_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.exists() else None

        if fisheye_crop is not None:
            left, top, right, bottom = fisheye_crop
            H_orig, W_orig = img_np.shape[:2]
            img_np = img_np[top:bottom, left:right]
            if mask_np_raw is not None:
                mask_np_raw = mask_np_raw[top:bottom, left:right]
            orig_size = (W_orig, H_orig)
        else:
            H, W = img_np.shape[:2]
            orig_size = (W, H)

        H, W = img_np.shape[:2]
        img_vis = img_np

        if fisheye_mask_bool is not None:
            img_np = np.where(fisheye_mask_bool[..., None], img_np, 0)

        img_resized = cv2.resize(img_np, target_size, interpolation=cv2.INTER_LINEAR)

        mask_np = None
        if mask_np_raw is not None:
            mask_arr = mask_np_raw
            if fisheye_crop is None:
                mask_arr = mask_arr[300:2700, 800:3200]
            mask_np = cv2.resize(mask_arr, target_size, interpolation=cv2.INTER_LINEAR)

        return img_name, img_vis, img_resized, mask_np, W, H, orig_size

    # Step 3: 并行读取所有唯一图像
    image_data_list = parallel_execution(
        unique_images,
        action=load_single_image,
        num_processes=num_workers,
        print_progress=True,
        sequential=False,
        desc=f"load images",
    )
    image_dict = {data[0]: data[1:] for data in image_data_list}

    # Step 4: 组装 pairs (no edge mask needed for grid sampling)
    def assemble_pair(idx):
        pair_line = all_pairs[idx]
        img_A_name, img_B_name = pair_line.strip().split(" ")

        img_A_vis, img_A_resized, mask_A_np, W_A, H_A, orig_A = image_dict[img_A_name]
        img_B_vis, img_B_resized, mask_B_np, W_B, H_B, orig_B = image_dict[img_B_name]

        input_images = torch.from_numpy(np.stack([img_A_resized, img_B_resized], axis=0)).permute(0, 3, 1, 2).float()

        fisheye_meta = None
        if fisheye_crop is not None:
            fisheye_meta = {
                "orig_A": orig_A,
                "orig_B": orig_B,
                "crop_left": fisheye_crop[0],
                "crop_top": fisheye_crop[1],
            }

        return img_A_name, img_B_name, img_A_vis, img_B_vis, mask_A_np, mask_B_np, fisheye_meta, input_images, W_A, H_A, W_B, H_B

    # Step 5: 分 chunk 组装 pairs + 推理，避免一次性 assemble 所有 pairs 导致内存爆炸
    local_pair_results = {}
    local_image_size = {}
    chunk_size = args.chunk_size
    total = len(all_pairs)

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_indices = list(range(chunk_start, chunk_end))
        print(f"Assembling pairs [{chunk_start}:{chunk_end}] / {total}")

        outputs = parallel_execution(
            chunk_indices,
            action=assemble_pair,
            num_processes=num_workers,
            print_progress=True,
            sequential=False,
            desc=f"assemble pairs [{chunk_start}:{chunk_end}]",
        )
        img_A_name_list, img_B_name_list, img_A_list, img_B_list, mask_A_np_list, mask_B_np_list, fisheye_meta_list, input_images_list, W_A_list, H_A_list, W_B_list, H_B_list = zip(
            *outputs
        )

        chunk_len = len(chunk_indices)
        pbar = tqdm(range(chunk_len), total=chunk_len, desc=f"inference [{chunk_start}:{chunk_end}]")

        for i in pbar:
            img_A_name = img_A_name_list[i]
            img_B_name = img_B_name_list[i]
            img_A = img_A_list[i]
            img_B = img_B_list[i]
            mask_A_np = mask_A_np_list[i]
            mask_B_np = mask_B_np_list[i]
            fisheye_meta = fisheye_meta_list[i]
            input_images = input_images_list[i]

            W_A = W_A_list[i]
            H_A = H_A_list[i]
            W_B = W_B_list[i]
            H_B = H_B_list[i]

            img_A_name, img_B_name, pair_result, sizes = process_single_pair(
                model,
                device,
                args.process_res,
                args.grid_step,
                args.overlap_threshold,
                args.n_sample,
                img_A_name,
                img_B_name,
                img_A,
                img_B,
                mask_A_np,
                mask_B_np,
                output_dir,
                args.vis,
                fisheye_meta=fisheye_meta,
                input_images=input_images,
                W_A=W_A,
                H_A=H_A,
                W_B=W_B,
                H_B=H_B,
                seed=seed,
                conf_threshold=args.conf_threshold,
            )

            if img_A_name not in local_pair_results:
                local_pair_results[img_A_name] = {}
            local_pair_results[img_A_name][img_B_name] = pair_result
            local_image_size.update(sizes)

        pbar.close()
        del outputs

    return local_pair_results, local_image_size


def main():
    parser = argparse.ArgumentParser(description="Parallel Kosmo matching with grid-based sampling.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--load_from", default=None, help="Path of checkpoint to load.")
    parser.add_argument(
        "--scene_dir",
        "-s",
        type=str,
        required=True,
        help="Path to the scene directory",
    )
    parser.add_argument(
        "--pair",
        "-p",
        type=str,
        default="pose_pair.txt",
        help="Path to the image pair file.",
    )
    parser.add_argument("--grid_step", type=int, default=10,
                        help="Grid sampling step on the warp field. "
                             "With process_res=840, step=10 gives ~84x84=7056 candidate points.")
    parser.add_argument("--overlap_threshold", type=float, default=0.5,
                        help="Minimum overlap_AB to keep a grid point.")
    parser.add_argument("--n_sample", type=int, default=0,
                        help="Target total sample count. If grid yields fewer points, "
                             "random confidence-weighted sampling fills up to n_sample. "
                             "0 means no fill (grid-only).")
    parser.add_argument("--conf_threshold", type=float, default=None,
                        help="Minimum precision confidence to keep a match point. "
                             "None means no confidence filtering.")
    parser.add_argument("--process_res", type=int, default=840)
    parser.add_argument("--output", "-o", type=str, default="kosmo_match")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--fisheye_mode", action="store_true", help="Enable fisheye crop + elliptical mask mode.")
    parser.add_argument("--fisheye_crop", type=int, nargs=4, default=[500, 0, 3500, 3000], help="Crop box (left, top, right, bottom) for fisheye images.")
    parser.add_argument("--fisheye_mask_shrink", type=int, default=100, help="Elliptical mask shrinkage in pixels.")
    parser.add_argument("--num_workers", type=int, default=8, help="Total worker processes.")
    parser.add_argument("--prefetch_size", type=int, default=100, help="Prefetch queue depth per worker.")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Chunk size for pair assembly to limit memory usage.")
    parser.add_argument("--seed", type=int, default=42)

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)

    print(f"grid_step: {args.grid_step}, overlap_threshold: {args.overlap_threshold}, n_sample: {args.n_sample}, conf_threshold: {args.conf_threshold}")

    if args.load_from in ["None", "none", "null"]:
        args.load_from = None

    scene_dir = Path(args.scene_dir)
    image_pair_path = scene_dir / args.pair

    with open(image_pair_path, "r") as f:
        all_pairs = f.readlines()

    # 过滤重复 pairs：保留 A-B，丢弃 B-A
    seen_pairs = set()
    filtered_pairs = []
    for pair_line in all_pairs:
        parts = pair_line.strip().split(" ")
        if len(parts) != 2:
            continue
        img_A, img_B = parts
        pair_key = tuple(sorted([img_A, img_B]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            filtered_pairs.append(pair_line)
    print(f"Filtered pairs: {len(all_pairs)} -> {len(filtered_pairs)}")

    if args.vis:
        filtered_pairs = filtered_pairs[:10]

    pair_results, image_size = worker_fn(args, cfg, filtered_pairs)

    print("Processing Pair Results ...")
    save_matches_hloc_nested_format(pair_results, out_dir=args.output, image_size=image_size)
    print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
