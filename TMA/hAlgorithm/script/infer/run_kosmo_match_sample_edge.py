import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path
from PIL import Image
import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import cv2
import torch
import random
from hAlgorithm.utils import (
    config_merge_args,
    file2dict,
    instantiate_from_config,
    get_obj_from_str,
    parse_unknown,
)

def set_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # 当前 GPU
        torch.cuda.manual_seed_all(seed)  # 所有 GPU
        np.random.seed(seed)  # NumPy
        random.seed(seed)  # Python 原生
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def numpy_to_pil(x: np.ndarray):
    assert x.dtype in [np.float32, np.uint8]
    if x.dtype == np.float32:
        assert x.min() >= 0.0 and x.max() <= 1.0
        x *= 255
        x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x):
    x = x.clone().detach().cpu().numpy()
    if len(x.shape) == 3:
        x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def quantize(pt, eps=1e-3):
    return tuple(np.round(pt / eps).astype(np.int64))


def safe_hdf5_name(name):
    """确保名称可用于 HDF5 group/dataset"""
    # 替换非法字符（HDF5 不允许 / \ 等）
    name = name.replace("/", "-").replace("\\", "-")
    return name

def get_normalized_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=overload_device or 'cuda')
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def kde(x: torch.Tensor, std: float = 0.1, half: bool = True) -> torch.Tensor:
    # use a gaussian kernel to estimate density
    if half:
        x = x.half()
    scores = (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density


def sample_with_edge(
    preds: dict[str, torch.Tensor],
    num_corresp: int,
    edge_mask: torch.Tensor = None,      # shape (H, W, 2)
    edge_sample_num: int = 0,
    seed = None,
):
    warp = preds["warp_AB"]
    confidence_AB = preds["overlap_AB"]
    precision_AB = preds["precision_AB"] if "precision_AB" in preds else None

    warp = warp[0]
    confidence_AB = confidence_AB[0].reshape(-1)
    if precision_AB is not None:
        precision_AB = precision_AB[0]

    H_A, W_A, two = warp.shape
    grid = get_normalized_grid(1, H_A, W_A)[0]  # (H, W, 2), normalized [-1, 1]
    matches_AB = torch.cat((grid, warp), dim=-1).reshape(-1, 4)

    matches = matches_AB
    confidence = confidence_AB.reshape(-1)
    if precision_AB is not None:
        precision = precision_AB.reshape(-1, 2, 2)
    else:
        precision = None

    # --- Step 1: 原始采样（不变）---
    expansion_factor = 4
    # Mask out matches that go outside [-1, 1] (i.e., invalid)
    valid_mask = matches.abs().amax(dim=-1).le(1 - 1 / H_A).float()
    confidence = confidence * valid_mask

    # 防止所有 confidence 为 0
    if confidence.sum() == 0:
        confidence = torch.ones_like(confidence)
    
    num_to_sample = min(expansion_factor * num_corresp, len(confidence))
    set_seed(seed)
    corresp_inds = torch.multinomial(confidence, num_to_sample, replacement=False)
    sampled_matches = matches[corresp_inds]
    sampled_confidence = confidence[corresp_inds]
    if precision is not None:
        sampled_precision = precision[corresp_inds]
    else:
        sampled_precision = None

    # --- Step 2: 边缘引导采样（新增）---
    edge_matches = []
    edge_confidences = []
    edge_precisions = []

    if edge_sample_num > 0 and edge_mask is not None:
        # edge_mask: (H, W, 2) -> [edge_A, edge_B]
        edge_A = edge_mask[:, :, 0]  # (H, W)
        edge_B = edge_mask[:, :, 1]  # (H, W)

        # 将 matches 的归一化坐标转为像素坐标（用于索引 edge_mask）
        # matches: (N, 4) -> [x_A, y_A, x_B, y_B] in [-1, 1]
        N = matches.shape[0]
        # 归一化坐标 → 像素坐标（左上角为 (0,0)，右下角为 (W-1, H-1)）
        def norm_to_pixel(coords, H, W):
            # coords: (..., 2) in [-1, 1]
            pixel = (coords + 1) * 0.5 * torch.tensor([W - 1, H - 1], device=coords.device)
            return pixel.round().long()

        pts_A = matches[:, :2]  # (N, 2)
        pts_B = matches[:, 2:]  # (N, 2)
        
        H_A, W_A, _ = edge_mask.shape
        pix_A = norm_to_pixel(pts_A, H_A, W_A)  # (N, 2)
        pix_B = norm_to_pixel(pts_B, H_A, W_A)  # 假设 H_B = H_A, W_B = W_A（如原注释）

        # Clamp to valid range
        pix_A = torch.stack([
            torch.clamp(pix_A[:, 0], min=0, max=W_A - 1),
            torch.clamp(pix_A[:, 1], min=0, max=H_A - 1)
        ], dim=1)

        pix_B = torch.stack([
            torch.clamp(pix_B[:, 0], min=0, max=W_A - 1),
            torch.clamp(pix_B[:, 1], min=0, max=H_A - 1)
        ], dim=1)

        # Check if within edge regions
        # edge_A[y, x] -> note: indexing is [row, col] = [y, x]
        in_edge_A = edge_A[pix_A[:, 1], pix_A[:, 0]] > 0  # (N,)
        in_edge_B = edge_B[pix_B[:, 1], pix_B[:, 0]] > 0  # (N,)

        edge_mask_match = in_edge_A & in_edge_B  # (N,)

        if edge_mask_match.any():
            edge_indices = torch.where(edge_mask_match)[0]
            edge_conf = confidence[edge_indices]

            # Avoid zero weights
            if edge_conf.sum() == 0:
                edge_conf = torch.ones_like(edge_conf)

            num_edge_to_sample = min(edge_sample_num, len(edge_indices))
            set_seed(seed)
            edge_sampled_inds = torch.multinomial(edge_conf, num_edge_to_sample, replacement=False)
            final_edge_inds = edge_indices[edge_sampled_inds]

            edge_matches.append(matches[final_edge_inds])
            edge_confidences.append(confidence[final_edge_inds])
            if precision is not None:
                edge_precisions.append(precision[final_edge_inds])

    # --- Step 3: 合并原始采样 + 边缘采样 ---
    all_matches = [sampled_matches]
    all_confidences = [sampled_confidence]
    all_precisions = [sampled_precision] if sampled_precision is not None else None

    if edge_matches:
        all_matches.extend(edge_matches)
        all_confidences.extend(edge_confidences)
        if all_precisions is not None:
            all_precisions.extend(edge_precisions)

    final_matches = torch.cat(all_matches, dim=0)
    final_confidences = torch.cat(all_confidences, dim=0)
    if all_precisions is not None:
        final_precision = torch.cat(all_precisions, dim=0)
    else:
        final_precision = None

    # --- Step 4: KDE 平衡采样（保持原逻辑）---
    density = kde(final_matches)
    p = 1 / (density + 1)
    p[density < 10] = 1e-7

    num_final = min(num_corresp, len(final_confidences))
    set_seed(seed)
    balanced_samples = torch.multinomial(p, num_samples=num_final, replacement=False)

    selected_matches = final_matches[balanced_samples]
    selected_confidence = final_confidences[balanced_samples]
    if final_precision is not None:
        selected_precision_0 = final_precision[balanced_samples][:, 0]
        selected_precision_1 = final_precision[balanced_samples][:, 1]
    else:
        selected_precision_0 = None
        selected_precision_1 = None

    return (
        selected_matches,
        selected_confidence,
        selected_precision_0,
        selected_precision_1,
    )


def to_pixel(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    return torch.stack(((x[..., 0] + 1) / 2 * W, (x[..., 1] + 1) / 2 * H), dim=-1)

def to_pixel_coordinates(
    warp: torch.Tensor, H_A: int, W_A: int, H_B: int, W_B: int
):
    return to_pixel(warp[..., :2], H=H_A, W=W_A), to_pixel(
        warp[..., 2:], H=H_B, W=W_B
    )


def save_matches_hloc_nested_format(matches, out_dir, eps=1e-3, **kwargs):
    os.makedirs(out_dir, exist_ok=True)
    keypoints_path = os.path.join(out_dir, "keypoints.h5")
    matching_path = os.path.join(out_dir, "matches.h5")
    
    # Clear files
    for p in [keypoints_path, matching_path]:
        if os.path.exists(p):
            os.remove(p)

    # Step 1: Build merged keypoints for every image from matches
    all_images = set()
    kpts_by_img = defaultdict(list)

    for imgA in matches:
        all_images.add(imgA)
        for imgB, m in matches[imgA].items():
            all_images.add(imgB)
            kpts_by_img[imgA].append(m['kptsA'])
            kpts_by_img[imgB].append(m['kptsB'])

    merged_kpts = {}
    merged_quant_to_idx = {}

    for img in tqdm(all_images, desc="Merging Keypoints"):
        if not kpts_by_img[img]:
            merged_kpts[img] = np.empty((0, 2), dtype=np.float32)
            merged_quant_to_idx[img] = {}
        else:
            all_pts = np.concatenate(kpts_by_img[img], axis=0)
            # 向量化去重逻辑
            quantized = np.round(all_pts / eps).astype(np.int64)
            # 使用 structured array 或 view 方法实现高效去重
            dt = np.dtype((np.void, quantized.dtype.itemsize * quantized.shape[1]))
            quantized_view = quantized.view(dt).ravel()
            _, unique_indices = np.unique(quantized_view, return_index=True)
            unique_indices.sort()  # 保持原始顺序（如需要 stability）
            merged = all_pts[unique_indices].astype(np.float32)
            merged_kpts[img] = merged

            # 向量化构建 quantize(pt) -> index 映射
            if merged.size > 0:
                q_arr = np.round(merged / eps).astype(np.int64)
                # 将每行转为 tuple，向量化方式
                q_tuples = [tuple(row) for row in q_arr]  # 仍需转 tuple，但循环在 C 层更快
                merged_quant_to_idx[img] = dict(zip(q_tuples, range(len(merged))))
            else:
                merged_quant_to_idx[img] = {}
    
    # Save keypoints.h5 (flat)
    with h5py.File(keypoints_path, "w", libver="latest") as fd:
        for img in tqdm(merged_kpts, desc="Saving keypoints"):
            kpts = merged_kpts[img]
            grp = fd.create_group(img)
            for k, v in kwargs.items():
                grp.create_dataset(k, data=v[img])
            if kpts.size > 0:
                ds = grp.create_dataset('keypoints', data=kpts.astype(np.float16))
                ds.attrs['uncertainty'] = 1.0
            else:
                grp.create_dataset('keypoints', data=np.empty((0, 2), dtype=np.float16))

    # Save matches.h5 in nested format: /imgA/imgB/{matches0, matching_scores0}
    processed = set()

    with h5py.File(matching_path, "w", libver="latest") as fd:
        for imgA in tqdm(merged_kpts, desc="Building nested matches"):
            N_A = len(merged_kpts[imgA])
            if N_A == 0:
                continue

            # Create group for imgA if not exists
            if safe_hdf5_name(imgA) not in fd:
                grp_A = fd.create_group(safe_hdf5_name(imgA))
            else:
                grp_A = fd[safe_hdf5_name(imgA)]

            # Find all neighbors of imgA
            neighbors = set()
            if imgA in matches:
                neighbors.update(matches[imgA].keys())
            for other in matches:
                if imgA in matches[other]:
                    neighbors.add(other)

            for imgB in neighbors:
                if imgB not in merged_kpts or len(merged_kpts[imgB]) == 0:
                    continue

                pair = (imgA, imgB)
                if pair in processed:
                    continue
                processed.add(pair)

                # Initialize match array
                matches_arr = -np.ones(N_A, dtype=np.int32)
                scores = np.zeros(N_A, dtype=np.float16)

                q2idx_A = merged_quant_to_idx[imgA]
                q2idx_B = merged_quant_to_idx[imgB]

                # Case 1: imgA as A
                if imgA in matches and imgB in matches[imgA]:
                    m = matches[imgA][imgB]
                    for i in range(len(m['kptsA'])):
                        qA = quantize(m['kptsA'][i], eps)
                        qB = quantize(m['kptsB'][i], eps)
                        if qA in q2idx_A and qB in q2idx_B:
                            idxA = q2idx_A[qA]
                            idxB = q2idx_B[qB]
                            matches_arr[idxA] = idxB
                            scores[idxA] = 1.0

                # Case 2: imgA as B
                if imgB in matches and imgA in matches[imgB]:
                    m = matches[imgB][imgA]
                    for i in range(len(m['kptsA'])):
                        qB_ = quantize(m['kptsA'][i], eps)
                        qA_ = quantize(m['kptsB'][i], eps)
                        if qA_ in q2idx_A and qB_ in q2idx_B:
                            idxA = q2idx_A[qA_]
                            idxB = q2idx_B[qB_]
                            matches_arr[idxA] = idxB
                            scores[idxA] = 1.0

                # Create sub-group: /imgA/imgB
                if safe_hdf5_name(imgB) in grp_A:
                    del grp_A[safe_hdf5_name(imgB)]
                subgrp = grp_A.create_group(safe_hdf5_name(imgB))

                # Save with FIXED dataset names
                subgrp.create_dataset("matches0", data=matches_arr)
                subgrp.create_dataset("matching_scores0", data=scores)

    print(f"✅ Saved nested matches to {matching_path}")


def rgb_to_edge_mask(
    img: Image.Image,
    low_threshold: int = 50,
    high_threshold: int = 150,
    dilate_kernel_size: int = 3,
    dilate_iterations: int = 1
) -> np.ndarray:
    """
    将 RGB PIL 图像转换为二值边缘图（numpy array, HxW, float32, 0 or 1）
    使用 Canny 边缘检测 + 可选膨胀（dilation）
    
    Args:
        img: 输入的 PIL RGB 图像
        low_threshold: Canny 低阈值
        high_threshold: Canny 高阈值
        dilate_kernel_size: 膨胀核大小（奇数，如 3, 5），设为 0 则不膨胀
        dilate_iterations: 膨胀迭代次数
    
    Returns:
        (H, W) 的 float32 数组，值为 0.0 或 1.0
    """
    # 转为 OpenCV 格式 (RGB -> BGR)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # 转灰度
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # 高斯模糊（减少噪声）
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny 边缘检测
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # 可选：膨胀边缘
    if dilate_kernel_size > 0 and dilate_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (dilate_kernel_size, dilate_kernel_size)
        )
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    # 转为 0/1 的 float32（便于后续与 torch 张量兼容）
    edge_binary = (edges > 0).astype(np.float32)
    return edge_binary

def get_fisheye_mask(H, W, crop):
    y, x = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing='ij'
    )
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0
    a = (W - 1) / 2.0   # x 方向半轴
    b = (H - 1) / 2.0   # y 方向半轴
    a = max(a - crop, 0.0)
    b = max(b - crop, 0.0)
    ellipse_norm = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2
    fisheye_mask = ellipse_norm <= 1.0
    return fisheye_mask

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Run RoMaV2 with a specified scene directory.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--load_from",
        default=None,
        help="Path of checkpoint to be load.",
    )
    parser.add_argument(
        "--scene_dir",
        "-s",
        type=str,
        required=True,
        help="Path to the scene directory (e.g., /mnt/nasTeam/AI/lx/results/hColmap_orig/...)"
    )
    parser.add_argument(
        "--pair",
        "-p",
        type=str,
        default="pose_pair.txt",
        help="Path to the image pair file. Default to pose_pair.txt"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=500,
        help="Number of sample matched per image pair, Default 500"
    )
    parser.add_argument(
        "--n_edge_sample",
        type=int,
        default=5000,
        help="Number of sample matched per image pair, Default 5000"
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=840,
        help="Number of sample matched per image pair, Default 5000"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="kosmo_match",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--vis", action='store_true',
        help='Whether to visualize match results'
    )
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)
    
    if args.load_from in ["None", "none", "null"]:
        args.load_from = None
    
    print("Loading model...")
    model = instantiate_from_config(cfg["model"]).cuda()
    model.eval()

    if args.load_from is not None:
        if args.load_from == "latest":
            args.load_from = os.path.join(os.path.dirname(args.config), "checkpoint/latest/ckpt.pth")
        elif args.load_from == "best":
            args.load_from = os.path.join(os.path.dirname(args.config), "checkpoint/best/ckpt.pth")

        model.load_checkpoint(ckpt_path=args.load_from)

    scene_dir = Path(args.scene_dir)
    image_dir = scene_dir / "images"
    mask_dir = scene_dir / "masks"
    image_pair_path = scene_dir / args.pair
    output_dir = os.path.join(scene_dir, args.output)

    with open(image_pair_path, 'r') as f:
        image_pair = f.readlines()

    image_size = {}
    pair_results = {}
    model_infer_size = args.process_res
    if args.vis:
        image_pair = image_pair[:10]


    for pair in tqdm(image_pair, desc="Running Kosmo Matching", unit='pair'):
        img_A_name, img_B_name = pair.strip().split(" ")
        img_A = Image.open(image_dir / img_A_name)
        img_B = Image.open(image_dir / img_B_name)
        
        # # TODO: DEBUG
        # fisheye_mask = get_fisheye_mask(2400, 2400, crop=10).float()
        # img_A = img_A.crop((800, 300, 3200, 2700))
        # img_B = img_B.crop((800, 300, 3200, 2700))

        mask_A_path = mask_dir / (os.path.splitext(img_A_name)[0] + '.png')
        mask_B_path = mask_dir / (os.path.splitext(img_B_name)[0] + '.png')
        mask_A = mask_B = None
        if mask_A_path.exists():
            mask_A = Image.open(mask_A_path)
            mask_A = mask_A.crop((800, 300, 3200, 2700))
            mask_A = torch.from_numpy(np.array(mask_A)).float().cuda() > 0
        if mask_B_path.exists():
            mask_B = Image.open(mask_B_path)
            mask_B = mask_B.crop((800, 300, 3200, 2700))
            mask_B = torch.from_numpy(np.array(mask_B)).float().cuda() > 0

        W_A, H_A = img_A.size
        W_B, H_B = img_B.size
        image_size[img_A_name] = np.array([H_A, W_A])
        image_size[img_B_name] = np.array([H_B, W_B])

        if args.n_edge_sample > 0:
            # 提取边缘并轻微膨胀（让边缘更鲁棒）
            edge_A = rgb_to_edge_mask(img_A, low_threshold=50, high_threshold=150, dilate_kernel_size=3, dilate_iterations=1)
            edge_B = rgb_to_edge_mask(img_B, low_threshold=50, high_threshold=150, dilate_kernel_size=3, dilate_iterations=1)
            # 拼接
            edge_mask_np = np.stack([edge_A, edge_B], axis=-1)  # (H, W, 2)
            rgb_edge_mask = torch.from_numpy(edge_mask_np).float().cuda()

        # Match densely for any image-like pair of inputs
        input_images = torch.stack([torch.as_tensor(np.array(img)).permute(2, 0, 1) for img in [img_A, img_B]], dim=0).float().cuda()
        # # TODO: DEBUG
        # input_images = input_images * fisheye_mask[None, None].cuda()
        input_images = torch.nn.functional.interpolate(
            input_images, size=(model_infer_size, model_infer_size), mode="bilinear", align_corners=True
        )
        input_images = (input_images / 127.5 - 1).unsqueeze(0) # 1, 2, 3, H, W
        meta_data = {}
        meta_data["frames"] = [1]
        meta_data["views"] = [2]
        meta_data["input_width"] = [model_infer_size]
        meta_data["input_height"] = [model_infer_size]
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                results = model.model(input_images, meta_data=meta_data)

        preds = results["match"]['final']
        # preds = results["match"]['coarse']
        confidence_AB = preds.pop('confidence_AB')
        overlap_AB = confidence_AB[..., :1].sigmoid()
        preds["overlap_AB"] = overlap_AB
        for key, val in preds.items():
            preds[key] = val.squeeze(1).float()
        
        if mask_A is not None and "overlap_AB" in preds:
            if mask_A.shape != preds["overlap_AB"].shape[1:3]:
                mask_A = torch.nn.functional.interpolate(
                    mask_A[None, None].float(), size=preds["overlap_AB"].shape[1:3], mode="bilinear", align_corners=False
                )[0, 0] > 0
            preds["overlap_AB"] = preds["overlap_AB"] * mask_A[None, ..., None].float()
        if mask_B is not None and "overlap_BA" in preds:
            if mask_B.shape != preds["overlap_BA"].shape[1:3]:
                mask_B = torch.nn.functional.interpolate(
                    mask_B[None, None].float(), size=preds["overlap_BA"].shape[1:3], mode="bilinear", align_corners=False
                )[0, 0] > 0
            preds["overlap_BA"] = preds["overlap_BA"] * mask_B[None, ..., None].float()
        
        # Sample matches for estimation
        assert args.n_edge_sample > 0
        matches, overlaps, precision_AB, precision_BA = sample_with_edge(
            preds, args.n_sample, rgb_edge_mask, args.n_edge_sample,
        )
        
        kptsA, kptsB = to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

        if args.vis:
            img_A_name_vis = Path(img_A_name).parent.name + '-' + Path(img_A_name).name[:-4]
            img_B_name_vis = Path(img_B_name).parent.name + '-' + Path(img_B_name).name[:-4]
            save_path = f"{output_dir}/kosmo_match_vis/{img_A_name_vis}_{img_B_name_vis}.jpg"
            if not Path(save_path).exists():
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            import torch.nn.functional as F
            warp_AB, overlap_AB = preds["warp_AB"][0], preds["overlap_AB"][0]

            x1 = (torch.tensor(np.array(img_A)) / 255).cuda().permute(2, 0, 1)
            x2 = (torch.tensor(np.array(img_B)) / 255).cuda().permute(2, 0, 1)

            im2_transfer_rgb = F.grid_sample(
                x2[None], warp_AB[None], mode="bilinear", align_corners=False
            )[0]
            warp_im = im2_transfer_rgb
            overlap = overlap_AB.squeeze()
            # white_im = torch.ones((H_A, 2 * W_A), device=device)
            white_im = torch.ones((warp_im.shape[-2], warp_im.shape[-1]), device=x1.device)
            vis_im = overlap * warp_im + (1 - overlap) * white_im
            if not Path(save_path).exists():
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            tensor_to_pil(vis_im).save(save_path)
        
            # vis match
            img_A_np = np.array(img_A)  # shape: (H_A, W_A, 3)
            img_B_np = np.array(img_B)  # shape: (H_B, W_B, 3)
            
            # 拼接原始图像（必须 copy！）
            combined_orig = np.concatenate([img_A_np, img_B_np], axis=1).copy()
            # breakpoint()
            W_A = img_A_np.shape[1]
            num_matches = kptsA.shape[0]
            if num_matches > 0:
                import cv2, random
                random.seed(42)
                # 随机选最多 30 个
                num_lines = min(30, num_matches)
                indices = np.random.choice(num_matches, size=num_lines, replace=False)

                for idx in indices:
                    # 注意：kptsA 和 kptsB 是 [x, y] 还是 [y, x]？
                    # 根据 RoMa 和 to_pixel_coordinates 的惯例，通常是 [x, y]（即 (u, v)）
                    # OpenCV 的 point 是 (x, y)，所以可以直接用

                    x_a, y_a = kptsA[idx]# W, H
                    x_b, y_b = kptsB[idx]
                    
                    x_a = x_a.item()
                    y_a = y_a.item()
                    x_b = x_b.item()
                    y_b = y_b.item()

                    # 转为整数（并确保在图像范围内）
                    pt1 = (int(round(x_a)), int(round(y_a)))
                    pt2 = (int(round(x_b + W_A)), int(round(y_b)))  # B 图在右侧，x 偏移 W_A

                    # 可选：clip 到图像边界（防止越界）
                    h_a, w_a = img_A_np.shape[:2]
                    h_b, w_b = img_B_np.shape[:2]
                    pt1 = (np.clip(pt1[0], 0, w_a - 1), np.clip(pt1[1], 0, h_a - 1))
                    pt2 = (np.clip(pt2[0], W_A, W_A + w_b - 1), np.clip(pt2[1], 0, h_b - 1))

                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.line(combined_orig, pt1, pt2, color=color, thickness=5)
                match_vis_path = os.path.splitext(save_path)[0] + '_matches.png'
                Image.fromarray(combined_orig).save(match_vis_path)

        if img_A_name in pair_results:
            pair_results[img_A_name][img_B_name] = {
                'kptsA': kptsA.cpu().numpy(),
                'kptsB': kptsB.cpu().numpy(),
            }
        else:
            pair_results[img_A_name] = {
                img_B_name: {
                    'kptsA': kptsA.cpu().numpy(),
                    'kptsB': kptsB.cpu().numpy(),
                }
            }

    print(f"Processing Pair Results ...")

    save_matches_hloc_nested_format(
        pair_results,
        out_dir=output_dir,
        image_size=image_size
    )
