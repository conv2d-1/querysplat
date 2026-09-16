"""
Run Kosmo Match dense matching on image pairs and write correspondences to HDF5.

Drop-in replacement for roma_correspondences.py: uses Kosmo Match instead of RoMa
while producing the same MargBA-compatible correspondence format (corres_i2j / visibility_i2j).

Images are read from disk (--image-dir), consistent with
run_kosmo_match_sample_grid_fill_conf_parallel_front.py.

Usage (run from TMA root):
  CUDA_VISIBLE_DEVICES=0,1 python hAlgorithm/script/infer/kosmo_match_mba_correspondences.py \
    --h5-path /path/to/custom.hdf5 \
    --image-dir /path/to/images \
    --pairs-path /path/to/pose_pair.txt \
    --frame-map /path/to/frame_names.json \
    --config /path/to/model_config.py \
    --load_from latest \
    [--process-res 784] \
    [--min-confidence 0.2] \
    [--min-visibility 0.1] \
    [--mba-root /path/to/Marginalized-Bundle-Adjustment]
"""
import os
import sys

sys.path.append(os.getcwd())

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from hAlgorithm.utils import (
    config_merge_args,
    file2dict,
    instantiate_from_config,
    parse_unknown,
)


def setup_mba_imports(mba_root):
    """Add MargBA root to sys.path so corre_utils can be imported."""
    if mba_root not in sys.path:
        sys.path.insert(0, mba_root)


# ------------------------------------------------------------------
#  Name -> index mapping (identical to roma_correspondences.py)
# ------------------------------------------------------------------

def parse_images_txt(path):
    images = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 10:
            images.append(dict(image_id=int(parts[0]), name=parts[9]))
            if i < len(lines):
                i += 1
    return images


def build_name_to_idx_from_colmap(colmap_dir):
    images = parse_images_txt(os.path.join(colmap_dir, 'images.txt'))
    images_sorted = sorted(images, key=lambda x: x['image_id'])
    return {img['name']: i for i, img in enumerate(images_sorted)}


def build_name_to_idx_from_frame_map(frame_map_path):
    with open(frame_map_path) as f:
        idx_to_name = json.load(f)
    return {name: int(idx) for idx, name in idx_to_name.items()}


# ------------------------------------------------------------------
#  Image reading (from disk files, consistent with
#  run_kosmo_match_sample_grid_fill_conf_parallel_front.py)
# ------------------------------------------------------------------

def read_rgb_from_file(image_dir, filename):
    """Read RGB image from disk as numpy uint8 array (H, W, 3)."""
    path = os.path.join(image_dir, filename)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------
#  Model loading
# ------------------------------------------------------------------

def load_kosmo_model(config_path, load_from, device):
    cfg = file2dict(config_path)
    model = instantiate_from_config(cfg["model"]).to(device)
    model.eval()
    if load_from is not None:
        ckpt = load_from
        if ckpt == "latest":
            ckpt = os.path.join(os.path.dirname(config_path), "checkpoint/latest/ckpt.pth")
        elif ckpt == "best":
            ckpt = os.path.join(os.path.dirname(config_path), "checkpoint/best/ckpt.pth")
        model.load_checkpoint(ckpt_path=ckpt)
        print(f"Loaded Kosmo Match model from {ckpt}")
    return model


# ------------------------------------------------------------------
#  Kosmo Match inference (single direction)
# ------------------------------------------------------------------

def build_normalized_grid(H, W, device):
    """Build cell-center normalized grid in [-1, 1], matching RoMa / Kosmo conventions."""
    ys = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device)
    xs = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)


def run_kosmo_single_direction(model, img_A_resized, img_B_resized, process_res, device):
    """Run Kosmo Match for A->B direction.

    Parameters
    ----------
    img_A_resized, img_B_resized : np.ndarray (H, W, 3) uint8, already resized to process_res.

    Returns
    -------
    corres : (1, H', W', 4)  -- [src_grid_xy, dst_warp_xy] in [-1, 1]
    certainty : (1, 1, H', W')  -- overlap probability in [0, 1]
    """
    # ---- prepare data ----
    # t0 = time.perf_counter()
    input_images = torch.from_numpy(
        np.stack([img_A_resized, img_B_resized], axis=0)
    ).permute(0, 3, 1, 2).float()
    input_images = (input_images / 127.5 - 1).unsqueeze(0).to(device)

    meta_data = {
        "frames": [1],
        "views": [2],
        "input_width": [process_res],
        "input_height": [process_res],
    }
    # torch.cuda.synchronize(device)
    # t1 = time.perf_counter()

    # ---- model inference ----
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            results = model.model(input_images, meta_data=meta_data, bidirectional=False)
    # torch.cuda.synchronize(device)
    # t2 = time.perf_counter()

    # ---- extract warp & certainty ----
    if "final" in results["match"]:
        preds = results["match"]["final"]
    elif "refiner" in results["match"]:
        preds = results["match"]["refiner"]
    else:
        preds = results["match"]["coarse"]

    warp = preds["warp_AB"].squeeze(1).float()                          # (1, H', W', 2)
    confidence_raw = preds["confidence_AB"]                              # (1, 1, H', W', C)
    certainty = confidence_raw[..., :1].sigmoid().squeeze(1).float()     # (1, H', W', 1)
    certainty = certainty.squeeze(-1).unsqueeze(1)                       # (1, 1, H', W')

    wrong = (warp.abs() > 1).sum(dim=-1) > 0
    certainty[:, 0][wrong] = 0.0
    warp = torch.clamp(warp, -1, 1)

    H, W = warp.shape[1], warp.shape[2]
    grid = build_normalized_grid(H, W, device).unsqueeze(0)
    corres = torch.cat([grid, warp], dim=-1)                             # (1, H', W', 4)
    # torch.cuda.synchronize(device)
    # t3 = time.perf_counter()

    # timings = {
    #     "prepare": t1 - t0,
    #     "infer": t2 - t1,
    #     "extract": t3 - t2,
    # }
    return corres, certainty


# ------------------------------------------------------------------
#  Background save (CPU-only, runs in thread pool)
# ------------------------------------------------------------------

def _save_pair_to_disk(corres2png_fn, write_corres_fn,
                       pair_fwd, pair_rev,
                       corres_fwd_cpu, cert_fwd_cpu, valid_fwd_cpu,
                       corres_rev_cpu, cert_rev_cpu, valid_rev_cpu,
                       vis_val, corres_dir, visibility_dir):
    """Convert to PNG and write to disk.  Pure CPU / IO — safe for background threads."""
    fwd_png = corres2png_fn(corres_fwd_cpu, cert_fwd_cpu.squeeze(), valid_fwd_cpu.squeeze())
    rev_png = corres2png_fn(corres_rev_cpu, cert_rev_cpu.squeeze(), valid_rev_cpu.squeeze())

    write_corres_fn(fwd_png, pair_fwd, corres_dir)
    write_corres_fn(rev_png, pair_rev, corres_dir)

    np.savetxt(os.path.join(visibility_dir, f'{pair_fwd}.txt'), np.array([vis_val]))
    np.savetxt(os.path.join(visibility_dir, f'{pair_rev}.txt'), np.array([vis_val]))


# ------------------------------------------------------------------
#  Multi-GPU worker (chunked pipeline)
# ------------------------------------------------------------------

def kosmo_worker(rank, world_size, args, pairs_with_idx):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    setup_mba_imports(args.mba_root)
    from MargBA.corres_estimator.corre_utils import (
        corres2png, compute_visibility, forward_backward_check, write_corres,
    )

    model = load_kosmo_model(args.config, args.load_from, device)

    my_pairs = pairs_with_idx[rank::world_size]
    intermediate_dir = os.path.join(os.path.dirname(args.h5_path), '_kosmo_intermediate')
    corres_dir = os.path.join(intermediate_dir, 'corres_i2j')
    visibility_dir = os.path.join(intermediate_dir, 'visibility_i2j')
    os.makedirs(corres_dir, exist_ok=True)
    os.makedirs(visibility_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  Step 1: Collect unique image indices for this worker's pairs
    # ------------------------------------------------------------------
    unique_idxs = set()
    for idx1, idx2 in my_pairs:
        unique_idxs.add(idx1)
        unique_idxs.add(idx2)
    unique_idxs = sorted(unique_idxs)

    # ------------------------------------------------------------------
    #  Step 2: Pre-load & resize all unique images once
    # ------------------------------------------------------------------
    target_size = (args.process_res, args.process_res)
    image_cache = {}
    for idx in tqdm(unique_idxs, desc=f"GPU {rank} load images", disable=(rank != 0)):
        img_uint8 = read_rgb_from_file(args.image_dir, args.idx_to_name[idx])
        image_cache[idx] = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_LINEAR)

    print(f"GPU {rank}: cached {len(image_cache)} unique images, processing {len(my_pairs)} pairs x 2 directions")

    # ------------------------------------------------------------------
    #  Step 3: Chunked pipeline — GPU infer ‖ CPU save
    #
    #  Main thread : inference + GPU post-proc (fb_check, visibility)
    #  Thread pool  : corres2png + write_corres + savetxt  (overlapped)
    # ------------------------------------------------------------------
    chunk_size = args.chunk_size
    num_save_workers = args.num_save_workers
    save_pool = ThreadPoolExecutor(max_workers=num_save_workers)
    prev_chunk_futures = []

    # total_t = {"prepare": 0.0, "infer": 0.0, "extract": 0.0, "gpu_post": 0.0}
    # total_save_wait = 0.0
    n_processed = 0
    n_saved = 0

    total_pairs = len(my_pairs)
    n_chunks = (total_pairs + chunk_size - 1) // chunk_size

    chunk_pbar = tqdm(total=n_chunks, desc=f"GPU {rank} chunks", position=rank * 2, leave=True)
    pair_pbar = tqdm(total=chunk_size, desc=f"GPU {rank} pairs", position=rank * 2 + 1, leave=True)

    for chunk_idx, chunk_start in enumerate(range(0, total_pairs, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, total_pairs)
        chunk_pairs = my_pairs[chunk_start:chunk_end]

        # ---- wait for *previous* chunk's saves before submitting new ones ----
        if prev_chunk_futures:
            for f in prev_chunk_futures:
                f.result()
        prev_chunk_futures = []

        # ---- reset pair progress bar for this chunk ----
        pair_pbar.reset(total=len(chunk_pairs))
        pair_pbar.set_description(f"GPU {rank} chunk {chunk_idx + 1}/{n_chunks}")

        # ---- GPU: inference + fb_check + visibility for this chunk ----
        for idx1, idx2 in chunk_pairs:
            pair_fwd = f"{str(idx1).zfill(6)}_{str(idx2).zfill(6)}"
            pair_rev = f"{str(idx2).zfill(6)}_{str(idx1).zfill(6)}"

            if os.path.exists(os.path.join(corres_dir, pair_fwd)):
                pair_pbar.update(1)
                continue

            img_src_resized = image_cache[idx1]
            img_dst_resized = image_cache[idx2]

            # ---- two-direction inference ----
            corres_src_dst, cert_src_dst = run_kosmo_single_direction(
                model, img_src_resized, img_dst_resized, args.process_res, device,
            )
            corres_dst_src, cert_dst_src = run_kosmo_single_direction(
                model, img_dst_resized, img_src_resized, args.process_res, device,
            )

            # ---- GPU post-proc: fb_check + visibility ----
            cert_src_dst[cert_src_dst < args.min_confidence] = 0.0
            cert_dst_src[cert_dst_src < args.min_confidence] = 0.0

            cyclic_err_fwd = forward_backward_check(corres_src_dst, corres_dst_src)
            cyclic_err_rev = forward_backward_check(corres_dst_src, corres_src_dst)

            valid_src_dst = (cyclic_err_fwd > 0) * (cert_src_dst > args.min_confidence)
            valid_dst_src = (cyclic_err_rev > 0) * (cert_dst_src > args.min_confidence)

            visibility = compute_visibility(valid_src_dst, valid_dst_src)
            n_processed += 1

            if visibility[0] < args.min_visibility:
                pair_pbar.update(1)
                continue
            if torch.sum(valid_src_dst[0]) == 0 or torch.sum(valid_dst_src[0]) == 0:
                pair_pbar.update(1)
                continue

            # ---- move to CPU and submit to save pool ----
            fut = save_pool.submit(
                _save_pair_to_disk,
                corres2png, write_corres,
                pair_fwd, pair_rev,
                corres_src_dst[0].cpu(), cert_src_dst[0].cpu(), valid_src_dst[0].cpu(),
                corres_dst_src[0].cpu(), cert_dst_src[0].cpu(), valid_dst_src[0].cpu(),
                visibility[0].item(), corres_dir, visibility_dir,
            )
            prev_chunk_futures.append(fut)
            n_saved += 1
            pair_pbar.update(1)

        chunk_pbar.update(1)

    # ---- drain final chunk ----
    if prev_chunk_futures:
        for f in prev_chunk_futures:
            f.result()

    pair_pbar.close()
    chunk_pbar.close()
    save_pool.shutdown()
    print(f"[GPU {rank}] Done: {n_processed} pairs inferred, {n_saved} saved.")


# ------------------------------------------------------------------
#  HDF5 merge (same logic as roma_correspondences.py)
# ------------------------------------------------------------------

def write_corres_to_hdf5(h5_path, intermediate_dir):
    import glob
    import shutil

    import natsort

    corres_dir = os.path.join(intermediate_dir, 'corres_i2j')
    visibility_dir = os.path.join(intermediate_dir, 'visibility_i2j')

    hfw = h5py.File(h5_path, 'a')

    if 'corres_i2j' not in hfw:
        hfw.create_group('corres_i2j')
    if 'visibility_i2j' not in hfw:
        hfw.create_group('visibility_i2j')

    pair_dirs = natsort.natsorted(glob.glob(os.path.join(corres_dir, '*')))
    n_written = 0
    for pair_path in tqdm(pair_dirs, desc="Writing to HDF5"):
        pair_name = os.path.basename(pair_path)
        if pair_name in hfw['corres_i2j']:
            continue

        grp = hfw['corres_i2j'].create_group(pair_name)
        for suffix in ['_x.png', '_y.png', '_conf.png']:
            img_name = f'{pair_name}{suffix}'
            img_path = os.path.join(pair_path, img_name)
            with open(img_path, 'rb') as f:
                img_data = np.frombuffer(f.read(), dtype=np.uint8)
            grp.create_dataset(img_name, data=img_data)

        vis_path = os.path.join(visibility_dir, f'{pair_name}.txt')
        if os.path.exists(vis_path):
            vis_val = np.loadtxt(vis_path)
            if f'{pair_name}.txt' not in hfw['visibility_i2j']:
                hfw['visibility_i2j'].create_dataset(f'{pair_name}.txt', data=vis_val)

        n_written += 1

    hfw.close()
    print(f"Written {n_written} correspondence entries to HDF5")

    shutil.rmtree(intermediate_dir)
    print(f"Cleaned up intermediate directory")


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run Kosmo Match dense matching and write MargBA-format correspondences to HDF5',
    )
    parser.add_argument('--h5-path', type=str, required=True,
                        help='Path to output HDF5 for correspondences')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory containing source RGB images')
    parser.add_argument('--pairs-path', type=str, required=True,
                        help='Path to pose_pair.txt')
    parser.add_argument('--colmap-dir', type=str, default=None,
                        help='Path to COLMAP sparse/0 (for name->index mapping)')
    parser.add_argument('--frame-map', type=str, default=None,
                        help='Path to frame_names.json (alternative to --colmap-dir)')
    parser.add_argument('--min-confidence', type=float, default=0.2)
    parser.add_argument('--min-visibility', type=float, default=0.1)

    parser.add_argument('--config', type=str, required=True,
                        help='Path to Kosmo Match model config (.py)')
    parser.add_argument('--load_from', default=None,
                        help='Checkpoint path, or "latest" / "best"')
    parser.add_argument('--process-res', type=int, default=784,
                        help='Model inference resolution')

    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Pairs per inference chunk before flushing saves to background')
    parser.add_argument('--num-save-workers', type=int, default=4,
                        help='Number of background threads for corres2png + file I/O')

    parser.add_argument('--mba-root', type=str, default=None,
                        help='Path to Marginalized-Bundle-Adjustment root (auto-detected if omitted)')

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    if args.mba_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(script_dir))
        ))
        args.mba_root = os.path.join(workspace_root, 'Marginalized-Bundle-Adjustment')

    if args.load_from in ("None", "none", "null"):
        args.load_from = None

    setup_mba_imports(args.mba_root)

    if args.frame_map:
        name_to_idx = build_name_to_idx_from_frame_map(args.frame_map)
    elif args.colmap_dir:
        name_to_idx = build_name_to_idx_from_colmap(args.colmap_dir)
    else:
        raise ValueError("Must specify either --colmap-dir or --frame-map")

    args.idx_to_name = {idx: name for name, idx in name_to_idx.items()}

    print("=== Reading pairs ===")
    with open(args.pairs_path) as f:
        pair_lines = [l.strip() for l in f if l.strip()]

    pairs_with_idx = []
    for line in pair_lines:
        parts = line.split()
        name1, name2 = parts[0], parts[1]
        if name1 in name_to_idx and name2 in name_to_idx:
            idx1, idx2 = name_to_idx[name1], name_to_idx[name2]
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            pairs_with_idx.append((idx1, idx2))

    pairs_with_idx = list(set(pairs_with_idx))
    print(f"  {len(pairs_with_idx)} unique pairs")

    print("=== Running Kosmo Match ===")
    world_size = torch.cuda.device_count()
    print(f"  Using {world_size} GPUs")

    mp.spawn(
        kosmo_worker,
        args=(world_size, args, pairs_with_idx),
        nprocs=world_size,
        join=True,
    )

    print("=== Writing to HDF5 ===")
    intermediate_dir = os.path.join(os.path.dirname(args.h5_path), '_kosmo_intermediate')
    write_corres_to_hdf5(args.h5_path, intermediate_dir)

    print("=== Done ===")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
