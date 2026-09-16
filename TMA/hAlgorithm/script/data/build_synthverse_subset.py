#!/usr/bin/env python3
"""Build SynthVerse Profile-B subset aligned with WorldTrack adt_mini query density.

Selects 50 scenes (T>=64), keeps first 64 frames, subsamples tracks to
Q_keep=min(430, candidate_pool) using frame-0 gating (visibility + projected
screen bounds).  SynthVerse ``trajs_3d`` is stored in world coordinates (c2w
extrinsics in JSON); gating projects via w2c at frame 0.

Usage:
    python build_synthverse_subset.py
    python build_synthverse_subset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults (Profile B)
# ---------------------------------------------------------------------------

DATA_ROOT = "/mnt/nasTeam2/AI/datasets/TMD/"
SRC_JSON = "SynthVerse/converted/train_mf_with_tracking.json"
OUT_JSON = "SynthVerse/converted/train_mf_with_tracking_subset50_64f_q430.json"
OUT_MANIFEST = "SynthVerse/converted/train_mf_with_tracking_subset50_64f_q430_manifest.json"
OUT_TRAJ_ROOT = "SynthVerse/converted_subset50_64f_q430"

NUM_SCENES = 50
NUM_FRAMES = 64
Q_TARGET = 430
SEED = 42
MIN_SCENE_CANDIDATES = 430  # prefer scenes with at least this many frame-0 queries

CATEGORIES = ("animal", "human", "interaction", "objects", "film", "embodied")
TRAJ_SUBDIRS = ("trajs_2d", "trajs_3d", "valids", "visibs")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _world_to_cam(xyz_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """[N,3] world points -> [N,3] camera coords."""
    pts = np.asarray(xyz_world, dtype=np.float64)
    n = pts.shape[0]
    hom = np.concatenate([pts, np.ones((n, 1), dtype=np.float64)], axis=1)
    return (hom @ np.asarray(w2c, dtype=np.float64).T)[:, :3]


def _project_cam_to_uv(xyz_cam: np.ndarray, cam_in: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(cam_in, dtype=np.float64).reshape(-1)[:4]
    z = xyz_cam[..., 2]
    safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
    u = (xyz_cam[..., 0] / safe_z) * fx + cx
    v = (xyz_cam[..., 1] / safe_z) * fy + cy
    return np.stack([u, v], axis=-1)


def _read_image_hw(rgb_path: str) -> Tuple[int, int]:
    """Return (H, W) from RGB header."""
    with Image.open(rgb_path) as im:
        w, h = im.size
    return int(h), int(w)


def compute_frame0_candidate_mask(
    *,
    trajs_3d_world: np.ndarray,
    visibs: np.ndarray,
    cam_in: Sequence[float],
    extrinsics_c2w: np.ndarray,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    """Frame-0 query gating (WorldTrack / Open-d4rt style, adapted for c2w)."""
    w2c0 = np.linalg.inv(np.asarray(extrinsics_c2w, dtype=np.float64))
    cam0 = _world_to_cam(trajs_3d_world, w2c0)
    uv0 = _project_cam_to_uv(cam0, cam_in)
    depth0 = cam0[:, 2]

    vis0 = np.asarray(visibs, dtype=bool)
    h, w = int(image_hw[0]), int(image_hw[1])

    mask = vis0.copy()
    mask &= np.isfinite(cam0).all(axis=-1)
    mask &= np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
    mask &= np.isfinite(uv0).all(axis=-1)
    mask &= (uv0[:, 0] >= 0.0) & (uv0[:, 0] < float(w))
    mask &= (uv0[:, 1] >= 0.0) & (uv0[:, 1] < float(h))
    return mask


def allocate_category_quotas(
    category_sizes: Dict[str, int],
    total: int,
) -> Dict[str, int]:
    """Proportional allocation (largest remainder) summing to *total*."""
    cats = [c for c in CATEGORIES if category_sizes.get(c, 0) > 0]
    if not cats:
        raise RuntimeError("No eligible scenes found")

    pool_total = sum(category_sizes[c] for c in cats)
    raw = {c: category_sizes[c] * total / pool_total for c in cats}
    base = {c: int(raw[c]) for c in cats}
    remainder = total - sum(base.values())
    order = sorted(cats, key=lambda c: (raw[c] - base[c]), reverse=True)
    for c in order[:remainder]:
        base[c] += 1
    return base


def stratified_select_scenes(
    scenes_by_cat: Dict[str, List[Tuple[str, int]]],
    *,
    num_scenes: int,
    seed: int,
    min_candidates: int,
) -> List[str]:
    """Select scenes stratified by category; prefer high candidate counts."""
    sizes = {c: len(scenes_by_cat.get(c, [])) for c in CATEGORIES}
    quotas = allocate_category_quotas(sizes, num_scenes)
    rng = np.random.default_rng(seed)
    selected: List[str] = []

    for cat in CATEGORIES:
        quota = quotas.get(cat, 0)
        if quota <= 0:
            continue
        pool = list(scenes_by_cat.get(cat, []))
        if not pool:
            raise RuntimeError(f"No scenes for category {cat!r} (quota={quota})")

        # Prefer scenes meeting min_candidates; fall back if needed.
        preferred = [x for x in pool if x[1] >= min_candidates]
        fallback = [x for x in pool if x[1] < min_candidates]
        ordered = preferred + fallback

        if len(ordered) < quota:
            raise RuntimeError(
                f"Category {cat!r}: need {quota} scenes but only {len(ordered)} eligible"
            )

        # Shuffle within preference tiers for reproducible diversity.
        pref_names = [s for s, _ in preferred]
        fall_names = [s for s, _ in fallback]
        rng.shuffle(pref_names)
        rng.shuffle(fall_names)
        cat_order = pref_names + fall_names
        picked = cat_order[:quota]
        selected.extend(picked)

    if len(selected) != num_scenes:
        raise RuntimeError(f"Selected {len(selected)} scenes, expected {num_scenes}")
    return selected


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def _resolve(root: str, rel: str) -> str:
    return os.path.join(root, rel)


def _scene_candidate_count(
    scene: str,
    frames: Sequence[dict],
    data_root: str,
) -> int:
    v0 = frames[0]["views"][0]
    trajs_3d = np.load(_resolve(data_root, v0["trajs_3d"]))
    visibs = np.load(_resolve(data_root, v0["visibs"]))
    rgb_path = _resolve(data_root, v0["rgb"])
    hw = _read_image_hw(rgb_path)
    mask = compute_frame0_candidate_mask(
        trajs_3d_world=trajs_3d,
        visibs=visibs,
        cam_in=v0["cam_in"],
        extrinsics_c2w=np.asarray(v0["extrinsics"]),
        image_hw=hw,
    )
    return int(mask.sum())


def subsample_scene(
    scene: str,
    frames: Sequence[dict],
    *,
    data_root: str,
    out_traj_root: str,
    num_frames: int,
    q_target: int,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """Subsample tracks and write npy; return per-scene manifest entry."""
    frames = frames[:num_frames]
    v0 = frames[0]["views"][0]
    rgb_path = _resolve(data_root, v0["rgb"])
    image_hw = _read_image_hw(rgb_path)

    trajs_3d_0 = np.load(_resolve(data_root, v0["trajs_3d"]))
    visibs_0 = np.load(_resolve(data_root, v0["visibs"]))
    candidate_mask = compute_frame0_candidate_mask(
        trajs_3d_world=trajs_3d_0,
        visibs=visibs_0,
        cam_in=v0["cam_in"],
        extrinsics_c2w=np.asarray(v0["extrinsics"]),
        image_hw=image_hw,
    )
    candidate_idx = np.flatnonzero(candidate_mask)
    n_candidates = int(candidate_idx.size)
    q_keep = min(q_target, n_candidates)

    rng = np.random.default_rng(seed)
    if q_keep < n_candidates:
        keep_idx = np.sort(rng.choice(candidate_idx, size=q_keep, replace=False))
    else:
        keep_idx = candidate_idx

    scene_out = os.path.join(data_root, out_traj_root, scene)
    if not dry_run:
        for sub in TRAJ_SUBDIRS:
            os.makedirs(os.path.join(scene_out, sub), exist_ok=True)

    n_tracks_orig = int(trajs_3d_0.shape[0])
    for t, frame in enumerate(frames):
        view = frame["views"][0]
        fstr = f"{t:06d}"
        arrays = {}
        for key in TRAJ_SUBDIRS:
            src = _resolve(data_root, view[key])
            arrays[key] = np.load(src)[keep_idx]

        if not dry_run:
            for key, arr in arrays.items():
                np.save(os.path.join(scene_out, key, f"{fstr}.npy"), arr)

    prefix = f"{out_traj_root}/{scene}"
    return {
        "scene": scene,
        "category": scene.split("_")[0],
        "num_frames": num_frames,
        "n_tracks_orig": n_tracks_orig,
        "n_candidates_frame0": n_candidates,
        "q_keep": q_keep,
        "image_hw": list(image_hw),
        "traj_prefix": prefix,
    }


def build_subset_json(
    src_mf: dict,
    selected_scenes: Sequence[str],
    manifest_entries: Sequence[dict],
    *,
    num_frames: int,
) -> dict:
    """Build mf_files JSON: reuse rgb/depth, point trajs to subset root."""
    manifest_by_scene = {m["scene"]: m for m in manifest_entries}
    out_mf = {}
    for scene in selected_scenes:
        src_frames = src_mf[scene][:num_frames]
        prefix = manifest_by_scene[scene]["traj_prefix"]
        new_frames = []
        for frame in src_frames:
            t = int(frame["frame_id"])
            fstr = f"{t:06d}"
            views = []
            for view in frame["views"]:
                nv = dict(view)
                nv["trajs_2d"] = f"{prefix}/trajs_2d/{fstr}.npy"
                nv["trajs_3d"] = f"{prefix}/trajs_3d/{fstr}.npy"
                nv["valids"] = f"{prefix}/valids/{fstr}.npy"
                nv["visibs"] = f"{prefix}/visibs/{fstr}.npy"
                views.append(nv)
            new_frames.append({"frame_id": frame["frame_id"], "views": views})
        out_mf[scene] = new_frames
    return {"mf_files": out_mf}


def run_build(args: argparse.Namespace) -> dict:
    data_root = args.data_root
    src_json_path = _resolve(data_root, args.src_json)

    with open(src_json_path, "r", encoding="utf-8") as f:
        src_payload = json.load(f)
    src_mf = src_payload.get("mf_files", src_payload)

    # Index eligible scenes and candidate counts.
    scenes_by_cat: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    print("Scanning eligible scenes (T>=64) and frame-0 candidate pools...")
    for scene, frames in tqdm(sorted(src_mf.items())):
        if len(frames) < args.num_frames:
            continue
        cat = scene.split("_")[0]
        if cat not in CATEGORIES:
            continue
        n_cand = _scene_candidate_count(scene, frames, data_root)
        scenes_by_cat[cat].append((scene, n_cand))

    for cat in CATEGORIES:
        scenes_by_cat[cat].sort(key=lambda x: x[1], reverse=True)

    selected = stratified_select_scenes(
        scenes_by_cat,
        num_scenes=args.num_scenes,
        seed=args.seed,
        min_candidates=args.min_scene_candidates,
    )
    print(f"Selected {len(selected)} scenes (seed={args.seed})")

    manifest_entries = []
    for i, scene in enumerate(tqdm(selected, desc="Subsample scenes")):
        entry = subsample_scene(
            scene,
            src_mf[scene],
            data_root=data_root,
            out_traj_root=args.out_traj_root,
            num_frames=args.num_frames,
            q_target=args.q_target,
            seed=args.seed + i,
            dry_run=args.dry_run,
        )
        manifest_entries.append(entry)

    q_values = [e["q_keep"] for e in manifest_entries]
    summary = {
        "profile": "B",
        "seed": args.seed,
        "num_scenes": len(selected),
        "num_frames": args.num_frames,
        "q_target": args.q_target,
        "scenes": selected,
        "per_scene": manifest_entries,
        "q_keep_median": float(statistics.median(q_values)),
        "q_keep_mean": float(statistics.mean(q_values)),
        "q_keep_min": int(min(q_values)),
        "q_keep_max": int(max(q_values)),
        "total_queries": int(sum(q_values)),
        "coordinate_note": (
            "trajs_3d is world coords; frame-0 gating uses w2c projection"
        ),
        "paths": {
            "data_root": data_root,
            "src_json": args.src_json,
            "out_json": args.out_json,
            "out_manifest": args.out_manifest,
            "out_traj_root": args.out_traj_root,
        },
    }

    if not args.dry_run:
        out_json_path = _resolve(data_root, args.out_json)
        out_manifest_path = _resolve(data_root, args.out_manifest)
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

        out_payload = build_subset_json(
            src_mf,
            selected,
            manifest_entries,
            num_frames=args.num_frames,
        )
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        with open(out_manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote JSON: {out_json_path}")
        print(f"Wrote manifest: {out_manifest_path}")

    print(
        f"Summary: scenes={len(selected)}, median_Q={summary['q_keep_median']:.1f}, "
        f"total_queries={summary['total_queries']}"
    )
    return summary


def smoke_test_dataset(
    data_root: str,
    json_rel: str,
    scenes: Sequence[str],
    *,
    num_smoke_scenes: int = 1,
) -> None:
    """Load a few scenes via PointOdysseyTrackDataset (default 1)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from hAlgorithm.datasets_4d.pointodyssey_track_dataset import PointOdysseyTrackDataset

    json_path = _resolve(data_root, json_rel)
    ds = PointOdysseyTrackDataset(
        phase="test",
        name="SynthVerseSubset",
        seed=0,
        data_root=data_root,
        data_path=json_path,
        track_data_path=_resolve(data_root, OUT_TRAJ_ROOT),
        mf_scene=list(scenes[:num_smoke_scenes]),
        mf_to_mv=False,
        clip_maxlen=4,
        test_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5] * 3,
                std=[127.5] * 3,
            ),
        ],
        debug=True,
    )
    print(f"Dataset length: {len(ds)}")
    for idx in range(min(num_smoke_scenes, len(ds))):
        sample = ds[idx]
        tr2d = sample.get("curr_trajs_2d")
        tr3d = sample.get("curr_trajs_3d")
        vis = sample.get("curr_visibs")
        print(
            f"  idx={idx} scene={sample.get('scene')} frame={sample.get('frame_id')} "
            f"trajs_2d={None if tr2d is None else tr2d.shape} "
            f"trajs_3d={None if tr3d is None else tr3d.shape} "
            f"visibs={None if vis is None else vis.shape}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--src-json", default=SRC_JSON)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-manifest", default=OUT_MANIFEST)
    p.add_argument("--out-traj-root", default=OUT_TRAJ_ROOT)
    p.add_argument("--num-scenes", type=int, default=NUM_SCENES)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    p.add_argument("--q-target", type=int, default=Q_TARGET)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--min-scene-candidates",
        type=int,
        default=MIN_SCENE_CANDIDATES,
        help="Prefer scenes with at least this many frame-0 candidates",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true", default=True)
    p.add_argument("--no-smoke-test", dest="smoke_test", action="store_false")
    p.add_argument(
        "--smoke-scenes",
        type=int,
        default=1,
        help="Number of scenes to load in smoke test (default: 1)",
    )
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="Skip build; run smoke test on existing out-json only",
    )
    return p.parse_args()


def _load_smoke_scenes_from_json(data_root: str, json_rel: str) -> List[str]:
    json_path = _resolve(data_root, json_rel)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mf = payload.get("mf_files", payload)
    return sorted(mf.keys())


def main() -> None:
    args = parse_args()
    if args.smoke_only:
        if not os.path.isfile(_resolve(args.data_root, args.out_json)):
            raise SystemExit(
                f"Missing subset JSON: {_resolve(args.data_root, args.out_json)}; "
                "run build without --smoke-only first"
            )
        scenes = _load_smoke_scenes_from_json(args.data_root, args.out_json)
        print(
            f"\nSmoke test only: PointOdysseyTrackDataset on "
            f"{min(args.smoke_scenes, len(scenes))} scene(s)..."
        )
        smoke_test_dataset(
            args.data_root,
            args.out_json,
            scenes,
            num_smoke_scenes=args.smoke_scenes,
        )
        return

    summary = run_build(args)
    if args.smoke_test and not args.dry_run:
        n = min(args.smoke_scenes, len(summary["scenes"]))
        print(f"\nSmoke test: PointOdysseyTrackDataset on {n} scene(s)...")
        smoke_test_dataset(
            args.data_root,
            args.out_json,
            summary["scenes"],
            num_smoke_scenes=args.smoke_scenes,
        )


if __name__ == "__main__":
    main()
