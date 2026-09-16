#!/usr/bin/env python3
"""Self-test for WorldTrack eval metrics (phase 1).

Checks:
  1. GT == Pred  => EPE ~ 0, APD ~ 1
  2. Optional cross-check against Open-d4rt on synthetic + real npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hAlgorithm.eval.worldtrack_json_loader import (  # noqa: E402
    compare_json_vs_npz,
    compute_opend4rt_frame0_query_indices,
    load_sequence_from_json,
    load_sequence_from_npz,
)
from hAlgorithm.utils import file2dict, instantiate_from_config  # noqa: E402
from hAlgorithm.modules.metrics.worldtrack_eval_metrics import (  # noqa: E402
    APD_GLOBAL_KEY,
    EPE_GLOBAL_KEY,
    TAU_GLOBAL_KEY,
    aggregate_results,
    format_subset_summary,
    metrics_for_sequence,
    tracks_cam_to_ref0_world,
)

DEFAULT_JSON = (
    "/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini/"
    "Tapvid3d_mini_extracted/adt_mini_test_mf_with_tracking.json"
)
DEFAULT_NPZ = (
    "/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini/adt_mini/Apartment_release_clean_seq131_0.npz"
)

OPEND4RT_ROOT = Path("/mnt/home/tcchen/workspace/Projects/Open-d4rt")
DEFAULT_NPZ = Path(
    "/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini/adt_mini/Apartment_release_clean_seq131_0.npz"
)


def _load_opend4rt_metrics_module():
    path = OPEND4RT_ROOT / "eval_track3d_in_worldtrack.py"
    if not path.is_file():
        return None
    if str(OPEND4RT_ROOT) not in sys.path:
        sys.path.insert(0, str(OPEND4RT_ROOT))
    spec = importlib.util.spec_from_file_location("opend4rt_worldtrack_eval", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        print(f"[skip] Open-d4rt eval import failed ({exc}); cross-check skipped.")
        return None
    return mod


def _assert_close(name: str, a: float, b: float, atol: float = 1e-9) -> None:
    if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) > atol:
        raise AssertionError(f"{name}: {a} vs {b} (atol={atol})")


def test_gt_equals_pred_synthetic(rng: np.random.Generator) -> Dict[str, Any]:
    gt = rng.normal(size=(64, 32, 3)).astype(np.float64)
    pred = gt.copy()
    metrics = metrics_for_sequence(gt, pred, compute_dyn=True, rng=rng)
    if metrics[EPE_GLOBAL_KEY] > 1e-9:
        raise AssertionError(f"synthetic GT=Pred: EPE={metrics[EPE_GLOBAL_KEY]}")
    if metrics[APD_GLOBAL_KEY] < 1.0 - 1e-9:
        raise AssertionError(f"synthetic GT=Pred: APD={metrics[APD_GLOBAL_KEY]}")
    return metrics


def _load_worldtrack_gt_npz(npz_path: Path, num_frames: int = 64) -> np.ndarray:
    seq = load_sequence_from_npz(npz_path, num_frames=num_frames, load_rgb=False)
    return seq.gt_tracks_world[:, seq.query_indices]


def test_opend4rt_query_indices_match_npz(
    npz_path: Path,
    num_frames: int = 64,
) -> None:
    """Query indices must match Open-d4rt eval loop on the same npz."""
    ref = _load_opend4rt_metrics_module()
    if ref is None:
        print("[skip] Open-d4rt eval module not found; query parity skipped.")
        return

    seq = load_sequence_from_npz(npz_path, num_frames=num_frames, load_rgb=False)
    sample = ref._load_worldtrack_sequence(npz_path, num_frames=num_frames)
    visible_mask = np.asarray(sample["visibility"][0], dtype=bool)
    query_uv = np.asarray(sample["tracks_uv"][0, visible_mask], dtype=np.float64)
    finite_mask = np.isfinite(query_uv).all(axis=-1)
    depth0 = np.asarray(sample["tracks_xyz_cam"][0, visible_mask, 2], dtype=np.float64)
    finite_mask &= np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
    ref_idx = np.flatnonzero(visible_mask)[finite_mask]

    ours = compute_opend4rt_frame0_query_indices(
        seq.tracks_xyz_cam, seq.visibility, seq.intrinsics
    )
    if not np.array_equal(ours, ref_idx):
        raise AssertionError(
            f"query_indices mismatch vs Open-d4rt: ours={ours.size} ref={ref_idx.size}"
        )
    if int(seq.num_frames) != int(sample["tracks_xyz_cam"].shape[0]):
        raise AssertionError(
            f"frame_count mismatch: loader={seq.num_frames} opend4rt={sample['tracks_xyz_cam'].shape[0]}"
        )
    print(f"[ok] Open-d4rt parity: T={seq.num_frames} Q={seq.num_queries}")


def test_gt_equals_pred_real(npz_path: Path, num_frames: int = 64) -> Dict[str, Any]:
    gt = _load_worldtrack_gt_npz(npz_path, num_frames=num_frames)
    metrics = metrics_for_sequence(gt, gt.copy(), compute_dyn=True)
    if metrics[EPE_GLOBAL_KEY] > 1e-9:
        raise AssertionError(f"real GT=Pred EPE={metrics[EPE_GLOBAL_KEY]}")
    if metrics[APD_GLOBAL_KEY] < 1.0 - 1e-9:
        raise AssertionError(f"real GT=Pred APD={metrics[APD_GLOBAL_KEY]}")
    return metrics


def cross_check_opend4rt(
    gt: np.ndarray,
    pred: np.ndarray,
    rng: np.random.Generator,
) -> None:
    ref = _load_opend4rt_metrics_module()
    if ref is None:
        print("[skip] Open-d4rt eval module not found; cross-check skipped.")
        return

    ours = metrics_for_sequence(gt, pred, compute_dyn=True, rng=rng)
    theirs = ref._metrics_for_sequence(gt_tracks_world=gt, pred_tracks_ref0=pred, compute_dyn=True)

    keys = [APD_GLOBAL_KEY, TAU_GLOBAL_KEY, EPE_GLOBAL_KEY, "avg_pts_pertraj", "epe_sim3_closed", "num_queries"]
    for key in keys:
        _assert_close(key, float(ours[key]), float(theirs[key]), atol=1e-12)
    print("[ok] TMA metrics match Open-d4rt on shared inputs.")


def test_dataset_query_indices_match_json(
    json_path: Path,
    data_root: Path,
    config_path: Path,
    sequence_name: str,
    num_frames: int = 64,
) -> None:
    """WorldTrackTrackDataset must use the same frame-0 queries as the JSON loader."""
    seq = load_sequence_from_json(
        json_path,
        sequence_name,
        data_root=data_root,
        num_frames=num_frames,
        load_rgb=False,
    )
    cfg = file2dict(str(config_path))
    basic = dict(cfg.get("data", {}).get("basic", {}))
    pipeline = dict(
        type="hAlgorithm.datasets_4d.worldtrack_track_dataset.WorldTrackTrackDataset",
        phase="test",
        name=json_path.stem,
        seed=0,
        data_root=str(data_root),
        data_path=str(json_path),
        num_frames=int(num_frames),
        sequence_names=[sequence_name],
        mf_to_mv=True,
        only_pointmap=True,
        with_pointmap=False,
        normalize_cameras=True,
        with_global_scale=True,
        track_points_nums=0,
        track_neg_ratio=0.0,
        depth_scale=1.0,
        min_depth=1e-3,
        max_depth=1000.0,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.sequential_sampler.SequentialClipSampler",
            view_num=int(num_frames),
            start_idx=0,
        ),
        test_transforms=basic.get("test_transforms", []),
        pipeline="hAlgorithm/configs/dataset/pipeline_14x/base4d.py",
        debug=False,
    )
    dataset = instantiate_from_config(pipeline)
    sample = dataset.get_mf_data_for_trainval(0)
    got = np.asarray(sample["worldtrack_query_indices"], dtype=np.int64)
    if not np.array_equal(got, seq.query_indices):
        diff = np.setdiff1d(seq.query_indices, got)
        raise AssertionError(
            f"query_indices mismatch: json={seq.query_indices.size} dataset={got.size} "
            f"only_in_json={diff[:8].tolist()}"
        )
    if int(sample["worldtrack_num_queries"]) != int(seq.num_queries):
        raise AssertionError(
            f"num_queries mismatch: json={seq.num_queries} dataset={sample['worldtrack_num_queries']}"
        )
    print(
        f"[ok] dataset query indices match JSON loader ({int(got.size)} queries, {sequence_name})"
    )


def cross_check_scaled_pred(rng: np.random.Generator) -> None:
    ref = _load_opend4rt_metrics_module()
    if ref is None:
        return
    gt = rng.normal(size=(64, 20, 3)).astype(np.float64)
    pred = gt * 2.5 + rng.normal(scale=0.02, size=gt.shape)
    cross_check_opend4rt(gt, pred, rng)


def main() -> int:
    parser = argparse.ArgumentParser(description="WorldTrack metrics phase-1 self-test.")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ), help="WorldTrack npz for real GT=Pred test.")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-opend4rt", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    print("=== 1/4 synthetic GT=Pred ===")
    syn = test_gt_equals_pred_synthetic(rng)
    print(
        f"  EPE={syn[EPE_GLOBAL_KEY]:.6f}  APD={syn[APD_GLOBAL_KEY]:.6f}  "
        f"queries={syn['num_queries']}"
    )

    if not args.skip_opend4rt:
        print("=== 2/4 cross-check Open-d4rt (scaled pred) ===")
        cross_check_scaled_pred(rng)

    if not args.skip_real:
        npz_path = Path(args.npz)
        print(f"=== 3/4 real GT=Pred ({npz_path.name}) ===")
        if not npz_path.is_file():
            print(f"[skip] npz not found: {npz_path}")
        else:
            real = test_gt_equals_pred_real(npz_path, num_frames=int(args.num_frames))
            print(
                f"  EPE={real[EPE_GLOBAL_KEY]:.6f}  APD={real[APD_GLOBAL_KEY]:.6f}  "
                f"queries={real['num_queries']}  dyn={real['dyn_count']}"
            )
            if not args.skip_opend4rt and npz_path.is_file():
                print("=== 4/4 cross-check Open-d4rt (real GT, scaled pred) ===")
                gt = _load_worldtrack_gt_npz(npz_path, num_frames=int(args.num_frames))
                pred = gt * 1.7 + rng.normal(scale=0.01, size=gt.shape)
                cross_check_opend4rt(gt, pred, rng)

    print("=== aggregate smoke test ===")
    summary = aggregate_results([syn, syn])
    print(format_subset_summary("synthetic", summary))

    print("=== JSON vs NPZ alignment (phase 2) ===")
    json_path = Path(args.npz).parent.parent / "Tapvid3d_mini_extracted/adt_mini_test_mf_with_tracking.json"
    if not json_path.is_file():
        json_path = Path(DEFAULT_JSON)
    npz_path = Path(args.npz) if Path(args.npz).is_file() else Path(DEFAULT_NPZ)
    if json_path.is_file() and npz_path.is_file():
        seq = "Apartment_release_clean_seq131_0"
        nf = int(args.num_frames)
        report = compare_json_vs_npz(
            json_path, npz_path, seq, data_root=npz_path.parent.parent, num_frames=nf
        )
        print(json.dumps(report, indent=2))
        if not report.get("aligned", False):
            raise AssertionError("JSON loader does not match NPZ/Open-d4rt GT")
        print("=== Open-d4rt query / frame parity (npz) ===")
        test_opend4rt_query_indices_match_npz(npz_path, num_frames=nf)
    else:
        print(f"[skip] json={json_path} npz={npz_path}")

    print("=== dataset vs JSON query indices (phase 2b) ===")
    if json_path.is_file():
        config_path = REPO_ROOT.parent / "TMA/results_D4RT_v2/test/wfm/config.py"
        if not config_path.is_file():
            config_path = Path(
                "/mnt/home/tcchen/workspace/TMA/results_D4RT_v2/test/wfm/config.py"
            )
        if config_path.is_file():
            test_dataset_query_indices_match_json(
                json_path,
                data_root=npz_path.parent.parent,
                config_path=config_path,
                sequence_name="Apartment_release_clean_seq131_0",
                num_frames=int(args.num_frames),
            )
        else:
            print(f"[skip] TMA config not found: {config_path}")
    else:
        print(f"[skip] json not found: {json_path}")

    print("All phase-1/2 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
