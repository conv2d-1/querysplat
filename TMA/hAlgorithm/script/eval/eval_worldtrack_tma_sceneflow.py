#!/usr/bin/env python3
"""Evaluate TMA (WFMQueryPipeline) on WorldTrack / Tapvid3d JSON with scene flow metrics.

This script is intentionally isolated from `eval_worldtrack_tma.py` (Dynamic Points / warp3d 3D):
it reuses the same data loading and inference, but metrics are **scene flow** on ``warp3d_delta``
(``avg_sf_global``, ``epe_sf_global``). Do not use ``epe_sf_global_dyn`` as Dynamic Points EPE.
See ``docs/WORLDTRACK_SCENEFLOW_AND_DYNAMIC_POINTS_EVAL.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hAlgorithm.eval.worldtrack_json_loader import (  # noqa: E402
    align_batch_queries_with_sequence,
    compare_json_vs_npz,
    filter_sequences_with_frame0_queries,
    list_sequences_in_json,
    load_sequence_from_json,
    prefer_worldtrack_json_path,
)
from hAlgorithm.eval.worldtrack_tma_infer import (  # noqa: E402
    Pred3DSource,
    _to_np,
    cam_space_epe_vs_json,
    extract_pred_tracks_cam_from_outputs,
    extract_pred_tracks_ref0_from_outputs,
    extract_tma_gt_cam_from_outputs,
    get_global_scale_from_batch,
    get_metric_extrinsics_for_sequence,
    pred_field_label,
    ref0_space_epe,
    resolve_pred_3d_source,
    summarize_pred_alignment,
    worldtrack_query_bank_override,
)
from hAlgorithm.eval.worldtrack_vis import (  # noqa: E402
    export_worldtrack_frame0_vis_ply,
    save_tracks_package,
    visualize_tracks_package,
)
from hAlgorithm.modules.metrics.worldtrack_sceneflow_eval_metrics import (  # noqa: E402
    SF_APD_GLOBAL_KEY,
    SF_EPE_GLOBAL_KEY,
    aggregate_results,
    format_subset_summary,
    metrics_for_sequence_scene_flow,
)
from hAlgorithm.utils import file2dict, instantiate_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldTrack scene flow eval for TMA (ref0-based).")
    p.add_argument("--config", required=True, help="TMA config.py (e.g. test/wfm/config.py).")
    p.add_argument("--load-from", required=True, help="Checkpoint .pth path.")
    p.add_argument(
        "--json",
        action="append",
        dest="json_paths",
        required=True,
        help="mf_files JSON path. Repeat for multiple subsets.",
    )
    p.add_argument(
        "--subset-name",
        action="append",
        dest="subset_names",
        help="Name per --json (default: json stem).",
    )
    p.add_argument("--data-root", default=None, help="Dataset root for relative paths in JSON.")
    p.add_argument(
        "--coord-convention",
        default="opend4rt",
        choices=("opend4rt", "pointodyssey"),
        help=(
            "mf_files coordinate layout: 'opend4rt' = ref0 trajs_3d + w2c extrinsics (TapVid3D); "
            "'pointodyssey' = world trajs_3d + c2w extrinsics (PO / SynthVerse)."
        ),
    )
    p.add_argument("--output-dir", default="tmp/eval_worldtrack_tma_sceneflow")
    p.add_argument("--num-frames", type=int, default=64)
    p.add_argument("--limit-seqs", type=int, default=1, help="<=0 means all sequences (default: 1).")
    p.add_argument("--seq-name", default="", help="Run only this sequence (overrides --limit-seqs ordering).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--verify-npz", default="", help="Optional npz path to verify JSON loader.")
    p.add_argument("--verify-seq", default="", help="Sequence name for --verify-npz.")
    p.add_argument("--save-per-sequence", action="store_true")
    p.add_argument("--mixed-precision", default="fp16", choices=("fp16", "bf16", "no"))
    p.add_argument(
        "--pred-3d-source",
        default="warp3d",
        choices=("warp3d", "warp3d_delta"),
        help=(
            "How to read per-pair 3D predictions: 'warp3d' = model absolute ref0 point; "
            "'warp3d_delta' = src_points + warp3d_delta (requires both). Default: warp3d."
        ),
    )
    p.add_argument(
        "--use-warp3d-delta",
        action="store_true",
        default=False,
        help="Shortcut for --pred-3d-source warp3d_delta (overrides --pred-3d-source if set).",
    )
    p.add_argument("--visualize", action="store_true", help="Save tracks package per sequence (optional).")
    p.add_argument(
        "--vis-frame0-pointmap-ply",
        action="store_true",
        help="Export frame-0 PLY (same as tracking eval; optional).",
    )
    p.add_argument("--vis-max-points", type=int, default=300)
    p.add_argument("--vis-fps", type=float, default=15.0)
    p.add_argument("--resume", action="store_true", help="Skip sequences that already have per-sequence JSON.")
    args = p.parse_args()
    args.pred_3d_source = resolve_pred_3d_source(
        pred_3d_source=args.pred_3d_source, use_warp3d_delta=bool(args.use_warp3d_delta)
    )
    return args


def _build_dataset(
    json_path: Path,
    data_root: Path,
    cfg: dict,
    num_frames: int,
    sequence_names: Optional[List[str]] = None,
    coord_convention: str = "opend4rt",
):
    basic = dict(cfg.get("data", {}).get("basic", {}))
    test_transforms = basic.get("test_transforms", [])
    pipeline = dict(
        type="hAlgorithm.datasets_4d.worldtrack_track_dataset.WorldTrackTrackDataset",
        phase="test",
        name=json_path.stem,
        seed=0,
        data_root=str(data_root),
        data_path=str(json_path),
        num_frames=int(num_frames),
        coord_convention=coord_convention,
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
        test_transforms=test_transforms,
        pipeline="hAlgorithm/configs/dataset/pipeline_14x/base4d.py",
        debug=False,
    )
    if sequence_names:
        pipeline["sequence_names"] = list(sequence_names)
    return instantiate_from_config(pipeline)


def _load_model(cfg: dict, ckpt_path: Path, device: torch.device):
    model = instantiate_from_config(cfg["model"])
    assert model is not None
    model.load_checkpoint(str(ckpt_path))
    model.to(device)
    if not hasattr(model, "device"):
        model.device = device
    if not hasattr(model, "dtype"):
        model.dtype = torch.float32
    model.eval()
    return model


def _resolve_inner_model(model):
    inner = getattr(model, "model", model)
    if hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod
    return inner


@contextmanager
def _worldtrack_skip_redundant_warp3d_gather() -> Iterator[None]:
    orig_gather = torch.gather

    def _gather(input, dim, index, *, out=None):
        if (
            dim == 1
            and input.ndim == 3
            and index.ndim == 3
            and index.shape[1] == input.shape[1]
            and int(index.max()) >= int(input.shape[1])
        ):
            return input
        return orig_gather(input, dim, index, out=out)

    torch.gather = _gather
    try:
        yield
    finally:
        torch.gather = orig_gather


@contextmanager
def _worldtrack_infer_compat(model) -> Iterator[None]:
    inner = _resolve_inner_model(model)
    orig_forward = inner.forward

    def _forward(*args, **kwargs):
        results = orig_forward(*args, **kwargs)
        if isinstance(results, dict):
            results = dict(results)
            for key in ("depth", "confidence", "global_points", "global_confidence"):
                results.pop(key, None)
            query = results.get("query")
            if query is not None and getattr(query, "uv", None) is not None:
                uv = query.uv
                if uv.ndim == 5:
                    query.uv = uv.squeeze(1)
        return results

    inner.forward = _forward
    try:
        yield
    finally:
        inner.forward = orig_forward


def _infer_batch(model, batch: dict, mixed_precision: str):
    if mixed_precision == "fp16":
        batch["use_amp"] = True
        batch["amp_dtype"] = torch.float16
    elif mixed_precision == "bf16":
        batch["use_amp"] = True
        batch["amp_dtype"] = torch.bfloat16
    else:
        batch["use_amp"] = False
        batch["amp_dtype"] = torch.float32

    with (
        torch.inference_mode(),
        worldtrack_query_bank_override(model, batch),
        _worldtrack_infer_compat(model),
        _worldtrack_skip_redundant_warp3d_gather(),
    ):
        return model.infer(**batch)


def _infer_batch_dense_frame0(model, batch: dict, mixed_precision: str):
    batch = dict(batch)
    for k in (
        "trajs_2d",
        "trajs_3d",
        "visibs",
        "valids",
        "worldtrack_query_indices",
        "worldtrack_num_queries",
    ):
        batch.pop(k, None)

    if mixed_precision == "fp16":
        batch["use_amp"] = True
        batch["amp_dtype"] = torch.float16
    elif mixed_precision == "bf16":
        batch["use_amp"] = True
        batch["amp_dtype"] = torch.bfloat16
    else:
        batch["use_amp"] = False
        batch["amp_dtype"] = torch.float32

    with (
        torch.inference_mode(),
        _worldtrack_infer_compat(model),
        _worldtrack_skip_redundant_warp3d_gather(),
    ):
        return model.infer(**batch)


def _extract_frame0_model_rgb_uint8(batch: Dict[str, Any]) -> np.ndarray:
    for key in ("image_show", "image"):
        tensor = batch.get(key)
        if tensor is None:
            continue
        arr = _to_np(tensor)
        if arr.ndim == 5:
            arr = arr[0, 0]
        elif arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr_f = np.asarray(arr, dtype=np.float64)
            if float(np.nanmax(arr_f)) <= 1.0 + 1e-3:
                arr = (np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
        return arr
    raise ValueError("batch has no image_show/image for PLY coloring")


def evaluate_subset(
    *,
    model,
    dataset,
    json_path: Path,
    data_root: Path,
    subset_name: str,
    num_frames: int,
    limit_seqs: int,
    seq_name: str,
    mixed_precision: str,
    pred_3d_source: Pred3DSource,
    save_per_sequence: bool,
    visualize: bool,
    vis_frame0_pointmap_ply: bool,
    vis_max_points: int,
    vis_fps: float,
    output_dir: Path,
    coord_convention: str = "opend4rt",
) -> Dict[str, Any]:
    collate_fn = dataset.get_collate_fn() if hasattr(dataset, "get_collate_fn") else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    results: List[Dict[str, Any]] = []
    if hasattr(dataset, "sequence_names") and dataset.sequence_names is not None:
        seq_names = list(dataset.sequence_names)
    else:
        seq_names = list_sequences_in_json(json_path)

    if seq_name:
        if seq_name not in seq_names:
            raise ValueError(f"Sequence {seq_name!r} not in {json_path}")
        seq_names = [seq_name]
    elif limit_seqs > 0:
        seq_names = seq_names[: int(limit_seqs)]

    subset_out = output_dir / subset_name
    subset_out.mkdir(parents=True, exist_ok=True)

    valid_names, skipped_no_query = filter_sequences_with_frame0_queries(
        json_path,
        seq_names,
        data_root=data_root,
        num_frames=num_frames,
        coord_convention=coord_convention,
    )
    if skipped_no_query:
        logging.warning(
            "Skipping %d sequences with no frame-0 visible queries: %s",
            len(skipped_no_query),
            ", ".join(skipped_no_query),
        )
        (subset_out / "skipped_no_frame0_queries.json").write_text(
            json.dumps({"sequences": skipped_no_query}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    seq_names = valid_names
    if not seq_names:
        logging.warning("No sequences left to evaluate for subset %s", subset_name)
        summary = aggregate_results(results)
        summary["sequences"] = results
        summary["skipped_no_frame0_queries"] = skipped_no_query
        (subset_out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    if hasattr(dataset, "sequence_names"):
        dataset.sequence_names = list(seq_names)
        dataset.data_infos = dataset.get_data_infos()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= len(seq_names):
            break
        seq_name_i = seq_names[batch_idx]

        # Use real frame-0 resolution for query/GT (loader reads RGB header when load_rgb=False).
        seq_gt = load_sequence_from_json(
            json_path,
            seq_name_i,
            data_root=data_root,
            num_frames=num_frames,
            load_rgb=False,
            coord_convention=coord_convention,
        )
        num_frames_eff = int(seq_gt.num_frames)
        query_idx = align_batch_queries_with_sequence(
            batch, seq_gt, sequence_name=seq_name_i
        )
        q_count = int(query_idx.shape[0])

        outputs_dense = None
        if vis_frame0_pointmap_ply:
            outputs_dense = _infer_batch_dense_frame0(model, batch, mixed_precision)

        outputs = _infer_batch(model, batch, mixed_precision)
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[-1], dict) and outputs[-1].get("type") is not None:
            batch = outputs.pop()

        ext_metric = get_metric_extrinsics_for_sequence(seq_gt, num_frames_eff)
        gscale = get_global_scale_from_batch(batch)

        pred_q = extract_pred_tracks_ref0_from_outputs(
            outputs,
            query_indices=np.arange(q_count, dtype=np.int64),
            num_frames=num_frames_eff,
            global_scale=gscale,
            extrinsics_w2c_metric=ext_metric,
            pred_3d_source=pred_3d_source,
        )
        pred_cam = extract_pred_tracks_cam_from_outputs(
            outputs,
            num_frames=num_frames_eff,
            q_count=q_count,
            extrinsics_w2c_metric=ext_metric,
            pred_3d_source=pred_3d_source,
        )

        gt_q = seq_gt.gt_tracks_world[:num_frames_eff, query_idx]
        pred_q = pred_q[:, :q_count]
        pred_cam = pred_cam[:, :q_count]

        # Scene flow prediction source:
        # use model `warp3d_delta` directly (per-pair displacement in ref0), not X(t)-X(0).
        pred_flow = np.full((num_frames_eff, q_count, 3), np.nan, dtype=np.float64)
        pred_flow[0] = 0.0
        for t in range(1, int(num_frames_eff)):
            track = None
            # Reuse internal finder from worldtrack_tma_infer through the public extractor path:
            # outputs is a list of per-view ReconstructOutput, each has `track_3d` dict for src_index.
            # Here, src_index is 0 and we need the (0,t) track.
            src_out = outputs[0] if len(outputs) > 0 else None
            track_dict = getattr(src_out, "track_3d", None) if src_out is not None else None
            if isinstance(track_dict, dict):
                track = track_dict.get(t, track_dict.get(int(t), None))
                if track is None:
                    for val in track_dict.values():
                        if getattr(val, "src_index", None) == 0 and getattr(val, "tgt_index", None) == t:
                            track = val
                            break
            if track is None:
                raise RuntimeError(f"WorldTrack scene flow eval: missing model track (src=0, tgt={t}).")
            delta = _to_np(getattr(track, "warp3d_delta", None))
            if delta is None:
                raise RuntimeError(
                    f"WorldTrack scene flow eval: missing model warp3d_delta for (src=0, tgt={t})."
                )
            delta = np.asarray(delta, dtype=np.float64)
            n = min(int(delta.shape[0]), q_count)
            pred_flow[t, :n] = delta[:n]

        align = summarize_pred_alignment(gt_q, pred_q)
        gt_cam_json = seq_gt.tracks_xyz_cam[:num_frames_eff, query_idx]
        gt_cam_tma = extract_tma_gt_cam_from_outputs(outputs, num_frames=num_frames_eff, q_count=q_count)
        epe_json_cam = cam_space_epe_vs_json(pred_cam, gt_cam_json)
        epe_ref0_vs_gt = ref0_space_epe(pred_q, gt_q)
        fin_tma = np.isfinite(pred_q) & np.isfinite(gt_cam_tma)
        epe_tma_tgt = (
            float(np.linalg.norm(pred_q[fin_tma] - gt_cam_tma[fin_tma], axis=-1).mean())
            if np.any(fin_tma)
            else float("nan")
        )

        metrics = metrics_for_sequence_scene_flow(
            gt_tracks_world=gt_q,
            pred_flow_ref0=pred_flow,
            pred_tracks_ref0=pred_q,
            compute_dyn=True,
        )
        metrics["sanity_epe_ref0_vs_gt"] = epe_ref0_vs_gt
        metrics["sanity_epe_cam_vs_json"] = epe_json_cam
        metrics["sanity_epe_ref0_vs_tma_tgt"] = epe_tma_tgt
        metrics["epe_after_scale_position"] = float(align.get("epe_after_scale_global", float("nan")))
        metrics["pred_3d_source"] = "warp3d_delta"
        metrics["pred_field"] = "warp3d_delta"
        metrics["video_name"] = seq_name_i
        metrics["num_queries"] = q_count
        metrics["num_queries_json"] = int(seq_gt.num_queries)
        metrics["num_frames_eff"] = num_frames_eff
        results.append(metrics)

        logging.info(
            "%s | %s | T=%d Q=%d | SF-APD/tau=%.4f SF-EPE=%.4f | "
            "scale_pos=%.4f scale_flow=%.4f pred_dyn=%d | ref0-vs-GT=%.3f pos-after-scale=%.3f "
            "cam-vs-JSON=%.3f ref0-vs-tmaGT=%.3f field=%s",
            subset_name,
            seq_name_i,
            num_frames_eff,
            metrics["num_queries"],
            float(metrics.get(SF_APD_GLOBAL_KEY, float("nan"))),
            float(metrics.get(SF_EPE_GLOBAL_KEY, float("nan"))),
            float(metrics.get("scale_global", float("nan"))),
            float(metrics.get("scale_global_flow_pred_dynamic", float("nan"))),
            int(metrics.get("pred_dyn_count", 0)),
            float(metrics.get("sanity_epe_ref0_vs_gt", float("nan"))),
            float(metrics.get("epe_after_scale_position", float("nan"))),
            float(metrics.get("sanity_epe_cam_vs_json", float("nan"))),
            float(metrics.get("sanity_epe_ref0_vs_tma_tgt", float("nan"))),
            metrics.get("pred_field", "warp3d"),
        )

        if save_per_sequence:
            (subset_out / f"{seq_name_i}.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        if visualize:
            seq_vis = load_sequence_from_json(
                json_path,
                seq_name_i,
                data_root=data_root,
                num_frames=num_frames_eff,
                load_rgb=True,
                coord_convention=coord_convention,
            )
            pred_aligned = pred_q * float(metrics.get("scale_global", 1.0))
            vis_dir = subset_out / seq_name_i
            save_tracks_package(
                vis_dir,
                video_rgb=seq_vis.video_rgb[:num_frames_eff],
                gt_tracks_world=gt_q,
                pred_tracks_ref0=pred_aligned,
                pred_tracks_ref0_raw=pred_q,
                visibility_tq=seq_vis.visibility[:num_frames_eff, query_idx],
                extrinsics_w2c=seq_vis.extrinsics_w2c[:num_frames_eff],
                intrinsics=seq_vis.intrinsics,
                gt_tracks_uv_pixel=seq_vis.tracks_uv[:num_frames_eff, query_idx],
                metrics=metrics,
                export_ply=True,
            )
            vis_paths = visualize_tracks_package(vis_dir, max_points=vis_max_points, fps=vis_fps)
            logging.info("Visualization: %s -> %s", vis_dir / "vis", vis_paths)

        if vis_frame0_pointmap_ply:
            if outputs_dense is None:
                raise RuntimeError("vis-frame0-pointmap-ply requires dense infer outputs.")
            model_rgb0 = _extract_frame0_model_rgb_uint8(batch)
            frame0_dir = (subset_out / seq_name_i / "vis") if (visualize or save_per_sequence) else subset_out
            frame0_dir.mkdir(parents=True, exist_ok=True)
            frame0_ply = frame0_dir / "frame0_vis_sparse_gt_dense_pred.ply"
            meta = export_worldtrack_frame0_vis_ply(
                frame0_ply,
                gt_tracks_ref0_frame0=gt_q[0],
                pred_tracks_ref0_frame0=pred_q[0],
                outputs_dense=outputs_dense,
                video_rgb_frame0=model_rgb0,
                align_pred=True,
            )
            (frame0_dir / "frame0_vis_sparse_gt_dense_pred.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    if save_per_sequence:
        all_results: List[Dict[str, Any]] = []
        for path in sorted(subset_out.glob("*.json")):
            if path.name in ("summary.json", "skipped_no_frame0_queries.json"):
                continue
            all_results.append(json.loads(path.read_text(encoding="utf-8")))
        summary = aggregate_results(all_results)
        summary["sequences"] = all_results
    else:
        summary = aggregate_results(results)
        summary["sequences"] = results
    if skipped_no_query:
        summary["skipped_no_frame0_queries"] = skipped_no_query
    (subset_out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.verify_npz:
        seq = args.verify_seq or list_sequences_in_json(args.json_paths[0])[0]
        report = compare_json_vs_npz(
            args.json_paths[0],
            args.verify_npz,
            seq,
            data_root=args.data_root,
            num_frames=args.num_frames,
        )
        print(json.dumps(report, indent=2))
        if not report.get("aligned", False):
            logging.warning("JSON vs NPZ verification reported mismatch (see above).")

    cfg = file2dict(args.config)
    logging.info("Prediction source: %s (%s)", args.pred_3d_source, pred_field_label(args.pred_3d_source))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, Path(args.load_from), device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root_for_json = Path(args.data_root) if args.data_root else None
    json_paths = [prefer_worldtrack_json_path(Path(p), data_root=data_root_for_json) for p in args.json_paths]
    if args.subset_names:
        if len(args.subset_names) != len(json_paths):
            raise ValueError("--subset-name count must match --json count")
        subset_names = list(args.subset_names)
    else:
        subset_names = [p.stem.replace("_test_mf_with_tracking", "") for p in json_paths]

    all_summary: Dict[str, Any] = {
        "inputs": {
            "config": args.config,
            "load_from": args.load_from,
            "num_frames": int(args.num_frames),
            "pred_3d_source": args.pred_3d_source,
            "pred_field": pred_field_label(args.pred_3d_source),
            "coord_convention": args.coord_convention,
            "json_paths": [str(p) for p in json_paths],
        },
        "subsets": {},
    }

    lines = ["WorldTrack TMA SceneFlow Summary"]
    for json_path, subset_name in zip(json_paths, subset_names):
        data_root = Path(args.data_root) if args.data_root else json_path.parent.parent
        seq_filter: Optional[List[str]] = None
        if args.seq_name:
            seq_filter = [args.seq_name]
        elif args.limit_seqs > 0:
            names = list_sequences_in_json(json_path)
            seq_filter = names[: int(args.limit_seqs)]
        else:
            seq_filter = list_sequences_in_json(json_path)

        if args.resume:
            subset_out = output_dir / subset_name
            done = {
                p.stem
                for p in subset_out.glob("*.json")
                if p.name not in ("summary.json", "skipped_no_frame0_queries.json")
            }
            if done:
                before = len(seq_filter)
                seq_filter = [s for s in seq_filter if s not in done]
                logging.info("Resume: skipping %d/%d sequences already in %s", before - len(seq_filter), before, subset_out)

        dataset = _build_dataset(
            json_path,
            data_root,
            cfg,
            args.num_frames,
            sequence_names=seq_filter,
            coord_convention=args.coord_convention,
        )
        summary = evaluate_subset(
            model=model,
            dataset=dataset,
            json_path=json_path,
            data_root=data_root,
            subset_name=subset_name,
            num_frames=args.num_frames,
            limit_seqs=args.limit_seqs,
            seq_name=args.seq_name,
            mixed_precision=args.mixed_precision,
            pred_3d_source=args.pred_3d_source,
            save_per_sequence=args.save_per_sequence,
            visualize=args.visualize,
            vis_frame0_pointmap_ply=args.vis_frame0_pointmap_ply,
            vis_max_points=args.vis_max_points,
            vis_fps=args.vis_fps,
            output_dir=output_dir,
            coord_convention=args.coord_convention,
        )
        all_summary["subsets"][subset_name] = summary
        lines.append(format_subset_summary(subset_name, summary))

    (output_dir / "summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

