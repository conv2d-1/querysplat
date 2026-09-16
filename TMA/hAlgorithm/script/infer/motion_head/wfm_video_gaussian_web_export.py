"""Export sparse 4DGS inference results for web / WebGL splat viewers.

Produces:
  web_viewer/manifest.json   — timeline, cameras, asset pointers
  web_viewer/frames/*.ply    — standard 3DGS PLY per time step
  web_viewer/scene_flow.npz  — ref means + per-frame displacements (optional interpolation)
  web_viewer/README.md       — loader notes for common viewers
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_GS_RESULT_KEYS = (
    "gs_opacity",
    "gs_scale",
    "gs_rotation",
    "gs_sh",
    "sparse_global_points",
    "sparse_scene_flow",
    "src_frame_idx",
    "query",
)


def attach_model_results_capture(pipeline) -> None:
    """Hook pipeline to retain sparse Gaussian tensors after model forward."""

    def _on_infer_model_results(_self, results: dict) -> None:
        captured = {}
        for key in _GS_RESULT_KEYS:
            if key not in results:
                continue
            val = results[key]
            if isinstance(val, torch.Tensor):
                captured[key] = val.detach().cpu()
            else:
                captured[key] = val
        _self._captured_gs_results = captured if captured else None

    import types

    pipeline._on_infer_model_results = types.MethodType(_on_infer_model_results, pipeline)
    pipeline._captured_gs_results = None


def _matrix_to_list(m: np.ndarray) -> list:
    return np.asarray(m, dtype=np.float64).tolist()


def _intrinsics_to_dict(k: np.ndarray, width: int, height: int) -> dict:
    k = np.asarray(k, dtype=np.float64)
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "K": _matrix_to_list(k),
    }


def _bootstrap_sh_from_image(
    gs_sh: torch.Tensor,
    query,
    image: torch.Tensor,
    ref_frame: int,
) -> torch.Tensor:
    """Fill SH DC from input RGB at query UVs (same as infer render)."""
    if query is None or not hasattr(query, "uv"):
        return gs_sh
    uv = query.uv.float()
    if uv.device != image.device:
        uv = uv.to(image.device)
    if uv.dim() == 2:
        uv = uv.unsqueeze(0)
    if uv.dim() == 4:
        uv = uv[:, ref_frame]
    rgb = (image[0, ref_frame].float() + 1.0) * 0.5
    grid = uv * 2.0 - 1.0
    grid = grid.unsqueeze(2)
    sampled = F.grid_sample(
        rgb.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(-1).permute(0, 2, 1)
    c0 = 0.28209479177387814
    gs_sh = gs_sh.clone()
    gs_sh[..., 0, :] = (sampled - 0.5) / c0
    return gs_sh


def export_web_viewer_bundle(
    *,
    captured: dict,
    outputs_list: list,
    image: torch.Tensor,
    out_dir: str,
    sequence_name: str,
    fps: float = 24.0,
    max_ply_points: Optional[int] = 8192,
    apply_viewer_coord_transform: bool = True,
) -> str | None:
    """Write ``web_viewer/`` export next to render outputs."""
    required = ("gs_opacity", "gs_scale", "gs_rotation", "gs_sh",
                "sparse_global_points", "sparse_scene_flow")
    if not all(k in captured for k in required):
        logger.warning("Web export skipped: missing sparse Gaussian tensors.")
        return None

    from hAlgorithm.modules.models2.gaussians.infinidepth_ply import export_ply
    from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
        align_sparse_query_geometry,
        build_sparse_dynamic_gaussians,
        canonicalize_scene_flow,
        compose_frame_means,
        squeeze_query_points,
        subsample_query_indices,
    )

    web_root = os.path.join(out_dir, "web_viewer")
    frames_dir = os.path.join(web_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    ref_frame = int(captured.get("src_frame_idx", 0))
    means_ref = squeeze_query_points(captured["sparse_global_points"].float())
    displacements = canonicalize_scene_flow(captured["sparse_scene_flow"].float())
    means_ref, displacements = align_sparse_query_geometry(
        means_ref, displacements, captured["gs_opacity"],
    )

    gs_opacity = captured["gs_opacity"]
    gs_scale = captured["gs_scale"]
    gs_rotation = captured["gs_rotation"]
    gs_sh = captured["gs_sh"]

    num_frames = int(displacements.shape[1])
    q = min(means_ref.shape[1], gs_opacity.shape[1], displacements.shape[2])

    query = captured.get("query")
    if query is not None and image is not None:
        img = image.cuda() if torch.cuda.is_available() and not image.is_cuda else image
        gs_sh = gs_sh.to(img.device)
        gs_sh = _bootstrap_sh_from_image(gs_sh, query, img, ref_frame)
        gs_sh = gs_sh.cpu()

    query_idx = subsample_query_indices(q, max_ply_points or q, means_ref.device)

    means_ref = means_ref[:, query_idx]
    displacements = displacements[:, :, query_idx]
    gs_opacity = gs_opacity[:, query_idx]
    gs_scale = gs_scale[:, query_idx]
    gs_rotation = gs_rotation[:, query_idx]
    gs_sh = gs_sh[:, query_idx]

    gs_opacity_delta = captured.get("gs_opacity_delta")
    gs_scale_delta = captured.get("gs_scale_delta")
    gs_rotation_delta = captured.get("gs_rotation_delta")
    if gs_opacity_delta is not None:
        gs_opacity_delta = gs_opacity_delta[:, :, query_idx]
        gs_scale_delta = gs_scale_delta[:, :, query_idx]
        gs_rotation_delta = gs_rotation_delta[:, :, query_idx]

    height = width = None
    if outputs_list and getattr(outputs_list[0], "dgs_render_rgb", None) is not None:
        height, width = outputs_list[0].dgs_render_rgb.shape[:2]

    cameras: list[dict[str, Any]] = []
    for t, out in enumerate(outputs_list):
        k = getattr(out, "intrinsics_pred", None)
        if k is None:
            k = getattr(out, "intrinsics", None)
        w2c = getattr(out, "extrinsics_pred", None)
        if w2c is None:
            w2c = getattr(out, "extrinsics", None)
        if k is None or w2c is None:
            continue
        k = np.asarray(k, dtype=np.float64)
        w2c = np.asarray(w2c, dtype=np.float64)
        h_t = height or int(k[1, 2] * 2) if k[1, 2] > 0 else 504
        w_t = width or int(k[0, 2] * 2) if k[0, 2] > 0 else 896
        try:
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError:
            c2w = np.eye(4)
        cameras.append({
            "frame": t,
            **_intrinsics_to_dict(k, w_t, h_t),
            "w2c": _matrix_to_list(w2c),
            "c2w": _matrix_to_list(c2w),
        })

    ply_paths: list[str] = []
    for t in range(num_frames):
        means_t = compose_frame_means(means_ref, displacements, t)
        gaussians = build_sparse_dynamic_gaussians(
            gs_opacity=gs_opacity,
            gs_scale=gs_scale,
            gs_rotation=gs_rotation,
            gs_sh=gs_sh,
            means=means_t,
            gs_opacity_delta=gs_opacity_delta,
            gs_scale_delta=gs_scale_delta,
            gs_rotation_delta=gs_rotation_delta,
            target_frame_idx=t,
        )
        ply_name = f"{t:06d}.ply"
        ply_path = os.path.join(frames_dir, ply_name)
        export_ply(
            means=gaussians.means[0],
            harmonics=gaussians.harmonics[0],
            opacities=gaussians.opacities[0],
            path=ply_path,
            scales=gaussians.scales[0],
            rotations=gaussians.rotations[0],
            shift_to_center=False,
            apply_coordinate_transform=apply_viewer_coord_transform,
        )
        ply_paths.append(ply_name)

    scene_flow_path = os.path.join(web_root, "scene_flow.npz")
    np.savez_compressed(
        scene_flow_path,
        means_ref=means_ref.numpy(),
        displacements=displacements.numpy(),
        ref_frame=np.int32(ref_frame),
        gs_opacity=gs_opacity.numpy(),
        gs_scale=gs_scale.numpy(),
        gs_rotation=gs_rotation.numpy(),
        gs_sh=gs_sh.numpy(),
    )

    manifest = {
        "format": "wfm_sparse_4dgs_v1",
        "sequence": sequence_name,
        "description": (
            "Per-frame standard 3DGS PLY + camera timeline + scene-flow NPZ. "
            "Load manifest.json in a custom WebGL timeline viewer, or import "
            "individual frames/*.ply into antimatter15/splat, SuperSplat, etc."
        ),
        "compatible_viewers": [
            "https://github.com/antimatter15/splat (static PLY per frame)",
            "https://github.com/playcanvas/supersplat (static PLY per frame)",
            "custom WebGL 4DGS loader (manifest + scene_flow.npz for interpolation)",
        ],
        "num_frames": num_frames,
        "fps": float(fps),
        "ref_frame": ref_frame,
        "num_gaussians": int(query_idx.numel()),
        "coordinate_note": (
            "PLY uses apply_coordinate_transform=True (x90°) when enabled for "
            "common WebGL viewers; cameras remain in model w2c/c2w space."
        ),
        "gaussians": {
            "mode": "per_frame_ply",
            "dir": "frames",
            "pattern": "{frame:06d}.ply",
            "files": ply_paths,
        },
        "scene_flow": {
            "file": "scene_flow.npz",
            "means_ref": "float32 [1, Q, 3] reference positions",
            "displacements": "float32 [1, T, Q, 3] per-frame offsets from ref",
            "usage": "means_t = means_ref + displacements[:, t]; interpolate t for arbitrary time",
        },
        "cameras": cameras,
    }

    manifest_path = os.path.join(web_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme = os.path.join(web_root, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            f"# Web viewer export — {sequence_name}\n\n"
            "## Static splat viewers (single frame)\n"
            "Open any `frames/XXXXXX.ply` in [antimatter15/splat](https://github.com/antimatter15/splat) "
            "or [SuperSplat](https://playcanvas.com/supersplat).\n\n"
            "## Timeline / arbitrary time & view\n"
            "Use `manifest.json` + `scene_flow.npz` in a custom loader:\n"
            "1. Parse `cameras[]` for per-frame or interpolated extrinsics/intrinsics.\n"
            "2. For time `t`, load `frames/{t:06d}.ply` OR reconstruct "
            "`means_t = means_ref + displacements[:, t]` from NPZ.\n"
            "3. Render with gsplat / WebGL splat rasterizer at any novel camera.\n\n"
            f"Frames: {num_frames}, Gaussians/frame: {int(query_idx.numel())}, FPS hint: {fps}\n"
        )

    logger.info("Web viewer export → %s", web_root)
    return web_root
