"""Stereo4D Hesai (left-rectified) 4D tracking dataset.

The public index JSON (``stereo4d_hesai_mf_{train,val}_with_tracking.json``) follows
the same *mf_files* layout as other TMD multi-frame datasets:

* Top level: ``{"mf_files": { "<scene_id>": [ { "frame_id", "views" }, ... ] }}``
* Each *view* entry (single camera per frame) provides:

  * ``rgb`` — RGBA PNG path (relative to ``data_root``)
  * ``depth`` — ``depth_plane_*.npy`` (H×W, float32, **OpenCV**-style z in metres;
    same z used for unprojection with ``cam_in``)
  * ``cam_in`` — ``[fx, fy, cx, cy]`` in **pixel** coordinates
  * ``extrinsics`` — 4×4 **world-to-camera** matrix as stored by the Hesai / Blender
    export (OpenGL-style camera: +Y up, camera looks down **-Z**).  This is *not*
    the same as OpenCV w2c (+Y down, +Z forward) until :meth:`convert_extrinsics`
    is applied.
  * ``trajs_2d``, ``trajs_3d``, ``visibs``, ``valids`` — per-frame ``.npy`` paths
  * ``trajs_3d`` is in a **single world frame** shared by the clip; it is
    independent of the camera-axis convention.  Converting *w2c* to OpenCV only
    changes how the same world point is expressed in camera coordinates.

Coordinate fix (``extrinsic_convention="gl_w2c"``)
------------------------------------------------
Left-multiply the raw 4×4 by ``diag(1, -1, -1, 1)`` so that the third row
(``-Z`` forward in GL) becomes ``+Z`` forward in OpenCV, while keeping the
world frame and ``trajs_3d`` unchanged.  This matches the depth / pinhole
unprojection used throughout :class:`hAlgorithm.datasets.base_dataset.BaseDataset`.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.getcwd())

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset

# OpenGL / Blender camera (Y up, -Z forward) -> OpenCV camera (Y down, +Z forward)
_GL_TO_CV_W2C = np.diag(np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float64))


class Stereo4DHesaiTrackDataset(BaseTrackDataset):
    """Loads Stereo4D Hesai clips with optional GL→OpenCV *w2c* conversion."""

    def __init__(
        self,
        extrinsic_convention: str = "gl_w2c",
        **kwargs,
    ):
        """
        Parameters
        ----------
        extrinsic_convention : str
            * ``"gl_w2c"`` (default) — JSON stores Blender/OpenGL *w2c*; convert to
              OpenCV *w2c* for the rest of the stack.
            * ``"opencv_w2c"`` or ``"w2c"`` — already OpenCV; no change.
        """
        self.extrinsic_convention = (extrinsic_convention or "gl_w2c").lower()
        super().__init__(**kwargs)

    def preprocess_data_info(self, data_info: dict) -> dict:
        """Normalise empty ``sem`` / ``motion_mask`` path strings to *None*."""
        patched = dict(data_info)
        for key in ("sem", "motion_mask"):
            if patched.get(key) == "":
                patched[key] = None
        return patched

    def convert_extrinsics(self, extrinsics_raw: np.ndarray) -> np.ndarray:
        """Map JSON *w2c* to the OpenCV *w2c* expected by depth / point maps."""
        e = np.asarray(extrinsics_raw, dtype=np.float64)
        if e.shape == (3, 4):
            m = np.eye(4, dtype=np.float64)
            m[:3, :4] = e
            e = m
        if self.extrinsic_convention in ("gl_w2c", "blender_w2c", "gl"):
            return (_GL_TO_CV_W2C @ e)
        if self.extrinsic_convention in (
            "opencv_w2c",
            "opencv",
            "w2c",
            "cv_w2c",
            "none",
            "identity",
        ):
            return e
        raise ValueError(
            f"Unknown extrinsic_convention={self.extrinsic_convention!r}; "
            'use "gl_w2c" or "opencv_w2c".'
        )


def _vis_output_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "_stereo4d_hesai_vis",
    )


def _height_colors_xyz(
    pts: np.ndarray,
    axis: int = 2,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> np.ndarray:
    """Vertex colours from height along one axis (blue→green→red) for structure cues."""
    coord = pts[:, axis].astype(np.float64)
    lo, hi = np.percentile(coord, [lo_pct, hi_pct])
    if hi <= lo + 1e-6:
        return np.full((len(pts), 3), 180, dtype=np.uint8)
    t = np.clip((coord - lo) / (hi - lo), 0.0, 1.0)
    r = (255 * t).astype(np.uint8)
    b = (255 * (1.0 - t)).astype(np.uint8)
    g = (255 * np.sin(np.pi * t)).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


def _visualize_merged_world_ply(
    dataset: Stereo4DHesaiTrackDataset,
    sample_index: int = 0,
    downsample: int = 6,
    depth_min: float = 0.05,
    depth_max: float = 20.0,
    max_points_per_frame: int = 35000,
    max_points_total: int = 120000,
    seed: int = 0,
    center_on_first_camera: bool = True,
    crop_radius_m: Optional[float] = 22.0,
    overlay_trajs: bool = True,
    traj_max_points: int = 8000,
    output_path: Optional[str] = None,
    write_height_ply: bool = True,
) -> Optional[str]:
    """Fuse multi-frame depth in **world** coordinates, then export viewer-friendly PLYs.

    Raw world coordinates are often huge and multi-frame RGB stacks look like
    rainbow soup in MeshLab.  This routine:

    #. Keeps depths in ``[depth_min, depth_max]`` (default max **20 m**, same idea as RRD).
    #. Optionally **subtracts the first camera centre** so the scene sits near the origin.
    #. Optionally drops points outside ``crop_radius_m`` after centering (default **22 m**).
    #. Caps total vertices with a global random subsample.
    #. Writes ``merged_world.ply`` — RGB depth points plus optional **red** ``trajs_3d``
       overlay (frame 0, valid points) for alignment checks vs GT.
    #. Writes ``merged_world_height.ply`` — **same XYZ** as the depth portion but vertex
       colour encodes height (Z in centered frame) so planar structure is visible without
       relying on RGB overlap.

    Returns the path to ``merged_world.ply``, or *None* on failure.
    """
    from hAlgorithm.datasets_4d.vis_utils import prepare_rgb_for_vis, save_ply, to_numpy

    rng = np.random.default_rng(seed)
    data = dataset[sample_index]

    for req in ("depth", "extrinsics", "intrinsics"):
        if req not in data:
            logging.error("Merged-world viz: missing %s", req)
            return None

    depth = to_numpy(data["depth"])
    extrinsics = to_numpy(data["extrinsics"])
    intrinsics = to_numpy(data["intrinsics"])

    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]

    num_frames = depth.shape[0]
    w2c0 = extrinsics[0] if extrinsics.ndim == 3 else extrinsics
    if w2c0.shape == (3, 4):
        m4 = np.eye(4, dtype=np.float64)
        m4[:3, :4] = w2c0
        w2c0 = m4
    c2w0 = np.linalg.inv(w2c0)
    cam_center_world = c2w0[:3, 3].copy()

    rgb_np, _rgb_key = prepare_rgb_for_vis(data)
    if rgb_np is None:
        logging.warning("Merged-world PLY: no RGB tensor in batch — depth points will be gray.")

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    for fi in range(num_frames):
        zmap = depth[fi]
        h, w = zmap.shape
        k = intrinsics[fi] if intrinsics.ndim == 3 else intrinsics
        w2c = extrinsics[fi] if extrinsics.ndim == 3 else extrinsics
        if w2c.shape == (3, 4):
            m4 = np.eye(4, dtype=np.float64)
            m4[:3, :4] = w2c
            w2c = m4
        c2w = np.linalg.inv(w2c)

        us = np.arange(0, w, downsample)
        vs = np.arange(0, h, downsample)
        uu, vv = np.meshgrid(us, vs)
        zs = zmap[vv, uu]
        m = (zs > depth_min) & (zs < depth_max)
        uu_f = uu[m].astype(np.float64)
        vv_f = vv[m].astype(np.float64)
        z_f = zs[m].astype(np.float64)
        if z_f.size == 0:
            continue

        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        xc = (uu_f - cx) / fx * z_f
        yc = (vv_f - cy) / fy * z_f
        pts_cam = np.stack([xc, yc, z_f], axis=1)
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        if rgb_np is not None and fi < rgb_np.shape[0]:
            cols = np.asarray(rgb_np[fi][vv[m], uu[m]])
            if cols.ndim == 2 and cols.shape[1] >= 4:
                cols = cols[:, :3]
            if cols.dtype != np.uint8:
                cols = np.clip(np.round(cols), 0, 255).astype(np.uint8)
        else:
            cols = np.full((pts_world.shape[0], 3), 160, dtype=np.uint8)

        if pts_world.shape[0] > max_points_per_frame:
            idx = rng.choice(pts_world.shape[0], max_points_per_frame, replace=False)
            pts_world = pts_world[idx]
            cols = cols[idx]

        all_pts.append(pts_world.astype(np.float32))
        all_cols.append(cols)

    if not all_pts:
        logging.error("Merged-world viz: no depth samples.")
        return None

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)

    offset = np.zeros(3, dtype=np.float64)
    if center_on_first_camera:
        offset = cam_center_world
    pts_c = pts.astype(np.float64) - offset

    if crop_radius_m is not None and crop_radius_m > 0:
        dist = np.linalg.norm(pts_c, axis=1)
        keep = dist <= crop_radius_m
        pts_c = pts_c[keep]
        cols = cols[keep]
        logging.info(
            "Merged-world: crop_radius=%.1f m kept %d / %d depth vertices",
            crop_radius_m,
            len(pts_c),
            len(pts),
        )

    if len(pts_c) > max_points_total:
        sel = rng.choice(len(pts_c), max_points_total, replace=False)
        pts_c = pts_c[sel]
        cols = cols[sel]

    depth_pts_final = pts_c.astype(np.float32)
    depth_cols_final = cols

    traj_pts_c = np.zeros((0, 3), dtype=np.float32)
    if overlay_trajs and "trajs_3d" in data:
        t3 = to_numpy(data["trajs_3d"])
        if t3.ndim == 3 and t3.shape[0] > 0:
            t0 = t3[0].astype(np.float64) - offset
            mask = np.isfinite(t0).all(axis=1)
            if "valids" in data:
                val = to_numpy(data["valids"])
                if val.ndim == 2 and val.shape[0] > 0:
                    mask &= val[0].astype(bool)
            t0 = t0[mask]
            if t0.shape[0] > traj_max_points:
                t0 = t0[rng.choice(t0.shape[0], traj_max_points, replace=False)]
            traj_pts_c = t0.astype(np.float32)

    out = output_path or os.path.join(_vis_output_dir(), "merged_world.ply")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    traj_rgb = np.tile(np.asarray([[255, 70, 70]], dtype=np.uint8), (len(traj_pts_c), 1))
    if len(traj_pts_c) > 0:
        pts_out = np.concatenate([depth_pts_final, traj_pts_c], axis=0)
        cols_out = np.concatenate([depth_cols_final, traj_rgb], axis=0)
    else:
        pts_out, cols_out = depth_pts_final, depth_cols_final

    save_ply(out, pts_out, cols_out)
    logging.info(
        "merged_world.ply: depth_pts=%d traj_pts=%d (centered=%s offset=%s)",
        len(depth_pts_final),
        len(traj_pts_c),
        center_on_first_camera,
        np.array2string(offset, precision=3, suppress_small=True),
    )

    if write_height_ply:
        hpath = str(Path(out).with_name("merged_world_height.ply"))
        hcol = _height_colors_xyz(depth_pts_final, axis=2)
        save_ply(hpath, depth_pts_final, hcol)
        logging.info("merged_world_height.ply: height-coloured depth (%d pts) -> %s", len(depth_pts_final), hpath)

    bb_min = pts_out.min(axis=0)
    bb_max = pts_out.max(axis=0)
    logging.info(
        "Axis-aligned bounds (centered frame): min=%s max=%s",
        np.array2string(bb_min, precision=3, suppress_small=True),
        np.array2string(bb_max, precision=3, suppress_small=True),
    )
    return out


if __name__ == "__main__":
    # Requires PyTorch (dataset stack). Example:
    #   conda activate acc && cd $TMA_ROOT && PYTHONPATH=. python hAlgorithm/datasets_4d/stereo4d_hesai_track_dataset.py
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    _root = "/mnt/nasTeam2/AI/datasets/TMD/"
    _json_val = os.path.join(_root, "Stereo4D_hesai/stereo4d_hesai_mf_val_with_tracking.json")

    dataset = Stereo4DHesaiTrackDataset(
        phase="test",
        name="Stereo4D_Hesai",
        seed=0,
        data_root=_root,
        data_path=_json_val,
        extrinsic_convention="gl_w2c",
        track_points_nums=256,
        track_neg_ratio=0.0,
        mf_to_mv=True,
        clip_maxlen=24,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=6,
            shuffle=False,
            seed=0,
        ),
        train_transforms=[
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
        min_depth=1e-3,
        max_depth=1000.0,
        with_pointmap=True,
        debug=True,
        normalize_cameras=False,
    )

    out_dir = _vis_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    logging.info("Stereo4D Hesai smoke test — outputs under %s", out_dir)

    # (1) Dense fusion in **first-camera** frame — uses poses exactly like train-time geometry.
    dataset.generate_merged_pointcloud(
        output_path=os.path.join(out_dir, "merged_registered_cam0.ply"),
        sample_index=0,
        downsample=4,
        max_points_per_frame=80000,
        depth_min=0.05,
        depth_max=20.0,
    )

    # (2) World-frame fusion: centred near origin + traj overlay + height-coloured sibling PLY.
    _visualize_merged_world_ply(
        dataset,
        sample_index=0,
        output_path=os.path.join(out_dir, "merged_world.ply"),
    )

    # (3) Rerun: RGB, cameras, merged depth cloud & 3D track classification colours.
    dataset.visualize_tracking_debug(
        sample_idx=0,
        output_path=os.path.join(out_dir, "tracking_debug_stereo4d_hesai.rrd"),
        downsample_pc=4,
        max_pc_points=50000,
        max_traj_vis=1200,
        depth_range=(0.05, 20.0),
    )

    # (4) Second validation clip — distinct scene / clip index (edit ``_EXTRA_RRD_INDEX``).
    _EXTRA_RRD_INDEX = 3
    if _EXTRA_RRD_INDEX < len(dataset):
        dataset.visualize_tracking_debug(
            sample_idx=_EXTRA_RRD_INDEX,
            output_path=os.path.join(
                out_dir,
                f"tracking_debug_stereo4d_hesai_sample{_EXTRA_RRD_INDEX:02d}.rrd",
            ),
            downsample_pc=4,
            max_pc_points=50000,
            max_traj_vis=1200,
            depth_range=(0.05, 20.0),
        )
    else:
        logging.warning(
            "Skip extra RRD: dataset len=%d < _EXTRA_RRD_INDEX+1",
            len(dataset),
        )
