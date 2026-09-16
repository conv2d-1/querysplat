"""Load WorldTrack / Tapvid3d sequences from ``mf_files`` JSON (+ on-disk npy/jpg).

Open-d4rt eval protocol (``eval_track3d_in_worldtrack.py``):
    - Clip length: ``min(num_frames, #images, #track_frames, #visibility_frames)``
    - Queries: frame-0 **visibility only** + finite projected UV + finite depth
      (does **not** use JSON ``valids`` for query selection)
    - GT trajectories in frame-0 reference world coordinates
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from hAlgorithm.modules.metrics.worldtrack_eval_metrics import tracks_cam_to_ref0_world

PathLike = Union[str, Path]

CoordConvention = Literal["opend4rt", "pointodyssey"]
DEFAULT_COORD_CONVENTION: CoordConvention = "opend4rt"


def normalize_coord_convention(coord_convention: Optional[str] = None) -> CoordConvention:
    """Normalize CLI / config aliases to a loader convention name."""
    if coord_convention is None or str(coord_convention).strip() == "":
        return DEFAULT_COORD_CONVENTION
    cc = str(coord_convention).strip().lower()
    if cc == "opend4rt":
        return "opend4rt"
    if cc in ("pointodyssey", "po", "po_synthverse", "synthverse", "point_odyssey"):
        return "pointodyssey"
    raise ValueError(
        f"Unknown coord_convention {coord_convention!r}; "
        "use 'opend4rt' (TapVid3D / Open-d4rt) or 'pointodyssey' (PO / SynthVerse)."
    )


def ref0_to_cam_tracks(points_ref0_tn3: np.ndarray, extrinsics_w2c: np.ndarray) -> np.ndarray:
    """Frame-0 ref coordinates [T,N,3] -> per-frame native camera coords [T,N,3]."""
    pts = np.asarray(points_ref0_tn3, dtype=np.float64)
    ext = np.asarray(extrinsics_w2c, dtype=np.float64)
    c2w0 = np.linalg.inv(ext[0])
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for t in range(int(pts.shape[0])):
        w2c_t = ext[t]
        p = pts[t]
        fin = np.isfinite(p).all(axis=-1)
        if not np.any(fin):
            continue
        ph = np.concatenate([p[fin], np.ones((int(fin.sum()), 1), dtype=np.float64)], axis=-1)
        world = (ph @ c2w0.T)[:, :3]
        wh = np.concatenate([world, np.ones((world.shape[0], 1), dtype=np.float64)], axis=-1)
        out[t, fin] = (wh @ w2c_t.T)[:, :3]
    return out


def extrinsics_c2w_to_w2c(extrinsics_c2w: np.ndarray) -> np.ndarray:
    """Invert PointOdyssey / SynthVerse ``extrinsics`` (c2w) to Open-d4rt w2c layout."""
    ext = np.asarray(extrinsics_c2w, dtype=np.float64)
    if ext.ndim == 2:
        return np.linalg.inv(ext)
    out = np.full_like(ext, np.nan, dtype=np.float64)
    for t in range(int(ext.shape[0])):
        out[t] = np.linalg.inv(ext[t])
    return out


def pointodyssey_world_to_ref0_world(
    tracks_world_tn3: np.ndarray,
    extrinsics_c2w: np.ndarray,
) -> np.ndarray:
    """Convert PO / SynthVerse world ``trajs_3d`` + c2w extrinsics to ref0-world layout."""
    pts = np.asarray(tracks_world_tn3, dtype=np.float64)
    c2w = np.asarray(extrinsics_c2w, dtype=np.float64)
    if pts.ndim != 3 or c2w.ndim != 3:
        raise ValueError(
            f"Expected tracks [T,N,3] and extrinsics [T,4,4], got {pts.shape} and {c2w.shape}"
        )
    w2c0 = np.linalg.inv(c2w[0])
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for t in range(int(pts.shape[0])):
        p = pts[t]
        fin = np.isfinite(p).all(axis=-1)
        if not np.any(fin):
            continue
        ph = np.concatenate([p[fin], np.ones((int(fin.sum()), 1), dtype=np.float64)], axis=-1)
        out[t, fin] = (ph @ w2c0.T)[:, :3]
    return out


def pointodyssey_cam_to_ref0_world(
    tracks_cam_tn3: np.ndarray,
    extrinsics_c2w: np.ndarray,
) -> np.ndarray:
    """Convert per-frame camera ``trajs_3d`` to ref0-world layout (legacy PO layout)."""
    pts = np.asarray(tracks_cam_tn3, dtype=np.float64)
    c2w = np.asarray(extrinsics_c2w, dtype=np.float64)
    if pts.ndim != 3 or c2w.ndim != 3:
        raise ValueError(
            f"Expected tracks [T,N,3] and extrinsics [T,4,4], got {pts.shape} and {c2w.shape}"
        )
    w2c0 = np.linalg.inv(c2w[0])
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for t in range(int(pts.shape[0])):
        p = pts[t]
        fin = np.isfinite(p).all(axis=-1)
        if not np.any(fin):
            continue
        ph = np.concatenate([p[fin], np.ones((int(fin.sum()), 1), dtype=np.float64)], axis=-1)
        world = (ph @ c2w[t].T)[:, :3]
        wh = np.concatenate([world, np.ones((world.shape[0], 1), dtype=np.float64)], axis=-1)
        out[t, fin] = (wh @ w2c0.T)[:, :3]
    return out


def normalize_mf_tracks_to_opend4rt(
    tracks_per_frame_tn3: np.ndarray,
    extrinsics_raw: np.ndarray,
    *,
    coord_convention: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(tracks_ref0, extrinsics_w2c)`` for WorldTrack metrics / query gating."""
    cc = normalize_coord_convention(coord_convention)
    tracks = np.asarray(tracks_per_frame_tn3, dtype=np.float64)
    extr = np.asarray(extrinsics_raw, dtype=np.float64)
    if cc == "opend4rt":
        return tracks, extr
    # PO / SynthVerse mf JSON: world ``trajs_3d`` + c2w ``extrinsics`` (see convert_to_pointodyssey.py).
    tracks_ref0 = pointodyssey_world_to_ref0_world(tracks, extr)
    extrinsics_w2c = extrinsics_c2w_to_w2c(extr)
    return tracks_ref0, extrinsics_w2c


@dataclass
class WorldTrackSequence:
    """One video sequence ready for WorldTrack metrics and TMA inference."""

    video_name: str
    num_frames: int
    num_tracks: int
    video_rgb: np.ndarray  # [T, H, W, 3] uint8
    tracks_xyz_cam: np.ndarray  # [T, N, 3] float64
    tracks_uv: np.ndarray  # [T, N, 2] float64
    visibility: np.ndarray  # [T, N] bool
    valids: np.ndarray  # [T, N] bool
    intrinsics: np.ndarray  # [4] fx, fy, cx, cy
    extrinsics_w2c: np.ndarray  # [T, 4, 4] float64
    gt_tracks_world: np.ndarray  # [T, N, 3] ref0 world (Open-d4rt)
    query_indices: np.ndarray  # [Q] int64 indices into N
    query_uv: np.ndarray  # [Q, 2] pixel coords at frame 0
    query_uv_norm: np.ndarray  # [Q, 2] in [0, 1]
    json_path: Optional[str] = None
    data_root: Optional[str] = None

    @property
    def num_queries(self) -> int:
        return int(self.query_indices.shape[0])

    def gt_tracks_world_queries(self) -> np.ndarray:
        """GT restricted to WorldTrack queries: [T, Q, 3]."""
        return self.gt_tracks_world[:, self.query_indices]


def resolve_worldtrack_path(
    rel_path: str,
    *,
    data_root: PathLike,
    json_path: Optional[PathLike] = None,
) -> Path:
    """Resolve a path stored in mf_files JSON."""
    p = Path(rel_path)
    if p.is_file():
        return p

    root = Path(data_root)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(root / p)
        candidates.append(root / p.name)
        if "Tapvid3d_mini_extracted/" in rel_path.replace("\\", "/"):
            suffix = rel_path.split("Tapvid3d_mini_extracted/")[-1]
            candidates.append(root / "Tapvid3d_mini_extracted" / suffix)
            candidates.append(root.parent / "Tapvid3d_mini_extracted" / suffix)

    if json_path is not None:
        jp = Path(json_path)
        candidates.append(jp.parent / p)
        candidates.append(jp.parent.parent / p)

    for cand in candidates:
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        f"Could not resolve path for {rel_path!r} under data_root={root}"
    )


def read_sequence_frame0_image_hw(
    json_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
) -> Tuple[int, int]:
    """Return (H, W) of frame-0 RGB for screen-space query gating.

    Always reads the actual image header (via OpenCV); never assumes 480x640.
    """
    import cv2

    json_path = Path(json_path)
    if data_root is None:
        data_root = json_path.parent.parent
    data_root = Path(data_root)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mf = payload.get("mf_files", payload)
    if sequence_name not in mf:
        raise KeyError(f"Sequence {sequence_name!r} not in {json_path}")
    frames = mf[sequence_name]
    if not frames:
        raise RuntimeError(f"Empty frame list for {sequence_name}")

    view = frames[0]["views"][0]
    rgb_path = resolve_worldtrack_path(
        view["rgb"], data_root=data_root, json_path=json_path
    )
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read RGB for image_hw: {rgb_path}")
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    return h, w


def infer_trajs_2d_fixed_rel_path(trajs_2d_rel: str) -> Optional[str]:
    """Infer fixed ``trajs_2d`` relative path from the original Tapvid3d layout.

    Example::

        Tapvid3d_mini_extracted/adt_mini/SEQ/trajs_2d/000000.npy
        -> Tapvid3d_mini_extracted/adt_mini_trajs2d_fixed/adt_mini/SEQ/trajs_2d/000000.npy
    """
    rel = str(trajs_2d_rel).replace("\\", "/")
    prefix = "Tapvid3d_mini_extracted/"
    if not rel.startswith(prefix) or "/trajs_2d/" not in rel:
        return None
    suffix = rel[len(prefix) :]
    subset, rest = suffix.split("/", 1)
    if not subset or not rest:
        return None
    return f"{prefix}{subset}_trajs2d_fixed/{subset}/{rest}"


def trajs_2d_path_candidates_from_view(view: Dict[str, Any]) -> List[str]:
    """Candidate ``trajs_2d`` paths: explicit ``trajs_2d_fixed`` then ``trajs_2d``."""
    refs: List[str] = []
    fixed = view.get("trajs_2d_fixed")
    if isinstance(fixed, str) and fixed.strip():
        refs.append(fixed.strip())

    orig = view.get("trajs_2d")
    if isinstance(orig, str) and orig.strip() and orig.strip() not in refs:
        refs.append(orig.strip())
    return refs


def resolve_trajs_2d_for_view(
    view: Dict[str, Any],
    *,
    data_root: PathLike,
    json_path: Optional[PathLike] = None,
) -> Path:
    """Resolve on-disk ``trajs_2d`` for one mf view (prefers ``trajs_2d_fixed``)."""
    refs = trajs_2d_path_candidates_from_view(view)
    if not refs:
        raise FileNotFoundError("view has no trajs_2d / trajs_2d_fixed path")

    last_err: Optional[FileNotFoundError] = None
    for ref in refs:
        try:
            return resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path)
        except FileNotFoundError as exc:
            last_err = exc
    assert last_err is not None
    raise last_err


def prefer_worldtrack_json_path(
    json_path: PathLike,
    *,
    data_root: Optional[PathLike] = None,
) -> Path:
    """Return the mf_files JSON path (``data_root`` unused; kept for API compat)."""
    del data_root
    return Path(json_path).resolve()


def _project_cam_to_uv(
    tracks_xyz_cam: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64).reshape(-1)[:4]
    z = tracks_xyz_cam[..., 2]
    safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
    u = (tracks_xyz_cam[..., 0] / safe_z) * fx + cx
    v = (tracks_xyz_cam[..., 1] / safe_z) * fy + cy
    return np.stack([u, v], axis=-1)


def resolve_opend4rt_frame_count(
    num_frames: int,
    *,
    num_images: int,
    num_track_frames: int,
    num_visibility_frames: int,
) -> int:
    """Match Open-d4rt ``_load_worldtrack_sequence`` clip length."""
    fc = min(
        int(num_frames),
        int(num_images),
        int(num_track_frames),
        int(num_visibility_frames),
    )
    if fc <= 0:
        raise ValueError(
            f"Non-positive clip length: num_frames={num_frames} images={num_images} "
            f"tracks={num_track_frames} visibility={num_visibility_frames}"
        )
    return fc


def compute_dynamic_point_mask(
    tracks_tn3: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """True for queries whose GT displacement from frame 0 exceeds ``threshold`` (meters).

    Use **world** ``trajs_3d`` for SynthVerse / PointOdyssey (static background is static in
    world). Use **ref0-world** tracks for TapVid3D / ``opend4rt``.
    """
    tracks = np.asarray(tracks_tn3, dtype=np.float64)
    if tracks.ndim != 3:
        raise ValueError(f"Expected tracks [T,N,3], got {tracks.shape}")
    disp = np.linalg.norm(tracks - tracks[0:1], axis=-1)
    with np.errstate(invalid="ignore"):
        max_disp = np.nanmax(disp, axis=0)
    return np.isfinite(max_disp) & (max_disp > float(threshold))


def filter_query_indices_by_dynamic_mask(
    query_indices: np.ndarray,
    dynamic_mask: np.ndarray,
) -> np.ndarray:
    """Keep only query indices marked dynamic in ``dynamic_mask`` (length N)."""
    qi = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    mask = np.asarray(dynamic_mask, dtype=bool).reshape(-1)
    if qi.size == 0:
        return qi
    keep = mask[qi]
    return qi[keep]


def _load_per_frame_npy_sequence(
    json_path: Path,
    sequence_name: str,
    *,
    data_root: Path,
    num_frames: int,
    key: str,
) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mf = payload.get("mf_files", payload)
    frames = mf[sequence_name][: min(int(num_frames), len(mf[sequence_name]))]

    def _load_npy(ref: Any) -> np.ndarray:
        if isinstance(ref, str):
            p = resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path)
            return np.load(p, allow_pickle=True)
        return np.asarray(ref)

    return np.stack([_load_npy(frame["views"][0][key]) for frame in frames], axis=0)


def dynamic_query_mask_for_sequence(
    json_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    coord_convention: Optional[str] = None,
    dynamic_query_threshold: float = 0.01,
) -> np.ndarray:
    """Per-track dynamic mask [N] for one sequence."""
    json_path = Path(json_path)
    if data_root is None:
        data_root = json_path.parent.parent
    data_root = Path(data_root)
    cc = normalize_coord_convention(coord_convention)

    if cc == "pointodyssey":
        tracks = _load_per_frame_npy_sequence(
            json_path,
            sequence_name,
            data_root=data_root,
            num_frames=num_frames,
            key="trajs_3d",
        )
    else:
        tracks_raw = _load_per_frame_npy_sequence(
            json_path,
            sequence_name,
            data_root=data_root,
            num_frames=num_frames,
            key="trajs_3d",
        )
        extr_raw = _load_per_frame_npy_sequence(
            json_path,
            sequence_name,
            data_root=data_root,
            num_frames=num_frames,
            key="extrinsics",
        )
        tracks, _ = normalize_mf_tracks_to_opend4rt(
            tracks_raw,
            extr_raw,
            coord_convention=cc,
        )
    return compute_dynamic_point_mask(tracks, dynamic_query_threshold)


def compute_opend4rt_frame0_query_indices(
    tracks_xyz_cam: np.ndarray,
    visibility: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_hw: Tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Frame-0 query indices — identical to Open-d4rt ``eval_track3d_in_worldtrack.py``.

    1. ``visible_mask = visibility[0]``
    2. On visible tracks: finite ``tracks_uv[0]`` and finite ``tracks_xyz_cam[0, :, 2] > 1e-8``
    3. UV from projecting ``trajs_3d`` with ``cam_in`` (not disk ``trajs_2d``)
    """
    tracks_uv = _project_cam_to_uv(tracks_xyz_cam, intrinsics)
    visible_mask = np.asarray(visibility[0], dtype=bool)
    if not np.any(visible_mask):
        return np.zeros((0,), dtype=np.int64)

    idx_vis = np.flatnonzero(visible_mask)
    query_uv = np.asarray(tracks_uv[0, idx_vis], dtype=np.float64)
    depth0 = np.asarray(tracks_xyz_cam[0, idx_vis, 2], dtype=np.float64)
    finite_mask = np.isfinite(query_uv).all(axis=-1)
    finite_mask &= np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
    # Screen-space gate: projected UV must be inside image bounds.
    h, w = int(image_hw[0]), int(image_hw[1])
    finite_mask &= (query_uv[:, 0] >= 0.0) & (query_uv[:, 0] < float(w))
    finite_mask &= (query_uv[:, 1] >= 0.0) & (query_uv[:, 1] < float(h))
    if not np.any(finite_mask):
        return np.zeros((0,), dtype=np.int64)
    return idx_vis[finite_mask].astype(np.int64)


def compute_frame0_query_indices_from_uv_and_depth(
    query_uv_frame0: np.ndarray,
    depth0: np.ndarray,
    visibility_frame0: np.ndarray,
    *,
    image_hw: Tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Frame-0 query indices using provided UV + depth (for ref0 trajs_3d datasets)."""
    visible_mask = np.asarray(visibility_frame0, dtype=bool)
    if not np.any(visible_mask):
        return np.zeros((0,), dtype=np.int64)

    idx_vis = np.flatnonzero(visible_mask)
    uv0 = np.asarray(query_uv_frame0, dtype=np.float64)[idx_vis]
    z0 = np.asarray(depth0, dtype=np.float64)[idx_vis]
    finite_mask = np.isfinite(uv0).all(axis=-1)
    finite_mask &= np.isfinite(z0) & (np.abs(z0) > 1e-8)
    # Screen-space gate: projected UV must be inside image bounds.
    h, w = int(image_hw[0]), int(image_hw[1])
    finite_mask &= (uv0[:, 0] >= 0.0) & (uv0[:, 0] < float(w))
    finite_mask &= (uv0[:, 1] >= 0.0) & (uv0[:, 1] < float(h))
    if not np.any(finite_mask):
        return np.zeros((0,), dtype=np.int64)
    return idx_vis[finite_mask].astype(np.int64)


def compute_frame0_query_indices(
    tracks_xyz_cam: np.ndarray,
    visibility: np.ndarray,
    valids: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Alias for :func:`compute_opend4rt_frame0_query_indices` (``valids`` ignored)."""
    del valids
    return compute_opend4rt_frame0_query_indices(tracks_xyz_cam, visibility, intrinsics)


def opend4rt_query_count_from_arrays(
    tracks_xyz_cam: np.ndarray,
    visibility: np.ndarray,
    intrinsics: np.ndarray,
) -> int:
    return int(
        compute_opend4rt_frame0_query_indices(tracks_xyz_cam, visibility, intrinsics).size
    )


def normalize_uv_to_01(
    query_uv: np.ndarray,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    out = np.asarray(query_uv, dtype=np.float64).copy()
    out[:, 0] /= float(max(w - 1, 1))
    out[:, 1] /= float(max(h - 1, 1))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def list_sequences_in_json(json_path: PathLike) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mf = payload.get("mf_files", payload)
    if not isinstance(mf, dict):
        raise ValueError(f"Invalid mf_files JSON: {json_path}")
    return sorted(mf.keys())


def frame0_query_indices_for_sequence(
    json_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    coord_convention: Optional[str] = None,
    dynamic_query_threshold: Optional[float] = None,
) -> np.ndarray:
    """Return frame-0 query indices; empty array if none (does not raise)."""
    json_path = Path(json_path)
    if data_root is None:
        data_root = json_path.parent.parent
    data_root = Path(data_root)
    cc = normalize_coord_convention(coord_convention)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mf = payload.get("mf_files", payload)
    if sequence_name not in mf:
        raise KeyError(f"Sequence {sequence_name!r} not in {json_path}")

    frames = mf[sequence_name]
    if not frames:
        return np.zeros((0,), dtype=np.int64)

    view = frames[0]["views"][0]

    def _load_npy(key: str) -> np.ndarray:
        ref = view[key]
        if isinstance(ref, str):
            p = resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path)
            return np.load(p)
        return np.asarray(ref)

    tracks_raw = np.asarray(_load_npy("trajs_3d"), dtype=np.float64)
    visibility = np.asarray(_load_npy("visibs"), dtype=bool)
    extr_raw = np.asarray(view["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(view["cam_in"], dtype=np.float64).reshape(-1)[:4]

    tracks_ref0, extr_w2c = normalize_mf_tracks_to_opend4rt(
        tracks_raw[np.newaxis, ...],
        extr_raw[np.newaxis, ...],
        coord_convention=cc,
    )
    cam0 = ref0_to_cam_tracks(tracks_ref0, extr_w2c)[0]
    depth0 = cam0[:, 2]
    uv0 = _project_cam_to_uv(cam0[np.newaxis, ...], intrinsics)[0]
    image_hw = read_sequence_frame0_image_hw(
        json_path, sequence_name, data_root=data_root
    )
    vis0 = visibility[0] if visibility.ndim == 2 else visibility
    query_indices = compute_frame0_query_indices_from_uv_and_depth(
        uv0, depth0, vis0, image_hw=image_hw
    )
    if dynamic_query_threshold is not None:
        dyn_mask = dynamic_query_mask_for_sequence(
            json_path,
            sequence_name,
            data_root=data_root,
            num_frames=num_frames,
            coord_convention=cc,
            dynamic_query_threshold=float(dynamic_query_threshold),
        )
        query_indices = filter_query_indices_by_dynamic_mask(query_indices, dyn_mask)
    return query_indices


def assert_query_uv_in_image_bounds(
    query_uv: np.ndarray,
    image_hw: Tuple[int, int],
) -> None:
    """Raise if any frame-0 query UV lies outside ``[0,W) x [0,H)`` (Open-d4rt gate)."""
    uv = np.asarray(query_uv, dtype=np.float64)
    h, w = int(image_hw[0]), int(image_hw[1])
    if uv.size == 0:
        return
    if not np.isfinite(uv).all():
        bad = int(np.sum(~np.isfinite(uv)))
        raise ValueError(f"{bad} query UVs are non-finite")
    oob = (
        (uv[:, 0] < 0.0)
        | (uv[:, 0] >= float(w))
        | (uv[:, 1] < 0.0)
        | (uv[:, 1] >= float(h))
    )
    if np.any(oob):
        n_oob = int(np.sum(oob))
        raise ValueError(
            f"{n_oob}/{uv.shape[0]} query UVs outside image bounds (H={h}, W={w})"
        )


def align_batch_queries_with_sequence(
    batch: dict,
    seq: WorldTrackSequence,
    *,
    sequence_name: str = "",
) -> np.ndarray:
    """Ensure eval uses the same query set / UVs as ``load_sequence_from_json`` (Scene Flow parity).

    - ``query_indices`` must match ``seq.query_indices`` (bounds-gated, projected UV).
    - Injects ``worldtrack_query_uv`` so inference does not rely on on-disk ``trajs_2d``.
    """
    qi = batch.get("worldtrack_query_indices")
    if qi is None:
        query_idx = np.asarray(seq.query_indices, dtype=np.int64)
    else:
        if hasattr(qi, "detach"):
            qi = qi.detach().cpu().numpy()
        query_idx = np.asarray(qi, dtype=np.int64)
        if query_idx.ndim == 2:
            query_idx = query_idx[0]
        query_idx = query_idx.reshape(-1)
    if not np.array_equal(query_idx, seq.query_indices):
        label = sequence_name or seq.video_name
        raise RuntimeError(
            f"{label}: dataset query_indices ({query_idx.size}) != "
            f"JSON loader ({seq.query_indices.size}); cannot align with Scene Flow eval"
        )
    h, w = int(seq.video_rgb.shape[1]), int(seq.video_rgb.shape[2])
    assert_query_uv_in_image_bounds(seq.query_uv, (h, w))
    batch["worldtrack_query_uv"] = np.asarray(seq.query_uv, dtype=np.float32)
    batch["worldtrack_query_indices"] = query_idx.copy()
    batch["worldtrack_num_queries"] = int(query_idx.size)
    batch["worldtrack_num_frames"] = int(seq.num_frames)
    return query_idx


def filter_sequences_with_frame0_queries(
    json_path: PathLike,
    sequence_names: Sequence[str],
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    coord_convention: Optional[str] = None,
    dynamic_query_threshold: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """Split sequences into those with valid frame-0 queries vs skipped."""
    valid: List[str] = []
    skipped: List[str] = []
    for name in sequence_names:
        qi = frame0_query_indices_for_sequence(
            json_path,
            name,
            data_root=data_root,
            num_frames=num_frames,
            coord_convention=coord_convention,
            dynamic_query_threshold=dynamic_query_threshold,
        )
        if int(qi.size) > 0:
            valid.append(name)
        else:
            skipped.append(name)
    return valid, skipped


def load_sequence_from_json(
    json_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    load_rgb: bool = True,
    coord_convention: Optional[str] = None,
    dynamic_query_threshold: Optional[float] = None,
) -> WorldTrackSequence:
    """Load one sequence from mf_files JSON (first ``num_frames`` frames).

    ``coord_convention``:
        - ``opend4rt`` (default): ``trajs_3d`` in ref0 world, ``extrinsics`` w2c (TapVid3D).
        - ``pointodyssey``: world ``trajs_3d``, ``extrinsics`` c2w (PO / SynthVerse mf JSON).
    """
    json_path = Path(json_path)
    if data_root is None:
        data_root = json_path.parent.parent
    data_root = Path(data_root)
    cc = normalize_coord_convention(coord_convention)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mf = payload.get("mf_files", payload)
    if sequence_name not in mf:
        raise KeyError(f"Sequence {sequence_name!r} not in {json_path}")

    frames = mf[sequence_name]
    if not frames:
        raise RuntimeError(f"Empty frame list for {sequence_name}")

    # Pre-cap JSON frame list; final clip may shrink further after loading arrays.
    list_cap = min(int(num_frames), len(frames))
    frames = frames[:list_cap]

    rgbs: List[np.ndarray] = []
    tracks_ref0_list: List[np.ndarray] = []
    vis_list: List[np.ndarray] = []
    val_list: List[np.ndarray] = []
    extr_list: List[np.ndarray] = []
    intrinsics: Optional[np.ndarray] = None

    for frame in frames:
        if len(frame.get("views", [])) != 1:
            raise ValueError(
                f"Expected 1 view per frame for {sequence_name}, got {len(frame.get('views', []))}"
            )
        view = frame["views"][0]

        if load_rgb:
            rgb_path = resolve_worldtrack_path(
                view["rgb"], data_root=data_root, json_path=json_path
            )
            import cv2

            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read RGB: {rgb_path}")
            rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        def _load_npy(key: str) -> np.ndarray:
            ref = view[key]
            if isinstance(ref, str):
                p = resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path)
                return np.load(p)
            return np.asarray(ref)

        # `trajs_3d` is treated as ref0-space tracks (already rewritten on disk).
        tracks_ref0_list.append(np.asarray(_load_npy("trajs_3d"), dtype=np.float64))
        vis_list.append(np.asarray(_load_npy("visibs"), dtype=bool))
        val_list.append(np.asarray(_load_npy("valids"), dtype=bool))

        if intrinsics is None:
            intrinsics = np.asarray(view["cam_in"], dtype=np.float64).reshape(-1)[:4]
        extr_list.append(np.asarray(view["extrinsics"], dtype=np.float64))

    tracks_raw_stack = np.stack(tracks_ref0_list, axis=0)
    visibility = np.stack(vis_list, axis=0)
    valids = np.stack(val_list, axis=0)
    extrinsics_raw = np.stack(extr_list, axis=0)

    num_images = len(rgbs) if load_rgb else int(tracks_raw_stack.shape[0])
    frame_count = resolve_opend4rt_frame_count(
        num_frames,
        num_images=num_images,
        num_track_frames=int(tracks_raw_stack.shape[0]),
        num_visibility_frames=int(visibility.shape[0]),
    )
    if load_rgb:
        rgbs = rgbs[:frame_count]
    tracks_raw_stack = tracks_raw_stack[:frame_count]
    visibility = visibility[:frame_count]
    valids = valids[:frame_count]
    extrinsics_raw = extrinsics_raw[:frame_count]

    tracks_ref0, extrinsics_w2c = normalize_mf_tracks_to_opend4rt(
        tracks_raw_stack,
        extrinsics_raw,
        coord_convention=cc,
    )

    # Keep a camera-space copy for sanity checks / UV projection.
    tracks_xyz_cam = ref0_to_cam_tracks(tracks_ref0, extrinsics_w2c)
    tracks_uv = _project_cam_to_uv(tracks_xyz_cam, intrinsics)
    gt_tracks_world = np.asarray(tracks_ref0, dtype=np.float64)
    if load_rgb:
        h, w = rgbs[0].shape[:2]
        video_rgb = np.stack(rgbs, axis=0)
    else:
        h, w = read_sequence_frame0_image_hw(
            json_path, sequence_name, data_root=data_root
        )
        video_rgb = np.zeros((frame_count, h, w, 3), dtype=np.uint8)

    query_indices = compute_frame0_query_indices_from_uv_and_depth(
        tracks_uv[0],
        tracks_xyz_cam[0, :, 2],
        visibility[0],
        image_hw=(h, w),
    )
    if dynamic_query_threshold is not None:
        if cc == "pointodyssey":
            dyn_mask = compute_dynamic_point_mask(tracks_raw_stack, dynamic_query_threshold)
        else:
            dyn_mask = compute_dynamic_point_mask(tracks_ref0, dynamic_query_threshold)
        query_indices = filter_query_indices_by_dynamic_mask(query_indices, dyn_mask)

    if query_indices.size == 0:
        raise RuntimeError(f"No valid frame-0 queries for {sequence_name}")

    query_uv = tracks_uv[0, query_indices]

    query_uv_norm = normalize_uv_to_01(query_uv, (h, w))

    return WorldTrackSequence(
        video_name=sequence_name,
        num_frames=frame_count,
        num_tracks=int(tracks_xyz_cam.shape[1]),
        video_rgb=video_rgb,
        tracks_xyz_cam=tracks_xyz_cam,
        tracks_uv=tracks_uv,
        visibility=visibility,
        valids=valids,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        gt_tracks_world=gt_tracks_world,
        query_indices=query_indices,
        query_uv=query_uv.astype(np.float32),
        query_uv_norm=query_uv_norm,
        json_path=str(json_path),
        data_root=str(data_root),
    )


def load_sequence_from_npz(
    npz_path: PathLike,
    *,
    num_frames: int = 64,
    load_rgb: bool = True,
) -> WorldTrackSequence:
    """Load the same fields from an official WorldTrack ``.npz`` (for cross-check)."""
    import cv2

    npz_path = Path(npz_path)
    pack = np.load(npz_path, allow_pickle=True)
    images_jpeg_bytes = np.asarray(pack["images_jpeg_bytes"])
    tracks_xyz_cam = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
    intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64)
    visibility = np.asarray(pack["visibility"], dtype=bool)
    extrinsics_w2c = np.asarray(pack["extrinsics_w2c"], dtype=np.float64)

    frame_count = resolve_opend4rt_frame_count(
        num_frames,
        num_images=int(images_jpeg_bytes.shape[0]),
        num_track_frames=int(tracks_xyz_cam.shape[0]),
        num_visibility_frames=int(visibility.shape[0]),
    )
    images_jpeg_bytes = images_jpeg_bytes[:frame_count]
    tracks_xyz_cam = tracks_xyz_cam[:frame_count]
    visibility = visibility[:frame_count]
    extrinsics_w2c = extrinsics_w2c[:frame_count]

    rgbs = []
    if load_rgb:
        for frame_bytes in images_jpeg_bytes:
            arr = np.frombuffer(frame_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to decode JPEG in {npz_path}")
            rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        video_rgb = np.stack(rgbs, axis=0)
    else:
        video_rgb = np.zeros((frame_count, 480, 640, 3), dtype=np.uint8)

    tracks_uv = _project_cam_to_uv(tracks_xyz_cam, intrinsics)
    valids = visibility.copy()
    # NPZ stores camera-space tracks; keep legacy conversion for NPZ loader.
    gt_tracks_world = tracks_cam_to_ref0_world(tracks_xyz_cam, extrinsics_w2c)
    query_indices = compute_opend4rt_frame0_query_indices(
        tracks_xyz_cam, visibility, intrinsics
    )
    query_uv = tracks_uv[0, query_indices]
    h, w = video_rgb.shape[1:3]
    query_uv_norm = normalize_uv_to_01(query_uv, (h, w))

    return WorldTrackSequence(
        video_name=npz_path.stem,
        num_frames=frame_count,
        num_tracks=int(tracks_xyz_cam.shape[1]),
        video_rgb=video_rgb,
        tracks_xyz_cam=tracks_xyz_cam,
        tracks_uv=tracks_uv,
        visibility=visibility,
        valids=valids,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        gt_tracks_world=gt_tracks_world,
        query_indices=query_indices,
        query_uv=query_uv.astype(np.float32),
        query_uv_norm=query_uv_norm,
        json_path=None,
        data_root=str(npz_path.parent),
    )


def compare_json_vs_npz(
    json_path: PathLike,
    npz_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Verify JSON loader matches npz + Open-d4rt GT construction."""
    seq_json = load_sequence_from_json(
        json_path, sequence_name, data_root=data_root, num_frames=num_frames, load_rgb=False
    )
    seq_npz = load_sequence_from_npz(npz_path, num_frames=num_frames, load_rgb=False)

    report: Dict[str, Any] = {
        "sequence": sequence_name,
        "num_frames": seq_json.num_frames,
        "num_tracks": seq_json.num_tracks,
        "num_queries_json": int(seq_json.num_queries),
        "num_queries_npz": int(seq_npz.num_queries),
    }

    def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")

    report["tracks_xyz_cam_max_diff"] = _max_diff(seq_json.tracks_xyz_cam, seq_npz.tracks_xyz_cam)
    report["visibility_equal"] = bool(np.array_equal(seq_json.visibility, seq_npz.visibility))
    report["gt_tracks_world_max_diff"] = _max_diff(seq_json.gt_tracks_world, seq_npz.gt_tracks_world)
    report["query_indices_equal"] = bool(np.array_equal(seq_json.query_indices, seq_npz.query_indices))

    report["aligned"] = (
        report["tracks_xyz_cam_max_diff"] < atol
        and report["visibility_equal"]
        and report["gt_tracks_world_max_diff"] < atol
        and report["query_indices_equal"]
    )
    return report
