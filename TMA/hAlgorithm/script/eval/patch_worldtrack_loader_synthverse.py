#!/usr/bin/env python3
"""Inject pointodyssey / SynthVerse coord support into vendored worldtrack_json_loader.py files."""

from __future__ import annotations

import re
from pathlib import Path

WORKSPACE = Path("/mnt/home/tcchen/workspace")

LOADER_PATHS = [
    WORKSPACE / "Projects/4RC/arc/eval/worldtrack_json_loader.py",
    WORKSPACE / "Projects/Any4D/any4d/eval/worldtrack_json_loader.py",
    WORKSPACE / "Projects/vdpm/eval/worldtrack_json_loader.py",
    WORKSPACE / "Projects/Open-d4rt/eval/worldtrack_json_loader.py",
    WORKSPACE / "Projects/SpaTrackerV2/evaluation/worldtrack_json_loader.py",
    WORKSPACE / "Projects/St4RTrack/evaluation/worldtrack_json_loader.py",
    WORKSPACE / "TraceAnything/eval/worldtrack_json_loader.py",
]

COORD_BLOCK = '''
CoordConvention = str
DEFAULT_COORD_CONVENTION = "opend4rt"


def normalize_coord_convention(coord_convention=None):
    if coord_convention is None or str(coord_convention).strip() == "":
        return DEFAULT_COORD_CONVENTION
    cc = str(coord_convention).strip().lower()
    if cc == "opend4rt":
        return "opend4rt"
    if cc in ("pointodyssey", "po", "po_synthverse", "synthverse", "point_odyssey"):
        return "pointodyssey"
    raise ValueError(
        f"Unknown coord_convention {coord_convention!r}; use 'opend4rt' or 'pointodyssey'."
    )


def ref0_to_cam_tracks(points_ref0_tn3, extrinsics_w2c):
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


def extrinsics_c2w_to_w2c(extrinsics_c2w):
    ext = np.asarray(extrinsics_c2w, dtype=np.float64)
    if ext.ndim == 2:
        return np.linalg.inv(ext)
    out = np.full_like(ext, np.nan, dtype=np.float64)
    for t in range(int(ext.shape[0])):
        out[t] = np.linalg.inv(ext[t])
    return out


def pointodyssey_world_to_ref0_world(tracks_world_tn3, extrinsics_c2w):
    pts = np.asarray(tracks_world_tn3, dtype=np.float64)
    c2w = np.asarray(extrinsics_c2w, dtype=np.float64)
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


def normalize_mf_tracks_to_opend4rt(tracks_per_frame_tn3, extrinsics_raw, *, coord_convention=None):
    cc = normalize_coord_convention(coord_convention)
    tracks = np.asarray(tracks_per_frame_tn3, dtype=np.float64)
    extr = np.asarray(extrinsics_raw, dtype=np.float64)
    if cc == "opend4rt":
        return tracks, extr
    tracks_ref0 = pointodyssey_world_to_ref0_world(tracks, extr)
    extrinsics_w2c = extrinsics_c2w_to_w2c(extr)
    return tracks_ref0, extrinsics_w2c

'''


def patch_loader(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "normalize_mf_tracks_to_opend4rt" in text:
        print(f"skip (already patched): {path}")
        return

    if "PathLike = Union[str, Path]" not in text:
        raise RuntimeError(f"Unexpected loader layout: {path}")

    text = text.replace(
        "PathLike = Union[str, Path]\n",
        "PathLike = Union[str, Path]\n" + COORD_BLOCK,
        1,
    )

    text = re.sub(
        r"def frame0_query_indices_for_sequence\(\s*json_path: PathLike,\s*sequence_name: str,\s*\*,\s*data_root: Optional\[PathLike\] = None,\s*num_frames: int = 64,\s*\)",
        "def frame0_query_indices_for_sequence(\n    json_path: PathLike,\n    sequence_name: str,\n    *,\n    data_root: Optional[PathLike] = None,\n    num_frames: int = 64,\n    coord_convention: Optional[str] = None,\n)",
        text,
        count=1,
    )

    # Replace frame0 body: after loading tracks_raw, use normalize + ref0_to_cam_tracks
    new_frame0 = """    tracks_raw = np.asarray(_load_npy("trajs_3d"), dtype=np.float64)
    visibility = np.asarray(_load_npy("visibs"), dtype=bool)
    extr_raw = np.asarray(view["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(view["cam_in"], dtype=np.float64).reshape(-1)[:4]
    tracks_ref0, extr_w2c = normalize_mf_tracks_to_opend4rt(
        tracks_raw[np.newaxis, ...],
        extr_raw[np.newaxis, ...],
        coord_convention=coord_convention,
    )
    cam0 = ref0_to_cam_tracks(tracks_ref0, extr_w2c)[0]
    depth0 = cam0[:, 2]
    uv0 = _project_cam_to_uv(cam0[np.newaxis, ...], intrinsics)[0]"""

    frame0_re = re.compile(
        r"    tracks_ref0 = np\.asarray\(_load_npy\(\"trajs_3d\"\), dtype=np\.float64\)\n"
        r"    visibility = np\.asarray\(_load_npy\(\"visibs\"\), dtype=bool\)\n"
        r"    extr0 = np\.asarray\(view\[\"extrinsics\"\], dtype=np\.float64\)\n"
        r"    intrinsics = np\.asarray\(view\[\"cam_in\"\], dtype=np\.float64\)\.reshape\(-1\)\[:4\]\n"
        r"(?:.*?\n)*?"
        r"    depth0 = cam0\[:, 2\]\n"
        r"    uv0 = _project_cam_to_uv\(cam0(?:\[np\.newaxis, \.\.\.\])?, intrinsics\)(?:\[0\])?",
        re.MULTILINE | re.DOTALL,
    )
    if not frame0_re.search(text):
        raise RuntimeError(f"frame0_query_indices body not found in {path}")
    text = frame0_re.sub(new_frame0, text, count=1)

    text = re.sub(
        r"def filter_sequences_with_frame0_queries\(\s*json_path: PathLike,\s*sequence_names: Sequence\[str\],\s*\*,\s*data_root: Optional\[PathLike\] = None,\s*num_frames: int = 64,\s*\)",
        "def filter_sequences_with_frame0_queries(\n    json_path: PathLike,\n    sequence_names: Sequence[str],\n    *,\n    data_root: Optional[PathLike] = None,\n    num_frames: int = 64,\n    coord_convention: Optional[str] = None,\n)",
        text,
        count=1,
    )
    text = text.replace(
        "qi = frame0_query_indices_for_sequence(json_path, name, data_root=data_root, num_frames=num_frames)",
        "qi = frame0_query_indices_for_sequence(\n            json_path, name, data_root=data_root, num_frames=num_frames, coord_convention=coord_convention\n        )",
        1,
    )

    text = re.sub(
        r"def load_sequence_from_json\(\s*json_path: PathLike,\s*sequence_name: str,\s*\*,\s*data_root: Optional\[PathLike\] = None,\s*num_frames: int = 64,\s*load_rgb: bool = True,\s*\)",
        "def load_sequence_from_json(\n    json_path: PathLike,\n    sequence_name: str,\n    *,\n    data_root: Optional[PathLike] = None,\n    num_frames: int = 64,\n    load_rgb: bool = True,\n    coord_convention: Optional[str] = None,\n)",
        text,
        count=1,
    )

    text = text.replace(
        "    data_root = Path(data_root)\n    with open(json_path, \"r\", encoding=\"utf-8\") as f:",
        "    data_root = Path(data_root)\n    cc = normalize_coord_convention(coord_convention)\n    with open(json_path, \"r\", encoding=\"utf-8\") as f:",
        1,
    )

    load_re = re.compile(
        r"    tracks_ref0 = np\.stack\(tracks_ref0_list, axis=0\)\n"
        r"    visibility = np\.stack\(vis_list, axis=0\)\n"
        r"(    valids = np\.stack\(val_list, axis=0\)\n)?"
        r"    extrinsics_w2c = np\.stack\(extr_list, axis=0\)",
        re.MULTILINE,
    )
    if not load_re.search(text):
        raise RuntimeError(f"load_sequence stack block not found in {path}")

    def _load_repl(m: re.Match) -> str:
        valids_line = m.group(1) if m.lastindex and m.group(1) else ""
        return (
            "    tracks_raw = np.stack(tracks_ref0_list, axis=0)\n"
            "    visibility = np.stack(vis_list, axis=0)\n"
            f"{valids_line}"
            "    extr_raw = np.stack(extr_list, axis=0)\n"
            "    tracks_ref0, extrinsics_w2c = normalize_mf_tracks_to_opend4rt(\n"
            "        tracks_raw, extr_raw, coord_convention=cc\n"
            "    )"
        )

    text = load_re.sub(_load_repl, text, count=1)

    path.write_text(text, encoding="utf-8")
    print(f"patched: {path}")


def main() -> None:
    for p in LOADER_PATHS:
        patch_loader(p)


if __name__ == "__main__":
    main()
