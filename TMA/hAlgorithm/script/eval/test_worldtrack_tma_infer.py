#!/usr/bin/env python3
"""Unit tests for WorldTrack prediction source composition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hAlgorithm.eval.worldtrack_tma_infer import (  # noqa: E402
    PRED_3D_SOURCE_WARP3D,
    PRED_3D_SOURCE_WARP3D_DELTA,
    _frame0_pred_positions,
    compose_warp3d_from_track,
    normalize_pred_3d_source,
    pred_field_label,
    resolve_pred_3d_source,
)


class _Track:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _OutputView:
    def __init__(self, track_3d: dict):
        self.track_3d = track_3d


def test_compose_warp3d_absolute():
    tr = _Track(warp3d=np.array([[1.0, 2.0, 3.0]]))
    out = compose_warp3d_from_track(tr, pred_3d_source=PRED_3D_SOURCE_WARP3D)
    assert np.allclose(out, [[1.0, 2.0, 3.0]])


def test_warp3d_mode_does_not_subtract_delta():
    """warp3d mode must return warp3d as-is, never warp3d - warp3d_delta."""
    tr = _Track(
        warp3d=np.array([[3.0, 1.0, 0.0]]),
        warp3d_delta=np.array([[0.0, 1.0, 0.0]]),
    )
    out = compose_warp3d_from_track(tr, pred_3d_source=PRED_3D_SOURCE_WARP3D)
    assert np.allclose(out, [[3.0, 1.0, 0.0]])
    assert not np.allclose(out, [[3.0, 0.0, 0.0]])


def test_compose_warp3d_delta_prefers_src_points():
    tr = _Track(
        src_index=0,
        tgt_index=1,
        src_points=np.array([[1.0, 0.0, 0.0]]),
        warp3d_delta=np.array([[0.0, 1.0, 0.0]]),
        warp3d=np.array([[9.0, 9.0, 9.0]]),
        src_3d_gt=np.array([[2.0, 0.0, 0.0]]),
    )
    out = compose_warp3d_from_track(tr, pred_3d_source=PRED_3D_SOURCE_WARP3D_DELTA)
    assert np.allclose(out, [[1.0, 1.0, 0.0]])


def test_compose_warp3d_delta_uses_frame0_src_when_track_has_no_src_points():
    tr = _Track(
        src_index=0,
        tgt_index=1,
        warp3d_delta=np.array([[0.0, 1.0, 0.0]]),
        warp3d=np.array([[1.0, 1.0, 0.0]]),
    )
    frame0 = np.array([[1.0, 0.0, 0.0]])
    out = compose_warp3d_from_track(
        tr, pred_3d_source=PRED_3D_SOURCE_WARP3D_DELTA, frame0_src=frame0
    )
    assert np.allclose(out, [[1.0, 1.0, 0.0]])


def test_frame0_prefers_src_points_on_pair_0t():
    outputs = [
        _OutputView(
            {
                1: _Track(
                    src_index=0,
                    tgt_index=1,
                    src_points=np.array([[1.0, 0.0, 0.0]]),
                    warp3d=np.array([[9.0, 9.0, 9.0]]),
                ),
                0: _Track(
                    src_index=0,
                    tgt_index=0,
                    warp3d=np.array([[2.0, 0.0, 0.0]]),
                ),
            }
        )
    ]
    pts = _frame0_pred_positions(outputs)
    assert np.allclose(pts, [[1.0, 0.0, 0.0]])


def test_frame0_falls_back_to_pair_00_warp3d():
    outputs = [
        _OutputView(
            {
                1: _Track(
                    src_index=0,
                    tgt_index=1,
                    warp3d=np.array([[9.0, 9.0, 9.0]]),
                    warp3d_delta=np.array([[0.0, 1.0, 0.0]]),
                ),
                0: _Track(
                    src_index=0,
                    tgt_index=0,
                    warp3d=np.array([[2.0, 0.0, 0.0]]),
                ),
            }
        )
    ]
    pts = _frame0_pred_positions(outputs)
    assert np.allclose(pts, [[2.0, 0.0, 0.0]])


def test_frame0_does_not_use_warp3d_minus_delta():
    outputs = [
        _OutputView(
            {
                1: _Track(
                    src_index=0,
                    tgt_index=1,
                    warp3d=np.array([[3.0, 1.0, 0.0]]),
                    warp3d_delta=np.array([[0.0, 1.0, 0.0]]),
                ),
            }
        )
    ]
    try:
        _frame0_pred_positions(outputs)
    except RuntimeError as exc:
        assert "(0, 0)" in str(exc) or "0, 0" in str(exc)
    else:
        raise AssertionError("expected missing (0,0) pair to raise")


def test_compose_warp3d_delta_rejects_gt_fallback():
    tr = _Track(
        src_index=0,
        tgt_index=1,
        warp3d_delta=np.array([[0.0, 1.0, 0.0]]),
        src_3d_gt=np.array([[2.0, 0.0, 0.0]]),
    )
    try:
        compose_warp3d_from_track(tr, pred_3d_source=PRED_3D_SOURCE_WARP3D_DELTA)
    except RuntimeError as exc:
        assert "src_points" in str(exc) or "frame-0" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when src_points is missing")


def test_resolve_pred_3d_source():
    assert resolve_pred_3d_source(pred_3d_source="warp3d") == PRED_3D_SOURCE_WARP3D
    assert resolve_pred_3d_source(use_warp3d_delta=True) == PRED_3D_SOURCE_WARP3D_DELTA
    assert normalize_pred_3d_source("delta") == PRED_3D_SOURCE_WARP3D_DELTA
    assert pred_field_label(PRED_3D_SOURCE_WARP3D_DELTA) == "warp3d_delta+src_points"


def test_sample_video_rgb_at_track_uv():
    from hAlgorithm.eval.worldtrack_vis import sample_video_rgb_at_track_uv

    video = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    video[0, 1, 2] = [10, 20, 30]
    video[1, 3, 4] = [40, 50, 60]
    uv = np.array([[[2.0, 1.0], [np.nan, 0.0]], [[4.0, 3.0], [1.0, 1.0]]], dtype=np.float64)
    mask = np.array([[True, False], [True, True]], dtype=bool)
    cols = sample_video_rgb_at_track_uv(video, uv, mask)
    assert cols.shape == (3, 3)
    assert np.allclose(cols[0], [10, 20, 30])
    assert np.allclose(cols[1], [40, 50, 60])
    assert np.allclose(cols[2], [0, 0, 0])


if __name__ == "__main__":
    test_compose_warp3d_absolute()
    test_warp3d_mode_does_not_subtract_delta()
    test_compose_warp3d_delta_prefers_src_points()
    test_compose_warp3d_delta_uses_frame0_src_when_track_has_no_src_points()
    test_frame0_prefers_src_points_on_pair_0t()
    test_frame0_falls_back_to_pair_00_warp3d()
    test_frame0_does_not_use_warp3d_minus_delta()
    test_compose_warp3d_delta_rejects_gt_fallback()
    test_resolve_pred_3d_source()
    test_sample_video_rgb_at_track_uv()
    print("ok")
