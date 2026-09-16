import logging
import os
import traceback

import cv2
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import map_coordinates

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    rr = None
    rrb = None
    logging.getLogger(__name__).warning("[VisMotion] rerun-sdk not found. 3D visualization will be disabled.")

from .utils import MotionVisUtils
from .gif_generator import GifGenerator

logger = logging.getLogger(__name__)


def _rerun_line_grid_hidden_kw():
    """Return ``{'line_grid': LineGrid3D(visible=False)}`` when the SDK exposes it.

    Older ``rerun`` builds omit ``LineGrid3D`` on ``rerun.blueprint``; try submodule imports,
    otherwise leave the default grid (empty dict).
    """
    if rrb is None:
        return {}
    LG = getattr(rrb, "LineGrid3D", None)
    if LG is None:
        try:
            from rerun.blueprint.archetypes.line_grid3d import LineGrid3D as LG
        except ImportError:
            try:
                from rerun.blueprint_archetypes import LineGrid3D as LG
            except ImportError:
                logger.debug(
                    "[VisMotion] LineGrid3D unavailable in this rerun SDK; spatial views keep default grid."
                )
                return {}
    try:
        return {"line_grid": LG(visible=False)}
    except TypeError:
        return {}


class MotionVisualizer3D:
    """Render dense motion and scene flow in 3D using Rerun."""

    SAMPLE_STEP = 4
    POINT_RADIUS = 0.015
    MAX_POINTS_PER_VIEW = 5000
    REF_FRAME = 0
    RR_NAME_PREFIX = "Motion"
    RRD_FILENAME_PREFIX = "vis_3d"
    GENERATE_FLOW_GIF = True

    _GT_OUTLIER_IQR_FACTOR = 50.0
    _GT_OUTLIER_MIN_SAMPLES = 20
    _GT_OUTLIER_MIN_THRESHOLD = 5.0
    _GT_DYNAMIC_THRESHOLD = 0.01

    # Frustum visual size (wire depth, line width, label dot) — 1/4 of previous defaults.
    FRUSTUM_SCALE = 0.15
    _FRUSTUM_LINE_STRIP_RADIUS = 0.003
    _FRUSTUM_GT_COLOR = np.array([50, 220, 50, 220], dtype=np.uint8)
    _FRUSTUM_PRED_COLOR = np.array([255, 140, 50, 220], dtype=np.uint8)
    _WORLD_PREFIXES = ("world", "world_gt_trajs", "world_pred")

    # Fading trail configuration for predicted trajectories
    TRAIL_LENGTH = 15             # Number of historical frames to show
    TRAIL_MIN_ALPHA = 0.3         # Opacity of oldest trail segment
    TRAIL_MAX_ALPHA = 1.0         # Opacity of newest trail segment
    TRAIL_MIN_RADIUS = 0.002      # Line width of oldest segment
    TRAIL_MAX_RADIUS = 0.006      # Line width of newest segment
    # When True, each trail uses a fixed hue from its reference-frame origin (spatial rainbow);
    # the same point keeps the same color across time. When False, use motion-direction colors.
    TRAIL_RAINBOW_COLORS = True

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def _prefer_gt(self):
        """When True (default), use GT pointmap / extrinsics for 3D vis."""
        return self.cfg.get("vis_3d_prefer_gt", True)

    @property
    def _show_ref_in_pred_view(self):
        """When False (default), hide the static reference pointcloud in
        the 'Predicted Target' view so predicted motion is not occluded."""
        return self.cfg.get("vis_3d_show_ref_in_pred_view", False)

    @property
    def _depth_max(self):
        """Maximum absolute coordinate value for 3D points.
        Points with any coordinate exceeding this are filtered out.
        Set via ``vis_3d_depth_max`` in the experiment config (default 20.0).
        """
        return float(self.cfg.get("vis_3d_depth_max", 20.0))

    @property
    def _view_coordinates(self):
        """World-axis convention for the Rerun viewer (default ``"RUB"``).

        The Rerun web viewer does not expose an entry for setting
        ``ViewCoordinates`` from the UI; without a static log on the root
        entity the 3D view has no notion of "up" and only mouse-orbit works.
        Override via ``vis_3d_view_coordinates`` in the experiment config
        with values such as ``"RUB"``, ``"RUF"``, ``"FLU"``, ``"RDF"``,
        ``"RIGHT_HAND_Y_UP"``, ``"RIGHT_HAND_Z_UP"``, etc.  Set to ``None``
        / empty string to disable logging the coordinate convention.
        """
        return self.cfg.get("vis_3d_view_coordinates", "RUB")

    def _log_view_coordinates(self):
        """Log the configured ``ViewCoordinates`` on the root entity (static).

        Logged on ``"/"`` so all child 3D views inherit the convention,
        giving the viewer a defined "up" direction.  Unknown names fall
        back to ``RUB`` with a warning; ``None`` / empty string disables.
        """
        if rr is None:
            return
        name = self._view_coordinates
        if name is None:
            return
        vc_name = str(name).strip().upper()
        if not vc_name:
            return
        vc = getattr(rr.ViewCoordinates, vc_name, None)
        if vc is None:
            logger.warning(
                "[VisMotion] unknown vis_3d_view_coordinates=%r; falling back to RUB",
                name,
            )
            vc = getattr(rr.ViewCoordinates, "RUB", None)
            if vc is None:
                return
        rr.log("/", vc, static=True)

    def _depth_valid(self, pts):
        """Return boolean mask: True where all |x|,|y|,|z| <= _depth_max and finite."""
        return np.isfinite(pts).all(axis=1) & (np.abs(pts).max(axis=1) <= self._depth_max)

    def _rerun_coord_row_mat(self):
        """3×3 row-vector map ``p_rr = p_in @ M`` (``rerun_coord_mat3`` overrides named frames).

        Preset ``rdf`` / ``z_up_to_rdf``: assumes pipeline world is right-handed with
        +X right, +Y forward, +Z up, and maps to **RDF** (+X right, +Y down, +Z forward).
        """
        cfg = getattr(self, "cfg", None) or {}
        custom = cfg.get("rerun_coord_mat3") or cfg.get("vis_3d_rerun_coord_mat3")
        if custom is not None:
            return np.asarray(custom, dtype=np.float64).reshape(3, 3)
        name = str(cfg.get("rerun_coord_frame") or cfg.get("vis_3d_rerun_coord_frame") or "default").strip().lower()
        if name in ("", "default", "none", "identity"):
            return np.eye(3, dtype=np.float64)
        if name in ("rdf", "z_up_to_rdf"):
            return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        logger.warning("[VisMotion] unknown rerun_coord_frame=%r; using identity", name)
        return np.eye(3, dtype=np.float64)

    def _rerun_axis_mul_post(self):
        """Optional length-3 scale applied *after* ``_rerun_coord_row_mat`` (e.g. ``[1,-1,1]``)."""
        cfg = getattr(self, "cfg", None) or {}
        s = cfg.get("rerun_axis_sign") or cfg.get("vis_3d_rerun_axis_sign")
        if s is None:
            return None
        a = np.asarray(s, dtype=np.float64).reshape(3)
        if np.allclose(a, 1.0):
            return None
        return a

    def _rerun_coord_apply_pts(self, arr):
        """Positions ``(..., 3)`` or ``(3,)`` for Rerun; free vectors use the same linear map."""
        if arr is None:
            return None
        x = np.asarray(arr, dtype=np.float64)
        if x.size == 0:
            return x
        M = self._rerun_coord_row_mat()
        ax = self._rerun_axis_mul_post()
        if np.allclose(M, np.eye(3)) and ax is None:
            return x
        if not np.allclose(M, np.eye(3)):
            if x.ndim == 1 and x.shape == (3,):
                x = (x.reshape(1, 3) @ M).ravel()
            else:
                x = x @ M
        if ax is not None:
            m = ax.reshape(*((1,) * max(0, x.ndim - 1)), 3)
            x = x * m
        return x

    def _rerun_coord_apply_vecs(self, arr):
        """Same as ``_rerun_coord_apply_pts`` (rotation + per-axis scale on directions)."""
        return self._rerun_coord_apply_pts(arr)

    def _setup_blueprint(self):
        if rrb is None:
            return
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(name="GT Target + Pred Flow", origin="world/", **_rerun_line_grid_hidden_kw()),
                    rrb.Horizontal(rrb.Spatial2DView(name="Reference RGB", origin="reference_rgb"),
                                   rrb.Spatial2DView(name="Target RGB", origin="target_rgb")), row_shares=[3, 1]),
                rrb.Vertical(
                    rrb.Spatial3DView(name="GT Trajectories", origin="world_gt_trajs/", **_rerun_line_grid_hidden_kw()),
                ),
                rrb.Vertical(
                    rrb.Spatial3DView(name="Predicted Target", origin="world_pred/", **_rerun_line_grid_hidden_kw()),
                ),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial3DView(name="Dynamic points (warp3d)", origin="world_warp3d_dynamic/", **_rerun_line_grid_hidden_kw()),
                        rrb.Spatial3DView(name="Scene flow (warp3d_delta)", origin="world_warp3d_delta_sf/", **_rerun_line_grid_hidden_kw()),
                    ),
                ),
                column_shares=[1, 1, 1, 1]), collapse_panels=True)
        rr.send_blueprint(blueprint)

    def _get_c2w(self, out):
        """Get camera-to-world matrix in *metric* world coordinates.

        ``extrinsics`` is stored with a normalized translation (divided by the
        depth-normalization scale in ``get_inputs()``), while ``pointmap_gt``
        and ``trajs_3d`` are stored in metric units.  Re-scaling the
        translation by ``prompt_scale`` before inversion gives a c2w that is
        consistent with both the dense point-cloud and the sparse trajectories.

        prefer_gt=True  → extrinsics (GT) first, fall back to extrinsics_pred
        prefer_gt=False → extrinsics_pred first, fall back to extrinsics
        """
        if self._prefer_gt:
            ext = getattr(out, 'extrinsics', None)
            if ext is None:
                ext = getattr(out, 'extrinsics_pred', None)
        else:
            ext = getattr(out, 'extrinsics_pred', None)
            if ext is None:
                ext = getattr(out, 'extrinsics', None)
        if ext is None:
            fi = getattr(out, 'frame_index', '?')
            vi = getattr(out, 'view_index', '?')
            logger.warning("[_get_c2w] frame=%s view=%s: extrinsics is None → using identity (points stay in camera space!)", fi, vi)
            return np.eye(4)
        ext_np = MotionVisUtils.to_numpy(ext).copy().astype(np.float64)
        # Restore metric translation: extrinsics[:3,3] was divided by
        # prompt_scale in get_inputs(); multiply back so that
        # inv(ext_metric) @ pts_cam_metric = pts_world_metric.
        scale_val = getattr(out, 'prompt_scale', None)
        if scale_val is not None:
            s = float(np.asarray(scale_val).flat[0])
            if s > 0:
                ext_np[:3, 3] *= s
        c2w = np.linalg.inv(ext_np)
        fi = getattr(out, 'frame_index', '?')
        vi = getattr(out, 'view_index', '?')
        logger.debug("[_get_c2w] frame=%s view=%s: ext[:3,3]=%s  c2w[:3,3]=%s  scale=%s",
                     fi, vi, ext_np[:3, 3].round(4), c2w[:3, 3].round(4), scale_val)
        return c2w

    def _get_pointmap(self, out):
        """Get pointmap, respecting GT preference.

        prefer_gt=True  → pointmap_gt first, fall back to pointmap (predicted)
        prefer_gt=False → pointmap (predicted) first, fall back to pointmap_gt
        """
        if self._prefer_gt:
            pmap = getattr(out, 'pointmap_gt', None)
            if pmap is None:
                pmap = getattr(out, 'pointmap', None)
        else:
            pmap = getattr(out, 'pointmap', None)
            if pmap is None:
                pmap = getattr(out, 'pointmap_gt', None)
        return pmap

    def _get_pointmap_world(self, out):
        """Return (pointmap, already_in_world_space) tuple.

        If ``pointmap_gt_global`` is available it is already transformed to the
        shared world coordinate frame (camera-0 space) by the dataset's
        ``normalize_cameras_base_first``.  In that case no additional c2w
        transformation is needed (identity is used instead).

        Falls back to ``_get_pointmap`` + ``_get_c2w`` when the pre-transformed
        pointmap is absent.
        """
        world_pmap = getattr(out, 'pointmap_gt_global', None)
        if world_pmap is not None:
            return world_pmap, True
        return self._get_pointmap(out), False

    def _get_pointmap_fallback(self, out):
        """Get the non-preferred pointmap (opposite of _get_pointmap)."""
        if self._prefer_gt:
            return getattr(out, 'pointmap', None)
        else:
            return getattr(out, 'pointmap_gt', None)

    # ------------------------------------------------------------------
    # Camera frustum helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frustum_lines(c2w, K, H, W, scale):
        """Compute 8 frustum wireframe segments in world space.

        Returns (cam_center [3], segments [8, 2, 3]).
        """
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        d = scale
        corners_cam = np.array([
            [-cx / fx * d,       -cy / fy * d,       d],
            [(W - cx) / fx * d,  -cy / fy * d,       d],
            [(W - cx) / fx * d,  (H - cy) / fy * d,  d],
            [-cx / fx * d,       (H - cy) / fy * d,  d],
        ])
        R, t = c2w[:3, :3], c2w[:3, 3]
        cw = (R @ corners_cam.T).T + t
        center = t.copy()
        segs = np.array([
            [center, cw[0]], [center, cw[1]], [center, cw[2]], [center, cw[3]],
            [cw[0], cw[1]], [cw[1], cw[2]], [cw[2], cw[3]], [cw[3], cw[0]],
        ])
        return center, segs

    def _log_frustums(
        self,
        out,
        tag,
        frame_idx,
        view_num=1,
        mv_outputs=None,
        static=False,
        panel_prefixes=None,
        pred_camera_only_prefixes=None,
    ):
        """Log GT and predicted camera frustums under one or more 3D panel roots.

        Args:
            out:        primary output object (view 0) for this frame.
            tag:        entity name prefix, e.g. ``"ref"`` or ``"tgt"``.
            frame_idx:  frame index (for label text).
            view_num:   total views; when >1 and *mv_outputs* is given,
                        frustums for every view of this frame are drawn.
            mv_outputs: full output list (needed for multi-view).
            static:     if True the log is time-independent.
            panel_prefixes: Rerun path prefixes without trailing slash (e.g. ``win1``).
                If None, uses ``_WORLD_PREFIXES`` (dense layout).
            pred_camera_only_prefixes: Panel prefixes for which only *pred* frustums are logged
                (GT cameras omitted). E.g. sparse warp panels ``world_warp3d_dynamic`` / ``world_warp3d_delta_sf``.
        """
        prefixes = self._WORLD_PREFIXES if panel_prefixes is None else tuple(panel_prefixes)
        pred_only = frozenset(pred_camera_only_prefixes) if pred_camera_only_prefixes else frozenset()
        outputs_to_draw = [(0, out)]
        if view_num > 1 and mv_outputs is not None:
            for v in range(1, view_num):
                idx = frame_idx * view_num + v
                if idx < len(mv_outputs):
                    outputs_to_draw.append((v, mv_outputs[idx]))

        for v_idx, cur_out in outputs_to_draw:
            K = MotionVisUtils.to_numpy(getattr(cur_out, 'intrinsics', None))
            if K is None:
                continue
            H, W = cur_out.pointmap_h, cur_out.pointmap_w
            v_suffix = f"_v{v_idx}" if view_num > 1 else ""

            configs = []
            ext_gt = MotionVisUtils.to_numpy(getattr(cur_out, 'extrinsics', None))
            if ext_gt is not None:
                configs.append(("gt", np.linalg.inv(ext_gt),
                                self._FRUSTUM_GT_COLOR, f"GT F{frame_idx}{v_suffix}"))
            ext_pred = MotionVisUtils.to_numpy(getattr(cur_out, 'extrinsics_pred', None))
            if ext_pred is not None:
                configs.append(("pred", np.linalg.inv(ext_pred),
                                self._FRUSTUM_PRED_COLOR, f"Pred F{frame_idx}{v_suffix}"))

            line_r = float(getattr(self, "_FRUSTUM_LINE_STRIP_RADIUS", 0.00075))
            center_r = self.POINT_RADIUS * 0.75
            for kind, c2w, color, label in configs:
                center, segs = self._frustum_lines(c2w, K, H, W, self.FRUSTUM_SCALE)
                center = self._rerun_coord_apply_pts(center)
                segs = self._rerun_coord_apply_pts(segs)
                entity = f"{tag}_{kind}{v_suffix}"
                for pfx in prefixes:
                    if kind == "gt" and pfx in pred_only:
                        continue
                    rr.log(f"{pfx}/cameras/{entity}/frustum",
                           rr.LineStrips3D(segs, colors=color, radii=line_r),
                           static=static)
                    rr.log(f"{pfx}/cameras/{entity}/center",
                           rr.Points3D([center], colors=[color[:3]],
                                       labels=[label],
                                       radii=center_r),
                           static=static)

    @staticmethod
    def _sample_3d_from_pointmap_bilinear(pointmap, coords_2d, depth_min=0.01, depth_max=20.0):
        """Nearest-neighbour lookup into the XYZ pointmap.

        Intentionally uses order=0 (nearest neighbour) instead of bilinear
        (order=1).  At object edges the XYZ channels of a pointmap have a
        sharp depth discontinuity between foreground and background.  Bilinear
        interpolation blends those two depths and creates phantom points that
        float between the two surfaces, making edge trajectories appear to land
        on the background.  Nearest-neighbour snaps to the closest pixel,
        preserving the hard edge.
        """
        H, W = pointmap.shape[:2]
        u, v = coords_2d[:, 0], coords_2d[:, 1]

        valid_mask = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return np.zeros((0, 3)), np.array([], dtype=int)

        u_v = np.clip(np.round(u[valid_mask]).astype(int), 0, W - 1)
        v_v = np.clip(np.round(v[valid_mask]).astype(int), 0, H - 1)
        pts = pointmap[v_v, u_v]  # direct integer indexing – no interpolation

        depth_mask = (pts[:, 2] > depth_min) & (pts[:, 2] < depth_max)
        return pts[depth_mask], valid_indices[depth_mask]

    def run(self, mv_outputs, out_dir, data_idx, meta_data=None):
        if rr is None:
            return

        rr.init(f"{self.RR_NAME_PREFIX}_{data_idx}", spawn=False)
        save_path = os.path.join(out_dir, "rerun_vis", f"{self.RRD_FILENAME_PREFIX}_{data_idx:06d}.rrd")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._log_view_coordinates()
        self._setup_blueprint()

        frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]
        logger.info("[Rerun3D] data_idx=%d, frames=%d, views=%d, total_outputs=%d, prefer_gt=%s",
                    data_idx, frame_num, view_num, len(mv_outputs), self._prefer_gt)

        try:
            ref_data = self._build_reference_pointcloud(mv_outputs, self.REF_FRAME, view_num)
        except Exception as e:
            logger.error("[Rerun3D] _build_reference_pointcloud failed: %s\n%s", e, traceback.format_exc())
            return

        if not ref_data["pts"]:
            logger.warning("[Rerun3D] reference pointcloud is empty — skipping .rrd save. "
                           "Check depth_mask / pointmap validity.")
            return
        ref_pts, ref_colors = np.concatenate(ref_data["pts"], axis=0), np.concatenate(ref_data["colors"], axis=0)
        logger.info("[Rerun3D] reference pointcloud: %d pts, Z range [%.3f, %.3f]",
                    len(ref_pts), ref_pts[:, 2].min(), ref_pts[:, 2].max())

        trajs_data = self._extract_gt_trajectories(mv_outputs, meta_data)

        flow_img_dir = os.path.join(os.path.dirname(save_path), f"flow_images_{data_idx:06d}")
        os.makedirs(flow_img_dir, exist_ok=True)

        ref_rgb = None
        ref_out_v0 = mv_outputs[self.REF_FRAME * view_num]
        if getattr(ref_out_v0, 'rgb', None) is not None:
            ref_rgb = MotionVisUtils.normalize_rgb(ref_out_v0.rgb)
            cv2.imwrite(os.path.join(flow_img_dir, f"rgb_f{self.REF_FRAME:03d}.png"), cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR))
            rr.log("reference_rgb", rr.Image(ref_rgb), static=True)

        ref_dgs_rgb = getattr(ref_out_v0, 'dgs_render_rgb', None)
        if ref_dgs_rgb is not None:
            ref_dgs_img = (np.clip(ref_dgs_rgb, 0, 1) * 255).astype(np.uint8)
            rr.log("dgs_render_rgb", rr.Image(ref_dgs_img), static=True)

        ref_pts_r = self._rerun_coord_apply_pts(ref_pts)
        rr.log("world/reference_pointcloud", rr.Points3D(ref_pts_r, colors=ref_colors, radii=self.POINT_RADIUS), static=True)
        rr.log("world_gt_trajs/reference_pointcloud", rr.Points3D(ref_pts_r, colors=ref_colors, radii=self.POINT_RADIUS * 0.8), static=True)
        if self._show_ref_in_pred_view:
            rr.log("world_pred/reference_pointcloud", rr.Points3D(ref_pts_r, colors=ref_colors, radii=self.POINT_RADIUS), static=True)

        self._log_static_query_overlay(mv_outputs, meta_data)

        self._rerun_log_sparse_frame0_init(mv_outputs, meta_data, ref_data, frame_num, view_num)

        # Reference frame camera frustums (static, all views)
        ref_out = mv_outputs[self.REF_FRAME * view_num]
        self._log_frustums(ref_out, "ref", self.REF_FRAME,
                           view_num=view_num, mv_outputs=mv_outputs, static=True)

        # History for fading trails: frame -> {positions, colors, origins, vectors}
        pred_history = {}

        for tgt_frame in range(frame_num):
            rr.set_time_sequence("target_frame", tgt_frame)
            try:
                # Target frame camera frustums (per-frame, view 0 only)
                tgt_out = mv_outputs[tgt_frame * view_num]
                self._log_frustums(tgt_out, "tgt", tgt_frame, static=False)

                if trajs_data["available"]:
                    self._visualize_gt_trajectory(mv_outputs, trajs_data, self.REF_FRAME, tgt_frame, meta_data)
                frame_positions = self._process_target_frame(
                    mv_outputs, meta_data, ref_data, self.REF_FRAME, tgt_frame, view_num, flow_img_dir, pred_history
                )
                if frame_positions is not None:
                    pred_history[tgt_frame] = frame_positions
                self._log_gaussian_renders(mv_outputs, tgt_frame, view_num, flow_img_dir)
            except Exception as e:
                logger.error("[Rerun3D] tgt_frame=%d failed: %s\n%s", tgt_frame, e, traceback.format_exc())
                continue

        rr.save(save_path)
        if self.GENERATE_FLOW_GIF:
            GifGenerator.generate_flow_gif(flow_img_dir, fps=2.0, size=280)

    def _rerun_log_sparse_frame0_init(self, mv_outputs, meta_data, ref_data, frame_num, view_num):
        """Hook for sparse Rerun: log frame-0 query init positions (default no-op)."""
        return

    def _log_gaussian_renders(self, mv_outputs, tgt_frame, view_num, flow_img_dir):
        """Log 4DGS rendered RGB and depth images to Rerun for a target frame."""
        tgt_out = mv_outputs[tgt_frame * view_num]

        dgs_rgb = getattr(tgt_out, 'dgs_render_rgb', None)
        if dgs_rgb is not None:
            dgs_rgb_img = (np.clip(dgs_rgb, 0, 1) * 255).astype(np.uint8)
            rr.log("dgs_render_rgb", rr.Image(dgs_rgb_img))
            cv2.imwrite(
                os.path.join(flow_img_dir, f"dgs_rgb_f{tgt_frame:03d}.png"),
                cv2.cvtColor(dgs_rgb_img, cv2.COLOR_RGB2BGR),
            )

        dgs_depth = getattr(tgt_out, 'dgs_render_depth', None)
        if dgs_depth is not None:
            depth_2d = np.squeeze(dgs_depth)
            if depth_2d.ndim == 3 and depth_2d.shape[0] == 1:
                depth_2d = depth_2d[0]
            valid = np.isfinite(depth_2d) & (depth_2d > 0)
            if valid.any():
                lo, hi = float(depth_2d[valid].min()), float(depth_2d[valid].max())
                depth_norm = np.clip((depth_2d - lo) / max(hi - lo, 1e-6), 0, 1)
            else:
                depth_norm = np.zeros_like(depth_2d)
            depth_colored = (cm.get_cmap('turbo')(depth_norm)[:, :, :3] * 255).astype(np.uint8)
            rr.log("dgs_render_depth", rr.Image(depth_colored))
            cv2.imwrite(
                os.path.join(flow_img_dir, f"dgs_depth_f{tgt_frame:03d}.png"),
                cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR),
            )

    def _compute_global_flow_range(self, mv_outputs, ref_frame, frame_num, view_num):
        """Compute global magnitude max and per-axis range across all targets.

        Returns (global_mag_max, global_xyz_range) where global_xyz_range is a
        dict with 'min' and 'max' arrays of shape (3,).  Used for consistent
        colour scaling across target frames.
        """
        all_mag_max, all_min, all_max = 0.0, None, None
        ref_out = mv_outputs[ref_frame * view_num]
        sf = MotionVisUtils.to_numpy(getattr(ref_out, 'scene_flow_pred', None)) \
            if getattr(ref_out, 'scene_flow_pred', None) is not None else None
        if sf is None:
            return 0.0, {"min": np.zeros(3), "max": np.ones(3)}
        for t in range(frame_num):
            if t == ref_frame or t >= sf.shape[0]:
                continue
            flow = sf[t]  # (3, H, W)
            mag = np.linalg.norm(flow, axis=0)
            all_mag_max = max(all_mag_max, float(mag.max()))
            fmin = flow.reshape(3, -1).min(axis=1)
            fmax = flow.reshape(3, -1).max(axis=1)
            all_min = np.minimum(all_min, fmin) if all_min is not None else fmin
            all_max = np.maximum(all_max, fmax) if all_max is not None else fmax
        if all_min is None:
            all_min, all_max = np.zeros(3), np.ones(3)
        return all_mag_max, {"min": all_min, "max": all_max}

    def _log_static_query_overlay(self, mv_outputs, meta_data):
        pass

    @staticmethod
    def _reshape_pmap(pmap_np, H, W):
        """Reshape flat (N, 3) pointmap to (H, W, 3) using exact dimensions."""
        N = pmap_np.shape[0]
        if N != H * W:
            raise ValueError(f"Pointmap size {N} does not match {H}×{W}={H * W}. "
                             "Pass the correct GT dimensions via pointmap_gt_h/w.")
        return pmap_np.reshape(H, W, 3)

    @staticmethod
    def _get_pmap_hw(out, prefer_gt=True):
        """Return (H, W) matching whichever pointmap _get_pointmap() will use."""
        if prefer_gt and getattr(out, 'pointmap_gt', None) is not None:
            gt_h = getattr(out, 'pointmap_gt_h', None)
            gt_w = getattr(out, 'pointmap_gt_w', None)
            if gt_h is not None and gt_w is not None:
                return gt_h, gt_w
        return out.pointmap_h, out.pointmap_w

    def _build_reference_pointcloud(self, mv_outputs, ref_frame, view_num):
        ref_data = {"pts": [], "colors": [], "valid_masks": {}, "pts_local": {}, "c2w": {}, "pmap_dims": {}}
        for v_idx in range(view_num):
            ref_out = mv_outputs[ref_frame * view_num + v_idx]
            pmap, is_world = self._get_pointmap_world(ref_out)
            if pmap is None:
                logger.debug("[_build_ref_pc] view %d: pointmap is None, skipping", v_idx)
                continue

            H_pmap, W_pmap = self._get_pmap_hw(ref_out, prefer_gt=True)
            pts_local = self._reshape_pmap(MotionVisUtils.to_numpy(pmap), H_pmap, W_pmap)
            pts_sub = pts_local[::self.SAMPLE_STEP, ::self.SAMPLE_STEP].reshape(-1, 3)

            valid_mask = self._depth_valid(pts_sub)

            pts_valid = pts_sub[valid_mask]
            if len(pts_valid) == 0:
                pmap_fb = self._get_pointmap_fallback(ref_out)
                if pmap_fb is not None and pmap_fb is not pmap:
                    logger.info("[_build_ref_pc] view %d: preferred pointmap all invalid, retrying with fallback", v_idx)
                    H_pmap, W_pmap = self._get_pmap_hw(ref_out, prefer_gt=False)
                    pts_local = self._reshape_pmap(MotionVisUtils.to_numpy(pmap_fb), H_pmap, W_pmap)
                    pts_sub = pts_local[::self.SAMPLE_STEP, ::self.SAMPLE_STEP].reshape(-1, 3)
                    valid_mask = self._depth_valid(pts_sub)
                    pts_valid = pts_sub[valid_mask]
                    is_world = False
            if len(pts_valid) == 0:
                continue

            c2w = np.eye(4) if is_world else self._get_c2w(ref_out)
            ref_data["c2w"][v_idx] = c2w
            ref_data["pts_local"][v_idx] = pts_sub
            ref_data["valid_masks"][v_idx] = valid_mask
            ref_data["pmap_dims"][v_idx] = (H_pmap, W_pmap)
            logger.debug("[_build_ref_pc] view %d: is_world=%s c2w_t=%s", v_idx, is_world, c2w[:3, 3].round(4))

            pts_world = MotionVisUtils.apply_c2w(c2w, pts_valid)
            ref_data["pts"].append(pts_world)

            if getattr(ref_out, 'rgb', None) is not None:
                rgb_sub = cv2.resize(MotionVisUtils.normalize_rgb(ref_out.rgb), (W_pmap, H_pmap))
                ref_data["colors"].append(rgb_sub[::self.SAMPLE_STEP, ::self.SAMPLE_STEP].reshape(-1, 3)[valid_mask].astype(np.uint8))
            else:
                ref_data["colors"].append(np.full((len(pts_world), 3), 180, dtype=np.uint8))
        return ref_data

    def _extract_gt_trajectories(self, mv_outputs, meta_data):
        data = {"available": False, "trajs_3d": None, "trajs_2d": None, "colors": None, "w2c_first": None}
        if not mv_outputs or getattr(mv_outputs[0], 'trajs_3d', None) is None:
            return data

        out = mv_outputs[0]
        data.update({"available": True, "trajs_3d": MotionVisUtils.to_numpy(out.trajs_3d),
                     "trajs_2d": MotionVisUtils.to_numpy(getattr(out, 'trajs_2d', None)),
                     "visibs": MotionVisUtils.to_numpy(getattr(out, 'trajs_visibs', None)),
                     "valids": MotionVisUtils.to_numpy(getattr(out, 'trajs_valids', None))})

        if getattr(out, 'motion_extrinsics', None) is not None:
            ext = MotionVisUtils.to_numpy(out.motion_extrinsics)
            data["w2c_first"] = ext[0] if ext.ndim == 3 else ext

        num_trajs = data["trajs_3d"].shape[1]
        orig_h, orig_w = MotionVisUtils.get_orig_dims(meta_data)
        ref_rgb = MotionVisUtils.normalize_rgb(out.rgb) if getattr(out, 'rgb', None) is not None else None

        if ref_rgb is not None and data["trajs_2d"] is not None:
            scale = [ref_rgb.shape[1] / orig_w if orig_w else 1.0, ref_rgb.shape[0] / orig_h if orig_h else 1.0]
            pts_2d = np.nan_to_num(data["trajs_2d"][self.REF_FRAME] * scale, nan=0.0, posinf=0.0, neginf=0.0)
            pts_mapped = pts_2d.astype(int)
            valid = (pts_mapped[:, 0] >= 0) & (pts_mapped[:, 0] < ref_rgb.shape[1]) & \
                    (pts_mapped[:, 1] >= 0) & (pts_mapped[:, 1] < ref_rgb.shape[0])
            colors = np.full((num_trajs, 3), 128, dtype=np.uint8)
            colors[valid] = ref_rgb[pts_mapped[valid, 1], pts_mapped[valid, 0]]
            data["colors"] = colors
        else:
            data["colors"] = (cm.get_cmap('rainbow')(np.linspace(0, 1, num_trajs))[:, :3] * 255).astype(np.uint8)

        rr.log("info/gt_trajs", rr.TextDocument(
            self._build_gt_info_summary(data, meta_data), media_type=rr.MediaType.MARKDOWN,
        ), static=True)
        return data

    def _build_gt_info_summary(self, data, meta_data):
        """Build a rich Markdown summary of the GT trajectory data."""
        trajs_3d = data["trajs_3d"]          # [F, N, 3]
        visibs = data.get("visibs")           # [F, N] or None
        valids = data.get("valids")           # [F, N] or None
        num_frames, num_trajs = trajs_3d.shape[:2]

        lines = [f"## GT Trajectories\n"]
        lines.append(f"- **Frames**: {num_frames}, **Trajs**: {num_trajs}")

        # Coordinate system info
        has_w2c = data.get("w2c_first") is not None
        lines.append(f"- **motion_extrinsics (w2c)**: {'available' if has_w2c else 'MISSING'}")
        lines.append(f"- **prefer_gt**: {self._prefer_gt}")

        # Resolution info
        orig_h, orig_w = MotionVisUtils.get_orig_dims(meta_data)
        if orig_h and orig_w:
            lines.append(f"- **Original resolution**: {orig_w}x{orig_h}")

        # Depth range at reference frame
        ref_pts = trajs_3d[self.REF_FRAME]
        finite_mask = np.isfinite(ref_pts).all(axis=1)
        if finite_mask.any():
            z_vals = ref_pts[finite_mask, 2]
            lines.append(f"- **Depth (Z) range at ref**: [{z_vals.min():.3f}, {z_vals.max():.3f}], "
                         f"mean={z_vals.mean():.3f}")

        # Motion magnitude statistics (ref -> all other frames)
        lines.append("\n### Motion Statistics\n")
        all_disps = []
        for f in range(num_frames):
            if f == self.REF_FRAME:
                continue
            disp = trajs_3d[f] - trajs_3d[self.REF_FRAME]
            mag = np.linalg.norm(disp, axis=1)
            valid_f = np.isfinite(mag)
            if visibs is not None:
                valid_f &= (visibs[self.REF_FRAME] > 0) & (visibs[f] > 0)
            if valids is not None:
                valid_f &= (valids[self.REF_FRAME] > 0) & (valids[f] > 0)
            if valid_f.any():
                all_disps.append(mag[valid_f])

        if all_disps:
            all_mag = np.concatenate(all_disps)
            dyn_mask = all_mag > self._GT_DYNAMIC_THRESHOLD
            n_dyn = dyn_mask.sum()
            lines.append(f"| Stat | Value |")
            lines.append(f"|------|-------|")
            lines.append(f"| Mean displacement | {all_mag.mean():.4f} m |")
            lines.append(f"| Median | {np.median(all_mag):.4f} m |")
            lines.append(f"| Max | {all_mag.max():.4f} m |")
            lines.append(f"| P95 | {np.percentile(all_mag, 95):.4f} m |")
            lines.append(f"| Dynamic (>{self._GT_DYNAMIC_THRESHOLD}m) | "
                         f"{n_dyn}/{len(all_mag)} ({100*n_dyn/len(all_mag):.1f}%) |")

        # Per-frame visibility/validity table
        lines.append("\n### Per-Frame Summary\n")
        lines.append("| Frame | Visible | Valid | Dynamic | Mean Disp |")
        lines.append("|-------|---------|-------|---------|-----------|")
        for f in range(num_frames):
            n_vis = int((visibs[f] > 0).sum()) if visibs is not None else num_trajs
            n_val = int((valids[f] > 0).sum()) if valids is not None else num_trajs
            if f == self.REF_FRAME:
                lines.append(f"| **{f} (ref)** | {n_vis}/{num_trajs} | {n_val}/{num_trajs} | - | - |")
                continue
            disp = trajs_3d[f] - trajs_3d[self.REF_FRAME]
            mag = np.linalg.norm(disp, axis=1)
            pair_valid = np.isfinite(mag)
            if visibs is not None:
                pair_valid &= (visibs[self.REF_FRAME] > 0) & (visibs[f] > 0)
            if valids is not None:
                pair_valid &= (valids[self.REF_FRAME] > 0) & (valids[f] > 0)
            if pair_valid.any():
                vm = mag[pair_valid]
                n_dyn_f = int((vm > self._GT_DYNAMIC_THRESHOLD).sum())
                mean_d = f"{vm.mean():.4f}"
            else:
                n_dyn_f = 0
                mean_d = "N/A"
            lines.append(f"| {f} | {n_vis}/{num_trajs} | {n_val}/{num_trajs} | "
                         f"{n_dyn_f} | {mean_d} |")

        return "\n".join(lines)

    def _build_gt_frame_info(self, tgt_frame, total_trajs, n_after_visval,
                             n_after_bounds, n_after_depth, n_after_outlier,
                             n_final, mags, outlier_threshold, q1, q3, iqr):
        """Build per-frame Markdown info for the GT Info panel."""
        lines = [f"## Frame 0 → {tgt_frame}\n"]
        lines.append("### Filtering Pipeline\n")
        lines.append(f"| Stage | Remaining | Removed |")
        lines.append(f"|-------|-----------|---------|")
        lines.append(f"| Total trajs | {total_trajs} | - |")
        lines.append(f"| After vis+val | {n_after_visval} | {total_trajs - n_after_visval} |")
        lines.append(f"| After bounds | {n_after_bounds} | {n_after_visval - n_after_bounds} |")
        lines.append(f"| After depth Z | {n_after_depth} | {n_after_bounds - n_after_depth} |")
        lines.append(f"| After outlier | {n_after_outlier} | {n_after_depth - n_after_outlier} |")
        lines.append(f"| After subsample | {n_final} | {n_after_outlier - n_final} |")

        if mags is not None and len(mags) > 0:
            dyn_mask = mags > self._GT_DYNAMIC_THRESHOLD
            lines.append(f"\n### Motion (before outlier filter)\n")
            lines.append(f"| Stat | Value |")
            lines.append(f"|------|-------|")
            lines.append(f"| Points | {len(mags)} |")
            lines.append(f"| Dynamic (>{self._GT_DYNAMIC_THRESHOLD}m) | "
                         f"{dyn_mask.sum()}/{len(mags)} ({100*dyn_mask.sum()/max(len(mags),1):.1f}%) |")
            lines.append(f"| Mean disp | {mags.mean():.4f} m |")
            lines.append(f"| Max disp | {mags.max():.4f} m |")
            lines.append(f"| Q1 / Q3 / IQR | {q1:.4f} / {q3:.4f} / {iqr:.4f} |")
            lines.append(f"| Outlier threshold | {outlier_threshold:.4f} m "
                         f"(floor={self._GT_OUTLIER_MIN_THRESHOLD:.1f}) |")

        return "\n".join(lines)

    def _visualize_gt_trajectory(self, mv_outputs, trajs_data, ref_frame, tgt_frame, meta_data):
        ref_out = mv_outputs[0]
        H, W = ref_out.pointmap_h, ref_out.pointmap_w

        total_trajs = trajs_data["trajs_3d"].shape[1]
        orig_h, orig_w = MotionVisUtils.get_orig_dims(meta_data, H, W)
        ref_2d_scaled = trajs_data["trajs_2d"][ref_frame].copy() * [W / orig_w, H / orig_h]

        # Stage 1: visibility + validity
        mask = np.ones(len(ref_2d_scaled), dtype=bool)
        if trajs_data["visibs"] is not None:
            mask &= (trajs_data["visibs"][ref_frame] > 0) & (trajs_data["visibs"][tgt_frame] > 0)
        if trajs_data["valids"] is not None:
            mask &= (trajs_data["valids"][ref_frame] > 0) & (trajs_data["valids"][tgt_frame] > 0)
        n_after_visval = int(mask.sum())

        # Stage 2: 2D bounds
        mask &= np.isfinite(ref_2d_scaled).all(axis=1) & (ref_2d_scaled[:, 0] >= 0) & (ref_2d_scaled[:, 0] < W - 1) & \
                (ref_2d_scaled[:, 1] >= 0) & (ref_2d_scaled[:, 1] < H - 1)
        n_after_bounds = int(mask.sum())

        orig_idx = np.where(mask)[0]
        if len(orig_idx) == 0:
            rr.log("info/gt_trajs", rr.TextDocument(
                self._build_gt_frame_info(tgt_frame, total_trajs, n_after_visval,
                                          n_after_bounds, 0, 0, 0, None,
                                          float('inf'), 0, 0, 0),
                media_type=rr.MediaType.MARKDOWN))
            return

        c2w_vis = self._get_c2w(ref_out)

        true_src_raw = trajs_data["trajs_3d"][ref_frame][orig_idx]
        gt_disp = trajs_data["trajs_3d"][tgt_frame][orig_idx] - true_src_raw

        if trajs_data["w2c_first"] is not None:
            pts_homo = np.concatenate([true_src_raw, np.ones((len(true_src_raw), 1))], axis=1)
            pts_cam = (trajs_data["w2c_first"] @ pts_homo.T).T
            src_world = (c2w_vis @ pts_cam.T).T[:, :3]

            gt_motion_world = ((c2w_vis[:3, :3] @ trajs_data["w2c_first"][:3, :3]) @ gt_disp.T).T
        else:
            src_world = true_src_raw
            gt_motion_world = gt_disp

        src_colors = trajs_data["colors"][orig_idx]

        gt_endpoints = src_world + gt_motion_world

        # Stage 3: depth / finite filter
        valid = self._depth_valid(src_world) & self._depth_valid(gt_endpoints)
        n_after_depth = int(valid.sum())

        # Stage 4: outlier filter (IQR-based with floor)
        # Detect erroneous jumps (e.g. Kubric4D visibs=True at occluded frames)
        # via IQR, but enforce a minimum threshold so that legitimate dynamic
        # points are never filtered in static-dominated scenes.
        mags = np.linalg.norm(gt_motion_world, axis=1)
        mags_before_outlier = mags[valid].copy() if valid.any() else np.array([])
        n_outliers = 0
        outlier_threshold = float('inf')
        q1, q3, iqr = 0.0, 0.0, 0.0
        if valid.sum() >= self._GT_OUTLIER_MIN_SAMPLES:
            valid_mags = mags[valid]
            q1, q3 = np.percentile(valid_mags, [25, 75])
            iqr = q3 - q1
            iqr_threshold = q3 + self._GT_OUTLIER_IQR_FACTOR * iqr
            outlier_threshold = max(iqr_threshold, self._GT_OUTLIER_MIN_THRESHOLD)
            inlier = mags <= outlier_threshold
            n_outliers = valid.sum() - (valid & inlier).sum()
            if n_outliers > 0:
                logger.info(
                    "[_visualize_gt_trajectory] tgt=%d: filtered %d/%d GT "
                    "trajectory outliers (threshold=%.4f [iqr=%.4f, floor=%.1f], "
                    "Q1=%.4f, Q3=%.4f, IQR=%.4f)",
                    tgt_frame, n_outliers, valid.sum(),
                    outlier_threshold, iqr_threshold, self._GT_OUTLIER_MIN_THRESHOLD,
                    q1, q3, iqr,
                )
            valid &= inlier
        n_after_outlier = int(valid.sum())

        if not valid.all():
            src_world = src_world[valid]
            gt_motion_world = gt_motion_world[valid]
            gt_endpoints = gt_endpoints[valid]
            src_colors = src_colors[valid]
        if len(src_world) == 0:
            rr.log("info/gt_trajs", rr.TextDocument(
                self._build_gt_frame_info(tgt_frame, total_trajs, n_after_visval,
                                          n_after_bounds, n_after_depth, n_after_outlier,
                                          0, mags_before_outlier, outlier_threshold,
                                          q1, q3, iqr),
                media_type=rr.MediaType.MARKDOWN))
            return

        n_before_subsample = len(src_world)
        if len(src_world) > self.MAX_POINTS_PER_VIEW:
            sub_mags = np.linalg.norm(gt_motion_world, axis=1)
            probs = (0.2 + 0.8 * (sub_mags / (sub_mags.max() + 1e-8)))
            idx = np.random.choice(len(src_world), self.MAX_POINTS_PER_VIEW, replace=False, p=probs / probs.sum())
            src_world, gt_motion_world, src_colors, gt_endpoints = \
                src_world[idx], gt_motion_world[idx], src_colors[idx], gt_endpoints[idx]
        n_final = len(src_world)

        # Log per-frame filtering info to GT Info panel
        rr.log("info/gt_trajs", rr.TextDocument(
            self._build_gt_frame_info(tgt_frame, total_trajs, n_after_visval,
                                      n_after_bounds, n_after_depth, n_after_outlier,
                                      n_final, mags_before_outlier, outlier_threshold,
                                      q1, q3, iqr),
            media_type=rr.MediaType.MARKDOWN))

        src_w = self._rerun_coord_apply_pts(src_world)
        gt_e = self._rerun_coord_apply_pts(gt_endpoints)
        gt_m = self._rerun_coord_apply_vecs(gt_motion_world)
        motion_colors = (MotionVisUtils.compute_motion_colors(gt_m) * 255).astype(np.uint8)

        rr.log("world_gt_trajs/gt_trajectory_reference_points", rr.Points3D(src_w, colors=src_colors, radii=self.POINT_RADIUS * 2.0))
        rr.log("world_gt_trajs/gt_trajectory_target_points", rr.Points3D(gt_e, colors=src_colors, radii=self.POINT_RADIUS * 2.0))
        rr.log("world_gt_trajs/gt_trajectories", rr.LineStrips3D(np.stack([src_w, gt_e], axis=1), colors=motion_colors, radii=0.012))
        rr.log("world_gt_trajs/gt_flow_arrows", rr.Arrows3D(origins=src_w, vectors=gt_m, colors=motion_colors))

    def _extract_target_frame_data(self, mv_outputs, ref_frame, tgt_frame, view_num):
        """Extract Target PC and RGB (Shared between Dense and Sparse)."""
        tgt_pts_all, tgt_colors_all, tgt_rgb_img = [], [], None
        for v_idx in range(view_num):
            tgt_out = mv_outputs[tgt_frame * view_num + v_idx]

            if v_idx == 0 and getattr(tgt_out, 'rgb', None) is not None:
                tgt_rgb_img = MotionVisUtils.normalize_rgb(tgt_out.rgb)

            tgt_pmap, is_world = self._get_pointmap_world(tgt_out)
            if tgt_pmap is not None:
                H_pmap, W_pmap = self._get_pmap_hw(tgt_out, prefer_gt=True)
                pts = self._reshape_pmap(MotionVisUtils.to_numpy(tgt_pmap), H_pmap, W_pmap)
                pts_sub = pts[::self.SAMPLE_STEP, ::self.SAMPLE_STEP].reshape(-1, 3)
                valid = self._depth_valid(pts_sub)
                if valid.any():
                    c2w = np.eye(4) if is_world else self._get_c2w(tgt_out)
                    pts_w = MotionVisUtils.apply_c2w(c2w, pts_sub[valid])
                    if getattr(tgt_out, 'rgb', None) is not None:
                        tc = cv2.resize(MotionVisUtils.normalize_rgb(tgt_out.rgb), (W_pmap, H_pmap))[::self.SAMPLE_STEP, ::self.SAMPLE_STEP].reshape(-1, 3)[valid]
                    else:
                        tc = np.full((len(pts_w), 3), [0, 200, 200])
                    tgt_pts_all.append(pts_w)
                    tgt_colors_all.append(tc.astype(np.uint8))
        return tgt_pts_all, tgt_colors_all, tgt_rgb_img

    def _process_target_frame(self, mv_outputs, meta_data, ref_data, ref_frame, tgt_frame, view_num, flow_img_dir, pred_history=None):
        """Process a target frame and return predicted positions for history tracking.

        Returns:
            dict with 'positions' (N, 3) and 'colors' (N, 3) for trail history, or None.
        """
        if pred_history is None:
            pred_history = {}

        tgt_pts_all, tgt_colors_all, tgt_rgb_img = self._extract_target_frame_data(mv_outputs, ref_frame, tgt_frame, view_num)
        pred_tgt_pts_all, pred_tgt_colors_all = [], []
        all_colors, all_ends, all_orgs, all_vecs = [], [], [], []
        flow_mag_img, flow_xyz_img = None, None

        for v_idx in range(view_num):
            ref_idx, tgt_idx = ref_frame * view_num + v_idx, tgt_frame * view_num + v_idx
            if ref_idx >= len(mv_outputs) or tgt_idx >= len(mv_outputs):
                continue
            ref_out = mv_outputs[ref_idx]

            if v_idx == 0 and getattr(ref_out, 'scene_flow_pred', None) is not None:
                sf = MotionVisUtils.to_numpy(ref_out.scene_flow_pred)
                if tgt_frame < sf.shape[0]:
                    flow_to_tgt = sf[tgt_frame]
                    mag = np.linalg.norm(flow_to_tgt, axis=0)
                    flow_mag_img = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else np.zeros_like(mag, dtype=np.uint8)
                    flow_xyz_img = np.full_like(flow_to_tgt.transpose(1, 2, 0), 128, dtype=np.uint8)
                    for c in range(3):
                        fc = flow_to_tgt[c]
                        if fc.max() > fc.min():
                            flow_xyz_img[..., c] = ((fc - fc.min()) / (fc.max() - fc.min()) * 255)

            if getattr(ref_out, 'scene_flow_pred', None) is None or v_idx not in ref_data["valid_masks"]:
                continue
            sf = MotionVisUtils.to_numpy(ref_out.scene_flow_pred)
            if tgt_frame >= sf.shape[0]:
                continue

            valid_mask = ref_data["valid_masks"][v_idx]
            _H_pm, _W_pm = ref_data["pmap_dims"].get(v_idx, (ref_out.pointmap_h, ref_out.pointmap_w))

            # Resize scene_flow to match pointmap_gt dimensions if they differ
            sf_frame = sf[tgt_frame]  # (3, H_sf, W_sf)
            H_sf, W_sf = sf_frame.shape[1], sf_frame.shape[2]
            if H_sf != _H_pm or W_sf != _W_pm:
                sf_frame = cv2.resize(sf_frame.transpose(1, 2, 0), (_W_pm, _H_pm)).transpose(2, 0, 1)

            flow_valid = sf_frame.transpose(1, 2, 0)[::self.SAMPLE_STEP, ::self.SAMPLE_STEP].reshape(-1, 3)[valid_mask]
            ref_pts_valid = ref_data["pts_local"][v_idx][valid_mask]
            c2w = ref_data["c2w"].get(v_idx)

            pred_w = MotionVisUtils.apply_c2w(c2w, ref_pts_valid + flow_valid)

            # Sample RGB colors - resize RGB to pointmap_gt dimensions to ensure alignment
            if getattr(ref_out, 'rgb', None) is not None:
                pc = cv2.resize(MotionVisUtils.normalize_rgb(ref_out.rgb), (_W_pm, _H_pm))[::self.SAMPLE_STEP, ::self.SAMPLE_STEP].reshape(-1, 3)[valid_mask].astype(np.uint8)
            else:
                pc = np.full((len(pred_w), 3), [100, 255, 255], dtype=np.uint8)

            pred_tgt_pts_all.append(pred_w)
            pred_tgt_colors_all.append(pc)

            ref_w = MotionVisUtils.apply_c2w(c2w, ref_pts_valid)
            flow_w = (c2w[:3, :3] @ flow_valid.T).T if c2w is not None else flow_valid
            ends = ref_w + flow_w

            # Subsample once after concat (not per-view random) for consistent trail matching
            all_colors.append((MotionVisUtils.compute_motion_colors(flow_w) * 255).astype(np.uint8))
            all_ends.append(ends)
            all_orgs.append(ref_w)
            all_vecs.append(flow_w)

        if tgt_pts_all:
            t_pts, t_cols = np.concatenate(tgt_pts_all), np.concatenate(tgt_colors_all)
            t_pts_r = self._rerun_coord_apply_pts(t_pts)
            rr.log("world/target_pointcloud", rr.Points3D(t_pts_r, colors=t_cols, radii=self.POINT_RADIUS))
            rr.log("world_gt_trajs/target_pointcloud", rr.Points3D(t_pts_r, colors=t_cols, radii=self.POINT_RADIUS * 0.8))

        if pred_tgt_pts_all:
            pp, pc = np.concatenate(pred_tgt_pts_all), np.concatenate(pred_tgt_colors_all)
            pm = self._depth_valid(pp)
            if pm.any():
                rr.log(
                    "world_pred/predicted_target_pointcloud",
                    rr.Points3D(self._rerun_coord_apply_pts(pp[pm]), colors=pc[pm], radii=self.POINT_RADIUS),
                )

        frame_result = None
        if all_ends:
            cat_ends_full = np.concatenate(all_ends)
            cat_orgs_full = np.concatenate(all_orgs)
            cat_vecs_full = np.concatenate(all_vecs)
            c_cols_full = np.concatenate(all_colors)
            sm_full = self._depth_valid(cat_orgs_full) & self._depth_valid(cat_ends_full)
            if sm_full.any():
                frame_result = {
                    'positions': self._rerun_coord_apply_pts(cat_ends_full[sm_full]),
                    'colors': c_cols_full[sm_full],
                    'origins': self._rerun_coord_apply_pts(cat_orgs_full[sm_full]),
                    'vectors': self._rerun_coord_apply_vecs(cat_vecs_full[sm_full]),
                }
                cat_ends = cat_ends_full[sm_full]
                cat_orgs = cat_orgs_full[sm_full]
                cat_vecs = cat_vecs_full[sm_full]
                c_cols = c_cols_full[sm_full]
                if len(cat_ends) > self.MAX_POINTS_PER_VIEW:
                    rng = np.random.default_rng(seed=42)
                    idx = rng.choice(len(cat_ends), self.MAX_POINTS_PER_VIEW, replace=False)
                    cat_ends = cat_ends[idx]
                    cat_orgs = cat_orgs[idx]
                    cat_vecs = cat_vecs[idx]
                    c_cols = c_cols[idx]
                cat_streaks = np.stack([cat_orgs, cat_ends], axis=1)
                cat_streaks_r = self._rerun_coord_apply_pts(cat_streaks)
                cat_ends_r = self._rerun_coord_apply_pts(cat_ends)
                cat_orgs_r = self._rerun_coord_apply_pts(cat_orgs)
                cat_vecs_r = self._rerun_coord_apply_vecs(cat_vecs)
                # GT Target + Pred Flow: ref→tgt streaks only (not duplicated in world_pred)
                rr.log("world/trajectories/streaks", rr.LineStrips3D(cat_streaks_r, colors=c_cols, radii=0.005))
                rr.log("world/trajectories/endpoints", rr.Points3D(cat_ends_r, colors=c_cols, radii=self.POINT_RADIUS * 0.8))
                rr.log("world/trajectories/arrows", rr.Arrows3D(origins=cat_orgs_r, vectors=cat_vecs_r, colors=c_cols))

        self._log_fading_trails(tgt_frame, frame_result, pred_history)

        if flow_mag_img is not None:
            rr.log("flow_magnitude", rr.Image(flow_mag_img))
            cv2.imwrite(os.path.join(flow_img_dir, f"magnitude_f{ref_frame:03d}_to_f{tgt_frame:03d}.png"), flow_mag_img)
        if flow_xyz_img is not None:
            rr.log("flow_xyz", rr.Image(flow_xyz_img))
            cv2.imwrite(os.path.join(flow_img_dir, f"flow_xyz_f{ref_frame:03d}_to_f{tgt_frame:03d}.png"), cv2.cvtColor(flow_xyz_img, cv2.COLOR_RGB2BGR))
        if tgt_rgb_img is not None:
            rr.log("target_rgb", rr.Image(tgt_rgb_img))
            cv2.imwrite(os.path.join(flow_img_dir, f"rgb_f{tgt_frame:03d}.png"), cv2.cvtColor(tgt_rgb_img, cv2.COLOR_RGB2BGR))

        return frame_result

    def _log_fading_trails(self, tgt_frame, current_data, pred_history):
        """Log fading multi-frame trails under world_pred only.

        Matches points across frames via quantized reference-frame origins so each
        physical point keeps one color when TRAIL_RAINBOW_COLORS is True.
        """
        if current_data is None:
            return

        cur_pos = current_data['positions']
        cur_colors = current_data['colors']
        cur_origins = current_data['origins']
        cur_vectors = current_data['vectors']
        n_points = len(cur_pos)
        if n_points == 0:
            return

        hist_frames = sorted([f for f in pred_history.keys() if f < tgt_frame])
        hist_frames = hist_frames[-(self.TRAIL_LENGTH - 1):]

        max_trail_points = min(self.MAX_POINTS_PER_VIEW, n_points)
        if n_points > max_trail_points:
            rng = np.random.default_rng(seed=42)
            trail_idx = rng.choice(n_points, max_trail_points, replace=False)
        else:
            trail_idx = np.arange(n_points)

        cur_pos_sub = cur_pos[trail_idx]
        cur_colors_sub = cur_colors[trail_idx]
        cur_origins_sub = cur_origins[trail_idx]
        cur_vectors_sub = cur_vectors[trail_idx]

        def origin_key(origin, precision=4):
            return tuple(np.round(origin, precision))

        if self.TRAIL_RAINBOW_COLORS:
            from matplotlib.colors import hsv_to_rgb
            o_min = cur_origins_sub.min(axis=0)
            o_max = cur_origins_sub.max(axis=0)
            o_rng = np.maximum(o_max - o_min, 1e-6)
            o_norm = (cur_origins_sub - o_min) / o_rng
            hue = (o_norm[:, 0] * 0.5 + o_norm[:, 1] * 0.3 + o_norm[:, 2] * 0.2) % 1.0
            hsv = np.stack([hue, np.full(len(hue), 0.95), np.full(len(hue), 1.0)], axis=1)
            rainbow_colors = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
        else:
            rainbow_colors = cur_colors_sub

        cur_origin_map = {}
        for orig, pos, col in zip(cur_origins_sub, cur_pos_sub, rainbow_colors):
            cur_origin_map[origin_key(orig)] = {'pos': pos, 'color': col}

        hist_origin_maps = {}
        for f in hist_frames:
            if f not in pred_history or pred_history[f] is None:
                continue
            hd = pred_history[f]
            if 'origins' not in hd or 'positions' not in hd:
                continue
            f_map = {}
            for o, p in zip(hd['origins'], hd['positions']):
                f_map[origin_key(o)] = p
            hist_origin_maps[f] = f_map

        common_origins = set(cur_origin_map.keys())
        for f in hist_frames:
            if f in hist_origin_maps:
                common_origins &= set(hist_origin_maps[f].keys())

        all_segments, all_seg_colors, all_seg_radii = [], [], []
        if common_origins and hist_frames:
            common_origins = list(common_origins)
            frame_sequence = hist_frames + [tgt_frame]
            n_segments = len(frame_sequence) - 1
            for seg_idx in range(n_segments):
                f_start = frame_sequence[seg_idx]
                f_end = frame_sequence[seg_idx + 1]
                if f_start not in hist_origin_maps:
                    continue
                seg_starts, seg_ends, seg_colors = [], [], []
                for ok in common_origins:
                    if ok not in hist_origin_maps[f_start]:
                        continue
                    pos_start = hist_origin_maps[f_start][ok]
                    if f_end == tgt_frame:
                        if ok not in cur_origin_map:
                            continue
                        pos_end = cur_origin_map[ok]['pos']
                        color = cur_origin_map[ok]['color']
                    elif f_end in hist_origin_maps and ok in hist_origin_maps[f_end]:
                        pos_end = hist_origin_maps[f_end][ok]
                        color = cur_origin_map[ok]['color']
                    else:
                        continue
                    seg_starts.append(pos_start)
                    seg_ends.append(pos_end)
                    seg_colors.append(color)
                if not seg_starts:
                    continue
                seg_starts = np.asarray(seg_starts, dtype=np.float64)
                seg_ends = np.asarray(seg_ends, dtype=np.float64)
                seg_colors = np.asarray(seg_colors, dtype=np.uint8)
                fade = (seg_idx + 1) / n_segments
                alpha = self.TRAIL_MIN_ALPHA + fade * (self.TRAIL_MAX_ALPHA - self.TRAIL_MIN_ALPHA)
                radius = self.TRAIL_MIN_RADIUS + fade * (self.TRAIL_MAX_RADIUS - self.TRAIL_MIN_RADIUS)
                segments = np.stack([seg_starts, seg_ends], axis=1)
                faded = (seg_colors.astype(np.float32) * alpha).astype(np.uint8)
                all_segments.append(segments)
                all_seg_colors.append(faded)
                all_seg_radii.extend([radius] * len(segments))

        if all_segments:
            cat_seg = np.concatenate(all_segments, axis=0)
            cat_col = np.concatenate(all_seg_colors, axis=0)
            rr.log("world_pred/trajectories/trails",
                   rr.LineStrips3D(cat_seg, colors=cat_col, radii=all_seg_radii))

        rr.log("world_pred/trajectories/endpoints",
               rr.Points3D(cur_pos_sub, colors=rainbow_colors, radii=self.POINT_RADIUS * 0.6))

        prev_frame = max(hist_frames) if hist_frames else None
        if prev_frame is not None and prev_frame in hist_origin_maps:
            ao, av, ac = [], [], []
            for ok in cur_origin_map:
                if ok in hist_origin_maps[prev_frame]:
                    p0 = hist_origin_maps[prev_frame][ok]
                    p1 = cur_origin_map[ok]['pos']
                    ao.append(p0)
                    av.append(p1 - p0)
                    ac.append(cur_origin_map[ok]['color'])
            if ao:
                rr.log("world_pred/trajectories/arrows",
                       rr.Arrows3D(origins=np.asarray(ao), vectors=np.asarray(av),
                                   colors=np.asarray(ac, dtype=np.uint8)))
        else:
            ac = (rainbow_colors.astype(np.float32) * 0.7).astype(np.uint8)
            rr.log("world_pred/trajectories/arrows",
                   rr.Arrows3D(origins=cur_origins_sub, vectors=cur_vectors_sub, colors=ac))


class SparseMotionVisualizer3D(MotionVisualizer3D):
    """Sparse motion Rerun: two main panels (GT target + flows); optional debug warp3d panels via config."""

    RR_NAME_PREFIX = "SparseMotion"
    RRD_FILENAME_PREFIX = "vis_sparse_3d"
    GENERATE_FLOW_GIF = False
    # Third / fourth Rerun panels (win3 / win4): predicted camera frustums only.
    _SPARSE_PRED_CAMERA_ONLY_PANELS = frozenset(("world_warp3d_dynamic", "world_warp3d_delta_sf"))

    @property
    def _sparse_rerun_enable_prediction(self):
        """When True, Rerun also shows prediction panels: ``warp3d`` / ``warp3d_delta``.

        Config (in ``rerun_vis_cfg``): ``enable_prediction`` (bool, default False).
        Legacy key ``vis_3d_sparse_rerun_enable_prediction`` is still accepted if ``enable_prediction`` is absent.
        """
        if "enable_prediction" in self.cfg:
            return bool(self.cfg.get("enable_prediction"))
        return bool(self.cfg.get("vis_3d_sparse_rerun_enable_prediction", False))

    @property
    def _sparse_rerun_camera_panel_prefixes(self):
        """Spatial3D panel entity roots that receive GT / pred camera frustums."""
        px = ["win1", "win2"]
        if self._sparse_rerun_enable_prediction:
            px.extend(["world_warp3d_dynamic", "world_warp3d_delta_sf"])
        return tuple(px)

    def _setup_blueprint(self):
        if rrb is None:
            return
        win1 = rrb.Vertical(
            rrb.Spatial3DView(
                name="GT Target + Pred Flow (warp3d − src_3d_gt)",
                origin="win1/",
                **_rerun_line_grid_hidden_kw(),
            ),
        )
        win2 = rrb.Vertical(
            rrb.Spatial3DView(
                name="GT Target + Pred Scene flow (warp3d_delta)",
                origin="win2/",
                **_rerun_line_grid_hidden_kw(),
            ),
        )
        if self._sparse_rerun_enable_prediction:
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    win1,
                    win2,
                    rrb.Vertical(
                        rrb.Spatial3DView(name="Dynamic points (warp3d)", origin="world_warp3d_dynamic/", **_rerun_line_grid_hidden_kw()),
                    ),
                    rrb.Vertical(
                        rrb.Spatial3DView(name="Scene flow (warp3d_delta)", origin="world_warp3d_delta_sf/", **_rerun_line_grid_hidden_kw()),
                    ),
                    column_shares=[1, 1, 1, 1],
                ),
                collapse_panels=True,
            )
        else:
            blueprint = rrb.Blueprint(
                rrb.Horizontal(win1, win2, column_shares=[1, 1]),
                collapse_panels=True,
            )
        rr.send_blueprint(blueprint)

    def run(self, mv_outputs, out_dir, data_idx, meta_data=None):
        """Sparse Rerun: two main 3D panels; optional debug panels for warp3d / warp3d_delta."""
        if rr is None:
            return

        rr.init(f"{self.RR_NAME_PREFIX}_{data_idx}", spawn=False)
        save_path = os.path.join(out_dir, "rerun_vis", f"{self.RRD_FILENAME_PREFIX}_{data_idx:06d}.rrd")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._log_view_coordinates()
        self._setup_blueprint()

        frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]
        logger.info(
            "[Rerun3D] sparse data_idx=%d, frames=%d, views=%d, outputs=%d, prefer_gt=%s, "
            "enable_prediction_panels=%s",
            data_idx,
            frame_num,
            view_num,
            len(mv_outputs),
            self._prefer_gt,
            self._sparse_rerun_enable_prediction,
        )
        logger.info(
            "[SparseMotionVis] Rerun: win1/win2 origin=src_3d_gt; "
            "win3/win4 origin=src_points (pred)"
        )

        try:
            ref_data = self._build_reference_pointcloud(mv_outputs, self.REF_FRAME, view_num)
        except Exception as e:
            logger.error("[Rerun3D] _build_reference_pointcloud failed: %s\n%s", e, traceback.format_exc())
            return

        if not ref_data["pts"]:
            logger.warning(
                "[Rerun3D] reference pointcloud is empty — skipping .rrd save. "
                "Check depth_mask / pointmap validity."
            )
            return

        flow_img_dir = os.path.join(os.path.dirname(save_path), f"flow_images_{data_idx:06d}")
        os.makedirs(flow_img_dir, exist_ok=True)

        ref_out = mv_outputs[self.REF_FRAME * view_num]
        self._log_frustums(
            ref_out,
            "ref",
            self.REF_FRAME,
            view_num=view_num,
            mv_outputs=mv_outputs,
            static=True,
            panel_prefixes=self._sparse_rerun_camera_panel_prefixes,
            pred_camera_only_prefixes=self._SPARSE_PRED_CAMERA_ONLY_PANELS,
        )

        self._log_win12_frame0_gt_pointmap(mv_outputs, ref_data, view_num)

        if self._sparse_rerun_enable_prediction:
            self._rerun_acc_warp3d_pts = []
            self._rerun_acc_warp3d_cols = []
            if rr is not None:
                rr.log("world_warp3d_dynamic/init_to_dyn_flow", rr.Clear(recursive=True))
            self._rerun_log_sparse_frame0_init(mv_outputs, meta_data, ref_data, frame_num, view_num)

        pred_history = {}
        for tgt_frame in range(frame_num):
            rr.set_time_sequence("target_frame", tgt_frame)
            try:
                tgt_out = mv_outputs[tgt_frame * view_num]
                self._log_frustums(
                    tgt_out,
                    "tgt",
                    tgt_frame,
                    view_num=view_num,
                    mv_outputs=mv_outputs,
                    static=False,
                    panel_prefixes=self._sparse_rerun_camera_panel_prefixes,
                    pred_camera_only_prefixes=self._SPARSE_PRED_CAMERA_ONLY_PANELS,
                )
                self._process_target_frame(
                    mv_outputs,
                    meta_data,
                    ref_data,
                    self.REF_FRAME,
                    tgt_frame,
                    view_num,
                    flow_img_dir,
                    pred_history,
                )
            except Exception as e:
                logger.error("[Rerun3D] tgt_frame=%d failed: %s\n%s", tgt_frame, e, traceback.format_exc())
                continue

        rr.save(save_path)
        if self.GENERATE_FLOW_GIF:
            GifGenerator.generate_flow_gif(flow_img_dir, fps=2.0, size=280)

    def _log_win12_frame0_gt_pointmap(self, mv_outputs, ref_data, view_num):
        """Frame-0 ``pointmap_gt`` in win1 / win2: grid subsample ``SAMPLE_STEP`` only (static, world)."""
        if rr is None or not mv_outputs:
            return
        v_idx = 0
        flat = self.REF_FRAME * view_num + v_idx
        if flat >= len(mv_outputs):
            return
        out_f0 = mv_outputs[flat]
        if not self._track3d_in_world_space(out_f0) and ref_data.get("c2w", {}).get(v_idx) is None:
            return
        pmap = getattr(out_f0, "pointmap_gt", None)
        if pmap is None:
            rr.log("win1/frame0_pointmap", rr.Clear(recursive=True))
            rr.log("win2/frame0_pointmap", rr.Clear(recursive=True))
            return
        ph, pw = self._get_pmap_hw(out_f0, prefer_gt=True)
        pmap3 = self._reshape_pmap(MotionVisUtils.to_numpy(pmap), ph, pw)
        if pmap3 is None or pmap3.ndim != 3:
            rr.log("win1/frame0_pointmap", rr.Clear(recursive=True))
            rr.log("win2/frame0_pointmap", rr.Clear(recursive=True))
            return
        pts_local = pmap3[:: self.SAMPLE_STEP, :: self.SAMPLE_STEP].reshape(-1, 3)
        vm = self._depth_valid(pts_local)
        if not vm.any():
            rr.log("win1/frame0_pointmap", rr.Clear(recursive=True))
            rr.log("win2/frame0_pointmap", rr.Clear(recursive=True))
            return
        pts_w = self._world_pts_from_track(out_f0, ref_data, v_idx, pts_local[vm])
        H, W = pmap3.shape[0], pmap3.shape[1]
        if getattr(out_f0, "rgb", None) is not None:
            rgb_rs = cv2.resize(MotionVisUtils.normalize_rgb(out_f0.rgb), (W, H))
            cols = rgb_rs[:: self.SAMPLE_STEP, :: self.SAMPLE_STEP].reshape(-1, 3)[vm].astype(np.uint8)
        else:
            cols = np.full((pts_w.shape[0], 3), 160, dtype=np.uint8)
        rd = self.POINT_RADIUS * 0.55
        pts_w_r = self._rerun_coord_apply_pts(pts_w)
        pc = rr.Points3D(pts_w_r, colors=cols, radii=rd)
        rr.log("win1/frame0_pointmap", pc, static=True)
        rr.log("win2/frame0_pointmap", pc, static=True)

    def _log_static_query_overlay(self, mv_outputs, meta_data):
        """Sparse four-panel layout: no reference RGB overlay."""
        return

    def _rerun_log_sparse_frame0_init(self, mv_outputs, meta_data, ref_data, frame_num, view_num):
        """Log static frame-0 query positions at predicted ``src_points`` (win3 / win4 only)."""
        if rr is None or not mv_outputs:
            return
        ref_track = mv_outputs[self.REF_FRAME * view_num]
        trd = getattr(ref_track, "track_3d", None)
        if not trd:
            return
        tr = trd[sorted(trd.keys(), key=lambda k: int(k))[0]]
        query_uv = getattr(tr, "query_uv", None)
        quv_np = MotionVisUtils.to_numpy(query_uv)
        if quv_np is None:
            return
        n = int(np.asarray(quv_np, dtype=np.float64).reshape(-1, 2).shape[0])
        if n <= 0:
            return
        ref_out = mv_outputs[self.REF_FRAME * view_num]
        org_w = self._resolve_track_origin_world_pred(ref_out, ref_data, 0, tr, n)
        if org_w is None or org_w.shape[0] < n:
            logger.info("[SparseMotionVis] frame0 init Rerun skipped (no src_points for track origin)")
            return
        rgb_row = self._rgb_from_query_uv(ref_out, query_uv, n)
        vm = self._depth_valid(org_w)
        if not vm.any():
            return
        pts, col = org_w[vm], rgb_row[vm]
        pts_r = self._rerun_coord_apply_pts(pts)
        rd = self.POINT_RADIUS * 0.9
        rr.log("world_warp3d_dynamic/init_frame0", rr.Points3D(pts_r, colors=col, radii=rd), static=True)
        rr.log("world_warp3d_delta_sf/init_frame0", rr.Points3D(pts_r, colors=col, radii=rd), static=True)
        logger.info("[SparseMotionVis] frame0 init Rerun: %d pts at src_points (pred)", len(pts))

    def _extract_gt_trajectories(self, mv_outputs, meta_data):
        data = super()._extract_gt_trajectories(mv_outputs, meta_data)
        if data["available"] and data["w2c_first"] is None:
            logger.warning(
                "[SparseMotionVis] motion_extrinsics missing! "
                "GT trajectory positions may be incorrect. "
                "Falling back to identity (no coordinate transform)."
            )
        return data

    def _track3d_in_world_space(self, ref_out):
        """``track_3d`` tensors follow ``pointmap_gt_global`` when present (already world)."""
        return getattr(ref_out, "pointmap_gt_global", None) is not None

    def _world_pts_from_track(self, ref_out, ref_data, v_idx, pts):
        if pts is None or len(pts) == 0:
            return pts
        if self._track3d_in_world_space(ref_out):
            return np.asarray(pts, dtype=np.float64)
        c2w = ref_data["c2w"].get(v_idx)
        return MotionVisUtils.apply_c2w(c2w, np.asarray(pts, dtype=np.float64))

    def _world_vecs_from_track(self, ref_out, ref_data, v_idx, vecs):
        if vecs is None or len(vecs) == 0:
            return vecs
        if self._track3d_in_world_space(ref_out):
            return np.asarray(vecs, dtype=np.float64)
        c2w = ref_data["c2w"].get(v_idx)
        if c2w is None:
            return np.asarray(vecs, dtype=np.float64)
        v = np.asarray(vecs, dtype=np.float64)
        return (c2w[:3, :3] @ v.T).T

    @staticmethod
    def _track3d_points_array(field, n):
        """``(n, 3)`` float64 slice or ``None`` if missing / too short."""
        if field is None:
            return None
        arr = MotionVisUtils.to_numpy(field)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] < n:
            return None
        return arr[:n]

    def _get_flow_delta_for_vis(self, tr, n):
        """Predicted ``warp3d_delta`` for Rerun win2 / warp3d_delta panels."""
        return self._track3d_points_array(getattr(tr, "warp3d_delta", None), n)

    def _get_track_src_pred(self, tr, n):
        """Predicted reference-frame 3D positions (``src_points`` from pred pointmap)."""
        return self._track3d_points_array(getattr(tr, "src_points", None), n)

    def _track3d_has_vis_delta(self, tr, w3, d_pred):
        """Whether this pair has predicted warp3d / warp3d_delta for arrows."""
        return d_pred is not None or w3 is not None

    def _resolve_track_origin_world(self, ref_out, ref_data, v_idx, src, n):
        """World-space arrow origins from GT ``src_3d_gt`` (win1 / win2)."""
        if src is not None and len(src) >= n:
            return self._world_pts_from_track(ref_out, ref_data, v_idx, np.asarray(src[:n], dtype=np.float64))
        return None

    def _resolve_track_origin_world_pred(self, ref_out, ref_data, v_idx, tr, n):
        """World-space arrow origins from predicted ``src_points`` (win3 / win4)."""
        src_pred = self._get_track_src_pred(tr, n)
        if src_pred is None:
            return None
        return self._world_pts_from_track(ref_out, ref_data, v_idx, src_pred)

    def _rgb_from_query_uv(self, ref_out, query_uv, n):
        """Sample reference-view RGB at normalised ``query_uv``; returns (n, 3) uint8."""
        if n <= 0:
            return np.zeros((0, 3), dtype=np.uint8)
        if query_uv is None or getattr(ref_out, "rgb", None) is None:
            return np.tile(np.array([[200, 200, 200]], dtype=np.uint8), (n, 1))
        quv = np.asarray(MotionVisUtils.to_numpy(query_uv), dtype=np.float64).reshape(-1, 2)[:n]
        rgb_raw = MotionVisUtils.normalize_rgb(ref_out.rgb)
        h_rgb, w_rgb = rgb_raw.shape[:2]
        rgb_px = np.clip(
            quv * [w_rgb - 1, h_rgb - 1],
            [0, 0],
            [w_rgb - 1, h_rgb - 1],
        ).astype(np.int64)
        return rgb_raw[rgb_px[:, 1], rgb_px[:, 0]].astype(np.uint8)

    def _log_win1_win2_gt_target_flows(self, mv_outputs, ref_data, ref_frame, tgt_frame, view_num):
        """Rerun win1 / win2: GT target PC + flow arrows (origins ``src_3d_gt``, win1: ``warp3d−src``, win2: ``warp3d_delta``)."""
        if rr is None:
            return

        tgt_pts_all, tgt_colors_all, _ = self._extract_target_frame_data(mv_outputs, ref_frame, tgt_frame, view_num)
        if tgt_pts_all:
            t_pts = np.concatenate(tgt_pts_all, axis=0)
            t_cols = np.concatenate(tgt_colors_all, axis=0)
            t_pts_r = self._rerun_coord_apply_pts(t_pts)
            rr.log("win1/gt_target", rr.Points3D(t_pts_r, colors=t_cols, radii=self.POINT_RADIUS))
            rr.log("win2/gt_target", rr.Points3D(t_pts_r, colors=t_cols, radii=self.POINT_RADIUS))
        else:
            rr.log("win1/gt_target", rr.Clear(recursive=True))
            rr.log("win2/gt_target", rr.Clear(recursive=True))

        ref_out = mv_outputs[ref_frame * view_num]
        trd = getattr(ref_out, "track_3d", None)
        if not trd:
            rr.log("win1/flow_arrows", rr.Clear(recursive=True))
            rr.log("win2/sf_arrows", rr.Clear(recursive=True))
            return

        v_idx = 0
        w1o, w1v, w1c = [], [], []
        w2o, w2v, w2c = [], [], []
        max_n = int(self.cfg.get("rerun_track3d_max_points", 50000))
        rng = np.random.default_rng(seed=9000 + tgt_frame)

        for tgt_key in sorted(trd.keys(), key=lambda k: int(k)):
            tgt_flat = int(tgt_key)
            if tgt_flat // view_num != tgt_frame:
                continue
            tr = trd[tgt_key]
            w3 = MotionVisUtils.to_numpy(getattr(tr, "warp3d", None))
            d_pred = MotionVisUtils.to_numpy(getattr(tr, "warp3d_delta", None))
            if w3 is not None:
                w3 = np.asarray(w3, dtype=np.float64).reshape(-1, 3)
            if d_pred is not None:
                d_pred = np.asarray(d_pred, dtype=np.float64).reshape(-1, 3)
            if not self._track3d_has_vis_delta(tr, w3, d_pred):
                continue

            src = MotionVisUtils.to_numpy(getattr(tr, "src_3d_gt", None))
            if src is not None:
                src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
            n = 0
            if w3 is not None:
                n = w3.shape[0]
            if d_pred is not None:
                n = min(n, d_pred.shape[0]) if n > 0 else d_pred.shape[0]
            if src is not None:
                n = min(n, src.shape[0]) if n > 0 else src.shape[0]

            query_uv = getattr(tr, "query_uv", None)
            quv_np = MotionVisUtils.to_numpy(query_uv)
            if quv_np is not None:
                n_uv = np.asarray(quv_np, dtype=np.float64).reshape(-1, 2).shape[0]
                if n_uv > 0:
                    n = min(n, n_uv) if n > 0 else n_uv
            if n <= 0:
                continue
            if w3 is not None:
                w3 = w3[:n]
            d_raw = self._get_flow_delta_for_vis(tr, n)
            if src is not None and src.shape[0] >= n:
                src = src[:n]
            else:
                src = None

            rgb_row = self._rgb_from_query_uv(ref_out, query_uv, n)
            org_w = self._resolve_track_origin_world(ref_out, ref_data, v_idx, src, n)

            if w3 is not None and org_w is not None:
                w3_w = self._world_pts_from_track(ref_out, ref_data, v_idx, w3)
                sm1 = self._depth_valid(w3_w) & self._depth_valid(org_w) & np.isfinite(org_w).all(axis=1)
                if sm1.any():
                    vd = w3_w[sm1] - org_w[sm1]
                    w1o.append(org_w[sm1])
                    w1v.append(vd)
                    w1c.append((MotionVisUtils.compute_motion_colors(vd) * 255).astype(np.uint8))

            if d_raw is not None:
                d_use = np.asarray(d_raw, dtype=np.float64)[:n]
                vec_w = self._world_vecs_from_track(ref_out, ref_data, v_idx, d_use)
                if org_w is not None:
                    sm2 = self._depth_valid(org_w) & self._depth_valid(org_w + vec_w) & np.isfinite(vec_w).all(axis=1)
                    if sm2.any():
                        w2o.append(org_w[sm2])
                        w2v.append(vec_w[sm2])
                        w2c.append((MotionVisUtils.compute_motion_colors(vec_w[sm2]) * 255).astype(np.uint8))

        if w1o:
            o1 = np.concatenate(w1o, axis=0)
            v1 = np.concatenate(w1v, axis=0)
            c1 = np.concatenate(w1c, axis=0)
            if len(o1) > max_n:
                s1 = rng.choice(len(o1), max_n, replace=False)
                o1, v1, c1 = o1[s1], v1[s1], c1[s1]
            o1r, v1r = self._rerun_coord_apply_pts(o1), self._rerun_coord_apply_vecs(v1)
            rr.log("win1/flow_arrows", rr.Arrows3D(origins=o1r, vectors=v1r, colors=c1))
        else:
            rr.log("win1/flow_arrows", rr.Clear(recursive=True))

        if w2o:
            o2 = np.concatenate(w2o, axis=0)
            v2 = np.concatenate(w2v, axis=0)
            c2 = np.concatenate(w2c, axis=0)
            if len(o2) > max_n:
                s2 = rng.choice(len(o2), max_n, replace=False)
                o2, v2, c2 = o2[s2], v2[s2], c2[s2]
            o2r, v2r = self._rerun_coord_apply_pts(o2), self._rerun_coord_apply_vecs(v2)
            rr.log("win2/sf_arrows", rr.Arrows3D(origins=o2r, vectors=v2r, colors=c2))
        else:
            rr.log("win2/sf_arrows", rr.Clear(recursive=True))

    def _log_track3d_warp3d_delta_rerun(self, ref_out, tgt_frame, view_num, ref_data, mv_outputs):
        """Log ``warp3d`` (cumulative per time) and ``warp3d_delta`` (scene flow) from ``track_3d``."""
        if rr is None:
            return
        trd = getattr(ref_out, "track_3d", None)
        if not trd:
            return

        v_idx = 0
        pts_w_list, pts_c_list = [], []
        org_w_list, vec_w_list, arr_col_list = [], [], []
        sf_pts_c_list = []

        for tgt_key in sorted(trd.keys(), key=lambda k: int(k)):
            tgt_flat = int(tgt_key)
            if tgt_flat // view_num != tgt_frame:
                continue
            tr = trd[tgt_key]
            w3 = MotionVisUtils.to_numpy(getattr(tr, "warp3d", None))
            d_pred = MotionVisUtils.to_numpy(getattr(tr, "warp3d_delta", None))
            if w3 is not None:
                w3 = np.asarray(w3, dtype=np.float64).reshape(-1, 3)
            if d_pred is not None:
                d_pred = np.asarray(d_pred, dtype=np.float64).reshape(-1, 3)
            if not self._track3d_has_vis_delta(tr, w3, d_pred):
                continue

            n = 0
            if w3 is not None:
                n = w3.shape[0]
            if d_pred is not None:
                n = min(n, d_pred.shape[0]) if n > 0 else d_pred.shape[0]

            query_uv = getattr(tr, "query_uv", None)
            quv_np = MotionVisUtils.to_numpy(query_uv)
            if quv_np is not None:
                n_uv = np.asarray(quv_np, dtype=np.float64).reshape(-1, 2).shape[0]
                if n_uv > 0:
                    n = min(n, n_uv) if n > 0 else n_uv
            if n <= 0:
                continue
            if w3 is not None:
                w3 = w3[:n]
            d_raw = self._get_flow_delta_for_vis(tr, n)

            rgb_row = self._rgb_from_query_uv(ref_out, query_uv, n)
            org_w = self._resolve_track_origin_world_pred(ref_out, ref_data, v_idx, tr, n)

            if w3 is not None:
                w3_w = self._world_pts_from_track(ref_out, ref_data, v_idx, w3)
                valid = self._depth_valid(w3_w)
                if valid.any():
                    pw = w3_w[valid]
                    pc = rgb_row[valid]
                    pts_w_list.append(pw)
                    pts_c_list.append(pc)

            if d_raw is not None:
                d_use = np.asarray(d_raw, dtype=np.float64)[:n]
                vec_w = self._world_vecs_from_track(ref_out, ref_data, v_idx, d_use)
                if org_w is not None:
                    sm = self._depth_valid(org_w) & self._depth_valid(org_w + vec_w) & np.isfinite(vec_w).all(axis=1)
                    if sm.any():
                        o_sm = org_w[sm]
                        v_sm = vec_w[sm]
                        org_w_list.append(o_sm)
                        vec_w_list.append(v_sm)
                        sf_pts_c_list.append(rgb_row[sm])
                        arr_col_list.append((MotionVisUtils.compute_motion_colors(v_sm) * 255).astype(np.uint8))

        max_n = int(self.cfg.get("rerun_track3d_max_points", 50000))
        rng = np.random.default_rng(seed=2024 + tgt_frame)

        if getattr(self, "_rerun_acc_warp3d_pts", None) is None:
            self._rerun_acc_warp3d_pts, self._rerun_acc_warp3d_cols = [], []

        if pts_w_list:
            pw_cur = np.concatenate(pts_w_list, axis=0)
            pc_cur = np.concatenate(pts_c_list, axis=0)
            self._rerun_acc_warp3d_pts.append(pw_cur)
            self._rerun_acc_warp3d_cols.append(pc_cur)

        if self._rerun_acc_warp3d_pts:
            pw = np.concatenate(self._rerun_acc_warp3d_pts, axis=0)
            pc = np.concatenate(self._rerun_acc_warp3d_cols, axis=0)
            if len(pw) > max_n:
                sel = rng.choice(len(pw), max_n, replace=False)
                pw, pc = pw[sel], pc[sel]
            pw_r = self._rerun_coord_apply_pts(pw)
            rr.log("world_warp3d_dynamic/points", rr.Points3D(pw_r, colors=pc, radii=self.POINT_RADIUS * 1.1))
        else:
            rr.log("world_warp3d_dynamic/points", rr.Clear(recursive=True))

        rr.log("world_warp3d_dynamic/init_to_dyn_flow", rr.Clear(recursive=True))

        if org_w_list:
            ow = np.concatenate(org_w_list, axis=0)
            vw = np.concatenate(vec_w_list, axis=0)
            ac = np.concatenate(arr_col_list, axis=0)
            spc = np.concatenate(sf_pts_c_list, axis=0)
            if len(ow) > max_n:
                sel = rng.choice(len(ow), max_n, replace=False)
                ow, vw, ac, spc = ow[sel], vw[sel], ac[sel], spc[sel]
            ends = ow + vw
            pts_sf = np.concatenate([ow, ends], axis=0)
            col_sf = np.concatenate([spc, spc], axis=0)
            ow_r, vw_r = self._rerun_coord_apply_pts(ow), self._rerun_coord_apply_vecs(vw)
            pts_sf_r = self._rerun_coord_apply_pts(pts_sf)
            rr.log("world_warp3d_delta_sf/points", rr.Points3D(pts_sf_r, colors=col_sf, radii=self.POINT_RADIUS * 0.95))
            rr.log("world_warp3d_delta_sf/arrows", rr.Arrows3D(origins=ow_r, vectors=vw_r, colors=ac))
        else:
            rr.log("world_warp3d_delta_sf/points", rr.Clear(recursive=True))
            rr.log("world_warp3d_delta_sf/arrows", rr.Clear(recursive=True))

    def _process_target_frame(self, mv_outputs, meta_data, ref_data, ref_frame, tgt_frame, view_num, flow_img_dir, pred_history=None):
        """Sparse Rerun: win1 / win2 always; prediction panels if ``rerun_vis_cfg.enable_prediction``."""
        if pred_history is None:
            pred_history = {}

        self._log_win1_win2_gt_target_flows(mv_outputs, ref_data, ref_frame, tgt_frame, view_num)
        ref_out = mv_outputs[ref_frame * view_num]
        if self._sparse_rerun_enable_prediction:
            self._log_track3d_warp3d_delta_rerun(ref_out, tgt_frame, view_num, ref_data, mv_outputs)
        return None