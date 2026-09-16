import logging
import os

import cv2
import numpy as np
import matplotlib.cm as cm

from .utils import MotionVisUtils

logger = logging.getLogger(__name__)


class MotionVisualizer:
    """Renders 2D diagnostic visualizations of sparse motion predictions."""

    # Maximum queries to draw per panel to keep rendering fast.
    MAX_DRAW_QUERIES = 300

    def __init__(self):
        try:
            import cv2
            import matplotlib.cm as cm_mpl
            self.cv2 = cv2
            self.cm = cm_mpl
            self.available = True
        except ImportError:
            self.available = False

    def render(self, out0, save_dir: str, **kwargs) -> None:
        if not self.available:
            return

        pred = self._as_nonempty(getattr(out0, 'track_pred', None))
        pred_flow_2d = self._as_nonempty(getattr(out0, 'pred_flow_2d', None))
        if pred is None and pred_flow_2d is None:
            return

        gt = self._as_nonempty(getattr(out0, 'track_gt', None))
        vis = self._as_nonempty(getattr(out0, 'track_vis_pred', None))
        rgb = getattr(out0, 'rgb', None)
        query_uv = self._as_nonempty(getattr(out0, 'motion_queries_uv', None))
        intrinsics = getattr(out0, 'intrinsics', None)

        H, W = rgb.shape[:2] if rgb is not None else (256, 256)
        valid_3d = None

        if pred is not None:
            q_sizes = [len(pred)]
            if gt is not None:
                q_sizes.append(len(gt))
            if vis is not None:
                q_sizes.append(len(vis))
            if query_uv is not None:
                q_sizes.append(len(query_uv))
            q_common = min(q_sizes) if q_sizes else 0
            if q_common == 0:
                pred = None
            else:
                pred = pred[:q_common]
                if gt is not None:
                    gt = gt[:q_common]
                if vis is not None:
                    vis = vis[:q_common]
                if query_uv is not None:
                    query_uv = query_uv[:q_common]
                valid_3d = vis > 0.5 if vis is not None else np.ones(q_common, dtype=bool)
                epe_colors, info = self._compute_epe(pred, gt, valid_3d)
                self._draw_query_points(rgb, query_uv, valid_3d, epe_colors, info, H, W, save_dir)

                if query_uv is not None and intrinsics is not None:
                    self._draw_motion_arrows(
                        rgb, pred, gt, query_uv, valid_3d, epe_colors,
                        intrinsics, H, W, save_dir,
                    )

        if pred is None:
            epe_colors = None
            info = None

        # 2D flow visualization: predicted vs GT UV correspondence in image space.
        gt_2d_tgt = self._as_nonempty(getattr(out0, 'motion_queries_gt_2d_tgt', None))
        tgt_frames = self._as_nonempty(getattr(out0, 'motion_queries_tgt_frame', None))
        if pred_flow_2d is not None and query_uv is not None:
            aligned = self._prepare_flow_inputs(
                src_uv=query_uv,
                pred_flow_2d=pred_flow_2d,
                gt_2d_tgt=gt_2d_tgt,
                tgt_frames=tgt_frames,
                valid=vis,
                context="render",
            )
            if aligned is None:
                pred_flow_2d = None
            else:
                query_uv, pred_flow_2d, gt_2d_tgt, tgt_frames, valid_2d = aligned
                if info is None:
                    info = f"Q={len(query_uv)} valid={int(valid_2d.sum())}"

        if pred_flow_2d is not None and query_uv is not None:
            outputs_list = kwargs.get('outputs_list', [out0])
            self._render_flow_2d(
                out0=out0,
                src_uv=query_uv,
                pred_flow_2d=pred_flow_2d,
                gt_2d_tgt=gt_2d_tgt,
                tgt_frames=tgt_frames,
                valid=valid_2d,
                outputs_list=outputs_list,
                H=H, W=W,
                save_dir=os.path.join(save_dir, "flow_2d"),
            )

        if info is None:
            num_queries = (
                len(pred_flow_2d) if pred_flow_2d is not None else
                len(query_uv) if query_uv is not None else 0
            )
            valid_fallback = (
                int(valid_2d.sum()) if pred_flow_2d is not None and query_uv is not None else 0
            )
            info = f"Q={num_queries} valid={valid_fallback}"

        logger.info("[SparseMotionPipeline] 2D vis: %s", info)

    @staticmethod
    def _as_nonempty(arr):
        if arr is None:
            return None
        try:
            return arr if len(arr) > 0 else None
        except TypeError:
            return arr

    def _prepare_flow_inputs(
        self,
        src_uv,
        pred_flow_2d,
        gt_2d_tgt,
        tgt_frames,
        valid,
        context: str,
    ):
        """Align flow-visualization inputs to a shared query dimension."""
        if src_uv is None or pred_flow_2d is None:
            return None

        src_uv = np.asarray(src_uv)
        pred_flow_2d = np.asarray(pred_flow_2d)
        if src_uv.ndim != 2 or src_uv.shape[-1] != 2:
            logger.warning("[flow2d] %s invalid src_uv shape %s; skipping.", context, src_uv.shape)
            return None
        if pred_flow_2d.ndim != 2 or pred_flow_2d.shape[-1] != 2:
            logger.warning(
                "[flow2d] %s invalid pred_flow_2d shape %s; skipping.",
                context,
                pred_flow_2d.shape,
            )
            return None

        if valid is None:
            valid = np.ones(src_uv.shape[0], dtype=bool)
        valid = np.asarray(valid).reshape(-1) > 0.5

        if tgt_frames is not None:
            tgt_frames = np.asarray(tgt_frames).reshape(-1)
            if tgt_frames.shape[0] == 0:
                tgt_frames = None

        if gt_2d_tgt is not None:
            gt_2d_tgt = np.asarray(gt_2d_tgt)
            if gt_2d_tgt.shape[0] == 0:
                gt_2d_tgt = None
            elif gt_2d_tgt.ndim != 2 or gt_2d_tgt.shape[-1] != 2:
                logger.warning(
                    "[flow2d] %s invalid gt_2d_tgt shape %s; drawing without GT overlay.",
                    context,
                    gt_2d_tgt.shape,
                )
                gt_2d_tgt = None

        lengths = {
            "src_uv": src_uv.shape[0],
            "pred_flow_2d": pred_flow_2d.shape[0],
            "valid": valid.shape[0],
        }
        if tgt_frames is not None:
            lengths["tgt_frames"] = tgt_frames.shape[0]
        if gt_2d_tgt is not None:
            lengths["gt_2d_tgt"] = gt_2d_tgt.shape[0]

        q_common = min(lengths.values()) if lengths else 0
        if q_common <= 0:
            logger.warning("[flow2d] %s has no usable queries after alignment: %s", context, lengths)
            return None

        if len(set(lengths.values())) > 1:
            logger.warning(
                "[flow2d] %s length mismatch %s; truncating to %d.",
                context,
                lengths,
                q_common,
            )

        src_uv = src_uv[:q_common]
        pred_flow_2d = pred_flow_2d[:q_common]
        valid = valid[:q_common]
        if tgt_frames is not None:
            tgt_frames = tgt_frames[:q_common]
        if gt_2d_tgt is not None:
            gt_2d_tgt = gt_2d_tgt[:q_common]

        return src_uv, pred_flow_2d, gt_2d_tgt, tgt_frames, valid

    def _compute_epe(self, pred, gt, valid):
        """Compute per-query End-Point Error and colormap."""
        if gt is None:
            info = f"Q={len(pred)} valid={int(valid.sum())}"
            return None, info

        epe = np.linalg.norm(pred - gt, axis=-1)
        valid_epe = epe[valid]
        mean_epe = float(valid_epe.mean()) if len(valid_epe) > 0 else 0.0
        med_epe = float(np.median(valid_epe)) if len(valid_epe) > 0 else 0.0
        info = f"Q={len(pred)} valid={int(valid.sum())}  mean_EPE={mean_epe:.4f}  med={med_epe:.4f}"

        cap = np.percentile(valid_epe, 90) + 1e-6 if len(valid_epe) > 0 else 1.0
        colors = (self.cm.RdYlGn_r(np.clip(epe / cap, 0, 1))[:, :3] * 255).astype(np.uint8)
        return colors, info

    def _base_canvas(self, rgb, H, W, darken=1.0):
        if rgb is not None:
            canvas = (np.clip(rgb[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
        else:
            canvas = np.ones((H, W, 3), dtype=np.uint8) * 200
        if darken < 1.0:
            canvas = (canvas.astype(float) * darken).astype(np.uint8)
        return np.ascontiguousarray(canvas)

    def _draw_query_points(self, rgb, query_uv, valid, epe_colors, info, H, W, save_dir):
        canvas = self._base_canvas(rgb, H, W)
        cv2 = self.cv2

        for thickness, color in [(2, (0, 0, 0)), (1, (255, 255, 255))]:
            cv2.putText(canvas, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, thickness, cv2.LINE_AA)

        if query_uv is not None:
            for i in range(len(query_uv)):
                if not valid[i]:
                    continue
                x, y = int(query_uv[i, 0] * (W - 1)), int(query_uv[i, 1] * (H - 1))
                if 0 <= x < W and 0 <= y < H:
                    c = tuple(int(v) for v in epe_colors[i, ::-1]) if epe_colors is not None else (0, 255, 0)
                    cv2.circle(canvas, (x, y), 3, c, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (x, y), 3, (0, 0, 0), 1, cv2.LINE_AA)

        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "query_points_epe.png"), canvas)

    def _draw_motion_arrows(self, rgb, pred, gt, query_uv, valid, epe_colors,
                            intrinsics, H, W, save_dir):
        cv2 = self.cv2
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        def _project(pt3d):
            z = max(pt3d[2], 1e-4)
            return int(pt3d[0] * fx / z + cx), int(pt3d[1] * fy / z + cy)

        def _src_pixel(i):
            return int(query_uv[i, 0] * (W - 1)), int(query_uv[i, 1] * (H - 1))

        canvas = self._base_canvas(rgb, H, W, darken=0.6)
        for i in range(len(pred)):
            if not valid[i]:
                continue
            sx, sy = _src_pixel(i)
            if not (0 <= sx < W and 0 <= sy < H):
                continue
            px, py = _project(pred[i])
            c = tuple(int(v) for v in epe_colors[i, ::-1]) if epe_colors is not None else (0, 200, 255)
            cv2.arrowedLine(canvas, (sx, sy), (px, py), c, 1, cv2.LINE_AA, tipLength=0.15)
            cv2.circle(canvas, (sx, sy), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "Pred motion (src->tgt)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_dir, "motion_arrows.png"), canvas)

        if gt is not None:
            canvas = self._base_canvas(rgb, H, W, darken=0.5)
            for i in range(len(pred)):
                if not valid[i]:
                    continue
                sx, sy = _src_pixel(i)
                if not (0 <= sx < W and 0 <= sy < H):
                    continue
                gx, gy = _project(gt[i])
                cv2.arrowedLine(canvas, (sx, sy), (gx, gy), (0, 200, 0), 1, cv2.LINE_AA, tipLength=0.15)
                px, py = _project(pred[i])
                cv2.arrowedLine(canvas, (sx, sy), (px, py), (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(canvas, "GT(green) vs Pred(red)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(save_dir, "gt_vs_pred_arrows.png"), canvas)

    # ── 2D Flow Visualization ─────────────────────────────────────────────

    @staticmethod
    def _position_colors(uv: np.ndarray, n: int) -> np.ndarray:
        """Return per-query BGR colors based on spatial UV position (rainbow hue)."""
        from matplotlib.colors import hsv_to_rgb
        hue = (uv[:n, 0] * 0.65 + uv[:n, 1] * 0.35) % 1.0
        hsv = np.stack([hue, np.full(n, 0.9), np.full(n, 0.95)], axis=1)
        rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
        return rgb[:, ::-1]   # RGB → BGR

    @staticmethod
    def _epe2d_colors(epe_px: np.ndarray, cmap_name: str = "RdYlGn_r") -> np.ndarray:
        """Map per-query 2D EPE (pixels) to BGR colors via a diverging colormap."""
        import matplotlib.cm as cm_mpl
        cap = np.percentile(epe_px[np.isfinite(epe_px)], 90) + 1e-6
        t = np.clip(epe_px / cap, 0, 1)
        rgba = cm_mpl.get_cmap(cmap_name)(t)
        bgr = (rgba[:, [2, 1, 0]] * 255).astype(np.uint8)
        return bgr

    def _rgb_canvas(self, rgb_float: np.ndarray | None, H: int, W: int,
                    darken: float = 1.0) -> np.ndarray:
        """Build an OpenCV BGR uint8 canvas from a float [0,1] RGB image."""
        if rgb_float is not None:
            canvas = (np.clip(rgb_float[:H, :W, ::-1], 0, 1) * 255).astype(np.uint8)
        else:
            canvas = np.full((H, W, 3), 40, dtype=np.uint8)
        if darken < 1.0:
            canvas = (canvas.astype(np.float32) * darken).astype(np.uint8)
        return np.ascontiguousarray(canvas)

    def _get_frame_rgb(self, outputs_list: list, frame_idx: int,
                       H: int, W: int) -> np.ndarray | None:
        """Retrieve the RGB image for a given frame index from the outputs list."""
        if frame_idx < len(outputs_list):
            return getattr(outputs_list[frame_idx], 'rgb', None)
        return None

    def _render_flow_2d(
        self,
        out0,
        src_uv: np.ndarray,
        pred_flow_2d: np.ndarray,
        gt_2d_tgt: np.ndarray | None,
        tgt_frames: np.ndarray | None,
        valid: np.ndarray,
        outputs_list: list,
        H: int,
        W: int,
        save_dir: str,
    ) -> None:
        """Generate per-target-frame 2D optical flow comparison panels.

        For each target frame, a 2×2 mosaic is written to *save_dir*:

        ┌─────────────────────────┬─────────────────────────┐
        │ Source + colored points │ Target: pred★ vs GT●    │
        ├─────────────────────────┼─────────────────────────┤
        │ GT(green) ↔ Pred(blue)  │ EPE heatmap on target   │
        │ arrows from source      │ (green=good, red=bad)   │
        └─────────────────────────┴─────────────────────────┘

        Args:
            out0:         First output object (carries source RGB).
            src_uv:       [Q, 2] normalised source UV coords.
            pred_flow_2d: [Q, 2] predicted flow Δ in normalised UV.
            gt_2d_tgt:    [Q, 2] GT target UV (normalised), or None.
            tgt_frames:   [Q] int — target frame index per query, or None.
            valid:        [Q] bool mask.
            outputs_list: Full list of ReconstructOutput for all frames.
            H, W:         Source image pixel dimensions.
            save_dir:     Output directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        cv2_mod = self.cv2

        aligned = self._prepare_flow_inputs(
            src_uv=src_uv,
            pred_flow_2d=pred_flow_2d,
            gt_2d_tgt=gt_2d_tgt,
            tgt_frames=tgt_frames,
            valid=valid,
            context="_render_flow_2d",
        )
        if aligned is None:
            return
        src_uv, pred_flow_2d, gt_2d_tgt, tgt_frames, valid = aligned

        src_rgb = getattr(out0, 'rgb', None)   # float [0,1] RGB
        src_frame_idx = getattr(out0, 'total_index', 0)

        # Determine unique target frames
        if tgt_frames is not None:
            unique_tgts = np.unique(tgt_frames).tolist()
        else:
            # Single-target fallback: assume all queries share one target
            unique_tgts = [1 if src_frame_idx == 0 else 0]
            tgt_frames = np.zeros(len(src_uv), dtype=np.int32) + unique_tgts[0]

        for tgt_f in unique_tgts:
            mask = valid & (tgt_frames == tgt_f)
            if not mask.any():
                continue

            q_uv     = src_uv[mask]          # [Qf, 2]  normalised
            q_flow   = pred_flow_2d[mask]     # [Qf, 2]  normalised delta
            pred_tgt = q_uv + q_flow          # predicted normalised target UV
            gt_tgt   = gt_2d_tgt[mask] if gt_2d_tgt is not None else None

            # Sub-sample for readability
            Qf = len(q_uv)
            idx_draw = (
                np.linspace(0, Qf - 1, min(Qf, self.MAX_DRAW_QUERIES), dtype=int)
                if Qf > self.MAX_DRAW_QUERIES
                else np.arange(Qf)
            )
            q_uv_d   = q_uv[idx_draw]
            pred_d   = pred_tgt[idx_draw]
            gt_d     = gt_tgt[idx_draw] if gt_tgt is not None else None

            colors_bgr = self._position_colors(q_uv_d, len(q_uv_d))

            # Pixel coordinates
            def _px(uv_norm):
                xy = uv_norm * np.array([W - 1, H - 1], dtype=np.float32)
                return xy.astype(np.int32)

            src_px   = _px(q_uv_d)
            pred_px  = _px(pred_d)
            gt_px    = _px(gt_d) if gt_d is not None else None

            # Compute EPE in pixels for this target frame
            if gt_tgt is not None:
                epe_px_all = np.linalg.norm((pred_tgt - gt_tgt) * [W, H], axis=-1)
                mean_epe   = float(np.mean(epe_px_all))
                med_epe    = float(np.median(epe_px_all))
                epe_info   = f"EPE mean={mean_epe:.1f}px  med={med_epe:.1f}px  N={Qf}"
                epe_colors = self._epe2d_colors(epe_px_all[idx_draw])
            else:
                epe_info   = f"N={Qf}  (no GT)"
                epe_colors = None

            tgt_rgb = self._get_frame_rgb(outputs_list, tgt_f, H, W)

            # ── Panel A: source image with colored source points ──────────
            pA = self._rgb_canvas(src_rgb, H, W)
            for k, (sx, sy) in enumerate(src_px):
                c = tuple(int(v) for v in colors_bgr[k])
                cv2_mod.circle(pA, (sx, sy), 4, c, -1, cv2_mod.LINE_AA)
                cv2_mod.circle(pA, (sx, sy), 4, (0, 0, 0), 1, cv2_mod.LINE_AA)
            self._put_label(cv2_mod, pA, f"Source (frame {src_frame_idx})", W)

            # ── Panel B: target image with pred★ and GT● ─────────────────
            pB = self._rgb_canvas(tgt_rgb, H, W, darken=0.75)
            for k in range(len(src_px)):
                c = tuple(int(v) for v in colors_bgr[k])
                px, py = pred_px[k]
                if 0 <= px < W and 0 <= py < H:
                    cv2_mod.drawMarker(pB, (px, py), c, cv2_mod.MARKER_STAR, 8, 1, cv2_mod.LINE_AA)
                if gt_px is not None:
                    gx, gy = gt_px[k]
                    if 0 <= gx < W and 0 <= gy < H:
                        cv2_mod.circle(pB, (gx, gy), 4, c, 1, cv2_mod.LINE_AA)
                        # Error segment: GT → pred
                        if epe_colors is not None:
                            ec = tuple(int(v) for v in epe_colors[k])
                            cv2_mod.line(pB, (gx, gy), (px, py), ec, 1, cv2_mod.LINE_AA)
            self._put_label(cv2_mod, pB,
                            f"Target (frame {tgt_f})  ★pred ●GT", W,
                            sub=epe_info)

            # ── Panel C: source with GT(green) vs Pred(blue) arrows ───────
            pC = self._rgb_canvas(src_rgb, H, W, darken=0.55)
            for k in range(len(src_px)):
                sx, sy = src_px[k]
                if not (0 <= sx < W and 0 <= sy < H):
                    continue
                px, py = pred_px[k]
                cv2_mod.arrowedLine(pC, (sx, sy), (px, py),
                                    (255, 100, 0), 1, cv2_mod.LINE_AA, tipLength=0.2)
                if gt_px is not None:
                    gx, gy = gt_px[k]
                    cv2_mod.arrowedLine(pC, (sx, sy), (gx, gy),
                                        (0, 200, 0), 1, cv2_mod.LINE_AA, tipLength=0.2)
                cv2_mod.circle(pC, (sx, sy), 2, (255, 255, 255), -1, cv2_mod.LINE_AA)
            legend = "GT(green) vs Pred(blue) from source"
            self._put_label(cv2_mod, pC, legend, W)

            # ── Panel D: EPE heatmap on target (green=small, red=large) ──
            pD = self._rgb_canvas(tgt_rgb, H, W, darken=0.55)
            if epe_colors is not None:
                for k in range(len(pred_px)):
                    px, py = pred_px[k]
                    if 0 <= px < W and 0 <= py < H:
                        ec = tuple(int(v) for v in epe_colors[k])
                        cv2_mod.circle(pD, (px, py), 5, ec, -1, cv2_mod.LINE_AA)
                        cv2_mod.circle(pD, (px, py), 5, (0, 0, 0), 1, cv2_mod.LINE_AA)
            self._put_label(cv2_mod, pD, "EPE heatmap (green=low, red=high)", W,
                            sub=epe_info if gt_tgt is not None else "no GT")

            # ── Assemble 2×2 mosaic ───────────────────────────────────────
            # Add 2-pixel white separator between panels
            sep_v = np.full((H, 2, 3), 220, dtype=np.uint8)
            sep_h = np.full((2, W * 2 + 2, 3), 220, dtype=np.uint8)
            top_row    = np.concatenate([pA, sep_v, pB], axis=1)
            bottom_row = np.concatenate([pC, sep_v, pD], axis=1)
            mosaic     = np.concatenate([top_row, sep_h, bottom_row], axis=0)

            out_path = os.path.join(save_dir, f"flow2d_tgt{tgt_f:03d}.png")
            cv2_mod.imwrite(out_path, mosaic)
            logger.info("[flow2d] saved %s  %s", out_path, epe_info)

    @staticmethod
    def _put_label(cv2_mod, canvas: np.ndarray, title: str, W: int,
                   sub: str | None = None) -> None:
        """Write a white-on-black title (and optional sub-line) onto the canvas."""
        for thickness, color in [(2, (0, 0, 0)), (1, (255, 255, 255))]:
            cv2_mod.putText(canvas, title, (8, 20), cv2_mod.FONT_HERSHEY_SIMPLEX,
                            0.45, color, thickness, cv2_mod.LINE_AA)
        if sub:
            for thickness, color in [(2, (0, 0, 0)), (1, (200, 230, 255))]:
                cv2_mod.putText(canvas, sub, (8, 38), cv2_mod.FONT_HERSHEY_SIMPLEX,
                                0.38, color, thickness, cv2_mod.LINE_AA)
