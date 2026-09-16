import logging
import os

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import map_coordinates

from .utils import MotionVisUtils

logger = logging.getLogger(__name__)


class MotionVisualizer2D:
    """Generates 2D correspondence videos comparing source/target and GT/pred."""

    GRID = 32
    REF_FRAME = 0
    REF_VIEW = 0
    MAX_POINTS = 200

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _generate_rainbow_colors(n_points, src_pts, H, W):
        """Generate colors based on spatial position (Vectorized)."""
        if n_points == 0:
            return []
        hue = (src_pts[:, 0] / W * 0.7 + src_pts[:, 1] / H * 0.3) % 1.0
        hsv = np.stack([hue, np.full(n_points, 0.9), np.full(n_points, 0.95)], axis=1)
        rgb = hsv_to_rgb(hsv) * 255
        return rgb[:, ::-1].astype(int).tolist()

    @staticmethod
    def _generate_magnitude_colors(magnitudes):
        """Generate color map from blue (small) to red (large) based on motion magnitude (Vectorized)."""
        if len(magnitudes) == 0:
            return []

        valid_mask = np.isfinite(magnitudes)
        valid_mags = magnitudes[valid_mask]

        colors = np.full((len(magnitudes), 3), 128, dtype=int)
        if len(valid_mags) == 0:
            return colors.tolist()

        max_mag = np.percentile(valid_mags, 95) + 1e-8
        t = np.clip(valid_mags / max_mag, 0, 1)

        b = np.piecewise(t, [t < 0.25, (t >= 0.25) & (t < 0.5)], [1, lambda x: 1 - (x - 0.25) * 4, 0])
        g = np.piecewise(t, [t < 0.25, (t >= 0.25) & (t < 0.75), t >= 0.75],
                         [lambda x: x * 4, 1, lambda x: 1 - (x - 0.75) * 4])
        r = np.piecewise(t, [t < 0.5, (t >= 0.5) & (t < 0.75)], [0, lambda x: (x - 0.5) * 4, 1])

        colors[valid_mask] = np.stack([b, g, r], axis=1) * 255
        return colors.astype(int).tolist()

    @staticmethod
    def _draw_correspondence_panel(img_src, img_tgt, src_pts, dst_pts, colors, H, W):
        canvas = np.concatenate([img_src, img_tgt], axis=1)
        for (sx, sy), (ex, ey), color in zip(src_pts.astype(int), dst_pts.astype(int), colors):
            if 0 <= ex < W and 0 <= ey < H:
                c_tuple = tuple(color)
                cv2.circle(canvas, (sx, sy), 1, c_tuple, -1, cv2.LINE_AA)
                cv2.circle(canvas, (ex + W, ey), 1, c_tuple, -1, cv2.LINE_AA)
                cv2.line(canvas, (sx, sy), (ex + W, ey), tuple(c // 2 for c in color), 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _draw_motion_trails_panel(img, src_pts, dst_pts, colors, H, W):
        canvas = img.copy()
        for (sx, sy), (ex, ey), color in zip(src_pts.astype(int), dst_pts.astype(int), colors):
            if 0 <= ex < W and 0 <= ey < H:
                c_tuple = tuple(color)
                cv2.arrowedLine(canvas, (sx, sy), (ex, ey), c_tuple, 1, cv2.LINE_AA, tipLength=0.2)
                cv2.circle(canvas, (sx, sy), 1, c_tuple, -1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _draw_gt_vs_pred_panel(img, gt_src_pts, gt_dst_pts, pred_src_pts, pred_dst_pts, H, W):
        canvas = (img.astype(float) * 0.7).astype(np.uint8)

        def draw_vectors(src, dst, color):
            if src is not None and dst is not None:
                for (sx, sy), (ex, ey) in zip(src.astype(int), dst.astype(int)):
                    if 0 <= sx < W and 0 <= sy < H and 0 <= ex < W and 0 <= ey < H:
                        cv2.arrowedLine(canvas, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA, tipLength=0.2)
                        cv2.circle(canvas, (sx, sy), 1, color, -1, cv2.LINE_AA)

        draw_vectors(gt_src_pts, gt_dst_pts, (0, 200, 0))
        draw_vectors(pred_src_pts, pred_dst_pts, (0, 0, 255))
        return canvas

    @staticmethod
    def _annotate_panels(p_pred, p_gt, p_cmp, p_trail, ref_f, tgt_f, H, W, has_gt):
        def put_text(img, text, pos, color=(255, 255, 255)):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        pred_title = f"PRED: Frame {ref_f} -> Frame {tgt_f}"
        put_text(p_pred, pred_title, (10, 18), (0, 200, 255))
        cv2.putText(p_pred, f"Source (F{ref_f})", (10, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(p_pred, f"Target (F{tgt_f})", (W + 10, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        gt_h = f"GT: Frame {ref_f} -> Frame {tgt_f}" if has_gt else "GT Correspondence (not available)"
        put_text(p_gt, gt_h, (10, 18), (0, 255, 0) if has_gt else (100, 100, 100))

        cmp_h = "GT (green) vs PRED (red)" if has_gt else "PRED only (no GT available)"
        put_text(p_cmp, cmp_h, (10, 18), (0, 255, 0) if has_gt else (0, 0, 255))
        put_text(p_trail, "Motion magnitude (blue=small, red=large)", (10, 18), (0, 255, 255))

    @staticmethod
    def _save_video(frames, output_path, fps=10):
        if not frames:
            return
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            out.write(frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

    @staticmethod
    def _get_gt_motion_2d(output, src_idx, tgt_idx, H, W, orig_H=None, orig_W=None):
        if getattr(output, 'trajs_2d', None) is None:
            return None, None

        trajs_2d = MotionVisUtils.to_numpy(output.trajs_2d)
        if max(src_idx, tgt_idx) >= trajs_2d.shape[0]:
            return None, None

        gt_src, gt_dst = trajs_2d[src_idx].copy(), trajs_2d[tgt_idx].copy()

        if orig_H != H or orig_W != W:
            gt_src *= [W / orig_W, H / orig_H]
            gt_dst *= [W / orig_W, H / orig_H]

        mask = np.ones(len(gt_src), dtype=bool)
        if getattr(output, 'trajs_visibs', None) is not None:
            visibs = MotionVisUtils.to_numpy(output.trajs_visibs)
            mask &= (visibs[src_idx] > 0) & (visibs[tgt_idx] > 0)
        if getattr(output, 'trajs_valids', None) is not None:
            valids = MotionVisUtils.to_numpy(output.trajs_valids)
            mask &= (valids[src_idx] > 0) & (valids[tgt_idx] > 0)

        mask &= (gt_src[:, 0] >= 0) & (gt_src[:, 0] < W) & (gt_src[:, 1] >= 0) & (gt_src[:, 1] < H) & \
                (gt_dst[:, 0] >= 0) & (gt_dst[:, 0] < W) & (gt_dst[:, 1] >= 0) & (gt_dst[:, 1] < H)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None, None

        if len(indices) > MotionVisualizer2D.MAX_POINTS:
            indices = np.random.choice(indices, MotionVisualizer2D.MAX_POINTS, replace=False)

        return gt_src[indices], gt_dst[indices]

    def run(self, mv_outputs, motion_out_dir, data_idx, meta_data=None):
        if getattr(mv_outputs[0], 'scene_flow_pred', None) is None:
            return

        os.makedirs(motion_out_dir, exist_ok=True)
        frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]
        orig_h, orig_w = MotionVisUtils.get_orig_dims(meta_data)

        ref_idx = self.REF_FRAME * view_num + self.REF_VIEW
        ref_out = mv_outputs[ref_idx]
        if ref_out.scene_flow_pred is None:
            return

        H, W = ref_out.pointmap_h, ref_out.pointmap_w
        depth, intrinsics = MotionVisUtils.get_depth_from_output(ref_out, H, W), MotionVisUtils.to_numpy(ref_out.intrinsics)
        if depth is None or intrinsics is None:
            return

        src_img = MotionVisUtils.get_image_from_output(ref_out, H, W)
        src_extrinsics = MotionVisUtils.to_numpy(getattr(ref_out, 'extrinsics', None))

        y_grid, x_grid = np.mgrid[self.GRID // 2:H:self.GRID, self.GRID // 2:W:self.GRID]
        src_pts = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)
        rainbow_colors = self._generate_rainbow_colors(len(src_pts), src_pts, H, W)

        video_frames = []
        for tgt_idx in range(frame_num * view_num):
            if tgt_idx == ref_idx or tgt_idx >= len(mv_outputs) or tgt_idx >= ref_out.scene_flow_pred.shape[0]:
                continue

            tgt_frame = tgt_idx // view_num
            target_out = mv_outputs[tgt_idx]
            tgt_img = MotionVisUtils.get_image_from_output(target_out, H, W)
            tgt_extrinsics = MotionVisUtils.to_numpy(getattr(target_out, 'extrinsics', None))

            flow_3d = ref_out.scene_flow_pred[tgt_idx].detach().cpu().numpy()
            abs_coords_2d = MotionVisUtils.project_scene_flow_to_2d(flow_3d, depth, intrinsics, src_extrinsics, tgt_extrinsics)

            u_end = map_coordinates(abs_coords_2d[0], [src_pts[:, 1], src_pts[:, 0]], order=1)
            v_end = map_coordinates(abs_coords_2d[1], [src_pts[:, 1], src_pts[:, 0]], order=1)
            dst_pts = np.stack([u_end, v_end], axis=-1)

            mag_colors = self._generate_magnitude_colors(np.linalg.norm(dst_pts - src_pts, axis=1))
            gt_src, gt_dst = self._get_gt_motion_2d(ref_out, ref_idx, tgt_idx, H, W, orig_h, orig_w)
            gt_colors = self._generate_rainbow_colors(len(gt_src), gt_src, H, W) if gt_src is not None else None

            p_pred = self._draw_correspondence_panel(src_img, tgt_img, src_pts, dst_pts, rainbow_colors, H, W)
            if gt_src is not None:
                p_gt = self._draw_correspondence_panel(src_img, tgt_img, gt_src, gt_dst, gt_colors, H, W)
            else:
                p_gt = np.ascontiguousarray((np.concatenate([src_img, tgt_img], axis=1).astype(float) * 0.5).astype(np.uint8))
                cv2.putText(p_gt, "No GT available", (W - 60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

            p_cmp = self._draw_gt_vs_pred_panel(src_img, gt_src, gt_dst, src_pts, dst_pts, H, W)
            p_trail = self._draw_motion_trails_panel(src_img, src_pts, dst_pts, mag_colors, H, W)

            self._annotate_panels(p_pred, p_gt, p_cmp, p_trail, self.REF_FRAME, tgt_frame, H, W, gt_src is not None)
            video_frames.append(np.concatenate([p_pred, p_gt, np.concatenate([p_cmp, p_trail], axis=1)], axis=0))

        if video_frames:
            video_path = os.path.join(motion_out_dir, f"motion_from_f{self.REF_FRAME:02d}_{data_idx:06d}.mp4")
            self._save_video(video_frames, video_path, fps=10)
            logger.info(f"Saved motion video ({len(video_frames)} frames): {video_path}")
