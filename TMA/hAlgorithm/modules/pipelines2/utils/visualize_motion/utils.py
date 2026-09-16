import logging

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class MotionVisUtils:
    """Helper methods for tensor conversion, normalization, and geometric projections."""

    @staticmethod
    def to_numpy(tensor):
        """Convert torch tensor to numpy array."""
        if torch is not None and isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    @staticmethod
    def normalize_rgb(img):
        """Normalize RGB image to 0-255 uint8 range."""
        img = MotionVisUtils.to_numpy(img)
        if img is None:
            return None
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return img

    @staticmethod
    def get_depth_from_output(output, H, W):
        """Priority: POINTMAP Z > Rendered Depth"""
        depth = None
        if getattr(output, 'pointmap', None) is not None:
            pts = MotionVisUtils.to_numpy(output.pointmap)
            if pts.ndim == 2:
                pts = pts.reshape(output.pointmap_h, output.pointmap_w, 3)
            depth = pts[..., 2]
        elif getattr(output, 'render_depth', None) is not None:
            depth = MotionVisUtils.to_numpy(output.render_depth)

        if depth is not None:
            if depth.ndim == 3:
                depth = depth.squeeze()
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        return depth

    @staticmethod
    def get_image_from_output(output, H, W):
        """Extract RGB image from output and convert to BGR for OpenCV."""
        if getattr(output, 'rgb', None) is not None:
            img = MotionVisUtils.normalize_rgb(output.rgb)
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return np.zeros((H, W, 3), dtype=np.uint8)

    @staticmethod
    def project_scene_flow_to_2d(scene_flow_3d, depth, intrinsics, src_extrinsics=None, tgt_extrinsics=None):
        """Project 3D scene flow to 2D pixel coordinates."""
        _, h, w = scene_flow_3d.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        z_t = np.maximum(depth, 1e-3)
        x_cam = (x_grid - cx) * z_t / fx
        y_cam = (y_grid - cy) * z_t / fy

        x_next, y_next, z_next = x_cam + scene_flow_3d[0], y_cam + scene_flow_3d[1], z_t + scene_flow_3d[2]
        z_next = np.maximum(z_next, 1e-3)

        if src_extrinsics is not None and tgt_extrinsics is not None:
            pts_src_cam = np.stack([x_next.flatten(), y_next.flatten(), z_next.flatten()], axis=0)
            src_c2w = np.linalg.inv(src_extrinsics)
            pts_world = src_c2w[:3, :3] @ pts_src_cam + src_c2w[:3, 3:4]
            pts_tgt_cam = tgt_extrinsics[:3, :3] @ pts_world + tgt_extrinsics[:3, 3:4]

            x_next = pts_tgt_cam[0].reshape(h, w)
            y_next = pts_tgt_cam[1].reshape(h, w)
            z_next = np.maximum(pts_tgt_cam[2].reshape(h, w), 1e-3)

        u_next = (x_next * fx / z_next) + cx
        v_next = (y_next * fy / z_next) + cy
        return np.stack([u_next, v_next], axis=0)

    @staticmethod
    def compute_motion_colors(vectors):
        """Compute HSV colors based on motion vector direction and magnitude."""
        if len(vectors) == 0:
            return np.zeros((0, 3))
        mags = np.linalg.norm(vectors, axis=1)
        max_mag = np.maximum(mags.max(), 1e-6)
        norm_vecs = vectors / (mags[:, np.newaxis] + 1e-8)

        hue = (np.arctan2(norm_vecs[:, 1], norm_vecs[:, 0]) + np.pi) / (2 * np.pi)
        norm_mag = np.clip(mags / max_mag, 0, 1)
        saturation, value = 0.3 + 0.7 * norm_mag, 0.5 + 0.5 * norm_mag

        return hsv_to_rgb(np.stack([hue, saturation, value], axis=1))

    @staticmethod
    def apply_c2w(c2w, pts):
        """Apply camera-to-world transformation to 3D points."""
        if c2w is None or len(pts) == 0:
            return pts
        pts_homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        return (c2w @ pts_homo.T).T[:, :3]

    @staticmethod
    def get_orig_dims(meta_data, default_h=None, default_w=None):
        """Extract origin_height and origin_width robustly from meta_data."""
        if meta_data is None:
            return default_h, default_w
        h = meta_data.get("origin_height", [default_h])[0]
        w = meta_data.get("origin_width", [default_w])[0]
        return getattr(h, 'item', lambda: h)(), getattr(w, 'item', lambda: w)()
