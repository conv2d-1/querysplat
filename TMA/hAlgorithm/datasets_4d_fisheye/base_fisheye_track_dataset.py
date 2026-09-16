"""Base class for fisheye 4D tracking datasets.

This extends BaseTrackDataset with proper fisheye camera handling:
- Circular valid region cropping
- Fisheye polynomial projection for static trajectory generation
- Fisheye boundary mask generation
"""

import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset
from hAlgorithm.datasets_4d_fisheye.utils import (
    fisheye_unproject,
    fisheye_project,
    detect_fisheye_valid_radius,
    generate_fisheye_boundary_mask,
)
from hAlgorithm.datasets_4d_fisheye.vis_utils import compute_effective_sensor_size


class BaseFisheyeTrackDataset(BaseTrackDataset):
    """Base class for fisheye 4D tracking datasets.

    Extends BaseTrackDataset with fisheye-specific functionality:
    - Automatic circular valid region detection and cropping
    - Correct fisheye polynomial projection for static trajectory generation
    - Fisheye boundary mask generation for downstream training

    Parameters
    ----------
    fisheye_k : list or None
        Polynomial distortion coefficients [k0, k1, k2, k3, k4].
        If None, loaded from data (requires 'distor_k' in HDF5).
    sensor_size : list or None
        Sensor size [width, height] in mm.
        If None, loaded from data (requires 'sensor_size' in HDF5).
    fisheye_crop : bool
        Whether to crop images to the valid circular region.
    fisheye_radius : float or None
        Manually specified valid region radius. If None, auto-detected.
    """

    def __init__(
        self,
        fisheye_k: list = None,
        sensor_size: list = None,
        fisheye_crop: bool = True,
        fisheye_radius: float = None,
        **kwargs,
    ):
        self.fisheye_k = np.array(fisheye_k) if fisheye_k is not None else None
        self.sensor_size = sensor_size
        self.fisheye_crop = fisheye_crop
        self.fisheye_radius_config = fisheye_radius

        self._detected_fisheye_radius: float = None
        self._fisheye_crop_params: dict = None

        super().__init__(**kwargs)

    def load_data(self, data_info):
        """Load a single frame's data with fisheye-specific processing."""
        data_batch = super().load_data(data_info)

        if self.fisheye_crop:
            data_batch = self._apply_fisheye_crop(data_batch)

        return data_batch

    def _get_fisheye_params(self, data_batch: dict) -> dict:
        """Get fisheye parameters from config or data_batch.

        Returns dict with keys: k, sensor_size, cx, cy, W, H
        """
        intrinsics = data_batch.get("curr_intrinsics")
        if intrinsics is None:
            raise ValueError("Intrinsics required for fisheye processing")

        fx, fy, cx, cy = intrinsics[:4]
        rgb = data_batch.get("curr_rgb")
        H, W = rgb.shape[:2] if rgb is not None else (900, 1200)

        k = self.fisheye_k
        sensor_size = self.sensor_size

        if k is None and "curr_distor_k" in data_batch:
            k = np.array(data_batch["curr_distor_k"])
        if sensor_size is None and "curr_sensor_size" in data_batch:
            sensor_size = data_batch["curr_sensor_size"]

        return {
            "k": k,
            "sensor_size": sensor_size,
            "cx": cx,
            "cy": cy,
            "fx": fx,
            "fy": fy,
            "W": W,
            "H": H,
        }

    def _compute_fisheye_crop_params(self, data_batch: dict) -> dict:
        """Compute cropping parameters for fisheye images."""
        rgb = data_batch.get("curr_rgb")
        if rgb is None:
            return None

        H, W = rgb.shape[:2]
        intrinsics = data_batch.get("curr_intrinsics")

        if intrinsics is not None:
            cx, cy = intrinsics[2], intrinsics[3]
        else:
            cx, cy = W / 2, H / 2

        if self.fisheye_radius_config is not None:
            radius = self.fisheye_radius_config
        elif self._detected_fisheye_radius is not None:
            radius = self._detected_fisheye_radius
        else:
            radius = detect_fisheye_valid_radius(rgb, cx, cy)
            self._detected_fisheye_radius = radius
            logging.info(f"Auto-detected fisheye radius: {radius:.1f} pixels")

        crop_size = int(2 * radius)
        crop_x0 = int(cx - radius)
        crop_y0 = int(cy - radius)

        crop_x0 = max(0, min(crop_x0, W - crop_size))
        crop_y0 = max(0, min(crop_y0, H - crop_size))

        return {
            "cx": cx,
            "cy": cy,
            "radius": radius,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_size": crop_size,
        }

    def _apply_fisheye_crop(self, data_batch: dict) -> dict:
        """Apply fisheye cropping to all relevant data in data_batch."""
        rgb = data_batch.get("curr_rgb")
        if rgb is None:
            return data_batch

        params = self._compute_fisheye_crop_params(data_batch)
        if params is None:
            return data_batch

        x0, y0, size = params["crop_x0"], params["crop_y0"], params["crop_size"]
        radius = params["radius"]

        def crop_image(img):
            if img is None:
                return None
            if img.ndim == 2:
                return img[y0 : y0 + size, x0 : x0 + size].copy()
            else:
                return img[y0 : y0 + size, x0 : x0 + size, ...].copy()

        data_batch["curr_rgb"] = crop_image(rgb)

        for key in ["curr_depth", "curr_depth_mask", "curr_sem", "curr_motion_mask"]:
            if key in data_batch and data_batch[key] is not None:
                data_batch[key] = crop_image(data_batch[key])

        intrinsics = data_batch.get("curr_intrinsics")
        if intrinsics is not None:
            new_intrinsics = list(intrinsics)
            new_intrinsics[2] = intrinsics[2] - x0  # cx
            new_intrinsics[3] = intrinsics[3] - y0  # cy
            data_batch["curr_intrinsics"] = new_intrinsics

        if "curr_trajs_2d" in data_batch and data_batch["curr_trajs_2d"] is not None:
            trajs_2d = data_batch["curr_trajs_2d"].copy()
            trajs_2d[:, 0] -= x0
            trajs_2d[:, 1] -= y0
            data_batch["curr_trajs_2d"] = trajs_2d

        new_cx = params["cx"] - x0
        new_cy = params["cy"] - y0
        boundary_mask = generate_fisheye_boundary_mask(size, size, new_cx, new_cy, radius)
        data_batch["curr_fisheye_boundary_mask"] = boundary_mask

        if "curr_depth_mask" in data_batch and data_batch["curr_depth_mask"] is not None:
            data_batch["curr_depth_mask"] = (
                data_batch["curr_depth_mask"].astype(bool) & boundary_mask
            ).astype(data_batch["curr_depth_mask"].dtype)

        self._fisheye_crop_params = {
            "crop_x0": x0,
            "crop_y0": y0,
            "crop_size": size,
            "radius": radius,
            "new_cx": new_cx,
            "new_cy": new_cy,
        }

        return data_batch

    def _generate_static_trajs(self, mf_data):
        """Generate static trajectories using fisheye projection model.

        Overrides the base class method to use correct fisheye geometry.
        """
        V = len(mf_data["data"])
        data_0 = mf_data["data"][0]

        dyn_trajs_2d = data_0.get("curr_trajs_2d")
        if dyn_trajs_2d is None or len(dyn_trajs_2d) == 0:
            return None

        num_static = int(len(dyn_trajs_2d) * self.static_traj_ratio)
        if num_static <= 0:
            return None

        depth_0 = data_0.get("curr_depth")
        if depth_0 is None:
            return None

        params = self._get_fisheye_params(data_0)
        k = params["k"]
        sensor_size = params["sensor_size"]

        if k is None or sensor_size is None:
            logging.warning(
                "Fisheye parameters (distor_k, sensor_size) not available. "
                "Falling back to base class static trajectory generation."
            )
            return super()._generate_static_trajs(mf_data)

        H, W = depth_0.shape
        cx, cy = params["cx"], params["cy"]
        
        # Compute effective sensor size for current (cropped/resized) image
        # The original sensor_size corresponds to original image dimensions (e.g., 1200x900)
        # After cropping, we need to adjust sensor_size to maintain correct pixel_size
        original_image_size = self._get_original_image_size()  # (1200, 900)
        current_image_size = (W, H)
        effective_sensor = compute_effective_sensor_size(
            sensor_size, original_image_size, current_image_size
        )
        sensor_w, sensor_h = effective_sensor

        static_mask = self._build_static_traj_sampling_mask(
            data_0, depth_0, dyn_trajs_2d
        )
        if static_mask is None or not static_mask.any():
            return None

        fisheye_mask = data_0.get("curr_fisheye_boundary_mask")
        if fisheye_mask is not None:
            static_mask &= fisheye_mask

        static_yx = np.argwhere(static_mask)
        if len(static_yx) == 0:
            return None
        num_static = min(num_static, len(static_yx))

        scene_name = (
            mf_data["info"][0].get("scene", "") if mf_data.get("info") else ""
        )
        frame_id = mf_data["info"][0].get("frame_id", 0) if mf_data.get("info") else 0
        seed = hash((scene_name, frame_id, num_static)) % (2**31)
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(static_yx), num_static, replace=False)
        sampled_y = static_yx[indices, 0]
        sampled_x = static_yx[indices, 1]

        d = depth_0[sampled_y, sampled_x]

        # Blender fisheye outputs radial depth (distance along ray)
        X_cam, Y_cam, Z_cam = fisheye_unproject(
            sampled_x.astype(np.float64),
            sampled_y.astype(np.float64),
            d.astype(np.float64),
            cx,
            cy,
            k,
            sensor_w,
            sensor_h,
            W,
            H,
            depth_is_along_ray=True,
        )

        p_cam_h = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(Z_cam)], axis=-1)

        extrinsics_0 = np.array(data_0["curr_extrinsics"], dtype=np.float64)
        E_c2w_0 = np.linalg.inv(extrinsics_0)
        p_world = (E_c2w_0 @ p_cam_h.T).T[:, :3]

        tol = self.static_traj_depth_tolerance
        result = []

        for t in range(V):
            dt = mf_data["data"][t]
            params_t = self._get_fisheye_params(dt)
            cx_t, cy_t = params_t["cx"], params_t["cy"]
            H_t, W_t = params_t["H"], params_t["W"]

            # Compute effective sensor size for this frame (may differ from frame 0)
            current_image_size_t = (W_t, H_t)
            effective_sensor_t = compute_effective_sensor_size(
                sensor_size, original_image_size, current_image_size_t
            )
            sensor_w_t, sensor_h_t = effective_sensor_t

            E_w2c_t = np.array(dt["curr_extrinsics"], dtype=np.float64)
            depth_t = dt["curr_depth"]

            p_world_h = np.concatenate(
                [p_world, np.ones((num_static, 1), dtype=np.float64)], axis=-1
            )
            p_cam_t = (E_w2c_t @ p_world_h.T).T[:, :3]

            u_t, v_t, _ = fisheye_project(
                p_cam_t[:, 0],
                p_cam_t[:, 1],
                p_cam_t[:, 2],
                cx_t,
                cy_t,
                k,
                sensor_w_t,
                sensor_h_t,
                W_t,
                H_t,
            )

            # Compute radial depth for comparison (Blender uses radial depth)
            radial_depth_t = np.sqrt(
                p_cam_t[:, 0]**2 + p_cam_t[:, 1]**2 + p_cam_t[:, 2]**2
            )

            in_bounds = (
                (u_t >= 0) & (u_t < W_t) & (v_t >= 0) & (v_t < H_t) & (p_cam_t[:, 2] > 0)
            )

            fisheye_mask_t = dt.get("curr_fisheye_boundary_mask")
            if fisheye_mask_t is not None:
                # Handle NaN values in u_t, v_t
                valid_uv = np.isfinite(u_t) & np.isfinite(v_t)
                u_int_check = np.zeros_like(u_t, dtype=np.int64)
                v_int_check = np.zeros_like(v_t, dtype=np.int64)
                u_int_check[valid_uv] = np.clip(np.round(u_t[valid_uv]).astype(np.int64), 0, W_t - 1)
                v_int_check[valid_uv] = np.clip(np.round(v_t[valid_uv]).astype(np.int64), 0, H_t - 1)
                in_circle = np.zeros(len(u_t), dtype=bool)
                in_circle[valid_uv] = fisheye_mask_t[v_int_check[valid_uv], u_int_check[valid_uv]]
                in_bounds = in_bounds & in_circle & valid_uv

            # Handle NaN values
            valid_uv = np.isfinite(u_t) & np.isfinite(v_t)
            u_int = np.zeros_like(u_t, dtype=np.int64)
            v_int = np.zeros_like(v_t, dtype=np.int64)
            u_int[valid_uv] = np.clip(np.round(u_t[valid_uv]).astype(np.int64), 0, W_t - 1)
            v_int[valid_uv] = np.clip(np.round(v_t[valid_uv]).astype(np.int64), 0, H_t - 1)
            
            rendered_depth = np.zeros(len(u_t), dtype=np.float64)
            rendered_depth[valid_uv] = depth_t[v_int[valid_uv], u_int[valid_uv]].astype(np.float64)
            
            # Compare radial depths (Blender fisheye uses radial depth)
            depth_ok = np.abs(rendered_depth - radial_depth_t) < tol * np.abs(radial_depth_t)
            depth_ok = depth_ok & valid_uv

            result.append(
                {
                    "trajs_2d": np.stack([u_t, v_t], axis=-1).astype(np.float32),
                    "trajs_3d": p_world.astype(np.float32),
                    "visibs": in_bounds & depth_ok,
                    "valids": in_bounds & (p_cam_t[:, 2] > 0),
                }
            )

        return result

    def _attach_fisheye_meta(self, mf_data_batch: dict) -> None:
        """Write fisheye camera metadata and invalid-region masks for the pipeline."""
        meta = mf_data_batch["meta_data"]
        num_frames = int(mf_data_batch["image"].shape[0])

        # Nested list so WFMQueryPipeline.get_camera_type() ([0][0]) resolves correctly.
        meta["camera_type"] = [["FISHEYE_BLENDER"]]

        k = torch.from_numpy(self.fisheye_k.astype(np.float32))
        meta["distort_k"] = k.unsqueeze(0).expand(num_frames, -1)

        output_h = int(meta["input_height"])
        output_w = int(meta["input_width"])
        if self.fisheye_crop and self._fisheye_crop_params is not None:
            crop_size = self._fisheye_crop_params["crop_size"]
            pre_resize_size = (crop_size, crop_size)
        else:
            pre_resize_size = self._get_original_image_size()
        eff_sensor = compute_effective_sensor_size(
            self.sensor_size, self._get_original_image_size(), pre_resize_size
        )
        meta["sensor_size"] = torch.tensor(eff_sensor, dtype=torch.float32).unsqueeze(0).expand(
            num_frames, -1
        )
        meta["crop_offset"] = torch.zeros(num_frames, 2, dtype=torch.float32)

    def _expand_fisheye_region(self, mask_hw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Broadcast a (H, W) valid-region mask to match *target* (V, 1, H', W')."""
        V = target.shape[0]
        region = mask_hw.view(1, 1, *mask_hw.shape).expand(V, 1, -1, -1)
        if region.shape[-2:] != target.shape[-2:]:
            region = F.interpolate(
                region.float(), size=target.shape[-2:], mode="nearest"
            ).bool()
        return region

    @staticmethod
    def _squeeze_to_hw(t: torch.Tensor) -> torch.Tensor:
        """Reduce leading dims so the result is a 2D (H, W) map."""
        while t is not None and t.ndim > 2:
            t = t.squeeze(0)
        return t

    def _align_raw_depth_pointmap_to_transformed(self, data_dict: dict) -> None:
        """Rebuild depth_raw / pointmap_raw from transformed depth (same H,W as image).

        BaseDataset uses pre-transform ``curr_depth`` for raw GT; after fisheye crop
        and ResizeKeepRatio that map no longer matches ``image`` / query UV, which
        breaks grid_sample + depth loss mask shapes in WFMQueryPipeline.
        """
        if not (self.with_depth_raw or self.with_pointmap_raw):
            return
        depth = data_dict.get("depth")
        depth_mask = data_dict.get("depth_mask")
        if depth is None or depth_mask is None:
            return

        depth_hw = self._squeeze_to_hw(depth.detach().float())
        mask_hw = self._squeeze_to_hw(depth_mask.detach().bool())

        if depth_hw.shape != mask_hw.shape:
            logging.warning(
                "depth / depth_mask shape mismatch after transforms: %s vs %s",
                tuple(depth_hw.shape),
                tuple(mask_hw.shape),
            )
            return

        if self.with_depth_raw:
            data_dict["depth_raw"] = depth_hw
            data_dict["depth_raw_mask"] = mask_hw.unsqueeze(0)

        if self.with_pointmap_raw:
            intrinsics = data_dict.get("intrinsics")
            if intrinsics is not None:
                data_dict["pointmap_raw"] = self.load_pointmap(depth_hw, intrinsics).float()
            if "depth_raw_mask" not in data_dict:
                data_dict["depth_raw_mask"] = mask_hw.unsqueeze(0)

    @staticmethod
    def _ensure_mask_has_channel(mf_data_batch: dict, keys=("depth_raw_mask", "depth_mask")) -> None:
        """Stacked clips should expose masks as (V, 1, H, W) for the MV-query pipeline."""
        for key in keys:
            if key not in mf_data_batch:
                continue
            m = mf_data_batch[key]
            if m is not None and m.ndim == 3:
                mf_data_batch[key] = m.unsqueeze(1)

    def get_data_for_trainval(
        self,
        idx,
        data_info=None,
        data_batch=None,
        transform_info=None,
        mf_debug=False,
    ):
        data_dict = super().get_data_for_trainval(
            idx,
            data_info=data_info,
            data_batch=data_batch,
            transform_info=transform_info,
            mf_debug=mf_debug,
        )
        self._align_raw_depth_pointmap_to_transformed(data_dict)
        return data_dict

    def get_mf_data_for_trainval(self, idx):
        """Assemble multi-frame data with fisheye boundary mask and camera metadata."""
        mf_data_batch = super().get_mf_data_for_trainval(idx)
        self._ensure_mask_has_channel(mf_data_batch)

        if self.fisheye_crop and "image" in mf_data_batch:
            _, _, output_h, output_w = mf_data_batch["image"].shape
            radius = min(output_h, output_w) / 2
            cx = output_w / 2
            cy = output_h / 2
            fisheye_mask = generate_fisheye_boundary_mask(
                int(output_h), int(output_w), cx, cy, radius
            )
            fisheye_mask_t = torch.from_numpy(fisheye_mask).bool()
            mf_data_batch["fisheye_boundary_mask"] = fisheye_mask_t

            if "depth_raw_mask" in mf_data_batch:
                region = self._expand_fisheye_region(
                    fisheye_mask_t, mf_data_batch["depth_raw_mask"]
                )
                mf_data_batch["depth_raw_mask"] = mf_data_batch["depth_raw_mask"] & region

            if "edge_mask" in mf_data_batch:
                region = self._expand_fisheye_region(
                    fisheye_mask_t, mf_data_batch["edge_mask"]
                )
                mf_data_batch["edge_mask"] = mf_data_batch["edge_mask"] | (~region)

        self._attach_fisheye_meta(mf_data_batch)
        return mf_data_batch

    def _get_original_image_size(self) -> tuple:
        """Get original image size (width, height) before any cropping/resizing.

        This is needed to correctly scale sensor_size for fisheye unprojection.
        Default is (1200, 900) for HaBlender fisheye data.
        """
        return (1200, 900)

    def load_pointmap(self, depth, intrinsics):
        """Generate point cloud from depth map using fisheye polynomial projection.

        Overrides the base class pinhole model with correct fisheye unprojection.
        
        IMPORTANT: Blender fisheye outputs RADIAL depth (distance along ray),
        not Z-depth (distance along optical axis).

        Args:
            depth: Depth map as tensor (1, H, W) or numpy array (H, W).
            intrinsics: 3x3 intrinsics matrix (used for cx, cy only).

        Returns:
            torch.Tensor: Point cloud map of shape (3, H, W).
        """
        import torch
        
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze(0).numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.numpy()

        H, W = depth.shape
        
        # Get fisheye parameters
        k = self.fisheye_k
        sensor_size = self.sensor_size
        
        if k is None or sensor_size is None:
            logging.warning(
                "Fisheye parameters not available for load_pointmap. "
                "Falling back to pinhole model (INCORRECT for fisheye!)."
            )
            return super().load_pointmap(depth, intrinsics)
        
        # Extract principal point from intrinsics
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Compute effective sensor size for current image dimensions.
        # IMPORTANT: We must use the pre-resize (post-crop) image size, because
        # resize changes pixel density but NOT the physical sensor extent covered.
        # compute_effective_sensor_size preserves original mm/pixel, which is only
        # correct for the crop step (no pixel density change). After resize, the
        # effective sensor = physical extent of cropped region, independent of
        # how many pixels currently represent it.
        original_image_size = self._get_original_image_size()
        if self.fisheye_crop and self._fisheye_crop_params is not None:
            crop_size = self._fisheye_crop_params["crop_size"]
            pre_resize_size = (crop_size, crop_size)
        else:
            pre_resize_size = original_image_size
        effective_sensor = compute_effective_sensor_size(
            sensor_size, original_image_size, pre_resize_size
        )
        sensor_w, sensor_h = effective_sensor
        
        # Create pixel coordinate grid (with 0.5 offset for pixel centers)
        u, v = np.meshgrid(
            np.arange(W, dtype=np.float64) + 0.5,
            np.arange(H, dtype=np.float64) + 0.5,
            indexing="xy"
        )
        
        # Unproject using fisheye model
        # Blender fisheye outputs RADIAL depth (distance along ray)
        X, Y, Z = fisheye_unproject(
            u.ravel(),
            v.ravel(),
            depth.ravel().astype(np.float64),
            cx,
            cy,
            k,
            sensor_w,
            sensor_h,
            W,
            H,
            depth_is_along_ray=True,  # Blender fisheye uses radial depth
        )
        
        # Reshape to (3, H, W)
        points = np.stack([X, Y, Z], axis=0).reshape(3, H, W)
        
        return torch.from_numpy(points).float()

    def save_tracking_gif(
        self,
        sample_idx=0,
        output_path="tracking_debug.gif",
        **kwargs,
    ):
        """Save tracking visualization GIF with correct fisheye unprojection."""
        from hAlgorithm.datasets_4d_fisheye.vis_utils import save_tracking_gif_fisheye

        return save_tracking_gif_fisheye(
            self,
            sample_idx=sample_idx,
            output_path=output_path,
            fisheye_k=self.fisheye_k,
            sensor_size=self.sensor_size,
            original_image_size=self._get_original_image_size(),
            **kwargs,
        )

    def visualize_tracking_debug(
        self,
        sample_idx=0,
        output_path="tracking_debug.rrd",
        **kwargs,
    ):
        """Visualize tracking debug with correct fisheye unprojection."""
        from hAlgorithm.datasets_4d_fisheye.vis_utils import visualize_tracking_debug_fisheye

        return visualize_tracking_debug_fisheye(
            self,
            sample_idx=sample_idx,
            output_path=output_path,
            fisheye_k=self.fisheye_k,
            sensor_size=self.sensor_size,
            original_image_size=self._get_original_image_size(),
            **kwargs,
        )
