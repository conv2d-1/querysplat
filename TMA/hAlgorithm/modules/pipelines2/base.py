import logging
import os

import torch
import torch.nn as nn

from hAlgorithm.utils import instantiate_from_config
from hAlgorithm.utils.checkpoint_util import load_state_dict_compatible


class Pipeline(nn.Module):
    def __init__(self, model, **kwargs):
        super(Pipeline, self).__init__()

        self.model = instantiate_from_config(model)
        self.cache_dict = dict()

    def get_train_parameters(self):
        return self.parameters()

    def accelerator_prepare(self, accelerator, optimizer, lr_scheduler, train_dataloader):
        weight_dtype = torch.float32
        if accelerator.mixed_precision in ["fp16", "fp8"]:
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        logging.info(f"weight_dtype: {weight_dtype}")

        self.cuda(accelerator.device)
        self.device = accelerator.device
        self.dtype = weight_dtype

        if train_dataloader is None:
            self.model = accelerator.prepare(self.model)
            return None, None, None

        (
            self.model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            self.model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
        return optimizer, train_dataloader, lr_scheduler

    def load_checkpoint(self, ckpt_path=None, state_dict=None):
        if ckpt_path is not None:
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state_dict is not None:
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            load_state_dict_compatible(
                self.model,
                state_dict,
                strict=False,
                log_prefix=f"load_checkpoint({ckpt_path})",
            )

    def save_checkpoint(self, accelerator, ckpt_dir=None):
        model = accelerator.unwrap_model(self.model)
        state_dict = model.state_dict()

        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "ckpt.pth")
            tmp_path = ckpt_path + ".tmp"
            try:
                torch.save(state_dict, tmp_path)
                os.replace(tmp_path, ckpt_path)
                logging.info(f"Model is saved to: {ckpt_path}")
            except Exception:
                logging.warning(f"Model is saved error!!")
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        return state_dict

    @staticmethod
    def normalize(data, scale):
        return data / scale if scale is not None else data

    @staticmethod
    def denormalize(data, scale):
        return data * scale if scale is not None else data

    @staticmethod
    def get_camera_type_from_meta(meta_data, default="PINHOLE"):
        if meta_data is None or "camera_type" not in meta_data:
            return default
        ct = meta_data["camera_type"]
        while isinstance(ct, (list, tuple)):
            if len(ct) == 0:
                return default
            ct = ct[0]
        return ct

    @staticmethod
    def _align_fisheye_batch_meta(tensor, n_flat, device=None):
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        if device is not None:
            tensor = tensor.to(device)
        if tensor.dim() == 3:
            tensor = tensor.reshape(-1, *tensor.shape[2:])
        elif tensor.dim() == 2:
            if tensor.shape[0] == 1 and n_flat > 1:
                tensor = tensor.expand(n_flat, -1)
            elif tensor.shape[0] > n_flat:
                tensor = tensor[:n_flat]
            elif tensor.shape[0] < n_flat:
                tensor = tensor[0:1].expand(n_flat, -1)
        return tensor

    def depth_to_points_from_meta(
        self,
        depth,
        K,
        meta_data,
        device=None,
        cache=True,
        frame_index=None,
    ):
        """Unproject depth using camera model from *meta_data* (matches fisheye dataset GT)."""
        camera_type = self.get_camera_type_from_meta(meta_data)
        kwargs = dict(camera_type=camera_type, device=device, cache=cache)
        if camera_type == "FISHEYE_BLENDER":
            for key, arg_name in (
                ("distort_k", "distort_k"),
                ("sensor_size", "sensor_size"),
                ("crop_offset", "crop_offset"),
            ):
                if key not in meta_data:
                    continue
                val = meta_data[key]
                if frame_index is not None:
                    if torch.is_tensor(val):
                        if val.dim() == 3:
                            val = val[0, frame_index]
                        elif val.dim() >= 2 and val.shape[0] > frame_index:
                            val = val[frame_index]
                        elif val.dim() == 2 and val.shape[0] == 1:
                            val = val[0]
                    elif isinstance(val, (list, tuple)) and len(val) > frame_index:
                        val = val[frame_index]
                kwargs[arg_name] = val
        return self.depth_to_points(depth, K, **kwargs)

    def depth_to_points(
        self,
        depth,
        K,
        device=None,
        cache=True,
        camera_type="PINHOLE",
        distort_k=None,
        sensor_size=None,
        crop_offset=None,
    ):

        if camera_type == "PINHOLE":
            device = device or self.device

            ndim = depth.ndim
            if ndim == 5:
                b, v, _, h, w = depth.shape
            elif ndim == 4:
                b, _, h, w = depth.shape
            else:
                raise ValueError(f"Invalid depth shape: {depth.shape}")

            cache_key = f"depth_to_points_{h}_{w}"

            if cache_key in self.cache_dict:
                points = self.cache_dict[cache_key].clone()
                if points.device != device:
                    points.to(device)
            else:
                grid_x, grid_y = torch.meshgrid(torch.arange(w) + 0.5, torch.arange(h) + 0.5, indexing="xy")
                points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().to(device)
                if cache:
                    self.cache_dict[cache_key] = points

            rays_d = K.inverse() @ points  # (B, 3, HW)

            if ndim == 5:
                pts = depth.flatten(3) * rays_d
                depth = pts.reshape(b, v, 3, h, w)
            elif ndim == 4:
                pts = depth.flatten(2) * rays_d
                depth = pts.reshape(b, 3, h, w)
            else:
                raise NotImplementedError
            return depth
        
        elif camera_type in ["FISHEYE_EQUIDISTANT"]:
            device = device or self.device

            ndim = depth.ndim
            if ndim == 5:
                b, v, _, h, w = depth.shape
            elif ndim == 4:
                b, _, h, w = depth.shape
            else:
                raise ValueError(f"Invalid depth shape: {depth.shape}")

            cache_key = f"fisheye_grid_{h}_{w}"

            if cache_key in self.cache_dict:
                grid = self.cache_dict[cache_key]
                if grid.device != device:
                    grid = grid.to(device)
            else:
                grid_x, grid_y = torch.meshgrid(
                    torch.arange(w, device=device) + 0.5,
                    torch.arange(h, device=device) + 0.5,
                    indexing="xy",
                )
                grid = torch.stack([grid_x, grid_y], dim=0).float()  # (2, H, W)
                if cache:
                    self.cache_dict[cache_key] = grid

            fx = K[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
            fy = K[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
            cx = K[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
            cy = K[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

            x_cam = (grid[0] - cx) / fx
            y_cam = (grid[1] - cy) / fy

            # Equidistant fisheye: theta = r
            r = torch.sqrt(x_cam ** 2 + y_cam ** 2)
            sin_r = torch.sin(r)
            cos_r = torch.cos(r)
            factor = torch.where(r > 1e-8, sin_r / r, torch.ones_like(r))

            rays_x = x_cam * factor
            rays_y = y_cam * factor
            rays_z = cos_r

            norm = torch.sqrt(rays_x ** 2 + rays_y ** 2 + rays_z ** 2)
            rays_x = rays_x / norm
            rays_y = rays_y / norm
            rays_z = rays_z / norm

            if ndim == 5:
                rays_d = torch.stack([rays_x, rays_y, rays_z], dim=2)  # (B, V, 3, H, W)
            else:
                rays_d = torch.stack([rays_x, rays_y, rays_z], dim=1)  # (B, 3, H, W)
            
            # NOTE: Normalize the depth by cos(theta) to get the euclidean depth
            # euclidean_depth = depth / cos_r.unsqueeze(-3)  # cos_r = cos(theta)
 
            # return euclidean_depth * rays_d

            return depth * rays_d

        elif camera_type == "FISHEYE_BLENDER":
            from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame_fisheye

            if distort_k is None or sensor_size is None:
                raise ValueError(
                    "distort_k and sensor_size are required for FISHEYE_BLENDER depth_to_points"
                )

            device = device or self.device
            ndim = depth.ndim
            if ndim == 5:
                b, v, _, h, w = depth.shape
                n_flat = b * v
                K_flat = K.reshape(n_flat, 3, 3)
                depth_flat = depth.reshape(n_flat, 1, h, w)
            elif ndim == 4:
                b, _, h, w = depth.shape
                n_flat = b
                K_flat = K.reshape(n_flat, 3, 3) if K.ndim > 3 else K
                depth_flat = depth
            else:
                raise ValueError(f"Invalid depth shape: {depth.shape}")

            distort_k = self._align_fisheye_batch_meta(distort_k, n_flat, device)
            sensor_size = self._align_fisheye_batch_meta(sensor_size, n_flat, device)
            crop_offset = self._align_fisheye_batch_meta(crop_offset, n_flat, device)

            rays_d = get_rays_in_camera_frame_fisheye(
                K_flat,
                h,
                w,
                fisheye_model="blender",
                k_coeffs=distort_k,
                sensor_size=sensor_size,
                crop_offset=crop_offset,
            )

            if ndim == 5:
                pts = depth_flat * rays_d
                return pts.reshape(b, v, 3, h, w)
            pts = depth_flat * rays_d
            return pts.reshape(b, 3, h, w)

        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

    def global_to_local(self, points, w2c):
        """
        glb: [b, v, 3, h, w], torch.Tensor
        extrinsics: [b, v, 4, 4], torch.Tensor
        scale: [b, v, 1, 1, 1], torch.Tensor
        """
        b, v, _, h, w = points.shape
        # Convert glb to homogeneous coordinates: [b, v, 4, h, w]
        glb_homo = torch.cat([points, points.new_ones((b, v, 1, h, w))], dim=2)  # [b, v, 4, h, w]
        # Reshape for matrix multiplication
        glb_homo = glb_homo.view(b, v, 4, -1)  # [b, v, 4, h*w]
        # Apply extrinsics transformation
        local_homo = torch.matmul(w2c, glb_homo)  # [b, v, 4, h*w]
        # Reshape back to original spatial shape
        local_homo = local_homo.view(b, v, 4, h, w)
        # Normalize by scale
        local = local_homo[:, :, :3, :, :]  # Keep only the first 3 channels
        return local
