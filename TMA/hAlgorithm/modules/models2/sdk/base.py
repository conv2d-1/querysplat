import logging
import time

import torch
import torch.nn as nn

from hAlgorithm.utils import instantiate_from_config


class Base(nn.Module):
    def __init__(
        self,
        rgb_encoder=None,
        depth_encoder=None,
        ray_encoder=None,
        extra_encoder=None,
        fuse_encoder=None,
        depth_head=None,
        normal_head=None,
        freeze_modules=None,
        load_modules=None,
        **kwargs,
    ):
        super(Base, self).__init__()

        # Store module names for freeze & introspection
        self.module_names = []
        self.freeze_modules = freeze_modules
        self.load_modules = load_modules or {}

        # Encoder
        self.rgb_encoder = self._instantiate_and_register(rgb_encoder, "rgb_encoder")
        self.depth_encoder = self._instantiate_and_register(depth_encoder, "depth_encoder")
        self.ray_encoder = self._instantiate_and_register(ray_encoder, "ray_encoder")
        self.extra_encoder = self._instantiate_and_register(extra_encoder, "extra_encoder")
        self.fuse_encoder = self._instantiate_and_register(fuse_encoder, "fuse_encoder")

        # Geometric Head
        self.depth_head = self._instantiate_and_register(depth_head, "depth_head")
        self.normal_head = self._instantiate_and_register(normal_head, "normal_head")

    def _instantiate_and_register(self, config, name):
        """Helper to instantiate module and register its name."""
        module = instantiate_from_config(config)
        if module is not None:
            self.module_names.append(name)
            self._load_module_ckpt(module, name)
        return module

    def _load_module_ckpt(self, module, name):
        """Load a checkpoint for a specific module if configured in load_modules.

        load_modules[name] can be:
          - str: checkpoint path, loaded with strict=True
          - dict: {"ckpt": path, "prefix": "model.xxx.", "strict": True}
            prefix is stripped from state_dict keys before loading.
        """
        if name not in self.load_modules:
            return

        load_cfg = self.load_modules[name]
        if isinstance(load_cfg, str):
            ckpt_path, prefix, strict = load_cfg, "", True
        else:
            ckpt_path = load_cfg["ckpt"]
            prefix = load_cfg.get("prefix", "")
            strict = load_cfg.get("strict", True)

        logging.info(f"Loading checkpoint for module '{name}' from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        if prefix:
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        missing, unexpected = module.load_state_dict(state_dict, strict=strict)
        logging.info(f"Loaded checkpoint for module '{name}': {len(state_dict)} keys, "
                     f"missing={len(missing)}, unexpected={len(unexpected)}")

    def freeze(self):
        """Freeze specified modules."""
        if self.freeze_modules is None:
            return
        for module_name in self.freeze_modules:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                if module is not None:
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                logging.warning(f"Module {module_name} not found for freezing.")

    def aggregator(self, rgb, prompt_depth, scale, intrinsics, ray_directions, w2c, meta_data):
        if rgb.ndim == 5:
            frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]

            b, n, c, h, w = rgb.shape
            assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

            rgb = rgb.view(b * n, c, h, w)

            if prompt_depth is not None:
                prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

            if scale is not None:
                scale = scale.view(b * n, *scale.shape[-3:])

            if intrinsics is not None:
                intrinsics = intrinsics.view(b * n, 3, 3)

            if w2c is not None:
                w2c = w2c.view(b * n, 4, 4)

        prompt_ray = prompt_extra = None

        if self.extra_encoder is not None:
            prompt_extra = self.extra_encoder(intrinsics=intrinsics, w2c=w2c, scale=scale, meta_data=meta_data)

        if self.ray_encoder is not None and intrinsics is not None:
            prompt_ray = self.ray_encoder(ray_directions=ray_directions, intrinsics=intrinsics, w2c=w2c, meta_data=meta_data)

        if self.depth_encoder is not None and prompt_depth is not None:
            prompt_depth = self.depth_encoder(prompt_depth, prompt_ray=prompt_ray, prompt_extra=prompt_extra, meta_data=meta_data)

        if self.rgb_encoder is not None:
            patch_features = self.rgb_encoder(
                rgb,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_extra=prompt_extra,
                meta_data=meta_data,
            )
        else:
            patch_features = rgb

        if self.fuse_encoder is not None:
            patch_features = self.fuse_encoder(
                patch_features,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_extra=prompt_extra,
                scale=scale,
                intrinsics=intrinsics,
                w2c=w2c,
                meta_data=meta_data,
            )

        return patch_features, prompt_depth

    def forward(
        self,
        rgb,
        prompt_depth=None,
        scale=None,
        w2c=None,
        intrinsics=None,
        ray_directions=None,
        meta_data=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        patch_features, prompt_depth = self.aggregator(rgb=rgb, prompt_depth=prompt_depth, scale=scale, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, meta_data=meta_data)

        results = dict()

        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                prompt_depth=prompt_depth,
                return_dict=True,
                meta_data=meta_data,
            )
            results.update(depth_results)

        if self.normal_head is not None:
            normal_results = self.normal_head(
                patch_features,
                meta_data=meta_data,
            )
            results.update(normal_results)

        return results


class MVBase(Base):
    def __init__(
        self,
        camera_encoder=None,
        ray_in_world_encoder=None,
        glb_points_head=None,
        camera_head=None,
        track_head=None,
        geometric_prob=None,
        cam_prob=None,
        depth_prob=None,
        **kwargs,
    ):
        super(MVBase, self).__init__(**kwargs)

        self.camera_encoder = self._instantiate_and_register(camera_encoder, "camera_encoder")
        self.ray_in_world_encoder = self._instantiate_and_register(ray_in_world_encoder, "ray_in_world_encoder")

        # MV Geometric Head
        self.glb_points_head = self._instantiate_and_register(glb_points_head, "glb_points_head")
        self.camera_head = self._instantiate_and_register(camera_head, "camera_head")
        self.track_head = self._instantiate_and_register(track_head, "track_head")

        # geometric input prob
        self.geometric_prob = geometric_prob
        self.cam_prob = cam_prob
        self.depth_prob = depth_prob

    def _modify_cam_token(self, cam_token, b, n, device, meta_data):
        """Hook for subclasses to augment the camera token (e.g., add time embeddings).

        Args:
            cam_token: [B, N, C] or None - the camera token from camera_encoder
            b: batch size
            n: number of frames/views
            device: torch device
            meta_data: metadata dict
        Returns:
            Modified cam_token, same shape as input (or a new tensor if cam_token was None)
        """
        return cam_token

    def _get_time_token(self, b, n, device, meta_data):
        """Hook for subclasses to produce a per-frame time token for concat injection.

        Returns a tensor of shape [B*N, 1, C] that will be concatenated as an
        independent token inside the fuse_encoder (MoVieS-style), or None to skip.
        """
        return None

    def aggregator(self, rgb, scale, prompt_depth, intrinsics, ray_directions, w2c, c2w, ray_world, rgb_mask, meta_data):
        if rgb.ndim == 5:
            frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]

            b, n, c, h, w = rgb.shape
            assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

            rgb = rgb.view(b * n, c, h, w)

            if prompt_depth is not None:
                prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

            if scale is not None:
                scale = scale.view(b * n, *scale.shape[-3:])

            if intrinsics is not None:
                intrinsics = intrinsics.view(b * n, 3, 3)

            if w2c is not None:
                w2c = w2c.view(b * n, 4, 4)

            if c2w is not None:
                c2w = c2w.view(b * n, 4, 4)

            if ray_directions is not None:
                ray_directions = ray_directions.view(b * n, *ray_directions.shape[-3:])

            if ray_world is not None:
                ray_world = ray_world.view(b * n, *ray_world.shape[-3:])

            if rgb_mask is not None:
                rgb_mask = rgb_mask.view(b * n, *rgb_mask.shape[-3:])
        else:
            n = 1
            b, c, h, w = rgb.shape

        cam_token = prompt_ray = prompt_ray_in_world = prompt_extra = None

        if self.training and self.geometric_prob is not None and self.geometric_prob < 1.0:
            geometric_input_mask = torch.rand(b, device=rgb.device) <= self.geometric_prob
            geometric_input_mask = geometric_input_mask[:, None].repeat(1, n)
        else:
            geometric_input_mask = None

        if self.extra_encoder is not None:
            prompt_extra = self.extra_encoder(intrinsics=intrinsics, w2c=w2c, c2w=c2w, scale=scale, meta_data=meta_data)

        if self.camera_encoder is not None:
            cam_token = self.camera_encoder(intrinsics=intrinsics, w2c=w2c, c2w=c2w, scale=scale, meta_data=meta_data)
            if self.training and self.cam_prob is not None and self.cam_prob < 1.0:
                cam_input_mask = torch.rand(b, device=rgb.device) <= self.cam_prob
                cam_input_mask = cam_input_mask[:, None].repeat(1, n)
                if geometric_input_mask is not None:
                    cam_input_mask *= geometric_input_mask
                cam_token *= cam_input_mask.unsqueeze(-1).float()
            elif self.training and geometric_input_mask is not None:
                cam_token *= geometric_input_mask.unsqueeze(-1).float()

        # Hook: allows subclasses to augment cam_token (e.g., add time embeddings, add-style)
        cam_token = self._modify_cam_token(cam_token, b, n, rgb.device, meta_data)

        # Hook: allows subclasses to produce a concat-style time token [B*N, 1, C]
        time_token = self._get_time_token(b, n, rgb.device, meta_data)

        if self.ray_encoder is not None and intrinsics is not None:
            prompt_ray = self.ray_encoder(ray_directions=ray_directions, intrinsics=intrinsics, w2c=w2c, c2w=c2w, rgb_mask=rgb_mask, meta_data=meta_data)

        if self.ray_in_world_encoder is not None:
            prompt_ray_in_world = self.ray_in_world_encoder(ray_directions=ray_world, intrinsics=intrinsics, w2c=w2c, c2w=c2w, rgb_mask=rgb_mask, meta_data=meta_data)

        if self.depth_encoder is not None and prompt_depth is not None:
            prompt_depth = self.depth_encoder(prompt_depth, prompt_ray=prompt_ray, prompt_ray_in_world=prompt_ray_in_world, prompt_extra=prompt_extra, meta_data=meta_data)
            if self.training and self.depth_prob is not None and self.depth_prob < 1.0:
                depth_input_mask = torch.rand(b, device=rgb.device) <= self.depth_prob
                depth_input_mask = depth_input_mask[:, None].repeat(1, n)
                if geometric_input_mask is not None:
                    depth_input_mask *= geometric_input_mask
                if prompt_depth.ndim == 3:
                    prompt_depth *= depth_input_mask.view(-1, 1, 1).float()
                else:
                    prompt_depth *= depth_input_mask.view(-1, 1, 1, 1).float()
            elif self.training and geometric_input_mask is not None:
                if prompt_depth.ndim == 3:
                    prompt_depth *= depth_input_mask.view(-1, 1, 1).float()
                else:
                    prompt_depth *= geometric_input_mask.view(-1, 1, 1, 1).float()

        if self.rgb_encoder is not None:
            patch_features = self.rgb_encoder(
                rgb,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_ray_in_world=prompt_ray_in_world,
                prompt_extra=prompt_extra,
                meta_data=meta_data,
            )
        else:
            patch_features = rgb.view(b, n, c, h, w)

        if self.fuse_encoder is not None:
            patch_features, pos, patch_start_idx = self.fuse_encoder(
                patch_features,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_ray_in_world=prompt_ray_in_world,
                prompt_extra=prompt_extra,
                scale=scale,
                intrinsics=intrinsics,
                w2c=w2c,
                c2w=c2w,
                cam_token=cam_token,
                time_token=time_token,
                meta_data=meta_data,
            )
        else:
            pos = patch_start_idx = None

        return patch_features, pos, patch_start_idx, prompt_depth

    def forward(
        self,
        rgb,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        query_points=None,
        rgb_mask=None,
        meta_data=None,
        return_features=False,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world, rgb_mask=rgb_mask, meta_data=meta_data
        )

        results = dict()

        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                prompt_depth=prompt_depth,
                meta_data=meta_data,
            )
            for key, val in depth_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        if self.glb_points_head is not None:
            glb_depth_results = self.glb_points_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in glb_depth_results.items():
                results["global_" + key] = val.view(b, n, *val.shape[-3:])

        if self.camera_head is not None and patch_start_idx is not None and patch_start_idx > 0:
            if isinstance(patch_features, (list, tuple)):
                camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
            else:
                camera_tokens = [patch_features[:, :, :patch_start_idx]]

            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                meta_data=meta_data,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        if self.normal_head is not None:
            normal_results = self.normal_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in normal_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        if return_features:
            results["patch_features"] = patch_features

        return results


class MVBase2(MVBase):
    def __init__(
        self,
        decoder_fp32=False,
        **kwargs,
    ):
        super(MVBase2, self).__init__(**kwargs)

        self.decoder_fp32 = decoder_fp32

    def decoder(self, b, n, patch_features, pos, patch_start_idx, prompt_depth, query_points, meta_data):
        results = dict()

        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                prompt_depth=prompt_depth,
                meta_data=meta_data,
            )
            for key, val in depth_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        if self.glb_points_head is not None:
            glb_depth_results = self.glb_points_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in glb_depth_results.items():
                results["global_" + key] = val.view(b, n, *val.shape[-3:])

        if self.camera_head is not None and patch_start_idx is not None and patch_start_idx > 0:
            if isinstance(patch_features, (list, tuple)):
                camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
            else:
                camera_tokens = [patch_features[:, :, :patch_start_idx]]

            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                meta_data=meta_data,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        if self.normal_head is not None:
            normal_results = self.normal_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in normal_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        return results

    def forward(
        self,
        rgb,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        query_points=None,
        rgb_mask=None,
        meta_data=None,
        return_features=False,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world, rgb_mask=rgb_mask, meta_data=meta_data
        )

        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                results = self.decoder(
                    b=b,
                    n=n,
                    patch_features=patch_features,
                    pos=pos,
                    patch_start_idx=patch_start_idx,
                    prompt_depth=prompt_depth,
                    query_points=query_points,
                    meta_data=meta_data,
                )
        else:
            results = self.decoder(
                b=b,
                n=n,
                patch_features=patch_features,
                pos=pos,
                patch_start_idx=patch_start_idx,
                prompt_depth=prompt_depth,
                query_points=query_points,
                meta_data=meta_data,
            )

        if return_features:
            results["patch_features"] = patch_features

        return results


class MemoryEfficientMV(MVBase):
    def __init__(self, **kwargs):
        super(MemoryEfficientMV, self).__init__(**kwargs)

    def aggregator(self, rgb, scale, prompt_depth, intrinsics, ray_directions, w2c, c2w, ray_world, rgb_mask, meta_data):
        device = rgb.device

        if rgb.ndim == 5:
            frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]

            b, n, c, h, w = rgb.shape
            assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

            rgb = rgb.view(b * n, c, h, w)

            if prompt_depth is not None:
                prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

            if scale is not None:
                scale = scale.view(b * n, *scale.shape[-3:])

            if intrinsics is not None:
                intrinsics = intrinsics.view(b * n, 3, 3)

            if w2c is not None:
                w2c = w2c.view(b * n, 4, 4)

            if c2w is not None:
                c2w = c2w.view(b * n, 4, 4)

            if ray_directions is not None:
                ray_directions = ray_directions.view(b * n, *ray_directions.shape[-3:])

            if ray_world is not None:
                ray_world = ray_world.view(b * n, *ray_world.shape[-3:])
            
            if rgb_mask is not None:
                rgb_mask = rgb_mask.view(b * n, *rgb_mask.shape[-3:])
        else:
            n = 1
            b, c, h, w = rgb.shape

        cam_token = prompt_ray = prompt_ray_in_world = prompt_extra = None

        if self.training and self.geometric_prob is not None and self.geometric_prob < 1.0:
            geometric_input_mask = torch.rand(b, device=rgb.device) <= self.geometric_prob
            geometric_input_mask = geometric_input_mask[:, None].repeat(1, n)
        else:
            geometric_input_mask = None

        if self.extra_encoder is not None:
            prompt_extra = self.extra_encoder(intrinsics=intrinsics, w2c=w2c, c2w=c2w, scale=scale, meta_data=meta_data)
            torch.cuda.empty_cache()

        if self.camera_encoder is not None:
            cam_token = self.camera_encoder(intrinsics=intrinsics, w2c=w2c, c2w=c2w, scale=scale, meta_data=meta_data)
            if self.training and self.cam_prob is not None and self.cam_prob < 1.0:
                cam_input_mask = torch.rand(b, device=rgb.device) <= self.cam_prob
                cam_input_mask = cam_input_mask[:, None].repeat(1, n)
                if geometric_input_mask is not None:
                    cam_input_mask *= geometric_input_mask
                cam_token *= cam_input_mask.unsqueeze(-1).float()
            elif self.training and geometric_input_mask is not None:
                cam_token *= geometric_input_mask.unsqueeze(-1).float()
            torch.cuda.empty_cache()

        if self.ray_encoder is not None and intrinsics is not None:
            prompt_ray = self.ray_encoder(ray_directions=ray_directions, intrinsics=intrinsics, w2c=w2c, c2w=c2w, rgb_mask=rgb_mask, meta_data=meta_data)
            torch.cuda.empty_cache()

        if self.ray_in_world_encoder is not None:
            prompt_ray_in_world = self.ray_in_world_encoder(ray_directions=ray_world, intrinsics=intrinsics, w2c=w2c, c2w=c2w, rgb_mask=rgb_mask, meta_data=meta_data)
            torch.cuda.empty_cache()

        if self.depth_encoder is not None and prompt_depth is not None:
            prompt_depth = self.depth_encoder(prompt_depth, prompt_ray=prompt_ray, prompt_ray_in_world=prompt_ray_in_world, prompt_extra=prompt_extra, meta_data=meta_data)
            torch.cuda.empty_cache()

        if self.rgb_encoder is not None:
            patch_features = self.rgb_encoder(
                rgb,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_ray_in_world=prompt_ray_in_world,
                prompt_extra=prompt_extra,
                meta_data=meta_data,
            )
        else:
            patch_features = rgb.view(b, n, c, h, w)

        # self.rgb_encoder = self.rgb_encoder.cpu()
        del rgb
        torch.cuda.empty_cache()

        if self.fuse_encoder is not None:
            patch_features, pos, patch_start_idx = self.fuse_encoder(
                patch_features,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_ray_in_world=prompt_ray_in_world,
                prompt_extra=prompt_extra,
                scale=scale,
                intrinsics=intrinsics,
                w2c=w2c,
                c2w=c2w,
                cam_token=cam_token,
                meta_data=meta_data,
                memory_efficient_infer=True,
            )

            # self.fuse_encoder = self.fuse_encoder.cpu()
            torch.cuda.empty_cache()

            patch_features = [tensor.to(device) for tensor in patch_features]
        else:
            pos = patch_start_idx = None

        return patch_features, pos, patch_start_idx, prompt_depth

    def forward(
        self,
        rgb,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        query_points=None,
        rgb_mask=None,
        meta_data=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        b, n, c, h, w = rgb.shape

        if meta_data is not None and "scene" in meta_data["data_info"]:
            scene = meta_data["data_info"]["scene"][0][0]
            logging.info(f"[Memory Efficient Infer], scene: {scene}")

        torch.cuda.synchronize()
        start = time.time()

        logging.info(f"[Memory Efficient Infer], image, {rgb.shape}!")
        logging.info("[Memory Efficient Infer], aggregator!")

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world, rgb_mask=rgb_mask, meta_data=meta_data
        )

        torch.cuda.synchronize()
        end = time.time()
        logging.info(f"[Memory Efficient Infer], aggregator time: {end - start:.3f}|{(end - start) / n:.3f}")

        torch.cuda.empty_cache()
        logging.info("[Memory Efficient Infer], head!")

        results = dict()

        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in depth_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        if self.glb_points_head is not None:
            glb_depth_results = self.glb_points_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in glb_depth_results.items():
                results["global_" + key] = val.view(b, n, *val.shape[-3:])

        if self.camera_head is not None and patch_start_idx is not None and patch_start_idx > 0:
            if isinstance(patch_features, (list, tuple)):
                camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
            else:
                camera_tokens = [patch_features[:, :, :patch_start_idx]]

            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                meta_data=meta_data,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        if self.normal_head is not None:
            normal_results = self.normal_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in normal_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        torch.cuda.synchronize()
        end2 = time.time()
        logging.info(f"[Memory Efficient Infer], head time: {end2 - end:.3f}|{(end2 - end) / n:.3f}")
        logging.info(f"[Memory Efficient Infer], total time: {end2 - start:.3f}|{(end2 - start) / n:.3f}")

        return results


class SVFlowBase(MVBase):
    def __init__(self, **kwargs):
        super(SVFlowBase, self).__init__(**kwargs)

    def aggregator(self, rgb, scale, prompt_depth, intrinsics, ray_directions, w2c, ray_world, meta_data):
        if rgb.ndim == 5:
            frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]

            b, n, c, h, w = rgb.shape
            assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

            rgb = rgb.view(b * n, c, h, w)

            if prompt_depth is not None:
                prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

            if scale is not None:
                scale = scale.view(b * n, *scale.shape[-3:])

            if intrinsics is not None:
                intrinsics = intrinsics.view(b * n, 3, 3)

            if w2c is not None:
                w2c = w2c.view(b * n, 4, 4)

            if ray_directions is not None:
                ray_directions = ray_directions.view(b * n, *ray_directions.shape[-3:])

            if ray_world is not None:
                ray_world = ray_world.view(b * n, *ray_world.shape[-3:])

        prompt_ray = prompt_ray_in_world = prompt_extra = None

        if self.extra_encoder is not None:
            prompt_extra = self.extra_encoder(intrinsics=intrinsics, w2c=w2c, scale=scale, meta_data=meta_data)

        if self.ray_encoder is not None and intrinsics is not None:
            prompt_ray = self.ray_encoder(ray_directions=ray_directions, intrinsics=intrinsics, w2c=w2c, meta_data=meta_data)

        if self.ray_in_world_encoder is not None:
            prompt_ray_in_world = self.ray_in_world_encoder(ray_directions=ray_world, intrinsics=intrinsics, w2c=w2c, meta_data=meta_data)

        if self.depth_encoder is not None and prompt_depth is not None:
            prompt_depth = self.depth_encoder(prompt_depth, prompt_ray=prompt_ray, prompt_ray_in_world=prompt_ray_in_world, prompt_extra=prompt_extra, meta_data=meta_data)

        patch_features = self.rgb_encoder(
            rgb,
            prompt_depth=prompt_depth,
            prompt_ray=prompt_ray,
            prompt_ray_in_world=prompt_ray_in_world,
            prompt_extra=prompt_extra,
            meta_data=meta_data,
        )

        if self.fuse_encoder is not None:
            patch_features, pos, patch_start_idx = self.fuse_encoder(
                patch_features,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_ray_in_world=prompt_ray_in_world,
                prompt_extra=prompt_extra,
                scale=scale,
                intrinsics=intrinsics,
                w2c=w2c,
                meta_data=meta_data,
            )
        else:
            pos = patch_start_idx = None

        return patch_features, pos, patch_start_idx, prompt_depth

    def forward(
        self,
        rgb,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        ray_world=None,
        query_points=None,
        meta_data=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        b, n, c, h, w = rgb.shape

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, ray_world=ray_world, meta_data=meta_data
        )

        results = dict()

        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                prompt_depth=prompt_depth,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in depth_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        if self.glb_points_head is not None:
            glb_depth_results = self.glb_points_head(
                patch_features,
                prompt_depth=prompt_depth,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in glb_depth_results.items():
                results["global_" + key] = val.view(b, n, *val.shape[-3:])

        if self.camera_head is not None and patch_start_idx is not None and patch_start_idx > 0:
            if isinstance(patch_features, (list, tuple)):
                camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
            else:
                camera_tokens = [patch_features[:, :, :patch_start_idx]]

            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                meta_data=meta_data,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        if self.normal_head is not None:
            normal_results = self.normal_head(
                patch_features,
                prompt_depth=prompt_depth,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in normal_results.items():
                results[key] = val.view(b, n, *val.shape[1:])

        return results
