import logging

import torch
import torch.nn as nn

from hAlgorithm.utils import instantiate_from_config


class Base(nn.Module):
    def __init__(
        self,
        rgb_encoder,
        depth_encoder=None,
        ray_encoder=None,
        extra_encoder=None,
        fuse_encoder=None,
        depth_head=None,
        scale_head=None,
        normal_head=None,
        freeze_modules=None,
        input_drop_config=None,
        **kwargs,
    ):
        super(Base, self).__init__()

        # Store module names for freeze & introspection
        self.module_names = []
        self.freeze_modules = freeze_modules if freeze_modules is not None else []

        #drop module
        self.input_drop_config = input_drop_config

        # Encoder
        self.rgb_encoder = self._instantiate_and_register(rgb_encoder, "rgb_encoder")
        self.depth_encoder = self._instantiate_and_register(depth_encoder, "depth_encoder")
        self.ray_encoder = self._instantiate_and_register(ray_encoder, "ray_encoder")
        self.extra_encoder = self._instantiate_and_register(extra_encoder, "extra_encoder")
        self.fuse_encoder = self._instantiate_and_register(fuse_encoder, "fuse_encoder")

        # Geometric Head
        self.depth_head = self._instantiate_and_register(depth_head, "depth_head")
        self.scale_head = self._instantiate_and_register(scale_head, "scale_head")
        self.normal_head = self._instantiate_and_register(normal_head, "normal_head")

    def _instantiate_and_register(self, config, name):
        """Helper to instantiate module and register its name."""
        module = instantiate_from_config(config)
        if module is not None:
            self.module_names.append(name)
        return module

    def freeze(self):
        """Freeze specified modules."""
        for module_name in self.freeze_modules:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                if module is not None:
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                logging.warning(f"Module {module_name} not found for freezing.")

    def forward_aggregator(self, rgb, prompt_depth, prompt_scale, intrinsics, w2c, meta_data):
        if rgb.ndim == 5:
            frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]

            b, n, c, h, w = rgb.shape
            assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

            rgb = rgb.view(b * n, c, h, w)

            if prompt_depth is not None:
                prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[2:])

            if prompt_scale is not None:
                prompt_scale = prompt_scale.view(b * n, *prompt_scale.shape[2:])
            
            if intrinsics is not None:
                intrinsics = intrinsics.view(b * n, *intrinsics.shape[2:])

            if w2c is not None:
                w2c = w2c.view(b * n, *w2c.shape[2:])

        prompt_ray = prompt_extra = None

        if self.extra_encoder is not None:
            prompt_extra = self.extra_encoder(
                intrinsics=intrinsics, w2c=w2c, prompt_scale=prompt_scale, meta_data=meta_data
            )

        if self.ray_encoder is not None and intrinsics is not None:
            prompt_ray = self.ray_encoder(intrinsics=intrinsics, w2c=w2c, meta_data=meta_data)

        if self.depth_encoder is not None and prompt_depth is not None:
            prompt_depth = self.depth_encoder(
                prompt_depth, prompt_ray=prompt_ray, prompt_extra=prompt_extra, meta_data=meta_data
            )

        # patch_features could be tokens or list of tokens
        patch_features = self.rgb_encoder(
            rgb,
            prompt_depth=prompt_depth,
            prompt_ray=prompt_ray,
            prompt_extra=prompt_extra,
            meta_data=meta_data,
        )

        if self.fuse_encoder is not None:
            patch_features = self.fuse_encoder(
                patch_features,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_extra=prompt_extra,
                prompt_scale=prompt_scale,
                intrinsics=intrinsics,
                w2c=w2c,
                meta_data=meta_data,
            )

        return patch_features, prompt_depth, prompt_ray, prompt_extra

    def forward(
        self,
        rgb,
        prompt_depth=None,
        prompt_scale=None,
        w2c=None,
        intrinsics=None,
        with_freeze=False,
        meta_data=None,
        **kwargs,
    ):
        if with_freeze:
            self.freeze()

        patch_features, prompt_depth, prompt_ray, prompt_extra = self.forward_aggregator(
            rgb, prompt_depth, prompt_scale, intrinsics, w2c, meta_data
        )
        results = {}
        features = {}
        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                prompt_features=prompt_depth,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
                return_dino_features=False,
            )
            results["pointmap"] = depth_results["pointmap"]

            if "confidence" in depth_results:
                results["confidence"] = depth_results["confidence"]

            if "mask" in depth_results:
                results["mask"] = depth_results["mask"]

            if "normal" in depth_results:
                results["normal"] = depth_results["normal"]

            if "dino_features" in depth_results:
                features["dino_features"] = depth_results["dino_features"]

            if "refine_features" in depth_results:
                features["refine_features"] = depth_results["refine_features"]

        if self.scale_head is not None:
            scale_results = self.scale_head(
                patch_features=patch_features,
                prompt_features=prompt_depth,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            results["scale"] = scale_results["scale"]

        if self.normal_head is not None:
            normal_results = self.normal_head(
                patch_features,
                prompt_features=prompt_depth,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
                additiontal_features=features,
            )
            results["normal"] = normal_results["normal"]

        return results


class MVBase(Base):
    def __init__(
        self,
        glb_depth_head=None,
        camera_head=None,
        track_head=None,
        **kwargs,
    ):
        super(MVBase, self).__init__(**kwargs)

        # MV Geometric Head
        self.glb_depth_head = self._instantiate_and_register(glb_depth_head, "glb_depth_head")
        self.camera_head = self._instantiate_and_register(camera_head, "camera_head")
        self.track_head = self._instantiate_and_register(track_head, "track_head")

    def forward(
        self,
        rgb,
        prompt_depth=None,
        prompt_scale=None,
        w2c=None,
        intrinsics=None,
        with_freeze=False,
        meta_data=None,
        **kwargs,
    ):
        if with_freeze:
            self.freeze()

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            b, c, h, w = rgb.shape

        patch_features, prompt_depth, prompt_ray, prompt_extra, camera_tokens = (
            self.froward_aggregator(rgb, prompt_depth, prompt_scale, intrinsics, w2c, meta_data)
        )

        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            depth = depth_results["pointmap"]
            results["pointmap"] = depth.view(b, n, *depth.shape[-3:])

            confidence = depth_results["confidence"]
            results["confidence"] = confidence.view(b, n, *confidence.shape[-3:])

        if self.glb_depth_head is not None:
            glb_depth_results = self.glb_depth_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            glb_pointmap = glb_depth_results["pointmap"]
            results["mv_depth"] = glb_pointmap.view(b, n, *glb_pointmap.shape[-3:])

            glb_confidence = glb_depth_results["confidence"]
            results["mv_depth_confidence"] = glb_confidence.view(b, n, *glb_confidence.shape[-3:])

        if self.camera_head is not None:
            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                prompt_features=prompt_features,
                meta_data=meta_data,
                local_feature_maps=local_feature_maps,
                glb_feature_maps=glb_feature_maps,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        return results
