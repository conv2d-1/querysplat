import torch
import torch.nn as nn

from hAlgorithm.modules.utils.ray import compute_rays
from hAlgorithm.utils import instantiate_from_config


class RefineModel(nn.Module):
    def __init__(
        self,
        rgb_encoder=None,
        prompt_encoder=None,
        decoder=None,
        depth_head=None,
        extrinsics_c2w=False,
        prompt_with_ray=False,
        prompt_with_direct=False,
        freeze_modules=[],
        **kwargs,
    ):
        super(RefineModel, self).__init__(**kwargs)

        # Store configurations and parameters
        self.freeze_modules = freeze_modules
        self.module_names = []

        self.extrinsics_c2w = extrinsics_c2w
        self.prompt_with_ray = prompt_with_ray
        self.prompt_with_direct = prompt_with_direct

        self.rgb_encoder = instantiate_from_config(rgb_encoder)
        if self.rgb_encoder is not None:
            self.module_names.append("rgb_encoder")

        self.prompt_encoder = instantiate_from_config(prompt_encoder)
        if self.prompt_encoder is not None:
            self.module_names.append("prompt_encoder")

        self.decoder = instantiate_from_config(decoder)
        if self.decoder is not None:
            self.module_names.append("decoder")

        self.depth_head = instantiate_from_config(depth_head)
        if self.depth_head is not None:
            self.module_names.append("depth_head")

    def freeze(self):
        for module_name in self.freeze_modules:
            if module_name in self.module_names:
                module = getattr(self, module_name)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(
        self,
        rgb=None,
        prompt_depth=None,
        extrinsics=None,
        intrinsics=None,
        with_freeze=False,
        meta_data=None,
        **kwargs,
    ):
        # Optionally freeze modules if specified
        if with_freeze:
            self.freeze()

        # Extract frame number and view number from metadata
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        # Unpack input dimensions for batch size (b), number of frames (n), channels (c), height (h), width (w)
        b, n, c, h, w = rgb.shape
        assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

        # Reshape input tensor for processing
        rgb = rgb.view(b * n, c, h, w)

        # If a depth prompt is provided, reshape it as well
        if prompt_depth is not None:
            prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

        if self.prompt_with_ray:
            ray_o, ray_d = compute_rays(
                c2w=extrinsics if self.extrinsics_c2w else extrinsics.inverse(),
                fxfycxcy=intrinsics[:, :, [0, 1, 0, 1], [0, 1, 2, 2]],
                h=h,
                w=w,
                device=rgb.device,
            )
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            rays = torch.cat([o_cross_d, ray_d], dim=2).to(rgb.dtype)
            rays = rays.view(b * n, *rays.shape[-3:])
            prompt_depth = torch.cat([prompt_depth, rays], dim=1)
        elif self.prompt_with_direct:
            _, ray_d = compute_rays(
                c2w=None,
                fxfycxcy=intrinsics[:, :, [0, 1, 0, 1], [0, 1, 2, 2]],
                h=h,
                w=w,
                device=rgb.device,
            )
            ray_d = ray_d.view(b * n, *ray_d.shape[-3:])
            prompt_depth = torch.cat([prompt_depth, ray_d], dim=1)

        prompt_features = rgb_features = None
        results = dict()

        # If a prompt encoder exists and a depth prompt is provided, encode the depth prompt
        if self.prompt_encoder is not None and prompt_depth is not None:
            prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)

        # Encode RGB images using the RGB encoder
        if self.rgb_encoder is not None:
            rgb_features = self.rgb_encoder(rgb, condition=prompt_features, meta_data=meta_data)

        # Further process features using the multi-view decoder
        if self.decoder is not None:
            patch_features = self.decoder(
                rgb_features,
                prompt_features=prompt_features,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
        else:
            patch_features = rgb_features

        # Process multi-view depth results
        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            results["depth"] = depth_results.pop("pointmap")
            results["depth"] = results["depth"].view(b, n, *results["depth"].shape[-3:])
            
            results["depth_confidence"] = depth_results.pop("confidence")
            results["depth_confidence"] = results["depth_confidence"].view(
                b, n, *results["depth_confidence"].shape[-3:]
            )

        return results

