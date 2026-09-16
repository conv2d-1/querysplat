import torch.nn as nn
from einops import rearrange

from hAlgorithm.modules.models.gaussiansv2.gaussian_adapter import GaussianAdapter
from hAlgorithm.modules.models.gaussiansv2.gaussian_adapter_voxelize import GaussianAdapterVoxelize
from hAlgorithm.modules.models.vggt.heads.dpt_head import DPTHead, custom_interpolate
from hAlgorithm.modules.utils.gaussians.projection import sample_image_grid


class GaussianHead(DPTHead):
    def __init__(
        self,
        features=256,
        cfg=None,
        voxelization=False,
        **kwargs,
    ):
        super(GaussianHead, self).__init__(features=features, feature_only=True, **kwargs)

        if voxelization:
            self.gaussian_adapter = GaussianAdapterVoxelize(cfg)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg)

        num_gaussian_parameters = self.gaussian_adapter.d_in + 2
        num_gaussian_parameters += 1  # predict opacity
        num_gaussian_parameters += 1  # predict mask
        if self.gaussian_adapter.offset:
            num_gaussian_parameters += 3  # prpedict xyz offset
        self.output_dim = num_gaussian_parameters

        self.head = nn.Sequential(
            nn.Conv2d(features, num_gaussian_parameters * 2, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(num_gaussian_parameters * 2, num_gaussian_parameters, 1, 1, 0, bias=True),
        )

    def scratch_forward_return_refine_features(self, features):
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        refine_features = []
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        refine_features.append(out)

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        refine_features.append(out)

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        refine_features.append(out)

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        refine_features.append(out)

        out = self.scratch.output_conv1(out)
        refine_features.append(out)

        return out, refine_features

    def forward(
        self, features, pos, patch_start_idx, depth, extrinsics, intrinsics, meta_data, **kwargs
    ):
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """

        H, W = meta_data["input_height"][0], meta_data["input_width"][0]
        patch_w = W // self.patch_size
        patch_h = H // self.patch_size

        aggregated_tokens_list = features

        B, N = aggregated_tokens_list[0].shape[0:2]

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx]

            if patch_start_idx is not None:
                if x.ndim == 4:
                    x = x[:, :, patch_start_idx:]
                elif x.ndim == 3:
                    x = x[:, patch_start_idx:]

            # x = x.view(BS, -1, x.shape[-1])
            if x.ndim == 4:
                x = x.reshape(-1, x.shape[-2], x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)

        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (
                int(patch_h * self.patch_size / self.down_ratio),
                int(patch_w * self.patch_size / self.down_ratio),
            ),
            mode="bilinear",
            align_corners=True,
        )

        gaussians = self.head(out)  # [B*V, C, H, W]

        # [B, V, C, H, W]
        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=B, v=N)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(gaussians, "b v c h w -> b v (h w) c")  # [B, V, H*W, 86]
        opacities = raw_gaussians[..., :1].unsqueeze(-1)  # [B, V, H*W, 1, 1]
        raw_gaussians = raw_gaussians[..., 1:]

        xy_ray, _ = sample_image_grid((H, W), out.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")  # [H*W, 1, 2]
        # [B*num_depths, V, H*W, 1, 84]
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=1,  # num_surfaces
        )
        xy_ray = xy_ray[None, None].repeat(B, N, 1, 1, 1)  # [B, V, H*W, 1, 2]

        gaussians = self.gaussian_adapter(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            rearrange(depth, "b v 1 h w -> b v (h w) () ()", b=B, v=N),
            opacities,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
            (H, W),
            input_images=None,
        )

        return gaussians
