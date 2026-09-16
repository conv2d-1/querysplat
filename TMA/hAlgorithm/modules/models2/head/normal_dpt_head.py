import logging

import torch

from hAlgorithm.modules.models.vggt.heads.dpt_head import DPTHead, activate_head, custom_interpolate


class NormalDPTHead(DPTHead):
    def __init__(
        self,
        patch_size=14,
        chunk_size=0,
        cat_sv_features=False,
        pretrain=None,
        **kwargs,
    ):
        kwargs["conf_activation"] = None
        super(NormalDPTHead, self).__init__(**kwargs)

        self.patch_size = patch_size
        self.chunk_size = chunk_size
        self.cat_sv_features = cat_sv_features
        self.pretrain = pretrain

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"VGGTDPTHead, load pretrain {self.pretrain}")

    def _forward_impl(
        self,
        aggregated_tokens_list,
        sv_features=None,
        patch_h=None,
        patch_w=None,
        num_views=None,
        patch_start_idx=None,
        view_start_idx=None,
        view_end_idx=None,
        meta_data=None,
    ):
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            view_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        H, W = meta_data["input_height"][0], meta_data["input_width"][0]

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx]

            if patch_start_idx is not None:
                if x.ndim == 4:
                    x = x[:, :, patch_start_idx:]
                elif x.ndim == 3:
                    x = x[:, patch_start_idx:]

            if view_start_idx is not None and view_end_idx is not None:
                x = x[:, view_start_idx:view_end_idx]
            
            if sv_features is not None and self.cat_sv_features:
                if isinstance(sv_features, (list, tuple)):
                    sv_features = sv_features[-1]
                sv_features = sv_features.view(-1, num_views, *sv_features.shape[-2:])
                if view_start_idx is not None and view_end_idx is not None:
                    sv_features = sv_features[:, view_start_idx:view_end_idx]
                x = torch.cat([x, sv_features], dim=-1)

            if x.ndim == 4:
                x = x.reshape(-1, x.shape[-2], x.shape[-1])

            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)

            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

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

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        preds = self.scratch.output_conv2(out)
        return_dict = dict(normal=preds)

        return return_dict

    def forward(
        self,
        features,
        sv_features=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        patch_start_idx=None,
        **kwargs,
    ):
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            view_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        aggregated_tokens_list = features

        S = aggregated_tokens_list[0].shape[1] if aggregated_tokens_list[0].ndim == 4 else None

        if self.training or self.chunk_size == 0:
            return self._forward_impl(
                aggregated_tokens_list,
                sv_features=sv_features,
                patch_h=patch_h,
                patch_w=patch_w,
                num_views=S,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )

        total_datas = {}
        for view_start_idx in range(0, S, self.chunk_size):
            view_end_idx = min(view_start_idx + self.chunk_size, S)

            outputs = self._forward_impl(
                aggregated_tokens_list,
                sv_features=sv_features,
                patch_h=patch_h,
                patch_w=patch_w,
                num_views=S,
                patch_start_idx=patch_start_idx,
                view_start_idx=view_start_idx,
                view_end_idx=view_end_idx,
                meta_data=meta_data,
            )

            for key, val in outputs.items():
                if key not in total_datas:
                    total_datas[key] = []
                total_datas[key].append(val)

        if isinstance(total_datas, list):
            total_datas = torch.stack(total_datas, dim=1)
        else:
            for key, val in total_datas.items():
                total_datas[key] = torch.stack(val, dim=1)

        return total_datas
