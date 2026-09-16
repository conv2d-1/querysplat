import logging

import torch

from hAlgorithm.modules.models.vggt.heads.dpt_head import DPTHead, activate_head, custom_interpolate


class VGGTDPTHead(DPTHead):
    def __init__(
        self,
        patch_size=14,
        pretrain=None,
        return_features=False,
        chunk_size=4,
        **kwargs,
    ):
        super(VGGTDPTHead, self).__init__(**kwargs)

        self.patch_size = patch_size
        self.pretrain = pretrain
        self.return_features = return_features
        self.chunk_size = chunk_size

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"VGGTDPTHead, load pretrain {self.pretrain}")

    def _forward_impl(
        self,
        aggregated_tokens_list,
        patch_h,
        patch_w,
        B,
        S,
        H,
        W,
        frames_start_idx=None,
        frames_end_idx=None,
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
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """

        if frames_start_idx is not None and frames_end_idx is not None:
            S = frames_end_idx - frames_start_idx

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx]

            if patch_start_idx is not None:
                if x.ndim == 4:
                    x = x[:, :, patch_start_idx:]
                elif x.ndim == 3:
                    x = x[:, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

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

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        if self.return_features:
            return_features = out
        else:
            return_features = None

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(
            out, activation=self.activation, conf_activation=self.conf_activation
        )

        preds = preds.view(B, S, *preds.shape[1:])
        if preds.shape[-1] == 1:
            preds = preds.squeeze(-1).unsqueeze(2)
        else:
            preds = preds.permute(0, 1, 4, 2, 3)

        conf = conf.view(B, S, *conf.shape[1:]).unsqueeze(2)

        if return_features is not None:
            return_features = return_features.view(B, S, *return_features.shape[1:])

        return preds, conf, return_features

    def forward(
        self,
        features,
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
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """

        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        aggregated_tokens_list = features

        if aggregated_tokens_list[0].ndim == 4:
            B, S = aggregated_tokens_list[0].shape[0:2]
        H, W = meta_data["input_height"][0], meta_data["input_width"][0]

        if not self.training:
            preds_list = []
            conf_list = []
            return_features_list = []
            for frames_start_idx in range(0, S, self.chunk_size):
                frames_end_idx = min(frames_start_idx + self.chunk_size, S)

                preds, conf, return_features = self._forward_impl(
                    aggregated_tokens_list,
                    patch_h,
                    patch_w,
                    B,
                    S,
                    H,
                    W,
                    frames_start_idx=frames_start_idx,
                    frames_end_idx=frames_end_idx,
                    patch_start_idx=patch_start_idx,
                )

                preds_list.append(preds)
                conf_list.append(conf)
                if return_features is not None:
                    return_features_list.append(return_features)

            return_dict = dict(
                pointmap=torch.cat(preds_list, dim=1).reshape(B * S, *preds_list[0].shape[2:]),
                confidence=torch.cat(conf_list, dim=1).reshape(B * S, *conf_list[0].shape[2:]),
                features=(
                    torch.cat(return_features_list, dim=1).reshape(
                        B * S, *return_features_list[0].shape[2:]
                    )
                    if return_features is not None and len(return_features) > 0
                    else None
                ),
            )

        else:
            preds, conf, return_features = self._forward_impl(
                aggregated_tokens_list,
                patch_h,
                patch_w,
                B,
                S,
                H,
                W,
                patch_start_idx=patch_start_idx,
            )

            return_dict = dict(
                pointmap=preds.reshape(B * S, *preds_list[0].shape[2:]),
                confidence=conf.reshape(B * S, *conf_list[0].shape[2:]),
                features=(
                    return_features.reshape(B * S, *return_features_list[0].shape[2:])
                    if return_features is not None
                    else None
                ),
            )

        return return_dict
