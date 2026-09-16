import torch

from .base import CombinedModel


class MFCombinedModel(CombinedModel):
    def __init__(
        self,
        rgb_encoder=None,
        prompt_encoder=None,
        decoder=None,
        head=None,
        low_mem=False,
        freeze_modules=[],
    ):
        super(MFCombinedModel, self).__init__(
            rgb_encoder=rgb_encoder,
            prompt_encoder=prompt_encoder,
            decoder=decoder,
            head=head,
            freeze_modules=freeze_modules,
        )
        self.low_mem = low_mem

    def preprocess_frames(self, x, prompt_depth, meta_data=None, grad_index=None):
        """
        A helper function to process frames either sequentially or in parallel based on memory constraints.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width).
        - prompt_depth (Tensor or None): Depth information for each frame.
        - grad_index (list): Whether to process frames without gradients.

        """
        b, n, c, h, w = x.shape

        if grad_index is None:
            grad_index = range(n)

        seq_mode = True
        if n == 1:
            seq_mode = False
        else:
            if len(grad_index) == 0:
                seq_mode = False
            if (not self.low_mem) and len(grad_index) == n:
                # low mem off and full grad
                seq_mode = False

        if seq_mode:
            rgb_features_list, prompt_features_list = [], []
            for frame_i in range(n):
                # if frame_i in grad_index:
                rgb, prompt = self._preprocess_frames_impl(
                    x[:, frame_i, ...],
                    prompt_depth[:, frame_i, ...],
                    meta_data=meta_data,
                    no_grad=(frame_i not in grad_index),
                )
                rgb_features_list.append(rgb)
                prompt_features_list.append(prompt)
            # organize b,n
            rgb_features_list = [
                [feat[i] for feat in rgb_features_list] for i in range(len(rgb_features_list[0]))
            ]
            if self.rgb_encoder.use_clstoken:
                rgb_features = [
                    (
                        torch.stack([f[0] for f in feat], dim=1),
                        torch.stack([f[1] for f in feat], dim=1),
                    )
                    for feat in rgb_features_list
                ]
            else:
                rgb_features = [torch.stack(feat, dim=1) for feat in rgb_features_list]
            prompt_features = torch.stack(prompt_features_list, dim=1)
        else:
            if len(grad_index) > 0:
                rgb_features, prompt_features = self._preprocess_frames_impl(
                    x, prompt_depth, meta_data=meta_data
                )
            else:
                with torch.no_grad():
                    rgb_features, prompt_features = self._preprocess_frames_impl(
                        x, prompt_depth, meta_data=meta_data
                    )

        return rgb_features, prompt_features

    def _preprocess_frames_impl(self, x, prompt_depth, meta_data=None, no_grad=False):
        """
        Implementation of frame processing logic.
        x: shape [b, n, c, h, w] or [b, c, h, w] image tensor
        prompt_depth: shape [b, n, c, h, w] or [b, c, h, w] pointmap tensor
        """
        assert x.ndim == prompt_depth.ndim
        if x.ndim == 4:
            if no_grad:
                with torch.no_grad():
                    return self.rgb_encoder(x, meta_data=meta_data), self.prompt_encoder(
                        prompt_depth, meta_data=meta_data
                    )
            else:
                return self.rgb_encoder(x, meta_data=meta_data), self.prompt_encoder(
                    prompt_depth, meta_data=meta_data
                )

        b, n, c, h, w = x.shape
        if no_grad:
            with torch.no_grad():
                rgb = self.rgb_encoder(
                    x.view(b * n, c, h, w),
                    meta_data=meta_data,
                )  # [(feat,cls)*4] or [feat*4]
                prompt = self.prompt_encoder(
                    prompt_depth.view(b * n, -1, h, w),
                    meta_data=meta_data,
                )
        else:
            rgb = self.rgb_encoder(
                x.view(b * n, c, h, w),
                meta_data=meta_data,
            )  # [(feat,cls)*4] or [feat*4]
            prompt = self.prompt_encoder(
                prompt_depth.view(b * n, -1, h, w),
                meta_data=meta_data,
            )
        # unpack b*n to b,n
        if self.rgb_encoder.use_clstoken:
            rgb = [(self.unpack_bn(feat[0], b, n), self.unpack_bn(feat[1], b, n)) for feat in rgb]
        else:
            rgb = [self.unpack_bn(feat, b, n) for feat in rgb]
        prompt = self.unpack_bn(prompt, b, n)

        return rgb, prompt

    def unpack_bn(self, bn_tensor, b, n):
        assert bn_tensor.shape[0] == b * n
        if n == 1:
            return bn_tensor.unsqueeze(1)
        unpack_shape = (b, n) + bn_tensor.shape[1:]
        return bn_tensor.view(unpack_shape)

    def postprocess_frames(self, paths, meta_data=None, grad_index=None):
        b, n, c, h, w = paths.shape

        if grad_index is None:
            grad_index = range(n)

        seq_mode = True
        if n == 1:
            seq_mode = False
        else:
            if len(grad_index) == 0:
                seq_mode = False
            if (not self.low_mem) and len(grad_index) == n:
                # low mem off and full grad
                seq_mode = False

        if seq_mode:
            results_list = []
            for frame_i in range(n):
                if frame_i in grad_index:
                    results_list.append(
                        self._postprocess_frames_impl(
                            paths=paths[:, frame_i, ...], meta_data=meta_data
                        )
                    )
                else:
                    with torch.no_grad():
                        results_list.append(
                            self._postprocess_frames_impl(
                                paths=paths[:, frame_i, ...], meta_data=meta_data
                            )
                        )
            results = dict()
            for key in results_list[0].keys():
                new_shape = (b, n) + results_list[0][key].shape[1:]
                results[key] = torch.stack([data[key] for data in results_list], dim=1).view(
                    new_shape
                )
        else:
            if len(grad_index) > 0:
                results = self._postprocess_frames_impl(paths=paths, meta_data=meta_data)
            else:
                with torch.no_grad():
                    results = self._postprocess_frames_impl(paths=paths, meta_data=meta_data)

        return results

    def _postprocess_frames_impl(self, paths, meta_data=None):
        patch_h, patch_w = meta_data["patch_h"], meta_data["patch_w"]
        if paths.ndim == 4:
            return self.head(paths, patch_h, patch_w, return_dict=True, meta_data=meta_data)

        b, n, c, h, w = paths.shape
        results = self.head(
            paths.view(b * n, c, h, w),
            patch_h,
            patch_w,
            return_dict=True,
            meta_data=meta_data,
        )

        for key in results.keys():
            results[key] = self.unpack_bn(results[key], b, n)

        return results

    def forward(
        self,
        x,
        prompt_depth,
        with_freeze=False,
        meta_data=None,
        grad_index=None,
        flow_mode=False,
        flow_clear=False,
        **kwargs
    ):
        """

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, num_frames * num_views, channels, height, width).
        - prompt_depth (Tensor or None): Depth information for guiding predictions.
        - freeze (bool): Whether to freeze certain layers during inference.
        """
        if with_freeze:
            self.freeze()

        assert meta_data is not None
        if "frames" not in meta_data and "views" not in meta_data:
            return super().forward(x, prompt_depth=prompt_depth, meta_data=meta_data)

        if flow_mode and flow_clear:
            self.decoder.clear_hidden()

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        # Unpack input dimensions for batch size (b), number of frames (n), channels (c), height (h), width (w)
        b, n, c, h, w = x.shape
        patch_h, patch_w = h // self.rgb_encoder.patch_size, w // self.rgb_encoder.patch_size
        assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

        if grad_index is None:
            grad_index = range(n)
        else:
            grad_index = [(grad_i + n) % n for grad_i in grad_index]

        # rgb, prompt encoder
        rgb_features, prompt_features = self.preprocess_frames(
            x=x,
            prompt_depth=prompt_depth,
            meta_data=meta_data,
            grad_index=grad_index,
        )  # [b,n,c,d]*4 , [b,n,c,h,w]

        # rgb, prompt decoder
        paths = self.decoder(
            rgb_features=rgb_features,
            prompt_features=prompt_features,
            meta_data=meta_data,
            grad_index=grad_index,
            flow_mode=flow_mode,
        )  # [b,n,c,h,w]

        # multi task head
        results = self.postprocess_frames(
            paths=paths,
            meta_data=meta_data,
            grad_index=grad_index,
        )  # dict of pointmap and confidence

        return results
