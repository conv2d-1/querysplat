import torch
from torch.utils.checkpoint import checkpoint

from hAlgorithm.modules.models2.sdk.base import MVBase2
from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.utils import cuda_timing_context


class MVQuery(MVBase2):
    """Query-based multi-view prediction model.

    Extends MVBase2 with a sparse query mechanism: instead of decoding directly from
    dense patch features, it generates query tokens (via query_banck), cross-attends
    them with encoder patch tokens (via query_feats_aggregator), and decodes the
    aggregated query features (via query_decoder) to produce per-query predictions.

    Pipeline:
        rgb -> aggregator (encoder + fuse) -> patch_tokens
        rgb -> query_banck -> query
        (patch_tokens, query) -> query_feats_aggregator -> feats
        feats -> query_decoder -> results

    Args:
        query_banck: Config for the query bank module that produces query tokens from RGB.
        query_feats_aggregator: Config for the module that cross-attends queries with patch tokens.
        query_decoder: Config for the head that decodes aggregated query features.
        timing: If True, enables CUDA timing for profiling each stage.
        **kwargs: Passed to MVBase2 (encoders, heads, freeze config, etc.).
    """

    def __init__(
        self,
        query_banck,
        query_feats_aggregator,
        query_decoder,
        timing=False,
        **kwargs,
    ):
        super(MVQuery, self).__init__(**kwargs)

        self.query_banck = self._instantiate_and_register(query_banck, "query_banck")
        self.query_feats_aggregator = self._instantiate_and_register(query_feats_aggregator, "query_feats_aggregator")
        self.query_decoder = self._instantiate_and_register(query_decoder, "query_decoder")

        self.timing = timing

    def forward(
        self,
        rgb,
        query_rgb=None,
        edge_mask=None,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        rgb_mask=None,
        meta_data=None,
        **kwargs,
    ):
        """
        Args:
            rgb: Input images, (B, N, C, H, W) for multi-view or (B, C, H, W) for single-view.
            query_rgb: Optional separate RGB input for query generation.
            edge_mask: Optional edge mask passed to query_banck.
            scale: Per-pixel scale factors.
            prompt_depth: Depth prompt for the encoder.
            intrinsics: Camera intrinsics (B, N, 3, 3).
            ray_directions: Per-pixel ray directions in camera space.
            w2c: World-to-camera transforms (B, N, 4, 4).
            c2w: Camera-to-world transforms (B, N, 4, 4).
            ray_world: Per-pixel ray directions in world space.
            meta_data: Dict with auxiliary info (frames, views, data_info, etc.).

        Returns:
            dict with query decoder outputs reshaped to (B, N, ...) and a "query" key
            holding the raw query tokens.
        """
        if self.training:
            self.freeze()

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        # Stage 1: dense feature extraction — encode RGB and fuse with geometric prompts
        with cuda_timing_context("dense aggregator", self.timing):
            patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
                rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world, rgb_mask=rgb_mask, meta_data=meta_data
            )

            # Strip prefix tokens (e.g. camera/cls tokens) to keep only spatial patch tokens
            if isinstance(patch_features, (list, tuple)):
                patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
            else:
                patch_tokens = [patch_features[:, :, patch_start_idx:]]

        # Stage 2: generate sparse query tokens from the input image
        with cuda_timing_context("query_banck", self.timing):
            query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)

        # Stage 3: cross-attend queries with dense patch tokens
        feats = self.query_feats_aggregator(x=patch_tokens, prompt_depth=prompt_depth, query=query, query_rgb=query_rgb, meta_data=meta_data)

        # Stage 4: decode aggregated features (optionally in fp32 during training)
        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                results = self.query_decoder(x=feats.float(), meta_data=meta_data)

                extra_results = self.decoder(
                    b=b,
                    n=n,
                    patch_features=patch_features,
                    pos=pos,
                    patch_start_idx=patch_start_idx,
                    prompt_depth=prompt_depth,
                    query_points=None,
                    meta_data=meta_data,
                )
        else:
            results = self.query_decoder(x=feats, meta_data=meta_data)

            extra_results = self.decoder(
                b=b,
                n=n,
                patch_features=patch_features,
                pos=pos,
                patch_start_idx=patch_start_idx,
                prompt_depth=prompt_depth,
                query_points=None,
                meta_data=meta_data,
            )

        results = {key: val.view(b, n, *val.shape[-2:]) for key, val in results.items()}
        results["query"] = query

        results.update(extra_results)

        return results


class MVQuery2(MVQuery):
    def __init__(self, chunk_size=None, **kwargs):
        super(MVQuery2, self).__init__(**kwargs)

        self.chunk_size = chunk_size

    def forward(
        self,
        rgb,
        query_rgb=None,
        edge_mask=None,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        rgb_mask=None,
        meta_data=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        # Stage 1: dense feature extraction — encode RGB and fuse with geometric prompts
        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world, rgb_mask=rgb_mask, meta_data=meta_data
        )

        # Strip prefix tokens (e.g. camera/cls tokens) to keep only spatial patch tokens
        if isinstance(patch_features, (list, tuple)):
            patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
        else:
            patch_tokens = [patch_features[:, :, patch_start_idx:]]

        # Stage 2: generate sparse query tokens from the input image
        query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)

        query_nums = query.uv.shape[0]
        if not self.training and self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
            results_list = []
            for q0 in range(0, query_nums, self.chunk_size):
                q1 = min(q0 + self.chunk_size, query_nums)

                chunk_query = BaseQuery(uv=query.uv[q0:q1])

                # Stage 3: cross-attend queries with dense patch tokens
                chunk_feats = self.query_feats_aggregator(x=patch_tokens, prompt_depth=prompt_depth, query=chunk_query, query_rgb=query_rgb, meta_data=meta_data)

                # Stage 4: decode aggregated features (optionally in fp32 during training)
                results_list.append(
                    self.query_decoder(x=chunk_feats, meta_data=meta_data)
                )

            results = dict()
            for key in results_list[0].keys():
                results[key] = torch.cat([res[key] for res in results_list], dim=1)

        else:
            # Stage 3: cross-attend queries with dense patch tokens
            feats = self.query_feats_aggregator(x=patch_tokens, prompt_depth=prompt_depth, query=query, query_rgb=query_rgb, meta_data=meta_data)

            # Stage 4: decode aggregated features (optionally in fp32 during training)
            if self.training and self.decoder_fp32:
                with torch.autocast(device_type=rgb.device.type, enabled=False):
                    results = self.query_decoder(x=feats.float(), meta_data=meta_data)
            else:
                results = self.query_decoder(x=feats, meta_data=meta_data)

        # Stage 5: decode aggregated features (optionally in fp32 during training)
        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                extra_results = self.decoder(
                    b=b,
                    n=n,
                    patch_features=patch_features,
                    pos=pos,
                    patch_start_idx=patch_start_idx,
                    prompt_depth=prompt_depth,
                    query_points=None,
                    meta_data=meta_data,
                )
        else:
            extra_results = self.decoder(
                b=b,
                n=n,
                patch_features=patch_features,
                pos=pos,
                patch_start_idx=patch_start_idx,
                prompt_depth=prompt_depth,
                query_points=None,
                meta_data=meta_data,
            )

        results = {key: val.view(b, n, *val.shape[-2:]) for key, val in results.items()}
        results["query"] = query

        results.update(extra_results)

        return results


class MVQuery3(MVQuery):
    """MVQuery2 extended with a pixel-aligned Gaussian splatting head.

    After the standard query-based depth prediction, an optional Gaussian head
    produces per-pixel 3D Gaussian parameters from dense backbone features,
    predicted depth, and input RGB. The Gaussian head only runs when queries
    cover the full image (full_uv=True), so that a dense depth map can be
    constructed from the query decoder output.

    Reference: InfiniDepth's GSPixelAlignPredictor architecture.

    Args:
        ffgs: Config for the Gaussian splatting head module.
        **kwargs: Passed to MVQuery2.
    """

    def __init__(self, ffgs=None, chunk_size=None, **kwargs):
        super(MVQuery3, self).__init__(**kwargs)
        self.ffgs = self._instantiate_and_register(ffgs, "ffgs")

        self.chunk_size = chunk_size
    
    def get_batch_views(self, rgb):
        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape
        return b, n

    def forward(
        self,
        rgb,
        query_rgb=None,
        edge_mask=None,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        rgb_mask=None,
        meta_data=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        b, n = self.get_batch_views(rgb)

        # Stage 1: dense feature extraction
        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics,
            ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world,
            rgb_mask=rgb_mask, meta_data=meta_data,
        )

        if isinstance(patch_features, (list, tuple)):
            patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
        else:
            patch_tokens = [patch_features[:, :, patch_start_idx:]]

        # Stage 2: generate sparse query tokens
        query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)

        # Stage 3 & 4: cross-attend and decode (inherited from MVQuery2)
        query_nums = query.uv.shape[0]
        if not self.training and self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
            results_list = []
            for q0 in range(0, query_nums, self.chunk_size):
                q1 = min(q0 + self.chunk_size, query_nums)
                chunk_query = BaseQuery(uv=query.uv[q0:q1])
                chunk_feats = self.query_feats_aggregator(
                    x=patch_tokens, prompt_depth=prompt_depth,
                    query=chunk_query, query_rgb=query_rgb, meta_data=meta_data,
                )
                results_list.append(self.query_decoder(x=chunk_feats, meta_data=meta_data))

            results = {}
            for key in results_list[0].keys():
                results[key] = torch.cat([res[key] for res in results_list], dim=1)
        else:
            feats = self.query_feats_aggregator(
                x=patch_tokens, prompt_depth=prompt_depth,
                query=query, query_rgb=query_rgb, meta_data=meta_data,
            )
            if self.training and self.decoder_fp32:
                with torch.autocast(device_type=rgb.device.type, enabled=False):
                    results = self.query_decoder(x=feats.float(), meta_data=meta_data)
            else:
                results = self.query_decoder(x=feats, meta_data=meta_data)

        # Stage 5: extra decoder heads (depth_head, normal_head, etc.)
        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                extra_results = self.decoder(
                    b=b, n=n, patch_features=patch_features, pos=pos,
                    patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
                    query_points=None, meta_data=meta_data,
                )
        else:
            extra_results = self.decoder(
                b=b, n=n, patch_features=patch_features, pos=pos,
                patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
                query_points=None, meta_data=meta_data,
            )

        results = {key: val.view(b, n, *val.shape[-2:]) for key, val in results.items()}
        results["query"] = query
        results.update(extra_results)

        # Stage 6: Gaussian splatting head (requires dense depth from full queries)
        if self.ffgs is not None and query.full_uv and results.get("depth") is not None:
            pred_depth = results["depth"].reshape(b, query.height, query.width, -1)
            gs_results = self.ffgs(
                patch_features=patch_features,
                patch_start_idx=patch_start_idx,
                rgb=rgb,
                depth=results["depth"],
                depth_width=query.width,
                depth_height=query.height,
                intrinsics=intrinsics,
                w2c=w2c,
                c2w=c2w,
                scale=scale,
            )
            results.update(gs_results)

        return results


class MVQuery4(MVBase2):
    """MVQuery extended with pair-wise cross-view matching and Gaussian splatting.

    Adds two capabilities on top of MVQuery:
    1. Pair-wise matching: given pair_idx, cross-attends query tokens between
       view pairs (via query_pair_feats_aggregator), decodes pair features
       (via query_pair_decoder), and optionally refines (via query_pair_refine).
    2. Gaussian splatting head (ffgs): produces per-pixel 3D Gaussian parameters
       when queries cover the full image (full_uv=True).

    Both single-view and pair-wise branches support chunked inference controlled
    by chunk_size to limit peak memory at test time.

    Args:
        query_pair_feats_aggregator: Config for pair-wise cross-view attention module.
        query_pair_decoder: Config for the head that decodes pair features.
        query_pair_refine: Config for optional refinement on pair decoder outputs.
        ffgs: Config for the Gaussian splatting head module.
        chunk_size: Max queries per chunk during inference (None = no chunking).
        **kwargs: Passed to MVQuery (encoders, heads, freeze config, etc.).
    """

    def __init__(
        self,
        query_banck,
        query_feats_aggregator=None,
        query_decoder=None,
        query_pair_feats_aggregator=None,
        query_pair_decoder=None,
        query_pair_decoder2=None,
        query_pair_refine=None, 
        query_pair_refine2=None,
        ffgs=None, 
        chunk_size=None,
        time_token_index=None,
        **kwargs
    ):
        super(MVQuery4, self).__init__(**kwargs)

        self.query_banck = self._instantiate_and_register(query_banck, "query_banck")
        self.query_feats_aggregator = self._instantiate_and_register(query_feats_aggregator, "query_feats_aggregator")
        self.query_decoder = self._instantiate_and_register(query_decoder, "query_decoder")

        self.query_pair_feats_aggregator = self._instantiate_and_register(query_pair_feats_aggregator, "query_pair_feats_aggregator")

        self.query_pair_decoder = self._instantiate_and_register(query_pair_decoder, "query_pair_decoder")
        self.query_pair_decoder2 = self._instantiate_and_register(query_pair_decoder2, "query_pair_decoder2")

        self.query_pair_refine = self._instantiate_and_register(query_pair_refine, "query_pair_refine")
        self.query_pair_refine2 = self._instantiate_and_register(query_pair_refine2, "query_pair_refine2")

        self.ffgs = self._instantiate_and_register(ffgs, "ffgs")

        self.chunk_size = chunk_size
        self.time_token_index = time_token_index
    
    def get_batch_views(self, rgb):
        """Extract batch size and view count from input tensor shape.

        Returns:
            (b, n): batch size and number of views (n=1 for single-view input).
        """
        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape
        return b, n

    def _get_time_token(self, b, n, device, meta_data):
        """Hook for subclasses to produce a per-frame time token for concat injection.

        Returns a tensor of shape [B*N, 1, C] that will be concatenated as an
        independent token inside the fuse_encoder (MoVieS-style), or None to skip.
        """
        return None

    def single_forward(self, rgb, patch_tokens, prompt_depth, query, query_rgb, meta_data):
        """Per-view query decoding: cross-attend queries with patch tokens, then decode.

        Args:
            rgb: Input images, used only to determine device for autocast.
            patch_tokens: Dense backbone features (list of (B, N, P, D) tensors).
            prompt_depth: Depth prompt from the encoder.
            query: BaseQuery with uv coordinates to sample.
            query_rgb: Optional separate RGB for query generation.
            meta_data: Auxiliary metadata dict.

        Returns:
            dict: Per-query predictions from query_decoder, e.g. {"depth": (B*N, Q, 1)}.
        """
        results = dict()

        if self.query_feats_aggregator is not None:
            feats = self.query_feats_aggregator(
                x=patch_tokens, prompt_depth=prompt_depth,
                query=query, query_rgb=query_rgb, meta_data=meta_data,
            )
            if self.query_decoder is None:
                return results
            if self.training and self.decoder_fp32:
                with torch.autocast(device_type=rgb.device.type, enabled=False):
                    return self.query_decoder(x=feats.float(), meta_data=meta_data)
            else:
                return self.query_decoder(x=feats, meta_data=meta_data)

        return results

    def pair_forward(self, rgb, pair_idx, patch_tokens, time_token, query, query_rgb, meta_data):
        """Pair-wise cross-view matching and decoding.

        For each view pair in pair_idx, performs cross-view feature matching
        (e.g. RoMa-style dense matching) between the paired views, then decodes
        the matched features to produce pair-wise predictions (e.g. relative depth,
        correspondences).

        Args:
            rgb: Input images, used only to determine device for autocast.
            pair_idx: List of (i, j) view index tuples defining pairs. len = num_pair.
            patch_tokens: Dense backbone features (list of (B, N, P, D) tensors).
            query: BaseQuery with uv coordinates to sample.
            query_rgb: Optional separate RGB for query generation.
            meta_data: Auxiliary metadata dict.

        Returns:
            dict: Pair-wise predictions, each value shaped (B * num_pair, Q, C).
                  If query_pair_refine is set, includes refined results as well.
        """
        results = dict()

        pyramids = single_feats = None
        if self.query_pair_feats_aggregator is not None:
            if getattr(self.query_pair_feats_aggregator, "return_dense_feats", False):
                feats, pyramids = self.query_pair_feats_aggregator(
                    x=patch_tokens, query=query, meta_data=meta_data,
                    pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                    time_token=time_token,
                )
            elif getattr(self.query_pair_feats_aggregator, "return_pyramids", False):
                feats, pyramids = self.query_pair_feats_aggregator(
                    x=patch_tokens, query=query, meta_data=meta_data,
                    pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                )
            elif getattr(self.query_pair_feats_aggregator, "return_src_hidden", False):
                feats, single_feats = self.query_pair_feats_aggregator(
                    x=patch_tokens, query=query, meta_data=meta_data,
                    pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                )
            else:
                feats = self.query_pair_feats_aggregator(
                    x=patch_tokens, query=query, meta_data=meta_data,
                    pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                    time_token=time_token,
                )

            if self.training and self.decoder_fp32:
                with torch.autocast(device_type=rgb.device.type, enabled=False):
                    results = self.query_pair_decoder(x=feats.float(), meta_data=meta_data)

                    if self.query_pair_decoder2 is not None:
                        single_results = self.query_pair_decoder2(x=single_feats.float(), meta_data=meta_data)
                        results.update(single_results)
            else:
                results = self.query_pair_decoder(x=feats, meta_data=meta_data)

                if self.query_pair_decoder2 is not None:
                    single_results = self.query_pair_decoder2(x=single_feats, meta_data=meta_data)
                    results.update(single_results)

            if self.query_pair_refine is not None:
                refine_results = self.query_pair_refine(
                    coarse_results=results,
                    query=query,
                    rgb=rgb,
                    meta_data=meta_data,
                    pair_idx=pair_idx,
                    query_rgb=query_rgb,
                )
                results.update(refine_results)

            if self.query_pair_refine2 is not None:
                refine2_results = self.query_pair_refine2(
                    coarse_results=results,
                    query=query,
                    rgb=rgb,
                    feats=pyramids,
                    meta_data=meta_data,
                    pair_idx=pair_idx,
                    query_rgb=query_rgb,
                )
                results.update(refine2_results)
        
        return results

    def forward(
        self,
        rgb,
        query=None,
        query_rgb=None,
        edge_mask=None,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        rgb_mask=None,
        meta_data=None,
        pair_idx=None,
        time_idx=None,
        **kwargs,
    ):
        """Full forward pass with single-view decoding, optional pair matching, and Gaussian splatting.

        Pipeline:
            Stage 1: Dense feature extraction (encoder + fuse) -> patch_tokens
            Stage 2: Sparse query generation (query_banck) -> query
            Stage 3-4: Single-view query decoding (query_feats_aggregator + query_decoder) -> per-view results
            Stage 5: Extra dense decoder heads (depth_head, normal_head, etc.) -> extra_results
            Stage 6: [optional] Gaussian splatting head (ffgs) when full_uv queries provide dense depth
            Stage 7: [optional] Pair-wise cross-view matching (query_pair_feats_aggregator +
                      query_pair_decoder + query_pair_refine) when pair_idx is provided

        At inference time, stages 3-4 and 7 support chunked processing (controlled by
        chunk_size) to cap peak GPU memory by iterating over query subsets.

        Args:
            rgb: Input images, (B, N, C, H, W) for multi-view or (B, C, H, W) for single-view.
            query_rgb: Optional separate RGB input for query generation.
            edge_mask: Optional edge mask passed to query_banck.
            scale: Per-pixel scale factors.
            prompt_depth: Depth prompt for the encoder.
            intrinsics: Camera intrinsics (B, N, 3, 3).
            ray_directions: Per-pixel ray directions in camera space.
            w2c: World-to-camera transforms (B, N, 4, 4).
            c2w: Camera-to-world transforms (B, N, 4, 4).
            ray_world: Per-pixel ray directions in world space.
            rgb_mask: Optional mask for valid RGB regions.
            meta_data: Dict with auxiliary info (frames, views, data_info, etc.).
            pair_idx: List of (i, j) view index tuples for pair-wise matching. None to skip.

        Returns:
            dict containing:
                - Per-view query results reshaped to (B, N, Q, C), e.g. "depth", "normal".
                - "query": the raw BaseQuery object.
                - Extra dense decoder outputs (from self.decoder).
                - [if ffgs] Gaussian splatting outputs (e.g. "gaussians").
                - [if pair_idx] Pair-wise results reshaped to (B, num_pair, Q, C) and "pair_idx".
        """
        if self.training:
            self.freeze()

        b, n = self.get_batch_views(rgb)

        # Stage 1: dense feature extraction
        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics,
            ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world,
            rgb_mask=rgb_mask, meta_data=meta_data,
        )

        if isinstance(patch_features, (list, tuple)):
            patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
        else:
            patch_tokens = [patch_features[:, :, patch_start_idx:]]

        # Stage 2: generate sparse query tokens
        if query is None:
            query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)
        query_nums = query.query_nums

        # Stage 3 & 4: single-view cross-attend and decode (chunked at inference)
        if not self.training and self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
            results_list = []
            for q0 in range(0, query_nums, self.chunk_size):
                q1 = min(q0 + self.chunk_size, query_nums)
                chunk_query = BaseQuery(uv=query.uv[q0:q1])
                results_list.append(
                    self.single_forward(rgb, patch_tokens, prompt_depth, chunk_query, query_rgb, meta_data)
                )
            results = {}
            for key in results_list[0].keys():
                results[key] = torch.cat([res[key] for res in results_list], dim=1)
        else:
            results = self.single_forward(rgb, patch_tokens, prompt_depth, query, query_rgb, meta_data)

        results = {key: val.view(b, n, *val.shape[-2:]) for key, val in results.items()}

        # Stage 5: extra dense decoder heads (depth_head, normal_head, etc.)
        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                extra_results = self.decoder(
                    b=b, n=n, patch_features=patch_features, pos=pos,
                    patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
                    query_points=None, meta_data=meta_data,
                )
        else:
            extra_results = self.decoder(
                b=b, n=n, patch_features=patch_features, pos=pos,
                patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
                query_points=None, meta_data=meta_data,
            )

        results.update(extra_results)

        # Stage 6: Gaussian splatting head (requires dense depth from full queries)
        if self.ffgs is not None and query.full_uv and results.get("depth") is not None:
            # pred_depth = results["depth"].reshape(b, query.height, query.width, -1)
            gs_results = self.ffgs(
                patch_features=patch_features,
                patch_start_idx=patch_start_idx,
                rgb=rgb,
                depth=results["depth"],
                depth_width=query.width,
                depth_height=query.height,
                intrinsics=intrinsics,
                w2c=w2c,
                c2w=c2w,
                scale=scale,
            )
            results.update(gs_results)
        
        # Stage 7: pair-wise cross-view matching (chunked at inference)
        if pair_idx is not None:
            num_pair = len(pair_idx)

            if self.time_token_index is not None:
                if isinstance(patch_features, (list, tuple)):
                    time_token = patch_features[-1][:, :, self.time_token_index]
                else:
                    time_token = patch_features[:, :, self.time_token_index]
            else:
                time_token = None

            if not self.training and self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
                pair_results_list = []
                for q0 in range(0, query_nums, self.chunk_size):
                    q1 = min(q0 + self.chunk_size, query_nums)

                    chunk_query = BaseQuery(uv=query.uv[q0:q1])
                    pair_results_list.append(
                        self.pair_forward(rgb, pair_idx=pair_idx, patch_tokens=patch_tokens, time_token=time_token, query=chunk_query, query_rgb=query_rgb, meta_data=meta_data)
                    )
                pair_results = dict()
                for key in pair_results_list[0].keys():
                    pair_results[key] = torch.cat([res[key] for res in pair_results_list], dim=1)
            else:
                pair_results = self.pair_forward(rgb, pair_idx=pair_idx, patch_tokens=patch_tokens, time_token=time_token, query=query, query_rgb=query_rgb, meta_data=meta_data)
            
            pair_results = {key: val.view(b, num_pair, *val.shape[-2:]) for key, val in pair_results.items()}
            pair_results["pair_idx"] = pair_idx
            results.update(pair_results)

        results["query"] = query

        return results


class MVQuery6(MVQuery4):
    """MVQuery extended with pair-wise cross-view matching and Gaussian splatting.
    """

    def __init__(self, sparse_gaussian_head=None, **kwargs):
        super(MVQuery6, self).__init__(**kwargs)
        self.sparse_gaussian_head = self._instantiate_and_register(
            sparse_gaussian_head, "sparse_gaussian_head",
        )

    def _run_sparse_gaussian_head(
        self,
        pair_flat: dict,
        pair_idx,
        batch_size: int,
        num_views: int,
        meta_data: dict,
        rgb=None,
        gaussian_query=None,
        intrinsics=None,
        w2c=None,
    ) -> dict:
        if self.sparse_gaussian_head is None or pair_idx is None:
            return {}
        query_feats = pair_flat.pop("_pair_query_feats", None)
        if query_feats is None:
            query_feats = pair_flat.pop("_single_feats", None)
        if query_feats is None:
            return {}

        from hAlgorithm.modules.pipelines2.utils.motion_utils import (
            prepare_decoupled_warp3d_delta_inplace,
        )

        prepare_decoupled_warp3d_delta_inplace(pair_flat)
        geometry_kwargs = {}
        if (
            getattr(self.sparse_gaussian_head, "geometry_source", "warp3d")
            == "camera_depth"
        ):
            geometry_kwargs["w2c"] = w2c

        return self.sparse_gaussian_head(
            single_feats=query_feats,
            pair_outputs=pair_flat,
            pair_idx=pair_idx,
            batch_size=batch_size,
            meta_data=meta_data,
            num_views=num_views,
            rgb=rgb,
            gaussian_query=gaussian_query,
            intrinsics=intrinsics,
            **geometry_kwargs,
        )

    def single_forward(self, rgb, patch_tokens, prompt_depth, query, query_rgb, meta_data):
        """Per-view query decoding: cross-attend queries with patch tokens, then decode.

        Args:
            rgb: Input images, used only to determine device for autocast.
            patch_tokens: Dense backbone features (list of (B, N, P, D) tensors).
            prompt_depth: Depth prompt from the encoder.
            query: BaseQuery with uv coordinates to sample.
            query_rgb: Optional separate RGB for query generation.
            meta_data: Auxiliary metadata dict.

        Returns:
            dict: Per-query predictions from query_decoder, e.g. {"depth": (B*N, Q, 1)}.
        """
        results = dict()

        if self.query_feats_aggregator is not None:
            if self.training:
                feats = checkpoint(
                    self.query_feats_aggregator,
                    x=patch_tokens, prompt_depth=prompt_depth,
                    query=query, query_rgb=query_rgb, meta_data=meta_data,
                    use_reentrant=False,
                )
            else:
                feats = self.query_feats_aggregator(
                    x=patch_tokens, prompt_depth=prompt_depth,
                    query=query, query_rgb=query_rgb, meta_data=meta_data,
                )
            if self.training and self.decoder_fp32:
                with torch.autocast(device_type=rgb.device.type, enabled=False):
                    return self.query_decoder(x=feats.float(), meta_data=meta_data)
            else:
                return self.query_decoder(x=feats, meta_data=meta_data)

        return results

    def pair_forward(self, rgb, pair_idx, patch_tokens, time_token, query, query_rgb, meta_data):
        """Pair-wise cross-view matching and decoding.

        For each view pair in pair_idx, performs cross-view feature matching
        (e.g. RoMa-style dense matching) between the paired views, then decodes
        the matched features to produce pair-wise predictions (e.g. relative depth,
        correspondences).

        Args:
            rgb: Input images, used only to determine device for autocast.
            pair_idx: List of (i, j) view index tuples defining pairs. len = num_pair.
            patch_tokens: Dense backbone features (list of (B, N, P, D) tensors).
            query: BaseQuery with uv coordinates to sample.
            query_rgb: Optional separate RGB for query generation.
            meta_data: Auxiliary metadata dict.

        Returns:
            dict: Pair-wise predictions, each value shaped (B * num_pair, Q, C).
                  If query_pair_refine is set, includes refined results as well.
        """
        results = dict()

        pyramids = single_feats = None
        if self.query_pair_feats_aggregator is not None:
            if getattr(self.query_pair_feats_aggregator, "return_dense_feats", False):
                if self.training:
                    feats, pyramids = checkpoint(
                        self.query_pair_feats_aggregator,
                        x=patch_tokens, query=query, meta_data=meta_data,
                        pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                        time_token=time_token,
                        use_reentrant=False,
                    )
                else:
                    feats, pyramids = self.query_pair_feats_aggregator(
                        x=patch_tokens, query=query, meta_data=meta_data,
                        pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                        time_token=time_token,
                    )
            elif getattr(self.query_pair_feats_aggregator, "return_pyramids", False):
                if self.training:
                    feats, pyramids = checkpoint(
                        self.query_pair_feats_aggregator,
                        x=patch_tokens, query=query, meta_data=meta_data,
                        pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                        use_reentrant=False,
                    )
                else:
                    feats, pyramids = self.query_pair_feats_aggregator(
                        x=patch_tokens, query=query, meta_data=meta_data,
                        pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                    )

            elif getattr(self.query_pair_feats_aggregator, "return_src_hidden", False):
                if self.training:
                    feats, single_feats = checkpoint(
                        self.query_pair_feats_aggregator,
                        x=patch_tokens, query=query, meta_data=meta_data,
                        pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                        use_reentrant=False,
                    )
                else:
                    feats, single_feats = self.query_pair_feats_aggregator(
                        x=patch_tokens, query=query, meta_data=meta_data,
                        pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                    )
            else:
                if self.training:
                    feats = checkpoint(
                        self.query_pair_feats_aggregator,
                        x=patch_tokens, query=query, meta_data=meta_data,
                        pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                        time_token=time_token,
                        use_reentrant=False,
                    )
                else:
                    feats = self.query_pair_feats_aggregator(
                        x=patch_tokens, query=query, meta_data=meta_data,
                        pair_idx=pair_idx, query_rgb=query_rgb, rgb=rgb,
                        time_token=time_token,
                    )

            if self.training and self.decoder_fp32:
                with torch.autocast(device_type=rgb.device.type, enabled=False):
                    results = self.query_pair_decoder(x=feats.float(), meta_data=meta_data)

                    if self.query_pair_decoder2 is not None:
                        single_results = self.query_pair_decoder2(x=single_feats.float(), meta_data=meta_data)
                        results.update(single_results)
            else:
                results = self.query_pair_decoder(x=feats, meta_data=meta_data)

                if self.query_pair_decoder2 is not None:
                    single_results = self.query_pair_decoder2(x=single_feats, meta_data=meta_data)
                    results.update(single_results)

            if self.query_pair_refine is not None:
                refine_results = self.query_pair_refine(
                    coarse_results=results,
                    query=query,
                    rgb=rgb,
                    meta_data=meta_data,
                    pair_idx=pair_idx,
                    query_rgb=query_rgb,
                )
                results.update(refine_results)

            if self.query_pair_refine2 is not None:
                refine2_results = self.query_pair_refine2(
                    coarse_results=results,
                    query=query,
                    rgb=rgb,
                    feats=pyramids,
                    meta_data=meta_data,
                    pair_idx=pair_idx,
                    query_rgb=query_rgb,
                )
                results.update(refine2_results)

            if single_feats is not None:
                results["_single_feats"] = single_feats
            if feats is not None:
                results["_pair_query_feats"] = feats
        
        return results

    def forward(
        self,
        rgb,
        query=None,
        query_rgb=None,
        edge_mask=None,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        rgb_mask=None,
        meta_data=None,
        pair_idx=None,
        time_idx=None,
        **kwargs,
    ):
        """Full forward pass with single-view decoding, optional pair matching, and Gaussian splatting.

        Pipeline:
            Stage 1: Dense feature extraction (encoder + fuse) -> patch_tokens
            Stage 2: Sparse query generation (query_banck) -> query
            Stage 3-4: Single-view query decoding (query_feats_aggregator + query_decoder) -> per-view results
            Stage 5: Extra dense decoder heads (depth_head, normal_head, etc.) -> extra_results
            Stage 6: [optional] Gaussian splatting head (ffgs) when full_uv queries provide dense depth
            Stage 7: [optional] Pair-wise cross-view matching (query_pair_feats_aggregator +
                      query_pair_decoder + query_pair_refine) when pair_idx is provided
            Stage 8: [optional] Sparse pair Dynamic 4DGS head on ``single_feats`` + pair motion

        At inference time, stages 3-4 and 7 support chunked processing (controlled by
        chunk_size) to cap peak GPU memory by iterating over query subsets.

        Args:
            rgb: Input images, (B, N, C, H, W) for multi-view or (B, C, H, W) for single-view.
            query_rgb: Optional separate RGB input for query generation.
            edge_mask: Optional edge mask passed to query_banck.
            scale: Per-pixel scale factors.
            prompt_depth: Depth prompt for the encoder.
            intrinsics: Camera intrinsics (B, N, 3, 3).
            ray_directions: Per-pixel ray directions in camera space.
            w2c: World-to-camera transforms (B, N, 4, 4).
            c2w: Camera-to-world transforms (B, N, 4, 4).
            ray_world: Per-pixel ray directions in world space.
            rgb_mask: Optional mask for valid RGB regions.
            meta_data: Dict with auxiliary info (frames, views, data_info, etc.).
            pair_idx: List of (i, j) view index tuples for pair-wise matching. None to skip.

        Returns:
            dict containing:
                - Per-view query results reshaped to (B, N, Q, C), e.g. "depth", "normal".
                - "query": the raw BaseQuery object.
                - Extra dense decoder outputs (from self.decoder).
                - [if ffgs] Gaussian splatting outputs (e.g. "gaussians").
                - [if pair_idx] Pair-wise results reshaped to (B, num_pair, Q, C) and "pair_idx".
                - [if sparse_gaussian_head] ``gs_*`` and ``sparse_*`` tensors for 4DGS.
        """
        if self.training:
            self.freeze()

        b, n = self.get_batch_views(rgb)

        # Stage 1: dense feature extraction
        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics,
            ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world,
            rgb_mask=rgb_mask, meta_data=meta_data,
        )

        if isinstance(patch_features, (list, tuple)):
            patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
        else:
            patch_tokens = [patch_features[:, :, patch_start_idx:]]

        # Stage 2: generate sparse query tokens
        if query is None:
            query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)
        query_nums = query.query_nums

        # Stage 3 & 4: single-view cross-attend and decode (chunked at inference)
        if self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
            results_list = []
            for q0 in range(0, query_nums, self.chunk_size):
                q1 = min(q0 + self.chunk_size, query_nums)
                if query.uv.ndim == 4:
                    chunk_query = BaseQuery(uv=query.uv[:, :, q0:q1])
                else:
                    chunk_query = BaseQuery(uv=query.uv[q0:q1])
                results_list.append(
                    self.single_forward(rgb, patch_tokens, prompt_depth, chunk_query, query_rgb, meta_data)
                )
            results = {}
            for key in results_list[0].keys():
                results[key] = torch.cat([res[key] for res in results_list], dim=1)
        else:
            results = self.single_forward(rgb, patch_tokens, prompt_depth, query, query_rgb, meta_data)

        results = {key: val.view(b, n, *val.shape[-2:]) for key, val in results.items()}

        # Stage 5: extra dense decoder heads (depth_head, normal_head, etc.)
        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                extra_results = self.decoder(
                    b=b, n=n, patch_features=patch_features, pos=pos,
                    patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
                    query_points=None, meta_data=meta_data,
                )
        else:
            extra_results = self.decoder(
                b=b, n=n, patch_features=patch_features, pos=pos,
                patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
                query_points=None, meta_data=meta_data,
            )

        results.update(extra_results)

        # Stage 6: Gaussian splatting head (requires dense depth from full queries)
        if self.ffgs is not None and query.full_uv and results.get("depth") is not None:
            # pred_depth = results["depth"].reshape(b, query.height, query.width, -1)
            gs_results = self.ffgs(
                patch_features=patch_features,
                patch_start_idx=patch_start_idx,
                rgb=rgb,
                depth=results["depth"],
                depth_width=query.width,
                depth_height=query.height,
                intrinsics=intrinsics,
                w2c=w2c,
                c2w=c2w,
                scale=scale,
            )
            results.update(gs_results)
        
        # Stage 7: pair-wise cross-view matching (chunked at inference)
        if pair_idx is not None:
            num_pair = len(pair_idx)

            if self.time_token_index is not None:
                if isinstance(patch_features, (list, tuple)):
                    time_token = patch_features[-1][:, :, self.time_token_index]
                else:
                    time_token = patch_features[:, :, self.time_token_index]
            else:
                time_token = None

            if self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
                pair_results_list = []
                for q0 in range(0, query_nums, self.chunk_size):
                    q1 = min(q0 + self.chunk_size, query_nums)
                    
                    if query.uv.ndim == 4:
                        chunk_query = BaseQuery(uv=query.uv[:, :, q0:q1])
                    else:
                        chunk_query = BaseQuery(uv=query.uv[q0:q1])

                    pair_results_list.append(
                        self.pair_forward(rgb, pair_idx=pair_idx, patch_tokens=patch_tokens, time_token=time_token, query=chunk_query, query_rgb=query_rgb, meta_data=meta_data)
                    )
                pair_results = dict()
                for key in pair_results_list[0].keys():
                    pair_results[key] = torch.cat([res[key] for res in pair_results_list], dim=1)
            else:
                pair_results = self.pair_forward(rgb, pair_idx=pair_idx, patch_tokens=patch_tokens, time_token=time_token, query=query, query_rgb=query_rgb, meta_data=meta_data)

            gs_results = self._run_sparse_gaussian_head(
                pair_flat=pair_results,
                pair_idx=pair_idx,
                batch_size=b,
                num_views=n,
                meta_data=meta_data,
                rgb=rgb,
                gaussian_query=query,
                intrinsics=intrinsics,
                w2c=w2c,
            )
            if gs_results:
                results.update(gs_results)

            pair_results = {
                key: val.view(b, num_pair, *val.shape[-2:])
                for key, val in pair_results.items()
            }
            pair_results["pair_idx"] = pair_idx
            results.update(pair_results)

        results["query"] = query

        return results
