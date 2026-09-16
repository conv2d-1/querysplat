import torch

from hAlgorithm.modules.models2.sdk.base import MVBase2
from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.utils import cuda_timing_context


class MatchQuery(MVBase2):
    """Query-based model for matching pair scenario.

    Replaces the dense match head (RoMaV2MatchHead) with a sparse query mechanism.
    Uses the same encoder backbone but decodes warp + confidence at sparse query
    locations instead of producing dense correspondences.

    Pipeline:
        rgb -> aggregator (fuse_encoder) -> multi-layer patch_tokens
        rgb -> query_banck -> query (sparse UV)
        (patch_tokens, query, pair_idx) -> query_feats_aggregator -> feats
            internally: roma cross-view matching + grid_sample at query UV
        feats -> query_decoder -> {warp, confidence}

    Output is organized by pair: (B, num_pair, Q, ...)
    """

    def __init__(
        self,
        query_banck,
        query_feats_aggregator,
        query_decoder,
        query_match_refine=None,
        timing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.query_banck = self._instantiate_and_register(query_banck, "query_banck")
        self.query_feats_aggregator = self._instantiate_and_register(query_feats_aggregator, "query_feats_aggregator")
        self.query_decoder = self._instantiate_and_register(query_decoder, "query_decoder")
        self.query_match_refine = self._instantiate_and_register(query_match_refine, "query_match_refine")
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
        pair_idx=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        if pair_idx is None:
            pair_idx = [(0, 1)]
        num_pair = len(pair_idx)

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        with cuda_timing_context("dense aggregator", self.timing):
            patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
                rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics,
                ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world,
                rgb_mask=rgb_mask, meta_data=meta_data,
            )

            if isinstance(patch_features, (list, tuple)):
                patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
            else:
                patch_tokens = [patch_features[:, :, patch_start_idx:]]

        with cuda_timing_context("query_banck", self.timing):
            query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)

        # MatchQueryAggregator: roma cross-view matching + query sampling
        # Output: (B * num_pair, Q, dim)
        feats = self.query_feats_aggregator(
            x=patch_tokens, query=query, meta_data=meta_data,
            pair_idx=pair_idx, query_rgb=query_rgb,
        )

        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                results = self.query_decoder(x=feats.float(), meta_data=meta_data)
        else:
            results = self.query_decoder(x=feats, meta_data=meta_data)

        if self.query_match_refine is not None:
            refine_results = self.query_match_refine(
                coarse_results=results,
                query=query,
                rgb=rgb,
                patch_tokens=patch_tokens,
                meta_data=meta_data,
                pair_idx=pair_idx,
                query_rgb=query_rgb,
            )
            results.update(refine_results)

        # Reshape from (B * num_pair, Q, C) -> (B, num_pair, Q, C)
        results = {key: val.view(b, num_pair, *val.shape[-2:]) for key, val in results.items()}
        results["query"] = query
        results["pair_idx"] = pair_idx

        return results



class MatchQuery2(MVBase2):
    """Query-based model for matching pair scenario.

    Replaces the dense match head (RoMaV2MatchHead) with a sparse query mechanism.
    Uses the same encoder backbone but decodes warp + confidence at sparse query
    locations instead of producing dense correspondences.

    Pipeline:
        rgb -> aggregator (fuse_encoder) -> multi-layer patch_tokens
        rgb -> query_banck -> query (sparse UV)
        (patch_tokens, query, pair_idx) -> query_feats_aggregator -> feats
            internally: roma cross-view matching + grid_sample at query UV
        feats -> query_decoder -> {warp, confidence}

    Output is organized by pair: (B, num_pair, Q, ...)
    """

    def __init__(
        self,
        query_banck,
        query_feats_aggregator,
        query_decoder,
        query_match_refine=None,
        chunk_size=None,
        timing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.query_banck = self._instantiate_and_register(query_banck, "query_banck")
        self.query_feats_aggregator = self._instantiate_and_register(query_feats_aggregator, "query_feats_aggregator")
        self.query_decoder = self._instantiate_and_register(query_decoder, "query_decoder")
        self.query_match_refine = self._instantiate_and_register(query_match_refine, "query_match_refine")

        self.chunk_size = chunk_size
        self.timing = timing
    
    def forward_sparse(self, rgb, pair_idx, patch_tokens, query, query_rgb, meta_data):
        # MatchQueryAggregator: roma cross-view matching + query sampling
        # Output: (B * num_pair, Q, dim)
        feats = self.query_feats_aggregator(
            x=patch_tokens, query=query, meta_data=meta_data,
            pair_idx=pair_idx, query_rgb=query_rgb,
        )

        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                results = self.query_decoder(x=feats.float(), meta_data=meta_data)
        else:
            results = self.query_decoder(x=feats, meta_data=meta_data)

        if self.query_match_refine is not None:
            refine_results = self.query_match_refine(
                coarse_results=results,
                query=query,
                rgb=rgb,
                meta_data=meta_data,
                pair_idx=pair_idx,
                query_rgb=query_rgb,
            )
            results.update(refine_results)
        
        return results


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
        pair_idx=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        if pair_idx is None:
            pair_idx = [(0, 1)]
        num_pair = len(pair_idx)

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        with cuda_timing_context("dense aggregator", self.timing):
            patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
                rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics,
                ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world,
                rgb_mask=rgb_mask, meta_data=meta_data,
            )

            if isinstance(patch_features, (list, tuple)):
                patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
            else:
                patch_tokens = [patch_features[:, :, patch_start_idx:]]

        with cuda_timing_context("query_banck", self.timing):
            query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)

        query_nums = query.uv.shape[0]
        if not self.training and self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
            results_list = []
            for q0 in range(0, query_nums, self.chunk_size):
                q1 = min(q0 + self.chunk_size, query_nums)

                chunk_query = BaseQuery(uv=query.uv[q0:q1])
                results_list.append(
                    self.forward_sparse(rgb, pair_idx, patch_tokens, chunk_query, query_rgb, meta_data)
                )
            results = dict()
            for key in results_list[0].keys():
                results[key] = torch.cat([res[key] for res in results_list], dim=1)
        else:
            results = self.forward_sparse(rgb, pair_idx, patch_tokens, query, query_rgb, meta_data)

        # Reshape from (B * num_pair, Q, C) -> (B, num_pair, Q, C)
        results = {key: val.view(b, num_pair, *val.shape[-2:]) for key, val in results.items()}
        results["query"] = query
        results["pair_idx"] = pair_idx

        return results


class MatchQuery3(MVBase2):
    def __init__(
        self,
        query_banck,
        query_feats_aggregator,
        query_decoder,
        query_match_refine=None,
        time_encoder=None,
        chunk_size=None,
        timing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.query_banck = self._instantiate_and_register(query_banck, "query_banck")
        self.query_feats_aggregator = self._instantiate_and_register(query_feats_aggregator, "query_feats_aggregator")
        self.query_decoder = self._instantiate_and_register(query_decoder, "query_decoder")
        self.query_match_refine = self._instantiate_and_register(query_match_refine, "query_match_refine")

        self.time_encoder = self._instantiate_and_register(time_encoder, "time_encoder")

        self.chunk_size = chunk_size
        self.timing = timing
    
    def forward_sparse(self, rgb, pair_idx, patch_tokens, query, query_rgb, meta_data):
        # MatchQueryAggregator: roma cross-view matching + query sampling
        # Output: (B * num_pair, Q, dim)
        feats = self.query_feats_aggregator(
            x=patch_tokens, query=query, meta_data=meta_data,
            pair_idx=pair_idx, query_rgb=query_rgb,
        )

        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                results = self.query_decoder(x=feats.float(), meta_data=meta_data)
        else:
            results = self.query_decoder(x=feats, meta_data=meta_data)

        if self.query_match_refine is not None:
            refine_results = self.query_match_refine(
                coarse_results=results,
                query=query,
                rgb=rgb,
                meta_data=meta_data,
                pair_idx=pair_idx,
                query_rgb=query_rgb,
            )
            results.update(refine_results)
        
        return results
    
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
        pair_idx=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        if pair_idx is None:
            pair_idx = [(0, 1)]
        num_pair = len(pair_idx)

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics,
            ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world,
            rgb_mask=rgb_mask, meta_data=meta_data,
        )

        if self.time_encoder is not None:
            assert num_pair == 1, "Time encoder only supports one pair"
            cond_view_idxs = torch.ones(b, n, device=rgb.device, dtype=torch.int64)  # [B, n], all views conditioned on view 1
            patch_features = self.time_encoder(rgb.view(b, n, c, h, w), patch_features, patch_start_idx, cond_view_idxs)

            if isinstance(patch_features, (list, tuple)):
                patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
            else:
                patch_tokens = [patch_features[:, :, patch_start_idx:]]

            pair_patch_tokens_list =[]
            for patch_tokens_i in patch_tokens:
                pair_patch_tokens = torch.stack([patch_tokens_i[:, i0] for (i0, i1) in pair_idx], dim=1)
                pair_patch_tokens_list.append(pair_patch_tokens)
            
            # 后续走 SV Query 的方案
            patch_tokens = pair_patch_tokens_list
            frame_num = meta_data["frames"]
            view_num = meta_data["views"]

            meta_data["frames"] = [1]
            meta_data["views"] = [num_pair]

        else:
            if isinstance(patch_features, (list, tuple)):
                patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
            else:
                patch_tokens = [patch_features[:, :, patch_start_idx:]]

        query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)

        query_nums = query.uv.shape[0]
        if not self.training and self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
            results_list = []
            for q0 in range(0, query_nums, self.chunk_size):
                q1 = min(q0 + self.chunk_size, query_nums)

                chunk_query = BaseQuery(uv=query.uv[q0:q1])
                results_list.append(
                    self.forward_sparse(rgb, pair_idx, patch_tokens, chunk_query, query_rgb, meta_data)
                )
            results = dict()
            for key in results_list[0].keys():
                results[key] = torch.cat([res[key] for res in results_list], dim=1)
        else:
            results = self.forward_sparse(rgb, pair_idx, patch_tokens, query, query_rgb, meta_data)

        if self.time_encoder is not None:
            meta_data["frames"] = frame_num
            meta_data["views"] = view_num

        # Reshape from (B * num_pair, Q, C) -> (B, num_pair, Q, C)
        results = {key: val.view(b, num_pair, *val.shape[-2:]) for key, val in results.items()}
        results["query"] = query
        results["pair_idx"] = pair_idx

        return results
