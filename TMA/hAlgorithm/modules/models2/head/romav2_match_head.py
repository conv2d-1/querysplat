from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from hAlgorithm.modules.models2.external.romav2.models.romav2.features import FineFeatures
from hAlgorithm.modules.models2.external.romav2.models.romav2.matcher import _compute_match_embeddings, _compute_head_preds
from hAlgorithm.modules.models2.external.romav2.models.romav2.refiner import Refiners
from hAlgorithm.modules.models2.external.romav2.models.romav2.romav2 import _interpolate_warp_and_confidence


from hAlgorithm.modules.models2.external.romav2.models.romav2.geometry import get_normalized_grid
from typing import Literal
from hAlgorithm.modules.models2.external.romav2.models.romav2.device import device
from hAlgorithm.modules.models2.external.romav2.models.romav2.vit import ViTModel, vit_from_name
from hAlgorithm.modules.models2.external.romav2.models.romav2.types import HeadType, MatcherStyle
from hAlgorithm.modules.models2.external.romav2.models.romav2.dpt import DPTHead

logger = logging.getLogger(__name__)

class Matcher(nn.Module):
    @dataclass(frozen=False)
    class Cfg:
        mv_vit: ViTModel = "vit_base"
        mv_vit_use_rope: bool = True
        mv_vit_position_mode: Literal["same"] = "same"
        mv_vit_attention_mode: Literal["alternating"] = "alternating"
        head: HeadType = "dpt-no-pos"
        # NOTE: 0.2 in RoMa
        temp: float = 0.1
        # NOTE: 8 in RoMa
        scale: float = 1
        # DPT dim
        dim: int = 1024
        warp_dim: int = 2
        confidence_dim: int = 1
        # mv config
        num_feature_layers: int = 2
        feat_dim: int = 1024
        embed_layers_idx = None
        
        pos_emb_dim: int = 1024
        enable_amp: bool = True
        style: MatcherStyle = "romav2"
        # ufm uses pos embedings for view B and no attn
        pos_embed_rope_rescale_coords: float | None = None

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        omega = 2 * torch.pi * torch.randn(cfg.dim // 2, 2)
        # self.omega = nn.Buffer(omega)
        self.register_buffer('omega', omega)
        
        # self.scale = nn.Buffer(torch.tensor(cfg.scale))
        self.register_buffer('scale', torch.tensor(cfg.scale))
        

        # self.temp = nn.Buffer(torch.tensor(cfg.temp))
        self.register_buffer('temp', torch.tensor(cfg.temp))
        if cfg.mv_vit is not None:
            self.mv_vit = vit_from_name(
                cfg.mv_vit,
                device=device,
                in_dim=cfg.feat_dim * cfg.num_feature_layers,
                out_dim=cfg.dim,
                multiview=True,
                use_rope=cfg.mv_vit_use_rope,
                mv_position_mode=cfg.mv_vit_position_mode,
                mv_attention_mode=cfg.mv_vit_attention_mode,
                pos_embed_rope_rescale_coords=cfg.pos_embed_rope_rescale_coords,
            )
        else:
            self.mv_vit = None
            logging.info(f"Using Embed Layers {self.cfg.embed_layers_idx} for mv_embed")
        self.head = DPTHead(
            dim_in=cfg.dim,
            out_dim=cfg.warp_dim + cfg.confidence_dim,
            pos_embed=False,
            feature_only=False,
            down_ratio=4,
        )

    def forward(
        self,
        f_list_A: list[torch.Tensor],
        f_list_B: list[torch.Tensor],
        img_A: torch.Tensor,
        img_B: torch.Tensor,
        bidirectional: bool,
    ):
        preds = {}
        f_A = torch.cat(f_list_A, dim=-1)
        f_B = torch.cat(f_list_B, dim=-1)
        B, H_A, W_A, D_feat = f_A.shape
        B, H_B, W_B, D_feat = f_B.shape
        x = get_normalized_grid(B, H_B, W_B)
        x_emb = nn.functional.linear(
            x.reshape(B, H_B * W_B, 2), self.scale * self.omega
        ).reshape(B, H_B, W_B, -1)
        pos_emb_grid = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

        if self.mv_vit is not None:
            assert D_feat == self.cfg.feat_dim * self.cfg.num_feature_layers, (
                "Feature dimension mismatch"
            )
            if self.cfg.enable_amp:
                with torch.autocast(device.type, torch.bfloat16, enabled=self.cfg.enable_amp):
                    f_mv_AB = self.mv_vit(torch.stack((f_A, f_B), dim=1))[
                        "x_norm_patchtokens"
                    ].reshape(B, 2, H_A, W_A, self.cfg.dim)
                    f_mv_A = f_mv_AB[:, 0]
                    f_mv_B = f_mv_AB[:, 1]
                f_mv_A = f_mv_A.float()
                f_mv_B = f_mv_B.float()
            else:
                f_mv_AB = self.mv_vit(torch.stack((f_A, f_B), dim=1))[
                    "x_norm_patchtokens"
                ].reshape(B, 2, H_A, W_A, self.cfg.dim)
                f_mv_A = f_mv_AB[:, 0]
                f_mv_B = f_mv_AB[:, 1]
        else:
            if self.cfg.embed_layers_idx is not None:
                f_mv_A = torch.cat([f_list_A[i] for i in self.cfg.embed_layers_idx], dim=-1)
                f_mv_B = torch.cat([f_list_B[i] for i in self.cfg.embed_layers_idx], dim=-1)
            else:
                f_mv_A = f_A
                f_mv_B = f_B

        assert H_A == H_B and W_A == W_B, "H_A and W_A must be equal to H_B and W_B"
        attn_AB_logits, attn_AB, match_emb_AB = _compute_match_embeddings(
            f_A=f_mv_A,
            f_B=f_mv_B,
            pos_emb_grid=pos_emb_grid,
            temp=self.temp,
            B=B,
            H_A=H_A,
            W_A=W_A,
            H_B=H_B,
            W_B=W_B,
        )
        warp_AB, confidence_AB = _compute_head_preds(
            f_list_A=f_list_A,
            match_emb_AB=match_emb_AB,
            f_mv_A=f_mv_A if self.mv_vit is not None else None,
            img_A=img_A,
            img_B=img_B,
            head=self.head,
            enable_amp=self.cfg.enable_amp,
        )

        if bidirectional:
            attn_BA_logits, attn_BA, match_emb_BA = _compute_match_embeddings(
                f_A=f_mv_B,
                f_B=f_mv_A,
                pos_emb_grid=pos_emb_grid,
                temp=self.temp,
                B=B,
                H_A=H_B,
                W_A=W_B,
                H_B=H_A,
                W_B=W_A,
            )
            warp_BA, confidence_BA = _compute_head_preds(
                f_list_A=f_list_B,
                match_emb_AB=match_emb_BA,
                f_mv_A=f_mv_B,
                img_A=img_B,
                img_B=img_A,
                head=self.head,
                enable_amp=self.cfg.enable_amp,
            )
        else:
            match_emb_BA = None
            attn_BA = None
            attn_BA_logits = None
            warp_BA = None
            confidence_BA = None

        preds["attn_AB_logits"] = attn_AB_logits
        preds["attn_AB"] = attn_AB
        preds["warp_AB"] = warp_AB
        preds["confidence_AB"] = confidence_AB

        preds["attn_BA_logits"] = attn_BA_logits
        preds["attn_BA"] = attn_BA
        preds["warp_BA"] = warp_BA
        preds["confidence_BA"] = confidence_BA
        return preds


class RoMaV2MatchHead(nn.Module):
    @dataclass(frozen=False)
    class Cfg:
        matcher: Matcher.Cfg = Matcher.Cfg()
        refiners: Refiners.Cfg = Refiners.Cfg()
        refiner_features: FineFeatures.Cfg = FineFeatures.Cfg()
        anchor_width: int = 512
        anchor_height: int = 512
        name: str = "RoMa v2"
        pretrained: str = "/mnt/netdata/Team/AI/personal/wsz/checkpoints/romav2/romav2_head.pt"
        denormalize: bool = False
        coarse_only = False
        use_feat_layers = None
        patch_size = 16
        freeze_modules : list[str] = None


    def __init__(self, cfg: Cfg | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            # default
            cfg = RoMaV2MatchHead.Cfg()
        # update cfg
        def update_nested(obj, updates):
            for k, v in updates.items():
                if not hasattr(obj, k):
                    logging.warning(f"Ignoring config {k}: {v}")
                    continue
                current = getattr(obj, k)
                if is_dataclass(current) and isinstance(v, dict):
                    # 递归更新嵌套 dataclass
                    update_nested(current, v)
                else:
                    # 直接赋值
                    setattr(obj, k, v)
        if kwargs:
            update_nested(cfg, kwargs)

        import os
        self.matcher = Matcher(cfg.matcher)
        self.cfg = cfg
        if not self.cfg.coarse_only:
            self.anchor_width = cfg.anchor_width
            self.anchor_height = cfg.anchor_height
            self.refiners = Refiners(cfg.refiners)
            self.refiner_features = FineFeatures(cfg.refiner_features)
        
        self.name = cfg.name
        
        if cfg.pretrained is not None and os.path.exists(cfg.pretrained):
            logging.info(f"Loading pretrained : {cfg.pretrained}")
            weights = torch.load(cfg.pretrained)
            res = self.load_state_dict(weights, strict=False)
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")
        self.denormalize = self.cfg.denormalize
        logger.info(f"{self.name} initialized.")
        self.freeze()

    def freeze(self):
        """Freeze specified modules."""
        if self.cfg.freeze_modules is None:
            return
        for module_name in self.cfg.freeze_modules:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                if module is not None:
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                logging.warning(f"Module {module_name} not found for freezing.")

    def _course_match(
        self,
        feat_A: list[torch.Tensor],
        feat_B: list[torch.Tensor],
        bidirectional: bool,
    ):
        predictions = OrderedDict()
        
        # coarse matching
        matcher_output = self.matcher(
            feat_A, feat_B, img_A=None, img_B=None, bidirectional=bidirectional
        )
        predictions["matcher"] = matcher_output
        return predictions

    def _get_refine_feat(self, x):
        if self.denormalize:
            x = (x + 1) * 0.5
        refine_feat = self.refiner_features(x)
        return refine_feat

    def _refine_match(
        self,
        predictions,
        refiner_features_A: torch.Tensor,
        refiner_features_B: torch.Tensor,
        img_H,
        img_W,
        bidirectional: bool,
        high_resolution: bool,
    ) -> dict:
        matcher_output = predictions["matcher"]
        warp_AB, confidence_AB = (
            matcher_output["warp_AB"],
            matcher_output["confidence_AB"],
        )
        if bidirectional:
            warp_BA, confidence_BA = (
                matcher_output["warp_BA"],
                matcher_output["confidence_BA"],
            )
        else:
            warp_BA = None
            confidence_BA = None
        # refine warp
        H, W = img_H, img_W
        scale_factor = torch.tensor(
            (W / self.anchor_width, H / self.anchor_height),
        )
        for patch_size_str, refiner in self.refiners.items():
            patch_size = int(patch_size_str)
            zero_out_precision = (
                (not high_resolution) and patch_size == 4
            )
            warp_AB, confidence_AB = _interpolate_warp_and_confidence(
                warp=warp_AB,
                confidence=confidence_AB,
                H=H,
                W=W,
                patch_size=patch_size,
                zero_out_precision=zero_out_precision,
            )
            if bidirectional:
                warp_BA, confidence_BA = _interpolate_warp_and_confidence(
                    warp=warp_BA,
                    confidence=confidence_BA,
                    H=H,
                    W=W,
                    patch_size=patch_size,
                    zero_out_precision=zero_out_precision,
                )
            f_patch_A = refiner_features_A[patch_size]
            f_patch_B = refiner_features_B[patch_size]
            scale_factor = scale_factor.to(f_patch_A.device)
            refiner_output_AB = refiner(
                f_A=f_patch_A,
                f_B=f_patch_B,
                prev_warp=warp_AB,
                prev_confidence=confidence_AB,
                scale_factor=scale_factor,
            )
            if bidirectional:
                refiner_output_BA = refiner(
                    f_A=f_patch_B,
                    f_B=f_patch_A,
                    prev_warp=warp_BA,
                    prev_confidence=confidence_BA,
                    scale_factor=scale_factor,
                )
            else:
                refiner_output_BA = None
            predictions[f"refiner_{patch_size}_AB"] = refiner_output_AB
            predictions[f"refiner_{patch_size}_BA"] = refiner_output_BA
            warp_AB, confidence_AB = (
                refiner_output_AB["warp"],
                refiner_output_AB["confidence"],
            )
            if bidirectional:
                warp_BA, confidence_BA = (
                    refiner_output_BA["warp"],
                    refiner_output_BA["confidence"],
                )
            predictions["warp_AB"] = warp_AB
            predictions["confidence_AB"] = confidence_AB
            if bidirectional:
                predictions["warp_BA"] = warp_BA
                predictions["confidence_BA"] = confidence_BA
            else:
                predictions["warp_BA"] = None
                predictions["confidence_BA"] = None
        return predictions

    def _forward_impl(
        self, 
        images: torch.Tensor, 
        patch_feats: list[torch.Tensor],
        refine_feats: dict,
        bidirectional: bool,
        pair_idx=None,
    ):
        b, n, c, h, w = images.shape
        if pair_idx is None:
            pair_idx = [(0, i) for i in range(1, n)]
            # pair_idx = [(0, 1)] # default to one pair
            assert n >= 2, "Please Specify pair index when n > 2"
        
        # organize feat to pair
        feat_A, feat_B = [], []
        for pair in pair_idx:
            feat_A.append(
                [feat[:, pair[0], ...] for feat in patch_feats]
            )
            feat_B.append(
                [feat[:, pair[1], ...] for feat in patch_feats]
            )
        num_feat = len(patch_feats)
        feat_A = [
            # B * Pair, P, C
            torch.cat([feat[i] for feat in feat_A], dim=0) for i in range(num_feat)
        ]
        feat_B = [
            # B * Pair, P, C
            torch.cat([feat[i] for feat in feat_B], dim=0) for i in range(num_feat)
        ]
        # coarse match
        predictions = self._course_match(feat_A, feat_B, bidirectional=bidirectional)

        num_pair = len(pair_idx)

        def reshape_to_pair(val):
            return val.reshape(b, num_pair, *val.shape[1:]) if val is not None else None

        if self.cfg.coarse_only:
            output = {
                "coarse":{
                    "warp_AB": reshape_to_pair(predictions["matcher"]["warp_AB"]),
                    "confidence_AB": reshape_to_pair(predictions["matcher"]["confidence_AB"]),
                },
                "pair_idx": pair_idx,
            }
            return output

        # get && organize refine_feat to pair
        refine_feat_A = {}
        refine_feat_B = {}
        for scale, val in refine_feats.items():
            # b, n, h, w, c
            scaled_refine_feat = val.reshape(b, n, *val.shape[1:])
            # b * pair, h, w, c
            refine_feat_A[scale] = torch.cat([
                scaled_refine_feat[:, pair[0], ...] for pair in pair_idx
            ], dim=0)
            refine_feat_B[scale] = torch.cat([
                scaled_refine_feat[:, pair[1], ...] for pair in pair_idx
            ], dim=0)
        
        # refine predictions
        predictions = self._refine_match(
            predictions, refine_feat_A, refine_feat_B, h, w,
            bidirectional=bidirectional, high_resolution=True
        )
        
        output = {
            "coarse": {},
            "final": {},
        }
        output["pair_idx"] = pair_idx
        # sort output shape to B, Pair H, W, 2
        
        # warp and confidence
        output["coarse"]["warp_AB"] = reshape_to_pair(predictions["matcher"]["warp_AB"])
        output["coarse"]["confidence_AB"] = reshape_to_pair(predictions["matcher"]["confidence_AB"])
        output["final"]["warp_AB"] = reshape_to_pair(predictions["warp_AB"])
        output["final"]["confidence_AB"] = reshape_to_pair(predictions["confidence_AB"])
        
        # refiner output
        for key, val in predictions.items():
            if key.startswith("refiner") and key.endswith("_AB"):
                output[key[:-3]] = {
                    "warp_AB": reshape_to_pair(val["warp"]),
                    "confidence_AB": reshape_to_pair(val["confidence"]),
                }
        
        return output

    def forward(
        self, 
        images: torch.Tensor, 
        patch_feats: list[torch.Tensor],
        bidirectional: bool = False,
        pair_idx=None,
        patch_start_idx=None,
        patch_h=None, patch_w=None,
        meta_data=None,
        **kwargs
    ):
        b, n, c, h, w = images.shape

        # take only patch feats
        if patch_start_idx is not None:
            patch_feats = [feat[..., patch_start_idx:, :] for feat in patch_feats]
        
        if self.cfg.use_feat_layers is not None:
            patch_feats = [patch_feats[i] for i in self.cfg.use_feat_layers]

        if patch_feats[0].ndim == 5:
            pass
        elif patch_h is not None and patch_w is not None:
            patch_feats = [
                # B,V,L,C -> # B,V,ph,pw,C
                feat.reshape(*feat.shape[:2], patch_h, patch_w, feat.shape[-1])
                for feat in patch_feats
            ]
        else:
            assert meta_data is not None, "Please provide patch height width info"
            patch_w = meta_data["input_width"][0] // self.cfg.patch_size
            patch_h = meta_data["input_height"][0] // self.cfg.patch_size
            patch_feats = [
                # B,V,L,C -> # B,V,ph,pw,C
                feat.reshape(*feat.shape[:2], patch_h, patch_w, feat.shape[-1])
                for feat in patch_feats
            ]

        if not self.cfg.coarse_only:
            refine_feats = self._get_refine_feat(images.reshape(b*n, c, h, w)) # b*n, h, w, c
        else:
            refine_feats = None
        return self._forward_impl(images, patch_feats, refine_feats, bidirectional, pair_idx)
