"""WFM pipeline v2 for Envision4D-inspired dual-query 4DGS."""
from __future__ import annotations

import logging

from hAlgorithm.modules.models2.sdk.query_dual_4dgs_v2 import MVQueryDual4DGSV2

from .mvfr_query_dual_4dgs import WFMQueryDual4DGSPipeline

logger = logging.getLogger(__name__)


class WFMQueryDual4DGSPipelineV2(WFMQueryDual4DGSPipeline):
    """V2 pipeline: passes GT depth into DGS render loss and expects MVQueryDual4DGSV2."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model = getattr(self, "model", None)
        if model is not None and not isinstance(model, MVQueryDual4DGSV2):
            logger.warning(
                "WFMQueryDual4DGSPipelineV2 expects MVQueryDual4DGSV2; got %s",
                type(model).__name__,
            )

    def get_dgs_render_loss_kwargs(
        self,
        image,
        target_local_depth=None,
        target_depth_mask=None,
        scale=None,
        name=None,
        **kwargs,
    ) -> dict:
        if target_local_depth is None:
            return {}
        b, n = image.shape[:2]
        depth = target_local_depth
        mask = target_depth_mask
        if depth.shape[0] == b * n:
            depth = depth.view(b, n, *depth.shape[1:])
        if mask is not None and mask.shape[0] == b * n:
            mask = mask.view(b, n, *mask.shape[1:])
        return dict(target_depth=depth, target_depth_mask=mask)
