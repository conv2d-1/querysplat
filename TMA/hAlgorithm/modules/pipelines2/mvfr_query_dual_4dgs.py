"""WFM pipeline with dual-query 4DGS training (sparse motion + dense Gaussian)."""
from __future__ import annotations

import logging

from hAlgorithm.modules.models2.sdk.query_dual_4dgs import MVQueryDual4DGS

from .mvfr_query_v4 import WFMQueryPipeline

logger = logging.getLogger(__name__)


class WFMQueryDual4DGSPipeline(WFMQueryPipeline):
    """Train motion on sparse traj queries and 4DGS on a separate dense UV grid.

    Requires ``MVQueryDual4DGS`` as ``model.type``.  The pipeline still builds
    the sparse ``query`` from GT trajectories (``get_motion_inputs``); the model
    automatically constructs the per-pixel ``gaussian_query`` for the Gaussian head.

    Existing motion / depth / camera losses are unchanged.  The 4DGS render loss
    reads ``gs_*`` tensors produced from the dense query branch only.
    """

    def __init__(
        self,
        enable_dense_gaussian_query: bool = True,
        dense_gaussian_dynamic_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_dense_gaussian_query = enable_dense_gaussian_query
        self.dense_gaussian_dynamic_only = dense_gaussian_dynamic_only

        model = getattr(self, "model", None)
        if model is not None and not isinstance(model, MVQueryDual4DGS):
            logger.warning(
                "WFMQueryDual4DGSPipeline expects MVQueryDual4DGS; got %s",
                type(model).__name__,
            )
        if model is not None and hasattr(model, "enable_dual_gaussian_query"):
            model.enable_dual_gaussian_query = enable_dense_gaussian_query

    def _should_build_dense_gaussian_query(self, is_dynamic: bool) -> bool:
        if not self.enable_dense_gaussian_query:
            return False
        if self.dense_gaussian_dynamic_only and not is_dynamic:
            return False
        return True

    def forward(self, batch, **kwargs):
        """Identical to :class:`WFMQueryPipeline` — dense query is built inside the model."""
        return super().forward(batch, **kwargs)
