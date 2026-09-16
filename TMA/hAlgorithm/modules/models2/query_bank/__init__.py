"""
Query Bank Module
==================
Utilities for query construction in sparse motion prediction.

This module contains:
- QueryBatch: Typed container for D4RT queries
- QueryConfig: Configuration for query construction  
- QueryBuilder: Constructs all query batches for training/inference
- create_queries_from_trajectories: Creates queries from GT trajectory annotations
- BaseQuery: Base query class for single-view queries
- RandomSampler: Random sampler for query selection
- QueryBank3/4/5: Query bank implementations for different use cases
"""

from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.modules.models2.query_bank.sampler import RandomSampler
from hAlgorithm.modules.models2.query_bank.single_view import (
    QueryBank5,
)
from hAlgorithm.modules.models2.query_bank.motion_query import (
    QueryBatch,
    QueryConfig,
    QueryBuilder,
    create_queries_from_trajectories,
)

__all__ = [
    # Base query classes
    "BaseQuery",
    "RandomSampler",
    # Single-view query banks
    "QueryBank5",
    # Motion query utilities
    "QueryBatch",
    "QueryConfig",
    "QueryBuilder",
    "create_queries_from_trajectories",
]
