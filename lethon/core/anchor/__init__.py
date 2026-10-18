"""
Lethon Contextual Semantic Anchor Core Module (Stage 2)
"""

from lethon.core.anchor.aggregator import (
    compute_anchor_indices,
    inject_anchor_embeddings,
    bidirectional_cross_attention,
    extract_anchor_kv_cache,
    SemanticAnchorAggregator,
)

__all__ = [
    "compute_anchor_indices",
    "inject_anchor_embeddings",
    "bidirectional_cross_attention",
    "extract_anchor_kv_cache",
    "SemanticAnchorAggregator",
]
