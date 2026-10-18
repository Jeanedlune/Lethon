"""
Lethon Stage 3: Dynamic Promising Token Subspace Masking (RLPT)
"""

from lethon.core.subspace.masking import (
    compute_promising_mask_topk,
    compute_promising_mask_topp,
    compute_promising_mask_minp,
    compute_subspace_projection_mask,
    apply_logit_mask,
    masked_softmax,
    compute_policy_gradient_variance,
    compute_rlpt_grpo_loss,
    PromisingTokenMasker,
)

__all__ = [
    "compute_promising_mask_topk",
    "compute_promising_mask_topp",
    "compute_promising_mask_minp",
    "compute_subspace_projection_mask",
    "apply_logit_mask",
    "masked_softmax",
    "compute_policy_gradient_variance",
    "compute_rlpt_grpo_loss",
    "PromisingTokenMasker",
]
