"""
Lethon Core RL Module
Menyediakan implementasi Sequential Decision MDP & CoT Length Truncation (Stage 4)
berdasarkan ThinkPrune (arXiv:2504.01296) dan TPRL (arXiv:2603.13394v1).
"""

from lethon.core.rl.cot_pruning import (
    extract_answer_from_text,
    clip_reasoning_tokens,
    compute_thinkprune_reward,
    compute_group_relative_advantage,
    ThoughtRedundancyAnalyzer,
    IterativePruningScheduler,
)
from lethon.core.rl.tprl import (
    TokenWiseAutoencoder,
    SharedAttentionModule,
    TPRLPolicyNetwork,
    TPRLValueNetwork,
    TPRLPruner,
    compute_tprl_reward,
    compute_gae_advantage,
    compute_lfd_loss,
)

from lethon.core.rl.entropy_regulation import (
    SemanticCluster,
    compute_semantic_entropy,
    SemanticEntropyCurriculum,
    compute_token_entropy,
    compute_token_advantage_covariance,
    TokenRegularizationSchedule,
    compute_sent_policy_loss,
)

__all__ = [
    "extract_answer_from_text",
    "clip_reasoning_tokens",
    "compute_thinkprune_reward",
    "compute_group_relative_advantage",
    "ThoughtRedundancyAnalyzer",
    "IterativePruningScheduler",
    "TokenWiseAutoencoder",
    "SharedAttentionModule",
    "TPRLPolicyNetwork",
    "TPRLValueNetwork",
    "TPRLPruner",
    "compute_tprl_reward",
    "compute_gae_advantage",
    "compute_lfd_loss",
    "SemanticCluster",
    "compute_semantic_entropy",
    "SemanticEntropyCurriculum",
    "compute_token_entropy",
    "compute_token_advantage_covariance",
    "TokenRegularizationSchedule",
    "compute_sent_policy_loss",
]

