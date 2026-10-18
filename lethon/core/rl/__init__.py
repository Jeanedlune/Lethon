"""
Lethon Core RL Module
Menyediakan implementasi Sequential Decision MDP & Token Pruning (Stage 4)
berdasarkan TPRL (arXiv:2603.13394v1), SENT (arXiv:2512.04359v3),
dan MAPPO Markov Game (arXiv:2503.23459).
"""

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
from lethon.core.rl.mappo_pruning import (
    MAPPOActorNetwork,
    MAPPOCriticNetwork,
    compute_mappo_token_rewards,
    apply_attention_mask_pruning,
    compute_mappo_gae,
    compute_mappo_policy_loss,
    compute_mappo_value_loss,
)

__all__ = [
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
    "MAPPOActorNetwork",
    "MAPPOCriticNetwork",
    "compute_mappo_token_rewards",
    "apply_attention_mask_pruning",
    "compute_mappo_gae",
    "compute_mappo_policy_loss",
    "compute_mappo_value_loss",
]
