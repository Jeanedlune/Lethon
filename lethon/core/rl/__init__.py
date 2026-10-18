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

__all__ = [
    "extract_answer_from_text",
    "clip_reasoning_tokens",
    "compute_thinkprune_reward",
    "compute_group_relative_advantage",
    "ThoughtRedundancyAnalyzer",
    "IterativePruningScheduler",
]
