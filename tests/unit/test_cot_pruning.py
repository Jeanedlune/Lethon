"""
Unit tests for Lethon Stage 4: Sequential Decision MDP & CoT Length Truncation
Berdasarkan arXiv:2504.01296 (ThinkPrune)
"""

import pytest
import numpy as np

from lethon.core.rl.cot_pruning import (
    extract_answer_from_text,
    clip_reasoning_tokens,
    compute_thinkprune_reward,
    compute_group_relative_advantage,
    ThoughtRedundancyAnalyzer,
    IterativePruningScheduler,
)


class TestAnswerExtraction:
    def test_extract_tag_answer(self):
        text = "<think>Let me compute 2+2=4.</think>\n<answer>4</answer>"
        assert extract_answer_from_text(text) == "4"

    def test_extract_tag_answer_multiline(self):
        text = "<think>Reasoning...</think>\n<answer>\n42\n</answer>"
        assert extract_answer_from_text(text) == "42"

    def test_extract_boxed_answer(self):
        text = "Therefore the result is \\boxed{36}."
        assert extract_answer_from_text(text) == "36"

    def test_extract_boxed_nested_braces(self):
        text = "We conclude \\boxed{\\frac{1}{2}}."
        assert extract_answer_from_text(text) == "\\frac{1}{2}"

    def test_extract_final_answer_markdown(self):
        text = "Done.\n\n**Final Answer:** 100\nSome extra footer text."
        assert extract_answer_from_text(text) == "100"

    def test_extract_missing_answer(self):
        text = "Thinking but no conclusion <think>Wait, let me think..."
        assert extract_answer_from_text(text) is None
        assert extract_answer_from_text("") is None


class TestTokenClipping:
    def test_clip_tokens_basic(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        clipped = clip_reasoning_tokens(tokens, 4)
        assert clipped == [1, 2, 3, 4]

    def test_clip_tokens_limit_exceeds_length(self):
        tokens = ["a", "b", "c"]
        clipped = clip_reasoning_tokens(tokens, 10)
        assert clipped == ["a", "b", "c"]

    def test_clip_tokens_zero_limit(self):
        tokens = [10, 20, 30]
        assert clip_reasoning_tokens(tokens, 0) == []

    def test_clip_tokens_negative_limit(self):
        with pytest.raises(ValueError, match="non-negatif"):
            clip_reasoning_tokens([1, 2], -1)


class TestThinkPruneReward:
    def test_reward_correct_within_budget(self):
        text = "<think>Compute 5*5</think><answer>25</answer>"
        tokens = list(range(100))
        reward = compute_thinkprune_reward(
            output_tokens=tokens,
            text_content=text,
            ground_truth="25",
            length_limit=200,
        )
        assert reward == 1.0

    def test_reward_efficiency_bonus(self):
        text = "<think>Compute</think><answer>25</answer>"
        tokens = list(range(50))  # 50 tokens out of 200 -> eff_ratio = 150/200 = 0.75
        reward = compute_thinkprune_reward(
            output_tokens=tokens,
            text_content=text,
            ground_truth="25",
            length_limit=200,
            lambda_eff=0.2,
        )
        assert reward == pytest.approx(1.0 + 0.2 * 0.75)

    def test_reward_overflow_zero_reward(self):
        # ThinkPrune key property: exceeding limit results in 0 reward
        text = "<think>Compute</think><answer>25</answer>"
        tokens = list(range(250))
        reward = compute_thinkprune_reward(
            output_tokens=tokens,
            text_content=text,
            ground_truth="25",
            length_limit=200,
        )
        assert reward == 0.0

    def test_reward_incorrect_answer(self):
        text = "<think>Compute</think><answer>24</answer>"
        tokens = list(range(100))
        reward = compute_thinkprune_reward(
            output_tokens=tokens,
            text_content=text,
            ground_truth="25",
            length_limit=200,
        )
        assert reward == 0.0

    def test_reward_custom_verifier(self):
        text = "<think>Compute</think><answer>40 miles</answer>"
        tokens = list(range(100))
        
        def custom_verifier(ext, gt):
            return "40" in ext and "40" in gt

        reward = compute_thinkprune_reward(
            output_tokens=tokens,
            text_content=text,
            ground_truth="40",
            length_limit=200,
            verifier_fn=custom_verifier,
        )
        assert reward == 1.0


class TestGroupRelativeAdvantage:
    def test_grpo_advantage_standard(self):
        rewards = [1.0, 0.0, 1.0, 0.0]
        adv = compute_group_relative_advantage(rewards)
        assert len(adv) == 4
        # Mean = 0.5, positive for 1.0, negative for 0.0
        assert adv[0] > 0.0
        assert adv[1] < 0.0
        assert pytest.approx(np.mean(adv), abs=1e-5) == 0.0

    def test_grpo_advantage_all_same(self):
        rewards = [1.0, 1.0, 1.0, 1.0]
        adv = compute_group_relative_advantage(rewards)
        np.testing.assert_allclose(adv, np.zeros(4))

    def test_grpo_advantage_dimension_error(self):
        with pytest.raises(ValueError, match="1D"):
            compute_group_relative_advantage(np.ones((2, 2)))


class TestThoughtRedundancyAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return ThoughtRedundancyAnalyzer()

    def test_count_keyword_frequencies(self, analyzer):
        text = (
            "Wait, let me rethink the problem. But wait, hold on! "
            "Let me confirm this theorem. Let me verify the roots. "
            "Therefore, let me compute the discriminant and hence we have the result."
        )
        freqs = analyzer.count_keyword_frequencies(text, token_count=100)
        
        assert freqs["hesitation"]["raw_count"] >= 3
        assert freqs["verification"]["raw_count"] >= 2
        assert freqs["core_computation"]["raw_count"] >= 3
        
        # Check scaling per 1k tokens (100 tokens -> scale is 10x)
        assert freqs["hesitation"]["per_1k_tokens"] == pytest.approx(freqs["hesitation"]["raw_count"] * 10.0)

    def test_estimate_problem_solving_phases(self, analyzer):
        text = (
            "We are given a math problem to solve.\n\n"
            "Wait, let me verify the theorem.\n\n"
            "Alternatively, consider a different approach.\n\n"
            "By computing the derivative, we find x=3.\n\n"
            "<answer>3</answer>"
        )
        phases = analyzer.estimate_problem_solving_phases(text)
        assert phases["understanding_problem"] > 0
        assert phases["verify_local_steps"] > 0
        assert phases["exploring_alternatives"] > 0
        assert phases["finalize_answer"] > 0
        total_prop = sum(phases.values())
        assert pytest.approx(total_prop, abs=1e-3) == 1.0

    def test_compute_redundancy_score(self, analyzer):
        # Text with heavy hesitation
        heavy_hesitation = "Wait... Wait, let me rethink... But wait, hold on... Wait, let me think again."
        score_high = analyzer.compute_redundancy_score(heavy_hesitation, token_count=50)
        
        # Text with direct computation
        clean_computation = "Therefore, let me compute the value. Hence substituting x=2 gives y=4. Done."
        score_low = analyzer.compute_redundancy_score(clean_computation, token_count=50)

        assert 0.0 <= score_low <= 1.0
        assert 0.0 <= score_high <= 1.0
        assert score_high > score_low


class TestIterativePruningScheduler:
    def test_scheduler_init_and_advance(self):
        scheduler = IterativePruningScheduler(schedule=[4000, 3000, 2000])
        assert scheduler.current_limit == 4000
        assert not scheduler.is_finished

        stage2 = scheduler.advance_stage()
        assert stage2 == 3000
        assert scheduler.current_limit == 3000

        stage3 = scheduler.advance_stage()
        assert stage3 == 2000
        assert scheduler.current_limit == 2000

        finished = scheduler.advance_stage()
        assert finished is None
        assert scheduler.is_finished

    def test_scheduler_invalid_schedule(self):
        with pytest.raises(ValueError):
            IterativePruningScheduler(schedule=[2000, 3000])  # Must be descending
        with pytest.raises(ValueError):
            IterativePruningScheduler(schedule=[])

    def test_select_best_checkpoint(self):
        scheduler = IterativePruningScheduler(
            schedule=[4000, 3000, 2000],
            max_relative_acc_drop=0.10,  # Max 10% drop
        )
        baseline_acc = 0.30  # Baseline AIME acc = 30% -> Min allowed = 0.27 (27%)
        
        candidates = [
            {"checkpoint_id": "step_100", "accuracy": 0.29, "avg_length": 6500},
            {"checkpoint_id": "step_200", "accuracy": 0.28, "avg_length": 5200},  # Shorter, satisfies >= 0.27
            {"checkpoint_id": "step_300", "accuracy": 0.25, "avg_length": 3500},  # Too low accuracy (< 0.27)
        ]

        best = scheduler.select_best_checkpoint(candidates, baseline_acc)
        assert best is not None
        assert best["checkpoint_id"] == "step_200"

    def test_select_best_checkpoint_none_qualifying(self):
        scheduler = IterativePruningScheduler(schedule=[4000, 2000], max_relative_acc_drop=0.05)
        baseline_acc = 0.50  # Min allowed = 0.475
        candidates = [
            {"checkpoint_id": "step_1", "accuracy": 0.40, "avg_length": 2000},
            {"checkpoint_id": "step_2", "accuracy": 0.45, "avg_length": 2500},
        ]
        assert scheduler.select_best_checkpoint(candidates, baseline_acc) is None
