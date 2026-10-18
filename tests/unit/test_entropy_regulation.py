"""
Unit tests for Entropy-Guided Regulation and Curriculum Learning (SENT)
Testing arXiv:2512.04359v3 formulas and implementation.
"""

import pytest
import numpy as np

from lethon.core.rl.entropy_regulation import (
    SemanticCluster,
    compute_semantic_entropy,
    SemanticEntropyCurriculum,
    compute_token_entropy,
    compute_token_advantage_covariance,
    TokenRegularizationSchedule,
    compute_sent_policy_loss,
)


class TestSemanticEntropy:
    def test_identical_responses_zero_entropy(self):
        responses = ["x = 4", "x = 4", "x = 4", "x = 4"]
        probabilities = [0.25, 0.25, 0.25, 0.25]
        h_se, clusters = compute_semantic_entropy(responses, probabilities)
        assert len(clusters) == 1
        assert pytest.approx(h_se, abs=1e-5) == 0.0

    def test_uniform_diverse_responses_max_entropy(self):
        responses = ["ans = 1", "ans = 2", "ans = 3", "ans = 4"]
        probabilities = [0.25, 0.25, 0.25, 0.25]
        h_se, clusters = compute_semantic_entropy(responses, probabilities)
        assert len(clusters) == 4
        # H = - 4 * (0.25 * log(0.25)) = log(4) approx 1.38629
        expected_h = np.log(4)
        assert pytest.approx(h_se, rel=1e-3) == expected_h

    def test_custom_answer_extractor(self):
        responses = [
            "Therefore, the solution is 42.",
            "Final result is 42!",
            "I obtain 99 as answer.",
        ]
        probabilities = [0.4, 0.4, 0.2]

        def simple_extractor(text: str) -> str:
            if "42" in text:
                return "42"
            return "99"

        h_se, clusters = compute_semantic_entropy(
            responses, probabilities, answer_extractor_fn=simple_extractor
        )
        assert len(clusters) == 2
        # Cluster '42' has prob 0.8, cluster '99' has prob 0.2
        # H = - (0.8*log(0.8) + 0.2*log(0.2)) approx 0.5004
        expected_h = -(0.8 * np.log(0.8) + 0.2 * np.log(0.2))
        assert pytest.approx(h_se, rel=1e-3) == expected_h

    def test_empty_responses(self):
        h_se, clusters = compute_semantic_entropy([], [])
        assert h_se == 0.0
        assert len(clusters) == 0

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            compute_semantic_entropy(["a"], [0.5, 0.5])


class TestSemanticEntropyCurriculum:
    def test_two_stage_partition(self):
        dataset = [
            {"id": "q1", "semantic_entropy": 1.5},
            {"id": "q2", "semantic_entropy": 0.2},
            {"id": "q3", "semantic_entropy": 0.8},
            {"id": "q4", "semantic_entropy": 2.1},
        ]
        curriculum = SemanticEntropyCurriculum(num_stages=2)
        stages = curriculum.partition_dataset(dataset)
        assert len(stages) == 2
        # Stage 1: lowest entropy (q2: 0.2, q3: 0.8)
        assert [item["id"] for item in stages[0]] == ["q2", "q3"]
        # Stage 2: highest entropy (q1: 1.5, q4: 2.1)
        assert [item["id"] for item in stages[1]] == ["q1", "q4"]

    def test_empty_dataset(self):
        curriculum = SemanticEntropyCurriculum(num_stages=3)
        stages = curriculum.partition_dataset([])
        assert len(stages) == 3
        assert all(len(s) == 0 for s in stages)

    def test_invalid_stages(self):
        with pytest.raises(ValueError):
            SemanticEntropyCurriculum(num_stages=0)


class TestTokenEntropyAndCovariance:
    def test_deterministic_token_entropy_near_zero(self):
        # 1-hot logit
        logits = np.array([[100.0, -100.0, -100.0]])
        h_t = compute_token_entropy(logits, is_logits=True)
        assert pytest.approx(h_t[0], abs=1e-4) == 0.0

    def test_uniform_token_entropy_max(self):
        logits = np.array([[1.0, 1.0, 1.0, 1.0]])
        h_t = compute_token_entropy(logits, is_logits=True)
        assert pytest.approx(h_t[0], rel=1e-3) == np.log(4)

    def test_token_advantage_covariance_formula(self):
        # Cov_t = (log_pi - mean_log_pi) * (A_t - mean_A)
        log_probs = np.array([-0.1, -0.5, -2.0, -1.0])
        advantages = np.array([2.0, 1.0, -1.0, -2.0])
        cov = compute_token_advantage_covariance(log_probs, advantages)
        assert len(cov) == 4

        # Manual check
        mean_lp = np.mean(log_probs)
        mean_adv = np.mean(advantages)
        expected_cov = (log_probs - mean_lp) * (advantages - mean_adv)
        np.testing.assert_allclose(cov, expected_cov, rtol=1e-5)

    def test_covariance_shape_mismatch(self):
        with pytest.raises(ValueError):
            compute_token_advantage_covariance(np.array([1.0]), np.array([1.0, 2.0]))


class TestTokenRegularizationSchedule:
    def test_hierarchical_beta_assignment(self):
        # 10 tokens with varying entropy and covariance
        # Low entropy: below 80th percentile
        entropies = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.5, 2.0])
        # tau_H (80th percentile) will be around 0.84
        # High covariance for token 0 and token 1
        covariances = np.array([0.001, 0.0005, 0.00001, 0.0, -0.001, 0.0, 0.0, 0.0, 0.05, 0.1])

        sched = TokenRegularizationSchedule(
            tau_H_percentile=80.0,
            tau_cov=0.0002,
            beta_high=2.0,
            beta_low=0.5,
            beta_base=0.0,
        )
        beta_con, stats = sched.compute_non_uniform_kl_coefficients(entropies, covariances)

        assert len(beta_con) == 10
        # Token 0: low entropy + high cov (>0.0002) -> beta_high (2.0)
        assert beta_con[0] == 2.0
        # Token 1: low entropy + high cov (>0.0002) -> beta_high (2.0)
        assert beta_con[1] == 2.0
        # Token 2: low entropy + low cov (<=0.0002) -> beta_low (0.5)
        assert beta_con[2] == 0.5
        # Token 8 and 9: high entropy (> tau_H) -> beta_base (0.0), even though cov is high!
        assert beta_con[8] == 0.0
        assert beta_con[9] == 0.0

        assert stats["num_high_cov_low_entropy"] == 2
        assert stats["total_tokens"] == 10


class TestSENTPolicyLoss:
    def test_sent_policy_loss_computation(self):
        log_probs = np.array([-0.2, -0.4, -0.8])
        old_log_probs = np.array([-0.2, -0.4, -0.8])  # ratio = 1.0
        ref_log_probs = np.array([-0.5, -0.3, -1.0])
        advantages = np.array([1.5, -0.5, 2.0])
        beta_con = np.array([2.0, 0.5, 0.0])

        loss, metrics = compute_sent_policy_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            beta_con=beta_con,
            clip_eps=0.2,
        )

        assert isinstance(loss, float)
        assert "loss" in metrics
        assert "mean_pg_objective" in metrics
        assert "mean_kl_penalty" in metrics
        # ratio is 1, so pg_obj = advantages
        expected_pg = np.mean(advantages)
        assert pytest.approx(metrics["mean_pg_objective"], rel=1e-4) == expected_pg
