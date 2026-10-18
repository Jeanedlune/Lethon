"""
Unit tests for Lethon Cross-Stage MAPPO & Markov Game Token Pruning.
Berdasarkan paper: arXiv:2503.23459 (RL4EViT)
"""

import pytest
import numpy as np

from lethon.core.rl.mappo_pruning import (
    MAPPOActorNetwork,
    MAPPOCriticNetwork,
    compute_mappo_token_rewards,
    apply_attention_mask_pruning,
    compute_mappo_gae,
    compute_mappo_policy_loss,
    compute_mappo_value_loss,
)


class TestMAPPOComponents:
    """Tests for Actor, Critic, and Reward calculation in MAPPO token pruning."""

    def test_actor_dimensions_and_probabilities(self):
        d_in = 64
        d_hidden = 256
        num_tokens = 16
        actor = MAPPOActorNetwork(d_in=d_in, d_hidden=d_hidden, seed=42)

        x = np.random.randn(num_tokens, d_in)
        logits = actor.forward_logits(x)
        assert logits.shape == (num_tokens, 2)

        probs = actor.get_action_probs(x)
        assert probs.shape == (num_tokens, 2)
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(num_tokens), rtol=1e-5)

        actions_det, selected_probs = actor.sample_actions(x, deterministic=True)
        assert actions_det.shape == (num_tokens,)
        assert set(np.unique(actions_det)).issubset({0, 1})
        assert len(selected_probs) == num_tokens

    def test_critic_dimensions_and_broadcasting(self):
        d_token = 64
        d_global = 64
        d_hidden = 256
        num_tokens = 16
        critic = MAPPOCriticNetwork(d_token=d_token, d_global=d_global, d_hidden=d_hidden, seed=42)

        s_local = np.random.randn(num_tokens, d_token)
        s_global = np.random.randn(d_global)  # 1D global summary

        values = critic.forward_value(s_local, s_global)
        assert values.shape == (num_tokens,)

        # Test batch broadcasting (B, N, d) and (B, d)
        batch_size = 4
        s_local_batch = np.random.randn(batch_size, num_tokens, d_token)
        s_global_batch = np.random.randn(batch_size, d_global)
        values_batch = critic.forward_value(s_local_batch, s_global_batch)
        assert values_batch.shape == (batch_size, num_tokens)

    def test_reward_formulation(self):
        """
        Verify Eq (1) & (2):
        r_1i = 0 if A=0, -1 if A=1
        r_2i = gamma / n_l
        R_total = alpha * r_1i + beta * r_2i
        """
        actions = np.array([0, 1, 1, 0, 1])  # 3 preserved (n_l = 3), 2 pruned
        task_acc = 1.0
        alpha = 0.8
        beta = 1.0

        r1, r2, R_total = compute_mappo_token_rewards(
            actions=actions,
            task_accuracy=task_acc,
            alpha=alpha,
            beta=beta,
        )

        # r1 check
        expected_r1 = np.array([0.0, -1.0, -1.0, 0.0, -1.0])
        np.testing.assert_allclose(r1, expected_r1)

        # r2 check: gamma / 3 = 1.0 / 3.0 = 0.333333
        expected_r2 = np.full(5, 1.0 / 3.0)
        np.testing.assert_allclose(r2, expected_r2, rtol=1e-4)

        # R_total check
        expected_R = alpha * expected_r1 + beta * expected_r2
        np.testing.assert_allclose(R_total, expected_R, rtol=1e-4)

    def test_attention_mask_pruning(self):
        """Verify -1000.0 mask bias on pruned tokens."""
        N = 4
        attn_scores = np.zeros((N, N))
        token_mask = np.array([1, 0, 1, 1])  # token 1 is pruned

        masked = apply_attention_mask_pruning(attn_scores, token_mask, mask_bias=-1000.0)

        # Tokens preserved with each other (0, 2, 3) must have bias 0.0
        assert masked[0, 0] == 0.0
        assert masked[0, 2] == 0.0
        assert masked[2, 3] == 0.0

        # Token 1 connections must have -1000.0
        assert masked[0, 1] == -1000.0
        assert masked[1, 0] == -1000.0
        assert masked[1, 1] == -1000.0

    def test_gae_and_loss_functions(self):
        num_layers = 3
        num_tokens = 5
        rewards = [np.random.randn(num_tokens) for _ in range(num_layers)]
        values = [np.random.randn(num_tokens) for _ in range(num_layers + 1)]

        advantages, returns = compute_mappo_gae(rewards, values, gamma_disc=0.99, lambda_gae=0.95)
        assert len(advantages) == num_layers
        assert len(returns) == num_layers
        assert advantages[0].shape == (num_tokens,)

        # Policy loss check
        log_probs_new = np.log(np.array([0.8, 0.2, 0.7]))
        log_probs_old = np.log(np.array([0.7, 0.3, 0.6]))
        adv = np.array([1.5, -0.5, 0.8])
        p_loss = compute_mappo_policy_loss(log_probs_new, log_probs_old, adv, clip_eps=0.2)
        assert isinstance(p_loss, float)

        # Value loss check
        v_new = np.array([1.2, 0.4, 2.1])
        v_old = np.array([1.0, 0.5, 2.0])
        ret = np.array([1.5, 0.2, 2.5])
        v_loss = compute_mappo_value_loss(v_new, v_old, ret, clip_eps=0.2)
        assert isinstance(v_loss, float)
        assert v_loss >= 0.0
