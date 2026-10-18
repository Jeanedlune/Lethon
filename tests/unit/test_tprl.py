"""
Unit tests for Lethon Stage 4 TPRL module (arXiv:2603.13394v1).
"""

import numpy as np
import pytest

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


class TestTokenWiseAutoencoder:
    def test_dimensions_and_reconstruction(self):
        N = 32
        d_v = 128
        d_l = 16
        ae = TokenWiseAutoencoder(d_v=d_v, d_hidden=64, d_l=d_l, seed=123)

        H = np.random.randn(N, d_v)
        Z = ae.encode(H)
        assert Z.shape == (N, d_l)

        H_rec = ae.decode(Z)
        assert H_rec.shape == (N, d_v)

        loss = ae.reconstruction_loss(H)
        assert loss > 0.0
        assert isinstance(loss, float)

    def test_freeze_unfreeze(self):
        ae = TokenWiseAutoencoder(d_v=64, d_hidden=32, d_l=8)
        assert not ae.frozen
        ae.freeze()
        assert ae.frozen
        ae.unfreeze()
        assert not ae.frozen

    def test_single_vector_input(self):
        ae = TokenWiseAutoencoder(d_v=64, d_hidden=32, d_l=8)
        h = np.random.randn(64)
        z = ae.encode(h)
        assert z.shape == (1, 8)
        h_rec = ae.decode(z[0])
        assert h_rec.shape == (1, 64)


class TestSharedAttentionAndPolicy:
    def test_shared_attention_forward(self):
        K_t = 20
        d_l = 32
        d_q = 32
        d_model = 32
        module = SharedAttentionModule(d_l=d_l, d_q=d_q, d_model=d_model, seed=42)

        Z = np.random.randn(K_t, d_l)
        q = np.random.randn(d_q)

        F = module.forward(Z, q)
        assert F.shape == (K_t + 1, d_model)

    def test_policy_probabilities_and_deterministic_actions(self):
        K_t = 24
        d_l = 32
        shared_attn = SharedAttentionModule(d_l=d_l, d_q=d_l, d_model=d_l, seed=42)
        policy = TPRLPolicyNetwork(shared_attn=shared_attn, d_model=d_l, lambda_disc=0.5, seed=42)

        Z = np.random.randn(K_t, d_l)
        q = np.random.randn(d_l)

        probs = policy.forward_probabilities(Z, q)
        assert probs.shape == (K_t,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

        # Deterministic actions with reasonable threshold
        actions, p = policy.deterministic_actions(Z, q, threshold=0.5)
        assert actions.shape == (K_t,)
        assert np.all(np.isin(actions, [0, 1]))
        assert np.sum(actions) >= 1  # Guard ensures at least 1 retained

    def test_policy_deterministic_fallback_guard(self):
        # Force very high threshold
        K_t = 10
        d_l = 16
        shared_attn = SharedAttentionModule(d_l=d_l, d_q=d_l, d_model=d_l, seed=42)
        policy = TPRLPolicyNetwork(shared_attn=shared_attn, d_model=d_l, seed=42)

        Z = np.random.randn(K_t, d_l)
        q = np.random.randn(d_l)

        # Threshold 1.5 is impossible to exceed, fallback should trigger
        actions, _ = policy.deterministic_actions(Z, q, threshold=1.5)
        assert np.sum(actions) == 1

    def test_policy_bernoulli_sampling(self):
        K_t = 30
        d_l = 16
        shared_attn = SharedAttentionModule(d_l=d_l, d_q=d_l, d_model=d_l, seed=42)
        policy = TPRLPolicyNetwork(shared_attn=shared_attn, d_model=d_l, lambda_disc=0.5, seed=42)

        Z = np.random.randn(K_t, d_l)
        q = np.random.randn(d_l)

        rng = np.random.RandomState(99)
        actions_step1, _ = policy.sample_actions(Z, q, step=1, rng=rng)
        actions_step3, _ = policy.sample_actions(Z, q, step=3, rng=rng)

        assert actions_step1.shape == (K_t,)
        assert actions_step3.shape == (K_t,)
        assert np.all(np.isin(actions_step1, [0, 1]))


class TestValueNetwork:
    def test_value_output_scalar(self):
        K_t = 15
        d_l = 24
        shared_attn = SharedAttentionModule(d_l=d_l, d_q=d_l, d_model=d_l, seed=42)
        value_net = TPRLValueNetwork(shared_attn=shared_attn, d_model=d_l, seed=42)

        Z = np.random.randn(K_t, d_l)
        q = np.random.randn(d_l)

        v = value_net.forward(Z, q)
        assert isinstance(v, float)


class TestRewardAndAdvantage:
    def test_compute_tprl_reward_batch(self):
        scores_curr = [0.85, 0.90, 0.75]
        scores_prev = [0.80, 0.90, 0.70]
        tokens_curr = [144, 192, 128]
        tokens_prev = [256, 256, 256]

        res = compute_tprl_reward(
            scores_current=scores_curr,
            scores_prev=scores_prev,
            tokens_current=tokens_curr,
            tokens_prev=tokens_prev,
            alpha=1.0,
            beta=0.1,
        )

        expected_delta_score = float(np.mean([0.05, 0.0, 0.05]))
        expected_reduction = float(np.mean([1.0 - 144 / 256, 1.0 - 192 / 256, 1.0 - 128 / 256]))

        assert pytest.approx(res["r_task"], rel=1e-4) == expected_delta_score
        assert pytest.approx(res["r_eff"], rel=1e-4) == 0.1 * expected_reduction
        assert pytest.approx(res["r_total"], rel=1e-4) == expected_delta_score + 0.1 * expected_reduction

    def test_compute_gae_advantage(self):
        rewards = [0.1, 0.2, 0.3]
        values = [0.5, 0.6, 0.7, 0.8]

        adv, ret = compute_gae_advantage(rewards, values, gamma=0.99, lam=0.95)
        assert len(adv) == 3
        assert len(ret) == 3
        # Advantage definition check: ret = adv + values[:T]
        np.testing.assert_allclose(ret, adv + np.array(values[:3]))

    def test_compute_lfd_loss(self):
        probs = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])

        loss = compute_lfd_loss(probs, labels)
        assert loss < 0.3  # Well-aligned predictions should have low BCE loss

        mismatched_loss = compute_lfd_loss(probs, 1 - labels)
        assert mismatched_loss > loss


class TestTPRLPruner:
    def test_pruner_one_shot(self):
        N = 64
        d_v = 128
        d_l = 16
        pruner = TPRLPruner(d_v=d_v, d_hidden=32, d_l=d_l, d_q=d_l, seed=42)

        H = np.random.randn(N, d_v)
        q = np.random.randn(d_l)

        H_retained, mask, stats = pruner.prune_one_shot(H, q, threshold=0.5)

        assert H_retained.shape[0] == stats["retained_tokens"]
        assert H_retained.shape[1] == d_v
        assert mask.shape == (N,)
        assert len(stats["active_indices"]) == stats["retained_tokens"]
        assert 0.0 <= stats["pruning_ratio"] <= 1.0

    def test_simulate_multistep_trajectory(self):
        N = 50
        d_v = 64
        d_l = 16
        pruner = TPRLPruner(d_v=d_v, d_hidden=32, d_l=d_l, d_q=d_l, seed=42)

        H = np.random.randn(N, d_v)
        q = np.random.randn(d_l)

        traj = pruner.simulate_multistep_trajectory(H, q, max_steps=3, step_threshold=0.45)
        assert len(traj) <= 3
        assert len(traj) > 0

        # Monotonically non-increasing tokens
        token_counts = [step["remaining_tokens"] for step in traj]
        for i in range(len(token_counts) - 1):
            assert token_counts[i] >= token_counts[i + 1]
