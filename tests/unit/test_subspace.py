import unittest
import numpy as np

from lethon.core.subspace import (
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


class TestSubspaceMasking(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_topk_masking_1d(self):
        # 10 tokens with arbitrary logits
        logits = np.array([0.1, 2.5, 0.3, 5.0, 1.2, 0.05, 4.1, 3.8, 0.2, 0.9])
        k = 4
        mask = compute_promising_mask_topk(logits, k=k)
        
        self.assertEqual(mask.shape, (10,))
        self.assertEqual(np.sum(mask), k)
        
        # Top 4 indices should be 3 (5.0), 6 (4.1), 7 (3.8), 1 (2.5)
        expected_top = {1, 3, 6, 7}
        active_indices = set(np.where(mask > 0.5)[0])
        self.assertEqual(active_indices, expected_top)

    def test_topk_masking_2d_batch(self):
        batch_size, vocab_size = 3, 20
        logits = np.random.randn(batch_size, vocab_size)
        k = 5
        mask = compute_promising_mask_topk(logits, k=k)

        self.assertEqual(mask.shape, (batch_size, vocab_size))
        # Setiap baris harus memiliki tepat K token aktif
        np.testing.assert_array_equal(np.sum(mask, axis=-1), np.full(batch_size, k))

    def test_topp_masking(self):
        # Probabilitas eksponensial menurun tajam
        probs = np.array([0.50, 0.25, 0.15, 0.06, 0.03, 0.01])
        # P = 0.80 -> token 0 (0.50) + token 1 (0.25) = 0.75 < 0.80, token 2 (0.15) -> total 0.90 >= 0.80
        mask = compute_promising_mask_topp(probs, p=0.80)
        self.assertEqual(mask.shape, (6,))
        active_indices = list(np.where(mask > 0.5)[0])
        self.assertEqual(active_indices, [0, 1, 2])

    def test_minp_masking(self):
        # max_prob = 0.60. min_p = 0.10 -> threshold = 0.06
        probs = np.array([0.60, 0.20, 0.08, 0.05, 0.04, 0.03])
        mask = compute_promising_mask_minp(probs, min_p=0.10)
        
        # Token 0 (0.60), 1 (0.20), 2 (0.08) >= 0.06 -> dipertahankan
        # Token 3 (0.05), 4 (0.04), 5 (0.03) < 0.06 -> dipangkas
        active_indices = list(np.where(mask > 0.5)[0])
        self.assertEqual(active_indices, [0, 1, 2])

    def test_subspace_projection_svd(self):
        N, d = 30, 16
        # Generate correlated hidden states
        latent = np.random.randn(N, 2)
        proj = np.random.randn(2, d)
        hidden_states = latent @ proj + 0.01 * np.random.randn(N, d)

        mask, scores, info = compute_subspace_projection_mask(
            hidden_states, energy_threshold=0.85
        )
        self.assertEqual(mask.shape, (N,))
        self.assertEqual(scores.shape, (N,))
        self.assertGreater(info["energy_ratio"], 0.80)
        self.assertGreater(np.sum(mask), 0)

    def test_apply_logit_mask_and_masked_softmax(self):
        logits = np.array([2.0, 1.0, 0.0, -1.0])
        mask = np.array([1.0, 1.0, 0.0, 0.0])  # hanya token 0 dan 1

        masked_logits = apply_logit_mask(logits, mask, mask_value=-1e9)
        self.assertAlmostEqual(masked_logits[0], 2.0)
        self.assertAlmostEqual(masked_logits[1], 1.0)
        self.assertLess(masked_logits[2], -1e8)
        self.assertLess(masked_logits[3], -1e8)

        # Probabilitas pada token 2 dan 3 harus bernilai 0.0
        probs = masked_softmax(logits, mask)
        self.assertAlmostEqual(probs[2], 0.0, places=6)
        self.assertAlmostEqual(probs[3], 0.0, places=6)
        # Probabilitas pada token 0 dan 1 harus menjumlah ke 1.0
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)
        # Ratio exp(2) / exp(1) = e approx 2.71828
        self.assertAlmostEqual(probs[0] / probs[1], np.e, places=4)

    def test_gradient_variance_reduction_prop_4_1(self):
        """
        Validasi eksak Proposition 4.1 paper RLPT:
        Optimasi di atas subruang Pt secara ketat mereduksi varians gradien (Delta V > 0)
        dengan mengeliminasi varians stokastik ribuan token ekor (tail tokens).
        """
        V = 1000
        # Simulasikan distribusi probabilitas LLM dengan konsentrasi pada top tokens
        raw_logits = np.random.randn(V) * 0.5
        raw_logits[0:8] += 7.0  # Head/promising tokens (> 97% prob mass)
        exp_z = np.exp(raw_logits - np.max(raw_logits))
        probs = exp_z / np.sum(exp_z)

        # K = 8 promising tokens
        mask = compute_promising_mask_topk(probs, k=8, is_probs=True)
        
        var_full, var_mask, delta_v = compute_policy_gradient_variance(
            probs, mask, advantage=2.5
        )

        # 1. Delta V harus secara ketat bernilai positif
        self.assertGreater(delta_v, 0.0)
        # 2. Total varians pada subruang harus lebih rendah dibanding ruang penuh
        self.assertLess(var_mask, var_full)

        # 3. Varians komponen tail pada ruang penuh bernilai positif
        tail_mask = 1.0 - mask
        tail_variance_full = float(np.sum(probs * (1.0 - probs) * (2.5 ** 2) * tail_mask))
        self.assertGreater(tail_variance_full, 0.0)
        # Pada RLPT, gradien komponen tail deterministik nol (var = 0)
        # Reduksi varians memiliki magnitudo positif signifikan
        self.assertGreater(delta_v / tail_variance_full, 0.2)

    def test_rlpt_grpo_loss(self):
        batch_size, vocab_size = 4, 50
        logits_theta = np.random.randn(batch_size, vocab_size)
        logits_old = np.random.randn(batch_size, vocab_size)

        mask = compute_promising_mask_topk(logits_old, k=4)
        
        # Pilih aksi valid dari promising tokens
        actions = np.array([np.where(mask[i] > 0.5)[0][0] for i in range(batch_size)])
        advantages = np.array([1.2, -0.8, 0.5, 2.0])

        loss, metrics = compute_rlpt_grpo_loss(
            logits_theta, logits_old, actions, mask, advantages, clip_eps=0.2
        )
        self.assertIsInstance(loss, float)
        self.assertIn("mean_ratio", metrics)
        self.assertIn("clip_fraction", metrics)
        self.assertGreaterEqual(metrics["clip_fraction"], 0.0)
        self.assertLessEqual(metrics["clip_fraction"], 1.0)

    def test_topp_masking_2d_batch(self):
        batch_size, vocab_size = 2, 5
        probs = np.array([
            [0.6, 0.2, 0.1, 0.06, 0.04],
            [0.3, 0.3, 0.2, 0.1, 0.1],
        ])
        mask = compute_promising_mask_topp(probs, p=0.75)
        self.assertEqual(mask.shape, (batch_size, vocab_size))
        # Row 0: 0.6 + 0.2 = 0.8 >= 0.75 -> 2 tokens
        self.assertEqual(np.sum(mask[0]), 2)
        # Row 1: 0.3 + 0.3 = 0.6 < 0.75, + 0.2 = 0.8 >= 0.75 -> 3 tokens
        self.assertEqual(np.sum(mask[1]), 3)

    def test_minp_masking_2d_batch(self):
        batch_size, vocab_size = 2, 4
        probs = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.5, 0.25, 0.15, 0.10],
        ])
        mask = compute_promising_mask_minp(probs, min_p=0.20)
        self.assertEqual(mask.shape, (batch_size, vocab_size))
        # Row 0: threshold = 0.2 * 0.8 = 0.16. Hanya token 0 (0.8) >= 0.16
        self.assertEqual(np.sum(mask[0]), 1)
        # Row 1: threshold = 0.2 * 0.5 = 0.10. Token 0, 1, 2, 3 >= 0.10 -> 4 tokens
        self.assertEqual(np.sum(mask[1]), 4)

    def test_error_handling_and_edge_cases(self):
        # Invalid P
        with self.assertRaises(ValueError):
            compute_promising_mask_topp(np.array([0.5, 0.5]), p=0.0)
        with self.assertRaises(ValueError):
            compute_promising_mask_topp(np.array([0.5, 0.5]), p=1.5)
        
        # Invalid min_p
        with self.assertRaises(ValueError):
            compute_promising_mask_minp(np.array([0.5, 0.5]), min_p=0.0)

        # Invalid temperature
        with self.assertRaises(ValueError):
            masked_softmax(np.array([1.0, 2.0]), np.array([1.0, 1.0]), temperature=0.0)

        # Mismatch shape in apply_logit_mask
        with self.assertRaises(ValueError):
            apply_logit_mask(np.array([1.0, 2.0]), np.array([1.0]))

        # Subspace SVD requires 2D
        with self.assertRaises(ValueError):
            compute_subspace_projection_mask(np.array([1.0, 2.0]))

    def test_promising_token_masker_class(self):
        masker = PromisingTokenMasker(strategy="topk", k=4)
        logits = np.random.randn(2, 30)
        probs, mask = masker.get_subspace_distribution(logits)

        self.assertEqual(probs.shape, (2, 30))
        self.assertEqual(mask.shape, (2, 30))
        np.testing.assert_array_equal(np.sum(mask, axis=-1), np.array([4, 4]))
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.array([1.0, 1.0]), atol=1e-5)

        # Test filtering
        masked_z, m = masker.filter_logits(logits)
        self.assertEqual(masked_z.shape, logits.shape)

        # Test topp strategy
        masker_topp = PromisingTokenMasker(strategy="topp", p=0.9)
        m_topp = masker_topp.create_mask(logits, is_probs=False)
        self.assertEqual(m_topp.shape, logits.shape)

        # Test minp strategy
        masker_minp = PromisingTokenMasker(strategy="minp", min_p=0.1)
        m_minp = masker_minp.create_mask(logits, is_probs=False)
        self.assertEqual(m_minp.shape, logits.shape)

        # Test subspace_svd strategy
        H = np.random.randn(20, 10)
        masker_svd = PromisingTokenMasker(strategy="subspace_svd", energy_threshold=0.8)
        m_svd = masker_svd.create_mask(logits=None, hidden_states=H)
        self.assertEqual(m_svd.shape, (20,))


if __name__ == "__main__":
    unittest.main()
