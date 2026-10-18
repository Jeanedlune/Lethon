import unittest
import numpy as np
from lethon.core.anchor import (
    compute_anchor_indices,
    inject_anchor_embeddings,
    bidirectional_cross_attention,
    extract_anchor_kv_cache,
    SemanticAnchorAggregator,
)


class TestSemanticAnchorAggregation(unittest.TestCase):
    def test_compute_anchor_indices_uniform(self):
        L = 510
        ratio = 15.0
        indices = compute_anchor_indices(seq_len=L, compression_ratio=ratio, strategy="uniform")
        
        expected_m = int(L // ratio)  # 34
        self.assertEqual(len(indices), expected_m)
        # Indeks harus unik dan strictly increasing
        self.assertTrue(np.all(np.diff(indices) > 0))
        # Indeks harus dalam batas rentang [0, L - 1]
        self.assertGreaterEqual(indices[0], 0)
        self.assertLess(indices[-1], L)

        # Uji formula chunk-centered: floor((k + 0.5) * L / M)
        for k in range(expected_m):
            expected_ik = int(np.floor((k + 0.5) * (L / expected_m)))
            self.assertEqual(indices[k], expected_ik)

    def test_compute_anchor_indices_edge_cases(self):
        # seq_len lebih kecil dari rasio kompresi -> tetap minimal 1 anchor
        indices = compute_anchor_indices(seq_len=5, compression_ratio=15.0)
        self.assertEqual(len(indices), 1)
        self.assertGreaterEqual(indices[0], 0)
        self.assertLess(indices[0], 5)

        # rasio kompresi 1.0 -> semua token menjadi anchor
        indices_full = compute_anchor_indices(seq_len=20, compression_ratio=1.0)
        self.assertEqual(len(indices_full), 20)
        np.testing.assert_array_equal(indices_full, np.arange(20))

        # Error cases
        with self.assertRaises(ValueError):
            compute_anchor_indices(seq_len=0, compression_ratio=5.0)
        with self.assertRaises(ValueError):
            compute_anchor_indices(seq_len=100, compression_ratio=0.5)

    def test_compute_anchor_indices_spectral_guided(self):
        L = 40
        ratio = 10.0  # 4 chunks
        # Buat spike energi pada indeks spesifik di setiap kuadran
        spectral_energies = np.zeros(L, dtype=np.float32)
        spectral_energies[3] = 10.0   # chunk 0 [0..10)
        spectral_energies[17] = 20.0  # chunk 1 [10..20)
        spectral_energies[25] = 15.0  # chunk 2 [20..30)
        spectral_energies[38] = 30.0  # chunk 3 [30..40)

        indices = compute_anchor_indices(
            seq_len=L,
            compression_ratio=ratio,
            strategy="spectral_guided",
            spectral_energies=spectral_energies,
        )
        self.assertEqual(len(indices), 4)
        np.testing.assert_array_equal(indices, [3, 17, 25, 38])

    def test_inject_anchor_embeddings(self):
        B, L, d = 2, 50, 64
        embeddings = np.random.randn(B, L, d).astype(np.float32)
        anchor_indices = np.array([5, 15, 25, 35, 45], dtype=np.int64)
        e_a = np.ones((1, 1, d), dtype=np.float32) * 2.5

        augmented, mask = inject_anchor_embeddings(
            embeddings=embeddings,
            anchor_indices=anchor_indices,
            anchor_embedding=e_a,
        )

        self.assertEqual(augmented.shape, (B, L, d))
        self.assertEqual(mask.shape, (L,))
        self.assertEqual(np.sum(mask), len(anchor_indices))

        # Token non-anchor harus persis sama (tidak termodifikasi)
        non_anchor_idx = [i for i in range(L) if i not in anchor_indices]
        np.testing.assert_allclose(
            augmented[:, non_anchor_idx, :],
            embeddings[:, non_anchor_idx, :],
            rtol=1e-6,
        )

        # Token anchor harus bertambah persis e_a
        for idx in anchor_indices:
            np.testing.assert_allclose(
                augmented[:, idx, :],
                embeddings[:, idx, :] + 2.5,
                rtol=1e-5,
            )

    def test_bidirectional_cross_attention_non_causal(self):
        B, L, d = 1, 10, 16
        # Pastikan bahwa token posisi 0 dapat melihat token posisi 9 (Non-Causal)
        q = np.random.randn(B, L, d).astype(np.float32)
        k = np.random.randn(B, L, d).astype(np.float32)
        v = np.random.randn(B, L, d).astype(np.float32)

        out1 = bidirectional_cross_attention(q, k, v)
        self.assertEqual(out1.shape, (B, L, d))

        # Modifikasi nilai token masa depan (indeks 9)
        v_modified = v.copy()
        v_modified[:, 9, :] += 50.0

        out2 = bidirectional_cross_attention(q, k, v_modified)

        # Pada causal attention, posisi 0 tidak boleh berubah.
        # Pada BIDIRECTIONAL attention, posisi 0 WAJIB berubah karena dapat melihat token 9!
        diff_at_pos_0 = np.max(np.abs(out1[:, 0, :] - out2[:, 0, :]))
        self.assertGreater(diff_at_pos_0, 1e-4)

    def test_extract_anchor_kv_cache(self):
        B, num_layers, L, d = 2, 4, 60, 32
        anchor_indices = np.array([10, 30, 50], dtype=np.int64)

        keys_list = [np.random.randn(B, L, d).astype(np.float32) for _ in range(num_layers)]
        values_list = [np.random.randn(B, L, d).astype(np.float32) for _ in range(num_layers)]

        comp_k, comp_v = extract_anchor_kv_cache(keys_list, values_list, anchor_indices)
        self.assertEqual(len(comp_k), num_layers)
        self.assertEqual(len(comp_v), num_layers)
        for l in range(num_layers):
            self.assertEqual(comp_k[l].shape, (B, 3, d))
            self.assertEqual(comp_v[l].shape, (B, 3, d))
            # Nilai harus persis slice dari layer asli
            np.testing.assert_array_equal(comp_k[l], keys_list[l][:, anchor_indices, :])

    def test_semantic_anchor_aggregator_compress_pipeline(self):
        hidden_size = 64
        aggregator = SemanticAnchorAggregator(
            hidden_size=hidden_size,
            default_compression_ratio=15.0,
            strategy="uniform",
        )

        B, L = 2, 150
        context_emb = np.random.randn(B, L, hidden_size).astype(np.float32)

        # Uji pada 3 rasio kompresi utama: 5x, 15x, 51x
        for r in [5.0, 15.0, 51.0]:
            result = aggregator.compress(context_emb, compression_ratio=r)
            expected_m = max(1, int(L // r))

            self.assertIn("compressed_embeddings", result)
            self.assertIn("anchor_indices", result)
            self.assertIn("num_anchors", result)
            self.assertEqual(result["num_anchors"], expected_m)
            self.assertEqual(result["compressed_embeddings"].shape, (B, expected_m, hidden_size))
            self.assertAlmostEqual(result["compression_ratio"], L / expected_m, places=2)


if __name__ == "__main__":
    unittest.main()
