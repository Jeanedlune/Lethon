import unittest
import numpy as np
from lethon.core.spectral import (
    spectral_filter_1d_downsample,
    spectral_token_saliency_prune,
    fourier_compressor_2d,
    calculate_layer_retention_ratio,
    fourier_compress_activation_2d,
    fourier_decompress_activation_2d,
)


class TestSpectralFiltering(unittest.TestCase):
    def test_spectral_filter_1d_downsample(self):
        B, N, d = 2, 100, 64
        x = np.random.randn(B, N, d).astype(np.float32)
        ratio = 0.6
        out = spectral_filter_1d_downsample(x, retention_ratio=ratio)
        self.assertEqual(out.shape, (B, 60, d))

    def test_spectral_token_saliency_prune(self):
        B, N, d = 2, 80, 32
        x = np.random.randn(B, N, d).astype(np.float32)
        ratio = 0.5
        pruned, indices = spectral_token_saliency_prune(x, retention_ratio=ratio)
        self.assertEqual(pruned.shape, (B, 40, d))
        self.assertEqual(indices.shape, (B, 40))
        # Check monotonic order of preserved indices
        for b in range(B):
            self.assertTrue(np.all(np.diff(indices[b]) > 0))

    def test_fourier_compressor_2d(self):
        B, d = 2, 32
        N = 8  # 64 tokens
        C = 4  # 16 tokens
        x = np.random.randn(B, N * N, d).astype(np.float32)
        out = fourier_compressor_2d(x, target_grid_c=C)
        self.assertEqual(out.shape, (B, C * C, d))

    def test_calculate_layer_retention_ratio(self):
        # Layer 1 should equal base_ratio (0.25)
        r_l1 = calculate_layer_retention_ratio(layer_idx=1, total_layers=16, base_ratio=0.25)
        self.assertAlmostEqual(r_l1, 0.25)
        # Final layer should equal 1.0 (conservative)
        r_last = calculate_layer_retention_ratio(layer_idx=16, total_layers=16, base_ratio=0.25)
        self.assertAlmostEqual(r_last, 1.0)
        # Mid layer should be monotonically increasing
        r_mid = calculate_layer_retention_ratio(layer_idx=8, total_layers=16, base_ratio=0.25)
        self.assertTrue(0.25 < r_mid < 1.0)

    def test_fourier_compress_activation_2d_roundtrip(self):
        S, D = 128, 64
        # Create a smooth low-frequency activation matrix
        t_s = np.linspace(0, 2 * np.pi, S)[:, None]
        t_d = np.linspace(0, 2 * np.pi, D)[None, :]
        activation = (np.sin(t_s) + np.cos(t_d)).astype(np.float32)

        core, orig_shape = fourier_compress_activation_2d(activation, seq_ratio=0.5, hidden_ratio=0.5)
        self.assertEqual(core.shape, (64, 32))
        self.assertEqual(orig_shape, (S, D))

        reconstructed = fourier_decompress_activation_2d(core, orig_shape)
        self.assertEqual(reconstructed.shape, (S, D))

        # Mean squared error on low-frequency signal should be small
        mse = np.mean((activation - reconstructed) ** 2)
        self.assertLess(mse, 0.05)


if __name__ == "__main__":
    unittest.main()

