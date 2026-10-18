import unittest
import numpy as np
from lethon.core.spectral import (
    spectral_filter_1d_downsample,
    spectral_token_saliency_prune,
    fourier_compressor_2d,
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


if __name__ == "__main__":
    unittest.main()
