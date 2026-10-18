"""
Lethon Spectral Filtering Core Module (Stage 1)
"""

from lethon.core.spectral.fft_filter import (
    spectral_filter_1d_downsample,
    spectral_token_saliency_prune,
    fourier_compressor_2d,
)

__all__ = [
    "spectral_filter_1d_downsample",
    "spectral_token_saliency_prune",
    "fourier_compressor_2d",
]
