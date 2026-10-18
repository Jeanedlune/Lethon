"""
Lethon Stage 1: Fast Spectral Fourier Filtering
Bebas parameter (Zero-weight), tanpa PyTorch heavyweight, thread-safe.
Implementasi berbasis temuan riset arXiv:2508.06038 (Fourier Compressor).
"""

from typing import Tuple, Optional
import numpy as np
import scipy.fft as sfft


def spectral_filter_1d_downsample(
    x: np.ndarray,
    retention_ratio: float = 0.7,
    workers: int = -1
) -> np.ndarray:
    """
    Skema A: Direct Continuous Spectral Downsampling via 1D-DCT/iDCT.
    
    Args:
        x: Input embedding tensor, shape (B, N, d) atau (N, d).
        retention_ratio: Rasio token yang dipertahankan rho in (0, 1].
                         0.7 berarti memangkas 30% frekuensi tinggi.
        workers: Jumlah thread CPU (-1 = seluruh core).
        
    Returns:
        Tensor terkompresi, shape (B, K, d) dengan K = max(1, floor(retention_ratio * N)).
    """
    ndim = x.ndim
    if ndim == 2:
        x = np.expand_dims(x, axis=0)
        
    B, N, d = x.shape
    K = max(1, int(N * retention_ratio))
    
    if K >= N:
        return x if ndim == 3 else x[0]
        
    # 1. Transformasi ke Domain Frekuensi via 1D-DCT Tipe-II Orthonormal
    # Akselerasi O(N log N) sepanjang axis sekuensial (axis=1)
    f_spectrum = sfft.dct(x, axis=1, type=2, norm="ortho", workers=workers)
    
    # 2. Cutoff Frekuensi Tinggi: Simpan hanya K koefisien frekuensi terendah
    f_low = f_spectrum[:, :K, :]
    
    # 3. Rekonstruksi Spasial Termampatkan via iDCT Orthonormal sepanjang K
    x_compressed = sfft.idct(f_low, axis=1, type=2, norm="ortho", workers=workers)
    
    return x_compressed if ndim == 3 else x_compressed[0]


def spectral_token_saliency_prune(
    x: np.ndarray,
    retention_ratio: float = 0.7,
    workers: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skema B: Discrete Token Selection berdasarkan stabilitas spektral.
    Menghasilkan token diskrit asli untuk pelestarian sintaks kode yang presisi.
    
    Args:
        x: Input tensor (B, N, d) atau (N, d).
        retention_ratio: Rasio token yang dipertahankan rho in (0, 1].
        workers: Jumlah thread CPU.
        
    Returns:
        pruned_x: Tensor dengan panjang K, shape (B, K, d).
        selected_indices: Indeks token terpilih, shape (B, K).
    """
    ndim = x.ndim
    if ndim == 2:
        x = np.expand_dims(x, axis=0)
        
    B, N, d = x.shape
    K = max(1, int(N * retention_ratio))
    
    if K >= N:
        indices = np.arange(N)[None, :].repeat(B, axis=0)
        return (x, indices) if ndim == 3 else (x[0], indices[0])
    
    # 1. Forward DCT
    f_spectrum = sfft.dct(x, axis=1, type=2, norm="ortho", workers=workers)
    
    # 2. Masking frekuensi tinggi (zeroing out k >= K)
    f_masked = np.zeros_like(f_spectrum)
    f_masked[:, :K, :] = f_spectrum[:, :K, :]
    
    # 3. Rekonstruksi sinyal acuan frekuensi rendah
    x_reconstructed = sfft.idct(f_masked, axis=1, type=2, norm="ortho", workers=workers)
    
    # 4. Saliency Score: Proyeksi energi komponen semantik rendah (Cosine fidelity)
    dot_prod = np.sum(x * x_reconstructed, axis=-1)
    norm_orig = np.linalg.norm(x, axis=-1) + 1e-8
    norm_rec = np.linalg.norm(x_reconstructed, axis=-1) + 1e-8
    saliency_score = dot_prod / (norm_orig * norm_rec)  # Shape (B, N)
    
    # 5. Top-K Selection terurut sesuai waktu/posisi asli
    selected_indices = np.argsort(-saliency_score, axis=1)[:, :K]
    selected_indices = np.sort(selected_indices, axis=1)  # Jaga urutan sekuensial
    
    batch_indices = np.arange(B)[:, None]
    pruned_x = x[batch_indices, selected_indices]
    
    return (pruned_x, selected_indices) if ndim == 3 else (pruned_x[0], selected_indices[0])


def fourier_compressor_2d(
    x: np.ndarray,
    target_grid_c: int,
    workers: int = -1
) -> np.ndarray:
    """
    Skema C: Algoritma Asli Fourier Compressor (arXiv:2508.06038) untuk 2D Vision Grid.
    
    Args:
        x: Visual patch tokens, shape (B, N^2, d).
        target_grid_c: Dimensi sisi grid terkompresi C (hasil akhir C^2 tokens).
        workers: Jumlah worker CPU.
        
    Returns:
        Tensor terkompresi shape (B, C^2, d).
    """
    B, total_tokens, d = x.shape
    N = int(np.sqrt(total_tokens))
    if N * N != total_tokens:
        raise ValueError(f"Total tokens ({total_tokens}) bukan merupakan kuadrat sempurna.")
    
    # Reshape ke grid 2D: (B, d, N, N)
    grid = np.transpose(x.reshape(B, N, N, d), (0, 3, 1, 2))
    
    # 2D-DCT: DCT pada axis baris lalu kolom
    f1 = sfft.dct(grid, axis=2, type=2, norm="ortho", workers=workers)
    f2 = sfft.dct(f1, axis=3, type=2, norm="ortho", workers=workers)
    
    # Truncate frekuensi tinggi: simpan sudut kiri-atas C x C
    f_crop = f2[:, :, :target_grid_c, :target_grid_c]
    
    # 2D-iDCT pada grid C x C
    r1 = sfft.idct(f_crop, axis=2, type=2, norm="ortho", workers=workers)
    r2 = sfft.idct(r1, axis=3, type=2, norm="ortho", workers=workers)
    
    # Reshape kembali ke (B, C^2, d)
    out = np.transpose(r2, (0, 2, 3, 1)).reshape(B, target_grid_c * target_grid_c, d)
    return out
