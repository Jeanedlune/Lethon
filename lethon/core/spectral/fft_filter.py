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


def calculate_layer_retention_ratio(
    layer_idx: int,
    total_layers: int,
    base_ratio: float = 0.25,
    gamma: float = 2.0
) -> float:
    """
    Menghitung rasio retensi adaptif per layer rho(l) sesuai temuan FourierCompress (arXiv:2510.16418).
    Layer awal (l=1) dipangkas agresif; layer dalam (l -> L) dipangkas konservatif.

    Args:
        layer_idx: Indeks layer aktif (1-indexed, misal 1 sampai total_layers).
        total_layers: Jumlah total layer transformer (misal 16 atau 32).
        base_ratio: Rasio retensi dasar pada Layer 1 (0.25 = ~4x compression).
        gamma: Eksponen kurvatur layer scheduling (default 2.0).

    Returns:
        Rasio retensi token/fitur yang diizinkan rho in [base_ratio, 1.0].
    """
    if total_layers <= 1:
        return 1.0
    normalized_depth = (layer_idx - 1) / (total_layers - 1)
    ratio = base_ratio + (1.0 - base_ratio) * (normalized_depth ** gamma)
    return float(np.clip(ratio, base_ratio, 1.0))


def fourier_compress_activation_2d(
    activation: np.ndarray,
    seq_ratio: float = 0.36,
    hidden_ratio: float = 0.36,
    workers: int = -1
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Skema D (Encoder/Client): FourierCompress 2D Spectral Activation Compression (arXiv:2510.16418).
    Mentransformasikan aktivasi (S, D) ke domain frekuensi 2D dan memotong blok frekuensi rendah.

    Args:
        activation: Matriks aktivasi Layer 1, shape (S, D).
        seq_ratio: Rasio frekuensi sequence yang dipertahankan rho_S (0.36 * 0.36 ~ 7.6x compression).
        hidden_ratio: Rasio frekuensi hidden size yang dipertahankan rho_D.
        workers: Thread CPU (-1 = all cores).

    Returns:
        core_spectrum: Blok koefisien frekuensi rendah (K_S, K_D) bernilai kompleks.
        orig_shape: Ukuran matriks asli (S, D) untuk zero-padding di receiver.
    """
    S, D = activation.shape
    K_S = max(1, int(S * seq_ratio))
    K_D = max(1, int(D * hidden_ratio))

    # 1. 2D Fast Fourier Transform
    spectrum_2d = sfft.fft2(activation, workers=workers)

    # 2. Retensi blok rectangular sudut kiri-atas (frekuensi rendah)
    core_spectrum = spectrum_2d[:K_S, :K_D]

    return core_spectrum, (S, D)


def fourier_decompress_activation_2d(
    core_spectrum: np.ndarray,
    orig_shape: Tuple[int, int],
    workers: int = -1
) -> np.ndarray:
    """
    Skema D (Decoder/Server): Rekonstruksi aktivasi via Zero-Padding & 2D-IFFT.
    Memanfaatkan simetri konjugat Hermitian A[u, v] = A*[S - u, D - v]
    untuk sinyal riil murni.

    Args:
        core_spectrum: Blok frekuensi rendah terkompresi, shape (K_S, K_D).
        orig_shape: Bentuk matriks asal (S, D).
        workers: Thread CPU.

    Returns:
        reconstructed_activation: Matriks aktivasi riil berukuran penuh (S, D).
    """
    S, D = orig_shape
    K_S, K_D = core_spectrum.shape

    # 1. Alokasi zero-padded spectrum ukuran penuh (S, D)
    padded_spectrum = np.zeros((S, D), dtype=np.complex128)
    padded_spectrum[:K_S, :K_D] = core_spectrum

    # 2. Pulihkan simetri konjugat Hermitian untuk koefisien riil murni:
    # A[u, v] = A*[S - u, D - v]
    for u in range(K_S):
        for v in range(K_D):
            if u == 0 and v == 0:
                continue
            padded_spectrum[(S - u) % S, (D - v) % D] = np.conj(core_spectrum[u, v])

    # 3. 2D Inverse Fast Fourier Transform & ekstraksi komponen riil
    reconstructed = sfft.ifft2(padded_spectrum, workers=workers)
    return np.real(reconstructed).astype(np.float32)


