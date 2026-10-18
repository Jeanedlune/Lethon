"""
Lethon Stage 2: Contextual Semantic Anchor Aggregation (SAC)
Berdasarkan temuan riset paper arXiv:2510.08907 (ICLR 2026).
Menyediakan mekanisme agregasi konteks ke token anchor via modifikasi
atensi dua arah (bidirectional attention) dan learnable anchor embedding.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_anchor_indices(
    seq_len: int,
    compression_ratio: float = 15.0,
    strategy: str = "uniform",
    spectral_energies: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Menghitung indeks token yang bertindak sebagai semantic anchor carriers (S subset C).
    
    Args:
        seq_len: Panjang total sekuens token L.
        compression_ratio: Rasio kompresi target r >= 1.0 (default: 15.0).
        strategy: 'uniform' (chunk-centered) atau 'spectral_guided'.
        spectral_energies: Array 1D energi spektral per token dari Stage 1 (opsional).
        
    Returns:
        np.ndarray: Array 1D integer berisi indeks anchor [M] dengan M = max(1, floor(L / r)).
    """
    if seq_len <= 0:
        raise ValueError("seq_len harus > 0")
    if compression_ratio < 1.0:
        raise ValueError("compression_ratio harus >= 1.0")

    num_anchors = max(1, int(math.floor(seq_len / compression_ratio)))

    if strategy == "spectral_guided" and spectral_energies is not None:
        if len(spectral_energies) != seq_len:
            raise ValueError(
                f"Ukuran spectral_energies ({len(spectral_energies)}) tidak cocok dengan seq_len ({seq_len})"
            )
        chunk_size = seq_len / num_anchors
        indices = []
        for k in range(num_anchors):
            start = int(k * chunk_size)
            end = int((k + 1) * chunk_size) if k < num_anchors - 1 else seq_len
            chunk = spectral_energies[start:end]
            best_local_idx = int(np.argmax(chunk))
            indices.append(start + best_local_idx)
        return np.array(sorted(list(set(indices))), dtype=np.int64)

    # Strategi Default: Uniform Chunk-Centered Selection (EPL & SAC)
    # i_k = floor((k + 0.5) * L / M)
    indices = [
        int(math.floor((k + 0.5) * (seq_len / num_anchors)))
        for k in range(num_anchors)
    ]
    return np.array(indices, dtype=np.int64)


def inject_anchor_embeddings(
    embeddings: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_embedding: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menyuntikkan vektor anchor embedding e_A secara aditif pada posisi token anchor:
    e_i = Emb(c_i) + I(i in S) * e_A
    
    Args:
        embeddings: Tensor representasi input (B, L, d) atau (L, d).
        anchor_indices: Array 1D berisi indeks anchor [M].
        anchor_embedding: Vektor embedding penanda e_A (d,) atau (1, d).
                          Jika None, diinisialisasi N(0, 1 / sqrt(d)).
                          
    Returns:
        augmented_embeddings: Tensor hasil modifikasi dengan bentuk yang sama dengan input.
        indicator_mask: Boolean mask array berukuran (L,) bernilai True pada posisi anchor.
    """
    orig_ndim = embeddings.ndim
    if orig_ndim == 2:
        x = np.expand_dims(embeddings, axis=0)
    else:
        x = embeddings.copy()

    B, L, d = x.shape

    if anchor_embedding is None:
        rng = np.random.default_rng(seed=42)
        e_a = rng.normal(loc=0.0, scale=1.0 / math.sqrt(d), size=(1, 1, d)).astype(x.dtype)
    else:
        e_a = np.asarray(anchor_embedding, dtype=x.dtype)
        while e_a.ndim < 3:
            e_a = np.expand_dims(e_a, axis=0)

    indicator_mask = np.zeros(L, dtype=bool)
    indicator_mask[anchor_indices] = True

    # Injeksi aditif pada posisi anchor
    x[:, indicator_mask, :] = x[:, indicator_mask, :] + e_a

    res = x if orig_ndim == 3 else x[0]
    return res, indicator_mask


def bidirectional_cross_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    scale: Optional[float] = None,
) -> np.ndarray:
    """
    Scaled Dot-Product Attention tanpa causal masking (Full Bidirectional Attention).
    Memungkinkan anchor tokens melihat seluruh token konteks (masa lalu dan masa depan).
    
    Formula:
        Attn(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
        
    Args:
        query: Query tensor (B, L_q, d) atau (B, H, L_q, d_h).
        key: Key tensor (B, L_k, d) atau (B, H, L_k, d_h).
        value: Value tensor (B, L_k, d) atau (B, H, L_k, d_h).
        scale: Faktor pembagi sqrt(d_k). Jika None, dihitung otomatis.
        
    Returns:
        np.ndarray: Output representasi atensi dua arah.
    """
    d_k = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    # Matriks perkalian Q @ K^T
    # Mendukung bentuk 3D (B, L, d) dan 4D (B, H, L, d)
    if query.ndim == 3:
        scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
        # Softmax sepanjang axis terakhir
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return np.matmul(attn_weights, value)
    elif query.ndim == 4:
        scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return np.matmul(attn_weights, value)
    else:
        raise ValueError(f"Dimensi query tidak valid: {query.ndim}. Harus 3D atau 4D.")


def extract_anchor_kv_cache(
    keys: Union[np.ndarray, List[np.ndarray]],
    values: Union[np.ndarray, List[np.ndarray]],
    anchor_indices: np.ndarray,
) -> Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
    """
    Mengekstraksi representasi Key dan Value pada posisi anchor tokens:
    K_tilde = K[:, S, :], V_tilde = V[:, S, :]
    
    Args:
        keys: Tensor Key (B, L, d) / (B, H, L, d_h) atau list tensor per layer.
        values: Tensor Value (B, L, d) / (B, H, L, d_h) atau list tensor per layer.
        anchor_indices: Array 1D indeks anchor [M].
        
    Returns:
        Tuple berisi (compressed_keys, compressed_values).
    """
    if isinstance(keys, list) and isinstance(values, list):
        comp_k = []
        comp_v = []
        for k_l, v_l in zip(keys, values):
            # Sequence axis adalah axis -2
            comp_k.append(np.take(k_l, anchor_indices, axis=-2))
            comp_v.append(np.take(v_l, anchor_indices, axis=-2))
        return comp_k, comp_v

    k_arr = np.asarray(keys)
    v_arr = np.asarray(values)
    return np.take(k_arr, anchor_indices, axis=-2), np.take(v_arr, anchor_indices, axis=-2)


class SemanticAnchorAggregator:
    """
    Kelas Orquestrator Stage 2 Lethon: Contextual Semantic Anchor Aggregation.
    Menjalankan alur lengkap kompresi konteks autoencoding-free.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        default_compression_ratio: float = 15.0,
        strategy: str = "uniform",
        seed: int = 42,
    ):
        self.hidden_size = hidden_size
        self.default_compression_ratio = default_compression_ratio
        self.strategy = strategy
        
        # Inisialisasi Learnable Anchor Embedding e_A ~ N(0, 1 / sqrt(d))
        rng = np.random.default_rng(seed=seed)
        self.anchor_embedding = rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(hidden_size),
            size=(1, 1, hidden_size),
        ).astype(np.float32)

    def compress(
        self,
        context_embeddings: np.ndarray,
        compression_ratio: Optional[float] = None,
        spectral_energies: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Menjalankan pipeline Stage 2 penuh:
        1. Menghitung indeks anchor.
        2. Menyuntikkan anchor embedding e_A.
        3. Menghitung representasi atensi dua arah penuh (bidirectional cross-attention).
        4. Mengekstraksi representasi anchor terkompresi pada indeks anchor.
        
        Args:
            context_embeddings: Tensor embedding konteks (B, L, d) atau (L, d).
            compression_ratio: Rasio kompresi r (opsional, default: default_compression_ratio).
            spectral_energies: Array 1D energi spektral per token dari Stage 1 (opsional).
            
        Returns:
            Dict[str, Any] berisi:
                - 'compressed_embeddings': Tensor representasi anchor terkompresi (B, M, d).
                - 'anchor_indices': Array indeks anchor [M].
                - 'indicator_mask': Boolean mask array [L].
                - 'num_anchors': Integer jumlah anchor M.
                - 'compression_ratio': Rasio kompresi aktual L / M.
        """
        orig_ndim = context_embeddings.ndim
        if orig_ndim == 2:
            x = np.expand_dims(context_embeddings, axis=0)
        else:
            x = context_embeddings.copy()

        B, L, d = x.shape
        ratio = compression_ratio or self.default_compression_ratio

        # 1. Hitung indeks anchor
        anchor_indices = compute_anchor_indices(
            seq_len=L,
            compression_ratio=ratio,
            strategy=self.strategy,
            spectral_energies=spectral_energies,
        )

        # 2. Suntikkan learnable anchor embedding e_A
        augmented, indicator_mask = inject_anchor_embeddings(
            embeddings=x,
            anchor_indices=anchor_indices,
            anchor_embedding=self.anchor_embedding,
        )

        # 3. Jalankan atensi dua arah penuh (bidirectional attention)
        # Anchor menyerap konteks masa lalu dan masa depan secara komprehensif
        contextualized = bidirectional_cross_attention(
            query=augmented,
            key=augmented,
            value=augmented,
        )

        # 4. Ekstraksi representasi terkompresi pada posisi anchor
        compressed_embeddings = np.take(contextualized, anchor_indices, axis=1)

        if orig_ndim == 2:
            compressed_embeddings = compressed_embeddings[0]

        num_anchors = len(anchor_indices)
        return {
            "compressed_embeddings": compressed_embeddings,
            "anchor_indices": anchor_indices,
            "indicator_mask": indicator_mask,
            "num_anchors": num_anchors,
            "compression_ratio": float(L / num_anchors),
        }

