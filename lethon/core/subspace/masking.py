"""
Lethon Stage 3: Dynamic Promising Token Subspace Masking (RLPT)
Berdasarkan paper:
- "Reinforcement Learning with Promising Tokens for Large Language Models" (arXiv:2602.03195v2, Feb 2026)
- "Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning" (arXiv:2512.04359)

Menyediakan mekanisme isolasi ruang aksi ke dalam low-rank promising token subspace,
eliminasi varians gradien ekor token tak relevan, logit masking deterministik,
dan formulasi loss RL terikat subruang (RLPT-GRPO).
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_promising_mask_topk(
    logits_or_probs: np.ndarray,
    k: int = 4,
    is_probs: bool = False,
) -> np.ndarray:
    """
    Membangun Promising Mask biner Mt in {0, 1}^|V| berdasarkan strategi Top-K ranking.
    Token dengan rank(v | st) <= K bernilai 1, sisanya bernilai 0.

    Args:
        logits_or_probs: Array 1D atau 2D (batch_size, vocab_size) dari logit atau probabilitas.
        k: Ukuran himpunan promising tokens (|Pt| = K). Default: 4.
        is_probs: Boolean apakah input berupa probabilitas atau logit (keduanya monotonik).

    Returns:
        np.ndarray: Mask biner boolean atau int (bentuk sama dengan input) dengan nilai 1 pada Top-K.
    """
    arr = np.asarray(logits_or_probs)
    if arr.ndim == 1:
        v_size = arr.shape[0]
        k_clamped = max(1, min(k, v_size))
        # Menggunakan partition untuk efisiensi O(V)
        topk_indices = np.argpartition(arr, -k_clamped)[-k_clamped:]
        mask = np.zeros(v_size, dtype=np.float32)
        mask[topk_indices] = 1.0
        return mask
    elif arr.ndim == 2:
        b_size, v_size = arr.shape
        k_clamped = max(1, min(k, v_size))
        topk_indices = np.argpartition(arr, -k_clamped, axis=-1)[:, -k_clamped:]
        mask = np.zeros((b_size, v_size), dtype=np.float32)
        np.put_along_axis(mask, topk_indices, 1.0, axis=-1)
        return mask
    else:
        raise ValueError(f"Input harus 1D atau 2D, diperoleh bentuk {arr.shape}")


def compute_promising_mask_topp(
    probs: np.ndarray,
    p: float = 0.95,
) -> np.ndarray:
    """
    Membangun Promising Mask biner berdasarkan Nucleus / Top-P sampling.
    Mempertahankan himpunan token terkecil yang probabilitas kumulatifnya >= p.

    Args:
        probs: Array 1D atau 2D probabilitas (harus ternormalisasi sum = 1 pada axis terakhir).
        p: Ambang batas probabilitas kumulatif in (0, 1].

    Returns:
        np.ndarray: Mask biner berukuran sama dengan probs.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if not (0.0 < p <= 1.0):
        raise ValueError("p harus berada dalam rentang (0.0, 1.0]")

    if arr.ndim == 1:
        sorted_indices = np.argsort(arr)[::-1]
        sorted_probs = arr[sorted_indices]
        cum_probs = np.cumsum(sorted_probs)

        # Selalu sertakan minimal token pertama
        mask_sorted = cum_probs - sorted_probs < p
        mask_sorted[0] = True

        mask = np.zeros_like(arr, dtype=np.float32)
        mask[sorted_indices[mask_sorted]] = 1.0
        return mask
    elif arr.ndim == 2:
        sorted_indices = np.argsort(arr, axis=-1)[:, ::-1]
        sorted_probs = np.take_along_axis(arr, sorted_indices, axis=-1)
        cum_probs = np.cumsum(sorted_probs, axis=-1)

        mask_sorted = (cum_probs - sorted_probs) < p
        mask_sorted[:, 0] = True

        mask = np.zeros_like(arr, dtype=np.float32)
        np.put_along_axis(mask, sorted_indices, mask_sorted.astype(np.float32), axis=-1)
        return mask
    else:
        raise ValueError(f"Input harus 1D atau 2D, diperoleh bentuk {arr.shape}")


def compute_promising_mask_minp(
    probs: np.ndarray,
    min_p: float = 0.05,
) -> np.ndarray:
    """
    Membangun Promising Mask biner berdasarkan Min-P sampling (Nguyen et al., 2024).
    Token dipertahankan jika: prob(v) >= min_p * max_prob.

    Args:
        probs: Array 1D atau 2D probabilitas.
        min_p: Faktor ambang batas relatif in (0, 1].

    Returns:
        np.ndarray: Mask biner berukuran sama dengan probs.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if not (0.0 < min_p <= 1.0):
        raise ValueError("min_p harus berada dalam rentang (0.0, 1.0]")

    if arr.ndim == 1:
        max_prob = np.max(arr)
        threshold = min_p * max_prob
        mask = (arr >= threshold).astype(np.float32)
        # Proteksi agar tidak kosong
        if not np.any(mask):
            mask[np.argmax(arr)] = 1.0
        return mask
    elif arr.ndim == 2:
        max_prob = np.max(arr, axis=-1, keepdims=True)
        threshold = min_p * max_prob
        mask = (arr >= threshold).astype(np.float32)
        # Proteksi argmax untuk setiap baris
        empty_rows = np.sum(mask, axis=-1) == 0
        if np.any(empty_rows):
            arg_maxes = np.argmax(arr[empty_rows], axis=-1)
            for idx, r_idx in enumerate(np.where(empty_rows)[0]):
                mask[r_idx, arg_maxes[idx]] = 1.0
        return mask
    else:
        raise ValueError(f"Input harus 1D atau 2D, diperoleh bentuk {arr.shape}")


def compute_subspace_projection_mask(
    hidden_states: np.ndarray,
    energy_threshold: float = 0.90,
    k_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Menghitung mask token berdasarkan proyeksi low-rank subspace via Truncated SVD.
    Untuk matriks representasi konteks H in R^{N x d}, menghitung basis subruang Vk
    dan skor signifikansi s_i = || h_i P_k ||_2^2.

    Args:
        hidden_states: Array 2D (seq_len, hidden_dim).
        energy_threshold: Rasio energi kumulatif nilai singular yang dipertahankan (0, 1].
        k_components: Jumlah komponen low-rank eksplisit (opsional; jika None ditentukan via energy_threshold).

    Returns:
        Tuple of:
            - mask: Array 1D boolean bernilai 1 untuk token dengan proyeksi di atas ambang batas.
            - scores: Array 1D skor proyeksi s_i per token.
            - info: Metadata dekomposisi (k_used, energy_ratio, total_variance).
    """
    H = np.asarray(hidden_states, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"hidden_states harus 2D (seq_len, hidden_dim), diperoleh {H.shape}")

    N, d = H.shape
    # Center matriks
    H_centered = H - np.mean(H, axis=0, keepdims=True)

    # Full/Truncated SVD pada representasi fitur
    # H_centered = U * S * Vt
    try:
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback jika terjadi singularitas
        return np.ones(N, dtype=np.float32), np.ones(N, dtype=np.float32), {"k_used": 1, "energy_ratio": 1.0}

    total_energy = np.sum(S**2)
    if total_energy > 0:
        cum_energy = np.cumsum(S**2) / total_energy
    else:
        cum_energy = np.ones_like(S)

    if k_components is not None:
        k = max(1, min(k_components, len(S)))
    else:
        k = int(np.searchsorted(cum_energy, energy_threshold)) + 1
        k = max(1, min(k, len(S)))

    # Basis proyeksi low-rank: Vk in R^{d x k}
    Vk = Vt[:k, :].T  # (d, k)
    
    # Proyeksi setiap token: h_i Vk => (N, k)
    H_proj = H_centered @ Vk
    # Skor signifikansi kuadrat magnitudo proyeksi
    scores = np.sum(H_proj**2, axis=-1)  # (N,)

    # Ambang batas dinamis: mean atau kuantil
    threshold = np.median(scores)
    mask = (scores >= threshold).astype(np.float32)

    # Proteksi minimal 1 token
    if not np.any(mask):
        mask[np.argmax(scores)] = 1.0

    actual_energy = float(cum_energy[k - 1]) if len(cum_energy) > 0 else 1.0
    info = {
        "k_used": float(k),
        "energy_ratio": actual_energy,
        "total_energy": float(total_energy),
    }
    return mask, scores.astype(np.float32), info


def apply_logit_mask(
    logits: np.ndarray,
    mask: np.ndarray,
    mask_value: float = -1e9,
) -> np.ndarray:
    """
    Menerapkan dynamic mask Mt ke logit model:
    z_masked = z + (1 - Mt) * mask_value

    Di mana mask_value default -1e9 (mendekati -infinity secara numerik aman).

    Args:
        logits: Array logit (1D atau 2D).
        mask: Mask biner bernilai 1 (promising) atau 0 (tail).
        mask_value: Nilai penalti untuk token yang di-mask (default: -1e9).

    Returns:
        np.ndarray: Logit ter-mask di mana token non-promising ditekan ke mask_value.
    """
    logits_arr = np.asarray(logits, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=np.float64)

    if logits_arr.shape != mask_arr.shape:
        raise ValueError(
            f"Dimensi logits ({logits_arr.shape}) dan mask ({mask_arr.shape}) harus identik."
        )

    # Logit masking deterministik
    masked_logits = np.where(mask_arr > 0.5, logits_arr, mask_value)
    return masked_logits


def masked_softmax(
    logits: np.ndarray,
    mask: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Menghitung probabilitas softmax ternormalisasi khusus pada subruang promising tokens Pt:
    pi_tilde(a | s) = Softmax(Logits(s) + (1 - Mt) * (-inf))_a

    Menghasilkan probabilitas deterministik 0 pada token ekor (tail), dan
    sum(pi_tilde(a | s)) = 1 pada token promising.

    Args:
        logits: Array logit (1D atau 2D).
        mask: Mask biner bernilai 1 (promising) atau 0 (tail).
        temperature: Parameter suhu decoding T > 0.

    Returns:
        np.ndarray: Distribusi probabilitas ternormalisasi di atas subruang Pt.
    """
    if temperature <= 0.0:
        raise ValueError("temperature harus > 0")

    masked_z = apply_logit_mask(logits / temperature, mask, mask_value=-1e9)
    
    # Numerically stable softmax
    if masked_z.ndim == 1:
        # Cari max hanya dari elemen yang aktif
        active = mask > 0.5
        if not np.any(active):
            return np.ones_like(logits) / len(logits)
        max_val = np.max(masked_z[active])
        exp_z = np.where(active, np.exp(masked_z - max_val), 0.0)
        sum_exp = np.sum(exp_z)
        if sum_exp == 0.0:
            probs = np.zeros_like(exp_z)
            probs[active] = 1.0 / np.sum(active)
            return probs
        return (exp_z / sum_exp).astype(np.float32)

    elif masked_z.ndim == 2:
        active = mask > 0.5
        # Max per row
        max_vals = np.zeros((masked_z.shape[0], 1), dtype=np.float64)
        for i in range(masked_z.shape[0]):
            row_active = active[i]
            if np.any(row_active):
                max_vals[i] = np.max(masked_z[i, row_active])
            else:
                max_vals[i] = 0.0

        exp_z = np.where(active, np.exp(masked_z - max_vals), 0.0)
        sum_exp = np.sum(exp_z, axis=-1, keepdims=True)
        # Proteksi row all-zeros
        zero_rows = (sum_exp.squeeze(-1) == 0.0)
        probs = np.zeros_like(exp_z)
        non_zero_rows = ~zero_rows
        if np.any(non_zero_rows):
            probs[non_zero_rows] = exp_z[non_zero_rows] / sum_exp[non_zero_rows]
        if np.any(zero_rows):
            for r_idx in np.where(zero_rows)[0]:
                probs[r_idx, active[r_idx]] = 1.0 / max(1, np.sum(active[r_idx]))
        return probs.astype(np.float32)

    else:
        raise ValueError(f"Input harus 1D atau 2D, diperoleh {masked_z.shape}")


def compute_policy_gradient_variance(
    probs: np.ndarray,
    mask: np.ndarray,
    advantage: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Menghitung perbandingan varians analitik estimator policy gradient:
    V(g_full) = sum_{v in V} pi_v (1 - pi_v) A_t^2
    V(g_mask) = sum_{i in Pt} pi_tilde_i (1 - pi_tilde_i) A_t^2
    Delta_V = V(g_full) - V(g_mask) approx sum_{k in Tt} pi_k (1 - pi_k) A_t^2

    Sesuai pembuktian Proposition 4.1 paper RLPT (arXiv:2602.03195v2).

    Args:
        probs: Probabilitas unmasked penuh over V (1D array).
        mask: Mask biner Pt (1D array).
        advantage: Nilai skalar advantage At (default: 1.0).

    Returns:
        Tuple of (var_full, var_mask, delta_variance):
            - var_full: Total varians gradien pada ruang vocabulary penuh.
            - var_mask: Total varians gradien pada subruang promising tokens.
            - delta_variance: Nilai reduksi varians (harus > 0).
    """
    p_full = np.asarray(probs, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    a2 = float(advantage ** 2)

    # 1. Varians Full
    var_full = float(np.sum(p_full * (1.0 - p_full) * a2))

    # 2. Renormalized probs on subspace Pt
    p_masked = np.where(m > 0.5, p_full, 0.0)
    sum_m = np.sum(p_masked)
    if sum_m > 0:
        p_renorm = p_masked / sum_m
    else:
        p_renorm = p_masked

    # 3. Varians Masked
    var_mask = float(np.sum(p_renorm * (1.0 - p_renorm) * a2))

    # 4. Reduksi varians
    delta_v = var_full - var_mask
    return var_full, var_mask, delta_v


def compute_rlpt_grpo_loss(
    pi_theta_logits: np.ndarray,
    pi_old_logits: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    advantages: np.ndarray,
    clip_eps: float = 0.2,
) -> Tuple[float, Dict[str, float]]:
    """
    Menghitung loss objective RLPT-GRPO (Persamaan 5 Paper):
    J_RLPT(theta) = E [ min( r_t A_t, clip(r_t, 1-eps, 1+eps) A_t ) ]
    di mana r_t = pi_tilde_theta(a_t|s_t) / pi_tilde_old(a_t|s_t)
    dihitung strictly pada subruang promising tokens yang didefinisikan oleh Mt.

    Args:
        pi_theta_logits: Logit model kebijakan aktif theta (batch_size, vocab_size).
        pi_old_logits: Logit model behavior policy old (batch_size, vocab_size).
        actions: Indeks token terpilih a_t (batch_size,).
        mask: Mask biner Mt (batch_size, vocab_size).
        advantages: Advantage skalar A_t per trajektori (batch_size,).
        clip_eps: Parameter GRPO clipping epsilon (default: 0.2).

    Returns:
        Tuple of:
            - loss: Nilai skalar loss negatif (untuk minimasi gradien descent).
            - metrics: Dict statistik (mean_ratio, clip_fraction, mean_advantage).
    """
    pi_theta = masked_softmax(pi_theta_logits, mask)
    pi_old = masked_softmax(pi_old_logits, mask)

    # Ambil probabilitas aksi terpilih
    b_indices = np.arange(len(actions))
    prob_theta_act = pi_theta[b_indices, actions]
    prob_old_act = pi_old[b_indices, actions]

    # Ratio r_t strictly on subspace
    # Hindari pembagian dengan nol
    ratio = prob_theta_act / np.maximum(prob_old_act, 1e-12)

    # GRPO Surrogate Loss
    surr1 = ratio * advantages
    clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr2 = clipped_ratio * advantages

    # Minimasi objeksi negatif -> loss
    obj = np.minimum(surr1, surr2)
    loss = -float(np.mean(obj))

    clip_frac = float(np.mean((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))))
    metrics = {
        "loss": loss,
        "mean_ratio": float(np.mean(ratio)),
        "clip_fraction": clip_frac,
        "mean_advantage": float(np.mean(advantages)),
    }
    return loss, metrics


class PromisingTokenMasker:
    """
    Modul Dynamic Promising Token Subspace Masker (RLPT) tingkat tinggi untuk Lethon.
    Mendukung berbagai strategi thresholding: 'topk', 'topp', 'minp', dan 'subspace_svd'.
    """

    def __init__(
        self,
        strategy: str = "topk",
        k: int = 4,
        p: float = 0.95,
        min_p: float = 0.05,
        energy_threshold: float = 0.90,
    ):
        self.strategy = strategy.lower()
        self.k = k
        self.p = p
        self.min_p = min_p
        self.energy_threshold = energy_threshold

    def create_mask(
        self,
        logits_or_probs: Optional[np.ndarray] = None,
        is_probs: bool = False,
        hidden_states: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Membangun Promising Mask biner Mt berdasarkan strategi yang dikonfigurasi.
        """
        arr = logits if logits is not None else logits_or_probs

        if self.strategy == "topk":
            if arr is None:
                raise ValueError("logits_or_probs diperlukan untuk strategi 'topk'")
            return compute_promising_mask_topk(arr, k=self.k, is_probs=is_probs)

        elif self.strategy == "topp":
            if arr is None:
                raise ValueError("logits_or_probs diperlukan untuk strategi 'topp'")
            if not is_probs:
                probs = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
                probs /= np.sum(probs, axis=-1, keepdims=True)
            else:
                probs = arr
            return compute_promising_mask_topp(probs, p=self.p)

        elif self.strategy == "minp":
            if arr is None:
                raise ValueError("logits_or_probs diperlukan untuk strategi 'minp'")
            if not is_probs:
                probs = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
                probs /= np.sum(probs, axis=-1, keepdims=True)
            else:
                probs = arr
            return compute_promising_mask_minp(probs, min_p=self.min_p)

        elif self.strategy == "subspace_svd":
            if hidden_states is None:
                raise ValueError("hidden_states diperlukan untuk strategi 'subspace_svd'")
            mask, _, _ = compute_subspace_projection_mask(
                hidden_states, energy_threshold=self.energy_threshold
            )
            return mask

        else:
            raise ValueError(f"Strategi tidak dikenal: {self.strategy}")

    def filter_logits(
        self,
        logits: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menerapkan filter masking ke logit dan mengembalikan (masked_logits, mask).
        """
        if mask is None:
            mask = self.create_mask(logits, is_probs=False)
        masked_z = apply_logit_mask(logits, mask)
        return masked_z, mask

    def get_subspace_distribution(
        self,
        logits: np.ndarray,
        mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menghasilkan distribusi probabilitas ternormalisasi di atas subruang promising tokens.
        """
        if mask is None:
            mask = self.create_mask(logits, is_probs=False)
        probs = masked_softmax(logits, mask, temperature=temperature)
        return probs, mask
