"""
Lethon Stage 4 Entropy-Guided Regulation & Curriculum Learning (SENT)
Berdasarkan paper: "Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning"
(arXiv:2512.04359v3)

Modul ini mengimplementasikan:
1. Perhitungan Semantic Entropy (SE) berbasis pengelompokan ekuivalensi semantik (klaster jawaban/makna).
2. Semantic Entropy Curriculum Scheduler (pengurutan data D_sorted = sort(D, key=H_SE) dan partisi N tahap).
3. Token-Level Entropy & Covariance Computation (log_pi - mean_log_pi) * (A_t - mean_A).
4. Non-Uniform KL Divergence Regularization (beta_con: 0 untuk token entropi tinggi,
   beta_low untuk token entropi rendah reguler, beta_high untuk token entropi rendah bervariansi tinggi).
5. Loss Objective SENT & Pencegahan Entropy Collapse pada pemadatan Chain-of-Thought (CoT).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Sequence
import numpy as np


@dataclass
class SemanticCluster:
    """Representasi klaster ekuivalensi semantik untuk kueri tertentu."""
    cluster_id: int
    representative_answer: str
    sample_indices: List[int] = field(default_factory=list)
    raw_probability_sum: float = 0.0
    normalized_probability: float = 0.0


def compute_semantic_entropy(
    responses: Sequence[str],
    probabilities: Sequence[float],
    answer_extractor_fn: Optional[Any] = None,
    normalize_probs: bool = True,
    epsilon: float = 1e-12,
) -> Tuple[float, List[SemanticCluster]]:
    """
    Menghitung Semantic Entropy (H_SE) untuk sekumpulan respons yang dihasilkan model untuk satu kueri q.
    
    Formula:
    P(C_k | q) = sum_{o in C_k} P(o | q)
    hat{P}(C_k | q) = P(C_k | q) / sum_{j=1}^K P(C_j | q)
    H_SE(q) = - sum_{k=1}^K hat{P}(C_k | q) * log(hat{P}(C_k | q) + eps)
    
    Args:
        responses: Daftar respons teks dari rollout M sampling.
        probabilities: Probabilitas sekuens/respons terkait P(o | q).
        answer_extractor_fn: Fungsi pemroses jawaban/makna (misal ekstraksi numerik/string kanonikal).
        normalize_probs: Apakah menormalisasi probabilitas klaster menjadi distribusi valid.
        epsilon: Nilai numerik kecil untuk mencegah log(0).
        
    Returns:
        Tuple (H_SE, list_of_clusters)
    """
    if len(responses) == 0:
        return 0.0, []
    if len(responses) != len(probabilities):
        raise ValueError("Jumlah respons harus sama dengan jumlah probabilitas.")

    # 1. Ekstraksi representasi makna / jawaban ekuivalen
    clusters_map: Dict[str, List[int]] = {}
    for idx, resp in enumerate(responses):
        if answer_extractor_fn is not None:
            ans_key = str(answer_extractor_fn(resp)).strip().lower()
        else:
            ans_key = str(resp).strip().lower()
        
        if ans_key not in clusters_map:
            clusters_map[ans_key] = []
        clusters_map[ans_key].append(idx)

    # 2. Hitung probabilitas per klaster semantik
    clusters: List[SemanticCluster] = []
    total_prob_sum = 0.0

    for c_id, (ans_key, sample_idxs) in enumerate(clusters_map.items()):
        c_prob = float(sum(probabilities[i] for i in sample_idxs))
        clusters.append(
            SemanticCluster(
                cluster_id=c_id,
                representative_answer=ans_key,
                sample_indices=sample_idxs,
                raw_probability_sum=c_prob,
            )
        )
        total_prob_sum += c_prob

    if total_prob_sum <= 0.0:
        total_prob_sum = 1.0

    # 3. Normalisasi distribusi dan kalkulasi entropi Shannon atas klaster
    h_se = 0.0
    for cluster in clusters:
        if normalize_probs:
            p_hat = cluster.raw_probability_sum / total_prob_sum
        else:
            p_hat = cluster.raw_probability_sum
        cluster.normalized_probability = float(p_hat)

        if p_hat > 0.0:
            h_se -= p_hat * np.log(p_hat + epsilon)

    return float(max(0.0, h_se)), clusters


class SemanticEntropyCurriculum:
    """
    Pengatur kurikulum data berbasis Semantic Entropy (H_SE).
    Mengurutkan dataset D berdasarkan tingkat kesulitan inheren (dari rendah ke tinggi):
    D_sorted = sort(D, key=H_SE)
    Membagi dataset ke dalam N tahap kurikulum (default: N=2 tahap).
    """

    def __init__(self, num_stages: int = 2):
        if num_stages < 1:
            raise ValueError("Jumlah tahap kurikulum minimal 1.")
        self.num_stages = num_stages

    def partition_dataset(
        self,
        dataset: Sequence[Dict[str, Any]],
        entropy_key: str = "semantic_entropy",
    ) -> List[List[Dict[str, Any]]]:
        """
        Mengurutkan dataset berdasarkan semantic_entropy dan membaginya ke dalam N tahap.
        """
        sorted_data = sorted(dataset, key=lambda item: item.get(entropy_key, 0.0))
        n_samples = len(sorted_data)
        if n_samples == 0:
            return [[] for _ in range(self.num_stages)]

        stage_size = int(np.ceil(n_samples / self.num_stages))
        stages: List[List[Dict[str, Any]]] = []
        
        for stage_idx in range(self.num_stages):
            start = stage_idx * stage_size
            end = min(n_samples, (stage_idx + 1) * stage_size)
            stages.append(list(sorted_data[start:end]))

        return stages


def compute_token_entropy(logits_or_probs: np.ndarray, is_logits: bool = True, epsilon: float = 1e-12) -> np.ndarray:
    """
    Menghitung entropi Shannon per posisi token t atas ruang kosakata V.
    
    H_t = - sum_{v in V} p_{t,v} * log(p_{t,v} + eps)
    
    Args:
        logits_or_probs: Matriks (..., vocab_size) berisi logit atau probabilitas.
        is_logits: Jika True, terapkan softmax numerik stabil terlebih dahulu.
    Returns:
        Array entropi token (...)
    """
    x = np.asarray(logits_or_probs, dtype=np.float64)
    if is_logits:
        # Stable softmax
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        probs = exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + epsilon)
    else:
        probs = x / (np.sum(x, axis=-1, keepdims=True) + epsilon)

    token_entropy = -np.sum(probs * np.log(np.maximum(probs, epsilon)), axis=-1)
    return np.maximum(0.0, token_entropy)


def compute_token_advantage_covariance(
    log_probs: np.ndarray,
    advantages: np.ndarray,
) -> np.ndarray:
    """
    Menghitung kovarians token-level antara confidence (log probability) dan sinyal keuntungan (advantage)
    sesuai Persamaan (12) paper SENT:
    
    Cov_t = (log pi(o_t | s_t) - mean(log pi)) * (A_t - mean(A))
    
    Args:
        log_probs: Array 1D log probabilitas token terpilih dalam sekuens/batch.
        advantages: Array 1D advantage yang dinormalisasi pada token tersebut.
    Returns:
        Array 1D kovarians per token.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)

    if lp.shape != adv.shape:
        raise ValueError(f"Shape mismatch antara log_probs {lp.shape} dan advantages {adv.shape}")

    if len(lp) == 0:
        return np.array([], dtype=np.float64)

    mean_lp = np.mean(lp)
    mean_adv = np.mean(adv)

    covariance = (lp - mean_lp) * (adv - mean_adv)
    return covariance


@dataclass
class TokenRegularizationSchedule:
    """
    Konfigurasi penjadwalan regulasi KL non-uniform pada token berdasarkan kriteria entropi & kovarians.
    Default hyperparameters terkalibrasi dari riset SENT (arXiv:2512.04359):
    - tau_H_percentile: 80% (top 80% token terendah entropinya dianggap rentan runtuh)
    - tau_cov: 0.0002 (ambang kovarians tinggi)
    - beta_high: 2.0 (regulasi ketat untuk low-entropy + high-covariance tokens)
    - beta_low: 0.5 (regulasi moderat untuk low-entropy tokens)
    - beta_base: 0.0 (bebas regulasi untuk high-entropy exploratory tokens)
    """
    tau_H_percentile: float = 80.0
    tau_cov: float = 0.0002
    beta_high: float = 2.0
    beta_low: float = 0.5
    beta_base: float = 0.0

    def compute_non_uniform_kl_coefficients(
        self,
        token_entropies: np.ndarray,
        token_covariances: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        r"""
        Menentukan koefisien regulasi KL (beta_con) per posisi token ot sesuai Persamaan (14):
        
        beta_con(t) =
            beta_high,  jika ot in T_high-cov (H_t < tau_H dan Cov_t > tau_cov)
            beta_low,   jika ot in T_low \ T_high-cov (H_t < tau_H dan Cov_t <= tau_cov)
            beta_base,  jika ot not in T_low (H_t >= tau_H)
        
        Returns:
            Tuple (beta_con array, metadata statistik)
        """
        entropies = np.asarray(token_entropies, dtype=np.float64)
        covariances = np.asarray(token_covariances, dtype=np.float64)
        n_tokens = len(entropies)

        if n_tokens == 0:
            return np.array([], dtype=np.float64), {}

        # Tentukan batas entropi rendah menggunakan persentil
        tau_H = float(np.percentile(entropies, self.tau_H_percentile))

        # Masking grup token
        mask_low_entropy = entropies < tau_H
        mask_high_cov = (covariances > self.tau_cov) & mask_low_entropy

        beta_con = np.full(n_tokens, self.beta_base, dtype=np.float64)
        beta_con[mask_low_entropy] = self.beta_low
        beta_con[mask_high_cov] = self.beta_high

        stats = {
            "tau_H_threshold": tau_H,
            "tau_cov_threshold": self.tau_cov,
            "total_tokens": n_tokens,
            "num_low_entropy": int(np.sum(mask_low_entropy)),
            "num_high_cov_low_entropy": int(np.sum(mask_high_cov)),
            "ratio_high_cov": float(np.mean(mask_high_cov)),
            "ratio_low_entropy": float(np.mean(mask_low_entropy)),
        }
        return beta_con, stats


def compute_sent_policy_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    advantages: np.ndarray,
    beta_con: np.ndarray,
    clip_eps: float = 0.2,
) -> Tuple[float, Dict[str, float]]:
    """
    Menghitung fungsi objektif loss SENT (Persamaan 15):
    
    J_SENT(theta) = E [ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) - beta_con * D_KL(pi_theta || pi_ref) ]
    
    di mana D_KL aproksimasi numerik per-token:
    D_KL(pi_theta || pi_ref) approx exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1
    atau standar log_pi - log_ref. Di sini digunakan formula KL per-token tak-bias k3:
    kl_div = (exp(log_ref - log_pi) - 1) - (log_ref - log_pi)
    
    Loss yang diminimalkan optimizer adalah -J_SENT(theta).
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    old_lp = np.asarray(old_log_probs, dtype=np.float64)
    ref_lp = np.asarray(ref_log_probs, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    b_con = np.asarray(beta_con, dtype=np.float64)

    # 1. Rasio probabilitas r_t(theta) = exp(log pi_theta - log pi_old)
    ratio = np.exp(lp - old_lp)

    # 2. Clipped policy gradient objective
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_objective = np.minimum(surr1, surr2)

    # 3. KL Divergence penalty per token terhadap model referensi (pi_ref)
    # Gunakan aproksimasi standar Schulman / reverse KL:
    # kl approx log(pi_theta / pi_ref)
    log_ratio_ref = lp - ref_lp
    # Atau k1 estimator: kl = log_ratio_ref
    kl_div = log_ratio_ref

    kl_penalty = b_con * kl_div

    # 4. Total objective per token dan loss skalar
    token_objective = pg_objective - kl_penalty
    total_loss = -float(np.mean(token_objective))

    metrics = {
        "mean_pg_objective": float(np.mean(pg_objective)),
        "mean_kl_penalty": float(np.mean(kl_penalty)),
        "mean_kl_div": float(np.mean(kl_div)),
        "mean_beta_con": float(np.mean(b_con)),
        "loss": total_loss,
    }
    return total_loss, metrics
