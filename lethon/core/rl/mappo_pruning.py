"""
Lethon Cross-Stage Optimization: Markov Game Token Pruning with Multi-Agent PPO (MAPPO)
Berdasarkan paper:
- "RL-based Token Pruning in ViTs: A Markov Game Approach" (arXiv:2503.23459, Lu et al., Fudan & Universite Paris Cite, 2025)

Fitur utama:
1. Actor Network (π_θ): Shared MLP untuk keputusan pruning lokal per-token (A_i ∈ {0, 1}).
2. Critic Network (V_ϕ): Shared MLP dengan evaluasi global via input gabungan [s_local || s_global].
3. Dual-Criteria Reward:
   - r_{1,i}^l = 0 jika dipangkas, -1 jika dipertahankan (kompetisi anggaran komputasi).
   - r_{2,i}^l = γ / (n_l + ε) (kolaborasi akurasi downstream diamortisasi terhadap jumlah token aktif).
   - R_i^l = α * r_{1,i}^l + β * r_{2,i}^l.
4. Non-Destructive Softmax Masking: Token yang dipangkas diisi bias -1000.0 pada attention matrix.
5. Markov Game Trajectory & GAE Multi-Agent Advantage Estimation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


class MAPPOActorNetwork:
    """
    Shared MLP Actor Network (π_θ) untuk setiap agen token.
    Ukuran arsitektur default sesuai paper: d -> 4d -> 256 -> 64 -> 2.
    """

    def __init__(
        self,
        d_in: int = 384,
        d_hidden: int = 1536,
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.seed = seed

        rng = np.random.RandomState(seed)

        # 5-layer MLP: d_in -> d_hidden -> 256 -> 64 -> 2
        self.W1 = rng.randn(d_in, d_hidden) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hidden)

        self.W2 = rng.randn(d_hidden, 256) * np.sqrt(2.0 / d_hidden)
        self.b2 = np.zeros(256)

        self.W3 = rng.randn(256, 64) * np.sqrt(2.0 / 256)
        self.b3 = np.zeros(64)

        self.W4 = rng.randn(64, 2) * np.sqrt(2.0 / 64)
        self.b4 = np.zeros(2)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass menghasilkan logits aksi [z0 (prune), z1 (preserve)].
        Args:
            x: Fitur lokal token, shape (..., d_in)
        Returns:
            logits biner, shape (..., 2)
        """
        h1 = self._gelu(np.dot(x, self.W1) + self.b1)
        h2 = self._gelu(np.dot(h1, self.W2) + self.b2)
        h3 = self._gelu(np.dot(h2, self.W3) + self.b3)
        logits = np.dot(h3, self.W4) + self.b4
        return logits

    def get_action_probs(self, x: np.ndarray) -> np.ndarray:
        """
        Menghitung distribusi probabilitas softmax atas aksi biner [P(prune), P(preserve)].
        """
        logits = self.forward_logits(x)
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)
        return probs

    def sample_actions(
        self,
        x: np.ndarray,
        deterministic: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mengambil keputusan pruning untuk setiap token.
        Args:
            x: Fitur lokal token (..., N, d_in)
            deterministic: Jika True, pilih argmax.
            rng: Random generator untuk stochastic sampling.
        Returns:
            actions: Array binary (0: prune, 1: preserve)
            probs: Probabilitas aksi yang terpilih
        """
        probs = self.get_action_probs(x)
        preserve_prob = probs[..., 1]

        if deterministic:
            actions = (preserve_prob >= 0.5).astype(np.int32)
        else:
            if rng is None:
                rng = np.random.RandomState()
            random_draw = rng.uniform(0.0, 1.0, size=preserve_prob.shape)
            actions = (random_draw < preserve_prob).astype(np.int32)

        # Probabilitas dari aksi terpilih
        selected_probs = np.where(actions == 1, probs[..., 1], probs[..., 0])
        return actions, selected_probs


class MAPPOCriticNetwork:
    """
    Centralized Critic Network (V_ϕ) yang mengevaluasi nilai keadaan
    dari sudut pandang global via fitur gabungan [s_local || s_global].
    Ukuran arsitektur default: 2d -> 4d -> 256 -> 64 -> 1.
    """

    def __init__(
        self,
        d_token: int = 384,
        d_global: int = 384,
        d_hidden: int = 1536,
        seed: int = 42,
    ) -> None:
        self.d_in = d_token + d_global
        self.d_hidden = d_hidden
        self.seed = seed

        rng = np.random.RandomState(seed)

        self.W1 = rng.randn(self.d_in, d_hidden) * np.sqrt(2.0 / self.d_in)
        self.b1 = np.zeros(d_hidden)

        self.W2 = rng.randn(d_hidden, 256) * np.sqrt(2.0 / d_hidden)
        self.b2 = np.zeros(256)

        self.W3 = rng.randn(256, 64) * np.sqrt(2.0 / 256)
        self.b3 = np.zeros(64)

        self.W4 = rng.randn(64, 1) * np.sqrt(2.0 / 64)
        self.b4 = np.zeros(1)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward_value(self, s_local: np.ndarray, s_global: np.ndarray) -> np.ndarray:
        """
        Menghitung nilai skalar V(s_local, s_global) untuk setiap agen token.
        Args:
            s_local: Shape (..., N, d_token)
            s_global: Shape (..., d_global) atau (..., N, d_global)
        Returns:
            V: Shape (..., N)
        """
        if s_global.ndim == s_local.ndim - 1:
            # Broadcast s_global sepanjang axis token N
            s_global = np.repeat(np.expand_dims(s_global, axis=-2), s_local.shape[-2], axis=-2)

        fused = np.concatenate([s_local, s_global], axis=-1)

        h1 = self._gelu(np.dot(fused, self.W1) + self.b1)
        h2 = self._gelu(np.dot(h1, self.W2) + self.b2)
        h3 = self._gelu(np.dot(h2, self.W3) + self.b3)
        val = np.dot(h3, self.W4) + self.b4
        return np.squeeze(val, axis=-1)


def compute_mappo_token_rewards(
    actions: np.ndarray,
    task_accuracy: float,
    alpha: float = 0.8,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menghitung reward Markov Game MAPPO sesuai Persamaan (1) & (2) paper:
    r_1,i^l = 0 jika action=0 (pruned), -1 jika action=1 (preserved)
    r_2,i^l = gamma_task / (n_l + eps)
    R_i^l = alpha * r_1,i^l + beta * r_2,i^l

    Args:
        actions: Binary actions array (N,) atau (B, N) di mana 0: prune, 1: preserve.
        task_accuracy: gamma task (skalar 0.0 hingga 1.0).
        alpha: Bobot penalti efisiensi / kompetisi.
        beta: Bobot reward akurasi teramortisasi / kolaborasi.
        eps: Small epsilon untuk stabilitas numerik.

    Returns:
        r1: Array reward efisiensi
        r2: Array reward akurasi
        R_total: Array total reward komposit
    """
    actions_arr = np.asarray(actions, dtype=np.float32)

    # 1. r_1i = 0 if pruned (A=0), -1 if preserved (A=1)
    r1 = np.where(actions_arr == 1.0, -1.0, 0.0)

    # 2. r_2i = gamma / n_l
    if actions_arr.ndim == 1:
        n_l = float(np.sum(actions_arr))
        r2 = np.full_like(actions_arr, fill_value=(task_accuracy / (n_l + eps)))
    else:
        # Multi-batch: sum across tokens dimension
        n_l = np.sum(actions_arr, axis=-1, keepdims=True)
        r2 = np.repeat(task_accuracy / (n_l + eps), actions_arr.shape[-1], axis=-1)

    R_total = alpha * r1 + beta * r2
    return r1, r2, R_total


def apply_attention_mask_pruning(
    attention_scores: np.ndarray,
    token_mask: np.ndarray,
    mask_bias: float = -1000.0,
) -> np.ndarray:
    """
    Menerapkan pemangkasan non-destruktif ke matriks perhatian (attention scores)
    sebelum Softmax, sesuai protokol DynamicViT & RL4EViT (arXiv:2503.23459 halaman 3):
    Jika token dipangkas (mask == 0), tambahkan -1000.0 agar nilai softmax = 0.

    Args:
        attention_scores: Matriks atensi (B, num_heads, N, N) atau (N, N).
        token_mask: Binary mask (B, N) atau (N,) dengan 1: preserved, 0: pruned.
        mask_bias: Nilai bias negatif besar (default -1000.0).

    Returns:
        masked_attention: Matriks atensi yang telah dimask.
    """
    scores = np.copy(attention_scores)
    mask = np.asarray(token_mask, dtype=np.float32)

    # Buat 2D mask: token j dipangkas -> kolom j dimask; token i dipangkas -> baris i dimask
    if mask.ndim == 1:
        # 1D: (N,) -> outer product
        mask_2d = np.outer(mask, mask)  # 1 jika keduanya aktif, 0 jika salah satu pruned
        penalty = (1.0 - mask_2d) * mask_bias
        return scores + penalty
    elif mask.ndim == 2:
        # 2D: (B, N) -> (B, 1, N, N)
        mask_2d = np.einsum("bi,bj->bij", mask, mask)
        penalty = (1.0 - mask_2d) * mask_bias
        if scores.ndim == 4:
            # (B, H, N, N)
            penalty = np.expand_dims(penalty, axis=1)
        return scores + penalty
    else:
        raise ValueError(f"Dimensi token_mask tidak didukung: {mask.shape}")


def compute_mappo_gae(
    rewards: List[np.ndarray],
    values: List[np.ndarray],
    gamma_disc: float = 0.99,
    lambda_gae: float = 0.95,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Menghitung Generalized Advantage Estimation (GAE) untuk Markov Game lintas layer.

    Args:
        rewards: List array reward [R^1, R^2, ..., R^L], masing-masing shape (N_tokens,).
        values: List array value [V^1, V^2, ..., V^L, V^{L+1}], masing-masing shape (N_tokens,).
        gamma_disc: Discount factor.
        lambda_gae: GAE parameter.

    Returns:
        advantages: List array advantage [A^1, ..., A^L].
        returns: List array target returns [G^1, ..., G^L].
    """
    num_layers = len(rewards)
    advantages = [np.zeros_like(r) for r in rewards]
    returns = [np.zeros_like(r) for r in rewards]

    gae = np.zeros_like(rewards[-1])

    for l in reversed(range(num_layers)):
        r_l = rewards[l]
        v_l = values[l]
        v_next = values[l + 1] if (l + 1 < len(values)) else np.zeros_like(v_l)

        # TD residual delta_l = r_l + gamma * V_{l+1} - V_l
        delta = r_l + gamma_disc * v_next - v_l
        gae = delta + gamma_disc * lambda_gae * gae

        advantages[l] = gae
        returns[l] = gae + v_l

    return advantages, returns


def compute_mappo_policy_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    clip_eps: float = 0.2,
) -> float:
    """
    Menghitung PPO Clipped Surrogate Loss untuk Actor Network sesuai Persamaan (3) paper:
    L_theta = - (1/M) * sum( min( r * A, clip(r, 1-eps, 1+eps) * A ) )

    Args:
        log_probs_new: Log probabilitas dari policy baru.
        log_probs_old: Log probabilitas dari policy lama.
        advantages: GAE advantage array.
        clip_eps: Epsilon clipping (default 0.2).

    Returns:
        loss skalar (untuk diminimalkan).
    """
    ratio = np.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    clipped_obj = np.minimum(surr1, surr2)
    return -float(np.mean(clipped_obj))


def compute_mappo_value_loss(
    values_new: np.ndarray,
    values_old: np.ndarray,
    returns: np.ndarray,
    clip_eps: float = 0.2,
) -> float:
    """
    Menghitung Clipped Value Loss untuk Critic Network sesuai Persamaan (4) paper:
    L_phi = (1/M) * sum( max( (V_new - R)^2, (clip(V_new, V_old +- eps) - R)^2 ) )

    Args:
        values_new: Estimasi value baru V_new.
        values_old: Estimasi value lama V_old.
        returns: Target returns R.
        clip_eps: Clipping threshold untuk value update.

    Returns:
        loss skalar.
    """
    v_diff1 = np.square(values_new - returns)
    v_clipped = np.clip(values_new, values_old - clip_eps, values_old + clip_eps)
    v_diff2 = np.square(v_clipped - returns)
    loss = np.maximum(v_diff1, v_diff2)
    return float(np.mean(loss))
