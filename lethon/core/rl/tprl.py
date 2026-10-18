"""
Lethon Stage 4: Language-Guided Token Compression with Reinforcement Learning (TPRL)
Based on: arXiv:2603.13394v1 (Language-Guided Token Compression with Reinforcement Learning in LVLMs)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class TokenWiseAutoencoder:
    """
    Lightweight token-wise autoencoder for compact state representation (Stage 4 TPRL).
    Compresses individual token feature vectors h_i in R^{d_v} to z_i in R^{d_l} (d_l << d_v),
    reducing feature dimensionality by ~16x without altering the number of visual/context tokens.
    """

    def __init__(
        self,
        d_v: int = 1024,
        d_hidden: int = 256,
        d_l: int = 64,
        seed: int = 42,
    ) -> None:
        self.d_v = d_v
        self.d_hidden = d_hidden
        self.d_l = d_l
        self.seed = seed
        self.frozen: bool = False

        rng = np.random.RandomState(seed)

        # Encoder weights: d_v -> d_hidden -> d_l
        self.W_e1 = rng.randn(d_v, d_hidden) * np.sqrt(2.0 / d_v)
        self.b_e1 = np.zeros(d_hidden)
        self.W_e2 = rng.randn(d_hidden, d_l) * np.sqrt(2.0 / d_hidden)
        self.b_e2 = np.zeros(d_l)

        # Decoder weights: d_l -> d_hidden -> d_v
        self.W_d1 = rng.randn(d_l, d_hidden) * np.sqrt(2.0 / d_l)
        self.b_d1 = np.zeros(d_hidden)
        self.W_d2 = rng.randn(d_hidden, d_v) * np.sqrt(2.0 / d_hidden)
        self.b_d2 = np.zeros(d_v)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit approximation."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Standard Layer Normalization over the last dimension."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def encode(self, H: np.ndarray) -> np.ndarray:
        """
        Compresses token sequence H in R^{N x d_v} to latent codes Z in R^{N x d_l}.
        """
        if H.ndim == 1:
            H = H.reshape(1, -1)
        h1 = self._gelu(self._layer_norm(H @ self.W_e1 + self.b_e1))
        Z = h1 @ self.W_e2 + self.b_e2
        return Z

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstructs latent codes Z in R^{N x d_l} back to original dimension R^{N x d_v}.
        """
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        h1 = self._gelu(self._layer_norm(Z @ self.W_d1 + self.b_d1))
        H_rec = h1 @ self.W_d2 + self.b_d2
        return H_rec

    def reconstruction_loss(self, H: np.ndarray) -> float:
        """
        Per-token mean squared reconstruction loss:
        L_rec = (1 / N) * sum_{i=1}^N ||h_i - D(z_i)||_2^2
        """
        Z = self.encode(H)
        H_rec = self.decode(Z)
        return float(np.mean(np.square(H - H_rec)))

    def freeze(self) -> None:
        """Freeze encoder parameters for downstream RL policy and inference."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze encoder parameters."""
        self.frozen = False


class SharedAttentionModule:
    """
    Shared Multi-Head Self-Attention module that jointly models the compressed
    visual tokens and the text query representation.
    """

    def __init__(
        self,
        d_l: int = 64,
        d_q: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.d_l = d_l
        self.d_q = d_q
        self.d_model = d_model
        self.n_heads = n_heads

        rng = np.random.RandomState(seed)

        # Query projection
        self.W_q_proj = rng.randn(d_q, d_model) * np.sqrt(1.0 / d_q)
        self.b_q_proj = np.zeros(d_model)

        # Visual token projection if d_l != d_model
        if d_l != d_model:
            self.W_z_proj = rng.randn(d_l, d_model) * np.sqrt(1.0 / d_l)
        else:
            self.W_z_proj = np.eye(d_model)

        # Multi-head attention projections
        self.W_Q = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.W_K = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.W_V = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.W_O = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)

    def forward(self, Z_t: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Processes concatenated sequence X = [Z_t; q_tilde] in R^{(K_t + 1) x d_model}.
        Returns enriched features F in R^{(K_t + 1) x d_model}.
        """
        K_t = Z_t.shape[0]
        if q.ndim == 1:
            q_vec = q.reshape(1, -1)
        else:
            q_vec = q

        q_tilde = q_vec @ self.W_q_proj + self.b_q_proj  # (1, d_model)
        Z_proj = Z_t @ self.W_z_proj  # (K_t, d_model)

        # Concatenate: (K_t + 1, d_model)
        X = np.vstack([Z_proj, q_tilde])

        # Self-Attention
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        scores = (Q @ K.T) / np.sqrt(self.d_model)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        attn_out = attn_weights @ V

        # Residual + LayerNorm
        F = attn_out @ self.W_O + X
        mean = np.mean(F, axis=-1, keepdims=True)
        var = np.var(F, axis=-1, keepdims=True)
        F = (F - mean) / np.sqrt(var + 1e-5)
        return F


class TPRLPolicyNetwork:
    """
    Policy network head that outputs per-token retention probabilities p_{t,i}.
    Supports Bernoulli exploration with step-wise discount decay lambda_disc^t during RL,
    and deterministic one-shot thresholding during inference.
    """

    def __init__(
        self,
        shared_attn: SharedAttentionModule,
        d_model: int = 64,
        lambda_disc: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.shared_attn = shared_attn
        self.d_model = d_model
        self.lambda_disc = lambda_disc

        rng = np.random.RandomState(seed)
        self.W_p1 = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.b_p1 = np.zeros(d_model)
        self.W_p2 = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.b_p2 = np.zeros(d_model)
        self.w_out = rng.randn(d_model) * np.sqrt(1.0 / d_model)
        self.b_out = 0.0

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward_probabilities(self, Z_t: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculates raw retention probabilities p_{t,i} in [0, 1] for all K_t visual tokens.
        """
        K_t = Z_t.shape[0]
        F = self.shared_attn.forward(Z_t, q)
        F_visual = F[:K_t]  # (K_t, d_model)

        h1 = self._gelu(F_visual @ self.W_p1 + self.b_p1)
        g = h1 @ self.W_p2 + self.b_p2 + F_visual

        logits = g @ self.w_out + self.b_out
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return p

    def sample_actions(
        self,
        Z_t: np.ndarray,
        q: np.ndarray,
        step: int = 1,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RL exploration sampling from Bernoulli distribution with step-wise discount factor:
        P(a_{t,i} = 1) = lambda_disc^t * p_{t,i}
        """
        if rng is None:
            rng = np.random.RandomState()

        p = self.forward_probabilities(Z_t, q)
        discounted_p = (self.lambda_disc ** step) * p
        discounted_p = np.clip(discounted_p, 0.0, 1.0)

        actions = (rng.rand(*discounted_p.shape) < discounted_p).astype(np.int32)
        return actions, p

    def deterministic_actions(
        self,
        Z_t: np.ndarray,
        q: np.ndarray,
        threshold: float = 0.55,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministic one-shot retention decision: a_i = 1{p_i > threshold}.
        Guarantees that at least one token is preserved.
        """
        p = self.forward_probabilities(Z_t, q)
        actions = (p > threshold).astype(np.int32)
        if np.sum(actions) == 0:
            actions[int(np.argmax(p))] = 1
        return actions, p


class TPRLValueNetwork:
    """
    Critic / Value network head that estimates state value V_phi(s_t)
    by mean-pooling enriched visual token features.
    """

    def __init__(
        self,
        shared_attn: SharedAttentionModule,
        d_model: int = 64,
        seed: int = 42,
    ) -> None:
        self.shared_attn = shared_attn
        self.d_model = d_model

        rng = np.random.RandomState(seed + 100)
        self.U1 = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.c1 = np.zeros(d_model)
        self.U2 = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.c2 = np.zeros(d_model)
        self.u = rng.randn(d_model) * np.sqrt(1.0 / d_model)
        self.b_v = 0.0

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward(self, Z_t: np.ndarray, q: np.ndarray) -> float:
        """
        Estimates V_phi(s_t) from aggregated token representations.
        """
        K_t = Z_t.shape[0]
        F = self.shared_attn.forward(Z_t, q)
        F_visual = F[:K_t]

        # Mean pooling: f_bar = (1 / K_t) * sum(f_i)
        f_bar = np.mean(F_visual, axis=0)

        # FFN + LayerNorm
        h1 = self._gelu(f_bar @ self.U1 + self.c1)
        h2 = h1 @ self.U2 + self.c2 + f_bar
        mean = np.mean(h2)
        var = np.var(h2)
        h2_norm = (h2 - mean) / np.sqrt(var + 1e-5)

        v = float(h2_norm @ self.u + self.b_v)
        return v


def compute_tprl_reward(
    scores_current: Union[float, List[float], np.ndarray],
    scores_prev: Union[float, List[float], np.ndarray],
    tokens_current: Union[int, List[int], np.ndarray],
    tokens_prev: Union[int, List[int], np.ndarray],
    alpha: float = 1.0,
    beta: float = 0.1,
) -> Dict[str, float]:
    """
    Computes TPRL dual-criterion reward over mini-batch of B samples:
    r_t = R_task + R_eff(t)
    R_task = alpha * (1/B) * sum(Score(H_t) - Score(H_{t-1}))
    R_eff  = beta  * (1/B) * sum(1 - K_t / K_{t-1})
    """
    sc = np.atleast_1d(scores_current)
    sp = np.atleast_1d(scores_prev)
    kc = np.atleast_1d(tokens_current).astype(np.float64)
    kp = np.atleast_1d(tokens_prev).astype(np.float64)

    # Avoid division by zero
    kp_safe = np.maximum(kp, 1.0)

    delta_score = sc - sp
    r_task = float(alpha * np.mean(delta_score))

    reduction_ratio = 1.0 - (kc / kp_safe)
    r_eff = float(beta * np.mean(reduction_ratio))

    r_total = float(r_task + r_eff)

    return {
        "r_total": r_total,
        "r_task": r_task,
        "r_eff": r_eff,
        "mean_delta_score": float(np.mean(delta_score)),
        "mean_reduction_ratio": float(np.mean(reduction_ratio)),
    }


def compute_gae_advantage(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Generalized Advantage Estimation (GAE) for trajectory of length T:
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    hat{A}_t = sum_{l=0}^{T - t - 1} (gamma * lam)^l * delta_{t+l}
    """
    T = len(rewards)
    deltas = np.zeros(T, dtype=np.float64)
    advantages = np.zeros(T, dtype=np.float64)

    for t in range(T):
        next_val = values[t + 1] if (t + 1) < len(values) else 0.0
        deltas[t] = rewards[t] + gamma * next_val - values[t]

    running_adv = 0.0
    for t in reversed(range(T)):
        running_adv = deltas[t] + (gamma * lam) * running_adv
        advantages[t] = running_adv

    returns = advantages + np.array(values[:T])
    return advantages, returns


def compute_lfd_loss(
    predicted_probs: np.ndarray,
    demonstration_actions: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Computes Binary Cross-Entropy loss for Learning from Demonstrations (LfD):
    L_BCE = - (1 / K) * sum [ a_demo * log(p) + (1 - a_demo) * log(1 - p) ]
    """
    p = np.clip(predicted_probs, eps, 1.0 - eps)
    a = demonstration_actions
    loss = -np.mean(a * np.log(p) + (1.0 - a) * np.log(1.0 - p))
    return float(loss)


class TPRLPruner:
    """
    High-level coordinator for Lethon Stage 4 TPRL token compression:
    1. Feature compression via token-wise autoencoder (N x d_v -> N x d_l)
    2. Multi-step MDP state transition and mask tracking on original token indices
    3. Deterministic one-shot inference thresholding
    """

    def __init__(
        self,
        autoencoder: Optional[TokenWiseAutoencoder] = None,
        policy: Optional[TPRLPolicyNetwork] = None,
        d_v: int = 1024,
        d_hidden: int = 256,
        d_l: int = 64,
        d_q: int = 64,
        seed: int = 42,
    ) -> None:
        if autoencoder is None:
            self.autoencoder = TokenWiseAutoencoder(
                d_v=d_v, d_hidden=d_hidden, d_l=d_l, seed=seed
            )
        else:
            self.autoencoder = autoencoder

        if policy is None:
            shared_attn = SharedAttentionModule(
                d_l=d_l, d_q=d_q, d_model=d_l, seed=seed
            )
            self.policy = TPRLPolicyNetwork(
                shared_attn=shared_attn, d_model=d_l, seed=seed
            )
        else:
            self.policy = policy

    def prune_one_shot(
        self,
        H: np.ndarray,
        q: np.ndarray,
        threshold: float = 0.55,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes one-shot deterministic pruning protocol:
        - Input: H in R^{N x d_v}, q in R^{d_q}
        - Output: H_retained in R^{K x d_v}, binary mask in {0, 1}^N, metadata stats
        """
        N = H.shape[0]

        # 1. Compress token features using frozen autoencoder
        Z = self.autoencoder.encode(H)

        # 2. Compute retention probabilities & threshold
        actions, probs = self.policy.deterministic_actions(Z, q, threshold=threshold)

        # 3. Apply mask to original uncompressed tokens
        retained_indices = np.where(actions == 1)[0]
        H_retained = H[retained_indices]

        stats = {
            "original_tokens": N,
            "retained_tokens": int(len(retained_indices)),
            "pruning_ratio": float(1.0 - len(retained_indices) / N),
            "mean_retention_prob": float(np.mean(probs)),
            "threshold": threshold,
            "active_indices": retained_indices.tolist(),
        }

        return H_retained, actions, stats

    def simulate_multistep_trajectory(
        self,
        H: np.ndarray,
        q: np.ndarray,
        max_steps: int = 3,
        step_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Simulates multi-step MDP trajectory (Z_{t+1} = {z_{t,i} | d_i = 1})
        useful for offline analysis and LfD validation.
        """
        N = H.shape[0]
        mask = np.ones(N, dtype=np.int32)
        active_indices = np.arange(N)

        trajectory = []
        Z_current = self.autoencoder.encode(H)

        for step in range(1, max_steps + 1):
            if len(active_indices) <= 1:
                break

            probs = self.policy.forward_probabilities(Z_current, q)
            step_actions = (probs > step_threshold).astype(np.int32)
            if np.sum(step_actions) == 0:
                step_actions[int(np.argmax(probs))] = 1

            # Update global mask
            pruned_local = np.where(step_actions == 0)[0]
            mask[active_indices[pruned_local]] = 0

            # Next step active set
            kept_local = np.where(step_actions == 1)[0]
            active_indices = active_indices[kept_local]
            Z_current = Z_current[kept_local]

            trajectory.append({
                "step": step,
                "remaining_tokens": int(len(active_indices)),
                "step_pruned_count": int(len(pruned_local)),
                "cumulative_pruning_ratio": float(1.0 - len(active_indices) / N),
                "active_indices": active_indices.copy(),
            })

        return trajectory
