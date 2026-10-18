"""
Lethon Stage 4: Sequential Decision MDP & CoT Length Truncation
Berdasarkan paper:
- "THINKPRUNE: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning" (arXiv:2504.01296, Apr 2025)
- "Language-Guided Token Compression with Reinforcement Learning in LVLMs (TPRL)" (arXiv:2603.13394v1)

Menyediakan fungsionalitas:
1. Operator clipping token penalaran (clip(Y, L)).
2. Evaluator reward biner dan komposit ThinkPrune (Zero-reward on truncation).
3. Penghitung Group Relative Advantage untuk GRPO.
4. ThoughtRedundancyAnalyzer (Keyword frequencies, 10 problem-solving phases, redundancy metrics).
5. IterativePruningScheduler (Manajemen kurikulum L_1 > L_2 > ... > L* dan checkpoint selection).
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


def extract_answer_from_text(text: str) -> Optional[str]:
    """
    Mengekstrak jawaban final dari teks penalaran LLM.
    Mendukung format:
    1. <answer>...</answer>
    2. \\boxed{...}
    3. **Final Answer:** ...
    
    Args:
        text: String teks penalaran atau output model.
        
    Returns:
        String jawaban jika ditemukan, atau None jika tag tidak tertutup atau tidak ditemukan.
    """
    if not text:
        return None

    # 1. Cek tag <answer>...</answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # 2. Cek format LaTeX \boxed{...}
    # Mencari \boxed{...} yang seimbang
    boxed_idx = text.rfind(r"\boxed{")
    if boxed_idx != -1:
        start_idx = boxed_idx + len(r"\boxed{")
        depth = 1
        i = start_idx
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start_idx:i - 1].strip()

    # 3. Cek format **Final Answer:** ...
    final_ans_match = re.search(r"\*\*Final Answer:\*\*\s*(.*)", text, flags=re.IGNORECASE)
    if final_ans_match:
        ans_line = final_ans_match.group(1).strip()
        # Ambil baris pertama atau hingga akhir paragraf
        return ans_line.split("\n")[0].strip()

    return None


def clip_reasoning_tokens(
    tokens: List[Union[int, str]],
    length_limit: int,
) -> List[Union[int, str]]:
    """
    Operator clipping sekuens token penalaran clip(Y; L).
    
    Args:
        tokens: Daftar token representasi.
        length_limit: Batas maksimum panjang token L.
        
    Returns:
        Daftar token terpotong dengan panjang min(len(tokens), length_limit).
    """
    if length_limit < 0:
        raise ValueError(f"length_limit harus non-negatif, diperoleh {length_limit}")
    return tokens[:length_limit]


def compute_thinkprune_reward(
    output_tokens: List[Union[int, str]],
    text_content: str,
    ground_truth: str,
    length_limit: int,
    lambda_eff: float = 0.0,
    verifier_fn: Optional[Any] = None,
) -> float:
    """
    Menghitung reward biner atau komposit ThinkPrune:
    R(Y, q; L) = 1 jika jawaban terekstrak dari clip(Y, L) dan benar, 0 jika lainnya.
    
    Jika panjang token melebihi length_limit, output terpotong sehingga jawaban
    tidak dapat diekstrak atau dianggap gagal, menghasilkan reward 0.0.
    
    Args:
        output_tokens: Daftar token yang dihasilkan.
        text_content: Teks lengkap hasil generasi.
        ground_truth: Jawaban referensi (ground truth).
        length_limit: Batas kuota token L.
        lambda_eff: Koefisien bonus efisiensi (default: 0.0 untuk ThinkPrune murni).
        verifier_fn: Fungsi pembanding kustom (opsional). Jika None, menggunakan perbandingan string.
        
    Returns:
        float: Nilai reward.
    """
    token_len = len(output_tokens)
    if token_len > length_limit:
        # Penalti keras: output terpotong melampaui limit L
        return 0.0

    # Ekstrak jawaban dari teks
    extracted = extract_answer_from_text(text_content)
    if extracted is None:
        return 0.0

    # Verifikasi kebenaran
    is_correct = False
    if verifier_fn is not None:
        is_correct = bool(verifier_fn(extracted, ground_truth))
    else:
        # Pembersihan dasar (whitespace, lower-case, dollar signs)
        clean_ext = extracted.replace("$", "").strip().lower()
        clean_gt = ground_truth.replace("$", "").strip().lower()
        is_correct = (clean_ext == clean_gt)

    if not is_correct:
        return 0.0

    # Jawaban benar
    base_reward = 1.0
    if lambda_eff > 0.0 and length_limit > 0:
        eff_ratio = max(0.0, (length_limit - token_len) / length_limit)
        return float(base_reward + lambda_eff * eff_ratio)

    return base_reward


def compute_group_relative_advantage(
    rewards: Union[List[float], np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Menghitung Group Relative Advantage untuk Group Relative Policy Optimization (GRPO):
    A_i = (r_i - mean(r)) / (std(r) + eps)
    
    Args:
        rewards: Array skalar reward untuk suatu grup rollout (ukuran G).
        eps: Nilai stabilisasi numerik pembagian.
        
    Returns:
        np.ndarray: Array keunggulan relatif berukuran sama.
    """
    r_arr = np.asarray(rewards, dtype=np.float32)
    if r_arr.ndim != 1:
        raise ValueError(f"rewards harus berupa array 1D, diperoleh dimensi {r_arr.ndim}")
    
    mean_r = np.mean(r_arr)
    std_r = np.std(r_arr)
    
    if std_r < eps:
        # Jika semua sampel dalam grup memperoleh reward yang identik (misal semua 0 atau semua 1)
        return np.zeros_like(r_arr)
        
    advantages = (r_arr - mean_r) / (std_r + eps)
    return advantages


class ThoughtRedundancyAnalyzer:
    """
    Menganalisis redundansi penalaran Chain-of-Thought (CoT) pada Long-Reasoning LLMs.
    Mengelompokkan kata kunci ke dalam 3 kategori perilaku:
    1. Hesitation / Self-Correction ("Wait", "Hold on", "Alternatively", dll.)
    2. Self-Verification ("Let me confirm", "Double-check", "Remember", dll.)
    3. Core Computation & Reasoning ("Let me compute", "Therefore", "Substitute", dll.)
    
    Serta mendeteksi proporsi 10 fase pemecahan masalah (Problem-Solving Phases).
    """

    HESITATION_KEYWORDS = [
        "wait",
        "but wait",
        "hold on",
        "alternatively",
        "let me think again",
        "let me rethink",
        "now i'm confused",
        "conflicting with my earlier thought",
    ]

    VERIFICATION_KEYWORDS = [
        "let me confirm",
        "double-check",
        "double check",
        "let me verify",
        "let me check",
        "remember",
        "is that correct",
        "just to make sure",
    ]

    CORE_COMPUTATION_KEYWORDS = [
        "let me compute",
        "therefore",
        "hence",
        "substituting",
        "substitute",
        "simplifying",
        "we have",
        "equation",
        "applying",
        "by definition",
    ]

    PROBLEM_SOLVING_PHASES = [
        "understanding_problem",
        "reformulating_problem",
        "applying_theorems",
        "breaking_down_subproblems",
        "computing_expressions",
        "substituting_values",
        "verify_local_steps",
        "reassess_whole_solution",
        "exploring_alternatives",
        "finalize_answer",
    ]

    def __init__(self, avg_tokens_per_word: float = 1.33):
        self.avg_tokens_per_word = avg_tokens_per_word

    def count_keyword_frequencies(
        self,
        text: str,
        token_count: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Menghitung frekuensi absolut dan frekuensi ternormalisasi per 1.000 token
        untuk setiap kategori kata kunci.
        
        Args:
            text: Teks penalaran CoT.
            token_count: Jumlah token dalam teks (jika None, diestimasi dari jumlah kata).
            
        Returns:
            Dict berisi frekuensi absolut dan 'per_1k_tokens' untuk tiap kategori.
        """
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        num_words = len(words)
        
        if token_count is None:
            eff_tokens = max(1, int(num_words * self.avg_tokens_per_word))
        else:
            eff_tokens = max(1, token_count)

        scale_per_1k = 1000.0 / eff_tokens

        results = {
            "hesitation": {"raw_count": 0, "per_1k_tokens": 0.0, "matches": {}},
            "verification": {"raw_count": 0, "per_1k_tokens": 0.0, "matches": {}},
            "core_computation": {"raw_count": 0, "per_1k_tokens": 0.0, "matches": {}},
        }

        # 1. Hesitation
        for kw in self.HESITATION_KEYWORDS:
            cnt = len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
            if cnt > 0:
                results["hesitation"]["matches"][kw] = cnt
                results["hesitation"]["raw_count"] += cnt

        # 2. Verification
        for kw in self.VERIFICATION_KEYWORDS:
            cnt = len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
            if cnt > 0:
                results["verification"]["matches"][kw] = cnt
                results["verification"]["raw_count"] += cnt

        # 3. Core Computation
        for kw in self.CORE_COMPUTATION_KEYWORDS:
            cnt = len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
            if cnt > 0:
                results["core_computation"]["matches"][kw] = cnt
                results["core_computation"]["raw_count"] += cnt

        for cat in ["hesitation", "verification", "core_computation"]:
            results[cat]["per_1k_tokens"] = round(results[cat]["raw_count"] * scale_per_1k, 4)

        return results

    def estimate_problem_solving_phases(
        self,
        text: str,
    ) -> Dict[str, float]:
        """
        Mengestimasi distribusi proporsi langkah pemecahan masalah berdasarkan
        segmentasi paragraf ('\\n\\n') dan penanda linguistik semantik.
        
        Returns:
            Dict proporsi (summing to ~1.0) untuk masing-masing fase.
        """
        # Pisahkan menjadi langkah/segmen berdasarkan double newline
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return {phase: 0.0 for phase in self.PROBLEM_SOLVING_PHASES}

        phase_counts = {phase: 0 for phase in self.PROBLEM_SOLVING_PHASES}

        for idx, para in enumerate(paragraphs):
            p_lower = para.lower()
            
            if idx == 0 and ("problem" in p_lower or "given" in p_lower or "we need to find" in p_lower):
                phase_counts["understanding_problem"] += 1
            elif "<answer>" in p_lower or "final answer" in p_lower or idx == len(paragraphs) - 1:
                phase_counts["finalize_answer"] += 1
            elif any(k in p_lower for k in ["alternatively", "another approach", "different way"]):
                phase_counts["exploring_alternatives"] += 1
            elif any(k in p_lower for k in ["verify the whole", "let me review", "reassess the solution"]):
                phase_counts["reassess_whole_solution"] += 1
            elif any(k in p_lower for k in ["double-check", "let me verify", "let me check", "hold on"]):
                phase_counts["verify_local_steps"] += 1
            elif any(k in p_lower for k in ["let", "assume", "rewrite", "reformulate", "substitute x ="]):
                phase_counts["reformulating_problem"] += 1
            elif any(k in p_lower for k in ["theorem", "property", "lemma", "formula", "identity"]):
                phase_counts["applying_theorems"] += 1
            elif any(k in p_lower for k in ["subproblem", "case 1", "case 2", "break down"]):
                phase_counts["breaking_down_subproblems"] += 1
            elif any(k in p_lower for k in ["substituting", "plugging in", "we get"]):
                phase_counts["substituting_values"] += 1
            else:
                # Default segment komputasi aljabar / aritmetika
                phase_counts["computing_expressions"] += 1

        total_steps = len(paragraphs)
        return {phase: round(count / total_steps, 4) for phase, count in phase_counts.items()}

    def compute_redundancy_score(
        self,
        text: str,
        token_count: Optional[int] = None,
    ) -> float:
        """
        Menghitung skor redundansi komposit S_red in [0, 1].
        Skor tinggi menunjukkan banyak keraguan/verifikasi berulang dibanding komputasi.
        """
        freqs = self.count_keyword_frequencies(text, token_count)
        hes_per_1k = freqs["hesitation"]["per_1k_tokens"]
        ver_per_1k = freqs["verification"]["per_1k_tokens"]
        core_per_1k = freqs["core_computation"]["per_1k_tokens"]

        # Rasio keraguan terhadap komputasi inti
        numerator = hes_per_1k * 2.0 + ver_per_1k
        denominator = numerator + max(1.0, core_per_1k)
        
        score = numerator / denominator
        return round(float(np.clip(score, 0.0, 1.0)), 4)


class IterativePruningScheduler:
    """
    Mengelola kurikulum pemangkasan panjang token iteratif (Iterative Length Pruning):
    L_1 > L_2 > ... > L* (misal: 4000 -> 3000 -> 2000).
    
    Menyediakan logika seleksi checkpoint terbaik berdasarkan kriteria toleransi validasi:
    Akurasi validasi >= (1.0 - max_relative_acc_drop) * Baseline_Acc
    dengan panjang rata-rata token terpendek.
    """

    def __init__(
        self,
        schedule: List[int] = [4000, 3000, 2000],
        max_relative_acc_drop: float = 0.10,
        target_token_reduction: float = 0.50,
    ):
        if not schedule or any(l <= 0 for l in schedule):
            raise ValueError(f"Jadwal schedule harus berupa list integer positif, diperoleh {schedule}")
        # Pastikan terurut menurun
        for i in range(len(schedule) - 1):
            if schedule[i] <= schedule[i + 1]:
                raise ValueError(f"Schedule harus terurut menurun, ditemukan {schedule[i]} <= {schedule[i+1]}")

        self.schedule = list(schedule)
        self.max_relative_acc_drop = max_relative_acc_drop
        self.target_token_reduction = target_token_reduction
        self.current_stage = 0

    @property
    def current_limit(self) -> int:
        """Mengembalikan limit token L pada stage aktif saat ini."""
        return self.schedule[self.current_stage]

    @property
    def is_finished(self) -> bool:
        """Boolean apakah seluruh tahapan schedule telah selesai."""
        return self.current_stage >= len(self.schedule)

    def select_best_checkpoint(
        self,
        candidates: List[Dict[str, Any]],
        baseline_accuracy: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Memilih checkpoint terbaik dari daftar kandidat evaluasi.
        Kriteria ThinkPrune (Section 3.3):
        1. Akurasi >= baseline_accuracy * (1.0 - max_relative_acc_drop)
        2. Memiliki rata-rata panjang output (avg_length) terpendek.
        
        Args:
            candidates: List dict yang memiliki field: 'checkpoint_id', 'accuracy', 'avg_length'.
            baseline_accuracy: Akurasi baseline awal (misal pada AIME22/23).
            
        Returns:
            Dict kandidat terbaik atau None jika tidak ada yang memenuhi syarat akurasi.
        """
        min_allowed_acc = baseline_accuracy * (1.0 - self.max_relative_acc_drop)
        valid_candidates = [c for c in candidates if c.get("accuracy", 0.0) >= min_allowed_acc]
        
        if not valid_candidates:
            return None

        # Sort berdasarkan avg_length terpendek (ascending)
        # Jika panjang sama, pilih akurasi tertinggi
        best = min(valid_candidates, key=lambda c: (c.get("avg_length", float("inf")), -c.get("accuracy", 0.0)))
        return best

    def advance_stage(self) -> Optional[int]:
        """
        Melangkah ke stage berikutnya dalam kurikulum.
        
        Returns:
            Limit token L pada stage baru, atau None jika sudah selesai.
        """
        self.current_stage += 1
        if self.is_finished:
            return None
        return self.current_limit
