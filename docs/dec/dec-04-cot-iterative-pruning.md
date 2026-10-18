# DEC-04: Sequential Decision MDP & CoT Length Truncation (Stage 4)

## 1. Overview
Modul **Sequential Decision MDP & CoT Length Truncation** mengotomatiskan pemangkasan langkah penalaran *Chain-of-Thought* (CoT) yang berulang atau bertele-tele pada model penalaran panjang (seperti DeepSeek-R1 / o1 / o3) melalui optimasi kebijakan berbasis Reinforcement Learning.

## 2. Formulasi Markov Decision Process (MDP)

Proses pemangkasan dimodelkan sebagai tuple $(S, A, P, R, \gamma)$:

1. **State ($s_t$):** Representasi ringkas dari langkah penalaran ke-$t$ yang dikodekan melalui *token-wise autoencoder* laten ($d_l \ll d_v$).
2. **Action ($a_t \in \{0, 1\}$):**
   - $a_t = 1$: Pertahankan langkah penalaran ke-$t$.
   - $a_t = 0$: Pangkas / singkirkan langkah penalaran ke-$t$.
3. **Reward ($r_t$):**
   $$r_t = R_{\text{task}} + \lambda_{\text{eff}} R_{\text{eff}}(t)$$
   - $R_{\text{task}}$: Skor akurasi jawaban akhir downstream task (1 untuk benar, 0 untuk salah).
   - $R_{\text{eff}}(t)$: Reward efisiensi:
     $$R_{\text{eff}}(t) = 1 - \frac{|T_{\text{retained}}|}{|T_{\text{total}}|}$$
     di mana $|T_{\text{retained}}|$ adalah jumlah token penalaran yang tersisa.

## 3. Iterative CoT Limits (ThinkPrune Mechanism)
Selama fase pelatihan RL, diterapkan batas token CoT bertahap (*iterative token limit* $L_{\text{limit}}$). Jika penalaran tidak menghasilkan jawaban final sebelum $L_{\text{limit}}$ terlampaui, penalaran dianggap gagal ($R_{\text{task}} = 0$). Hal ini memaksa model penalaran untuk:
1. Menghilangkan pola introspeksi berulang ("Wait, let me rethink...").
2. Memadatkan rute logika tanpa mengurangi presisi kesimpulan.
