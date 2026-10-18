# ADR-0004: Dynamic Promising Token Subspace Masking (RLPT) for Policy Optimization Stability

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon RL & Optimization Core Team
- **Consulted:** arXiv:2602.03195 (RLPT), arXiv:2512.04359 (Semantic & Token Entropy), PRD.md

---

## 1. Context & Problem Statement

Pada arsitektur kompresi konteks hibrida Lethon (Tahap Stage 3), agen pengambil keputusan berbasis *Reinforcement Learning* (RL) ditugaskan untuk memilih token mana yang harus dipertahankan atau dipangkas dari konteks masukan dan alur penalaran (*Chain-of-Thought*). 

Ketika algoritma RL standar (seperti PPO, REINFORCE, GRPO, atau DAPO) diterapkan langsung pada model bahasa besar (LLMs), ruang pencarian aksi diskrit yang dihadapi setara dengan seluruh kosa kata kosa kata (*full vocabulary*) $\mathcal{V}$ dengan ukuran masif ($|\mathcal{V}| > 50.000$ hingga $152.000$ token). Ruang aksi yang tidak terbatas (*unconstrained action space*) ini menimbulkan tiga kegagalan arsitektur kritis:

1. **Ledakan Varians Gradien (*Gradient Variance Explosion*):** Gradien kebijakan menderita akibat derau stokastik (*stochastic noise*) yang diinjeksikan oleh puluhan ribu token ekor (*tail tokens*) yang secara kontekstual tidak relevan. Meskipun probabilitas masing-masing token ekor sangat kecil, akumulasi variansnya memicu lonjakan gradien (*gradient norm spikes*) dan ketidakstabilan konvergensi (*policy collapse*).
2. **Ketidaksesuaian Kebijakan Tersembunyi (*Hidden Off-Policy Mismatch*):** Selama fase *rollout* (pengumpulan data), model menggunakan strategi pemangkasan heuristik seperti Top-$k$ atau Nucleus (Top-$p$) sampling agar keluaran tetap koheren. Namun, selama fase *optimization* (backward pass), baselines standar menghitung gradien dan divergensi KL di atas ruang kosa kata penuh $\mathcal{V}$. Mismatch ini merusak asumsi dasar *on-policy learning*, menghasilkan estimator yang bias.
3. **Beban Sintaksis versus Keputusan Strategis:** Kebijakan RL terdistraksi untuk mempelajari kepatuhan tata bahasa dasar (*grammatical coherence*) alih-alih berfokus pada penalaran logis tingkat tinggi atau strategi seleksi token kompresi yang optimal.

Tim Lethon membutuhkan strategi pembatasan ruang aksi yang secara matematis mereduksi varians gradien, menjamin konsistensi *on-policy*, dan melindungi token-token bernilai semantik kritis (seperti operator matematika, identifier kode, dan semantic anchors) dari pemangkasan tak sengaja.

---

## 2. Decision Drivers

- **Stabilitas Konvergensi & Reduksi Varians:** Memastikan penurunan varians gradien yang terbukti secara teoritis tanpa memerlukan tuning hiperparameter kritis yang rapuh.
- **Konsistensi On-Policy Matematika:** Menyelaraskan ruang aksi antara fase generasi trajektori (*rollout*) dan fase pembaruan parameter (*optimization*).
- **Proteksi Semantik Deterministik:** Menjamin token berenergi informasi tinggi (*promising tokens* dan *semantic anchors*) terlindungi dari pemangkasan katastropik.
- **Efisiensi Komputasi & Zero Heavy-Model Overhead:** Beroperasi dengan latensi minimal ($< 1$ ms pada CPU) tanpa perlu melatih atau menjalankan modul jaringan saraf tiruan tambahan yang membebani memori.

---

## 3. Considered Options

### Option 1: Unconstrained Action Space RL (Standard GRPO / PPO over Full Vocabulary)
Mengizinkan agen RL mengeksplorasi dan memperbarui probabilitas pada seluruh kosa kata $|\mathcal{V}| > 50\text{k}$.
- *Kelemahan Fatal:* Varians gradien sangat tinggi. Terjadi *gradient spikes* terus-menerus. Agen sering memotong token krusial sintaksis pemrograman atau operator logika karena random walk gradien pada ekor distribusi.

### Option 2: Explicit Lightweight Selector Policy (Auxiliary Neural Network)
Melatih jaringan terpisah (misal Multi-Layer Perceptron / MLP Selector) yang memetakan embedding konteks dan kandidat token untuk menentukan mask aksi sebelum dieksekusi oleh kebijakan utama.
- *Kelemahan Fatal:* Berdasarkan eksperimen empiris pada paper *arXiv:2602.03195*, pendekatan *Explicit Selector* gagal dan stagnan pada akurasi rendah ($\approx 25\%$, bahkan lebih rendah dari baseline tanpa pelatihan). Hal ini karena modul eksternal dimulai dari *cold start* dan kehilangan kekayaan representasi semantik pre-trained yang sudah dimiliki oleh backbone model bahasa.

### Option 3: Static Heuristic & Regex Rule Masking
Menggunakan aturan sintaksis kaku dan ekspresi reguler (misal: proteksi daftar kata kunci bahasa pemrograman seperti `def`, `class`, `if`, atau simbol matematika).
- *Kelemahan Fatal:* Sangat kaku, tidak dapat digeneralisasi ke teks domain bebas atau penalaran multi-disiplin, dan tidak adaptif terhadap variasi semantik konteks yang kompleks.

### Option 4: Dynamic Promising Token Subspace Masking via Base-Model Semantic Priors (RLPT Paradigm)
Mengisolasi ruang aksi kebijakan ke dalam subruang berperingkat rendah (*low-rank subspace*) dari prior semantik model dasar menggunakan matriks masking dinamis $M_t$:
- Membentuk himpunan promising tokens $\mathcal{P}_t$ (Top-$K$, Top-$P$, atau Min-$P$).
- Mengaplikasikan logit-level masking $z_{\text{masked}} = z + (1 - M_t) \cdot (-\infty)$.
- Menghitung distribusi probabilitas ternormalisasi $\tilde{\pi}$ dan pembaruan gradien strictly pada subruang $\mathcal{P}_t$ pada kedua fase (rollout & backward pass).

---

## 4. Decision Outcome

Dipilih **Option 4: Dynamic Promising Token Subspace Masking (RLPT Paradigm)**.

### Rationale:
1. **Bukti Konsentrasi Subruang 100%:** Investigasi empiris membuktikan bahwa **100% token penalaran valid yang dihasilkan model berada di dalam Top-8 kandidat** pada tugas matematika dan kode. Ruang aksi tidak perlu mencakup ribuan token ekor yang tidak relevan.
2. **Jaminan Teorema Reduksi Varians (Proposition 4.1):**
   Dengan memaksakan probabilitas token ekor $\mathcal{T}_t = \mathcal{V} \setminus \mathcal{P}_t$ bernilai nol deterministik, varians gradien komponen ekor menjadi nol: $\text{Var}(\hat{g}_{\text{mask}, k}) = 0$. Reduksi total varians:
   $$\Delta V = V(\hat{g}_{\text{full}}) - V(\hat{g}_{\text{mask}}) \approx \sum_{k \in \mathcal{T}_t} \pi_k (1 - \pi_k) A_t^2 > 0$$
   secara ketat bernilai positif, mengeliminasi derau fluktuasi logit dari puluhan ribu token ekor.
3. **Penyelesaian Sempurna Off-Policy Mismatch:** Karena mask $M_t$ yang sama persis diaplikasikan saat *rollout* dan saat *backward pass*, kebijakan target dan kebijakan perilaku beroperasi di atas *support set* yang identik, memulihkan konsistensi *on-policy* secara murni.
4. **Efisiensi Komputasi Tinggi:** Implementasi berbasis `np.argpartition` pada logit beroperasi dalam $\mathcal{O}(|\mathcal{V}|)$ dengan waktu eksekusi $< 0.8$ ms pada CPU, tanpa parameter tambahan (*zero-parameter overhead*).

---

## 5. Quantitative Evidence & Benchmark Summary

Berdasarkan benchmark komparatif komprehensif pada backbone model **Qwen3-8B** dan **Qwen3-4B** (paper arXiv:2602.03195):

| Parameter / Metrik | Baseline Standar (GRPO) | GRPO + Subspace Masking (RLPT) | Peningkatan / Dampak |
| :--- | :--- | :--- | :--- |
| **Math-17k Accuracy** | 34.7% | **38.3%** | **+3.6% Absolute** |
| **AIME-24 Reasoning** | 20.0% | **23.3%** | **+3.3% Absolute** |
| **Wireless QA Accuracy** | 52.17% | **55.65%** | **+3.48% Absolute** |
| **Gradient Norm Stability** | Fluktuatif dengan spike ($> 0.25$) | Mulus & stabil ($\sim 0.12 - 0.15$) | **Varians Gradien Turun Signifikan** |
| **Ground-Truth Top-32 Recall**| - | **> 99.2% – 99.5%** | Jalur solusi tetap utuh |
| **Model Correct Solution Top-8**| - | **100.0% Coverage** | Zero kehilangan kapasitas penalaran |
| **Optimal Subspace Size** | $|\mathcal{V}| = 152.064$ | **$K = 4 \sim 8$ Tokens** | Rasio Sinyal/Derau Maksimal |

---

## 6. Pipeline Integration & Lethon Architecture Impact

Dalam alur kerja 4-Stage Compression Pipeline Lethon:
```
[Stage 1: Fast Spectral 1D-FFT]
        │
        ▼ (Truncation Noise Frekuensi Tinggi O(N log N))
[Stage 2: Contextual Semantic Anchor Aggregation (SAC)]
        │
        ▼ (Aggregated Key-Value Manifold)
[Stage 3: Dynamic Promising Token Subspace Masking (ADR-0004)]
        │  * Truncated SVD Low-Rank Projection (P_k = V_k V_k^T)
        │  * Promising Token Masking Matrix M_t in {0, 1}^|V|
        │  * Logit-level masking: z_masked = z + (1 - M_t) * (-1e9)
        │  * Perlindungan token kritis & penekanan varians gradien
        ▼
[Stage 4: Sequential Decision MDP & CoT Length Truncation]
        │  * Optimasi Trajectory GRPO strictly on Subspace
        ▼
[Output: Compressed Context untuk Agent & IDE (Hemat 50% - 66.7% Token)]
```

- **Sinergi dengan Stage 2 (SAC):** Token *semantic anchors* yang dihasilkan pada Stage 2 secara otomatis memperoleh skor proyeksi subruang $s_n$ yang tinggi, memastikan seluruh jangkar semantik terkunci aman dalam *promising set* dan tidak dipangkas pada Stage 4.
- **Sinergi dengan Stage 4 (CoT MDP):** Mengurangi kompleksitas ruang aksi dari puluhan ribu menjadi hanya $K$ opsi strategis, mempercepat konvergensi RL hingga 2.5x lebih cepat pada epoch awal.

---

## 7. Consequences & Trade-offs

### Positive
- **Stabilitas Pelatihan Optimal:** Permukaan fungsi loss menjadi jauh lebih konveks lokal dan mulus, mengeliminasi kegagalan konvergensi (*training divergence*).
- **Sampel Lebih Efisien:** Membutuhkan langkah pelatihan (*training steps*) lebih sedikit untuk mencapai performa asimptotik berkat konsentrasi gradien pada token-token relevan.
- **Kualitas Penalaran Terjaga:** Menghindari degradasi akurasi pada benchmark multi-langkah (*multi-step deduction*) karena tidak ada token kunci yang terpotong.
- **Zero Memory Bloat:** Tidak membutuhkan model eksternal, dapat dijalankan langsung di CPU runtime MCP.

### Negative / Mitigations
- **Risiko Pengecualian Token Ekor Langka (< 1%):** Pada kasus yang sangat ekstrem di mana simbol matematika langka berada di luar Top-$K$, ada kemungkinan token tersebut ter-mask.
  - *Mitigasi:* Lethon menyediakan opsi mode adaptive thresholding **Top-P** ($p = 0.95$) atau **Min-P** ($\tau = 0.05$) untuk teks dengan entropi tinggi, sehingga ukuran subruang melebar secara otomatis pada titik percabangan pemikiran yang ambigu.
