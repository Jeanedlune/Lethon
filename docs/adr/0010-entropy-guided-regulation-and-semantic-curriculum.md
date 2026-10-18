# ADR-0010: Entropy-Guided Non-Uniform KL Regulation and Semantic Curriculum Learning (SENT) for LLM Reasoning Stability

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture Guild & Algorithmic Core Team
- **Consulted Literature:**
  - `arXiv:2512.04359v3`: *Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning (SENT)* (Cao et al., Nanjing University & China Mobile NineVerse, 2026)
  - `arXiv:2504.01296`: *THINKPRUNE: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning* (2025)
  - `arXiv:2602.03195v2`: *Reinforcement Learning with Promising Tokens (RLPT)* (2026)
  - `arXiv:2603.13394v1`: *Language-Guided Token Compression with Reinforcement Learning in LVLMs (TPRL)* (2026)
  - `PRD.md`: *Lethon System Requirements & 4-Stage Architecture (Stage 4 CoT & Trajectory Optimization)*
  - `docs/dec/dec-06-entropy-collapse-curriculum-sent.md`: *Technical & Mathematical Specifications of SENT*

---

## 1. Context & Problem Statement

Pada arsitektur engine kompresi konteks Lethon ([PRD.md](../../PRD.md)), pemangkasan alur pikir penalaran panjang (*Chain-of-Thought / CoT*) pada **Stage 4** dilatih menggunakan metode *Reinforcement Learning with Verifiable Rewards* (RLVR) seperti GRPO ([DEC-04](file:///workspaces/Lethon/docs/dec/dec-04-cot-iterative-pruning.md)). Tujuannya adalah memangkas 50% hingga 66.7% token redundan tanpa menjatuhkan akurasi penalaran downstream.

Namun, pascapelatihan RLVR murni berbasis akurasi (*binary outcome rewards*) menghadapi patologi fatal: **Entropy Collapse**.
1. **Penyempitan Ruang Eksplorasi Cepat:**  
   Pembaruan *Policy Gradient* standar menghasilkan penurunan entropi secara monoton karena kovarians positif kuat antara log-probabilitas aksi terpilih dan reward positif. Entropi kebijakan meluncur mendekati nol ($H(\pi_\theta) \to 0$).
2. **Kerapuhan Saat Kompresi Panjang CoT:**  
   Ketika dikenakan *iterative token limit clipping* (misal pemotongan langkah $4k \to 3k \to 2k$), kebijakan yang mengalami *entropy collapse* tidak memiliki diversitas representasi untuk merumuskan ulang langkah pembuktian secara ringkas. Model mengalami *reasoning collapse* (membuang langkah verifikasi esensial, memicu penurunan akurasi drastis).
3. **Kegagalan Regulasi Entropi Konvensional:**  
   - Penalti entropi seragam (*uniform entropy bonus* $+\lambda H$) menyebabkan *entropy explosion*—model berhalusinasi dan kehilangan logika deduktif matematika.
   - Masking token entropi rendah secara heuristik (*80% masking*) membuang sinyal optimasi pada token stabil dan rentan terhadap ketidakstabilan numerik.
   - Memberikan bonus pada token berentropi tinggi justru memberi insentif pada kata-kata fungsi bahasa (*filler tokens* seperti "the", "of", "to") yang tidak memiliki nilai semantik penalaran.

Oleh karena itu, tim arsitektur perlu menetapkan keputusan strategis: **Bagaimana merancang mekanisme regulasi entropi dan kurikulum data yang menjamin stabilitas eksplorasi penalaran LLM selama optimasi RL kompresi CoT tanpa merusak koherensi logis?**

---

## 2. Decision Drivers

- **Pencegahan Entropy Collapse & Explosion:** Menjaga entropi kebijakan pada ambang sehat ($H(\pi_\theta) \ge H_{\min} > 0$) sepanjang pelatihan tanpa memicu ledakan acak.
- **Proteksi Titik Keputusan Kritis (*Non-Uniform Token Treatment*):** Mengidentifikasi dan memproteksi token deterministik berisiko tinggi (*high-covariance low-entropy tokens*) tanpa mengekang token kreatif/eksploratif.
- **Perkembangan Kesulitan Bertahap (*Scaffolded Learning*):** Menghindari lonjakan kesulitan data di awal yang memicu divergensi kebijakan atau pembaruan gradien destruktif.
- **Preservasi Refleksi & Verifikasi Mandiri (*Self-Correction*):** Menjaga kapasitas model untuk memeriksa ulang langkah antara (*backtracking*) meskipun panjang CoT dikompresi.
- **Kompatibilitas Penuh dengan Pipeline Hybrid:** Mampu beroperasi selaras dengan *Subspace Masking* (Stage 3 - RLPT) dan *Sequential MDP* (Stage 4 - ThinkPrune/TPRL).

---

## 3. Considered Options

### Option 1: Standard GRPO with Uniform Constant KL Penalty
- Menggunakan penalti KL standar $\beta \mathcal{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ seragam di seluruh posisi token dengan koefisien kecil tetap ($\beta = 0.001$).
- *Karakteristik:* Sangat sederhana, namun gagal mencegah entropy collapse. Entropi tetap meluncur bebas menuju nol pada token berkovarians tinggi, mematikan kemampuan eksplorasi model di benchmark sulit (AIME24 hanya mencapai 43.3% Pass@32).

### Option 2: Uniform Entropy Bonus / Entropy Maximization Objective
- Menambahkan suku penalti negatif $-\lambda \sum_t H_t$ pada loss untuk memaksa entropi tetap tinggi.
- *Karakteristik:* Memicu *entropy explosion*. Model kehilangan kepastian pada konsep dasar matematika dan kode, logika penalaran terurai menjadi keluaran acak yang tidak valid.

### Option 3: Low-Entropy Token Masking (80% Pruning Heuristic)
- Mengabaikan gradien policy pada $80\%$ token berentropi terendah pada setiap batch.
- *Karakteristik:* Mencegah collapse parsial, tetapi memperlakukan seluruh $80\%$ token secara homogen. Padahal sebagian besar token tersebut adalah kata sambung alami yang tidak berdampak. Gradien yang terputus-putus juga menyebabkan instabilitas konvergensi.

### Option 4: Dual-Level SENT (Semantic Entropy Curriculum + Non-Uniform KL Covariance Regulation - Selected)
- **Tingkat Data (Global):** Mengelompokkan solusi ke dalam klaster ekuivalensi semantik, mengukur *Semantic Entropy* ($H_{\text{SE}}$), dan mengatur jadwal pelatihan dua tahap ($N=2$): dimulai dari data berentropi semantik rendah (stabil) menuju data berentropi semantik tinggi (eksploratif).
- **Tingkat Algoritmik (Lokal Token-Wise):** Menerapkan regulasi KL non-seragam adaptif $\beta_{\text{con}}$:
  - $\beta_{\text{con}} = 0.0$ untuk token entropi tinggi ($H_t \ge \tau_H$, eksplorasi penuh).
  - $\beta_{\text{con}} = \beta_{\text{low}} = 0.5$ untuk token entropi rendah bervariansi normal ($H_t < \tau_H, \text{Cov}_t \le \tau_{\text{cov}}$).
  - $\beta_{\text{con}} = \beta_{\text{high}} = 2.0$ khusus untuk token entropi rendah bervariansi tinggi ($\text{Cov}_t > \tau_{\text{cov}}$), pemicu utama kejatuhan entropi.

---

## 4. Evaluasi Komparatif & Bukti Empiris

| Dimensi Evaluasi | Option 1: Uniform GRPO | Option 2: Entropy Bonus | Option 3: Token Masking | Option 4: SENT (Selected) |
| :--- | :--- | :--- | :--- | :--- |
| **Dinamika Entropi** | Collapse cepat ($H \to 0$) | Explosion ($H \to \infty$, liar) | Degradasi bertahap | **Terkendali & Sehat ($H \ge H_{\min}$)** |
| **AIME24 Pass@32** | 43.33% | 70.00% (tidak stabil) | 73.33% | **76.67% (Puncak Akurasi)** |
| **AIME24 Avg@16** | 18.96% | 20.62% | 24.79% | **25.21% (Konsistensi Tertinggi)** |
| **AIME25 Pass@32** | 30.00% | 43.33% | 43.33% | **46.67% (+16.67 vs GRPO)** |
| **MATH500 Avg@16** | 77.26% | 79.98% | 80.76% | **81.74% (+4.48 vs GRPO)** |
| **OlympiadBench Avg@16** | 36.59% | 40.11% | 42.65% | **42.99% (+6.40 vs GRPO)** |
| **Rata-Rata 6 Benchmark (Pass@K)** | 57.61% | 65.29% | 65.31% | **68.57% (Melampaui semua baseline)** |
| **Rata-Rata 6 Benchmark (Avg@K)** | 35.13% | 39.56% | 40.52% | **44.01% (Margin keunggulan +8.88)** |
| **Perilaku Penalaran** | Rantai pendek, tanpa cek ulang | Liar, banyak halusinasi | Rantai moderat | **Rantai kaya refleksi & verifikasi mandiri** |

### Bukti Empiris Kunci dari arXiv:2512.04359v3:
1. **Teorema Dekomposisi Entropi (Teorema 2 Paper):**  
   Dinamika perubahan entropi terbukti terbagi menjadi dua komponen berlawanan: $\Delta H \approx - \text{Term 1 (Decay)} + \text{Term 2 (KL-Induced Preservation)}$. Dengan memberikan $\beta_{\text{high}} = 2.0$ pada token berkovarians tinggi, Term 2 mengimbangi Term 1 secara presisi, menghasilkan kestabilan asimtotik tanpa lonjakan liar.
2. **Superioritas Kurikulum Dua Tahap ($N=2$):**  
   Studi ablasi membuktikan pembagian dua tahap menghasilkan performa rata-rata tertinggi di semua benchmark (MATH500 81.74 vs 80.75 pada 3 tahap; AIME24 25.21 vs 23.54). Tahap 1 mengunci pemahaman deterministik, Tahap 2 menumbuhkan eksplorasi tingkat lanjut.
3. **Generalisasi Lintas Domain (LiveCodeBench):**  
   Ketika dievaluasi pada penalaran kode di luar distribusi latih (OOD), model terlatih SENT mengungguli GRPO sebesar **+3.87 poin pada Pass@16** dan **+2.05 poin pada Avg@8**, membuktikan bahwa kebiasaan eksplorasi dan verifikasi mandiri yang dipelajari bersifat universal.

---

## 5. Decision Outcome & Rationale

Tim arsitektur memutuskan untuk mengadopsi **Option 4: Dual-Level SENT (Semantic Entropy Curriculum + Non-Uniform KL Covariance Regulation)** sebagai standar regulasi pelatihan RL pada **Lethon Stage 4**.

### Rationale Utama:
1. **Menghilangkan Akar Penyebab Entropy Collapse:**  
   Bukan dengan melarang determinisme, melainkan dengan memproteksi token-token berkovarians tinggi yang menghubungkan keyakinan probabilitas dengan keunggulan reward, sehingga model tidak menutup pintu terhadap penalaran alternatif.
2. **Menjaga Integritas Kompresi CoT (ThinkPrune Synergy):**  
   Saat panjang CoT dipangkas hingga 50%–65.5%, kestabilan entropi memastikan bahwa pemotongan panjang tidak membunuh langkah verifikasi kritis, melainkan membuang redundansi linguistik yang sesungguhnya.
3. **Konvergensi Halus Tanpa Parameter Eksternal:**  
   Seluruh penghitungan entropi token dan kovarians berjalan langsung di atas logit dan advantage yang sudah tersedia pada batch pelatihan PPO/GRPO, tanpa membutuhkan reward model eksternal tambahan.

---

## 6. Implementation Notes & Architecture Links

- **Spesifikasi Konsep & Matematika Lengkap:** Didefinisikan di [`docs/dec/dec-06-entropy-collapse-curriculum-sent.md`](file:///workspaces/Lethon/docs/dec/dec-06-entropy-collapse-curriculum-sent.md).
- **Modul Python Core:** Diimplementasikan di [`lethon/core/rl/entropy_regulation.py`](file:///workspaces/Lethon/lethon/core/rl/entropy_regulation.py).
- **Pengujian Unit & Validasi Numerik:** Berhasil lolos pengujian pada [`tests/unit/test_entropy_regulation.py`](file:///workspaces/Lethon/tests/unit/test_entropy_regulation.py).
