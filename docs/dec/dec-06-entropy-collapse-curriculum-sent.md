# DEC-06: Entropy Collapse Prevention & Semantic Curriculum Learning (SENT)

## 1. Executive Summary & Context

Modul **Entropy Collapse Prevention & Semantic Curriculum Learning (SENT)** merupakan pilar stabilitas optimasi pada tahap akhir (*Stage 4*) pipeline kompresi konteks Lethon ([PRD.md](../../PRD.md)), beroperasi berdampingan dengan pemadatan *Chain-of-Thought* (*ThinkPrune*, [DEC-04](dec-04-cot-iterative-pruning.md)) dan kompresi token sekuensial (*TPRL*, [DEC-05](dec-05-tprl-token-compression-mdp.md)). Modul ini dirancang untuk memecahkan fenomena paling destruktif pada pascapelatihan model penalaran panjang (*reasoning LLMs*) berbasis *Reinforcement Learning with Verifiable Rewards* (RLVR): **Entropy Collapse**.

Pada pelatihan RLVR berbasis algoritma seperti PPO atau GRPO ([DeepSeek R1](https://arxiv.org/abs/2501.12948)), model dihargai secara ketat berdasarkan ketepatan solusi akhir (*binary accuracy reward*). Pola optimasi yang murni berorientasi pada akurasi ini memicu penurunan entropi kebijakan secara drastis (*precipitous entropy collapse*). Akibatnya:
1. **Hilangnya Kapasitas Eksplorasi:** Distribusi probabilitas token menyusut tajam menjadi deterministik sempit. Model kehilangan kemampuan untuk menjelajahi jalur penalaran alternatif (*alternative reasoning trajectories*), terjebak dalam optimum lokal, dan gagal melakukan refleksi diri (*self-correction / backtracking*).
2. **Kerapuhan Pemangkasan Token CoT:** Saat Lethon menerapkan *iterative length clipping* atau pemangkasan token sekuensial, kebijakan yang mengalami *entropy collapse* akan merespons pemotongan panjang langkah dengan kepanikan struktural—memotong langkah kritis penalaran dan menjatuhkan akurasi downstream secara tajam.
3. **Kegagalan Solusi Reguler:** Upaya konvensional dengan memaksakan penalti entropi seragam (*all-token entropy bonus*) justru memicu *entropy explosion* (kebijakan melantur dan kehilangan arah penalaran logis), sementara pemangkasan/penutupan token entropi rendah secara heuristik (*token masking 80%*) mengabaikan struktur internal token dan mengorbankan stabilitas numerik.

Berdasarkan riset terobosan **SENT (Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning, *arXiv:2512.04359v3*)**, Lethon menerapkan strategi penanganan entropi dual-level yang sinergis:
- **Tingkat Data (Global):** *Semantic Entropy-Guided Curriculum Learning* yang mengelompokkan data ke dalam klaster ekuivalensi semantik dan mengurutkan kesulitan pelatihan dari entropi semantik rendah ke tinggi ($D_{\text{sorted}} = \text{sort}(D, \text{key}=H_{\text{SE}})$) dalam jadwal multi-tahap ($N=2$).
- **Tingkat Algoritmik (Lokal Token-Wise):** *Non-Uniform KL Divergence Regularization* yang menerapkan penalti KL secara bertingkat: membebaskan token berentropi tinggi ($\beta_{\text{con}} = 0$), memberikan regulasi moderat pada token berentropi rendah ($\beta_{\text{low}} = 0.5$), dan menerapkan regulasi ketat ($\beta_{\text{high}} = 2.0$) khusus pada porsi token berentropi rendah yang memiliki kovarians tinggi terhadap sinyal keuntungan (*high-covariance low-entropy tokens*).

Hasil empiris membuktikan bahwa formulasi ini menjaga tingkat entropi sehat sepanjang pelatihan, menaikkan performa penalaran hingga **$68.57\%$ Pass@K** dan **$44.01\%$ Avg@K** (mengungguli seluruh baseline PPO/GRPO berorientasi entropi pada 6 benchmark penalaran: AIME24, AIME25, AMC23, MATH500, OlympiadBench, dan Minerva), serta menghasilkan rantai penalaran mandiri yang lebih kaya verifikasi reflektif.

---

## 2. Landasan Teori & Analisis Paper Acuan (arXiv:2512.04359v3)

### 2.1 Patologi Entropy Collapse pada RLVR

Dalam pembaruan *Policy Gradient* standar (seperti PPO/GRPO), gradien kebijakan dinyatakan sebagai:
$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{(q, a) \sim \mathcal{D}, o \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=1}^{|o|} \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t}) A_t \right]$$

Berdasarkan analisis teoritis perubahan logit pada softmax policies (Lemma 1 & Teorema 1 paper):
$$\Delta H = H(\pi^{k+1}_\theta) - H(\pi^k_\theta) \approx - \eta \mathbb{E}_{s_t} \left[ \text{Cov}_t \left( \log \pi^k_\theta(o_t \mid s_t), \pi^k_\theta(o_t \mid s_t) A_t \right) \right]$$

Karena token dengan probabilitas tinggi pada respons yang berhasil memiliki advantage $A_t > 0$, kovarians $\text{Cov}_t$ bernilai positif besar secara konsisten. Akibatnya, $\Delta H < 0$ pada hampir setiap langkah optimasi, menyebabkan entropi kebijakan meluncur turun tanpa batas (*monotonic entropy decay*) hingga mencapai nol ($H(\pi_\theta) \to 0$).

### 2.2 Dilema Solusi Sebelumnya

| Pendekatan | Mekanisme | Kelemahan Fatal |
| :--- | :--- | :--- |
| **Vanilla GRPO** | Tanpa kontrol entropi eksplisit | *Entropy collapse* cepat; eksplorasi mati; rentan terjebak di optimum lokal. |
| **GRPO + Entropy Bonus** (`w/ En`) | Menambahkan $+\lambda \sum H_t$ pada loss | *Entropy explosion*; kebijakan terlalu acak, merusak kemampuan inferensi terstruktur matematika/kode. |
| **Entropy Advantage Shaping** (`w/ Adv`) | $A_t + \psi(H_t)$ clipped shaping | Menghambat token bernilai deterministik valid; tidak menyelaraskan kurikulum data. |
| **Low-Entropy Masking** (`w/ Mask`) | Mengabaikan token berentropi rendah ($80\%$) dari gradien | Membuang sinyal pembelajaran pada token stabil; instabilitas gradien pada transisi konteks. |
| **High-Entropy Reward** (`w/ High En`) | Memberi reward jika $H_t \ge \tau$ | Menghargai kata-kata fungsi linguistik sepele (*filler tokens* seperti "the", "of", "to") yang kebetulan memiliki entropi tinggi. |
| **SENT (Lethon Stage 4)** | **Curriculum Semantik + Non-Uniform KL Covariance Regulation** | **Mempertahankan stabilitas entropi ($H(\pi_\theta) \ge H_{\min} > 0$), mencegah ledakan dan keruntuhan, mengisolasi titik keputusan kritis.** |

---

## 3. Formulasi Matematis Eksak SENT

### 3.1 Data-Level: Semantic Entropy (SE) & Semantic Clustering

Berbeda dengan entropi token lokal yang mengukur keacakan distribusi kosa kata instan, **Semantic Entropy** mengukur diversitas solusi yang bermakna ekuivalen untuk suatu masalah $q$.

1. **Sampling Respons Diversifikasi:**  
   Untuk setiap kueri $q \in \mathcal{D}$, model melakukan rollout sebanyak $M$ respons dengan *nucleus sampling* suhu penuh ($T = 1.0$):
   $$\{o_1, o_2, \dots, o_M\} \sim \pi_\theta(\cdot \mid q), \quad \{P(o_1 \mid q), \dots, P(o_M \mid q)\}$$

2. **Partisi Klaster Ekuivalensi Semantik:**  
   Sekuens respons dipartisi ke dalam $K$ kelas ekuivalensi semantik $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$. Untuk tugas penalaran formal (matematika, logika kode), ekuivalensi ditentukan melalui ekuivalensi jawaban akhir kanonikal:
   $$o_i \sim o_j \iff \text{CanonicalAnswer}(o_i) = \text{CanonicalAnswer}(o_j)$$

3. **Probabilitas Klaster & Normalisasi:**  
   Probabilitas untuk masing-masing kelas semantik $C_k$ diakumulasikan dan dinormalisasi:
   $$P(C_k \mid q) = \sum_{o \in C_k} P(o \mid q)$$
   $$\hat{P}(C_k \mid q) = \frac{P(C_k \mid q)}{\sum_{j=1}^K P(C_j \mid q)}$$

4. **Kalkulasi Semantic Entropy ($H_{\text{SE}}$):**  
   $$H_{\text{SE}}(q) = - \sum_{k=1}^K \hat{P}(C_k \mid q) \log \hat{P}(C_k \mid q)$$
   - Nilai $H_{\text{SE}}(q)$ rendah: Model konsisten menghasilkan jawaban semantik yang sama (masalah tergolong mudah/stabil bagi model).
   - Nilai $H_{\text{SE}}(q)$ tinggi: Model menghasilkan jawaban semantik yang bercabang/kontradiktif (masalah menantang dan membutuhkan kapasitas eksplorasi tinggi).

### 3.2 Semantic Entropy Curriculum Scheduler

Pelatihan diatur secara bertahap dari masalah berentropi semantik rendah ke tinggi guna menghindari lonjakan gradien agresif di awal pelatihan:
1. **Pengurutan Dataset Global:**  
   $$\mathcal{D}_{\text{sorted}} = \text{sort}\left(\mathcal{D}, \text{key} = H_{\text{SE}}\right)$$
2. **Partisi Multi-Tahap:**  
   Berdasarkan studi ablasi paper acuan, **jadwal dua tahap ($N = 2$)** memberikan keseimbangan Pareto optimal antara adaptasi kesulitan bertahap dan efisiensi waktu konvergensi:
   - **Tahap 1 ($\mathcal{D}_1$):** $50\%$ data dengan $H_{\text{SE}}$ terendah. Melatih kestabilan penalaran dasar dan membentuk token-token deterministik yang valid.
   - **Tahap 2 ($\mathcal{D}_2$):** $50\%$ data dengan $H_{\text{SE}}$ tertinggi. Melatih kapasitas eksplorasi, refleksi diri, dan pemecahan masalah rumit tanpa meruntuhkan kebijakan dasar.

---

### 3.3 Algorithmic-Level: Non-Uniform KL Divergence Regularization

#### 3.3.1 Token-Level Entropy
Untuk setiap posisi token $o_t$ dalam sekuens keluaran $o$, entropi kebijakan saat ini atas seluruh ruang kosa kata $\mathcal{V}$ dihitung sebagai:
$$H_t(q, o_{<t}) = - \sum_{v \in \mathcal{V}} \pi_\theta(v \mid q, o_{<t}) \log \pi_\theta(v \mid q, o_{<t})$$

Subset token berentropi rendah yang rentan terhadap optimasi berlebih diidentifikasi melalui ambang persentil $\tau_H$ (default persentil $80\%$):
$$\mathcal{T}_{\text{low}} = \{o_t \mid H_t(q, o_{<t}) < \tau_H\}$$

#### 3.3.2 Token-Level Advantage Covariance
Tidak semua token berentropi rendah berkontribusi sama terhadap keruntuhan kebijakan. Token yang memiliki korelasi kuat antara tingkat kepercayaan diri (*confidence*) dan sinyal reward adalah pemicu utama kejatuhan entropi. Kovarians dihitung pada setiap batch rollout:
$$\text{Cov}_t = \left( \log \pi_\theta(o_t \mid q, o_{<t}) - \frac{1}{N} \sum_{j=1}^N \log \pi_\theta(o_j \mid q, o_{<j}) \right) \cdot \left( A_t - \frac{1}{N} \sum_{j=1}^N A_j \right)$$

Subset token berkovarians tinggi di dalam kelompok berentropi rendah diisolasi melalui ambang $\tau_{\text{cov}}$ (default $\tau_{\text{cov}} = 0.0002$):
$$\mathcal{T}_{\text{high-cov}} = \{o_t \in \mathcal{T}_{\text{low}} \mid \text{Cov}_t > \tau_{\text{cov}}\}$$

#### 3.3.3 Hierarki Koefisien KL Non-Uniform ($\beta_{\text{con}}$)
Untuk memberikan rem proteksi proporsional tanpa menghambat eksplorasi pada token kreatif:
$$\beta_{\text{con}}(o_t) = \begin{cases}
\beta_{\text{high}} = 2.0, & \text{jika } o_t \in \mathcal{T}_{\text{high-cov}} \subseteq \mathcal{T}_{\text{low}} \quad \text{(Kritis: Proteksi Maksimal)} \\
\beta_{\text{low}} = 0.5, & \text{jika } o_t \in \mathcal{T}_{\text{low}} \setminus \mathcal{T}_{\text{high-cov}} \quad \text{(Moderat: Mencegah Over-optimasi)} \\
\beta_{\text{base}} = 0.0, & \text{jika } o_t \notin \mathcal{T}_{\text{low}} \quad \text{(Bebas: Eksplorasi Penuh)}
\end{cases}$$

---

### 3.4 Fungsi Objektif SENT & Pembuktian Teoretis Preservasi Entropi

Fungsi objektif optimasi SENT menggabungkan data kurikulum $\mathcal{D}_n$ dan penalti KL non-uniform:
$$\mathcal{J}_{\text{SENT}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}_n, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( r_t^i(\theta) \hat{A}_t^i, \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t^i \right) - \beta_{\text{con}}(o_t) \mathcal{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right) \right]$$

#### Teorema 2: Preservasi Entropi Terkendali
Melalui proyeksi gradien ke ruang logit dan ekspansi Taylor orde pertama:
$$\theta^{k+1}_{s_t, o_t} - \theta^k_{s_t, o_t} = \eta \left( \pi^k_\theta(o_t \mid s_t) A_t - \beta_{\text{con}} \nabla_\theta \mathcal{D}_{\text{KL}}(\pi^k_\theta \parallel \pi_{\text{ref}}) \right)$$

Substitusi ke dalam Lemma 1 menghasilkan dekomposisi dinamika entropi dua suku:
$$H(\pi^{k+1}_\theta) - H(\pi^k_\theta) \approx \underbrace{- \eta \mathbb{E}_{s_t} \left[ \text{Cov}_t \left( \log \pi^k_\theta, \pi^k_\theta A_t \right) \right]}_{\text{Term 1: Vanilla Entropy Decay } (< 0)} + \underbrace{\eta \mathbb{E}_{s_t} \left[ \beta_{\text{con}} \text{Cov}_t \left( \log \pi^k_\theta, \nabla_\theta \mathcal{D}_{\text{KL}}(\pi^k_\theta \parallel \pi_{\text{ref}}) \right) \right]}_{\text{Term 2: KL-Induced Entropy Preservation } (> 0)}$$

**Interpretasi Mekanistis:**
- Pada vanilla policy gradient, **Term 1** mendominasi dan menarik entropi menuju nol secara monoton.
- Pada SENT, **Term 2** bertindak sebagai gaya pemulih (*restoring force*) yang menarik distribusi kembali ke prior referensi $\pi_{\text{ref}}$. Karena $\beta_{\text{high}} = 2.0$ difokuskan tepat pada token dengan kovarians tertinggi, Term 2 memberikan kompensasi positif terkuat tepat di mana Term 1 berusaha meruntuhkan entropi paling tajam.
- Hal ini menjamin eksistensi batas bawah entropi sehat:
  $$H(\pi_\theta) \ge H_{\min} > 0, \quad \forall k$$

---

## 4. Integrasi dengan Arsitektur Hybrid Lethon

Dalam arsitektur produk Lethon ([PRD.md](../../PRD.md)), modul SENT bekerja terintegrasi pada **Stage 4**:

```
[Input Reasoning Context / CoT Stream]
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 1: Spectral FFT Pass     │  --> Truncation noise frekuensi tinggi
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 2: Semantic Anchor (SAC) │  --> In-Context Bidirectional Aggregation
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 3: Subspace Masking (RLPT)│ --> Low-rank Promising Token Action Masking
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Stage 4: Sequential MDP & CoT Compression with SENT        │
  │                                                             │
  │  1. Semantic Entropy Curriculum (D_1 Easy -> D_2 Hard)      │
  │  2. Iterative CoT Pruning (ThinkPrune 4k -> 3k -> 2k)       │
  │  3. Non-Uniform KL Covariance Regulation (beta_con)         │
  │  ==> Policy Collapse Prevented during CoT Shortening        │
  └────────────────┬────────────────────────────────────────────┘
                   │
                   ▼
[Compact High-Reasoning Context] (Hemat 50-66.7% Token, PPL 1.90 vs 1.91)
```

1. **Proteksi Saat Pemangkasan Agresif CoT (*ThinkPrune*):**  
   Ketika limit token dipotong bertahap ($4k \to 3k \to 2k$), reward pemotongan panjang memberikan tekanan gradien besar. Tanpa SENT, model cenderung memangkas langkah verifikasi matematika kritis agar tidak terkena penalti panjang. Regulasi $\beta_{\text{high}}$ pada token kritis menjaga model tetap mempertahankan langkah pembuktian esensial.
2. **Harmonisasi dengan Ruang Aksi Subspace Masking (*Stage 3 - RLPT*):**  
   RLPT membatasi aksi pada *promising tokens* ($K=4 \sim 8$), sementara SENT mengatur penalti KL per-token pada posisi terpilih. Keduanya bekerja tanpa friksi: RLPT menekan varians ruang aksi, sedangkan SENT menjaga kestabilan entropi di sepanjang sekuens.

---

## 5. Bukti Empiris & Hasil Benchmark (arXiv:2512.04359v3)

### 5.1 Perbandingan Performa Multi-Benchmark (Model 1.5B)

Evaluasi pada model *DeepSeek-R1-Distill-Qwen-1.5B* di 6 benchmark penalaran utama:

| Benchmark | Metrik | Vanilla GRPO | w/ En | w/ Mask | w/ Clip | w/ Cov | SENT (Ours) | Keunggulan SENT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AIME24** | Pass@32 | 43.33 | 56.67 | 73.33 | 70.00 | 53.33 | **76.67** | **+33.34 vs GRPO, +3.34 vs Mask** |
| | Avg@16 | 18.96 | 16.46 | 24.79 | 22.50 | 21.67 | **25.21** | **+6.25 vs GRPO** |
| **AIME25** | Pass@32 | 30.00 | 43.33 | 43.33 | 36.67 | 40.00 | **46.67** | **+16.67 vs GRPO** |
| | Avg@16 | 19.17 | 15.63 | 17.92 | 18.54 | 17.08 | **21.46** | **+2.29 vs GRPO** |
| **AMC23** | Pass@32 | 92.50 | 95.00 | 95.00 | 90.00 | 97.50 | **97.50** | **+5.00 vs GRPO** |
| | Avg@16 | 57.03 | 60.47 | 68.28 | 67.03 | 65.94 | **70.16** | **+13.13 vs GRPO** |
| **MATH500** | Pass@32 | 93.20 | 95.00 | 94.40 | 93.80 | 94.80 | **95.20** | **+2.00 vs GRPO** |
| | Avg@16 | 77.26 | 73.47 | 80.76 | 80.24 | 80.31 | **81.74** | **+4.48 vs GRPO** |
| **OlympiadBench** | Pass@32 | 61.19 | 64.30 | 69.19 | 68.74 | 67.70 | **70.22** | **+9.03 vs GRPO** |
| | Avg@16 | 36.59 | 29.43 | 42.65 | 41.44 | 39.72 | **42.99** | **+6.40 vs GRPO** |
| **Minerva** | Pass@32 | 41.54 | 39.71 | 45.59 | 44.49 | 43.01 | **46.32** | **+4.78 vs GRPO** |
| | Avg@16 | 36.76 | 36.03 | 38.97 | 38.60 | 40.44 | **42.65** | **+5.89 vs GRPO** |
| **Rata-Rata (Pass@K)** | Pass@K | 57.61 | 61.59 | 65.31 | 63.16 | 62.95 | **68.57** | **Puncak Akurasi Global (+10.96)** |
| **Rata-Rata (Avg@K)** | Avg@K | 35.13 | 36.74 | 40.52 | 39.91 | 39.12 | **44.01** | **Konsistensi Tertinggi (+8.88)** |

### 5.2 Ablasi Desain Kurikulum ($N$ Tahapan)

| Konfigurasi Kurikulum | AIME24 Pass@16 | AIME24 Avg@16 | MATH500 Pass@16 | MATH500 Avg@16 | OlympiadBench Pass@16 | OlympiadBench Avg@16 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tanpa Kurikulum** | 63.33 | 25.00 | 93.80 | 81.60 | 66.22 | 42.72 |
| **Dua Tahap ($N=2$, Selected)** | **63.33** | **25.21** | **94.20** | **81.74** | **67.26** | **42.99** |
| **Tiga Tahap ($N=3$)** | 60.00 | 23.54 | 94.20 | 80.75 | 66.37 | 41.94 |

*Insight:* Dua tahap ($N=2$) memberikan perolehan tertinggi secara konsisten. Pembagian 3 tahap memperpanjang durasi transisi dan menurunkan efisiensi konvergensi model penalaran.

---

## 6. Spesifikasi Teknis Implementasi Python (`lethon.core.rl.entropy_regulation`)

Modul teknis diwujudkan dalam berkas [`lethon/core/rl/entropy_regulation.py`](file:///workspaces/Lethon/lethon/core/rl/entropy_regulation.py) dengan komponen:
1. `compute_semantic_entropy`: Menghitung $H_{\text{SE}}(q)$ dari respons rollout dan klasterisasi jawaban akhir kanonikal.
2. `SemanticEntropyCurriculum`: Mengelola pengurutan dataset pelatihan $\mathcal{D}_{\text{sorted}}$ dan pemisahan ke dalam $N=2$ tahapan kurikulum.
3. `compute_token_entropy`: Mengukur Shannon entropy per posisi token atas kosa kata $\mathcal{V}$.
4. `compute_token_advantage_covariance`: Menghitung $\text{Cov}_t = (\log \pi_t - \mu_{\log \pi})(A_t - \mu_A)$.
5. `TokenRegularizationSchedule`: Menghasilkan tensor koefisien non-uniform $\beta_{\text{con}} \in \{0.0, 0.5, 2.0\}$ berdasarkan ambang batas $\tau_H$ (persentil 80%) dan $\tau_{\text{cov}} = 0.0002$.
6. `compute_sent_policy_loss`: Mengkalkulasi fungsi loss komposit terpotong PPO/GRPO dengan penalti diferensial $\beta_{\text{con}} \mathcal{D}_{\text{KL}}$.
