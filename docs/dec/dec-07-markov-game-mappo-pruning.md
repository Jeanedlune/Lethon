# DEC-07: Markov Game Token Pruning & Multi-Agent PPO (MAPPO) Cross-Stage Optimization

## 1. Executive Summary & Context

Modul **Markov Game Token Pruning & Multi-Agent PPO (MAPPO) Cross-Stage Optimization** merupakan komponen optimasi koordinasi antar-layer (*Cross-Stage Optimization*) pada arsitektur hybrid **Lethon** ([PRD.md](../../PRD.md)). Modul ini beroperasi pada koordinasi lintas layer (*inter-layer transformer coordination*) untuk kompresi representasi token dan KV-Cache, menjembatani representasi yang telah difilter oleh domain spektral (*Stage 1*, [DEC-01](dec-01-spectral-filtering-math.md)), diagregasi oleh *Semantic Anchors* (*Stage 2*, [DEC-02](dec-02-semantic-anchor-aggregation.md)), dan dibatasi oleh *Subspace Masking* (*Stage 3*, [DEC-03](dec-03-rlpt-promising-token-masking.md)).

Dalam kompresi token tradisional berbasis heuristik (seperti *attention thresholding*, *cosine similarity*, atau *fixed-ratio pruning* seperti DynamicViT), keputusan pemangkasan token bersifat:
1. **Statik & Non-Adaptif:** Rasio pemangkasan per layer dipatok secara manual (*manually defined ratio*), sehingga gagal beradaptasi dengan variasi kompleksitas input konteks/kueri yang dinamis.
2. **Layer-Agnostik & Myopic:** Mengabaikan sifat sekuensial hierarkis dari representasi Transformer, di mana pemangkasan sebuah token pada layer $l$ secara permanen mengubah distribusi aktivasi dan atensi seluruh layer berikutnya ($l+1, \dots, L$).
3. **Single-Agent Bottleneck:** Jika diformulasikan sebagai satu agen RL global (*Single-Agent PPO*), ruang aksi mengalami ledakan kombinatorial sebesar $2^N$ untuk $N$ token, memicu varians gradien tinggi dan *credit assignment problem* yang parah antar-token.

Berdasarkan terobosan riset **RL4EViT (RL-based Token Pruning in ViTs: A Markov Game Approach, *arXiv:2503.23459*)**, modul ini memformulasikan pemangkasan token lintas layer sebagai **Markov Game** multi-tahap yang diselesaikan menggunakan **Multi-Agent Proximal Policy Optimization (MAPPO)** dengan paradigma *Centralized Training with Decentralized Execution (CTDE)*:
- **Individualized Agent Treatment:** Setiap token yang masuk dipetakan sebagai agen individual mandiri ($a_i^l \in \{0, 1\}$) yang dikendalikan oleh *shared parameter actor network* $\pi_\theta$ berbasis fitur lokal token.
- **Global Sequential Evaluation:** State transisi antar-layer dimodelkan sebagai Markov Game trajectory $(S_1, A_1, R_1, \dots, S_L, A_L, R_L)$. Evaluasi nilai dilakukan oleh *centralized critic network* $V_\phi$ yang menerima fitur global lingkungan (task-specific token / semantic summary) yang digabungkan dengan fitur lokal agen.
- **Dual-Criteria Competitive-Collaborative Reward:** Fungsi reward dirancang untuk menyeimbangkan efisiensi komputasi ($r_{1,i}^l$, penalti preservasi token yang memicu kompetisi alokasi anggaran komputasi) dan akurasi downstream ($r_{2,i}^l$, reward akurasi kolektif yang diamortisasi terhadap jumlah token tersisa untuk memicu kolaborasi).

Hasil validasi empiris membuktikan bahwa formulasi Markov Game MAPPO ini mampu memangkas FLOPs inferensi hingga **$44\%$** dengan degradasi akurasi sangat minimal (**$0.4\%$** tanpa fine-tuning, dan **$0.1\%$** dengan fine-tuning), serta secara natural memprioritaskan pemangkasan agresif pada layer awal (*earliest possible stage pruning*) tanpa memotong token informasi kritis.

---

## 2. Landasan Teori & Analisis Paper Acuan (arXiv:2503.23459)

### 2.1 Keterbatasan Pendekatan Eksisting

Metode kompresi token kontemporer terbagi menjadi dua paradigma utama:
1. **Handcrafted Heuristics (EViT, ToMe, ATS, Tri-Level):** Mengandalkan skor atensi $A_{i,j}$ dari *class token* atau *query token*. Metode ini tidak adaptif terhadap distribusi fitur laten non-linear dan memerlukan kalibrasi manual untuk setiap domain data baru.
2. **Supervised Token Pruning (DynamicViT):** Menggunakan Multi-Layer Perceptron (MLP) untuk memprediksi *keep score* setiap token. Namun, rasio pruning pada setiap stage ($p_1, p_2, p_3$) dikunci sebagai hyperparameter kaku (misal $[0.7, 0.7, 0.7]$). Model tidak memiliki fleksibilitas untuk membuang $90\%$ token pada layer awal untuk input yang sederhana, atau mempertahankan token pada input yang rumit.

### 2.2 Keunggulan Multi-Agent RL Dibanding Single-Agent PPO

| Dimensi | Heuristik Statis / DynamicViT | Single-Agent PPO | Multi-Agent PPO (MAPPO - DEC-07) |
| :--- | :--- | :--- | :--- |
| **Ruang Aksi** | Deterministik / Heuristik | Eksponensial: $\{0, 1\}^N$ ($2^N$ kombinasi aksi) | Terdesentralisasi: $\{0, 1\}$ per agen token |
| **Adaptivitas Data** | Rendah (rasio tetap per layer) | Tinggi, namun lambat konvergen | Sangat Tinggi (Pareto-optimal adaptif) |
| **Credit Assignment** | Tidak ada | Global, tidak mampu membedakan kontribusi token spesifik | Presisi via *Agent-Specific Advantage Estimation* (GAE) |
| **Sensitivitas Sekuensial** | Terputus antar layer | Menengah | Lengkap via *Markov Game Trajectory* lintas layer |
| **Overhead Parameter** | Ringan | Berat (output layer ukuran $N$) | Minimal (shared MLP kecil lintas agen) |

---

## 3. Formulasi Matematis Eksak Markov Game

Pemangkasan token sekuensial antar blok transformer $l = 1, 2, \dots, L$ diformulasikan sebagai Markov Game $\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}_{i \in \mathcal{N}}, \mathcal{P}, \{\mathcal{R}_i\}_{i \in \mathcal{N}}, \gamma \rangle$.

### 3.1 Komponen Game

1. **Himpunan Agen ($\mathcal{N}^l$):**  
   Pada layer $l$, setiap token $i \in \{1, \dots, N_l\}$ (di luar token penanda global seperti `[CLS]` atau `[ANCHOR]`) bertindak sebagai agen individual $a_i^l$. Jumlah agen aktif pada layer $l$ adalah $N_l$.
2. **State Ruang ($\mathcal{S}^l$):**  
   State global $S^l \in \mathbb{R}^{N_l \times d}$ adalah matriks representasi token sebelum pemangkasan pada layer $l$:
   $$S^l = F^l(T^{l-1})$$
   di mana $F^l$ adalah blok Transformer ke-$l$ (Self-Attention + FFN), dan $T^{l-1}$ adalah representasi token pasca-pruning dari layer sebelumnya.
   - **Fitur Lokal Agen ($s_{i,\text{local}}^l \in \mathbb{R}^d$):** Vektor embedding dari token ke-$i$ pada layer $l$.
   - **Fitur Global Lingkungan ($s_{\text{global}}^l \in \mathbb{R}^d$):** Vektor embedding token representasi global (misal `[CLS]` token atau hasil pooling semantik konteks).
   - **Fitur Gabungan untuk Critic ($s_{i,\text{critic}}^l \in \mathbb{R}^{2d}$):**
     $$s_{i,\text{critic}}^l = [s_{i,\text{local}}^l \,\|\, s_{\text{global}}^l]$$
3. **Ruang Aksi Bersama ($\mathcal{A}^l = \prod_{i=1}^{N_l} \mathcal{A}_i^l$):**  
   Aksi dari setiap agen $i$ adalah keputusan biner:
   $$A_i^l \in \{0, 1\}$$
   di mana $A_i^l = 0$ merepresentasikan keputusan untuk memangkas (*prune*) token, dan $A_i^l = 1$ merepresentasikan keputusan untuk mempertahankan (*preserve*) token.
   Aksi bersama (*joint action*) layer $l$ dinotasikan sebagai vektor biner:
   $$A^l = [A_1^l, A_2^l, \dots, A_{N_l}^l]^\top \in \{0, 1\}^{N_l}$$
4. **Fungsi Transisi Probabilistik ($\mathcal{P}$):**  
   Transisi dari state $S^l$ ke $S^{l+1}$ ditentukan oleh eksekusi aksi bersama $A^l$ dan pemrosesan blok Transformer berikutnya:
   $$T^l = S^l \odot A^l$$
   $$S^{l+1} \sim \mathcal{P}(S^{l+1} \mid S^l, A^l) = \delta(S^{l+1} - F^{l+1}(T^l))$$
   Dalam implementasi komputasi batch tensor, token yang dipangkas ($A_i^l = 0$) tidak langsung di-slice dari memori (yang merusak keseragaman dimensi batch), melainkan di-mask pada matriks self-attention:
   $$\text{AttnMask}_{i, j}^l = \begin{cases} 0, & \text{jika } A_i^l = 1 \text{ dan } A_j^l = 1 \\ -1000.0, & \text{jika } A_i^l = 0 \text{ atau } A_j^l = 0 \end{cases}$$
5. **Trajektori Markov Game ($\tau$):**  
   Untuk seluruh kedalaman kompresi, trajektori didefinisikan sebagai:
   $$\tau = (S^1, A^1, R^1, S^2, A^2, R^2, \dots, S^L, A^L, R^L)$$

---

## 4. Mekanisme Multi-Agent PPO (MAPPO) & Arsitektur Jaringan

### 4.1 Desentralisasi Eksekusi dengan Parameter-Sharing

Untuk menjaga overhead parameter tetap minimal dan skalabel terhadap jumlah token yang besar, seluruh agen berbagi parameter pada satu Actor Network $\pi_\theta$ dan satu Critic Network $V_\phi$.

```
                        Input Tokens S^l in R^{N_l x d}
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       [Local Token Feature s_i^l]              [Global Context Feature s_g^l]
                 │                                       │
                 ▼                                       │
        Actor Network π_θ (MLP)                          │
                 │                                       │
                 ▼                                       │
      Action Sampling A_i^l in {0, 1}                    │
                 │                                       │
                 ├───────────────────────┬───────────────┘
                 │                       ▼
                 │             Concatenate [s_i^l || s_g^l]
                 │                       │
                 │                       ▼
                 │            Critic Network V_φ (MLP)
                 │                       │
                 ▼                       ▼
      Token Masking / Pruning       State-Value V(s_i^l, s_g^l)
                 │                       │
                 └───────────────┬───────┘
                                 ▼
                     GAE Advantage Estimation A_i^l
```

1. **Actor Network ($\pi_\theta$):**
   - **Input:** Fitur lokal token $s_{i,\text{local}}^l \in \mathbb{R}^d$.
   - **Arsitektur:** MLP 5-layer dengan fungsi aktivasi GELU:
     $$d \to 4d \to 256 \to 64 \to 2$$
   - **Output:** Logit biner $[z_0, z_1]$, menghasilkan probabilitas pemangkasan via Softmax:
     $$\pi_\theta(A_i^l = a \mid s_{i,\text{local}}^l) = \frac{e^{z_a}}{e^{z_0} + e^{z_1}}$$

2. **Critic Network ($V_\phi$):**
   - **Input:** Fitur gabungan agen-global $s_{i,\text{critic}}^l = [s_{i,\text{local}}^l \,\|\, s_{\text{global}}^l] \in \mathbb{R}^{2d}$.
   - **Arsitektur:** MLP 5-layer dengan fungsi aktivasi GELU:
     $$2d \to 4d \to 256 \to 64 \to 1$$
   - **Output:** Estimasi nilai skalar $V_\phi(s_{i,\text{critic}}^l) \in \mathbb{R}$.

---

## 5. Dinamika Reward: Sinergi Kolaborasi & Kompetisi

Tujuan optimasi kompresi token adalah memaksimalkan efisiensi komputasi (meminimalkan token tersisa) tanpa mengorbankan akurasi tugas downstream. Untuk mencapai Pareto-optimalitas ini, didefinisikan dua komponen reward untuk agen $i$ pada layer $l$:

### 5.1 Komponen Reward Efisiensi ($r_{1,i}^l$ - Kompetisi)

Reward efisiensi menghukum agen yang memilih untuk mempertahankan token, sehingga memicu kompetisi antar-agen untuk memperebutkan kuota retensi:
$$r_{1,i}^l = \begin{cases} 0, & \text{jika } A_i^l = 0 \quad (\text{Token dipangkas / hemat komputasi}) \\ -1, & \text{jika } A_i^l = 1 \quad (\text{Token dipertahankan / membebani biaya}) \end{cases}$$

### 5.2 Komponen Reward Akurasi Teramortisasi ($r_{2,i}^l$ - Kolaborasi)

Reward akurasi memberikan insentif kolektif jika tugas downstream berhasil diselesaikan, namun diamortisasi terhadap jumlah total token yang dipertahankan pada layer tersebut ($n_l$):
$$r_{2,i}^l = \frac{\gamma_{\text{task}}}{n_l + \epsilon}$$
di mana:
- $\gamma_{\text{task}} \in [0, 1]$ adalah metrik performa downstream task (misal: 1 jika klasifikasi/penalaran benar, 0 jika salah; atau skor retensi F1/ROUGE).
- $n_l = \sum_{j=1}^{N_l} A_j^l$ adalah jumlah total token yang dipertahankan pada layer $l$.
- $\epsilon = 10^{-6}$ mencegah pembagian dengan nol.

> **Analisis Dinamika Kolaborasi-Kompetisi:**
> - Pembagian dengan $n_l$ memberikan efek regulasi yang krusial: semakin sedikit token yang dipertahankan untuk mencapai keberhasilan downstream, semakin besar reward yang diterima oleh masing-masing token yang selamat.
> - Hal ini memaksa agen untuk berkolaborasi hanya mempertahankan token yang benar-benar esensial (*maximum information density*).

### 5.3 Total Reward Komposit

Total reward untuk agen $i$ pada layer $l$ adalah kombinasi linear dengan bobot trade-off $\alpha$ dan $\beta$:
$$R_i^l = \alpha \cdot r_{1,i}^l + \beta \cdot r_{2,i}^l = - \alpha \cdot A_i^l + \beta \cdot \frac{\gamma_{\text{task}}}{\sum_{j=1}^{N_l} A_j^l + \epsilon}$$

Rasio $\frac{\alpha}{\beta}$ bertindak sebagai tuas kontrol model complexity (GFLOPs):
- **Nilai $\frac{\alpha}{\beta}$ lebih tinggi:** Mengutamakan efisiensi; model memangkas lebih banyak token (GFLOPs rendah).
- **Nilai $\frac{\alpha}{\beta}$ lebih rendah:** Mengutamakan akurasi; model lebih permisif mempertahankan token.

---

## 6. Algoritma Optimasi & Pembaruan Kebijakan MAPPO

### 6.1 Perhitungan Generalized Advantage Estimation (GAE)

Untuk trajektori Markov Game sepanjang $L$ layer, reward gabungan agen dievaluasi menggunakan GAE ($\gamma_{\text{disc}}, \lambda_{\text{gae}}$):
$$\delta_l^i = R_i^l + \gamma_{\text{disc}} V_\phi(s_{i,\text{critic}}^{l+1}) - V_\phi(s_{i,\text{critic}}^l)$$
$$\hat{A}_i^l = \sum_{k=0}^{L - l} (\gamma_{\text{disc}} \lambda_{\text{gae}})^k \delta_{l+k}^i$$

### 6.2 MAPPO Policy Clipped Objective

Actor network diperbarui untuk memaksimalkan fungsi objektif terklip:
$$\mathcal{L}_\theta = \frac{1}{\sum_{b=1}^B M^b} \sum_{b=1}^B \sum_{m=1}^{M^b} \min \left( r_{\theta, m}^b \hat{A}_m^b, \; \text{clip}(r_{\theta, m}^b, 1 - \epsilon_\theta, 1 + \epsilon_\theta) \hat{A}_m^b \right)$$
di mana:
- $r_{\theta, m}^b = \frac{\pi_\theta(A_m^b \mid s_m^b)}{\pi_{\theta_{\text{old}}}(A_m^b \mid s_m^b)}$ adalah importance sampling ratio.
- $\epsilon_\theta \in [0.1, 0.2]$ adalah clipping threshold.
- $M^b = \sum_{l=1}^L N_l^b$ adalah total seluruh agen pada semua layer dalam trajektori sampel $b$.

### 6.3 MAPPO Value Loss with Clipping

Critic network diperbarui dengan meminimalkan value loss terklip terhadap target return $R_m^b$:
$$\mathcal{L}_\phi = \frac{1}{\sum_{b=1}^B M^b} \sum_{b=1}^B \sum_{m=1}^{M^b} \max \left( \left(V_\phi(s_m^b) - \hat{R}_m^b\right)^2, \; \left(\text{clip}\left(V_\phi(s_m^b), V_{\phi_{\text{old}}}(s_m^b) \pm \epsilon_\phi\right) - \hat{R}_m^b\right)^2 \right)$$

---

## 7. Integrasi dengan Arsitektur Lethon (KV-Cache Compression)

Dalam sistem Lethon, formulasi Markov Game MAPPO ini diterapkan untuk menyelesaikan **Multi-Layer KV-Cache Pruning** pada fase decoding LLM:

```
                  Layer l-1 KV-Cache
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│             Lethon Layer l Attention Block             │
│                                                        │
│  1. Extract Local Token Key Vectors k_i in R^d         │
│  2. Compute Query/Context Anchor Summary s_global      │
│  3. MAPPO Actor π_θ -> Binary Decision A_i in {0, 1}   │
│  4. Apply Attention Mask to Pruned KV Slots (-1000.0)  │
│  5. Forward Only Active KV Pairs to Layer l+1          │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
                  Layer l+1 KV-Cache
```

1. **Earliest-Stage Pruning Phenomenon:**  
   Konsisten dengan temuan empiris pada DeiT-B (Gambar 5 arXiv:2503.23459), agen Markov Game yang terlatih belajar membuang redundansi terbesar pada layer-layer awal ($l \in \{1, \dots, L/3\}$). Hal ini menghasilkan penghematan kumulatif maksimal pada alokasi memori KV-Cache di seluruh layer inferensi berikutnya.
2. **Koordinasi dengan Semantic Anchors (Stage 2):**  
   Token *anchor* bertindak sebagai penyedia fitur $s_{\text{global}}$ yang stabil untuk mengevaluasi apakah sebuah token lokal unik masih membawa informasi yang belum terserap oleh anchor.

---

## 8. Verifikasi & Pengujian Teknis

Implementasi teknis mencakup modul Python mandiri di `lethon/core/rl/mappo_pruning.py` dengan cakupan pengujian unit komprehensif di `tests/unit/test_mappo_pruning.py`:
1. **Mathematical Invariant Test:** Verifikasi reward $r_{1,i}^l \in \{0, -1\}$, amortisasi $r_{2,i}^l = \frac{\gamma}{n_l}$, dan skala komposit $R_i^l$.
2. **Actor & Critic Network Dimension Invariant:** Validasi shape keluaran logit $\pi_\theta \in \mathbb{R}^{B \times N \times 2}$ dan nilai scalar $V_\phi \in \mathbb{R}^{B \times N}$.
3. **Attention Masking Consistency:** Verifikasi bahwa token yang dipangkas ($A_i = 0$) mendapatkan nilai mask $-1000.0$ dan probabilitas Softmax mendekati $0.0$.
4. **Markov Game Trajectory Collector:** Validasi keutuhan riwayat multi-layer $(S^l, A^l, R^l)_{l=1}^L$ dan perhitungan GAE multi-step.
