# DEC-05: Sequential Decision MDP & Autoencoder Compression (Stage 4)

## 1. Executive Summary & Context

Modul **Sequential Decision MDP & Autoencoder Compression** merupakan komponen inti tahap akhir (*Stage 4*) pada pipeline kompresi konteks Lethon ([PRD.md](../../PRD.md)), beroperasi berdampingan dengan pemadatan alur penalaran CoT (*ThinkPrune*, [DEC-04](dec-04-cot-iterative-pruning.md)). Modul ini dirancang untuk memecahkan hambatan inferensi pada model multimodal (*Large Vision-Language Models* / LVLMs dan model konteks panjang): **beban komputasi eksponensial akibat masifnya token visual/konteks awal serta kelemahan metode pemangkasan heuristik satu langkah (*one-shot heuristic pruning*)**.

Metode pemangkasan token visual konvensional (seperti ToMe, FastV, SparseVLM, dan PruMerge) umumnya mengandalkan skor heuristik statis berbasis *attention weights* atau *cosine similarity*. Pendekatan ini memiliki dua kelemahan fundamental:
1. **Task-Agnostic:** Keputusan pemangkasan token terisolasi dari kebutuhan semantik kueri downstream pengguna (*language query*), sehingga sering kali membuang token visual berukuran kecil namun krusial untuk menjawab pertanyaan spesifik.
2. **One-Shot Rigidity:** Memperlakukan kompresi sebagai eliminasi seketika tanpa memodelkan dinamika penyempurnaan informasi bertahap (*progressive information refinement*) dan ketergantungan antar-langkah (*sequential step-to-step dependencies*).

Berdasarkan riset terobosan pada **TPRL (Language-Guided Token Compression with Reinforcement Learning in LVLMs, *arXiv:2603.13394v1*)**, Lethon Stage 4 mengadopsi formulasi **Sequential Markov Decision Process (MDP)**:
- **State Representation Compact:** Menggunakan *self-supervised token-wise autoencoder* yang memadatkan dimensi fitur setiap token ($d_v \to d_l$, reduksi $16\times$) sebelum diteruskan ke policy network, menjaga overhead komputasi policy tetap di bawah $1.5\%$ dari total inferensi.
- **Dual-Criterion Composite Reward:** Mengoptimalkan kompromi Pareto antara akurasi tugas downstream ($R_{\text{task}}$ berbasis gain diferensial) dan efisiensi komputasi ($R_{\text{eff}}$ berbasis rasio pemangkasan langkah).
- **Learning from Demonstrations (LfD) Warm-Start:** Mengatasi *sample inefficiency* dan kegagalan eksplorasi pada ruang aksi kombinatorial $2^{K_t}$ dengan melakukan inisialisasi bobot policy melalui supervisi trajektori heuristik, dilanjutkan dengan *fine-tuning* berbasis *Proximal Policy Optimization* (PPO).

Hasil empiris membuktikan bahwa formulasi ini mampu memangkas hingga **$66.7\%$ token visual**, mereduksi **FLOPs sebesar $54.2\%$**, memangkas ukuran **KV-Cache hingga $66.6\%$**, dan mempercepat latensi inferensi hingga **$53.8\%$**, dengan degradasi akurasi rata-rata hanya **$0.7\%$** di 7 benchmark multimodal standar, bahkan melampaui baseline *Upper Bound* tanpa pruning pada MMBench ($102.3\%$).

---

## 2. Landasan Teori & Analisis Paper Acuan (arXiv:2603.13394v1)

### 2.1 Arsitektur Tiga Tahap Sinergis TPRL

Arsitektur TPRL tersusun atas tiga fase pelatihan yang saling melengkapi (Gambar 1 paper):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ (a) Autoencoder Training for State Compression                              │
│                                                                             │
│  Original Token h_i (d_v=1024) ──► [Encoder E] ──► z_i (d_l=64)             │
│                                                      │                      │
│  Reconstructed h_hat_i ◄────────── [Decoder D] ◄─────┘                      │
│  Minimasi Loss: L_rec = (1/N) * sum ||h_i - D(z_i)||^2                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Freeze Encoder E)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ (b) Learning from Demonstrations (Supervised Warm-Start)                     │
│                                                                             │
│  Trajectory Heuristik (Vispruner): s_t=(Z_t, q) ──► Action Label a_demo     │
│  Policy Network pi_theta(s_t) ──► Retention Probabilities p_{t,i}           │
│  Minimasi Loss: L_BCE(theta) = CrossEntropy(p_{t,i}, a_demo)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Initialized Policy pi_theta_0)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ (c) Reinforcement Learning Fine-Tuning via PPO                              │
│                                                                             │
│  Trajectory Rollout (T_max = 3 steps) ──► Reward r_t = R_task + R_eff(t)   │
│  Advantage Estimation via GAE (delta_t, A_hat_t)                            │
│  Policy & Value Optimization: L(theta, phi) = -L_CLIP + c1*L_VF - c2*L_S    │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Autoencoder State Compression (Fase a):**  
   Token visual $h_i \in \mathbb{R}^{d_v}$ memiliki dimensi fitur yang besar ($d_v = 1024$ pada CLIP-ViT-L/14 atau $d_v = 4096$ pada representasi internal LLM). Menjalankan modul policy atensi langsung pada dimensi asli memicu pembengkakan parameter dan latensi berlebih. Melalui autoencoder berparameter ringan, representasi state diproyeksikan ke ruang laten kompak $z_i \in \mathbb{R}^{d_l}$ ($d_l = 64$ atau $256$, kompresi $16\times$). Setelah konvergen, encoder dibekukan (*frozen*), dan decoder dibuang saat tahap RL dan inferensi.
2. **Learning from Demonstrations (Fase b):**  
   Ruang aksi token-wise bersifat biner untuk setiap token ($a_t \in \{0, 1\}^{K_t}$), menghasilkan $2^{K_t}$ konfigurasi keputusan pada setiap langkah $t$. Jika dilatih dari inisialisasi acak (*RL from scratch*), gradien eksplorasi mengalami varians ekstrem dan terjebak pada kebijakan suboptimal (bahkan runtuh ke solusi trivial di mana seluruh token dibuang atau seluruh token dipertahankan). Dengan meniru trajektori bertingkat dari *reference pruner* (seperti Vispruner pada rasio pemangkasan $0\% \to 25\% \to 50\%$), policy network memperoleh *semantic prior* yang kuat mengenai relevansi spasial visual.
3. **Sequential PPO Fine-Tuning (Fase c):**  
   Menggunakan policy terinisialisasi, sistem mengeksplorasi trajektori pemangkasan multi-langkah ($T_{\max} = 3$). Pada setiap langkah, policy memprediksi probabilitas retensi $p_{t,i}$, mengeksekusi aksi $a_{t,i}$, menerima umpan balik kinerja downstream $R_{\text{task}}$ serta bonus efisiensi $R_{\text{eff}}$, dan mengoptimalkan fungsi objektif PPO terkoreksi kliping.

---

## 3. Formulasi Matematis Eksak Language-Guided MDP

Proses kompresi token sekuensial dimodelkan sebagai *Markov Decision Process* terdisiplin yang dinyatakan oleh tuple:
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

### 3.1 State Space ($\mathcal{S}$)

Diberikan input visual awal berupa sekuens $N$ token $H = [\mathbf{h}_1, \dots, \mathbf{h}_N]^\top \in \mathbb{R}^{N \times d_v}$ dan kueri instruksi/bahasa teks $q \in \mathbb{R}^{d_q}$.

1. **Kompresi Fitur Token-Wise:**  
   Encoder terlatih $\mathcal{E}: \mathbb{R}^{d_v} \to \mathbb{R}^{d_l}$ memetakan setiap token secara independen:
   $$\mathbf{z}_i = \mathcal{E}(\mathbf{h}_i), \quad \forall i \in \{1, \dots, N\}, \quad d_l \ll d_v$$
   Matriks representasi laten awal didefinisikan sebagai $Z_0 = [\mathbf{z}_1, \dots, \mathbf{z}_N] \in \mathbb{R}^{N \times d_l}$.

2. **Representasi State pada Langkah $t$ ($s_t \in \mathcal{S}$):**  
   Pada langkah keputusan $t \in \{1, \dots, T_{\max}\}$, state didefinisikan sebagai pasangan matriks representasi token yang masih bertahan $Z_t$ dan vektor kueri teks $q$:
   $$s_t = (Z_t, q)$$
   di mana $Z_t \in \mathbb{R}^{K_t \times d_l}$ memuat kode laten dari $K_t$ token yang belum dipangkas ($K_0 = N$).

3. **Vektor Pelacak Mask Global ($m \in \{0, 1\}^N$):**  
   Untuk menjaga korespondensi indeks token asli $H$ saat diteruskan ke downstream LLM, sistem mengelola vektor biner global $m \in \{0, 1\}^N$.  
   - Inisialisasi: $m^{(0)}[idx] = 1, \quad \forall idx \in \{1, \dots, N\}$.
   - Pembaruan: Jika token dengan indeks asli $idx$ diputuskan untuk dibuang pada langkah $t$, maka:
     $$m^{(t)}[idx] = 0$$
   - Indeks asli $idx$ bersifat invarian terhadap reduksi token perantara. Representasi laten terkompresi $Z_t$ hanya digunakan oleh policy network dan **tidak pernah diteruskan ke LLM**.

### 3.2 Action Space ($\mathcal{A}$)

Pada langkah $t$, agen mengevaluasi seluruh $K_t$ token dalam $Z_t$ dan menghasilkan vektor keputusan retensi biner:
$$a_t = \{d_i \mid i \in \{1, \dots, K_t\}, \; d_i \in \{0, 1\}\}$$
- $d_i = 1$: Token ke-$i$ dipertahankan (*retained*).
- $d_i = 0$: Token ke-$i$ dipangkas (*pruned*).

### 3.3 State Transition Dynamics ($\mathcal{P}$)

Dinamika transisi bersifat deterministik bersyarat pada aksi $a_t$. Kumpulan kode representasi laten untuk langkah berikutnya $t+1$ dibentuk melalui subsetting:
$$Z_{t+1} = \{\mathbf{z}_{t,i} \mid d_i = 1, \; i \in \{1, \dots, K_t\}\} \in \mathbb{R}^{K_{t+1} \times d_l}$$
di mana jumlah token aktif menyusut: $K_{t+1} = \sum_{i=1}^{K_t} d_i \le K_t$.

### 3.4 Dual-Criterion Reward Function ($\mathcal{R}$)

Fungsi reward pada langkah $t$ menyeimbangkan akurasi tugas downstream dan efisiensi komputasi, dihitung di atas mini-batch berukuran $B$ sampel untuk mereduksi varians estimasi:
$$r_t = R_{\text{task}}(t) + R_{\text{eff}}(t)$$

1. **Reward Performa Tugas Downstream ($R_{\text{task}}$):**  
   Mengukur perubahan diferensial skor downstream antara langkah $t$ dan langkah $t-1$:
   $$R_{\text{task}}(t) = \alpha \cdot \frac{1}{B} \sum_{b=1}^{B} \left[ \mathrm{Score}\left(H_t^{(b)}, q^{(b)}\right) - \mathrm{Score}\left(H_{t-1}^{(b)}, q^{(b)}\right) \right]$$
   di mana $H_t^{(b)} = \{\mathbf{h}_i^{(b)} \mid m_t^{(b)}[i] = 1\}$ adalah himpunan token asli yang masih aktif setelah pemangkasan langkah $t$. Formulasi diferensial ini memastikan agen dihukum jika pemangkasan langkah $t$ mendegradasi akurasi, dan diberi reward positif jika pemangkasan berhasil mempertahankan/menaikkan akurasi (misal melalui filtering noise latar belakang).

2. **Reward Efisiensi Komputasi ($R_{\text{eff}}$):**  
   Memberikan insentif langsung atas persentase token yang berhasil dibuang pada langkah aktif:
   $$R_{\text{eff}}(t) = \beta \cdot \frac{1}{B} \sum_{b=1}^{B} \left(1 - \frac{K_t^{(b)}}{K_{t-1}^{(b)}}\right)$$
   dengan parameter default TPRL: $\alpha = 1.0$ dan $\beta = 0.1$.

### 3.5 Discount Factor ($\gamma$) & Step-Wise Action Sampling

- Faktor diskonto temporal: $\gamma \in [0, 1]$ (default: $\gamma = 0.99$).
- **Step-Wise Discounted Exploration:**  
  Untuk mendorong pemangkasan yang lebih agresif pada langkah-langkah lanjut serta mempercepat konvergensi, probabilitas retensi $p_{t,i}$ didiskonto oleh faktor langkah $\lambda_{\text{disc}} \in (0, 1)$ (default: $\lambda_{\text{disc}} = 0.5$):
  $$P(a_{t,i} = 1) = \lambda_{\text{disc}}^t \cdot p_{t,i}$$
  $$P(a_{t,i} = 0) = 1 - \lambda_{\text{disc}}^t \cdot p_{t,i}$$
  Aksi selama fase RL disampel dari distribusi Bernoulli: $a_{t,i} \sim \mathrm{Bernoulli}\left(\lambda_{\text{disc}}^t p_{t,i}\right)$.

- **Deterministic One-Shot Inference Protocol:**  
  Saat inferensi deployment, proses pemangkasan dieksekusi secara deterministik satu langkah (*one-shot*): policy memprediksi probabilitas retensi $p_i$, dan token dipangkas jika berada di bawah ambang batas $\tau$:
  $$a_i = \mathbb{I}(p_i > \tau)$$
  Sekuens token terpilih $\tilde{H} = \{\mathbf{h}_{idx} \mid m[idx] = 1\}$ langsung dikirimkan ke downstream LLM. Nilai ambang batas $\tau$ dapat disetel sesuai target rasio kompresi:
  - $\tau = 0.55 \implies \sim 192$ token tersisa (pemangkasan $66.7\%$).
  - $\tau = 0.67 \implies \sim 144$ token tersisa (pemangkasan $75.0\%$).
  - $\tau = 0.74 \implies \sim 128$ token tersisa (pemangkasan $77.8\%$).

---

## 4. Arsitektur Jaringan: Autoencoder, Policy, dan Value Network

```
                                [State s_t = (Z_t, q)]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [Z_t in R^{K_t x d_l}]                          [q in R^{d_q}]
                  │                                               │
                  │                                         [Linear Proj]
                  │                                               │
                  └───────────────┬───────────────────────────────┘
                                  ▼
                   Concatenation X = [Z_t ; q_tilde]  ((K_t + 1) x d)
                                  │
                                  ▼
                     Multi-Head Self-Attention (MHSA)
                                  │
                                  ▼
                     LayerNorm & Residual Connection
                                  │
                                  ▼
               Enriched Sequence Features F = [f_1, ..., f_{K_t}, f_q]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
   [Policy Head]                                     [Value Head]
   Token-wise: g_i = FFN(f_i)                        Mean Pooling: f_bar = (1/K_t) * sum(f_i)
   Retention: p_{t,i} = sigma(w^T g_i + b)           State Value: V_phi(s_t) = u^T FFN(f_bar) + b_v
```

### 4.1 Arsitektur Token-Wise Autoencoder

Autoencoder dilatih secara *self-supervised* sebelum fase RL untuk memadatkan fitur individual token tanpa mengubah jumlah token $N$.
- **Encoder ($\mathcal{E}$):** 2-layer MLP dengan proyeksi $d_v \to 256 \to d_l$. Dilengkapi LayerNorm dan aktivasi GELU:
  $$\mathbf{z}_i = \mathcal{E}(\mathbf{h}_i) = W_e^{(2)} \cdot \text{GELU}\left(\text{LayerNorm}\left(W_e^{(1)} \mathbf{h}_i + \mathbf{b}_e^{(1)}\right)\right) + \mathbf{b}_e^{(2)}$$
  di mana $d_v = 1024$ (CLIP) dan $d_l = 64$ (rasio kompresi $16\times$).
- **Decoder ($\mathcal{D}$):** Struktur simetris merekonstruksi representasi asli $\hat{\mathbf{h}}_i = \mathcal{D}(\mathbf{z}_i)$ melalui proyeksi $d_l \to 256 \to d_v$.
- **Fungsi Objektif Rekonstruksi:**
  $$\mathcal{L}_{\text{rec}} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{h}_i - \mathcal{D}(\mathbf{z}_i)\|_2^2$$
Setelah pelatihan selesai, $\mathcal{E}$ dibekukan (*frozen*), sedangkan $\mathcal{D}$ dieliminasi dari memori komputasi inferensi.

### 4.2 Shared Attention Module

Untuk meminimalkan jumlah parameter dan memungkinkan interaksi semantik menyeluruh antara token visual terkompresi dan teks instruksi, Policy Network ($\pi_\theta$) dan Value Network ($V_\phi$) berbagi modul ekstraksi fitur berbasis atensi:
1. Proyeksi kueri teks ke dimensi model $d$: $\tilde{q} = W_q^{\text{proj}} q + b_q^{\text{proj}} \in \mathbb{R}^{1 \times d}$.
2. Konkatenasi sekuens: $X = [Z_t; \tilde{q}] \in \mathbb{R}^{(K_t + 1) \times d}$.
3. Multi-Head Self-Attention:
   $$\text{Attention}(X) = \text{softmax}\left(\frac{X W_Q (X W_K)^\top}{\sqrt{d_k}}\right) X W_V$$
4. LayerNorm dan Residual:
   $$F = \text{LayerNorm}(X + \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O) = [\mathbf{f}_1, \dots, \mathbf{f}_{K_t}, \mathbf{f}_q]$$

### 4.3 Policy Network Head ($\pi_\theta$)

Untuk setiap token visual ke-$i$, policy network menghitung probabilitas retensi:
$$\mathbf{g}_i = \text{LayerNorm}\left(\mathbf{f}_i + W_2 \cdot \text{GELU}(W_1 \mathbf{f}_i + \mathbf{b}_1) + \mathbf{b}_2\right)$$
$$p_{t,i} = \sigma\left(\mathbf{w}^\top \mathbf{g}_i + b\right)$$
di mana $\sigma(x) = \frac{1}{1 + e^{-x}}$.

### 4.4 Value Network Head ($V_\phi$)

Value network mengestimasi nilai skalar state $V(s_t)$ dengan mengagregasi representasi seluruh token aktif melalui *mean pooling*:
$$\bar{\mathbf{f}} = \frac{1}{K_t} \sum_{i=1}^{K_t} \mathbf{f}_i$$
$$V_\phi(s_t) = \mathbf{u}^\top \text{LayerNorm}\left(\bar{\mathbf{f}} + U_2 \cdot \text{GELU}(U_1 \bar{\mathbf{f}} + \mathbf{c}_1) + \mathbf{c}_2\right) + b_v$$

---

## 5. Learning from Demonstrations (LfD) & PPO Policy Optimization

### 5.1 Supervised Pre-Training via LfD

Untuk menghindari jebakan konvergensi lambat dan kebijakan suboptimal dari eksplorasi acak, policy network $\pi_\theta$ terlebih dahulu dilatih dengan supervised learning meniru trajektori heuristik dari Vispruner:
1. **Generasi Trajektori Demonstrasi:**  
   Untuk setiap sampel data latih, dihasilkan trajektori berjenjang bertahap dengan rasio pemangkasan bertambah:
   $$0\% \text{ mask} \implies 25\% \text{ mask} \implies 50\% \text{ mask}$$
   Setiap trajektori dicatat sebagai pasangan state-aksi $(s_t, a_t^{\text{demo}})$, di mana label aksi per-token:
   $$a_{t,i}^{\text{demo}} = \begin{cases} 1, & \text{jika token } i \text{ dipertahankan oleh heuristik pada langkah } t \\ 0, & \text{lainnya} \end{cases}$$
2. **Minimasi Binary Cross-Entropy Loss:**
   $$\mathcal{L}_{\text{BCE}}(\theta) = -\frac{1}{|\mathcal{D}_{\text{demo}}|} \sum_{\tau \in \mathcal{D}_{\text{demo}}} \sum_{t} \sum_{i=1}^{K_t} \left[ a_{t,i}^{\text{demo}} \log p_{t,i} + (1 - a_{t,i}^{\text{demo}}) \log(1 - p_{t,i}) \right]$$
Inisialisasi ini memberikan pemahaman dasar relevansi visual (*foundational visual salience*) yang stabil sebelum dioptimalkan lebih lanjut oleh RL.

### 5.2 Generalized Advantage Estimation (GAE)

Advantage temporal $\hat{A}_t$ dihitung menggunakan Generalized Advantage Estimation (Schulman et al., 2015):
$$\delta_t = r_t + \gamma V_{\phi_{\text{old}}}(s_{t+1}) - V_{\phi_{\text{old}}}(s_t)$$
$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}$$
dengan parameter default $\gamma = 0.99$ dan $\lambda = 0.95$.

### 5.3 Fungsi Objektif PPO Multi-Term

Total loss yang diminimasi untuk memperbarui parameter policy $\theta$ dan value network $\phi$:
$$\mathcal{L}(\theta, \phi) = -\mathcal{L}^{\text{CLIP}}(\theta) + c_1 \mathcal{L}^{\text{VF}}(\phi) - c_2 \mathcal{L}^{\text{S}}(\theta)$$

1. **Clipped Surrogate Policy Loss:**
   $$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
   di mana $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ adalah *importance sampling ratio*, dan $\epsilon = 0.2$.
2. **Value Function Regression Loss:**
   $$\mathcal{L}^{\text{VF}}(\phi) = \hat{\mathbb{E}}_t \left[ \left(V_\phi(s_t) - V_t^{\text{target}}\right)^2 \right]$$
3. **Entropy Regularization:**
   $$\mathcal{L}^{\text{S}}(\theta) = \hat{\mathbb{E}}_t \left[ \mathcal{H}(\pi_\theta(\cdot \mid s_t)) \right]$$
dengan bobot penyeimbang default: $c_1 = 0.5$ dan $c_2 = 0.01$.

---

## 6. Analisis Empiris Benchmark & Bukti Kinerja

### 6.1 Evaluasi Komparatif pada LLaVA-1.5-7B (Tabel 1 Paper)

| Metode Pemangkasan | Retained Token | GQA | MMBench | MME | POPE | SEED | SQAI | VQAv2 | Avg. Relative (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Upper Bound (Uncompressed)** | 576 | 62.0 | 64.6 | 1553.4 | 85.9 | 60.3 | 70.4 | 78.9 | **100.0%** |
| ToMe (ICLR 2023) | 192 | 54.3 | 60.5 | 1485.0 | 72.4 | 53.1 | 65.2 | 67.9 | 90.5% |
| FastV (ECCV 2024) | 192 | 52.7 | 61.2 | 1312.5 | 64.7 | 50.9 | 65.4 | 67.2 | 86.0% |
| SparseVLM (2024) | 192 | 57.6 | 62.5 | 1473.0 | 83.6 | 53.0 | 67.2 | 75.6 | 94.2% |
| PruMerge+ (ICCV 2025) | 144 | 61.3 | 64.9 | 1462.4 | 84.0 | - | 68.3 | 76.8 | 97.6% |
| Vispruner (ICCV 2025) | 192 | 58.1 | 62.9 | 1475.0 | 84.9 | - | 69.2 | 76.5 | 96.4% |
| **TPRL (Ours @ 192 tokens)** | **192\*** | **61.1** | **66.1** | **1554.1** | **85.6** | **58.9** | **69.0** | **78.6** | **99.3% (-0.7% Drop)** |
| **TPRL (Ours @ 144 tokens)** | **144\*** | **60.7** | **65.8** | **1539.4** | **84.0** | **58.5** | **68.1** | **77.5** | **98.4% (-1.6% Drop)** |
| **TPRL (Ours @ 128 tokens)** | **128\*** | **60.2** | **65.4** | **1533.2** | **83.2** | **58.2** | **67.7** | **76.8** | **97.7% (-2.3% Drop)** |

*Catatan Kritis:*  
Pada benchmark **MMBench**, TPRL mencatatkan skor **66.1** ($102.3\%$ relatif terhadap Upper Bound 64.6). Agen RL berhasil menyaring noise latar belakang yang tidak relevan, secara efektif membersihkan gangguan visual sehingga memudahkan LLM memahami pertanyaan konseptual.

### 6.2 Evaluasi pada Skala Model Lain (Tabel 2 & 3 Paper)

- **LLaVA-1.5-13B (Tabel 2):**  
  Pada kapasitas 192 token, TPRL mempertahankan **$99.2\%$** rata-rata performa baseline (GQA: 62.8 vs 63.3; MMBench: 68.95 vs 68.64). Pada 128 token ($77.8\%$ pruning rate), performa tetap bertahan di **$97.1\%$**, jauh mengungguli FastV yang jatuh ke $80.5\%$.
- **Qwen2.5-VL-3B (Tabel 3):**  
  Dengan arsitektur vision NaViT beresolusi dinamis, TPRL pada 144 token mencapai **$98.1\%$** rata-rata performa baseline, dan pada VQA-v2 mencapai skor 77.9 vs 77.4 ($100.6\%$).

### 6.3 Analisis Efisiensi Komputasi & KV-Cache (Tabel 4 Paper)

Total kompleksitas inferensi dimodelkan sebagai:
$$\mathcal{F}_{\text{total}} = \mathcal{F}_{\text{vision}} + \mathcal{F}_{\text{pruner}} + \mathcal{F}_{\text{prefill}} + \mathcal{F}_{\text{decode}}$$

| Konfigurasi Model | Token Count | Avg. Accuracy | Cache Storage (MB) | Reduksi Memori (%) | Latency (ms) | Speedup (%) | FLOPs (TeraFLOPs) | Reduksi FLOPs (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Upper Bound Baseline** | 576 | 100.0% | 302.4 | - | 57.8 | - | 9.6 | - |
| FastV Heuristic | 192 | 86.0% | 100.8 | 66.7% | 23.1 | 60.0% | 2.0 | 79.2% |
| SparseVLM Heuristic | 192 | 94.2% | 100.8 | 66.7% | 26.0 | 55.0% | 3.9 | 59.4% |
| **TPRL (Lethon Stage 4 @ 192)** | **192\*** | **99.3%** | **100.9** | **66.6%** | **26.7** | **53.8%** | **4.4** | **54.2%** |
| **TPRL (Lethon Stage 4 @ 144)** | **144\*** | **98.4%** | **75.7** | **75.0%** | **22.6** | **60.9%** | **3.7** | **61.4%** |
| **TPRL (Lethon Stage 4 @ 128)** | **128\*** | **97.7%** | **67.3** | **77.7%** | **21.2** | **63.3%** | **3.5** | **63.5%** |

### 6.4 Rekapitulasi Studi Ablasi Kunci

1. **Efek Learning from Demonstrations (Tabel 6):**  
   - Vispruner statis murni: Akurasi relatif $97.0\%$ (VQAv2), $95.0\%$ (MME).
   - RL dari inisialisasi acak (*from scratch*): Akurasi jatuh ke $96.3\%$ (VQAv2) dan $95.9\%$ (POPE) karena eksplorasi gagal di ruang aksi $2^{K_t}$.
   - **TPRL (LfD + PPO):** Mencapai **$99.6\%$** (VQAv2) dan **$99.7\%$** (POPE), membuktikan sinergi penting antara inisialisasi terarah dan optimasi RL.
2. **Dimensi Kompresi State $d_l$ (Gambar 2):**  
   Memadatkan fitur dari 4096 ke 256 menghasilkan kurva konvergensi dan akurasi final yang identik dengan representasi penuh tanpa kompresi. Namun pemadatan ekstrem ke $d_l = 128$ menimbulkan *information bottleneck* yang menurunkan performa. Rasio kompresi $16\times$ ($1024 \to 64$) diadopsi sebagai konfigurasi optimal.
3. **Horizon Pemangkasan Maksimum $T_{\max}$ (Tabel 7):**  
   - $T_{\max} = 1$: Rel. Acc $95.2\%$
   - $T_{\max} = 2$: Rel. Acc $98.8\%$
   - **$T_{\max} = 3$:** Rel. Acc **$99.6\%$** (optimal)
   - $T_{\max} \ge 4$: Akurasi mulai saturasi dan menurun ($96.2\%$ dan $93.7\%$) akibat akumulasi eror compounding.
4. **Arsitektur PolicyNet (Tabel 8):**  
   - MLP-only (tanpa atensi): Rata-rata akurasi drop ke $87.4\% - 89.6\%$.
   - **Shared Multi-Head Attention + MLP:** Mencapai **$99.6\% - 102.3\%$**, membuktikan bahwa interaksi relasional global antar-token sangat krusial untuk keputusan pemangkasan yang cerdas.
5. **Robustness Algoritma RL (Gambar 3):**  
   PPO, MAPO, dan GRPO mencapai kinerja retensi yang sebanding ($>98.5\%$). PPO dipilih karena kestabilan numerik dan ketersediaan pustaka mature.

---

## 7. Harmonisasi dengan Pipeline Lethon & PRD Vision

Pada pipeline 4-tahap Lethon ([PRD.md](../../PRD.md)):

```
Input Tokens / Context
          │
          ▼
┌────────────────────────────────────────┐
│ Stage 1: Fast Spectral 1D-FFT          │  --> Parameter-free noise truncation O(N log N)
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 2: Contextual Semantic Anchor    │  --> Aggregation ke in-context anchor tokens
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 3: Promising Subspace Masking    │  --> Low-rank subspace action restriction
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 4: Sequential Decision MDP Dual  │
│  ├── A. Token Autoencoder MDP (TPRL)   │  --> Language-guided progressive token selection
│  └── B. CoT Length Truncation (TP)     │  --> Iterative reasoning step compression
└──────────────────┬─────────────────────┘
                   │
                   ▼
Output Compressed Context (Hemat 50% - 66.7% Token, Acc Drop < 0.7%)
```

1. **Pre-Filtering:** Stage 1 (Spectral FFT) dan Stage 2 (SAC) telah memangkas redundansi frekuensi tinggi dan memadatkan konteks menjadi sekuens yang lebih bersih.
2. **Subspace Constraint:** Stage 3 (RLPT Subspace Masking) membatasi kandidat aksi ke *promising subspace*, memangkas varians probabilitas retensi.
3. **Stage 4 Dual Optimization:**
   - Untuk masukan konteks/multimodal visual: Modul **TPRL** menjalankan proses keputusan sekuensial MDP dengan state laten $Z_t$ dan kueri instruksi $q$, mereduksi hingga $66.7\%$ token dengan menjaga akurasi downstream task.
   - Untuk alur penalaran panjang (*Long CoT*): Modul **ThinkPrune** menerapkan *iterative length limits* untuk mengeliminasi keraguan verbal dan redundansi pemikiran.
4. **Zero-Proxy Interoperability:** Representasi terkompresi $Z_t$ hanya hidup di dalam policy inference loop; token akhir yang diteruskan ke agent host via MCP Tools (`prune_context`) adalah token asli $\tilde{H}$ yang indeksnya diatur oleh bitmask $m$.

---

## 8. Spesifikasi Algoritmik & Implementasi Python

```python
"""
Lethon Stage 4: Sequential Decision MDP Token Compression (TPRL Module)
Sesuai: arXiv:2603.13394v1 (Language-Guided Token Compression with RL in LVLMs)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List

class TokenWiseAutoencoder:
    """
    Autoencoder berparameter ringan untuk kompresi state representasi token.
    Memetakan d_v (misal 1024) -> d_l (misal 64) -> d_v.
    """
    def __init__(self, d_v: int = 1024, d_hidden: int = 256, d_l: int = 64, seed: int = 42):
        self.d_v = d_v
        self.d_hidden = d_hidden
        self.d_l = d_l
        rng = np.random.RandomState(seed)
        
        # Encoder weights
        self.W_e1 = rng.randn(d_v, d_hidden) * np.sqrt(2.0 / d_v)
        self.b_e1 = np.zeros(d_hidden)
        self.W_e2 = rng.randn(d_hidden, d_l) * np.sqrt(2.0 / d_hidden)
        self.b_e2 = np.zeros(d_l)
        
        # Decoder weights
        self.W_d1 = rng.randn(d_l, d_hidden) * np.sqrt(2.0 / d_l)
        self.b_d1 = np.zeros(d_hidden)
        self.W_d2 = rng.randn(d_hidden, d_v) * np.sqrt(2.0 / d_hidden)
        self.b_d2 = np.zeros(d_v)
        
        self.frozen = False

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def encode(self, H: np.ndarray) -> np.ndarray:
        """H: (N, d_v) -> Z: (N, d_l)"""
        h1 = self._gelu(self._layer_norm(H @ self.W_e1 + self.b_e1))
        Z = h1 @ self.W_e2 + self.b_e2
        return Z

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Z: (N, d_l) -> H_rec: (N, d_v)"""
        h1 = self._gelu(self._layer_norm(Z @ self.W_d1 + self.b_d1))
        H_rec = h1 @ self.W_d2 + self.b_d2
        return H_rec

    def reconstruction_loss(self, H: np.ndarray) -> float:
        Z = self.encode(H)
        H_rec = self.decode(Z)
        return float(np.mean(np.square(H - H_rec)))


class TPRLPruningPolicy:
    """
    Policy network berbasis Shared Self-Attention untuk language-guided token pruning.
    """
    def __init__(self, d_l: int = 64, d_q: int = 64, d_model: int = 64, seed: int = 42):
        self.d_l = d_l
        self.d_q = d_q
        self.d_model = d_model
        rng = np.random.RandomState(seed)
        
        # Query projection
        self.W_q_proj = rng.randn(d_q, d_model) * np.sqrt(1.0 / d_q)
        
        # Attention projection
        self.W_Q = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.W_K = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.W_V = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.W_O = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        
        # Policy head (FFN + sigmoid)
        self.W_p1 = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.b_p1 = np.zeros(d_model)
        self.W_p2 = rng.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.b_p2 = np.zeros(d_model)
        self.w_out = rng.randn(d_model) * np.sqrt(1.0 / d_model)
        self.b_out = 0.0

    def compute_retention_probabilities(self, Z_t: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Z_t: (K_t, d_l)
        q: (d_q,)
        Returns: p: (K_t,) probabilitas retensi per token
        """
        K_t = Z_t.shape[0]
        q_proj = (q @ self.W_q_proj).reshape(1, self.d_model) # (1, d)
        
        # Concat [Z_t; q_tilde] -> (K_t + 1, d)
        X = np.vstack([Z_t, q_proj])
        
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
        
        # Visual tokens only: (K_t, d)
        F_visual = F[:K_t]
        
        # Policy FFN
        h1 = F_visual @ self.W_p1 + self.b_p1
        h1_gelu = 0.5 * h1 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h1 + 0.044715 * np.power(h1, 3))))
        g = h1_gelu @ self.W_p2 + self.b_p2 + F_visual
        
        # Sigmoid
        logits = g @ self.w_out + self.b_out
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return p


class TPRLPruner:
    """
    Manager eksekusi kompresi token sekuensial TPRL:
    - Autoencoder compression
    - Binary mask tracking
    - Deterministic one-shot inference thresholding
    """
    def __init__(self, autoencoder: TokenWiseAutoencoder, policy: TPRLPruningPolicy):
        self.autoencoder = autoencoder
        self.policy = policy

    def prune_one_shot(self, H: np.ndarray, q: np.ndarray, threshold: float = 0.55) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Inference protocol:
        H: (N, d_v) representasi token asli
        q: (d_q,) vektor kueri teks
        threshold: tau ambang batas retensi (default 0.55 untuk target ~192 token dari 576)
        
        Returns:
            H_retained: (K, d_v) token asli yang dipertahankan
            mask: (N,) binary retention mask
            stats: statistik kompresi
        """
        N = H.shape[0]
        # 1. State compression via frozen autoencoder
        Z = self.autoencoder.encode(H)
        
        # 2. Compute probabilities
        probs = self.policy.compute_retention_probabilities(Z, q)
        
        # 3. Deterministic thresholding
        mask = (probs > threshold).astype(np.int32)
        
        # Guard: minimal 1 token dipertahankan jika semua probabilitas rendah
        if np.sum(mask) == 0:
            top_idx = int(np.argmax(probs))
            mask[top_idx] = 1
            
        retained_indices = np.where(mask == 1)[0]
        H_retained = H[retained_indices]
        
        stats = {
            "original_tokens": N,
            "retained_tokens": int(np.sum(mask)),
            "pruning_ratio": float(1.0 - (np.sum(mask) / N)),
            "mean_retention_prob": float(np.mean(probs)),
            "threshold": threshold,
        }
        return H_retained, mask, stats
```

---

## 9. Kesimpulan & Rekomendasi Integrasi

1. **Efisiensi Superior dengan Near-Lossless Accuracy:**  
   Formulasi Sequential MDP TPRL membuktikan bahwa kompresi token visual/konteks hingga $66.7\%$ dapat dicapai dengan degradasi akurasi downstream di bawah $0.7\%$, sekaligus memangkas beban KV-Cache sebesar $66.6\%$ dan FLOPs sebesar $54.2\%$.
2. **Kemandirian State Laten:**  
   Penggunaan representasi laten ringkas ($d_l = 64$) memastikan komputasi policy network memiliki overhead negligible ($<15$ ms) tanpa membebani LLM downstream.
3. **Penyempurnaan Stage 4 Lethon:**  
   Modul TPRL beroperasi secara harmonis melengkapi modul *ThinkPrune*: TPRL menangani kompresi spasial/kontekstual sebelum token masuk ke LLM, sedangkan ThinkPrune menangani pemadatan langkah penalaran temporal saat generasi respons.
