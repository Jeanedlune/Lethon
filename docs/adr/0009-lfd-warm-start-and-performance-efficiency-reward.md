# ADR-0009: Learning from Demonstrations (LfD) Warm-Start and PPO Performance-Efficiency Reward Optimization for Sequential Token Pruning

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture Guild & Algorithmic Core Team
- **Consulted Literature:**
  - `arXiv:2603.13394v1`: *Language-Guided Token Compression with Reinforcement Learning in LVLMs (TPRL)* (Cao et al., UESTC / HIT / XAUAT, 2026)
  - `arXiv:2504.01296`: *THINKPRUNE: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning* (Hou et al., 2025)
  - `arXiv:2602.03195v2`: *Reinforcement Learning with Promising Tokens (RLPT)* (2026)
  - `arXiv:2510.08907`: *Semantic-Anchor Compression (SAC) for LLMs* (2025)
  - `PRD.md`: *Lethon System Requirements & 4-Stage Architecture (Stage 4 Sequential MDP & Autoencoder Compression)*
  - `docs/dec/dec-05-tprl-token-compression-mdp.md`: *Technical & Mathematical Specifications of Stage 4 TPRL*

---

## 1. Context & Problem Statement

Pada arsitektur engine kompresi konteks Lethon ([PRD.md](../../PRD.md)), pemrosesan konteks melewati tahapan penyaringan frekuensi (Stage 1 - FFT), agregasi jangkar semantik (Stage 2 - SAC), dan pembatasan subruang aksi prior (Stage 3 - RLPT Subspace Masking). Pada **Stage 4 (Sequential Decision MDP & Autoencoder Compression)**, sistem dihadapkan pada tugas memangkas token input/visual secara dinamis berdasarkan instruksi bahasa teks pengguna (*language-guided token compression*).

Tantangan komputasi dan optimasi terbesar dalam melatih kebijakan pemangkasan token (*token pruning policy*) berbasis *Reinforcement Learning* (RL) adalah:
1. **Ledakan Ruang Aksi Kombinatorial:**  
   Pada setiap langkah sekuensial $t$, terdapat $K_t$ token kandidat ($K_0 = 576$ hingga ribuan token). Karena setiap token dapat dipangkas atau dipertahankan ($d_i \in \{0, 1\}$), ukuran ruang aksi pada satu langkah adalah $2^{K_t}$.
2. **Kegagalan Eksplorasi Acak (*Exploration Failure*):**  
   Melatih agen RL dari inisialisasi bobot acak (*RL from scratch*) dengan sinyal reward downstream yang jarang (*sparse*) menyebabkan varians gradien ekstrem, ketidakstabilan konvergensi, atau keruntuhan kebijakan (*policy collapse*) di mana agen membuang semua token atau mempertahankan semua token tanpa seleksi semantik.
3. **Keterbatasan Heuristik Statis:**  
   Metode kompresi heuristik non-RL (seperti Vispruner, FastV, ToMe) menawarkan titik awal yang stabil namun bersifat kaku (*task-agnostic*), gagal beradaptasi dengan kebutuhan semantik spesifik dari kueri hilir (*downstream query*), dan tidak memodelkan dependensi bertahap antar-langkah.

Oleh karena itu, tim arsitektur perlu menetapkan keputusan strategis: **Bagaimana merancang strategi pelatihan kebijakan pemangkasan token yang efisien sampel (*sample-efficient*), stabil secara numerik, dan mampu menyeimbangkan retensi akurasi tugas downstream dengan penghematan komputasi yang agresif?**

---

## 2. Decision Drivers

- **Stabilitas Konvergensi & Efisiensi Sampel:** Mencegah keruntuhan kebijakan dan varians gradien tinggi selama eksplorasi di ruang aksi kombinatorial $2^{K_t}$.
- **Near-Lossless Task Retention (Akurasi Drop $\le 0.7\%$):** Memastikan token yang tersisa mempertahankan informasi kritis untuk menjawab pertanyaan teks pengguna secara akurat.
- **Penghematan Komputasi Signifikan (Target Reduksi Token $\ge 50\% - 66.7\%$):** Menekan ukuran KV-Cache hingga $\ge 60\%$, mereduksi FLOPs inferensi hingga $\ge 50\%$, dan mempercepat waktu respons model.
- **Formulasi Reward Seimbang (Pareto Optimal):** Menghindari bias di mana agen hanya mengejar efisiensi (membuang semua token) atau hanya mengejar akurasi (tidak memangkas token sama sekali).
- **Kecepatan Inferensi & Minimal Parameter Overhead:** Modul kebijakan harus ringan, dapat dijalankan deterministik dalam satu langkah (*one-shot inference*), dengan overhead $<15$ ms.

---

## 3. Considered Options

### Option 1: Pure RL from Scratch with End-Task Sparse Reward
- Menginisialisasi policy network secara acak (Gaussian initialization).
- Agen mengeksplorasi trajektori pemangkasan token visual langsung menggunakan reward akhir dari skor evaluasi downstream LLM.
- *Karakteristik:* Bebas dari bias heuristik sebelumnya, namun membutuhkan jutaan langkah interaksi, sangat rentan terhadap *sparse reward failure*, dan menghasilkan kebijakan lokal yang buruk.

### Option 2: Static Supervised Heuristic Imitation (No RL Fine-Tuning)
- Melatih policy network hanya dengan *Behavior Cloning* (BC) meniru keputusan pemangkasan dari metode heuristik referensi (misal Vispruner) menggunakan loss *Binary Cross-Entropy*.
- *Karakteristik:* Pelatihan sangat cepat dan stabil, tetapi performa dibatasi (*ceiling effect*) oleh kelemahan heuristik referensi yang tidak sadar tugas downstream (*task-agnostic*), tanpa kemampuan adaptasi dinamis terhadap kueri teks.

### Option 3: Two-Phase Hybrid: Learning from Demonstrations (LfD) Warm-Start + Sequential PPO Fine-Tuning with Composite Incremental Reward (TPRL Paradigm - Selected)
- **Fase 1 (Supervised LfD Warm-Start):** Policy network diinisialisasi terlebih dahulu melalui supervised learning di atas trajektori demonstrasi heuristik bertahap ($0\% \to 25\% \to 50\%$ pruning mask) menggunakan $\mathcal{L}_{\text{BCE}}$. Ini menanamkan pemahaman awal tentang relevansi visual spasial.
- **Fase 2 (Sequential PPO Optimization):** Policy yang telah di-warm start kemudian dioptimalkan melalui *Proximal Policy Optimization* (PPO) multi-langkah ($T_{\max} = 3$) dengan fungsi reward diferensial:
  $$r_t = R_{\text{task}}(t) + R_{\text{eff}}(t)$$
  - $R_{\text{task}}(t) = \alpha \cdot \Delta \text{Score}_t$: Memberi insentif pada preservasi/peningkatan akurasi downstream.
  - $R_{\text{eff}}(t) = \beta \cdot (1 - K_t / K_{t-1})$: Memberi insentif pada reduksi rasio token per langkah.
- Menggunakan *step-wise exploration decay* ($\lambda_{\text{disc}}^t$) selama pelatihan RL, dan *deterministic thresholding* ($\mathbb{I}(p_i > \tau)$) saat inferensi produksi.

---

## 4. Evaluasi Komparatif & Bukti Empiris

| Dimensi Evaluasi | Option 1: RL from Scratch | Option 2: Supervised Imitation Only | Option 3: LfD + PPO Dual-Reward (TPRL - Selected) |
| :--- | :--- | :--- | :--- |
| **Stabilitas Pelatihan** | Sangat Rendah (Varians gradien ekstrem) | Sangat Tinggi (Supervised konvergen cepat) | **Tinggi & Terkontrol** (Warm-start mencegah divergensi) |
| **Kebutuhan Sampel Latih** | Sangat Besar ($>10\times$ interaksi) | Rendah | **Efisien** (50k demonstrasi + 200 epoch RL) |
| **Akurasi Downstream (VQAv2)** | 96.3% (Degradasi 3.7%) | 97.0% (Degradasi 3.0%) | **99.6% (Near-lossless, drop hanya 0.4%)** |
| **Akurasi Downstream (MMBench)** | 98.5% | 97.1% | **102.3% (Melampaui baseline tanpa pruning!)** |
| **Akurasi Downstream (MME)** | 96.7% | 95.0% | **99.4% (Preservasi persepsi visual tinggi)** |
| **Akurasi Downstream (POPE)** | 95.9% | 98.8% | **99.7% (Eliminasi halusinasi objek)** |
| **Adaptabilitas Semantik Kueri** | Acak / Tidak stabil | Buruk (Kaku terhadap aturan heuristik) | **Optimal** (Kueri teks memandu atensi seleksi) |
| **Kemampuan Menembus Ceiling Heuristik** | Gagal | Mustahil (Terbatas oleh data guru) | **Berhasil** (+2.6% s.d. +5.2% di atas Vispruner) |

### Bukti Empiris Kunci dari arXiv:2603.13394v1:

1. **Ablasi Inisialisasi Kebijakan (Tabel 6 Paper):**  
   - Menggunakan Vispruner statis murni menghasilkan skor rata-rata terbatas (MME: 95.0%, MMBench: 97.1%).
   - Melatih RL dari inisialisasi acak (*RL from scratch*) menghasilkan performa terendah di VQAv2 ($96.3\%$) dan POPE ($95.9\%$) karena agen tersesat di ruang pencarian $2^{K_t}$.
   - Kombinasi **LfD Warm-Start + PPO** melesat ke **$99.6\%$ (VQAv2)** dan **$102.3\%$ (MMBench)**, membuktikan bahwa LfD adalah prasyarat mutlak untuk keberhasilan RL pada kompresi token visual.
2. **Efisiensi Komputasi & KV-Cache (Tabel 4 Paper):**  
   - Pemangkasan ke 192 token ($66.7\%$ reduksi) menghemat memori KV-Cache dari **302.4 MB ke 100.9 MB (hemat 66.6%)**.
   - Latensi turun dari **57.8 ms ke 26.7 ms (speedup 53.8%)**.
   - Beban FLOPs terpangkas dari **9.6 T ke 4.4 T (reduksi 54.2%)**.
3. **Horizon Sekuensial Optimal (Tabel 7 Paper):**  
   Eksperimen variasi langkah $T_{\max} \in \{1, 2, 3, 4, 5\}$ menunjukkan bahwa $T_{\max} = 3$ memberikan performa puncak ($99.6\%$). Nilai $T_{\max} > 3$ memicu penurunan performa karena eror compounding pada estimasi state antar-langkah.

---

## 5. Decision Outcome & Rationale

Tim arsitektur memutuskan untuk mengadopsi **Option 3: Two-Phase Hybrid (Learning from Demonstrations Warm-Start + Sequential PPO Fine-Tuning with Composite Performance-Efficiency Reward)** sebagai arsitektur resmi modul kompresi token sekuensial pada **Lethon Stage 4**.

### Rationale Utama:

1. **Mengatasi Bottleneck Ruang Aksi Kombinatorial via LfD:**  
   Supervisi berbasis demonstrasi heuristik menyuntikkan *inductive bias* yang memangkas volume pencarian kebijakan secara drastis. Agen memulai RL bukan dari ketidaktahuan total, melainkan dari posisi yang sudah memiliki akurasi $\sim 97\%$, sehingga fase PPO murni difokuskan untuk *fine-tuning* adaptasi bahasa downstream.
2. **Formulasi Reward Diferensial Mencegah Reward Hacking:**  
   Penggunaan $\Delta \text{Score} = \text{Score}(H_t) - \text{Score}(H_{t-1})$ mengisolasi dampak langsung dari keputusan pemangkasan pada langkah $t$. Ditambah penalti/bonus efisiensi proporsional $\beta (1 - K_t / K_{t-1})$ dengan bobot moderat ($\beta = 0.1$), sistem tidak tergoda untuk membuang token sembarangan demi efisiensi instan.
3. **Penyaringan Noise Multimodal Menghasilkan Super-Baseline:**  
   Peningkatan skor MMBench dari 64.6 ke 66.1 ($102.3\%$) membuktikan bahwa RL yang dipandu kueri teks secara aktif membuang *visual clutter* dan artefak latar belakang yang selama ini membingungkan LLM, memberikan keuntungan akurasi ganda selain penghematan komputasi.
4. **Kompatibilitas Zero-Proxy & Deterministic Inference:**  
   Meskipun pelatihan RL bersifat probabilistik multi-langkah ($T_{\max} = 3$), saat inferensi produksi agen beroperasi dalam mode **satu langkah deterministik** melalui ambang batas probabilitas $\tau$ ($a_i = \mathbb{I}(p_i > \tau)$). Hal ini menjamin latensi inferensi pruner tetap stabil di bawah **$1.5$ ms**, memenuhi standar *Zero-Proxy MCP*.

---

## 6. Consequences & Implementasi Teknis

### Dampak Positif:
- **Penghematan Token Masif:** Mampu memangkas hingga $66.7\%$ token visual/konteks awal dengan degradasi akurasi rata-rata $<0.7\%$.
- **Pengurangan KV-Cache:** Menurunkan konsumsi VRAM KV-cache hingga $66.6\%$, melipatgandakan throughput konkurensi pada server serving.
- **Sinergi Pipeline Penuh:** Melengkapi Stage 1 (FFT Noise Removal), Stage 2 (SAC Anchors), dan Stage 3 (RLPT Subspace Masking), serta berjalan paralel dengan pemadatan CoT ThinkPrune pada model reasoning.

### Mitigasi & Pertimbangan Teknis:
- **Ketergantungan Data Demonstrasi:** Memerlukan dataset pasangan gambar-teks dengan label demonstrasi heuristik (50K trajektori) untuk fase pre-training LfD. *Mitigasi:* Menggunakan dataset terbuka (GQA, MMBench, VQAv2) dengan pipeline ekstraksi otomatis berbasis Vispruner.
- **Kalibrasi Ambang Batas $\tau$:** Pengguna MCP Tool `prune_context` dapat memilih mode preservasi:
  - Mode `CONSERVATIVE`: $\tau = 0.55$ ($\sim 66.7\%$ pruning rate, target akurasi $99.3\%$).
  - Mode `BALANCED`: $\tau = 0.67$ ($\sim 75.0\%$ pruning rate, target akurasi $98.4\%$).
  - Mode `AGGRESSIVE`: $\tau = 0.74$ ($\sim 77.8\%$ pruning rate, target akurasi $97.7\%$).
- **Fallback Minimum Token:** Jika seluruh probabilitas retensi berada di bawah $\tau$, sistem secara deterministik mempertahankan minimal 1 token dengan probabilitas retensi tertinggi untuk mencegah input kosong (*empty sequence error*).
