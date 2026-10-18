# ADR-0007: Semantic Anchor Aggregation (SAC) vs Rigid Autoencoders and Heuristic Token Pruning for Context Compression

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture Guild & Algorithmic Core Team
- **Consulted Literature:**
  - `arXiv:2510.08907`: *Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors* (Liu et al., ICLR 2026)
  - `arXiv:2307.06945`: *In-context Autoencoder for Context Compression in a Large Language Model (ICAE)* (Ge et al., 2024)
  - `arXiv:2403.12968`: *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression* (Pan et al., ACL 2024)
  - `arXiv:2409.14364`: *Position IDs Matter: An Enhanced Position Layout for Efficient Context Compression (EPL)* (Zhao et al., 2025)
  - `PRD.md`: *Lethon System Requirements & 4-Stage Architecture (Stage 2 Semantic Aggregation)*
  - `docs/dec/dec-02-semantic-anchor-aggregation.md`: *Technical & Mathematical Specifications of Stage 2*

---

## 1. Context & Problem Statement

Pada arsitektur engine kompresi konteks Lethon ([PRD.md](../../PRD.md)), setelah teks panjang atau jejak *Chain-of-Thought* (CoT) dibersihkan dari noise frekuensi tinggi oleh **Stage 1 (Fast Spectral Fourier Filtering)**, sistem memerlukan mekanisme kompresi semantik berkepadatan tinggi pada **Stage 2** untuk memadatkan konteks dengan rasio kompresi agresif (**$5\times$ hingga $51\times$**) sebelum diteruskan ke **Stage 3 (Dynamic Subspace Masking - RLPT)** dan **Stage 4 (CoT MDP Truncation)**.

Dalam literatur rekayasa kompresi konteks LLM, terdapat dua paradigma dominan:
1. **Rigid In-Context Autoencoders (Soft Compression):** Menambahkan token kompresi khusus acak (*special soft tokens*) di akhir konteks dan melatihnya menggunakan fungsi objektif *Autoencoding* (AE) untuk merekonstruksi seluruh token dokumen kata per kata (misal ICAE, 500xCompressor).
2. **Heuristic Hard Token Pruning:** Memangkas dan membuang token-token tertentu secara permanen berdasarkan estimasi entropi model kecil (LLMLingua-2), threshold skor atensi, atau pemilihan acak tanpa memperbarui representasi token yang tersisa.

Namun, pengujian pada skenario penalaran panjang (*complex QA*, *multi-hop reasoning*, dan *codebase synthesis*) mengungkapkan kegagalan mendasar dari kedua paradigma ini:
* **Autoencoder Kaku** membebani kapasitas token pembawa dengan keharusan menghafal tanda baca dan detail sintaksis sepele, memicu **konflik ortogonalitas gradien** terhadap tugas penalaran (*gradient orthogonality*), serta membutuhkan fase *pretraining* yang mahal dan lambat.
* **Hard Token Pruning** merusak integritas sintaksis dokumen, memicu hilangnya entitas kunci secara ireversibel, dan tidak mampu melakukan sintesis informasi antar-kalimat karena token yang tersisa tidak disesuaikan untuk mewakili token yang terbuang.

Oleh karena itu, arsitektur Lethon harus menetapkan: Pendekatan kompresi representasi apa yang paling optimal untuk Stage 2 guna menjamin retensi akurasi penalaran downstream tertinggi dengan latensi dan biaya komputasi minimal?

---

## 2. Decision Drivers

- **Preservasi Akurasi Penalaran Kompleks ($< 0.7\%$ Degradasi):** Mempertahankan integritas semantik untuk tugas penalaran multi-hop, *long-context question answering*, dan peringkasan dokumen panjang.
- **Eliminasi Misalignment Pelatihan Autoencoding:** Menghindari interferensi optimasi akibat tugas rekonstruksi literal yang ortogonal terhadap pemodelan bahasa task-specific.
- **Kesesuaian Topologi Manifold Representasi (Zero Distribution Shift):** Memastikan representasi terkompresi (Key-Value pairs) berada di dalam manifold distribusi token konteks asli sehingga dapat dipahami langsung oleh target LLM tanpa proyeksi rumit.
- **Efisiensi FLOPs & Eliminasi Padding Sekuens:** Menghindari penambahan panjang sekuens input ($s + k$) yang memperberat komputasi atensi kuadratik transformer.
- **Kompatibilitas Mulus dengan Stage 3 (RLPT Subspace Masking):** Representasi terkompresi harus menyediakan basis kovarians yang stabil untuk proyeksi subspace ber-rank rendah (*low-rank subspace projection*).

---

## 3. Considered Options

### Option 1: Rigid In-Context Autoencoder (ICAE / 500xCompressor Paradigm)
- Mengalokasikan $k$ token khusus acak di akhir sekuens konteks.
- Melatih model dengan kombinasi loss rekonstruksi $\mathcal{L}_{AE} = -\log P(C \mid \tilde{M})$ dan pemodelan bahasa $\mathcal{L}_{LM}$.
- Mengekstraksi hidden state atau KV cache dari $k$ token khusus tersebut.

### Option 2: Heuristic Hard Token Pruning (LLMLingua-2 / Entropy Thresholding)
- Menggunakan classifier token terdistilasi atau model bahasa kecil kausal (SLM) untuk menghitung skor entropi atau probabilitas retensi setiap token.
- Membuang token berprobabilitas rendah secara permanen dan menyambungkan kembali token yang tersisa sebagai teks mentah.
- Tanpa pelatihan representasi adaptif.

### Option 3: Semantic Anchor Aggregation (SAC Architecture - Selected)
- Memilih subset token representatif langsung dari dokumen asli ($S \subseteq C$) sebagai *anchor tokens* menggunakan pembagian blok berjarak sama (*uniform chunk-centered selection*).
- Menandai token anchor dengan **Learnable Anchor Embedding ($e_A$)** aditif: $e_i = \text{Emb}(c_i) + \mathbb{I}(c_i \in S) \cdot e_A$.
- Mengganti *causal attention mask* dengan **Full Bidirectional Attention Mask** ($M_{i,j} = 0$) pada encoder kompresi.
- Melatih sistem secara **Autoencoding-Free** (hanya dengan $\mathcal{L}_{LM}$ dan $\mathcal{L}_{\text{task}}$, membuang total $\mathcal{L}_{AE}$).
- Mengekstraksi representasi Key-Value multi-layer pada indeks anchor untuk disisipkan sebagai prefix KV cache pada target LLM.

---

## 4. Evaluasi Komparatif & Trade-off Analysis

| Dimensi Evaluasi | Option 1: Rigid Autoencoder (ICAE/500x) | Option 2: Hard Token Pruning (LLMLingua-2) | Option 3: Semantic Anchor (SAC - Lethon) |
| :--- | :--- | :--- | :--- |
| **In-Domain QA (MRQA Avg F1 @ 15x)** | 44.50 – 49.35 | 31.83 (hanya mampu 5x) | **54.95** ($\mathbf{+5.60 \text{ hingga } +10.45}$) |
| **Out-of-Domain Generalization (F1)**| 30.81 – 34.04 | 27.90 | **39.26** ($\mathbf{+5.22 \text{ hingga } +8.45}$) |
| **Biaya Pretraining & Stabilitas** | Boros (Konflik gradien $\mathcal{L}_{AE} \perp \mathcal{L}_{LM}$) | Ringan (Hanya inference SLM) | **Efisien (26%–31% lebih cepat, tanpa AE)** |
| **Penyelarasan Manifold KV (t-SNE)** | Buruk (Kluster terpisah & RoPE mismatch) | N/A (Teks terpotong-potong) | **Sempurna (Menyatu seragam dengan konteks)** |
| **Komputasi Sekuens (FLOPs)** | Tinggi (Beban sekuens bertambah $s + k$) | Rendah | **Optimal ($1.19\times$ lebih hemat dari 500x)** |
| **Integritas Konteks Panjang** | Rentan hilang fokus pada kompresi tinggi | Rusak parah (Sintaks dan relasi putus) | **Sangat Baik (Focused attention maps)** |
| **Kompatibilitas Stage 3 (RLPT)** | Rendah (Varians embedding acak tinggi) | Rendah (Ruang diskrit tidak terdiferensiasi) | **Sangat Tinggi (Representasi kontinu mulus)** |

---

## 5. Decision Outcome & Rationale

Tim arsitektur memutuskan untuk memilih **Option 3: Semantic Anchor Aggregation (SAC)** sebagai fondasi teknis untuk **Stage 2 Context Compression Lethon**.

### Rationale Utama:

1. **Resolusi Konflik Ortogonalitas Gradien (arXiv:2510.08907):**  
   Pengukuran gradien membuktikan bahwa arah penurunan loss rekonstruksi literal ($\nabla \mathcal{L}_{AE}$) dan loss pemahaman bahasa ($\nabla \mathcal{L}_{LM}$) saling tegak lurus ($\cos(\theta) \approx 0$). Menghapus $\mathcal{L}_{AE}$ sepenuhnya membebaskan kapasitas model untuk fokus mengabstraksi konsep semantik penting, menghasilkan perplexity bahasa terbaik ($10.79$ vs $12.35$ pada ICAE).
2. **Keunggulan Inherent Semantic Priors vs Token Acak:**  
   Memanfaatkan token yang sudah ada di dalam konteks sebagai wadah kompresi memberikan keuntungan *natural prior* bahasa, mengeliminasi kebutuhan fase pemanasan (*cold start*) yang memakan ribuan jam GPU. Injeksi vektor $e_A$ secara terbukti cukup memberikan sinyal pembeda yang tegas kepada transformer.
3. **Penyelarasan Sempurna pada Manifold Representasi Key-Value:**  
   Visualisasi t-SNE membuktikan bahwa representasi KV dari anchor SAC menyebar merata (*uniformly distributed*) pada ruang fitur konteks asli tanpa membentuk kluster anomali di batas luar. Target LLM dapat mengonsumsi representasi ini secara instan layaknya membaca konteks aslinya sendiri (*zero semantic dissonance*).
4. **Keunggulan Bidirectional Cross-Visibility:**  
   Berbeda dengan arsitektur autoregresif yang membatasi pandangan token hanya ke belakang, atensi dua arah memungkinkan anchor di awal dokumen merangkum kesimpulan di akhir dokumen, mengeliminasi sindrom *"lost-in-the-middle"*.
5. **Efisiensi Algoritmik & Eliminasi Token Padding:**  
   SAC tidak menambahkan $k$ token khusus ke dalam sekuens komputasi atensi. Hal ini memangkas FLOPs sebesar $1.19\times$ pada rasio $5\times$ dan mempercepat latensi inferensi kompresi ($243.87\text{ ms}$ vs $257.43\text{ ms}$ per batch).

---

## 6. Implementation Guidelines & Architectural Constraints

1. **Penetapan Skema Seleksi Default:**  
   Implementasi pada `lethon.core.anchor` wajib menggunakan **Uniform Chunk-Centered Selection** sebagai algoritma pemilihan indeks default:
   $$i_k = \left\lfloor \left(k + \frac{1}{2}\right) \cdot \frac{L}{M} \right\rfloor$$
   karena telah terbukti secara empiris memberikan cakupan global terbaik dengan margin stabilitas tertinggi.
2. **Larangan Penambahan Loss Autoencoding ($\mathcal{L}_{AE}$):**  
   Pipeline pelatihan Lethon Stage 2 dilarang menyertakan fungsi loss rekonstruksi token literal $\mathcal{L}_{AE}$. Pelatihan hanya menggunakan kombinasi $\mathcal{L}_{LM}$ (prediktif masa depan) dan $\mathcal{L}_{\text{task}}$ (downstream objective).
3. **Bidirectional Masking Enforcement:**  
   Saat forward pass kompresi dilakukan pada encoder, matriks atensi wajib diatur ke `torch.zeros((1, 1, L, L))` (unmasked) agar seluruh token saling berinteraksi dua arah.
4. **Layer-wise KV Extraction:**  
   Representasi yang disimpan dan diteruskan ke Stage 3 adalah pasangan tensor Key-Value di setiap layer transformer pada posisi indeks anchor: $\{(\tilde{K}^{(l)}, \tilde{V}^{(l)})\}_{l=1}^{N_{\text{layers}}}$.
5. **Ukuran Sub-Konteks Maksimal (510 Tokens):**  
   Untuk mempertahankan resolusi atensi lokal dan stabilitas komputasi, teks masukan dipartisi menjadi segmen-segmen sub-konteks berukuran 510 token sebelum diagregasikan ke anchor tokens masing-masing.
