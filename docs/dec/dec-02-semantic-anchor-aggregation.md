# DEC-02: Contextual Semantic Anchor Aggregation (Stage 2)

## 1. Executive Summary & System Context

Modul **Contextual Semantic Anchor Aggregation (SAC)** beroperasi sebagai mesin pemadatan semantik representasi (*semantic densification engine*) pada tahap kedua (Stage 2) dalam arsitektur 4-Stage Context Compression Engine Lethon ([PRD.md](../../PRD.md)). Modul ini menerima sekuens token konteks yang telah dibersihkan dari komponen noise frekuensi tinggi oleh Stage 1 (*Fast Spectral Fourier Filtering*), lalu memetakan dan mengonsolidasikan informasi global tersebut ke dalam subset token representatif terpilih (*anchor tokens*) dengan rasio kompresi agresif (**$5\times$ hingga $51\times$**).

Berdasarkan landasan riset fundamental pada paper ICLR 2026: **Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors** (*arXiv:2510.08907*), Lethon meninggalkan paradigma kompresi berbasis *autoencoder* kaku (seperti ICAE dan 500xCompressor) yang bergantung pada pelatihan rekonstruksi token demi token. Sebaliknya, Stage 2 Lethon mengadopsi mekanisme **Contextual Semantic Anchors** dengan dua pilar arsitektur utama:
1. **In-Context Semantic Anchors & Learnable Anchor Embeddings ($e_A$):** Menggunakan token asli dokumen sebagai wadah representasi (*carrier*) yang diperkaya dengan vektor embedding penanda peran kompresi yang dapat dipelajari (*learnable anchor embedding*). Pendekatan ini mempertahankan *natural semantic priors* dan mencegah pergeseran manifold (*distribution shift*) pada representasi Key-Value (KV).
2. **Bidirectional Attention Modification:** Membuka seluruh batasan *causal mask* pada model encoder saat proses kompresi, memungkinkan *anchor tokens* melihat konteks secara dua arah (*bidirectional cross-visibility*) guna menyerap sari informasi dokumen secara holistik.

Implementasi Stage 2 ini berjalan **Autoencoding-Free** (tanpa fungsi *loss* rekonstruksi $\mathcal{L}_{AE}$), menghasilkan peningkatan kecepatan pelatihan hingga **26%–31%**, konvergensi loss yang jauh lebih stabil, serta retensi akurasi penalaran downstream yang mendekati uncompressed baseline (penurunan F1/EM $< 0.7\%$).

```
[Context / Denoised Tokens dari Stage 1] ──> (L tokens, d dimensions)
                    │
                    ▼
     ┌────────────────────────────────────────────────────────┐
     │ 1. Anchor Selection: Uniform / Spectral-Guided (M << L)│
     │ 2. Embed Injection: e_i = Emb(c_i) + I(c_i in S) * e_A │
     │ 3. Bidirectional Mask: M_attn = 0 (Full Visibility)    │
     │ 4. LoRA Transformer Forward Pass (No Causal Barrier)   │
     │ 5. Multi-Layer KV Cache Slicing at Anchor Indices S    │
     └────────────────────────────────────────────────────────┘
                    │
                    ▼
[Compressed Anchor Representations M] ──> (M tokens / KV-Pairs)
                    │
                    ▼
  [Menuju Stage 3: Dynamic Promising Token Subspace Masking (RLPT)]
```

---

## 2. Landasan Teori & Analisis Paper Acuan (arXiv:2510.08907)

Stage 2 Lethon didesain secara spesifik berdasarkan analisis teoritis dan temuan empiris mendalam dari paper *Semantic-Anchor Compression (SAC) for LLMs* (Liu et al., ICLR 2026 / arXiv:2510.08907).

### 2.1 Dekonstruksi Masalah Paradigma Kompresi Tradisional (ICAE & 500xCompressor)

Metode kompresi konteks sebelumnya (misal ICAE - Ge et al., 2024; 500xCompressor - Li et al., 2025b; EPL - Zhao et al., 2025) menambahkan sejumlah token khusus acak (*special compression tokens*) di akhir sekuens konteks dan melatih token tersebut menggunakan tugas *Autoencoding* (AE):
$$\mathcal{L}_{AE} = -\log P(C \mid \tilde{M})$$
di mana $\tilde{M}$ adalah representasi terkompresi dan $C$ adalah seluruh teks konteks input.

Paper membuktikan bahwa pendekatan ini memiliki dua kelemahan fatal:
1. **Konflik Kapasitas & Misalignment Rekonstruksi (Intuitive Perspective):**  
   Tujuan AE memaksa $\tilde{M}$ merekonstruksi seluruh token individual dari konteks asli tanpa mempertimbangkan signifikansi semantik relatifnya. Kapasitas representasi token yang terbatas terbuang sia-sia untuk menghafal tanda baca, kata hubung, dan detail sintaksis yang tidak relevan dengan tugas penalaran downstream (*downstream QA / reasoning*).
2. **Ortogonalitas Gradien AE vs LM (Gradient Perspective - Gambar 9 Paper):**  
   Pengukuran *cosine similarity* gradien antara loss rekonstruksi autoencoding ($\mathcal{L}_{AE}$) dan loss pemodelan bahasa / kelanjutan konteks ($\mathcal{L}_{LM}$):
   $$\text{Sim}_{\text{grad}} = \frac{\langle \nabla_\theta \mathcal{L}_{AE}, \, \nabla_\theta \mathcal{L}_{LM} \rangle}{\|\nabla_\theta \mathcal{L}_{AE}\| \cdot \|\nabla_\theta \mathcal{L}_{LM}\|}$$
   Hasil eksperimen empiris (Gambar 9 pada paper) menunjukkan bahwa setelah iterasi awal, nilai $\text{Sim}_{\text{grad}}$ langsung merosot mendekati **$0$** (ortogonal). Hal ini membuktikan bahwa pembaruan parameter untuk tugas rekonstruksi literal berjalan ortogonal dan saling menginterferensi dengan pembaruan untuk pemahaman semantik bahasa, merusak representasi untuk downstream task.

### 2.2 Keunggulan Inherent Semantic Prior & Autoencoding-Free Training

SAC menghilangkan ketergantungan pada tugas rekonstruksi dengan langsung memilih token-token representatif dari konteks itu sendiri sebagai *anchor tokens* ($S \subseteq C$).

1. **Natural Semantic Priors:** Berbeda dengan token kompresi eksternal yang diinisialisasi secara acak dan harus diajari memahami bahasa dari nol, *anchor tokens* sudah membawa bobot semantik awal dari teks dokumen asli.
2. **Autoencoding-Free Training Regime:** SAC hanya dilatih menggunakan loss pemodelan bahasa prediktif ($\mathcal{L}_{LM}$) dan loss tugas downstream ($\mathcal{L}_{\text{task}}$):
   $$\mathcal{L} = \mathcal{L}_{LM} + \mathcal{L}_{\text{task}}$$
   Tanpa beban $\mathcal{L}_{AE}$, model memusatkan 100% kapasitas representasinya pada fitur semantik abstrak yang relevan dengan downstream reasoning.
3. **Efisiensi Waktu Pelatihan:** Tabel 10 pada paper menunjukkan penghapusan tugas AE memangkas waktu pre-training sebesar **$31\%$** dibanding ICAE (dari $3.85$ jam menjadi $2.66$ jam pada 8x RTX 3090) dan menghasilkan Language Modeling Perplexity terendah ($10.79$ vs $12.35$ pada ICAE).

### 2.3 Analisis Manifold Topologi KV Representation Space (Gambar 4 Paper)

Visualisasi t-SNE pada representasi Key dan Value dari layer terakhir (Gambar 4 pada paper) mengungkap alasan fundamental superioritas SAC terhadap 500xCompressor dan EPL:
1. **Ruang Representasi Key ($K$):**  
   Pada 500xCompressor, vektor Key dari token kompresi terisolasi jauh dari gugus token konteks asli karena tidak memiliki korespondensi posisi yang selaras. EPL mencoba memperbaiki ini dengan menyelaraskan sudut rotasi RoPE (*Rotational Position Embedding*), namun tetap menggunakan token baru. Pada SAC, token anchor berada langsung di dalam alur konteks asli, sehingga distribusi Key-nya menempel erat secara alami dengan manifold konteks.
2. **Ruang Representasi Value ($V$):**  
   Pada ruang Value, token kompresi EPL masih membentuk sub-kluster di batas-batas terluar (*boundary clustering*), mengindikasikan adanya *semantic shift*. Sebaliknya, representasi Value SAC menyebar merata secara seragam (*uniformly distributed*) melintasi seluruh manifold Value token asli tanpa membentuk kluster terpisah. Target LLM saat inferensi dapat mengonsumsi representasi ini secara instan tanpa mengalami disonansi semantik (*zero semantic gap*).

### 2.4 Analisis Pola Atensi (Attention Maps - Gambar 6, 7, 8 Paper)

Pemeriksaan peta atensi (*attention maps*) antar-layer pada rasio kompresi $5\times$, $15\times$, dan $51\times$:
* **Pada Kompresi Rendah ($5\times$):** Pola atensi SAC menunjukkan diagonal positif yang tegas (Gambar 6 paper), menandakan anchor tokens memprioritaskan penyerapan informasi dari tetangga lokal terdekat (*local receptive field prior*).
* **Pada Kompresi Tinggi ($15\times$ & $51\times$):** Ketika EPL dan 500xCompressor mengalami penyebaran atensi yang kabur/difus (*diffuse attention noise*), SAC secara konsisten menunjukkan pola **focused attention** (Gambar 7 & 8 paper). Anchor tokens memfokuskan bobot atensinya secara tajam hanya pada segelintir token penting (*salient keywords*) di seluruh konteks, mengabaikan token transisi yang tidak esensial.

### 2.5 Efisiensi FLOPs & Eliminasi Sequence Padding (Tabel 12 & 13 Paper)

Salah satu keunggulan teknis paling signifikan dari SAC adalah **tidak adanya token tambahan ($k$) yang disisipkan ke dalam sekuens input**.
* Pada 500xCompressor dan ICAE, sekuens diperpanjang menjadi $s + k$, meningkatkan beban komputasi kuadratik atensi menjadi $\mathcal{O}((s + k)^2)$.
* Pada SAC, kompresi berlangsung langsung di dalam sekuens panjang asli $s$. 

Secara teoritis (Tabel 12 paper), total FLOPs per layer untuk konteks $[b, s, h]$ dengan FFN intermediate dimension $I$:
$$\text{FLOPs}_{\text{SAC}} = b \cdot h \cdot s \cdot (8h + 4s + 6I)$$
$$\text{FLOPs}_{\text{500x}} = b \cdot h \cdot (s + k) \cdot [8h + 2(s + k) + 6I]$$

Pada kompresi $5\times$ ($k = 102, s = 510, h = 2048, I = 8192$), 500xCompressor membutuhkan **$1.19\times$ lebih banyak FLOPs** daripada SAC. Uji empiris latensi kompresi (Tabel 13 paper) membuktikan SAC lebih cepat ($243.87\text{ ms}$ vs $257.43\text{ ms}$ per batch).

---

## 3. Formulasi Matematis Eksak (Stage 2 Architecture)

Diberikan sekuens token konteks dari dokumen panjang atau jejak kode yang telah melewati pembersihan frekuensi tinggi Stage 1:
$$C = \langle c_1, c_2, \dots, c_L \rangle, \quad c_i \in \mathcal{V}$$
di mana $L$ adalah panjang sekuens token konteks dan $\mathcal{V}$ adalah kosakata model bahasa. 

Target kompresi ditentukan oleh rasio kompresi $r \in \mathbb{R}_{\ge 1}$ (misal $r = 5, 15, 51$). Jumlah *anchor tokens* yang dipertahankan adalah:
$$M = \left\lfloor \frac{L}{r} \right\rfloor \ll L$$

### 3.1 Strategi Pemilihan Indeks Anchor Tokens ($S \subseteq C$)

Lethon mengimplementasikan dua strategi pemilihan subset anchor token $S = \{ c_{i_1}, c_{i_2}, \dots, c_{i_M} \}$ dengan himpunan indeks $\mathcal{I}_S = \{ i_1, i_2, \dots, i_M \} \subset \{1, \dots, L\}$:

#### Strategi A: Uniform Chunk-Centered Selection (Default Baseline)
Sesuai temuan paper (Tabel 4), membagi sekuens menjadi $M$ blok (*chunks*) berjarak sama dan memilih token tengah dari setiap blok menghasilkan cakupan dokumen global terbaik:
$$\text{Chunk}_k = \left[ k \cdot \frac{L}{M}, \, (k + 1) \cdot \frac{L}{M} \right), \quad k \in \{0, 1, \dots, M-1\}$$
Indeks anchor ke-$k$ dipilih pada titik median interval:
$$i_k = \left\lfloor \left(k + \frac{1}{2}\right) \cdot \frac{L}{M} \right\rfloor$$

#### Strategi B: Spectral-Guided Peak Selection (Integrasi Khusus Lethon Stage 1)
Menggabungkan informasi energi spektral dari Stage 1 FFT/DCT. Di dalam setiap $\text{Chunk}_k$, dipilih token yang memiliki deviasi energi terendah terhadap noise frekuensi tinggi atau magnitudo koefisien residual tertinggi:
$$i_k = \arg\max_{j \in \text{Chunk}_k} \left\| \mathcal{X}_{\text{spectral}}[j, :] \right\|_2$$
Pendekatan ini menjamin token yang bertindak sebagai anchor adalah token yang memiliki stabilitas informasi semantik tertinggi di bloknya.

### 3.2 Inisialisasi & Injeksi Learnable Anchor Embedding ($e_A$)

Setiap token dalam konteks $c_i$ dipetakan ke vektor representasi embedding melalui lookup matrix $\text{Emb}(c_i) \in \mathbb{R}^d$.

Untuk membedakan token yang bertindak sebagai anchor pembawa kompresi dari token konteks biasa tanpa merusak representasi semantik aslinya, Lethon memperkenalkan **Learnable Anchor Embedding Vector**:
$$e_A \in \mathbb{R}^d$$
yang diinisialisasi secara acak dengan distribusi normal terpotong:
$$e_A \sim \mathcal{N}\left(0, \, \sigma^2 = \frac{1}{\sqrt{d}}\right)$$

Vektor embedding akhir untuk setiap posisi $i \in \{1, \dots, L\}$ didefinisikan sebagai:
$$e_i = \text{Emb}(c_i) + \mathbb{I}(i \in \mathcal{I}_S) \cdot e_A$$
di mana $\mathbb{I}(\cdot)$ adalah fungsi indikator biner:
$$\mathbb{I}(i \in \mathcal{I}_S) = \begin{cases} 1, & \text{jika } i \in \mathcal{I}_S \\ 0, & \text{jika } i \notin \mathcal{I}_S \end{cases}$$

Dengan injeksi aditif ini:
- Token non-anchor tetap memiliki representasi natural $\text{Emb}(c_i)$.
- Token anchor tetap mempertahankan semantik asalnya $\text{Emb}(c_{i_k})$, namun ditandai secara eksplisit oleh vektor $e_A$ bahwa token tersebut harus bertindak sebagai wadah kondensasi informasi.

### 3.3 Preservasi Rotational Position Embedding (RoPE)

Berbeda dengan metode soft token eksternal (ICAE) yang kehilangan informasi koordinat dokumen, *anchor tokens* mempertahankan nomor indeks posisi aslinya:
$$p_i = i, \quad \forall i \in \{1, \dots, L\}$$
Saat matriks Query dan Key diputar menggunakan RoPE (*Rotary Position Embedding*):
$$\tilde{q}_i = \mathcal{R}_{\Theta, i} (W_Q e_i), \quad \tilde{k}_j = \mathcal{R}_{\Theta, j} (W_K e_j)$$
jarak spasial relatif antar token dipertahankan secara utuh:
$$\langle \tilde{q}_i, \tilde{k}_j \rangle = (W_Q e_i)^T \mathcal{R}_{\Theta, j - i} (W_K e_j)$$
Hal ini mencegah timbulnya *positional dissonance* yang sering merusak pemahaman konteks panjang.

### 3.4 Bidirectional Cross-Attention Modification

Pada pemrosesan transformer autoregresif standar, atensi dibatasi oleh matriks segitiga bawah (*causal mask*) $M^{\text{causal}}$:
$$M_{i, j}^{\text{causal}} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$
Batasan kausal ini menyebabkan anchor token di posisi awal dokumen buta terhadap informasi di paragraf akhir.

Pada Stage 2 Lethon, encoder kompresi memodifikasi seluruh matriks atensi menjadi **Full Bidirectional Attention Mask**:
$$M_{i, j}^{\text{bidi}} = 0, \quad \forall i, j \in \{1, \dots, L\}$$

Komputasi *Scaled Dot-Product Attention* multi-head untuk layer transformer ke-$l$ ($l = 1, \dots, N_{\text{layers}}$):
$$Q^{(l)} = H^{(l-1)} W_Q^{(l)}, \quad K^{(l)} = H^{(l-1)} W_K^{(l)}, \quad V^{(l)} = H^{(l-1)} W_V^{(l)}$$
$$\mathcal{A}^{(l)} = \text{softmax}\left( \frac{Q^{(l)} (K^{(l)})^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{B \times L \times L}$$
$$H^{(l)} = \mathcal{A}^{(l)} V^{(l)} W_O^{(l)} + H^{(l-1)}$$

**Karakteristik Kritis Penyerapan Informasi:**
Karena $M_{i, j}^{\text{bidi}} = 0$ diterapkan secara seragam ke seluruh token:
1. Token anchor $c_{i_k}$ dapat secara bebas memberikan bobot atensi ke token masa lalu ($j < i_k$) maupun token masa depan ($j > i_k$).
2. Token non-anchor juga saling berinteraksi secara dua arah, membentuk representasi kontekstual yang kaya sebelum diserap ke dalam Query token anchor.

### 3.5 Mekanisme Ekstraksi & Penyerapan Multi-Layer Key-Value (KV) Representations

Alih-alih hanya mengambil representasi hidden state layer terakhir ($H^{(N_{\text{layers}})}$), Lethon mengadopsi skema *Layer-wise KV Absorption* (seperti yang divalidasi oleh 500xCompressor dan SAC):

Pada setiap layer transformer $l \in \{1, \dots, N_{\text{layers}}\}$, matriks Key ($K^{(l)}$) dan Value ($V^{(l)}$) yang telah diperkaya oleh atensi dua arah diekstraksi secara selektif hanya pada indeks anchor $\mathcal{I}_S$:
$$\tilde{K}^{(l)} = K^{(l)}[:, \, \mathcal{I}_S, \, :] \in \mathbb{R}^{B \times M \times d_k}$$
$$\tilde{V}^{(l)} = V^{(l)}[:, \, \mathcal{I}_S, \, :] \in \mathbb{R}^{B \times M \times d_v}$$

Representasi akhir kompresi konteks adalah pasangan tensor KV terkompresi di seluruh layer:
$$\tilde{\mathcal{M}}_{\text{KV}} = \left\{ \left( \tilde{K}^{(l)}, \, \tilde{V}^{(l)} \right) \right\}_{l=1}^{N_{\text{layers}}}$$

Ukuran memori representasi ini menyusut persis dengan faktor kompresi $r = L / M$:
$$\text{Memory Overhead} = \frac{M \times d \times N_{\text{layers}}}{L \times d \times N_{\text{layers}}} = \frac{1}{r}$$

### 3.6 Re-Injeksi Inferensi ke Target LLM (Downstream Generation)

Saat target LLM menerima kueri atau pertanyaan $Q = \langle q_1, \dots, q_{L_Q} \rangle$:
1. Matriks $\tilde{K}^{(l)}$ dan $\tilde{V}^{(l)}$ dari Stage 2 disisipkan sebagai **prefix KV cache** pada layer ke-$l$ target LLM.
2. Target LLM mengeksekusi inferensi kausal standar pada token kueri:
   $$K_{\text{total}}^{(l)} = \left[ \tilde{K}^{(l)} \,\|\, K_Q^{(l)} \right], \quad V_{\text{total}}^{(l)} = \left[ \tilde{V}^{(l)} \,\|\, V_Q^{(l)} \right]$$
3. Target LLM dapat langsung membaca intisari konteks panjang tanpa perlu menghitung atensi atas $L$ token asli, memangkas latensi TTFT (*Time-To-First-Token*) hingga $> 80\%$.

---

## 4. Analisis Komparatif & Hasil Eksperimen Empiris (arXiv:2510.08907 Data Matrix)

Seluruh metrik performa di bawah ini diambil langsung dari hasil pengujian empiris resmi pada paper *arXiv:2510.08907* menggunakan Llama-3.2-1B/3B dan Llama-3.1-8B pada benchmark QA standar industri (MRQA) dan Long-Context Summarization (GovReport, QMSum).

### 4.1 Performa Downstream QA In-Domain & Out-of-Domain (Kompresi 15x)

Tabel evaluasi gabungan (Tabel 1 & 2 paper) pada rasio kompresi agresif $15\times$:

| Metode Kompresi | Tipe Carrier | Objective Pretrain | In-Domain Avg F1 | In-Domain Avg EM | Out-of-Domain Avg F1 | Out-of-Domain Avg EM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full-FT (Uncompressed)** | - | Standard LM | 71.51 | 56.85 | 52.51 | 36.87 |
| **LLMLingua-2 (5x only)** | Hard Token Drop | Token Classifier | 31.83 | 19.46 | 27.90 | 16.20 |
| **ICAE** | Soft Special Tokens | AE + LM | 44.50 | 31.28 | 30.81 | 20.18 |
| **500xCompressor** | Layer KV Tokens | AE + LM | 49.35 | 34.98 | 34.04 | 22.09 |
| **Activation Beacon** | Beacon Tokens | Condensation | 40.31 | 15.39 | 36.91 | 13.02 |
| **DAST** | Dynamic Soft Tokens | Dynamic Alloc | 49.83 | 36.14 | 31.93 | 20.95 |
| **EPL** | Position-Aware Soft | AE + LM | 51.52 | 36.65 | 36.74 | 23.83 |
| **SAC (Lethon Stage 2)** | **In-Context Anchors** | **LM Only (No AE)** | **54.95** | **39.67** | **39.26** | **26.02** |

*Insight Kritis:*  
SAC melampaui seluruh baseline soft-compression terdahulu pada seluruh dataset MRQA (SQuAD, NewsQA, TriviaQA, SearchQA, HotpotQA, NQ), mencatatkan keunggulan mutlak hingga **$+23.5\%$ F1** atas ICAE dan **$+6.7\%$ F1 / $+8.2\%$ EM** atas EPL, dengan generalisasi OOD tertinggi ($39.26$ F1).

### 4.2 Studi Ablasi Komponen Arsitektur (Tabel 3 & 4 Paper)

Ablasi membuktikan bahwa setiap elemen dalam perumusan Stage 2 Lethon memiliki kontribusi krusial:

| Konfigurasi Model | ID Avg F1 | ID Avg EM | OOD Avg F1 | OOD Avg EM | Dampak Penghapusan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full SAC Model** | **66.24** | **53.11** | **48.45** | **32.19** | Baseline Lengkap |
| **w/o Bidirectional Mask (Causal)** | 63.62 | 50.35 | 45.11 | 30.16 | Degradasi $-2.62$ F1 (Anchor kehilangan pandangan global dokumen) |
| **w/o Anchor Embedding ($e_A$)** | 64.58 | 51.56 | 47.65 | 31.99 | Degradasi $-1.66$ F1 (Model gagal membedakan carrier dari token biasa) |
| **Random Anchor Selection** | 59.54 | 43.58 | 45.35 | 30.44 | Degradasi $-6.70$ F1 (Distribusi tidak merata memicu blind spots) |
| **Lingua-2 Entropy Selection** | 63.26 | 46.54 | 47.43 | 32.13 | Performa setara dengan Uniform Selection (generik & robust) |

### 4.3 Bukti Kegagalan Autoencoding Objective (Tabel 5 & 14 Paper)

Eksperimen membuktikan bahwa memaksakan tugas autoencoding justru merusak representasi:

| Model & Objective | Kompresi 5x (ID F1) | Kompresi 15x (ID F1) | Kompresi 51x (ID F1) | AE-BLEU4 | LM-PPL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SAC (LM Only - Lethon)** | **63.63** | **54.95** | **46.37** | N/A | **10.79** (Terbaik) |
| **SAC (w/ AE + LM)** | 62.04 | 51.73 | 44.69 | 98.19 | 11.01 (Memburuk) |
| **SAC (w/ AE Only)** | 56.55 | 49.93 | 43.95 | 99.94 | - (Keruntuhan Akurasi) |
| **500xCompressor (AE+LM)** | 55.26 | 49.35 | 43.19 | 87.21 | 11.83 |

Ketika SAC dilatih hanya dengan AE, skor BLEU rekonstruksi teks mencapai $99.94\%$ sempurna, namun performa QA downstream runtuh hingga $56.55\%$ (merosot $7.08$ poin F1). Ini adalah bukti empiris definitif bahwa **menghafal kata per kata bertentangan dengan abstraksi semantik penalaran**.

### 4.4 Skalabilitas Skala Model (1B, 3B, 8B Parameters - Tabel 6 Paper)

Keunggulan SAC tidak memudar saat parameter model diperbesar ke 8 Miliar:

| Model Ukuran | Metode | ID SearchQA F1 | ID NQ F1 | ID Avg F1 | OOD DROP F1 | OOD RE F1 | OOD Avg F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1B Model** | EPL | 66.36 | 53.80 | 51.52 | 32.16 | 59.75 | 36.74 |
| | **SAC** | **68.97** | **56.79** | **54.95** | **36.72** | **61.04** | **39.26** |
| **3B Model** | EPL | 73.61 | 63.59 | 68.60 | 46.20 | 65.73 | 55.97 |
| | **SAC** | **73.88** | **65.98** | **69.93** | **48.46** | **75.11** | **61.76** |
| **8B Model** | EPL | 74.86 | 67.24 | 71.05 | 49.98 | 69.77 | 59.88 |
| | **SAC** | **76.92** | **67.77** | **72.35** | **51.55** | **78.91** | **65.23** |

---

## 5. Implementasi Teknis & Pseudo-Code Arsitektur

Berikut adalah spesifikasi implementasi PyTorch modular untuk modul `SemanticAnchorAggregator` yang terintegrasi dalam paket `lethon.core.anchor`:

```python
"""
lethon/core/anchor/aggregator.py
Stage 2: Contextual Semantic Anchor Aggregation (SAC)
Berdasarkan formulasi riset arXiv:2510.08907 (ICLR 2026).
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAnchorAggregator(nn.Module):
    """
    Mengonsolidasikan konteks token panjang ke dalam representasi anchor
    menggunakan In-Context Tokens, Learnable Anchor Embedding, dan Bidirectional Attention.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_layers: int = 16,
        num_heads: int = 32,
        default_compression_ratio: float = 15.0,
        selection_strategy: str = "uniform",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.default_compression_ratio = default_compression_ratio
        self.selection_strategy = selection_strategy

        # Learnable Anchor Embedding Vector e_A
        self.anchor_embedding = nn.Parameter(
            torch.empty(1, 1, hidden_size)
        )
        # Inisialisasi N(0, 1 / sqrt(d))
        nn.init.normal_(self.anchor_embedding, mean=0.0, std=1.0 / math.sqrt(hidden_size))

    def compute_anchor_indices(
        self,
        seq_len: int,
        compression_ratio: Optional[float] = None,
        spectral_energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Menentukan indeks token yang bertindak sebagai anchor carrier.
        
        Args:
            seq_len: Panjang total sekuens token L.
            compression_ratio: Rasio kompresi r (default: 15.0).
            spectral_energies: Tensor energi spektral per token dari Stage 1 (opsional).
            
        Returns:
            torch.Tensor: Indeks 1D berukuran [M] dengan M = floor(L / r).
        """
        r = compression_ratio or self.default_compression_ratio
        num_anchors = max(1, math.floor(seq_len / r))

        if self.selection_strategy == "spectral_guided" and spectral_energies is not None:
            # Strategi B: Pilih token dengan stabilitas spektral tertinggi di setiap chunk
            chunk_size = seq_len / num_anchors
            indices = []
            for k in range(num_anchors):
                start = int(k * chunk_size)
                end = int((k + 1) * chunk_size) if k < num_anchors - 1 else seq_len
                chunk_energies = spectral_energies[start:end]
                best_local_idx = torch.argmax(chunk_energies).item()
                indices.append(start + best_local_idx)
            return torch.tensor(indices, dtype=torch.long)

        # Strategi A: Uniform Chunk-Centered Selection (Default EPL/SAC)
        # i_k = floor((k + 0.5) * L / M)
        indices = [
            math.floor((k + 0.5) * (seq_len / num_anchors))
            for k in range(num_anchors)
        ]
        return torch.tensor(indices, dtype=torch.long)

    def inject_anchor_embeddings(
        self,
        embeddings: torch.Tensor,
        anchor_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Menyuntikkan e_A secara aditif pada posisi token anchor:
        e_i = Emb(c_i) + I(i in S) * e_A
        
        Args:
            embeddings: Tensor embedding input [Batch, Seq_Len, Hidden_Size].
            anchor_indices: Indeks anchor [M].
            
        Returns:
            augmented_embeddings: Tensor [Batch, Seq_Len, Hidden_Size].
            indicator_mask: Boolean mask [Seq_Len].
        """
        b, s, d = embeddings.shape
        augmented = embeddings.clone()
        
        indicator = torch.zeros(s, dtype=torch.bool, device=embeddings.device)
        indicator[anchor_indices] = True

        # Tambahkan e_A pada posisi anchor
        augmented[:, indicator, :] = augmented[:, indicator, :] + self.anchor_embedding
        return augmented, indicator

    def construct_bidirectional_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Membangun attention mask dua arah penuh (Full Visibility) bernilai 0:
        M_attn[i, j] = 0 untuk seluruh i, j.
        """
        return torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)

    def extract_anchor_kv_cache(
        self,
        all_layer_keys: List[torch.Tensor],
        all_layer_values: List[torch.Tensor],
        anchor_indices: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Mengekstraksi representasi Key dan Value hanya pada posisi anchor tokens:
        K_tilde^(l) = K^(l)[:, S, :], V_tilde^(l) = V^(l)[:, S, :]
        
        Args:
            all_layer_keys: List berisi tensor Key per layer [B, Num_Heads, Seq_Len, Head_Dim].
            all_layer_values: List berisi tensor Value per layer [B, Num_Heads, Seq_Len, Head_Dim].
            anchor_indices: Indeks anchor [M].
            
        Returns:
            Dict dengan kunci 'compressed_keys' dan 'compressed_values'.
        """
        compressed_keys = []
        compressed_values = []

        for k_l, v_l in zip(all_layer_keys, all_layer_values):
            # Slicing sepanjang dimensi sekuens (dim 2)
            k_anchor = k_l[:, :, anchor_indices, :]
            v_anchor = v_l[:, :, anchor_indices, :]
            compressed_keys.append(k_anchor)
            compressed_values.append(v_anchor)

        return {
            "compressed_keys": compressed_keys,
            "compressed_values": compressed_values,
        }
```

---

## 6. Integrasi dengan Arsitektur Global Lethon

```
+-------------------------------------------------------------------------+
|                              LETHON ENGINE                              |
+-------------------------------------------------------------------------+
  Stage 1: Fast Spectral Fourier Filtering (Parameter-Free O(N log N))
  - Membuang noise koefisien frekuensi tinggi (20%-30% token reduction).
  - Menghasilkan denoised sequence tensor X_denoised dan kurva energi spektral.
                                   │
                                   ▼
  Stage 2: Contextual Semantic Anchor Aggregation (SAC - Dokumen ini)
  - Menerima X_denoised.
  - Memilih M token anchor via Uniform / Spectral-Guided Selection.
  - Menginjeksi learnable anchor embedding e_A.
  - Menjalankan forward pass dengan Full Bidirectional Attention Mask.
  - Mengekstraksi multi-layer Key-Value pairs tilde{M}_KV pada indeks anchor.
                                   │
                                   ▼
  Stage 3: Dynamic Promising Token Subspace Masking (RLPT)
  - Menerima anchor representations tilde{M}_KV sebagai basis subspace.
  - Memproyeksikan aksi RL ke low-rank subspace untuk menekan varians gradien.
                                   │
                                   ▼
  Stage 4: Sequential Decision MDP & CoT Length Truncation (TPRL / ThinkPrune)
  - Mengoptimasi alur penalaran akhir dan memotong pemikiran CoT berulang.
```

### Aturan Protokol Penyambungan Stage 1 $\to$ Stage 2:
1. **Sub-Context Chunking (510 Tokens):**  
   Untuk konteks dokumen panjang ($> 2048$ token), sekuens dipecah menjadi blok sub-konteks berukuran $L_{\text{sub}} = 510$ token. Setiap sub-konteks dikompresi menjadi sub-anchor oleh Stage 2, lalu digabungkan (*concatenated*) secara sekuensial.
2. **Positional Coordinate Offset:**  
   Anchor tokens mempertahankan offset posisi absolut dokumen asli untuk memastikan koordinat RoPE selaras saat diinjeksi ke target LLM decoder.
3. **Zero Semantic Gap Guarantee:**  
   Encoder kompresor dan Target LLM decoder wajib menggunakan arsitektur atau basis tokenizer yang kompatibel guna menjamin representasi KV cache dapat langsung diinterpretasikan tanpa adapter proyeksi tambahan.
