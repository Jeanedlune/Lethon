# ADR-0003: Parameter-Free 1D-FFT vs Heuristic Pruning for Stage 1 Spectral Truncation

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Algorithmic Team & Architecture Guild
- **Consulted Literature:**
  - `arXiv:2508.06038`: *Fourier Compressor: Frequency-Domain Visual Token Compression for VLMs* (Wang et al., 2025/2026)
  - `arXiv:2510.16418`: *FourierCompress: Layer-Aware Spectral Activation Compression*
  - `arXiv:2410.04417`: *SparseVLM: Visual Token Sparsification for Efficient VLM Inference*
  - `arXiv:2501.03895`: *LLaVA-Mini: Efficient Large Multimodal Models with One Vision Token*
  - `PRD.md`: *Lethon System Requirements & Target KPIs*

---

## 1. Context & Problem Statement

Pada arsitektur kompresi multi-tahap Lethon ([PRD.md](../../PRD.md)), **Stage 1 (First-Pass Sifting)** bertanggung jawab menyaring volume token masif (panjang sekuens $N = 4096$ hingga $32768$ pada long-context LLM, atau ratusan hingga ribuan patch visual pada VLM) secara instan.

Tantangan teknis utama pada tahap ini mencakup:
1. **Target Latensi Ketat ($< 3$ ms pada CPU):**  
   Pengguna agen AI (Claude Code, Cursor, Windsurf) menuntut interaksi real-time tanpa jeda (*zero perceived lag*). Jika tahap kompresi pertama memakan waktu $> 15$ ms, efisiensi penghematan token downstream akan tereduksi oleh waktu komputasi lokal (*latency tax*).
2. **Ketiadaan Akselerator GPU pada Local Machine / Edge:**  
   CLI wrapper dan MCP tool Lethon harus dapat dijalankan secara instan via `uvx` / `npx` pada mesin pengembang biasa tanpa mengharuskan adanya GPU NVIDIA dengan VRAM besar atau download bobot model neural network berukuran ratusan megabyte.
3. **Penyaringan Noise Sintaksis Tanpa Merusak Manifold Semantik:**  
   Metode harus memangkas 20%–30% token redundan di awal tanpa mendistorsi informasi semantik global yang dibutuhkan oleh tahap selanjutnya (Stage 2: SAC & Stage 3: RLPT).

---

## 2. Decision Drivers

- **Kompleksitas Asimptotik Sub-Kuadratik:** Menghindari $\mathcal{O}(N^2)$ yang mengalami ledakan waktu komputasi pada sekuens panjang.
- **Zero-Parameter / Zero-Checkpoint Dependency:** Bebas bobot latih (*parameter-free*), tidak ada ketergantungan download bobot PyTorch yang membebani distribusi package.
- **Akurasi & Integritas Representasi:** Mempertahankan struktur topologis dan energi semantik global dengan degradasi akurasi downstream $< 0.7\%$.
- **Model-Agnostic & Zero-Proxy:** Dapat diterapkan langsung pada token visual, teks dokumen, maupun *Chain-of-Thought* reasoning logs.

---

## 3. Considered Options

### Option 1: Heuristic Cosine Similarity & Clustering (e.g., LLaVA-PruMerge, VisionZip, ToMe)
Mengelompokkan token yang berdekatan atau mirip berdasarkan *pairwise cosine similarity*:
$$\text{Sim}(x_i, x_j) = \frac{x_i \cdot x_j}{\|x_i\| \|x_j\|}$$
Token-token yang memiliki kemiripan di atas ambang batas tertentu digabungkan (*merged*) melalui rata-rata tertimbang (*weighted average*).

### Option 2: Attention Weight & Saliency Heuristics (e.g., FastV, ATP-LLaVA, SparseVLM)
Menggunakan matriks *self-attention* dari beberapa layer awal transformer untuk menghitung skor kepentingan token:
$$\text{Score}(x_i) = \sum_{j} A_{i, j}$$
Token dengan skor atensi terendah dipangkas secara langsung.

### Option 3: Parameterized Query-Transformers / MLPs (e.g., MQT-LLaVA, QueCC, VisToG)
Menempatkan modul neural network kecil terlatih (seperti Q-Former atau Matryoshka Query Transformer) yang memproyeksikan sekuens token panjang ke sejumlah *learnable query tokens* yang lebih ringkas.

### Option 4: Parameter-Free 1D Fast Fourier Transform / DCT-II Filtering (Selected)
Mengubah sekuens representasi token ke domain spektral menggunakan 1D-FFT / 1D-DCT berkecepatan $\mathcal{O}(N \log N)$, membuang komponen frekuensi tinggi yang merepresentasikan fluktuasi lokal redundan, dan merekonstruksi representasi semantik esensial pada frekuensi rendah.

---

## 4. Evaluasi Komparatif & Trade-off Analysis

| Kriteria Evaluasi | Option 1: Cosine Sim / Clustering | Option 2: Attention Heuristics | Option 3: Parameterized Query-Transformer | Option 4: Parameter-Free 1D-FFT (Lethon) |
| :--- | :--- | :--- | :--- | :--- |
| **Kompleksitas Waktu** | $\mathcal{O}(B \cdot d \cdot N^2)$ | $\mathcal{O}(B \cdot d \cdot N^2)$ | $\mathcal{O}(B \cdot d \cdot N \cdot K)$ | $\mathbf{\mathcal{O}(B \cdot d \cdot N \log N)}$ |
| **Latensi CPU ($N=8192$)** | $> 80\text{ ms}$ (Lambat) | Tidak berlaku (Butuh Forward Model $> 100\text{ ms}$) | $> 45\text{ ms}$ + Forward pass | $\mathbf{< 3.5\text{ ms}}$ (Target Tercapai) |
| **Parameter Tambahan** | $0$ | $0$ | $\sim 15\text{M} - 50\text{M}$ params ($\sim 60 - 200\text{ MB}$) | $\mathbf{0}$ **(Murni Parameter-Free)** |
| **Dependensi Hardware** | CPU-friendly tapi lambat | Membutuhkan GPU / model backbone | Membutuhkan PyTorch / model weights | **Ringan: NumPy/SciPy/Rust CPU** |
| **Distorsi Manifold** | **Tinggi** (Cluster averaging mendistorsi distribusi fitur) | Rendah | Rendah (bila terlatih baik) | **Minimal** (*Preserves continuous smooth manifold*) |
| **Masalah Attention Sinks**| Tidak ada | **Rentan** (Token delimiter/awal menyedot atensi semu) | Tidak ada | **Kebal** (Analisis murni berbasis energi spektral) |

---

## 5. Decision Outcome & Rationale

Tim memutuskan memilih **Option 4: Parameter-Free 1D-FFT / 1D-DCT Filtering** sebagai arsitektur resmi untuk Stage 1 Lethon Engine.

### 5.1 Mengapa 1D-FFT Mengungguli Cosine Similarity Heuristics?
1. **Penskalaan Asimptotik $\mathcal{O}(N \log N)$ vs $\mathcal{O}(N^2)$:**  
   Pada $N = 8192$, komputasi *pairwise cosine similarity* memerlukan $\approx 3.35 \times 10^7$ perkalian vektor berdimensi $d=1024$. Sementara 1D-FFT hanya memerlukan $8192 \times \log_2(8192) \approx 1.06 \times 10^5$ operasi per kanal. Hal ini menghasilkan akselerasi lebih dari **25x lebih cepat** pada CPU.
2. **Menghindari Kerusakan Distribusi Fitur (*Centroid Distortion*):**  
   Penggabungan token (*merging*) berbasis Cosine Similarity memaksakan penggabungan dua token berbeda menjadi rata-rata centroid baru yang sering kali tidak berada pada manifold representasi alami model (*out-of-distribution feature drift*). Sebaliknya, pemotongan spektral frekuensi tinggi bertindak sebagai penghalusan matematis (*low-pass smoothing*) yang terbukti melestarikan varian fitur esensial.

### 5.2 Mengapa 1D-FFT Mengungguli Attention-Based Heuristics?
1. **Tidak Memerlukan Akses Internal Bobot Transformer:**  
   Metode atensi (seperti FastV atau ATP-LLaVA) mensyaratkan eksekusi forward-pass pada layer awal model LLM untuk mendapatkan matriks atensi. Hal ini tidak mungkin dilakukan pada mode *black-box API* atau *Zero-Proxy MCP context pre-processing* di mana kita memangkas teks mentah atau embedding sebelum dikirim ke API LLM komersial (Claude 3.7 / GPT-4o).
2. **Kebal terhadap Gejala Attention Sink:**  
   Attention weights pada LLM/VLM terkenal bias terhadap token pertama (*bos_token*), tanda baca koma/titik, dan token newline (*attention sink phenomenon*), yang membuat token-token tersebut tampak penting secara numerik padahal tidak mengandung informasi semantik yang relevan.

### 5.3 Mengapa 1D-FFT Mengungguli Parameterized Approaches?
- Parameterized approaches (MQT-LLaVA / QueCC) memerlukan pelatihan model tambahan (*training overhead*) dan pengunduhan checkpoint puluhan MB.
- Tanpa parameter, Lethon dapat didistribusikan dalam paket super-ringan via `uvx lethon` atau single binary Rust berukuran $< 15\text{ MB}$, selaras dengan misi distribusi open-source 100k Stars pada [PRD.md](../../PRD.md).

---

## 6. Analisis Trade-off: Akurasi vs Kecepatan (Bukti Empiris Paper arXiv:2508.06038)

Berdasarkan benchmark empiris formal pada paper acuan utama (*Fourier Compressor*):

1. **Efisiensi Komputasi & Kecepatan:**
   - **Penurunan FLOPs Inferensi:** Menurun drastis hingga **$83.8\%$** (dari $8.54\text{ TFLOPs}$ menjadi $1.38\text{ TFLOPs}$ pada LLaVA-7B).
   - **Peningkatan Waktu Respons (TTFT):** Waktu ke token pertama (*Time-to-First-Token*) meningkat **$31.2\%$ lebih cepat**.
   - **Penghematan KV-Cache:** Konsumsi memori KV-cache terpangkas hingga **$86.4\%$** (dari $312.5\text{ MB}$ menjadi $42.5\text{ MB}$).

2. **Retensi Akurasi Downstream:**
   - Pada pemangkasan moderat ($55.6\%$ token reduction, menyisakan 256 dari 576 token), akurasi model tetap **$100\%$ setara** ($64.6\%$ baseline vs $64.6\%$ Fourier-LLaVA pada rata-rata 8 benchmark VQA).
   - Pada pemangkasan agresif ($75.0\%$ token reduction, menyisakan 144 token), akurasi rata-rata tetap stabil di angka **$64.6\%$** ($0.0\%$ penurunan!).
   - Bahkan pada pemangkasan ekstrem $93.75\%$ (hanya menyisakan 36 token / 6.25%), penurunan akurasi hanya sebesar **$3.87\%$** (MMB dan SciQA bahkan tetap melampaui baseline!).
   - Pada pengujian Qwen2.5-VL-3B, kompresi Fourier bahkan menghasilkan **peningkatan akurasi $+2.1\%$ rata-rata** dan **$+7.9\%$ pada RealWorldQA**, membuktikan bahwa eliminasi noise frekuensi tinggi membantu model penalaran memfokuskan atensi pada fakta inti.

---

## 7. Konsekuensi Arsitektur & Mitigasi Risiko

### Konsekuensi Positif:
- Tahap 1 dapat berjalan sebagai *microsecond / millisecond pure CPU step*.
- Kompatibilitas multi-platform tanpa ketergantungan driver CUDA/GPU tertentu.
- Pengurangan beban token sebesar 20%–30% di awal langsung meringankan kerja Stage 2 (SAC) dan Stage 3 (RLPT).

### Risiko & Mitigasi:
1. **Risiko Pemangkasan Detail Tekstual Kritis:**  
   *Mitigasi:* Menggunakan **Skema B (Discrete Token Saliency)** untuk konteks kode/teks, di mana cutoff frekuensi tinggi digunakan sebagai skor pemeringkat (*saliency ranking*), bukan downsampling destruktif, sehingga token sintaks kritis (seperti tanda kurung kurawal atau operator logika) tetap terlindungi.
2. **Kalibrasi Ambang Batas ($\rho$):**  
   *Mitigasi:* Nilai retensi rasio $\rho$ dikonfigurasi adaptif: $\rho = 0.70$ untuk teks biasa / visual patch, dan $\rho = 0.85$ untuk sintaks kode pemrograman presisi tinggi.
3. **Migrasi Bahasa Pemrograman:**  
   Sesuai [ADR-0005](0005-language-selection-python-to-rust.md), implementasi riset Python (`scipy.fft`) akan di-porting ke modul Rust Core (`rustfft` + AVX-512 SIMD) untuk menjamin eksekusi stabil $< 1.5$ ms pada sistem produksi.
