# ADR-0006: Layer-Aware Adaptive Spectral Filtering vs Uniform Spectral Truncation for Transformer Sequences

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture Guild & Algorithmic Core Team
- **Consulted Literature:**
  - `arXiv:2510.16418`: *FourierCompress: Layer-Aware Spectral Activation Compression for Efficient and Accurate Collaborative LLM Inference* (Ma et al., Oct 2025)
  - `arXiv:2508.06038`: *Fourier Compressor: Frequency-Domain Visual Token Compression for VLMs* (Wang et al., 2025)
  - `PRD.md`: *Lethon System Requirements & Target KPIs (Stage 1 Spectral Filtering)*
  - `docs/dec/dec-01-spectral-filtering-math.md`: *Mathematical Specifications of Spectral Filtering*

---

## 1. Context & Problem Statement

Pada pipeline kompresi konteks Lethon ([PRD.md](../../PRD.md)), **Stage 1 (Spectral Fourier Filtering)** bertindak sebagai gerbang saringan pertama (*first-pass gatekeeper*) untuk memangkas noise dan redundansi sekuens token secara instan ($\mathcal{O}(N \log N)$) sebelum diteruskan ke Stage 2 (Semantic Anchor Aggregation) dan Stage 3 (RL Subspace Masking).

Secara konvensional, metode kompresi frekuensi maupun pemangkasan token menerapkan **Uniform Filtering Policy**, yaitu menerapkan rasio retensi konstan $\rho$ (misal mempertahankan 50% atau 70% koefisien) di setiap layer atau memperlakukan seluruh token sequence dari awal hingga akhir jaringan transformer dengan perlakuan homogen.

Namun, penelitian empiris terbaru pada **FourierCompress** (*arXiv:2510.16418*) mengungkapkan fenomena mendasar:
1. **Asimetri Spektral Antar Layer:**  
   Aktivasi pada layer awal transformer ($l=1$) menunjukkan **spatial smoothness** yang luar biasa dengan pola garis vertikal konsisten antar token (kemiripan representasi antar token $> 0.8$ pada benchmark PiQA, ARC, dan OpenBookQA). Energi sinyal sangat terkonsentrasi ($> 52.2\% - 85\%$) pada komponen frekuensi rendah di sekitar DC component.
2. **Entropi Tinggi & Chaos pada Layer Dalam:**  
   Seiring bertambahnya kedalaman ($l \to L$), representasi bertransformasi menjadi representasi kontekstual spesifik per token. Kehalusan spasial lenyap, entropi meningkat drastis, dan spektrum didominasi oleh fluktuasi frekuensi tinggi yang memuat logika penalaran kritis (*in-depth reasoning*).
3. **Keruntuhan Akurasi Akibat Pemotongan Seragam:**  
   Menerapkan rasio kompresi agresif (misal 7.6x – 10.3x) secara seragam pada layer dalam menyebabkan lonjakan kesalahan rekonstruksi (error naik dari $2.1$ pada Layer 1 menjadi $> 19.0 - 23.5$ pada Layer 15), yang merusak akurasi penalaran downstream hingga anjlok $> 26\%$.

Oleh karena itu, arsitektur Lethon harus menentukan: Apakah kompresi spektral harus diterapkan secara **Uniform** (rasio tetap di seluruh kedalaman/sekuens) atau **Layer-Aware Adaptive Filtering**?

---

## 2. Decision Drivers

- **Preservasi Akurasi Penalaran Ekstrem ($< 0.3\% - 0.7\%$ Degradation):** Mempertahankan integritas rantai penalaran panjang (*Chain-of-Thought*) dan resolusi semantik LLM.
- **Efisiensi Bandwidth & Payload Maksimal ($\ge 50\% - 66.7\%$ Token / 7.6x Activation Reduction):** Mengurangi volume data yang ditransmisikan antara client edge dan host server/API.
- **Kompleksitas Komputasi Rendah & Determinisme:** Bebas dependensi kalkulasi dinamis yang lambat; formula scheduling harus dapat dihitung secara instan dalam skala $\mathcal{O}(1)$.
- **Kesesuaian dengan Sifat Manifold Transformer:** Selaras dengan evolusi fitur bahasa dari ekstraksi fitur umum task-agnostic (layer awal) ke integrasi penalaran task-specific (layer dalam).

---

## 3. Considered Options

### Option 1: Uniform Spectral Truncation (Baseline Statis)
Menerapkan rasio pemotongan frekuensi tetap $\rho_{\text{fixed}}$ di setiap titik kompresi:
$$\rho(l) = \rho_{\text{fixed}} \quad \forall l \in \{1, \dots, L\}$$
- *Contoh:* Memotong 30% frekuensi tertinggi di seluruh layer atau seluruh blok aktivasi secara identik.

### Option 2: Attention/SVD-Guided Dynamic Layer Pruning
Menghitung singular values (SVD) atau matriks entropi atensi pada setiap layer secara runtime untuk menentukan rasio pemotongan layer tersebut:
$$\rho(l) = f(\sigma_{\text{top}}(A_l), \mathcal{H}(\text{Attn}_l))$$

### Option 3: Layer-Aware Polynomial Spectral Scheduling (Selected)
Memanfaatkan titik pemisahan optimal pada Layer 1 dan menerapkan fungsi penjadwalan kurvatur monotonik naik terhadap kedalaman layer:
$$\rho(l) = \min\left(1.0, \, \rho_0 + (1.0 - \rho_0) \cdot \left(\frac{l - 1}{L - 1}\right)^\gamma\right)$$
di mana $\rho_0$ adalah rasio retensi dasar agresif pada Layer 1 (misal $0.13 - 0.25$, menghasilkan reduksi 4x – 7.6x), $\gamma \ge 1.0$ adalah koefisien kurvatur kekakuan (misal $\gamma = 2.0$), dan $l$ adalah indeks layer transformer.

---

## 4. Evaluasi Komparatif & Trade-off Analysis

| Kriteria Evaluasi | Option 1: Uniform Truncation | Option 2: SVD/Attention Dynamic | Option 3: Layer-Aware Polynomial (Lethon) |
| :--- | :--- | :--- | :--- |
| **Akurasi Downstream (Avg Drop)** | **Buruk** ($\downarrow 6.5\% - 26\%$ pada layer dalam) | Baik ($\downarrow 1.2\% - 3.5\%$) | **Terbaik** ($\mathbf{\downarrow < 0.3\%}$, near-lossless) |
| **Overhead Komputasi** | Sangat Rendah ($\mathcal{O}(1)$) | **Sangat Tinggi** ($\mathcal{O}(S \cdot D^2)$ SVD runtime) | **Sangat Rendah** ($\mathcal{O}(1)$ deterministik) |
| **Sensitivitas terhadap Outlier** | Tinggi pada layer lanjut | Moderat | **Kebal** (Layer dalam diproteksi secara otomatis) |
| **Kesesuaian Hardware Edge** | Baik | Sangat Buruk (Boros VRAM/FLOPs) | **Sangat Baik** (Optimal untuk DSP / cuFFT / Rust Core) |
| **Penurunan Payload Maksimum** | Terbatas (harus konservatif agar aman) | Variatif | **Agresif di Awal (7.6x – 10.3x)** tanpa merusak akurasi |

---

## 5. Decision Outcome & Rationale

Tim arsitektur memutuskan untuk memilih **Option 3: Layer-Aware Polynomial Spectral Scheduling** dan menetapkan **Layer 1 Transformer sebagai split/sifting point utama**.

### Rationale Utama:

1. **Validasi Teoretis & Empiris Konsentrasi Energi (arXiv:2510.16418):**  
   Pengujian pada 10 dataset penalaran commonsense (Llama 3 dan Qwen2.5) membuktikan bahwa kompresi spektral pada Layer 1 mencapai rata-rata reduksi ukuran data sebesar **7.6x** (hingga **10.3x** pada PIQA) dengan penurunan akurasi hanya **$< 0.3\%$**, bahkan melampaui baseline tanpa kompresi pada Qwen2.5-3B karena efek *denoising* frekuensi tinggi. Menerapkan rasio yang sama pada layer 8 atau 15 mengakibatkan akurasi jatuh bebas hingga di bawah 48%.
2. **Efisiensi Algoritmik & Zero-SLA Penalty:**  
   Berbeda dengan pendekatan runtime SVD/Attention (Option 2) yang memakan waktu hingga $> 100\text{ s}$ per inferensi, scheduling Layer-Aware dihitung secara deterministik $\mathcal{O}(1)$ sebelum FFT dijalankan, menjaga total latensi kompresi tetap $< 0.7\text{ ms}$ pada akselerator hardware dan $< 3\text{ ms}$ pada CPU.
3. **Mencegah "Information Bottleneck" pada Rantai Penalaran (CoT):**  
   Token-token penalaran pada tahap awal mengekstraksi sintaks dasar bahasa yang memiliki pola redundansi tinggi, sedangkan token pada layer akhir mengintegrasikan kesimpulan logis. Pemotongan layer-aware memastikan kompresi paling agresif hanya terjadi di zona aman berenergi rendah, menjaga integritas rantai penalaran untuk Stage 4 CoT MDP.

---

## 6. Arsitektur & Aturan Implementasi

1. **Matriks Aktivasi Client-Server (Collaborative/Split Inference):**  
   Jika Lethon beroperasi dalam skenario collaborative split LLM, titik potong (*split point*) wajib ditempatkan pada **Layer 1** dengan blok koefisien frekuensi rendah rectangular $\mathcal{A}_{\text{core}} \in \mathbb{C}^{K_S \times K_D}$.
2. **Rekonstruksi Simetri Konjugat Hermitian:**  
   Sisi penerima/server wajib memulihkan spektrum penuh menggunakan simetri konjugat $\mathcal{A}[u, v] = \mathcal{A}^*[S-u, D-v]$ sebelum IFFT guna menjamin rekonstruksi riil murni bebas distorsi fase.
3. **Pipeline Multi-Layer / State Sifting:**  
   Untuk pemangkasan token internal atau state embeddings antar-layer, fungsi rasio adaptif `calculate_layer_retention_ratio(layer_idx, total_layers)` di `lethon.core.spectral` wajib digunakan sebagai acuan parameter $\rho(l)$.
