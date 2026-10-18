# ADR-0011: Multi-Agent Markov Game (MAPPO) vs Single-Agent and Heuristic Cross-Layer Pruning for KV-Cache Compression

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture Guild & Algorithmic Core Team
- **Consulted Literature:**
  - `arXiv:2503.23459v1`: *Reinforcement Learning-based Token Pruning in Vision Transformers: A Markov Game Approach (RL4EViT)* (Lu et al., Fudan University & Université Paris Cité, Mar 2025)
  - `arXiv:2603.13394v1`: *Language-Guided Token Compression with Reinforcement Learning in LVLMs (TPRL)* (2026)
  - `arXiv:2510.08907`: *Semantic-Anchor Compression (SAC) for LLMs* (ICLR 2026)
  - `arXiv:2512.04359v3`: *Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning (SENT)* (2026)
  - `PRD.md`: *Lethon System Requirements & 4-Stage Architecture (Multi-Agent Inter-Layer Optimization)*
  - `docs/dec/dec-07-markov-game-mappo-pruning.md`: *Technical & Mathematical Specifications of DEC-07*

---

## 1. Context & Problem Statement

Pada arsitektur kompresi konteks Lethon ([PRD.md](../../PRD.md)), model Transformer dan LLM penalaran panjang menghasilkan beban komputasi dan memori yang masif pada Key-Value (KV) Cache selama fase *prefill* dan *autoregressive decoding*. Redundansi semantik terdistribusi secara tidak seragam di sepanjang kedalaman layer (*layer depth*) model.

Dalam merancang mekanisme pemangkasan token dan KV-Cache antar layer (*cross-layer / inter-layer pruning*), terdapat beberapa karakteristik krusial:
1. **Dampak Sekuensial Antar-Layer:**  
   Keputusan untuk mempertahankan atau memangkas sebuah token pada layer awal ($l$) secara permanen mengubah distribusi aktivasi, atensi, dan representasi semantik pada seluruh layer downstream ($l+1, \dots, L$). Keputusan yang bersifat independen per layer (*myopic / layer-agnostic*) memotong token yang esensial bagi layer akhir.
2. **Keterbatasan Pendekatan Rasio Kaku (DynamicViT / Static Heuristics):**  
   Metode konvensional memaksakan rasio pruning yang seragam atau terjadwal kaku (misal mempertahankan 70% token di setiap stage). Akibatnya, sistem tidak mampu melakukan *early aggressive pruning* saat input sederhana, atau menahan token saat konteks membutuhkan penalaran presisi tinggi.
3. **Ledakan Ruang Aksi pada Single-Agent PPO:**  
   Jika pemangkasan seluruh token diserahkan kepada satu agen sentral (*single-agent formulation*), ruang keputusan diskrit meledak sebesar $2^N$ untuk sekuens sepanjang $N$ token. Hal ini memicu varians gradien ekstrim dan kegagalan *credit assignment*—agen tidak dapat membedakan token mana yang menyebabkan kegagalan downstream.

Oleh karena itu, tim arsitektur perlu menetapkan keputusan strategis: **Kapan dan bagaimana paradigma Multi-Agent Markov Game (MAPPO) harus diterapkan pada arsitektur Lethon untuk kompresi KV-Cache dan token lintas layer, serta bagaimana batasannya dibandingkan metode Stage 1 (Spektral), Stage 2 (SAC), dan Stage 4 (TPRL/ThinkPrune)?**

---

## 2. Decision Drivers

- **Preservasi Akurasi Downstream:** Meminimalkan degradasi akurasi downstream ($< 0.5\%$) pada tingkat kompresi token yang agresif ($> 40\%$).
- **Adaptivitas Terhadap Kompleksitas Input:** Rasio pemangkasan harus fleksibel secara data-adaptif di setiap layer, bukan rasio konstan kaku.
- **Efisiensi Memori & Latensi KV-Cache:** Mampu memangkas token sesegera mungkin pada layer-layer awal (*earliest stage pruning*) untuk meminimalkan beban memori KV-cache secara kumulatif.
- **Skalabilitas Komputasi & Minimal Parameter:** Mekanisme multi-agen tidak boleh menambahkan jutaan parameter baru atau memicu inferensi yang lambat.
- **Harmoni Pipeline Hybrid Lethon:** Bekerja sinergis dengan penyaringan spektral Fourier (Stage 1), penyerapan Semantic Anchors (Stage 2), dan regulasi entropi (Stage 4).

---

## 3. Considered Options

### Option 1: Heuristic Layer-Wise Thresholding (Attention / Cosine Similarity Pruning)
- Memangkas token pada setiap layer berdasarkan skor atensi akumulatif atau *cosine distance* terhadap query/class token dengan threshold tetap.
- *Karakteristik:* Tanpa pelatihan (zero-parameter), namun rentan memotong token kunci yang memiliki atensi rendah di layer awal tetapi krusial di layer akhir (*myopic failure*). Akurasi turun drastis jika kompresi $> 30\%$.

### Option 2: Supervised MLP Predictor with Fixed Ratios (DynamicViT-style)
- Menggunakan MLP kecil pada layer tertentu untuk memprediksi probabilitas retensi token, tetapi mengunci rasio token yang dipertahankan pada nilai konstan per layer (misal $\rho = [0.7, 0.7, 0.7]$).
- *Karakteristik:* Lebih adaptif dibanding heuristik murni, tetapi batasan rasio statis membatasi fleksibilitas model untuk membuang redundansi di layer awal (Gambar 5 arXiv:2503.23459). Membutuhkan *distillation loss* yang kompleks.

### Option 3: Single-Agent Reinforcement Learning (Centralized PPO)
- Satu agen sentral menerima representasi seluruh token dan mengeluarkan vektor biner seleksi ukuran $N$.
- *Karakteristik:* Ruang aksi raksasa ($2^N$). Eksplorasi sangat tidak efisien, konvergensi lambat, dan performa inferior (Ablasi Table II arXiv:2503.23459: Top-1 Acc hanya 76.5% vs 76.8% MAPPO pada DeiT-S).

### Option 4: Multi-Agent Markov Game (MAPPO) with Shared Actor and Centralized Critic (Selected)
- Setiap token dimodelkan sebagai agen individual dengan aksi biner $A_i^l \in \{0, 1\}$.
- Seluruh agen berbagi bobot pada satu Actor Network $\pi_\theta$ ringan (fitur lokal) dan satu Critic Network $V_\phi$ terpusat (fitur lokal + fitur global konteks).
- Dinamika reward komposit kompetitif-kolaboratif:
  $$R_i^l = \alpha \cdot r_{1,i}^l + \beta \cdot \frac{\gamma_{\text{task}}}{n_l + \epsilon}$$
  di mana $r_{1,i}^l = -A_i^l$ (penalti preservasi) dan komponen kedua adalah reward akurasi teramortisasi atas jumlah token aktif $n_l$.
- Transisi status sekuensial antar blok transformer dimodelkan sebagai Markov Game trajectory $(S^1, A^1, R^1, \dots, S^L, A^L, R^L)$.

---

## 4. Evaluasi Komparatif & Bukti Empiris

| Dimensi Evaluasi | Option 1: Heuristic Attention | Option 2: DynamicViT (Fixed Ratios) | Option 3: Single-Agent PPO | Option 4: MAPPO Markov Game (Selected) |
| :--- | :--- | :--- | :--- | :--- |
| **Formulasi Masalah** | Aturan Ambang Statis | Prediksi Terawasi Berbobot Tetap | Single MDP (Aksi Global $2^N$) | **Markov Game Multi-Tahap (CTDE)** |
| **Ruang Aksi Agen** | N/A | Heuristik Top-$K$ | $2^N$ (Kombinatorial Eksplosif) | **$\{0, 1\}$ per Agen Individual** |
| **Adaptivitas Rasio Layer** | Rendah (ambang statis) | Nol (rasio dikunci manual) | Dinamis tetapi tidak stabil | **Sangat Tinggi (Data-Driven Pareto)** |
| **Akurasi DeiT-S (ImageNet-1k)** | 76.8% (EViT) / 78.6% (ToMe) | 74.0% (-5.9% drop) | 76.5% (-3.4% drop) | **78.6% – 79.4% (Hanya -0.5% drop)** |
| **Inference Speedup** | $\times 1.21$ | $\times 1.17$ | $\times 1.15$ | **$\times 1.61$ (+44% Speedup)** |
| **Profil Pemangkasan Layer** | Tersebar merata | Terpaksa merata sesuai rasio | Tidak menentu | **Agresif di Layer Awal (Optimal)** |
| **Overhead Parameter** | 0 | 0.7M (MLPs) | 2.5M+ (High-dim heads) | **0.2M (Shared Lightweight MLP)** |

### Bukti Empiris Kunci dari arXiv:2503.23459:
1. **Trade-Off Akurasi vs Speedup Tertinggi (Tabel I Paper):**  
   Pada DeiT-S dengan komputasi 2.6 GFLOPs, MAPPO mempertahankan akurasi 78.6% dengan kecepatan 2041.4 images/s ($\times 1.61$), mengalahkan DynamicViT (58.3%, 1980 images/s) dan EViT (76.8%, 2144 images/s).
2. **Superioritas Multi-Agent atas Single-Agent (Tabel II Paper):**  
   Pengujian ablasi langsung membuktikan bahwa perlakuan individual pada setiap token melalui MAPPO secara konsisten melampaui Single-Agent PPO pada semua tingkat kompleksitas (78.6% vs 78.3% pada 2.6 GFLOPs; 81.4% vs 81.0% pada 11.6 GFLOPs).
3. **Fenomena Pruning di Layer Paling Awal (Gambar 5 Paper):**  
   Visualisasi kualitatif membuktikan bahwa MAPPO secara natural belajar memangkas sebagian besar token redundan pada transformer block ke-3 (layer awal), dan hanya meneruskan token esensial (seperti mata dan paruh burung) ke block 6 dan 9. Sebaliknya, DynamicViT terpaksa membagi kuota pruning secara merata karena keterbatasan rasio tetapnya.

---

## 5. Decision Outcome & Rationale: Kapan Multi-Agent Pruning Relevan?

Tim arsitektur memutuskan untuk mengadopsi **Option 4: Multi-Agent Markov Game (MAPPO)** sebagai arsitektur koordinasi lintas layer (*Cross-Stage Optimization*) pada **Lethon**.

### Matriks Relevansi Penerapan pada Pipeline Lethon:

1. **SANGAT RELEVAN (Digunakan):**
   - **Cross-Layer KV-Cache Pruning pada LLM Backbone:** Ketika model melakukan inferensi konteks panjang multi-layer, pemangkasan KV pairs di layer-layer awal ($l \in [1, L/3]$) memberikan penghematan alokasi memori kuadratik untuk seluruh layer selanjutnya.
   - **Multi-Stage Vision-Language Redundancy Elimination:** Pada encoder visual dan cross-attention layers, di mana token visual memiliki redundansi spasial masif yang dapat dihilangkan sebelum proyeksi ke LLM text layers.
   - **Konfigurasi Budget Dinamis:** Menggunakan kontrol rasio $\frac{\alpha}{\beta}$ untuk menggeser model secara mulus antara mode hemat daya (*ultra-fast latency*) dan mode akurasi tinggi (*high-fidelity reasoning*).

2. **TIDAK RELEVAN (Gunakan Modul Lain):**
   - **Filter Noise Token Frekuensi Tinggi Instan:** Jangan gunakan RL untuk membuang noise di input awal; gunakan **Stage 1 Fast Fourier FFT** ([DEC-01](file:///workspaces/Lethon/docs/dec/dec-01-spectral-filtering-math.md)) yang bebas parameter dan beroperasi dalam $\mathcal{O}(N \log N)$ ($< 3\text{ms}$).
   - **Konteks Panjang Tanpa Fine-Tuning Model:** Untuk konteks dokumen eksternal di sisi API/MCP tool, gunakan **Stage 2 Semantic Anchor Aggregation (SAC)** ([DEC-02](file:///workspaces/Lethon/docs/dec/dec-02-semantic-anchor-aggregation.md)) yang tidak memerlukan modifikasi bobot attention backbone.
   - **Pemadatan Langkah Penalaran CoT:** Untuk memotong repetisi berpikir verbal model (*Chain-of-Thought*), gunakan **Stage 4 ThinkPrune/SENT** ([DEC-04](file:///workspaces/Lethon/docs/dec/dec-04-cot-iterative-pruning.md), [DEC-06](file:///workspaces/Lethon/docs/dec/dec-06-entropy-collapse-curriculum-sent.md)).

---

## 6. Implementation Guidelines & Guardrails

1. **Parameter Sharing:** Semua agen token pada seluruh layer wajib berbagi satu Actor Network $\pi_\theta$ dan satu Critic Network $V_\phi$.
2. **Non-Destructive Mini-Batch Masking:** Dalam forward pass batch PyTorch/NumPy, token yang dipangkas ($A_i^l = 0$) tidak dihapus dari tensor dimensi, melainkan di-mask dengan nilai bias $-1000.0$ pada attention logits sebelum Softmax.
3. **Alternating Training Protocol:** Melatih parameter RL ($\pi_\theta, V_\phi$) dan fine-tuning backbone transformer secara bergantian (ganjil: optimasi RL dengan backbone beku; genap: fine-tuning backbone dengan RL beku) untuk menjamin stabilitas konvergensi.
