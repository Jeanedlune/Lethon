# Product Requirement Document (PRD): Lethon
**System Name:** Lethon (Spectral-RL Hybrid Context Compression Engine)  
**Target Release:** Q4 2026  
**Document Version:** 1.0.0-Draft  
**Target Distribution:** GitHub Open-Source (Goal: 100k Stars)  

---

## 1. Executive Summary & Vision

**Lethon** adalah engine pemangkasan token dan kompresi konteks AI *hybrid* generasi baru yang dirancang khusus untuk era **Agentic AI & Long-Reasoning LLMs** (Claude Code, Cursor, Windsurf, OpenAI Codex, dan Google Antigravity). 

Dengan mengombinasikan **Domain Frekuensi Fourier (1D-FFT)**, **Semantic Anchor Aggregation (SAC)**, dan **Reinforcement Learning Subspace Masking (RLPT/TPRL/ThinkPrune)**, Lethon mampu memangkas volume token dan panjang penalaran (*Chain-of-Thought*) hingga **50%–66.7%** dengan penurunan akurasi kurang dari **0.7%**.

Sistem diintegrasikan secara **Zero-Friction / Zero-Proxy** melalui **Model Context Protocol (MCP)** dan didistribusikan langsung via `uvx` / `npx`, memungkinkan pengembang dan enterprise menghemat biaya API serta *latency* secara instan tanpa perlu mengubah *workflow* atau menjalankan proxy server di background.

---

## 2. Problem Statement & Target Market

### 2.1 Permasalahan Utama (Pain Points)
1. **Explosive KV-Cache & Latency Overhead:** Model penalaran panjang (seperti DeepSeek-R1) dan input multimodal beresolusi tinggi menghasilkan puluhan ribu token. Hal ini memicu *memory bottleneck* pada KV-cache serta memperlambat waktu respons (*time-to-first-token* dan *generation speed*).
2. **Tagihan API Membengkak:** Developer dan tim engineer yang menggunakan agen otomatis (Claude Code, Cursor) mengalami lonjakan konsumsi token hingga jutaan token per hari untuk membaca repositori kode dan *log trace*.
3. **Kegagalan Metode Pruning Heuristik Tradisional:** Pendekatan *static pruning* berbasis *attention weights* atau *cosine similarity* biasa bersifat kaku (*one-shot*), merusak alur logika penalaran (*Chain-of-Thought*), dan mengabaikan kebutuhan semantik spesifik dari tugas downstream.

### 2.2 Target Audience
* **AI Developers & Power Users:** Pengguna Claude Code, Cursor, Windsurf, Codex, dan Antigravity yang menginginkan alur kerja lebih kencang dan hemat token.
* **AI Framework Developers:** Tim yang membangun sistem RAG, Multi-Agent, dan aplikasi LLM berbasis LangChain, LlamaIndex, atau Vercel AI SDK.
* **Enterprise Edge / Local AI Deployments:** Organisasi yang menjalankan LLM di perangkat *edge* atau infrastruktur lokal dengan keterbatasan RAM/VRAM.

---

## 3. Landasan Teori & Literatur Paper Terkait

Lethon dibangun di atas 8 paper riset bereputasi tinggi (2025–2026):

1. **Fourier Compressor: Frequency-Domain Visual Token Compression for VLMs** (`arXiv:2508.06038`)  
   *Inovasi:* Membuktikan non-uniformity distribusi informasi semantik pada domain frekuensi. Menggunakan 1D Fast Fourier Transform (FFT) dengan kompleksitas $\mathcal{O}(N \log N)$ untuk membuang redundansi frekuensi tinggi, menekan FLOPs inferensi hingga 83.8% tanpa parameter tambahan.
2. **FourierCompress: Layer-Aware Spectral Activation Compression** (`arXiv:2510.16418`)  
   *Inovasi:* Memanfaatkan *low-frequency smoothness* pada layer awal transformer untuk mengompresi aktivasi sekuens hingga 7.6x lebih hemat memori/bandwidth dengan konsentrasi energi pada koefisien spektral rendah.
3. **ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning** (`arXiv:2504.01296`)  
   *Inovasi:* Mengajarkan LLM untuk memotong langkah penalaran CoT berulang melalui RL dengan *iterative token limit*. Mengeliminasi *thought redundancy* hingga 50% pada benchmark AIME24 dengan penurunan akurasi hanya ~2%.
4. **Reinforcement Learning with Promising Tokens (RLPT)** (`arXiv:2602.03195`)  
   *Inovasi:* Membuktikan secara teoritis bahwa jalur penalaran valid terkonsentrasi pada *low-rank subspace*. Memisahkan keputusan strategis dari pembuatan token dengan menerapkan *masking* dinamis pada *promising tokens*, menekan varians gradien dan menstabilkan RL.
5. **Language-Guided Token Compression with Reinforcement Learning in LVLMs (TPRL)** (`arXiv:2603.13394v1`)  
   *Inovasi:* Memformulasikan kompresi token sekuensial sebagai *Markov Decision Process* (MDP). Menggunakan *self-supervised autoencoder* untuk kompresi state, *Learning from Demonstrations* (LfD) untuk *warm start*, dan PPO untuk optimasi reward performa vs efisiensi.
6. **Semantic-Anchor Compression (SAC) for LLMs** (`arXiv:2510.08907`)  
   *Inovasi:* Menggantikan pelatihan *autoencoding* yang kaku dengan agregasi informasi konteks langsung ke dalam *anchor tokens* menggunakan *learnable anchor embeddings* dan *bidirectional attention modification*.
7. **Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning** (`arXiv:2512.04359`)  
   *Inovasi:* Mencegah *entropy collapse* saat RL melakukan kompresi dengan memadukan *semantic entropy-guided curriculum learning* dan regulasi KL non-seragam pada token-token keputusan kritis.
8. **RL-based Token Pruning in ViTs: A Markov Game Approach** (`arXiv:2503.23459`)  
   *Inovasi:* Memformulasikan pemangkasan token antar layer sebagai *Markov Game* yang diselesaikan via *Multi-Agent Proximal Policy Optimization* (MAPPO).

---

## 4. Arsitektur Produk & Metode Hybrid (4-Stage Pipeline)

Metodologi Lethon mengawinkan teknik spektral, agregasi semantik, dan RL sekuensial ke dalam 4 tahapan pemrosesan:

```
[Input Context / Codebase / CoT Stream]
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 1: Spectral FFT Pass     │  --> Truncation noise frekuensi tinggi O(N log N)
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 2: Semantic Anchor (SAC) │  --> Aggregation ke anchor tokens via Bidirectional Attention
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 3: Subspace Masking (RLPT)│ --> Low-rank Promising Token action masking
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 4: Sequential MDP/GRPO   │  --> RL Trajectory & CoT Truncation Optimization
  └────────────────┬────────────────┘
                   │
                   ▼
[Compressed Context untuk Agent / IDE] (Hemat 50% - 66.7% Token)
```

### Detail Tahapan Metode:

### Stage 1: Fast Spectral Fourier Filtering
* **Prinsip Matematika:** Mengubah representasi embedding/aktivasi sekuens konteks $X \in \mathbb{R}^{N \times d}$ ke domain spektral menggunakan Fast Fourier Transform 1D.
* **Mekanisme:** Informasi utama konteks terkonsentrasi pada komponen spektral frekuensi rendah (*low-frequency energy concentration*). Frekuensi tinggi merepresentasikan fluktuasi lokal dan noise redundan.
* **Karakteristik:** Berjalan tanpa parameter tambahan (*parameter-free*) dalam skala kompleksitas $\mathcal{O}(N \log N)$, langsung memangkas 20%–30% volume token noise di awal secara instan.

### Stage 2: Contextual Semantic Anchor Aggregation (SAC)
* **Prinsip Matematika:** Konteks panjang dipetakan ke sejumlah *anchor tokens* yang bertindak sebagai pemuat informasi utama.
* **Mekanisme:** Menggunakan *learnable anchor embedding* dan *bidirectional attention modification*. Anchor tokens diizinkan melihat seluruh konteks secara dua arah (*bidirectional*) untuk menyerap *Key-Value (KV) representations* secara efisien sebelum diteruskan ke LLM.

### Stage 3: Dynamic Promising Token Subspace Masking (RLPT)
* **Prinsip Matematika:** Mengisolasi pilihan aksi agen RL ke dalam *low-rank subspace* dari prior semantik model dasar.
* **Mekanisme:** Daripada membiarkan RL mencari aksi di seluruh ruang kosa kata/token yang masif (yang memicu varians gradien tinggi), sistem menerapkan *dynamic masking matrix* untuk membatasi ruang keputusan hanya pada subset *promising tokens*.

### Stage 4: Sequential Decision MDP & CoT Length Truncation (TPRL & ThinkPrune)
* **Prinsip Matematika:** Formulasi *Markov Decision Process* $(S, A, P, R, \gamma)$ di mana state dinamis diwakili oleh latent code dari *token-wise autoencoder* ($d_l \ll d_v$).
* **Fungsi Reward $r_t$:**
  $$r_t = R_{\text{task}} + R_{\text{eff}}(t)$$
  - $R_{\text{task}}$: Mengukur retensi skor akurasi/penalaran downstream task.
  - $R_{\text{eff}}(t)$: Memberikan penalti efisiensi berdasarkan rasio token yang dibuang.
* **Iterative CoT Limits:** Untuk model penalaran panjang, diterapkan *iterative token limit* selama pelatihan RL. Pemikiran yang tidak selesai sebelum batas token mendapatkan reward nol, memaksa model belajar melakukan konsolidasi langkah berpikir (*compact logic execution*).

---

## 5. Spesifikasi Integrasi Produk (MCP Protocol & CLI)

### 5.1 Arsitektur Distribusi Zero-Proxy
Lethon beroperasi langsung di dalam proses agen tanpa membutuhkan server proxy terpisah. Integrasi dilakukan melalui dua mekanisme utama:

1. **Model Context Protocol (MCP) Server:**
   * Dijalankan via runner instan (`uvx` untuk Python / `npx` untuk Node.js).
   * Mendaftarkan MCP Tools bawaan:
     - `prune_context`: Memangkas berkas kode, dokumentasi, atau log raksasa sebelum dimasukkan ke context window.
     - `compress_cot`: Memotong dan merapikan alur pikir *Chain-of-Thought* berulang dari model reasoning.
2. **CLI Executable Wrapper:**
   * Pembungkus perintah terminal instan untuk mengeksekusi Claude Code atau CLI AI lainnya dengan mode kompresi aktif secara otomatis.

### 5.2 Spesifikasi Antarmuka MCP Tools
* **Tool Name:** `prune_context`
  - **Input Parameters:**
    - `context_text` (String): Teks/kode sumber yang akan dikompresi.
    - `compression_ratio` (Float): Target rasio retensi token (misal: 0.4 untuk kompresi 60%).
    - `preservation_mode` (Enum: `BALANCED`, `AGGRESSIVE_COT`, `SPECTRAL_FAST`).
  - **Output:**
    - `compressed_text` (String): Teks hasil pemangkasan semantik.
    - `metrics` (Object): Statistik jumlah token awal, token akhir, persentase penghematan, dan estimasi waktu eksekusi.

---

## 6. Target Key Performance Indicators (KPIs) & Benchmark

| Metrik | Baseline (Uncompressed) | Target Lethon |
| :--- | :--- | :--- |
| **Token Reduction Rate** | 0% | **50.0% – 66.7%** |
| **Inference FLOPs Reduction** | 0% | **> 54.0%** |
| **Average Accuracy Drop** | 0% | **< 0.7%** (MMBench, VQA-v2, GQA) |
| **CoT Thinking Step Reduction** | 0% | **50.0%** (AIME24 Benchmark) |
| **Execution Latency Impact** | - | **< 15ms overhead** per compression pass |

---

## 7. Roadmap Pengembangan & Strategi Rilis GitHub (100k Stars Strategy)

### Fase 1: Core Engine & Paper Benchmark Suite (Bulan 1)
* Implementasi modul matematika Fourier 1D-FFT dan SAC *anchor selection*.
* Pengembangan lingkungan simulasi RL (MDP & Subspace Masking).
* Publikasi benchmark reproduksibel pada dataset AIME24, MMBench, dan GSM8K.

### Fase 2: MCP Server & Packaging (`uvx`/`npx`) (Bulan 2)
* Pembuatan server MCP standar kompatibel dengan Claude Code, Cursor, Windsurf, dan Antigravity.
* Penerbitan paket ke PyPI (`uvx Lethon`) dan NPM (`npx Lethon`).
* Dokumentasi lengkap dengan gif animasi *live token savings*.

### Fase 3: Developer Dashboard & IDE Extension (Bulan 3)
* Pembuatan ekstensi VS Code / Cursor untuk visualisasi penghematan token secara real-time.
* Peluncuran kampanye repositori open-source GitHub dengan *showcase video* dan *benchmarking suite*.

---

*Dokumen PRD ini disusun sebagai panduan teknis bebas kode untuk persiapan pembacaan literatur dan pengerjaan repositori.*
