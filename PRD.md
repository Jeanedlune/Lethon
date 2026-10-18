# Product Requirement Document (PRD): Lethon
**System Name:** Lethon (Spectral-RL Context Compression Engine)  
**Target Release:** Q4 2026  
**Document Version:** 1.1.0  
**Target Distribution:** GitHub Open-Source (Goal: 100k Stars)  

---

## 1. Executive Summary & Vision

**Lethon** adalah engine pemangkasan token dan kompresi konteks AI berkecepatan tinggi yang dirancang khusus untuk era **Agentic AI** (Claude Code, Cursor, Windsurf, OpenAI Codex, dan Google Antigravity). 

Dengan mengombinasikan **Domain Frekuensi Fourier (1D-FFT)**, **Semantic Anchor Aggregation (SAC)**, dan **Reinforcement Learning Subspace Masking (RLPT/TPRL)**, Lethon mampu memangkas volume token codebase, log terminal, dan git diff hingga **50%–66.7%** dengan penurunan akurasi kurang dari **0.7%** dan latensi eksekusi di bawah **5 milidetik**.

Sistem diintegrasikan secara **Zero-Friction / Zero-Proxy** melalui **Model Context Protocol (MCP)** dan didistribusikan langsung via `uvx` / `npx`, memungkinkan pengembang dan enterprise menghemat biaya API serta latensi secara instan tanpa perlu mengubah *workflow* atau menjalankan proxy server di background.

---

## 2. Problem Statement & Target Market

### 2.1 Permasalahan Utama (Pain Points)
1. **Explosive Context & Latency Overhead:** Agen otomatis membaca ribuan baris file dan log trace, memicu *memory bottleneck* pada context window serta memperlambat waktu respons.
2. **Tagihan API Membengkak:** Developer dan tim engineer yang menggunakan agen otomatis mengalami lonjakan konsumsi token hingga jutaan token per hari untuk membaca repositori kode dan terminal output.
3. **Kegagalan Metode Pruning Heuristik Tradisional:** Pendekatan *static pruning* biasa merusak sintaks kode (menghapus kurung kurawal, indentasi, atau deklarasi fungsi penting).

### 2.2 Target Audience
* **AI Developers & Power Users:** Pengguna Claude Code, Cursor, Windsurf, Codex, dan Antigravity yang menginginkan alur kerja lebih kencang dan hemat token.
* **AI Framework Developers:** Tim yang membangun sistem RAG, Multi-Agent, dan aplikasi LLM berbasis LangChain, LlamaIndex, atau Vercel AI SDK.
* **Enterprise Edge / Local AI Deployments:** Organisasi yang menjalankan LLM di perangkat *edge* atau infrastruktur lokal dengan keterbatasan RAM/VRAM.

---

## 3. Landasan Teori & Literatur Paper Terkait

Lethon dibangun di atas 7 paper riset bereputasi tinggi (2025–2026):

1. **Fourier Compressor: Frequency-Domain Visual Token Compression for VLMs** (`arXiv:2508.06038`)  
   *Inovasi:* Membuktikan non-uniformity distribusi informasi semantik pada domain frekuensi. Menggunakan 1D Fast Fourier Transform (FFT) dengan kompleksitas $\mathcal{O}(N \log N)$ untuk membuang redundansi frekuensi tinggi, menekan FLOPs inferensi hingga 83.8% tanpa parameter tambahan.
2. **FourierCompress: Layer-Aware Spectral Activation Compression** (`arXiv:2510.16418`)  
   *Inovasi:* Memanfaatkan *low-frequency smoothness* pada layer awal transformer untuk mengompresi aktivasi sekuens hingga 7.6x lebih hemat memori/bandwidth dengan konsentrasi energi pada koefisien spektral rendah.
3. **Reinforcement Learning with Promising Tokens (RLPT)** (`arXiv:2602.03195`)  
   *Inovasi:* Membuktikan secara teoritis bahwa jalur representasi valid terkonsentrasi pada *low-rank subspace*. Memisahkan keputusan strategis dari pembuatan token dengan menerapkan *masking* dinamis pada *promising tokens*, menekan varians gradien dan mengunci sintaks kritis.
4. **Language-Guided Token Compression with Reinforcement Learning in LVLMs (TPRL)** (`arXiv:2603.13394v1`)  
   *Inovasi:* Memformulasikan kompresi token sekuensial sebagai *Markov Decision Process* (MDP). Menggunakan *self-supervised autoencoder* untuk kompresi state, *Learning from Demonstrations* (LfD) untuk *warm start*, dan PPO untuk optimasi reward performa vs efisiensi.
5. **Semantic-Anchor Compression (SAC) for LLMs** (`arXiv:2510.08907`)  
   *Inovasi:* Menggantikan pelatihan *autoencoding* yang kaku dengan agregasi informasi konteks langsung ke dalam *anchor tokens* menggunakan *learnable anchor embeddings* dan *bidirectional attention modification*.
6. **Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning** (`arXiv:2512.04359`)  
   *Inovasi:* Mencegah *entropy collapse* saat RL melakukan kompresi dengan memadukan *semantic entropy-guided curriculum learning* dan regulasi KL non-seragam pada token-token keputusan kritis.
7. **RL-based Token Pruning in ViTs: A Markov Game Approach** (`arXiv:2503.23459`)  
   *Inovasi:* Memformulasikan pemangkasan token antar layer sebagai *Markov Game* yang diselesaikan via *Multi-Agent Proximal Policy Optimization* (MAPPO).

---

## 4. Arsitektur Produk & Metode Hybrid (3-Stage Specialized Pipeline)

Metodologi Lethon mengawinkan teknik spektral, agregasi semantik, dan RL subspace masking:

```
[Input: Large Codebase / Terminal Logs / Git Diff]
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 1: Spectral FFT Pass     │  --> Truncation noise frekuensi tinggi O(N log N)
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 2: Semantic Anchor (SAC) │  --> Aggregation & Error Isolation
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Stage 3: Subspace Masking (RLPT)│ --> Syntax Protection & Invariant Locking
  └────────────────┬────────────────┘
                   │
                   ▼
[Compressed Context untuk Agent / IDE] (Hemat 50% - 66.7% Token, < 5ms latency)
```

---

## 5. Spesifikasi Integrasi Produk (MCP Protocol & CLI)

### 5.1 Arsitektur Distribusi Zero-Proxy
Lethon beroperasi langsung di dalam proses agen tanpa membutuhkan server proxy terpisah. Integrasi dilakukan melalui Model Context Protocol (MCP) dan CLI.

### 5.2 Spesifikasi Antarmuka MCP Tools
1. **`prune_code`**: Memangkas berkas source code besar sambil menjaga keutuhan 100% deklarasi fungsi, struct, class, dan syntax brackets.
2. **`prune_logs`**: Memangkas log terminal spam dan mengisolasi stack trace error utama.
3. **`prune_diff`**: Membersihkan file lockfile dan binary asset dari output `git diff`.
4. **`prune_context`**: Universal router untuk teks umum.

---

## 6. Target Key Performance Indicators (KPIs) & Benchmark

| Metrik | Baseline (Uncompressed) | Target Lethon |
| :--- | :--- | :--- |
| **Token Reduction Rate** | 0% | **50.0% – 66.7%** |
| **Inference FLOPs Reduction** | 0% | **> 54.0%** |
| **Average Accuracy Drop** | 0% | **< 0.7%** |
| **Execution Latency Impact** | - | **< 5ms overhead** per compression pass |
| **Syntax Invariant Retention**| 0% | **100% (Zero Broken Code)** |

---

## 7. Roadmap Pengembangan & Strategi Rilis GitHub (100k Stars Strategy)

### Fase 1: Core Engine & Benchmark Suite (Bulan 1)
* Implementasi modul matematika Fourier 1D-FFT dan SAC *anchor selection*.
* Pengembangan proteksi sintaks RLPT Subspace Masking.
* Pengujian pada repositori kode nyata dan log build compiler.

### Fase 2: MCP Server & Packaging (`uvx`/`npx`) (Bulan 2)
* Pembuatan server MCP standar kompatibel dengan Claude Code, Cursor, Windsurf, dan Antigravity.
* Penerbitan paket ke PyPI (`uvx lethon`) dan NPM (`npx lethon`).
* Dokumentasi lengkap dengan gif animasi *live token savings*.

### Fase 3: Developer Dashboard & IDE Extension (Bulan 3)
* Pembuatan ekstensi VS Code / Cursor untuk visualisasi penghematan token secara real-time.
* Peluncuran repositori open-source GitHub.
