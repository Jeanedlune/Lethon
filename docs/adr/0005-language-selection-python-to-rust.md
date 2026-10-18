# ADR-0005: Dual-Track Language Strategy (Python Research to Rust Production Core)

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture & Performance Team
- **Consulted:** ADR-0001, ADR-0002, PRD Section 7

---

## 1. Context & Problem Statement
Lethon dirancang sebagai engine pemangkasan konteks berkecepatan tinggi untuk developer tool (Claude Code, Cursor, Windsurf).
Terdapat dua kebutuhan yang saling bertolak belakang:
1. **Kebutuhan Riset & Eksplorasi (Fase Awal):** Cepat memvalidasi 8 paper riset, fleksibilitas tensor matrix, ekosistem tokenizer HuggingFace, dan simulasi training RL.
2. **Kebutuhan Pengguna / Developer Experience (Fase Produksi):** Wajib super enteng (< 25MB), instan (< 1ms startup time), zero-dependency, dan tidak boleh membuat IDE lag atau laptop panas.

Jika kita memaksakan runtime Python penuh (terutama dependensi PyTorch ~1.5GB) ke laptop user biasa, user akan enggan menggunakan Lethon karena berat dan lambat.

## 2. Decision Drivers
- **Zero Friction Adoption:** Startup time instant (< 15ms total overhead).
- **Lightweight Distribution:** Ukuran paket < 25MB (single binary tanpa beban PyTorch ratusan MB).
- **Research Agility:** Kemudahan mereproduksi rumus matematika paper (FFT, SVD, PPO/GRPO).

## 3. Decision Outcome: The Dual-Track Strategy
Diputuskan menggunakan **Dual-Track Language Strategy (Python-First Research, Rust-Core Production)**:

### Fase 1: Riset & Algorithmic Validation (Python Track)
- **Scope:** Validasi matematika di `lethon/core/` (Fourier 1D-FFT, SAC, Subspace Masking, MDP CoT).
- **Rule:** Runtime inferensi untuk MCP **dilarang bergantung pada `torch` penuh**. Cukup gunakan `numpy`/`scipy` ringan agar cold-start cepat. `torch` hanya diizinkan untuk script training/benchmarking riset.
- **Milestone Selesai:** Algoritma terbukti memangkas 50%–66.7% token dengan akurasi drop < 0.7% pada benchmark AIME24/GSM8K.

### Fase 2 & 3: Transisi Produksi ke Rust (Rust Track)
Kapan Rust resmi masuk dan menggantikan engine utama? **Tepat saat masuk fase produksi & rilis publik (Bulan 2-3 / Fase 2-3 PRD):**

1. **Rust Core Porting (`crates/lethon-core`):**
   - Mengubah komputasi FFT 1D dan matrix projection dari NumPy ke Rust murni menggunakan SIMD / portable Rust math.
   - Keuntungan: Latensi turun dari ~10ms ke **< 0.8ms**, RAM usage cuma hitungan megabyte.
2. **PyO3 Python Bindings (`crates/lethon-py`):**
   - Python package `lethon` tetap ada di PyPI, tetapi engine matematis di dalamnya di-compile dari Rust binary (`.so` / `.pyd`) agar tetap secepat kilat.
3. **Standalone Standalone CLI & MCP Binary (`crates/lethon-cli`):**
   - Untuk pengguna `npx` dan `uvx`, didistribusikan single binary Rust (~5–10MB).
   - User Cursor/Claude Code bisa langsung jalan tanpa perlu install Python runtime sama sekali di sistem mereka.

---

## 4. Kriteria Pemicu Migrasi ke Rust (Trigger Checklist)
Migrasi ke Rust dimulai ketika:
- [ ] Pipeline 4-Stage di Python sudah lolos validasi benchmark akurasi (< 0.7% drop).
- [ ] Kontrak input/output tensor dan batas CoT sudah stabil (tidak sering berubah skema).
- [ ] Persiapan rilis packaging GitHub v1.0.0 (distribusi massal via `npx` / `uvx`).

---

## 5. Consequences & Trade-offs
### Positive
- **Kecepatan Riset:** Eksplorasi paper tidak terhambat oleh *strict borrow checker* Rust di awal.
- **Kecepatan User:** Pengguna akhir mendapatkan binary Rust super enteng tanpa kompromi performa.

### Negative / Mitigations
- Perlu penulisan ulang kode kernel Stage 1-3 dari Python ke Rust saat rilis produksi.
- *Mitigasi:* Arsitektur dibuat modular per stage (DEC-01 s/d DEC-04) dengan input/output test vectors yang jelas, sehingga porting ke Rust tinggal mengikuti test suite yang sudah ada.
