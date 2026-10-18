# ADR-0001: 4-Stage Hybrid Spectral-RL Pipeline Architecture

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture & Research Team
- **Consulted:** PRD v1.0.0, arXiv:2508.06038, arXiv:2510.08907, arXiv:2602.03195, arXiv:2504.01296

---

## 1. Context & Problem Statement
Model penalaran panjang (misal DeepSeek-R1, OpenAI o1/o3-mini) dan AI agents (Claude Code, Cursor) memakan puluhan ribu token konteks dan langkah penalaran CoT (Chain-of-Thought). Metode kompresi heuristik satu tahap (seperti pemangkasan *cosine similarity* atau *attention weight thresholding*) gagal mempertahankan konsistensi penalaran logis dan sering kali memicu *accuracy drop* yang besar (> 5%).

Diperlukan arsitektur kompresi bertahap yang mampu memilah noise berfrekuensi tinggi dengan cepat tanpa parameter tambahan, memadatkan semantik ke anchor token, dan memangkas redundansi penalaran RL secara adaptif.

## 2. Decision Drivers
- **Token Reduction Target:** Mampu memangkas volume token antara 50.0% – 66.7%.
- **Accuracy Drop:** Harus di bawah 0.7% pada benchmark standar (MMBench, AIME24, GSM8K).
- **Latency Overhead:** Total pipeline latency overhead harus < 15ms.
- **Modularity:** Setiap tahap harus dapat diaktifkan atau dilewati sesuai profil beban kerja (`BALANCED`, `AGGRESSIVE_COT`, `SPECTRAL_FAST`).

## 3. Considered Options
1. **Option 1:** Single-Stage RL Agent (End-to-end token pruner).
2. **Option 2:** Pure Heuristic / Attention-Weight Dropping (Static FastV style).
3. **Option 3:** Hybrid 4-Stage Hierarchical Engine:
   - *Stage 1:* Fast Spectral Fourier 1D Filtering (Parameter-free $\mathcal{O}(N \log N)$).
   - *Stage 2:* Semantic-Anchor Compression (SAC) via Bidirectional Attention.
   - *Stage 3:* Dynamic Promising Token Subspace Masking (RLPT).
   - *Stage 4:* Sequential Decision MDP & Token Pruning (TPRL & SENT).

## 4. Decision Outcome
Dipilih **Option 3 (Hybrid 4-Stage Hierarchical Engine)**.

### Rationale:
- Stage 1 langsung membuang 20-30% token noise berfrekuensi tinggi dalam hitungan fraksi milidetik tanpa parameter (`O(N log N)`).
- Stage 2 mengonsolidasi representasi semantik luas ke anchor tokens sebelum masuk ke ranah keputusan token diskrit.
- Stage 3 memecahkan kendala klasik RL (ruang aksi kosa kata terlalu besar) dengan memproyeksikan aksi ke *low-rank subspace* untuk memproteksi sintaks kode.
- Stage 4 mengoptimasi pemangkasan token konteks sekuensial secara adaptif berdasarkan target reward akurasi vs efisiensi.

## 5. Consequences & Trade-offs
### Positive
- Mendapatkan kecepatan eliminasi noise di awal (Stage 1) sekaligus kecerdasan penalaran dinamis di akhir (Stage 4).
- Akurasi penalaran terjaga karena *low-rank subspace* menjaga token-token krusial dari pemangkasan yang keliru.

### Negative / Mitigations
- Kompleksitas integrasi antar stage memerlukan interface tensor dan state MDP yang terstandarisasi.
- *Mitigasi:* Dibangun modul `lethon.core.pipeline.CompressionPipeline` dengan kontrak input-output yang ketat.
