# ADR-0004: Low-Rank Dynamic Action Masking (RLPT) for Policy Optimization

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon RL & Optimization Team
- **Consulted:** arXiv:2602.03195 (RLPT), arXiv:2512.04359

---

## 1. Context & Problem Statement
Ketika menerapkan Reinforcement Learning (PPO/GRPO) untuk memilih token mana yang harus dipertahankan atau dipangkas, ruang pencarian aksi diskrit yang sangat besar (seukuran vocab atau ratusan baris CoT) menghasilkan varians gradien yang sangat tinggi. Hal ini kerap memicu ketidakstabilan pelatihan (*policy collapse*) atau agen memotong token-token kunci (seperti operator matematika atau indentasi kode).

## 2. Decision Drivers
- Stabilitas konvergensi RL.
- Proteksi terhadap pemangkasan token bernilai semantik tinggi.
- Kemampuan generalisasi di lintas tugas penalaran (penalaran matematika vs pemahaman kode).

## 3. Considered Options
1. **Option 1:** Unconstrained Action Space PPO (semua token bebas dipilih untuk dipangkas).
2. **Option 2:** Static rule-based heuristic masking (misal: proteksi kata kunci tertentu saja).
3. **Option 3:** Dynamic Low-Rank Subspace Masking (RLPT paradigm).

## 4. Decision Outcome
Dipilih **Option 3: Dynamic Low-Rank Subspace Masking**.

### Rationale:
- Jalur penalaran valid terbukti matematis terkonsentrasi pada *low-rank subspace* dari representasi semantik.
- Matriks masking dinamis memproyeksikan probabilitas aksi hanya pada subset token yang memiliki kontribusi varians informasi signifikan (*promising tokens*), membatasi agen agar tidak memotong token kritis dan secara drastis menurunkan varians gradien.

## 5. Consequences & Trade-offs
### Positive
- Pelatihan kebijakan RL lebih cepat konvergen dan stabil.
- Zero degradasi pada token krusial sintaksis pemrograman.

### Negative / Mitigations
- Diperlukan komputasi proyeksi subspace (SVD / low-rank linear projection) sebelum pengambilan keputusan RL.
