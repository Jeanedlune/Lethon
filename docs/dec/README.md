# Design Engineering Concepts (DEC)

Index dokumentasi teknis, landasan matematis, dan spesifikasi arsitektur modul untuk **Lethon Engine**.

| Dokumen | Nama Modul | Tahap Pipeline | Paper Acuan Utama | Deskripsi Teknis Singkat |
| :--- | :--- | :--- | :--- | :--- |
| [DEC-01](dec-01-spectral-filtering-math.md) | Fast Spectral Fourier Filtering & Layer-Aware Activation Compression | Stage 1 | arXiv:2508.06038, arXiv:2510.16418 | Formulasi 1D/2D FFT/DCT-II ortonormal $\mathcal{O}(N \log N)$, analisis kehalusan & konsentrasi energi spektral layer awal transformer, rekonstruksi simetri konjugat Hermitian, scheduling layer-aware, serta implementasi NumPy/SciPy (< 3ms CPU). |
| [DEC-02](dec-02-semantic-anchor-aggregation.md) | Contextual Semantic Anchor Aggregation (SAC) | Stage 2 | arXiv:2510.08907 | Pemetaan konteks panjang ke sejumlah *anchor tokens* melalui modifikasi atensi dua arah (*bidirectional attention*) dan learnable anchor embeddings. |
| [DEC-03](dec-03-rlpt-promising-token-masking.md) | Dynamic Promising Token Subspace Masking (RLPT) | Stage 3 | arXiv:2602.03195 | Isolasi ruang aksi RL ke dalam *low-rank subspace* dari prior semantik untuk menekan varians gradien dan menstabilkan optimasi kebijakan kompresi. |
| [DEC-04](dec-04-cot-iterative-pruning.md) | Sequential Decision MDP & CoT Length Truncation | Stage 4 | arXiv:2504.01296, arXiv:2603.13394 | Formulasi MDP untuk pemangkasan alur pikir penalaran panjang (*Chain-of-Thought*) berulang dengan *iterative token limits* dan reward seimbang akurasi vs efisiensi. |
