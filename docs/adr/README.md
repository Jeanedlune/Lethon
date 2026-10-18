# Architecture Decision Records (ADR)

Index dan ringkasan keputusan arsitektur strategis untuk pengembangan **Lethon**.

| ID | Title | Status | Date | Decision Summary |
| :--- | :--- | :--- | :--- | :--- |
| [ADR-0001](0001-hybrid-spectral-rl-architecture.md) | 4-Stage Hybrid Spectral-RL Pipeline | Accepted | 2026-10-18 | Menggabungkan Fourier 1D, SAC, Subspace Masking, dan MDP CoT Pruning untuk kompresi hingga 66.7%. |
| [ADR-0002](0002-zero-proxy-mcp-integration.md) | Model Context Protocol (MCP) Over Proxy | Accepted | 2026-10-18 | Menggunakan standar MCP stdio/SSE dibanding proxy HTTP/SOCKS untuk zero-proxy distribution via `uvx`/`npx`. |
| [ADR-0003](0003-parameter-free-spectral-fft.md) | Parameter-Free 1D-FFT vs Heuristic Pruning for Stage 1 | Accepted | 2026-10-18 | Memilih 1D-FFT/DCT-II $\mathcal{O}(N \log N)$ dibanding Cosine Similarity & Attention Heuristics: latensi $< 3$ ms pada CPU, zero-parameter, tanpa torch berat. |
| [ADR-0004](0004-action-space-subspace-masking.md) | Low-Rank Dynamic Action Masking (RLPT) | Accepted | 2026-10-18 | Membatasi ruang aksi RL ke subspace promising tokens guna menekan varians gradien. |
| [ADR-0005](0005-language-selection-python-to-rust.md) | Dual-Track Strategy (Python Research to Rust Core) | Accepted | 2026-10-18 | Python untuk validasi riset awal; porting ke single Rust binary saat siap rilis produksi. |
| [ADR-0006](0006-layer-aware-vs-uniform-spectral-filtering.md) | Layer-Aware vs Uniform Spectral Filtering for Transformer Sequences | Accepted | 2026-10-18 | Menetapkan Layer 1 sebagai split/compression point utama dengan polynomial scheduling $\rho(l)$ dan rekonstruksi simetri konjugat Hermitian: reduksi payload hingga 7.6x – 10.3x, akurasi terjaga $(< 0.3\%$ loss). |
| [ADR-0007](0007-semantic-anchor-vs-autoencoder-and-token-pruning.md) | Semantic Anchor (SAC) vs Rigid Autoencoders and Heuristic Pruning | Accepted | 2026-10-18 | Memilih in-context semantic anchors dengan bidirectional attention dan pelatihan autoencoding-free: eliminasi konflik ortogonalitas gradien $\mathcal{L}_{AE} \perp \mathcal{L}_{LM}$, preservasi manifold KV tanpa distribution shift, hemat 1.19x FLOPs, dan unggul hingga +23.5% F1 atas ICAE. |
