<div align="center">

# ⚡ Lethon

**Spectral-RL Hybrid Context Compression Engine**  
*Cut Token Overhead & CoT Reasoning Length by 50%–66.7% with < 0.7% Accuracy Drop.*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io/)

</div>

---

## 🚀 Key Highlights

- **4-Stage Hybrid Engine:** Mengombinasikan 1D-FFT Spektral ($\mathcal{O}(N \log N)$), Semantic Anchor Aggregation (SAC), Subspace Action Masking (RLPT), dan MDP CoT Pruning (ThinkPrune).
- **Zero-Proxy / Zero-Friction:** Berjalan langsung via **Model Context Protocol (MCP)** dengan `uvx lethon` atau `npx lethon`. Tidak perlu repot mengurus reverse proxy lokal.
- **Agentic & IDE Ready:** Terintegrasi langsung dengan Claude Code, Cursor, Windsurf, dan Google Antigravity.

---

## 📂 Repository Structure

- `lethon/`: Core Python package engine, MCP server, dan CLI interface.
- `docs/adr/`: [Architecture Decision Records](docs/adr/README.md) (ADR-0001 s/d ADR-0004).
- `docs/dec/`: [Design Engineering Concepts](docs/dec/README.md) (DEC-01 s/d DEC-04).
- `research/`: Kumpulan paper PDF asli dan hasil parsing teks/gambar per paper.
- `scripts/`: Tool otomasi riset (ekstraksi PDF, downloader).

---

## ⚡ Quickstart

```bash
# Jalankan via UVX tanpa instalasi
uvx lethon mcp

# Atau install lokal
pip install -e .
lethon --help
```

---

## 📖 Technical Documentation

Lihat [PRD.md](PRD.md) untuk roadmap dan spesifikasi produk lengkap, serta [docs/adr/](docs/adr/) & [docs/dec/](docs/dec/) untuk arsitektur teknis dan landasan teoritis.