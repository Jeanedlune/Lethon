<div align="center">

# Lethon

**Spectral-RL Context Compression Engine for AI Coding Agents**  
*Cut Token Volume in Codebases, Logs, & Git Diffs by 50%–66.7% with Zero Perceived Latency (< 5ms).*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io/)

</div>

---

## Key Highlights

- **Specialized Pruning Engines:**
  - `prune_code`: 1D-FFT noise filtering + RLPT Subspace Masking to protect 100% of syntax invariants (functions, classes, brackets).
  - `prune_logs`: Semantic Anchor (SAC) error isolation to strip spam logs and capture root-cause stack traces.
  - `prune_diff`: Sequential MDP filtering to eliminate bloated lockfiles and generated assets from git diffs.
  - `prune_context`: Universal smart router for general files.
- **Zero-Proxy & Parameter-Free:** No heavy PyTorch model weights or daemon proxy. Runs in milliseconds on CPU.
- **MCP Native:** Integrated with Claude Code, Cursor, Windsurf, OpenAI Codex, and Google Antigravity.

---

## Repository Structure

- `lethon/`: Core Python package engine, MCP server, and CLI interface.
- `docs/adr/`: [Architecture Decision Records](docs/adr/README.md).
- `docs/dec/`: [Design Engineering Concepts](docs/dec/README.md).
- `research/`: Original research papers (PDFs) and extracted text/diagrams.
- `tests/`: Complete unit and integration test suite.

---

## Quickstart

```bash
# Run MCP server via UVX
uvx lethon mcp

# Or install locally
pip install -e .
lethon --help
```

---

## Technical Documentation

See [PRD.md](PRD.md) for full specifications, and [docs/adr/](docs/adr/) & [docs/dec/](docs/dec/) for technical decisions and theoretical foundations.