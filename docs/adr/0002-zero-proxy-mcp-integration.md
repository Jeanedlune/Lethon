# ADR-0002: Model Context Protocol (MCP) Integration Over Reverse Proxy

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture & Integration Team
- **Consulted:** PRD v1.0.0 Section 5, Anthropic MCP Specification

---

## 1. Context & Problem Statement
Alat kompresi prompt/konteks terdahulu (seperti prompt-caching proxies atau LLM gatekeepers) mengharuskan pengguna menjalankan daemon/server proxy lokal yang mencegat panggilan HTTP OpenAI/Anthropic API (`http://localhost:8080/v1/...`). Hal ini menimbulkan friksi tinggi bagi developer:
1. Masalah sertifikat SSL/TLS custom.
2. Konflik port lokal dan manajemen background process/daemon.
3. Sulit dikonfigurasi pada IDE yang tidak mengekspos base URL custom.

Lethon menargetkan distribusi viral open-source dengan adopsi tanpa friksi (*Zero-Friction / Zero-Proxy*).

## 2. Decision Drivers
- Kemudahan instalasi instan (`uvx lethon` atau `npx lethon`).
- Kompatibilitas bawaan dengan ekosistem modern: Claude Code, Cursor, Windsurf, dan Google Antigravity.
- Keamanan sandbox tanpa mengintersepsi kredensial API key developer.

## 3. Considered Options
1. **Option 1:** Local HTTP Reverse Proxy Server.
2. **Option 2:** Patching / Hooking SDK Python/Node.js secara runtime.
3. **Option 3:** Standard Model Context Protocol (MCP) Tools & Resources.

## 4. Decision Outcome
Dipilih **Option 3: Standard Model Context Protocol (MCP)** dengan CLI wrapper opsional.

### Rationale:
- MCP adalah standar terbuka yang didukung langsung oleh Cursor, Claude Code, Windsurf, dan Antigravity.
- Agen dapat memanggil tool `prune_context` dan `compress_cot` secara on-demand sebelum meletakkan berkas raksasa ke dalam context window.
- Eksekusi via `uvx lethon mcp` tidak membutuhkan instalasi permanen dan tidak memicu konflik port jaringan.

## 5. Consequences & Trade-offs
### Positive
- Zero network proxy overhead; proses berjalan langsung di stdio klien.
- Pengguna cukup menambahkan 1 baris konfigurasi di `mcpServers`.

### Negative / Mitigations
- Klien harus mendukung MCP. Untuk lingkungan non-MCP (seperti bash scripting biasa), disediakan CLI executable wrapper `lethon run`.
