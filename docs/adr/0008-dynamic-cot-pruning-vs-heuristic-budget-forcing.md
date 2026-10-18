# ADR-0008: Dynamic RL CoT Pruning (ThinkPrune) vs Heuristic Budget-Forcing and Static Text Truncation for Stage 4

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** Lethon Architecture Guild & Algorithmic Core Team
- **Consulted Literature:**
  - `arXiv:2504.01296`: *THINKPRUNE: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning* (Hou et al., UCSB / MIT-IBM / MIT CSAIL, 2025)
  - `arXiv:2501.19393`: *s1: Simple test-time scaling* (Muennighoff et al., 2025)
  - `arXiv:2412.20993`: *Efficiently Serving LLM Reasoning Programs with Certaindex* (Fu et al., 2024)
  - `arXiv:2412.21187`: *Do Not Think That Much for 2+3=? On the Overthinking of o1-like LLMs* (Chen et al., 2024)
  - `arXiv:2603.13394v1`: *Language-Guided Token Compression with Reinforcement Learning in LVLMs (TPRL)* (2026)
  - `PRD.md`: *Lethon System Requirements & 4-Stage Architecture (Stage 4 Sequential MDP & CoT Length Truncation)*
  - `docs/dec/dec-04-cot-iterative-pruning.md`: *Technical & Mathematical Specifications of Stage 4*

---

## 1. Context & Problem Statement

Pada arsitektur engine kompresi konteks Lethon ([PRD.md](../../PRD.md)), setelah input awal dibersihkan dari noise frekuensi tinggi oleh **Stage 1 (Spectral FFT)**, dipadatkan ke dalam token jangkar oleh **Stage 2 (Semantic Anchors - SAC)**, dan diproyeksikan ke subruang aksi prior oleh **Stage 3 (Subspace Masking - RLPT)**, sistem menangani pemrosesan model penalaran panjang (*Long-Reasoning LLMs* seperti DeepSeek-R1, OpenAI o1/o3, DeepScaleR, dan QwQ-32B) pada **Stage 4**.

Model penalaran panjang yang dilatih dengan *Reinforcement Learning with Verifiable Rewards* (RLVR) menunjukkan kemampuan luar biasa dalam memecahkan soal kompleks berkat munculnya perilaku *self-reflection* dan verifikasi eksploratif. Namun, ketiadaan batasan efisiensi pada reward DeepSeek-R1 menciptakan fenomena **"Overthinking"**:
- Model sering menghasilkan 10.000 hingga 15.000+ token per jawaban pada benchmark matematika dan kode (seperti MATH-500 dan AIME24).
- Sebagian besar token tersebut dihabiskan untuk membaca ulang soal, memparafrase secara bertele-tele, mengekspresikan keraguan verbal berulang (*"Wait, let me double check...", "Hold on, another thought..."*), serta menjalankan loop verifikasi hingga 4–5 kali untuk masalah yang solusinya sudah ditemukan pada percobaan pertama.
- Hal ini menimbulkan ledakan ukuran KV-Cache (*memory footprint bottleneck*), menurunkan kecepatan inferensi (*tokens per second*), dan melipatgandakan tagihan API bagi pengembang agen AI.

Untuk mengatasi pemborosan token penalaran ini, tim engineering dihadapkan pada pilihan arsitektural: **Bagaimana cara memangkas panjang Chain-of-Thought (CoT) hingga 50% tanpa merusak akurasi logika konklusif dan tanpa mengorbankan keterbacaan (*readability*) alur penalaran?**

---

## 2. Decision Drivers

- **Preservasi Akurasi Penalaran Ekstrem (AIME24 Drop $\le 2\% - 2.5\%$):** Menjaga ketelitian logika pada tugas-tugas kompetisi matematika tingkat lanjut (AIME, OlympiadBench, MATH-500).
- **Eliminasi "Truncation Blindness":** Menghindari pemotongan paksa yang membunuh proses berpikir di tengah kalkulasi aktif.
- **Efisiensi Token Agresif (Penghematan $\ge 50\%$ Volume CoT):** Memangkas separuh pemikiran berulang pada model 1.5B hingga 32B.
- **Adaptabilitas Kesulitan Tugas (*Difficulty-Aware Pruning*):** Model harus secara fleksibel memangkas token secara masif pada soal sederhana, namun tetap mempertahankan kemampuan berpikir mendalam (*deep thinking*) pada soal yang sangat rumit.
- **Preservasi Keterbacaan & Distribusi Bahasa (Zero Entropy / Language Collapse):** Mencegah fenomena degradasi bahasa (seperti bahasa campuran atau simbol kriptik pada R1-Zero), dibuktikan oleh kestabilan nilai *perplexity* (PPL).
- **Zero Runtime Inference Overhead:** Tidak membutuhkan jaringan eksternal (*critic/verifier model*) di background saat deployment produksi.

---

## 3. Considered Options

### Option 1: Heuristic Test-Time Budget-Forcing / Early Exit (s1 Paradigm)
- Model dibiarkan beroperasi dengan bobot asli tanpa pelatihan ulang.
- Saat proses autoregresif mencapai kuota token tertentu ($L$), sistem memotong inferensi secara paksa dengan menginjeksi token penutup `</think>` dan memaksa model merangkum jawaban melalui prompt instan `\n\n**Final Answer:**\n\n`.
- *Karakteristik:* Sangat mudah diimplementasikan, zero-training cost.

### Option 2: Static Rule-Based Regex Stripping & Entropy Cutoff (LLMLingua Heuristic)
- Menganalisis teks CoT yang dihasilkan secara post-processing menggunakan daftar aturan regex (misal menghapus kalimat yang diawali *"Wait"*, *"Let me double-check"*) atau menghapus token dengan entropi rendah menggunakan model terdistilasi kecil.
- *Karakteristik:* Menghilangkan kata-kata tertentu dari log keluaran, tetapi tidak mengurangi biaya komputasi KV-cache selama proses generasi LLM.

### Option 3: Dynamic RL CoT Pruning with Iterative Length Clipping (ThinkPrune Paradigm - Selected)
- Melanjutkan pelatihan RL (menggunakan algoritma GRPO) pada model penalaran panjang dengan menambahkan batasan kuota panjang token ($L$).
- Menerapkan **Strict Length Clipping:** Trajektori yang dihasilkan dipotong pada panjang $L$ (`clip(Y, L)`). Jika jawaban belum selesai atau terpotong sebelum tag `</answer>`, reward dinyatakan **0**.
- Menggunakan kurikulum **Iterative Length Pruning Schedule** ($L_1 > L_2 > \dots > L^*$, misal $4k \to 3k \to 2k$) dengan kriteria checkpoint selection berbasis retensi akurasi validasi (toleransi drop $\le 10\%$ pada AIME22/AIME23 dengan panjang terpendek).
- *Karakteristik:* Model secara intrinsik belajar mengoptimalkan alur logika, memadatkan penalaran, dan mengeliminasi langkah periferal sejak token pertama.

---

## 4. Evaluasi Komparatif & Trade-off Analysis

| Kriteria Evaluasi | Option 1: Test-Time Budget-Forcing (s1) | Option 2: Static Regex / Entropy Stripping | Option 3: Dynamic RL ThinkPrune (Lethon Stage 4) |
| :--- | :--- | :--- | :--- |
| **Akurasi di Bawah Budget Ketat ($L \le 2000$)** | **Anjlok Katastropik** (Penurunan $> 20\% - 40\%$) | Tidak relevan (Model tetap generate penuh) | **Terjaga Sangat Kuat** (Drop hanya ~2.3% AIME24) |
| **Penanganan "Truncation Blindness"** | Gagal total (Terpotong saat kalkulasi berjalan) | Gagal (Konteks terputus tidak koheren) | **Sempurna** (Model menyelesaikan logika sebelum $L$) |
| **Penghematan KV-Cache & Waktu Generasi** | Parsial (Sering menghasilkan jawaban salah) | **Nol** (Generasi asli tetap 15k token) | **Maksimal** (Penghematan token $50\% - 65.5\%$) |
| **Performa pada Model Unsaturated (R1-1.5B)** | Penurunan tajam | Tidak ada perbaikan logika | **Meningkat (+0.6% Akurasi, -65.5% Token)** |
| **Preservasi Deep Thinking pada Soal Sulit** | Hilang (Terbatas kaku pada limit) | N/A | **Adaptif** (Otomatis berekspansi di AIME24) |
| **Keterbacaan Bahasa (Perplexity)** | Terpotong tidak tuntas | Rusak sintaksisnya | **Sempurna** (PPL 1.90 vs 1.91 baseline) |
| **Runtime Overhead saat Inferensi** | Ringan | Menambah latensi parsing teks | **Nol** (Internal dalam bobot inferensi) |

### Bukti Empiris dari Benchmark ThinkPrune (*arXiv:2504.01296*):
1. **DeepSeek-R1-Distill-Qwen-1.5B pada AIME24:**
   - Baseline: Akurasi $29.4\%$, Panjang $15.484$ token.
   - Budget-Forcing (Option 1 @ 2k budget): Akurasi jatuh ke $<10\%$ karena model kehabisan kuota saat parafrase soal.
   - ThinkPrune Iterative ($4k \to 3k \to 2k$): Akurasi **$27.1\%$** (hanya turun $2.3\%$), Panjang terpangkas ke **$5.631$ token** (**$-63.6\%$**).
   - Rata-rata 4 Benchmark (MATH-500, AIME24, AMC23, OlympiadBench): Akurasi **meningkat dari $56.8\%$ menjadi $57.4\%$** dengan reduksi token sebesar **$65.5\%$** (dari $10.355$ ke $3.574$ token).
2. **DeepScaleR-1.5B-Preview:**
   - Baseline: Akurasi $65.7\%$, Panjang $5.914$ token.
   - ThinkPrune Iterative: Akurasi $63.2\%$ (hanya drop $2.5\%$), Panjang $3.370$ token (**$-43.0\%$**).
3. **QwQ-32B:**
   - Baseline: Akurasi $85.6\%$, Panjang $8.763$ token.
   - ThinkPrune Iterative: Akurasi $82.4\%$ (drop $3.2\%$), Panjang $4.494$ token (**$-48.7\%$**).

---

## 5. Decision Outcome & Rationale

Tim arsitektur memutuskan untuk memilih **Option 3: Dynamic RL CoT Pruning with Iterative Length Clipping (ThinkPrune Paradigm)** sebagai standar metodologi resmi untuk **Stage 4 Sequential MDP & CoT Length Truncation** pada Lethon.

### Rationale Utama:

1. **Resolusi "Truncation Blindness" Melalui Adaptasi Kebijakan Intrinsik:**  
   Metode heuristik (Option 1) memperlakukan model sebagai kotak hitam statis dan memotong proses berpikir dari luar. Hal ini secara fundamental cacat karena model tidak menyadari sisa budget-nya hingga token delimiter disuntikkan secara paksa. Sebaliknya, ThinkPrune melatih model dengan sinyal prompt eksplisit (*"The output of the assistant should be within L tokens"*) dan penalti reward nol pada clipping. Model belajar memprioritaskan komputasi esensial sejak token pertama.
2. **Pergeseran Alokasi Budget dari Periferal ke Logika Inti:**  
   Analisis perilaku membuktikan bahwa ThinkPrune tidak menghilangkan logika esensial:
   - Frase keraguan berulang (*"Wait"*, *"Hold on"*) anjlok hingga $70\%$.
   - Proporsi langkah komputasi ekspresi dan aplikasi teorema justru naik dari $32.3\%$ ke $46.9\%$.
   - Verifikasi berulang 4–5 kali diringkas menjadi 1 putaran verifikasi terarah yang efisien.
3. **Keunggulan Kurikulum Iteratif ($L_1 > L_2 > \dots > L^*$):**  
   Penurunan limit token secara langsung (one-shot 2k) memicu shock gradient yang merusak stabilitas model, terutama pada arsitektur besar. Kurikulum bertahap ($4k \to 3k \to 2k$) memberikan transisi halus yang memungkinkan model memulihkan akurasi dan menghasilkan kompresi yang lebih dalam (pada DeepScaleR, iterative $2k$ unggul $+1.5\%$ dibanding one-shot $2k$).
4. **Generalisasi Adaptif terhadap Tingkat Kesulitan Soal:**  
   Ketika dievaluasi pada runtime dengan token limit terbuka, model yang dilatih ThinkPrune tidak terjebak pada panjang tetap. Pada soal sederhana (MATH-500), model memangkas lebih dari $60\%$ token; namun pada soal AIME24 yang rumit, model secara adaptif menggunakan hingga $5.600 - 7.600$ token untuk menuntaskan solusi.
5. **Preservasi Keterbacaan Penuh Tanpa Degradasi Bahasa:**  
   Tidak seperti varian R1-Zero tanpa regulasi, perplexity model pasca-ThinkPrune tetap identik dengan model dasar ($1.90$ vs $1.91$), menjamin output tetap berupa bahasa alami yang terstruktur dan mudah di-audit oleh pengembang dalam toolchain agen AI.

---

## 6. Implementation Guidelines & Architectural Constraints in Lethon

1. **Kompatibilitas Standar Format Tag:**  
   Implementasi pada `lethon.core.rl` wajib mematuhi pemisahan tag `<think>...</think>` dan `<answer>...</answer>` sesuai konvensi DeepSeek-R1.
2. **Integrasi Operator Length Clipping:**  
   Modul evaluasi reward wajib menerapkan clipping token keras sebelum fungsi ekstraksi jawaban dipanggil:
   ```python
   def compute_thinkprune_reward(output_tokens: list, max_limit: int, verifier) -> float:
       if len(output_tokens) > max_limit:
           return 0.0
       extracted = verifier.extract_answer(output_tokens)
       return 1.0 if (extracted and verifier.is_correct(extracted)) else 0.0
   ```
3. **Kriteria Seleksi Checkpoint Validasi:**  
   Pada kurikulum iteratif, kriteria kelulusan checkpoint antar-tahap didasarkan pada subset validasi AIME22/AIME23:
   $$\text{Akurasi}(\theta_{\text{selected}}) \ge 0.90 \times \text{Akurasi}(\theta_{\text{baseline}}), \quad \min \mathbb{E}[|Y|]$$
4. **Handling Fallback MCP Server:**  
   Jika model dipanggil via MCP Server Lethon (`prune_context`), sistem menerapkan parsing cerdas yang mendeteksi tag `<think>` yang belum tertutup dan secara aman menyisipkan penutup tanpa menyebabkan error format JSON-RPC.
