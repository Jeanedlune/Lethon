# DEC-04: Sequential Decision MDP & CoT Length Truncation (Stage 4)

## 1. Executive Summary & Context

Modul **Sequential Decision MDP & CoT Length Truncation** merupakan pertahanan pemadatan tahap akhir (*Stage 4*) pada pipeline kompresi konteks Lethon ([PRD.md](../../PRD.md)). Modul ini dirancang khusus untuk memecahkan dilema komputasi terbesar pada era model penalaran panjang (*Long-Reasoning LLMs* seperti DeepSeek-R1, OpenAI o1/o3, DeepScaleR, dan QwQ-32B): **sindrom overthinking dan thought redundancy**.

Model reasoning yang dilatih menggunakan *Reinforcement Learning with Verifiable Rewards* (RLVR) cenderung menghasilkan alur penalaran (*Chain-of-Thought* / CoT) yang masif—sering kali melampaui 10.000 hingga 15.000 token per solusi pada benchmark seperti MATH-500 dan AIME24, meskipun persoalan tersebut secara esensial dapat dipecahkan dalam kurang dari 1.000 hingga 3.000 token. Sebagian besar token tambahan tersebut terbuang untuk:
1. Introspeksi dan keraguan verbal berulang (*hesitation & self-correction* seperti *"Wait, let me rethink...", "Hold on, but wait..."*).
2. Verifikasi lokal dan global yang redundan (*repetitive self-verification*).
3. Parafrase ulang soal secara bertele-tele (*problem paraphrasing*).

Berdasarkan temuan riset fundamental pada **ThinkPrune** (*arXiv:2504.01296*), pendekatan pemotongan paksa saat inferensi (*test-time budget forcing* seperti S1) mengalami degradasi akurasi katastropik karena memutus alur penalaran sebelum LLM sempat menyelesaikan komputasi inti atau merumuskan kesimpulan. 

Lethon Stage 4 mengintegrasikan paradigma **Continuous Reinforcement Learning with Length Clipping & Iterative Length Pruning**:
- Menghukum penalaran yang melampaui kuota token $L$ dengan reward nol ($R = 0$) melalui *strict output clipping*, memaksa model mengoptimalkan alur logika internalnya secara kompak.
- Menerapkan **Iterative Pruning Schedule** ($L_1 > L_2 > \dots > L^*$) dengan validasi retensi akurasi (toleransi $\le 10\%$ pada subset validasi kompetisi matematika seperti AIME22/AIME23) untuk mencegah *policy collapse*.
- Mempertahankan ketelitian pemikiran (*deep thinking*) pada masalah rumit, memangkas panjang token CoT hingga **50%–65.5%** pada AIME24 dan MATH-500, dengan degradasi akurasi downstream hanya **~2%** (bahkan meningkatkan rata-rata akurasi pada model yang belum tersaturasi), serta menjaga nilai *perplexity* (readability) tetap stabil.

---

## 2. Landasan Teori & Analisis Paper Acuan (arXiv:2504.01296 & arXiv:2603.13394)

### 2.1 Patologi "Overthinking" pada Long-Reasoning LLMs

Inference-time scaling melalui RLVR memicu kemunculan kemampuan refleksi diri (*self-reflection*) dan verifikasi eksploratif. Namun, ketiadaan batasan panjang keluaran pada fungsi reward orisinal DeepSeek-R1 menciptakan insentif yang salah: model mengeksplorasi cabang logika sembarang tanpa urgensi temporal.

Analisis empiris ThinkPrune (Bairu Hou et al., UCSB & MIT-IBM, April 2025) mengungkap fakta struktural:
- Pada model **DeepSeek-R1-Distill-Qwen-1.5B**, panjang respons rata-rata pada AIME24 mencapai **15.484 token**, dengan akurasi baseline **29.4%**.
- Pada model tersaturasi **DeepScaleR-1.5B-Preview**, panjang CoT mencapai **9.463 token** (akurasi 40.3%).
- Pada model skala besar **QwQ-32B**, panjang respons mencapai **13.822 token** (akurasi 78.8%).

### 2.2 Kegagalan Budget-Forcing Heuristik (S1 Paradox)

Metode heuristik budget-forcing (*test-time truncation*, Muennighoff et al., 2025) memaksa model keluar dini (*early exit*) begitu batas token tercapai dengan menyisipkan delimiter token penutup `</think>` dan memaksakan ekstraksi jawaban langsung `**Final Answer:**`.

Hasil eksperimen pada Gambar 1(a) dan Gambar 3 paper ThinkPrune membuktikan kelemahan fatal budget-forcing:
1. **Truncation Blindness:** Jika model dialokasikan budget 2.000 token, model menghabiskan 453 token hanya untuk parafrase problem. Ketika mencapai 2.000 token, model berada di tengah-tengah percobaan kalkulasi (*attempt 2*).
2. Pemotongan paksa mengorbankan rantai logika yang belum tuntas, memicu jawaban salah (\boxed{40} alih-alih \boxed{36}).
3. Kurva akurasi budget-forcing (garis biru pada Gambar 1(b)) anjlok drastis (*steep performance cliff*) seiring pengetatan budget token.

Sebaliknya, **ThinkPrune** (garis kuning) melatih model melalui RL untuk memadatkan pemikiran sejak langkah pertama: pemahaman masalah diringkas menjadi 166 token, verifikasi diringkas menjadi 1 putaran terarah (324 token), dan model berhasil merumuskan jawaban yang tepat sebelum batas 2.000 token terlampaui.

---

## 3. Formulasi Matematis Eksak

Proses kompresi CoT dan truncasi sekuensial dimodelkan ke dalam kerangka **Markov Decision Process (MDP)** terdisiplin yang terintegrasi dengan algoritma **Group Relative Policy Optimization (GRPO)**.

### 3.1 Formulasi State, Action, dan Transisi MDP

Diberikan query tugas $q \in \mathcal{Q}$ dan trajektori generasi penalaran $Y = (y_1, y_2, \dots, y_T)$, di mana setiap token $y_t \in \mathcal{V}$. Trajektori terbagi menjadi dua segmen semantik:
$$Y = [Y_{\text{think}}, Y_{\text{answer}}]$$
di mana $Y_{\text{think}} = \langle \text{<think>} \dots \text{</think>} \rangle$ dan $Y_{\text{answer}} = \langle \text{<answer>} \dots \text{</answer>} \rangle$.

1. **State Space ($s_t \in \mathcal{S}$):**
   State pada langkah autoregresif $t$ adalah tuple konteks query dan prefix token yang telah dihasilkan:
   $$s_t = (q, y_{<t}) = (q, y_1, y_2, \dots, y_{t-1})$$
   Pada arsitektur kompresi laten TPRL (*arXiv:2603.13394*), $s_t$ dapat dikompresi ke dalam representasi laten $\mathbf{z}_t = \mathcal{E}_\phi(s_t) \in \mathbb{R}^{d_l}$ menggunakan encoder ringkas ($d_l \ll d_v$).

2. **Action Space ($a_t \in \mathcal{A}$):**
   Terdapat dua formulasi aksi yang bekerja selaras:
   - **Formulasi Generatif (Token-Level Policy $\pi_\theta(y_t \mid s_t)$):** Model memilih token berikutnya dari kosa kata $\mathcal{V}$ yang telah dibatasi oleh Promising Subspace Masking Stage 3 ($M_t \odot \pi_\theta$).
   - **Formulasi Sequential Meta-Action ($\alpha_t \in \{0, 1\}$):** Keputusan biner pada tingkat langkah/paragraf penalaran:
     $$\alpha_t = \begin{cases} 1, & \text{pertahankan / lanjutkan elaborasi langkah penalaran } t \\ 0, & \text{pangkas / terminasi cabang penalaran } t \text{ dan lompat ke kesimpulan} \end{cases}$$

3. **Panjang Efektif Trajektori ($|Y|$):**
   Total token yang dihasilkan hingga token penutup $\text{EOS}$ atau batas maksimum:
   $$T = |Y| = |Y_{\text{think}}| + |Y_{\text{answer}}|$$

### 3.2 Operator Length Clipping

Diberikan batas panjang token $L \in \mathbb{N}^+$. Operator clipping $\text{clip}(Y; L)$ didefinisikan sebagai:
$$\text{clip}(Y; L) = (y_1, y_2, \dots, y_{\min(T, L)})$$

Jika $T > L$, seluruh token $y_{L+1}, \dots, y_T$ dibuang secara permanen sebelum evaluasi reward dilakukan.

### 3.3 Formulasi Fungsi Reward

#### 3.3.1 Reward Biner ThinkPrune (Zero-Reward Length Clipping)
Sesuai Persamaan 1 pada paper ThinkPrune (*arXiv:2504.01296*), reward biner diverifikasi melalui fungsi ekstraksi jawaban $\text{ExtractAnswer}(\cdot)$ dan verifier deterministik $\text{IsCorrect}(\cdot, q)$:

$$R_{\text{clip}}(Y, q; L) = \begin{cases} 
1, & \text{jika } \text{ExtractAnswer}(\text{clip}(Y; L)) \neq \emptyset \;\land\; \text{IsCorrect}(\text{ExtractAnswer}(\text{clip}(Y; L)), q^*) = \text{True} \\
0, & \text{lainnya (jawaban salah, format rusak, atau terpotong sebelum tag } \text{</answer>})
\end{cases}$$

**Implikasi Kritis:** Jika model bernalar dengan sangat benar tetapi baru mencapai tag `</answer>` pada token $L+1$, operasi $\text{clip}(Y; L)$ membuang jawaban tersebut. Hasil ekstraksi menjadi kosong ($\emptyset$), sehingga model memperoleh reward **0**. Hal ini memberikan sinyal penalti gradien yang sangat tajam tanpa memerlukan rekayasa hyperparameter reward yang rapuh.

#### 3.3.2 Reward Efisiensi Komposit Lethon
Untuk mempercepat konvergensi pemadatan pada mode hybrid, Lethon menambahkan reward efisiensi teratur $\mathcal{R}_{\text{eff}}$:

$$R_{\text{composite}}(Y, q; L) = R_{\text{clip}}(Y, q; L) \cdot \left[ 1 + \lambda_{\text{eff}} \cdot R_{\text{eff}}(Y; L) \right] - \beta_{\text{pen}} \cdot \mathbb{I}(|Y| > L)$$

di mana reward efisiensi token $R_{\text{eff}}(Y; L)$ diformulasikan sebagai rasio penghematan budget:
$$R_{\text{eff}}(Y; L) = \max\left(0, \frac{L - |Y|}{L}\right) = 1 - \frac{|Y|}{L}$$

dengan parameter:
- $\lambda_{\text{eff}} \in [0.0, 0.25]$: Bobot bonus penghematan token (hanya aktif jika jawaban benar).
- $\beta_{\text{pen}} \in [0.0, 1.0]$: Penalti langsung terhadap pelanggaran batas token ($|Y| > L$). Pada formulasi standar ThinkPrune, $\beta_{\text{pen}} = 0$ karena reward nol dari kegagalan ekstraksi sudah cukup kuat.

### 3.4 Optimasi Kebijakan via Group Relative Policy Optimization (GRPO)

Untuk menghindari kebutuhan model *Critic* (*Value Network*) yang memakan alokasi VRAM besar, Lethon menggunakan algoritma GRPO (Shao et al., 2024; DeepSeek-AI, 2025).

Untuk setiap pertanyaan training $q$, model melakukan rollout sebanyak $G$ kandidat trajektori penalaran $\{Y_1, Y_2, \dots, Y_G\} \sim \pi_{\theta_{\text{old}}}(q)$ di bawah limit $L$. Reward dihitung untuk masing-masing sampel: $r_i = R(Y_i, q; L)$.

1. **Normalisasi Skor Keuntungan Relatif Kelompok (Group Relative Advantage):**
   $$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

2. **Fungsi Objektif GRPO dengan Length Constraint:**
   $$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{Y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|Y_i|} \sum_{t=1}^{|Y_i|} \min\left( \frac{\pi_\theta(y_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid s_{i,t})} \hat{A}_i, \; \text{clip}\left(\frac{\pi_\theta(y_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid s_{i,t})}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

---

## 4. Analisis Empiris Benchmark AIME24 & MATH-500

Berikut adalah rekapitulasi data empiris komprehensif dari evaluasi ThinkPrune (*arXiv:2504.01296*, Tabel 1 & Gambar 2-3):

### 4.1 Tabel Kinerja Lengkap: One-Shot vs Iterative Pruning

| Model Backbone | Strategi Pemangkasan | Akurasi MATH-500 (%) | Akurasi AIME24 (%) | Akurasi AMC23 (%) | Akurasi Olympiad (%) | Rata-rata Akurasi (%) | Panjang Token MATH-500 | Panjang Token AIME24 | Rata-rata Panjang Token | Reduksi Token (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-R1-Distill-Qwen-1.5B** | *Original Model* | 82.9 | **29.4** | 70.3 | 44.7 | 56.8 | 5.560 | 15.484 | 10.355 | Baseline |
| | One-Shot 4k | 83.8 | 29.0 | 73.6 | 46.5 | 58.2 | 2.709 | 8.301 | 5.232 | -49.5% |
| | One-Shot 3k | 83.7 | 27.8 | 71.8 | 44.9 | 57.1 | 2.557 | 7.968 | 4.940 | -52.3% |
| | One-Shot 2k | 82.9 | 27.0 | 72.2 | 45.6 | 56.9 | 2.356 | 7.574 | 4.650 | -55.1% |
| | Iterative 4k $\to$ 3k | 83.9 | 26.9 | 71.4 | 46.0 | 57.1 | 2.209 | 6.389 | 4.062 | -60.8% |
| | **Iterative 4k $\to$ 3k $\to$ 2k** | **83.2** | **27.1** | **73.2** | **46.2** | **57.4** | **1.938** | **5.631** | **3.574** | **-65.5%** |
| **DeepScaleR-1.5B-Preview** | *Original Model* | 88.5 | **40.3** | 81.2 | 52.7 | 65.7 | 3.084 | 9.463 | 5.914 | Baseline |
| | One-Shot 4k | 87.1 | 37.2 | 80.2 | 51.4 | 64.0 | 2.212 | 6.366 | 4.037 | -31.7% |
| | One-Shot 2k | 86.1 | 33.3 | 77.7 | 49.7 | 61.7 | 1.880 | 5.528 | 3.429 | -42.0% |
| | **Iterative 4k $\to$ 3k $\to$ 2k** | **86.9** | **36.5** | **79.4** | **50.1** | **63.2** | **1.881** | **5.301** | **3.370** | **-43.0%** |
| **QwQ-32B** | *Original Model* | 95.1 | **78.8** | 97.5 | 71.1 | 85.6 | 4.289 | 13.822 | 8.763 | Baseline |
| | One-Shot 4k | 94.0 | 76.3 | 95.8 | 68.2 | 83.5 | 2.552 | 8.787 | 5.300 | -39.5% |
| | **Iterative 4k $\to$ 3k $\to$ 2k** | **93.8** | **72.5** | **95.9** | **67.3** | **82.4** | **2.162** | **7.631** | **4.494** | **-48.7%** |

### 4.2 Analisis Wawasan Kunci Benchmark

1. **Efek Model Unsaturated vs Saturated:**  
   Pada model yang belum jenuh (*unsaturated*) seperti DeepSeek-R1-Distill-Qwen-1.5B (hanya melewati SFT dari jejak R1 asli tanpa RL lanjutan), pelatihan ThinkPrune justru **meningkatkan rata-rata akurasi** dari $56.8\%$ menjadi **$57.4\%$**, sambil memotong panjang token sebesar **$65.5\%$** (dari 10.355 menjadi 3.574 token). Hal ini membuktikan bahwa overthinking pada model distilasi awal bersifat kontraproduktif dan menimbulkan bias kebingungan (*hallucinatory confusion*).
2. **Retensi Penalaran Ekstrem AIME24:**  
   Pada benchmark AIME24 yang terkenal sangat sulit, panjang penalaran model 1.5B terpotong dari **15.484 token menjadi 5.631 token** (**reduksi 63.6%**). Akurasi hanya mengalami penurunan minimal dari 29.4% ke 27.1% (penurunan absolut 2.3%, atau relatif ~7.8%, jauh di bawah batas toleransi kritis 10%).
3. **Keunggulan Mutlak Iterative Pruning atas One-Shot:**  
   Pada DeepScaleR-1.5B, strategi bertahap $4k \to 3k \to 2k$ menghasilkan akurasi **$86.9\%$** pada MATH-500 dibandingkan $86.1\%$ pada One-Shot 2k, dengan jumlah token yang lebih hemat (1.881 vs 1.880 token). Transisi bertahap memberikan kurikulum adaptasi bagi model untuk belajar mengonsolidasikan rute penalaran.
4. **Generalisasi Skala Uji (*Test-Time Generalization*):**  
   Meskipun model dilatih dengan limit ketat $L \in \{2000, 3000, 4000\}$, ketika diuji pada soal yang sangat sulit di AIME24 dengan batas konteks terbuka (32.768 token), model **secara adaptif memperpanjang pemikirannya hingga 5.631 – 7.631 token**. Ini membuktikan bahwa ThinkPrune tidak merusak kemampuan penalaran mendalam (*deep thinking*), melainkan menghilangkan kebiasaan bertele-tele pada langkah-langkah trivial.

---

## 5. Pola Identifikasi Redundansi Penalaran (*Thought Redundancy Taxonomy*)

Berdasarkan analisis frekuensi kata kunci (*Figure 4 paper*) dan segmentasi 10 fase penalaran oleh GPT-4o (*Figure 5 & Appendix A.2*), Lethon mengklasifikasikan dinamika pemikiran CoT menjadi 3 kelas perilaku dan 10 fase problem-solving:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │          Taksonomi Redundansi CoT ThinkPrune            │
                  └────────────────────────────┬────────────────────────────┘
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         ▼                                     ▼                                     ▼
 ┌───────────────────────┐           ┌───────────────────┐               ┌───────────────────────┐
 │ Kelas 1: Hesitation   │           │ Kelas 2: Verify   │               │ Kelas 3: Core Logic   │
 │ & Self-Correction     │           │ & Reassessment    │               │ & Computation         │
 ├───────────────────────┤           ├───────────────────┤               ├───────────────────────┤
 │ • "Wait", "But wait"  │           │ • "Double-check"  │               │ • "Let me compute"    │
 │ • "Hold on"           │           │ • "Let me confirm"│               │ • "Therefore"         │
 │ • "Alternatively"     │           │ • "Remember"      │               │ • "Applying theorem"  │
 ├───────────────────────┤           ├───────────────────┤               ├───────────────────────┤
 │ Anjlok drastis (-70%) │           │ Turun moderat     │               │ Frekuensi per 1K      │
 │ Indikasi keraguan     │           │ Menghilangkan loop│               │ token MENINGKAT       │
 │ tanpa dasar dibuang   │           │ verifikasi berulang│              │ (+15% s.d. +25%)      │
 └───────────────────────┘           └───────────────────┘               └───────────────────────┘
```

### 5.1 Perubahan Frekuensi Keyword Penalaran (Per 1.000 Token)

1. **Frase Keraguan & Koreksi Diri (*Hesitation / Self-Correction*):**
   - *"Wait"* & *"But wait"*: Turun lebih dari **$60\% - 75\%$**.
   - *"Hold on"*: Turun secara signifikan.
   - *"Alternatively"*: Berkurang tajam karena model tidak lagi mengeksplorasi jalan buntu tanpa alasan yang jelas.
2. **Frase Verifikasi (*Self-Verification*):**
   - *"Double-check"*, *"Let me confirm"*, *"Remember"*: Mengalami penurunan moderat. Model mempertahankan tepat **satu putaran verifikasi terarah** di akhir kalkulasi sebelum merilis jawaban, alih-alih melakukan 4–5 putaran verifikasi berulang.
3. **Frase Komputasi Inti (*Core Computation & Deduction*):**
   - *"Let me compute"*, *"Therefore"*, *"Substitute"*: **Frekuensi relatif per 1.000 token justru meningkat**. Porsi kapasitas token dialokasikan langsung untuk eksekusi aljabar, penyederhanaan ekspresi, dan aplikasi teorema.

### 5.2 Distribusi 10 Fase Penalaran (*Reasoning Phases*)

Berdasarkan anotasi semantik segmentasi chunk:

| Indeks | Fase Pemecahan Masalah (*Problem-Solving Phase*) | Karakteristik Perilaku | Proporsi Sebelum Pruning | Proporsi Setelah ThinkPrune | Status Transformasi |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | **Understanding the Problem** | Membaca data, identifikasi variabel, dan parafrase target | 4.7% | 3.7% | Dipadatkan / diringkas |
| 2 | **Reformulating the Problem** | Mengubah variabel, menulis ulang ekspresi, substitusi indeks | 6.9% | 2.7% | Dieliminasi jika trivial |
| 3 | **Applying Known Theorems/Properties** | Penggunaan rumus baku, identitas geometri, teorema aljabar | 10.1% | **15.3%** | **Meningkat Signifikan** |
| 4 | **Breaking Down into Subproblems** | Dekomposisi masalah menjadi kasus-kasus terpisah | 4.4% | 3.7% | Stabil / efisien |
| 5 | **Computing or Simplifying Expressions** | Manipulasi aljabar, evaluasi aritmetika, faktorisasi | 22.2% | **31.6%** | **Dominan Utama (+9.4%)** |
| 6 | **Substituting Known Values or Results** | Substitusi konstanta dan hasil antara ke persamaan | 1.7% | 6.3% | Meningkat |
| 7 | **Reassess and Verify Local Steps** | Pemeriksaan ketelitian aritmetika pada langkah tertentu | 10.9% | 8.7% | Terkonsolidasi |
| 8 | **Reassess the Whole Solution** | Tinjauan menyeluruh terhadap konsistensi logika solusi | 15.2% | 11.1% | Dibatasi 1 kali iterasi |
| 9 | **Exploring Alternative Approaches** | Mencoba cara kedua/ketiga yang tidak diperlukan | 23.9% | **13.3%** | **Terpangkas Masif (-10.6%)** |
| 10 | **Finalize and Present the Answer** | Merangkum jawaban akhir dan sintaksis output | 7.4% | 4.0% | Langsung to-the-point |

### 5.3 Preservasi Readability & Perplexity

Kekhawatiran umum dalam pelatihan pemadatan RL (seperti pada model R1-Zero) adalah runtuhnya keterbacaan bahasa (*language degradation*), di mana model menghasilkan bahasa campuran (*code-switching*) atau simbol-simbol terkompresi yang tidak dapat dibaca manusia (*cryptic tokens*).

Pengukuran *perplexity* (PPL) menggunakan evaluator bahasa matematis independen (`Qwen2.5-Math-7B`) pada dataset MATH-500 menunjukkan:
- **DeepSeek-R1-1.5B-Distill:** Baseline PPL = **1.91** $\to$ ThinkPrune PPL = **1.90** (stabil).
- **DeepScaleR-1.5B-Preview:** Baseline PPL = **1.95** $\to$ ThinkPrune PPL = **2.02** ($\Delta < +0.07$).
- **QwQ-32B:** Baseline PPL = **2.37** $\to$ ThinkPrune PPL = **2.24** (membaik).

Hal ini membuktikan bahwa penalti panjang berbasis clipping mempertahankan distribusi bahasa alami dengan kejelasan logika yang utuh.

---

## 6. Arsitektur Komponen & Spesifikasi Algoritmik

```
                          ┌────────────────────────────────────────────────────────┐
                          │     Lethon Stage 4 CoT Truncation Architecture         │
                          └───────────────────────────┬────────────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
 ┌───────────────────────────────┐    ┌───────────────────────────────┐    ┌───────────────────────────────┐
 │   Iterative Length Schedule   │    │  Thought Redundancy Analyzer  │    │  GRPO Clipping Reward Engine  │
 │ (4k -> 3k -> 2k Multi-Round)  │    │ (Keywords & 10-Phase Tracker) │    │ (Verification & Eff Penalty)  │
 ├───────────────────────────────┤    ├───────────────────────────────┤    ├───────────────────────────────┤
 │ • Schedule Manager            │    │ • Regex Phrase Matcher        │    │ • Deterministic Answer Match  │
 │ • Checkpoint Gatekeeper       │    │ • Phase Segmenter             │    │ • Zero-Reward on Overflow     │
 │ • AIME Validation Tolerance   │    │ • Entropy Drift Monitor       │    │ • Group Advantage Normalizer  │
 └───────────────────────────────┘    └───────────────────────────────┘    └───────────────────────────────┘
```

Modul Stage 4 terdiri atas tiga sub-sistem utama:
1. **`IterativePruningScheduler`:** Mengelola kurikulum transisi limit token $L_1 > L_2 > \dots > L^*$, melakukan evaluasi periodik tiap $N$ step pada validation set (AIME22/AIME23), dan menerapkan kriteria seleksi checkpoint terbaik (akurasi dalam toleransi relatif $\le 10\%$ dengan panjang output terpendek).
2. **`ThoughtRedundancyAnalyzer`:** Menganalisis teks CoT mentah atau sekuens token yang dihasilkan, mendeteksi densitas kata kunci keraguan vs komputasi, mengidentifikasi fase penalaran, dan mengukur rasio kompresi.
3. **`RewardClippingEvaluator`:** Mengeksekusi operator $\text{clip}(Y; L)$, mengekstrak tag `<answer>`, memvalidasi kebenaran matematis, dan mengembalikan skalar reward biner atau komposit.

---

## 7. Pseudo-Code & Spesifikasi Implementasi

### 7.1 Algoritma Iterative Length Pruning via GRPO

```python
"""
Algoritma: Iterative CoT Pruning dengan GRPO dan Strict Length Clipping
Sesuai: arXiv:2504.01296 (ThinkPrune)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class IterativeThinkPruneTrainer:
    def __init__(
        self,
        base_model,
        tokenizer,
        verifier,
        length_schedule: List[int] = [4000, 3000, 2000],
        max_relative_acc_drop: float = 0.10,
        group_size: int = 16,
        lambda_eff: float = 0.1,
    ):
        self.model = base_model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.length_schedule = length_schedule
        self.max_relative_acc_drop = max_relative_acc_drop
        self.group_size = group_size
        self.lambda_eff = lambda_eff

    def evaluate_validation(self, model, val_dataset, current_limit: int) -> Tuple[float, float]:
        """
        Mengevaluasi pass@1 accuracy dan rata-rata token output pada validation set (misal AIME22+23).
        """
        correct_count = 0
        total_tokens = 0
        total_samples = len(val_dataset)

        for example in val_dataset:
            prompt = self.format_training_prompt(example["question"], current_limit)
            output_tokens, raw_text = model.generate(prompt, max_length=current_limit * 2)
            
            # Evaluasi di bawah batasan clipping
            clipped_text = self.clip_text_to_tokens(raw_text, current_limit)
            extracted_ans = self.verifier.extract_answer(clipped_text)
            
            if extracted_ans is not None and self.verifier.check_correct(extracted_ans, example["ground_truth"]):
                correct_count += 1
            total_tokens += len(output_tokens)

        acc = correct_count / total_samples
        avg_len = total_tokens / total_samples
        return acc, avg_len

    def compute_grpo_rewards(
        self, 
        rollouts: List[str], 
        ground_truth: str, 
        length_limit: int
    ) -> np.ndarray:
        """
        Menghitung reward biner dengan length clipping dan komponen efisiensi.
        """
        rewards = np.zeros(len(rollouts), dtype=np.float32)
        
        for idx, text in enumerate(rollouts):
            token_count = len(self.tokenizer.encode(text))
            
            # Operator Clipping
            if token_count > length_limit:
                # Output terpotong sebelum selesai -> jawaban hilang
                rewards[idx] = 0.0
                continue
                
            extracted_ans = self.verifier.extract_answer(text)
            is_correct = (extracted_ans is not None) and self.verifier.check_correct(extracted_ans, ground_truth)
            
            if is_correct:
                # Reward dasar benar = 1.0 + bonus efisiensi
                eff_ratio = max(0.0, (length_limit - token_count) / length_limit)
                rewards[idx] = 1.0 + self.lambda_eff * eff_ratio
            else:
                rewards[idx] = 0.0

        return rewards

    def run_pruning_curriculum(self, train_dataset, val_dataset):
        current_model = self.model
        baseline_acc, _ = self.evaluate_validation(current_model, val_dataset, current_limit=self.length_schedule[0])
        min_allowed_acc = baseline_acc * (1.0 - self.max_relative_acc_drop)

        for stage_idx, current_limit in enumerate(self.length_schedule):
            print(f"=== Memulai Iterasi Pruning {stage_idx+1}: Limit = {current_limit} token ===")
            best_checkpoint = None
            best_len = float("inf")

            for epoch in range(num_epochs_per_stage):
                for batch in train_dataset:
                    # 1. Rollout group of responses
                    prompts = [self.format_training_prompt(q, current_limit) for q in batch["question"]]
                    rollouts_group = current_model.sample_groups(prompts, G=self.group_size)

                    # 2. Hitung reward clipped
                    rewards = [
                        self.compute_grpo_rewards(group, gt, current_limit)
                        for group, gt in zip(rollouts_group, batch["ground_truth"])
                    ]

                    # 3. Hitung group advantage & update policy
                    advantages = self.compute_group_advantages(rewards)
                    self.update_policy_grpo(current_model, rollouts_group, advantages)

                # Evaluasi periodik untuk checkpoint selection
                val_acc, val_len = self.evaluate_validation(current_model, val_dataset, current_limit)
                if val_acc >= min_allowed_acc and val_len < best_len:
                    best_len = val_len
                    best_checkpoint = current_model.save_checkpoint()

            if best_checkpoint is not None:
                current_model.load_checkpoint(best_checkpoint)
                print(f"Iterasi {stage_idx+1} selesai. Selected Checkpoint: Avg Len = {best_len:.1f}")

        return current_model
```

---

## 8. Integrasi dengan Pipeline Lethon & Batasan Runtime

### 8.1 Sinergi Stage 1 hingga Stage 4

1. **Stage 1 (Spectral FFT):** Menyingkirkan token noise frekuensi tinggi dari konteks input awal secara parameter-free dalam $<3$ ms.
2. **Stage 2 (Semantic Anchors - SAC):** Memadatkan konteks dokumen/repositori ke dalam anchor tokens via bidirectional attention ($5\times - 51\times$).
3. **Stage 3 (Promising Token Masking - RLPT):** Membatasi ruang aksi generasi pada token-token kritis ($K = 4 \sim 8$), menekan varians gradien hingga $\Delta V > 0$.
4. **Stage 4 (CoT MDP Truncation):** Menerapkan kebijakan pemikiran ringkas yang telah dilatih ThinkPrune. Ketika model memproses query, pola CoT tidak lagi mengulang-ulang keraguan atau verifikasi sia-sia, menghasilkan penyelesaian masalah dalam batas token 50%–65% lebih singkat tanpa kehilangan akurasi konklusif.

### 8.2 Runtime Profiling & Target Spesifikasi

- **Inference Latency Overhead:** Zero runtime overhead. Kebijakan pemadatan diinternalisasi ke dalam bobot model selama fase pelatihan RL (tidak memerlukan ekstraksi jaringan critic saat inferensi).
- **Format Tag Kompatibilitas:** Standar tag DeepSeek-R1 `<think> ... </think>` dan `<answer> ... </answer>`.
- **Perplexity Boundary:** Degradasi PPL bahasa pada benchmark MATH-500 dibatasi $\Delta \text{PPL} \le 0.15$ untuk mencegah degradasi bahasa.
- **Fail-Safe Fallback:** Jika respons terpotong sebelum tag `</think>` tertutup saat batas keras runtime tercapai, Lethon MCP menyisipkan penutup terstruktur dan mengembalikan rute logika parsial ke agen tanpa memicu exception fatal.
