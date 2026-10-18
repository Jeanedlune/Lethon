# DEC-02: Contextual Semantic Anchor Aggregation (Stage 2)

## 1. Overview
Modul **Contextual Semantic Anchor Aggregation (SAC)** bertugas mengonsolidasikan informasi global dari konteks teks panjang ke dalam sejumlah representasi kompak (*anchor tokens*) menggunakan mekanisme modifikasi atensi dua arah (*bidirectional cross-attention*).

## 2. Landasan Matematis

Diberikan token konteks setelah Stage 1:
$$H \in \mathbb{R}^{L \times d}$$
dan kumpulan $M$ *anchor tokens* yang dapat dipelajari (*learnable anchor embeddings*):
$$A^{(0)} \in \mathbb{R}^{M \times d}, \quad \text{dengan } M \ll L$$

### 2.1 Bidirectional Attention Mechanism
Berbeda dengan causal masking pada autoregressive LLM, proses penyerapan konteks oleh anchor dilakukan secara dua arah:
$$Q = A^{(0)} W_Q, \quad K = H W_K, \quad V = H W_V$$
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

Anchor token $A$ menyerap sari representasi informasi dari seluruh konteks $H$ tanpa batasan causal temporal, memungkinkan penyimpanan intisari dokumen dalam ukuran tensor yang jauh lebih kecil.

### 2.2 Re-Injection ke LLM Pipeline
Anchor tokens yang telah diperkaya $\tilde{A} \in \mathbb{R}^{M \times d}$ kemudian disisipkan di awal konteks (prefix) atau menggantikan segmen konteks panjang yang redundan saat inferensi LLM dilakukan.
