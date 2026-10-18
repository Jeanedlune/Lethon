# DEC-03: Dynamic Promising Token Subspace Masking (Stage 3)

## 1. Overview
Modul **Dynamic Promising Token Subspace Masking (RLPT)** bertugas melindungi token-token bernilai informasi kritis (seperti simbol sintaksis, operator logika, identifier variabel) dari risiko pemangkasan tak sengaja saat agen RL membuat keputusan seleksi token.

## 2. Landasan Matematis

Berdasarkan analisis teoritis pada paper RLPT (arXiv:2602.03195), trajektori penalaran yang valid selalu berada di dalam *low-rank subspace* dari representasi semantik model.

### 2.1 Subspace Decomposition
Untuk hidden states $H \in \mathbb{R}^{L \times d}$, dilakukan dekomposisi nilai singular terpotong (Truncated SVD) atau proyeksi matriks peringkat rendah:
$$H = U \Sigma V^T$$

Basis proyeksi $P_k = V_k V_k^T$ mengidentifikasi subruang $k$-dimensi dari representasi paling dominan.

### 2.2 Promising Token Score
Skor signifikansi setiap token $i$ dihitung berdasarkan magnitudo proyeksinya ke subruang:
$$s_i = \| h_i P_k \|_2^2$$

### 2.3 Dynamic Action Masking Matrix
Diberikan threshold dinamis $\tau$, matriks mask aksi $M_{\text{action}} \in \{0, -\infty\}^L$ didefinisikan sebagai:
$$M_{\text{action}}(i) = \begin{cases}
0 & \text{jika } s_i < \tau \quad (\text{token boleh dipangkas}) \\
-\infty & \text{jika } s_i \ge \tau \quad (\text{promising token, dilarang dipangkas})
\end{cases}$$

Dengan menambahkan $M_{\text{action}}$ pada logit kebijakan RL ($\pi(a|s) = \text{softmax}(z + M_{\text{action}})$), ruang pencarian aksi menjadi stabil dan varians gradien turun drastis.
