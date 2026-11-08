# 02 — Mathematical Specification & Surgical Bugfix Plan (`M-01` .. `M-07`)

> **Tujuan Dokumen**: Memberikan spesifikasi matematika eksak, batas presisi numerik IEEE-754, bentuk tensor (*tensor shapes*), dan lokasi bedah kode untuk **7 bug matematika yang telah dibuktikan secara forensik** pada `src/kilat/`.  
> **Standar Bukti**: Setiap bedah kode di dokumen ini tidak dianggap selesai sebelum lulus `torch.autograd.gradcheck` dalam `torch.float64` (toleransi $< 10^{-6}$) dan uji ekuivalensi referensi.

---

## Ringkasan Eksekutif 7 Bedah Matematika

| ID | Komponen & Lokasi File Kanonik | Akar Masalah Matematika yang Dijaga | Dampak Diam-Diam bila Dilanggar (*Silent Failure*) |
| :---: | :--- | :--- | :--- |
| **M-01** | `src/kilat/models/kernels/flash_decay.py` & `src/kilat/models/attention.py` | Kernel CUDA `flash_decay_fwd_kernel` wajib dibungkus `FlashDecayFunction` (`autograd.Function`); rumus naif $(1-\lambda^{i+1})/(1-\lambda+10^{-6})$ runtuh saat $\lambda \to 1^-$. | Seluruh cabang *Global Decay* (`log_lambda`, `v_proj_global`) kehilangan gradien di GPU; normalisasi meledak menuju $0$ saat $\lambda \to 1$. |
| **M-02** | `src/kilat/models/blocks.py` (`RMSNorm`) | Varians `RMSNorm.forward` wajib di-*upcast* ke `float32` (dan mempertahankan `float64` untuk `gradcheck`). | *Overflow* menjadi `Inf` $\to$ keluaran `0.0` / `NaN` begitu $|x| > 256$ pada AMP FP16. |
| **M-03** | `src/kilat/engine/optimizer.py` (`create_optimizer`) | Seluruh parameter `ndim < 2`, `RMSNorm`, `LayerNorm`, `.bias`, dan `log_lambda` wajib masuk grup `weight_decay = 0.0`. | `attn.log_lambda` dan `q_a_norm`/`kv_a_norm` tergerus *weight decay*, merusak horizon memori jangka panjang ($\lambda \to 0.5$). |
| **M-04** | `src/kilat/models/ffn.py` (`DeepSeekMoE`) | `expert_fraction` dibagi $K_{\text{eff}}$; renormalisasi Top-1 dilewati agar gradien *router* hidup; mendukung `num_shared_experts >= 0` & mode `bias_free`; *router* dihitung di FP32. | `aux_loss` membengkak sebesar faktor $K$; *router* Top-1 buta terhadap *LM loss*; konfigurasi tanpa *shared expert* gagal. |
| **M-05** | `src/kilat/models/attention.py` (`LayerHybridCache`) | `seq_len` disimpan eksplisit di `LayerHybridCache`; mendukung *chunked prefill* ($N > 1$); RoPE disimpan pasca-rotasi di cache. | Output generasi meledak $\sim 10\times$ pada `recall_ratio=0.0`; *chunked prefill* kehilangan konteks masa lalu. |
| **M-06** | `src/kilat/models/transformer.py`, `src/kilat/engine/trainer.py`, `src/kilat/engine/checkpoint.py` | Pemisahan `ce_loss` vs `aux_loss` untuk `ppl = exp(ce_loss)` berbobot token; skala akumulasi parsial $m_w$; `compute_total_steps` memakai `ceil`; skala residual $1/\sqrt{2L}$. | Metrik PPL MoE bias tinggi; gradien langkah terakhir *epoch* mengecil semu; LR scheduler melampaui batas langkah; varians aktivasi tumbuh linear terhadap kedalaman $L$. |
| **M-07** | `src/kilat/models/{blocks,ffn,transformer}.py` | Nol pemanggilan sinkronisasi `.all()` / `.any()` / `.item()` di dalam `nn.Module.forward()` (`debug_nan_checks=False`). | Menyumbat *CUDA stream* (sinkronisasi CPU-GPU di setiap sub-layer) dan mematahkan `torch.compile(fullgraph=True)`. |

---

## M-01: Autograd Analitik Eksak $O(N)$ & Stabilitas $\lambda \to 1^-$ pada `triton_global_decay`

### 1. Persamaan Forward Eksak (Bebas Singularitas $1 - \lambda$)
Untuk input nilai $V \in \mathbb{R}^{B \times N \times H_g \times D_h}$ dan parameter peluruhan per-head $\lambda \in (0, 1)^{H_g}$ dengan $\lambda_h = \sigma(\alpha_h)$ (`alpha = self.log_lambda`), definisikan rekurensi kausal untuk indeks posisi $i \in \{0, 1, \dots, N-1\}$:

$$s_{b, 0, h} = v_{b, 0, h}, \qquad s_{b, i, h} = \lambda_h s_{b, i-1, h} + v_{b, i, h}$$

$$z_{0, h} = 1, \qquad z_{i, h} = \lambda_h z_{i-1, h} + 1, \qquad z'_{0, h} = 0, \qquad z'_{i, h} = \lambda_h z'_{i-1, h} + z_{i-1, h}$$

$$y_{b, i, h} = \frac{s_{b, i, h}}{z_{i, h}}$$

**Keunggulan Matematis Rekurensi $(z_i, z'_i)$**:
- Karena $\lambda_h \in [0, 1]$, terjamin $z_{i, h} \ge 1$ untuk seluruh $i \ge 0$. **Tidak ada pembagian dengan $(1 - \lambda_h)$**, sehingga tidak pernah muncul galat pembagian nol atau pembatalan katastrofik (*catastrophic cancellation*) bahkan pada titik ekstrem $\lambda_h = 1.0$ ($z_i = i+1$, $z'_i = \frac{i(i+1)}{2}$).
- Seluruh akumulator rekurensi ($s, z, z'$) di dalam kernel Triton maupun jalur CPU **wajib** disimpan dalam `float32` (atau `float64` saat `V.dtype == torch.float64` untuk `gradcheck`), lalu di-*cast* kembali ke `V.dtype` saat menulis $y$.

### 2. Rumus Tertutup Stabil untuk Loncatan Indeks $t$ (Incremental Decode)
Saat *incremental decoding* berada pada posisi sekuens $t = \text{cache.seq\_len}$ (tanpa memindai ulang $0 \dots t$), jangan pernah menghitung `(1 - lam**(t+1)) / (1 - lam + 1e-6)`. Gunakan parameterisasi log-ruang $u_h = \ln \sigma(\alpha_h) = -\operatorname{softplus}(-\alpha_h) = \operatorname{logsigmoid}(\alpha_h) \le 0$ dalam `float32`/`float64` dan fungsi analitik mulus $\varphi_1(w)$:

$$\varphi_1(w) = \begin{cases} \dfrac{\operatorname{expm1}(w)}{w}, & |w| \ge 10^{-4} \\[8pt] 1 + \dfrac{w}{2} + \dfrac{w^2}{6} + \dfrac{w^3}{24}, & |w| < 10^{-4} \end{cases}$$

Maka untuk sembarang $n = t + 1 \ge 1$ (bahkan $n = 10^6$) dan sembarang $\alpha_h \in (-\infty, +\infty)$:

$$z_{t, h} = n \cdot \frac{\varphi_1(n u_h)}{\varphi_1(u_h)}$$

Rumus ini akurat hingga presisi mesin penuh tanpa batas atas panjang konteks $n$.

### 3. Derivasi Backward Analitik Eksak $O(N)$ (`FlashDecayFunction.backward`)
Misalkan gradien hulu $\frac{\partial \mathcal{L}}{\partial y} = G \in \mathbb{R}^{B \times N \times H_g \times D_h}$. Definisikan gradien ter-skala $\tilde{g}_{b, i, h} = \frac{g_{b, i, h}}{z_{i, h}}$.

1. **Gradien terhadap $V$ (Reverse Scan $i = N-1 \dots 0$)**:
   $$r_{b, N-1, h} = \tilde{g}_{b, N-1, h}, \qquad r_{b, i, h} = \tilde{g}_{b, i, h} + \lambda_h r_{b, i+1, h}$$
   $$\frac{\partial \mathcal{L}}{\partial v_{b, i, h}} = r_{b, i, h}$$

2. **Gradien terhadap $\lambda$ (Tanpa Buffer Tambahan $O(BND)$!)**:
   Memanfaatkan identitas adjoin $\sum_{i=0}^{N-1} \langle \tilde{g}_i, q_i \rangle = \sum_{i=1}^{N-1} \langle r_i, s_{i-1} \rangle$ serta $s_{i-1} = z_{i-1} y_{i-1}$, gradien terhadap $\lambda_h$ dapat dihitung langsung di dalam *reverse scan* yang sama hanya dengan menyimpan output forward $Y$ dan tabel kecil $(z, z') \in \mathbb{R}^{N \times H_g}$:
   $$\frac{\partial \mathcal{L}}{\partial \lambda_h} = \sum_{b=0}^{B-1} \sum_{i=1}^{N-1} \left( z_{i-1, h} \langle r_{b, i, h}, y_{b, i-1, h} \rangle - z'_{i, h} \langle \tilde{g}_{b, i, h}, y_{b, i, h} \rangle \right)$$
   *(Catatan: untuk $i=0$, $z'_{0,h} = 0$ sehingga suku $i=0$ bernilai identik nol).*

### 4. Kontrak Implementasi Kode
- Buat `class FlashDecayFunction(torch.autograd.Function)` di `src/kilat/models/kernels/flash_decay.py`:
  - `forward(ctx, V, lam)` mendukung jalur `@triton.jit` `flash_decay_fwd_kernel` saat `V.is_cuda and HAS_TRITON and V.dtype != torch.float64` dan jalur PyTorch terkompilasi/vektor saat CPU atau `torch.float64`. Simpan `ctx.save_for_backward(y, lam, z, z_prime)`.
  - `backward(ctx, grad_out)` memanggil `@triton.jit` `flash_decay_bwd_kernel` pada CUDA (dengan akumulasi `tl.float32` + `tl.atomic_add` atau reduksi per-program untuk `grad_lam`) dan pemindaian balik eksak untuk CPU/`float64`. Kembalikan `(grad_V.to(V.dtype), grad_lam.to(lam.dtype))`.
- Sediakan fungsi murni `_reference_global_decay_fp64(V, lam)` (matriks segitiga bawah kausal eksak $M_{i,j} = \frac{\lambda^{i-j}}{z_i}$ untuk $j \le i$) khusus untuk pengujian ekuivalensi Gate 2 & Gate 3.

---

## M-02: Upcasting Varians FP32 pada `RMSNorm`

### 1. Lokasi & Masalah
`src/kilat/models/blocks.py` (`RMSNorm.forward`) jika dihitung naif:
```python
norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
x_normed = x * torch.rsqrt(norm_x + self.eps)
return self.weight * x_normed
```
Pada `torch.float16`, nilai maksimum yang dapat direpresentasikan adalah $65504$. Begitu elemen aktivasi residual $|x_k| > 255.9$, maka $x_k^2 > 65504 \implies +\infty$, sehingga `torch.rsqrt(inf) == 0.0` (atau `inf * 0.0 == NaN`).

### 2. Persamaan & Implementasi Target
Untuk input $x \in \mathbb{R}^{\dots \times D}$ dengan dtype sembarang $\tau \in \{\text{fp16}, \text{bf16}, \text{fp32}, \text{fp64}\}$:
$$\tilde{x} = \operatorname{cast}_{\text{fp32}}(x) \quad (\text{kecuali jika } \tau = \text{fp64}, \text{ pertahankan fp64})$$
$$\operatorname{RMSNorm}(x) = \operatorname{cast}_{\tau}\!\left(\tilde{x} \cdot \frac{1}{\sqrt{\frac{1}{D}\sum_{k=1}^{D} \tilde{x}_k^2 + \epsilon}}\right) \odot w$$

Implementasi eksak:
```python
compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
x_up = x.to(compute_dtype)
variance = x_up.pow(2).mean(dim=-1, keepdim=True)
x_normed = (x_up * torch.rsqrt(variance + self.eps)).to(x.dtype)
return x_normed * self.weight
```
*(Perhatikan penjaga `torch.float64`: ini mencegah downcasting FP64 $\to$ FP32 yang akan menggagalkan `torch.autograd.gradcheck`!)*

---

## M-03: Partisi Weight Decay Berbasis Dimensi & Tipe Modul (`create_optimizer`)

### 1. Lokasi & Masalah
Pada `src/kilat/engine/optimizer.py`, pencocokan naif hanya memakai:
`no_decay_keywords = ("bias", "LayerNorm", "rms", "ln_f", "rms1", "rms2")`.
- `attn.log_lambda` memiliki bentuk `(n_global_heads,)` (`ndim == 1`). Dengan $\lambda = \sigma(\text{log\_lambda})$ diinisialisasi pada $\ln 9 \approx 2.197$ ($\lambda = 0.9$), *weight decay* menarik `log_lambda` menuju $0 \implies \lambda \to 0.5$ bahkan ketika gradien tugas menginginkan memori panjang!
- `attn.q_a_norm.weight` dan `attn.kv_a_norm.weight` adalah `nn.LayerNorm`, tetapi nama atributnya tidak mengandung string `"LayerNorm"`, sehingga ikut terkena *weight decay*.

### 2. Aturan Partisi Eksak (Standar Karpathy / TorchTitan / OLMo)
Suatu parameter $\theta$ dengan nama `param_name` yang berada di dalam modul `module` masuk ke grup **`no_decay` (`weight_decay = 0.0`)** jika dan hanya jika memenuhi **minimal satu** syarat berikut:
1. `param.ndim < 2` (mencakup seluruh `bias`, bobot `RMSNorm`/`LayerNorm`, dan parameter skalar/vektor per-head seperti `log_lambda`).
2. `isinstance(module, (nn.LayerNorm, RMSNorm, nn.Embedding))` — catatan: untuk `wte` (`nn.Embedding`), sediakan opsi `decay_embeddings: bool = False` (default `False` sesuai praktik LLM modern, atau ikuti flag eksplisit agar tidak mematahkan tes lama yang mengecek grup parameter — verifikasi terhadap `tests/unit/features/training/test_training_components.py`!).
3. Nama parameter berakhiran `.bias` atau mengandung kata kunci normalisasi (`norm`, `rms`, `ln_`, `log_lambda` secara *case-insensitive*).

> **Catatan Anti-Halusinasi pada Tes Eksisting**: Sebelum mengaktifkan pengecualian `nn.Embedding`, baca `tests/unit/features/training/test_training_components.py` untuk memastikan asersi jumlah parameter di grup `decay` vs `no_decay` kompatibel. `param.ndim < 2` + pencocokan *case-insensitive* `("bias", "layernorm", "norm", "rms", "ln_", "log_lambda")` sudah 100% melindungi `log_lambda`, `q_a_norm`, `kv_a_norm`, `rms1`, `rms2`, `ln_f`, dan seluruh `bias` tanpa mengubah perlakuan matriks 2D!

---

## M-04: Matematika Eksak `DeepSeekMoE` (Load Balance, Top-1 Gradient, Shared Experts $\ge 0$, & Zero-Sync Dispatch)

### 1. Empat Invarian Matematis di `src/kilat/models/ffn.py`

1. **Presisi Router Logits (`float32`)**:
   Untuk input token teratakan $X \in \mathbb{R}^{T \times D}$ ($T = B \cdot S$), hitung logit router dalam FP32 (kecuali saat FP64 untuk `gradcheck`):
   $$Z_{\text{router}} = X_{\text{up}} W_{\text{router}}^\top, \qquad P = \operatorname{softmax}(Z_{\text{router}}, \text{dim}=-1)$$
   di mana $P \in \mathbb{R}^{T \times N_r}$, $N_r = \text{num\_experts} \times \text{fine\_grained\_factor}$ adalah jumlah *routed experts* total, dan $K_{\text{eff}} = \text{active\_experts} \times \text{fine\_grained\_factor}$ adalah jumlah expert aktif per token.

2. **Penjaga Gradien Router saat $K_{\text{eff}} == 1$**:
   Pilih $(W_{\text{top}}, I_{\text{top}}) = \operatorname{TopK}(P, K_{\text{eff}})$.
   $$\tilde{W}_{\text{top}} = \begin{cases} \dfrac{W_{\text{top}}}{\sum_{k=1}^{K_{\text{eff}}} W_{\text{top}, k} + \epsilon}, & \text{jika } K_{\text{eff}} > 1 \text{ dan } \text{renormalize\_topk}=\text{True} \\[8pt] W_{\text{top}}, & \text{jika } K_{\text{eff}} == 1 \end{cases}$$
   Dengan demikian, saat $K_{\text{eff}} = 1$, $\tilde{W}_{\text{top}} = P_{t, i^*(t)}$ tetap bergantung pada parameter router dan $\nabla_{W_{\text{router}}} \mathcal{L}_{\text{CE}} \neq \mathbf{0}$.

3. **Persamaan Load-Balancing Loss Eksak (DeepSeek-V2 Eq. 16 / Switch Transformer)**:
   Definisikan fraksi penugasan expert $f_i$ dan probabilitas rata-rata $P_i$ untuk $i \in \{0, \dots, N_r - 1\}$:
   $$f_i = \frac{N_r}{K_{\text{eff}} \cdot T} \sum_{t=1}^{T} \mathbb{1}\!\left[i \in I_{\text{top}}(t)\right], \qquad \bar{P}_i = \frac{1}{T} \sum_{t=1}^{T} P_{t, i}$$
   $$\mathcal{L}_{\text{aux}} = \alpha_{\text{aux}} \sum_{i=0}^{N_r - 1} f_i \cdot \bar{P}_i$$
   **Properti Invarian Wajib**: Jika routing seragam sempurna ($\bar{P}_i = \frac{1}{N_r}$ dan setiap expert menerima tepat $\frac{K_{\text{eff}} T}{N_r}$ penugasan), maka $f_i = \frac{N_r}{K_{\text{eff}} T} \cdot \frac{K_{\text{eff}} T}{N_r} = 1$, sehingga $\sum_{i=0}^{N_r - 1} f_i \bar{P}_i = \sum_{i=0}^{N_r - 1} 1 \cdot \frac{1}{N_r} = 1.0$, dan $\mathcal{L}_{\text{aux}} = \alpha_{\text{aux}}$ **independen dari nilai $K_{\text{eff}}$ maupun $N_r$**!
   - Hitung jumlah kemunculan expert secara vektor tanpa loop Python menggunakan `torch.bincount(I_top.reshape(-1), minlength=N_r)`!

4. **Dukungan `num_shared_experts >= 0` & Opsional DeepSeek-V3 Bias-Free Balancing**:
   - Ganti `validate_positive_int("num_shared_experts", num_shared_experts)` pada `ffn.py:198` menjadi `validate_non_negative_int("num_shared_experts", num_shared_experts)`. Jika `num_shared_experts == 0`, inisialisasi `shared_out = torch.zeros_like(x_flat)`.
   - Hapus percabangan sinkron `if expert_mask.any():` di `ffn.py:310`. Gunakan pengurutan indeks token (`sorted_expert_ids, perm = I_top.reshape(-1).sort()`, `counts = bincount(...)`) atau pemrosesan tanpa sinkronisasi host per-expert sehingga bebas *GPU stall* (atau ketika di CPU dengan tensor kecil, tetap bebas `.any()` GPU).
   - Sediakan mode opsional `router_balance_mode: Literal["aux_loss", "bias_free"] = "aux_loss"` di `KilatConfig`: pada mode `"bias_free"` (gaya DeepSeek-V3), seleksi Top-K menggunakan $\operatorname{TopK}(P + b_{\text{expert}})$ sementara bobot penggabungan tetap mengambil $P$ asli tanpa bias, `aux_loss = 0.0`, dan bias $b_{\text{expert}}$ diperbarui tanpa gradien di akhir langkah optimisasi: $b_i \leftarrow b_i + \gamma \operatorname{sign}(\bar{c} - c_i)$. Default tetap `"aux_loss"` untuk menjaga 100% kompatibilitas mundur.

---

## M-05: State Machine Eksplisit `HybridCache` (Perbaikan `recall_ratio=0.0`, Chunked Prefill, & RoPE Cache)

### 1. Struktur Data `HybridCache`
Alih-alih meneruskan tuple mentah `(new_global_state, new_latent_kv)` yang kehilangan informasi panjang sekuens saat `n_recall_heads == 0`, definisikan kelas ringan `LayerHybridCache` (tetap mendukung `__iter__` dan `__getitem__` 2-elemen `(global_state, latent_kv)` untuk kompatibilitas mundur penuh dengan pemanggil lama!):

```python
@dataclass
class LayerHybridCache:
    global_state: torch.Tensor          # (B, H_global, D_head) dalam float32
    latent_kv: torch.Tensor             # (B, S_past, D_latent)
    seq_len: int                        # Panjang token masa lalu aktual S_past >= 0
    rope_k: Optional[torch.Tensor] = None  # (B, H_recall, S_past, D_rope) SUDAH ter-rotasi RoPE!

    def __iter__(self):
        # Kompatibilitas mundur bagi kode lama yang melakukan: g_state, l_kv = past_kv
        yield self.global_state
        yield self.latent_kv
```

### 2. Persamaan Transisi Multi-Token (`N >= 1`) dengan Cache
Misalkan cache masa lalu berada pada panjang $T_0 = \text{seq\_len} \ge 0$ dengan akumulator tak-ternormalisasi $S_{T_0 - 1} = \text{global\_state} \in \mathbb{R}^{B \times H_g \times D_h}$, dan input baru memiliki panjang $N \ge 1$ (baik *single-token decode* $N=1$ maupun *chunked prefill* $N > 1$):
1. **Cabang Global Decay (`n_global_heads > 0`)**:
   Untuk langkah lokal $j \in \{0, \dots, N-1\}$ (posisi absolut $t = T_0 + j$):
   $$S_{T_0 + j} = \lambda S_{T_0 + j - 1} + V_{\text{new}, j} \qquad (S_{-1} = \mathbf{0})$$
   $$y_{T_0 + j} = \frac{S_{T_0 + j}}{z_{T_0 + j}(\lambda)}$$
   di mana $z_{T_0 + j}(\lambda)$ dihitung memakai rumus stabil $\varphi_1$ dari **M-01** dengan $t = T_0 + j$ dari `cache.seq_len` (**bukan** `latent_kv.shape[1]`). Simpan `new_global_state = S_{T_0 + N - 1}` (FP32) dan `new_seq_len = T_0 + N`.
2. **Cabang MLA Recall & Decoupled RoPE (`n_recall_heads > 0`)**:
   - Konkatenasi `latent_kv_total = cat([cache.latent_kv, new_latent_kv], dim=1)` berdimensi `(B, T_0 + N, D_latent)`.
   - Pada `KilatAttentionRoPE`: putar RoPE **hanya** pada query baru (`pos = T_0 .. T_0 + N - 1`) dan key baru (`pos = T_0 .. T_0 + N - 1`), lalu konkatenasikan `rope_k_total = cat([cache.rope_k, new_rope_k_rotated], dim=2)`. Ini memangkas kompleksitas langkah decode dari $O(T^2)$ rotasi ulang menjadi $O(1)$ per langkah dan menjamin kesamaan numerik dengan *prefill* penuh!
   - Masker kausal persegi panjang untuk *chunked prefill* ($N > 1$ dengan $T_0 > 0$): query indeks lokal $j \in [0, N-1]$ boleh menghadiri key indeks absolut $k \in [0, T_0 + j]$.

---

## M-06: Pemisahan $\mathcal{L}_{\text{CE}}$ vs $\mathcal{L}_{\text{aux}}$, Token-Weighted PPL, Akumulasi Parsial, & Skala Residual $\frac{1}{\sqrt{2L}}$

### 1. Kontrak Output Model & Pure Perplexity
Pada `KilatTransformer.forward` (`src/kilat/models/transformer.py`):
- Hitung *cross-entropy loss* murni di `float32`:
  $$\mathcal{L}_{\text{CE}} = \operatorname{CrossEntropy}\!\left(\operatorname{cast}_{\text{fp32}}(\text{logits}_{:, :-1, :}),\; \text{labels}_{:, 1:},\; \text{ignore\_index}=-100,\; \text{reduction}=\text{"mean"}\right)$$
  serta jumlah token valid $T_{\text{valid}} = \sum_{b, t} \mathbb{1}[\text{labels}_{b, t+1} \neq -100]$.
  - **Penjaga Starvasi Token (`T_valid == 0`)**: Jika seluruh label dalam suatu *micro-batch* bernilai `-100`, `F.cross_entropy(..., reduction="mean")` di PyTorch mengembalikan **`NaN`** ($0/0$)! Wajib pasang penjaga: jika `T_valid == 0`, kembalikan `(logits * 0.0).sum()` (nol terdiferensiasi tanpa `NaN`) dan catat peringatan pada `FlightRecorder`.
- Total objektif optimisasi adalah:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{aux}}$$
- Simpan metrik terakhir yang bebas grafik (`detach()`) atau kembalikan objek `KilatModelOutput` yang **tetap kompatibel 100% dengan unpacking tuple 2/3 elemen `(logits, loss)` atau `(logits, loss, presents)`** (misalnya dengan menyimpan `_last_loss_breakdown = {"ce_loss": ..., "aux_loss": ..., "valid_tokens": ...}` pada atribut modul atau subclass tuple bernama `ModelOutput`), sehingga kode lama `logits, loss = model(x, labels=y)` **tidak pernah patah**, sementara `KilatTrainer` dapat membaca `ce_loss`, `aux_loss`, dan `valid_tokens` secara eksak!
- **Evaluasi Token-Weighted & Pure PPL (`trainer.py`)**:
  Pada `_default_eval_loop`:
  $$\overline{\mathcal{L}}_{\text{eval, CE}} = \frac{\sum_{m=1}^{M} \mathcal{L}_{\text{CE}, m} \cdot T_{\text{valid}, m}}{\max\!\left(1,\; \sum_{m=1}^{M} T_{\text{valid}, m}\right)}, \qquad \text{eval\_ppl} = \exp\!\left(\min(\overline{\mathcal{L}}_{\text{eval, CE}}, 20.0)\right)$$
  Log secara terpisah: `train_ce_loss`, `train_aux_loss`, `train_loss` (total), `eval_loss` (murni CE), `eval_aux_loss`, dan `eval_ppl` (murni $\exp(\overline{\mathcal{L}}_{\text{eval, CE}})$).

### 2. Penyelarasan Akumulasi Gradien Parsial & `compute_total_steps`
Misalkan panjang `train_dataloader` adalah $B_{\text{total}}$ *micro-batches* per *epoch*, dan faktor akumulasi gradien adalah $G = \text{gradient\_accumulation\_steps}$:
1. **Total Optimizer Steps (`optimizer.py` / `trainer_utils.py`)**:
   Pada `training_mode == "epochs"`:
   $$S_{\text{epoch}} = \left\lceil \frac{B_{\text{total}}}{G} \right\rceil = \frac{B_{\text{total}} + G - 1}{G} \quad (\text{pembagian bulat atas}), \qquad S_{\text{total}} = S_{\text{epoch}} \times E$$
2. **Skala Gradien Eksak pada Jendela Parsial (`trainer.py`)**:
   Untuk jendela akumulasi ke-$w$, hitung ukuran jendela aktual:
   $$m_w = \min\!\left(G,\; B_{\text{total}} - \text{window\_start\_idx}\right)$$
   Bagi *loss* setiap *micro-batch* di dalam jendela ke-$w$ dengan **`m_w`** (bukan selalu `G`)!
   - Untuk seluruh jendela penuh, $m_w = G$ (identik dengan sebelumnya).
   - Untuk jendela sisa di akhir *epoch* ($1 \le m_w < G$), rata-rata gradien atas $m_w$ *micro-batch* memiliki ekspektasi skala yang **100% tak-bias** ($\frac{1}{m_w}\sum_{j=1}^{m_w} \nabla \ell_j$).

### 3. Skala Inisialisasi Kedalaman Residual (GPT-2 / DeepSeek Standard)
Di `KilatTransformer._init_weights` (`src/kilat/models/transformer.py`), matriks proyeksi keluar yang menjumlahkan langsung ke jalur residual utama (`attn.c_proj.weight`, `ffn.down.weight` baik pada `SwiGLU`, `shared_experts`, maupun `routed_experts`) diinisialisasi ulang setelah pemanggilan `self.apply(self._init_weights)` dengan simpangan baku:

$$\sigma_{\text{residual}} = \frac{\text{config.initializer\_range}}{\sqrt{2 \cdot \text{config.n\_layer}}}$$

Ini menjaga varians aliran residual $\operatorname{Var}(h^{(L)}) = \mathcal{O}(1)$ tidak meledak seiring bertambahnya `n_layer`.

---

## M-07: Penghapusan 450+ Sinkronisasi Stream CUDA di Jalur Panas `forward()`

### 1. Masalah
Pemanggilan `validate_finite_tensor(...)` di dalam `SwiGLU.forward`, `DeepSeekMoE.forward`, `FeedForward.forward`, dan `KilatTransformer.forward` mengeksekusi `if not torch.isfinite(tensor).all():`, yang menyisipkan `cudaStreamSynchronize` implisit ratusan kali dalam satu langkah *forward*.

### 2. Spesifikasi Perbaikan
- Tambahkan flag konfigurasi ringan pada `KilatConfig`: `debug_nan_checks: bool = False` (default **`False`**).
- Hapus seluruh pemanggilan `validate_finite_tensor` tanpa syarat dari dalam `nn.Module.forward()`. Jika `self.debug_nan_checks` bernilai `True` (hanya dipakai saat mendiagnosis divergensi), barulah validator dijalankan.
- Pertahankan fungsi `validate_finite_tensor` di `src/kilat/utils/validators.py` agar seluruh pengujian unit `tests/unit/features/utils/test_validators.py` tetap lulus 100%.
- Pindahkan deteksi `NaN`/`Inf` produksi ke batas akhir langkah optimisasi di `KilatTrainer` dan `FlightRecorderCallback` (satu pengecekan pada skalar `loss` per langkah optimisasi, bukan 450 pengecekan per *micro-batch*).
