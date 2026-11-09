# 07 — Kilat Future Architecture Blueprint: Macro-Synergy & Interleaved Stacking

> **Doc ID**: `DOC-KILAT-07-MACRO-SYNERGY`  
> **Status**: Accepted Engineering Specification (Future Roadmap)  
> **Target Release**: Kilat Next-Gen Sub-Quadratic Hybrid LLM  
> **Preceding Reference**: [`02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md`](02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md), [`06_MODEL_CERTIFICATION_AND_BENCHMARK_PLAN.md`](06_MODEL_CERTIFICATION_AND_BENCHMARK_PLAN.md)

---

## 1. Executive Summary & Problem Diagnosis

Pada arsitektur Kilat-v2 standar (`recall_ratio=0.5`), penggabungan Global Decay dan Multi-Head Latent Attention (MLA) dilakukan secara intra-layer split:

$$\mathbf{y}_t = W_O \begin{bmatrix} \mathbf{y}_t^{(g)} \\ \mathbf{y}_t^{(r)} \end{bmatrix} = W_O^{(g)} \mathbf{y}_t^{(g)} + W_O^{(r)} \mathbf{y}_t^{(r)}$$

### Keterbatasan Kritis Concatenation Statis
1. **Superposisi Linier Buta Konteks**: Matriks $W_O^{(g)}$ dan $W_O^{(r)}$ bersifat statis. Model tidak dapat secara dinamis mematikan noise peluruhan kontinu saat menghadapi token kueri yang membutuhkan penarikan asosiatif murni.
2. **Kebutaan Representasi Hulu**: Kueri MLA $\mathbf{q}_t$ dibentuk murni dari token instan $\mathbf{x}_t$ tanpa memanfaatkan akumulasi riwayat $\mathbf{h}_t$ yang telah dihitung Global Decay.
3. **Bottleneck Kapasitas Sekuens Panjang**: Pada evaluasi MQAR ($N=256$, 16 pasang key-value), 2 kepala MLA di setiap layer mengalami interferensi sinyal, menyebabkan akurasi runtuh ke **15.62%**.

---

## 2. Bukti Empiris Eksperimen GPU

Pengujian langsung dilakukan di atas hardware CUDA dengan model micro ($d=64, H=4, L=2$):

### A. Akurasi & Ketajaman Asosiatif (MQAR)
- **Kilat-v2 Baseline (Concat 50/50)**:
  - $N=64$: 97.10% (1200 langkah) / 67.19% (400 langkah)
  - $N=128$: 100.00% (1000 langkah) / 13.67% (400 langkah)
  - $N=256$: **15.62%** (1000 langkah) ❌ *Stagnasi Kritis*
- **Macro Synergy (Interleaved L0 Decay $\to$ L1 MLA)**:
  - $N=64$: **98.80%** (1000 langkah) / 74.22% (400 langkah)
  - $N=128$: **100.00%** (1000 langkah) / 33.20% (400 langkah)
  - $N=256$: **96.09%** (1000 langkah) 🏆 **(+80.47% peningkatan mutlak)**

### B. Profil Efisiensi Komputasi & Memori
1. **KV Cache Footprint per Batch**:
   - $N=2048$: Ref Transformer = 1,024 KB | Kilat Concat = 320 KB | **Macro Synergy = 160 KB (6.4x vs Ref, 2.0x vs Base)**.
   - $N=8192$: Ref Transformer = 4,096 KB | Kilat Concat = 1,280 KB | **Macro Synergy = 640 KB**.
   - *Penyebab*: Layer 0 tidak memerlukan KV Cache (0 bita).
2. **Throughput Prefill**:
   - $N=512$: Macro Synergy = **423,808 tok/s** (+59% dibanding Kilat Concat 266,162 tok/s).
   - $N=1024$: Macro Synergy = **276,082 tok/s** (Tercepat, mengalahkan Transformer 269k tok/s).
3. **Keandalan VRAM**:
   - Reference Transformer mengalami **CUDA Out of Memory (OOM)** pada $N=4096$, sedangkan arsitektur Macro Synergy dan Kilat-v2 berjalan tanpa OOM.

---

## 3. Spesifikasi Arsitektur Interleaved Stacking

```
Layer 0 (Genap)  : 100% Pure Global Decay Contextualizer
                   - 4 Heads Triton Parallel Prefix Scan
                   - Komputasi O(N), KV Cache = 0 BYTES
                   - Output: Representasi temporal kausal matang

Layer 1 (Ganjil) : 100% Pure Latent MLA Associative Pointer
                   - 4 Heads Latent Attention + Decoupled RoPE
                   - Menggunakan kueri yang sudah diperkaya konteks Layer 0
                   - Mendedikasikan 100% kapasitas untuk penarikan memori tajam
```

---

## 4. Rencana Transisi Kode & Kompatibilitas

Untuk rilis mendatang:
1. `KilatConfig` akan diperluas dengan parameter `layer_layout: Literal["hybrid_50_50", "interleaved"]`.
2. `KilatTransformer` akan menginisialisasi blok secara selang-seling jika `layer_layout="interleaved"`.
3. Invarian cache `LayerHybridCache` tetap terjaga dengan `latent_kv=None` pada layer genap.
