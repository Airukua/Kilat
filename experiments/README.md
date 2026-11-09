# TinyStories Pre-Training & Benchmark Experiments

Direktori ini berisi seluruh kebutuhan untuk mereplikasi dan melatih model pada dataset **TinyStories** secara mandiri:

---

## 1. File Eksperimen (Flat Layout)

- `train_gpu.yaml`: Konfigurasi latihan GPU (35M parameters, Hybrid Sub-Quadratic Global Decay + Latent MLA + RoPE + SwiGLU).
- `baseline_gpu.yaml`: Konfigurasi baseline untuk perbandingan (Standard Dense Attention Transformer).
- `quick_smoke.yaml`: Konfigurasi uji cepat / debugging.
- `train.npy` & `eval.npy`: Tokenized dataset TinyStories (memmap).
- `eval_prompts.yaml`: 44 Canonical Evaluation Prompts resmi dari paper peneliti TinyStories (Microsoft Research).
- `eval.py`: Script benchmark evaluasi mandiri (mengukur Perplexity, Repetition Rate, dan Text Generation).

---

## 2. Cara Melatih Model Sendiri

### A. Siapkan Dataset Bersih (Bebas Kebocoran & Berbatas EOS)
```bash
python experiments/prepare_clean_data.py --train-stories 50000 --eval-stories 5000
```

### B. Audit Integritas Data & Kebocoran N-Gram
Pastikan token EOS hadir dan kontaminasi n-gram 0.00% (CLEAN):
```bash
python tools/audit/data_auditor.py --train experiments/train.npy --eval experiments/eval.npy
```

### C. Pre-Flight Assurance Verification
Pastikan konfigurasi memenuhi semua syarat stabilitas numerik:
```bash
kilat verify -c experiments/train_gpu.yaml
```

### D. Mulai Latihan di GPU
```bash
kilat train -c experiments/train_gpu.yaml
```
*(Checkpoint akan disimpan secara otomatis di `checkpoints/tinystories_gpu`)*

---

## 3. Cara Menguji & Membongkar Hasil Latihan

### A. Uji Falsifikasi Perilaku & Kontrol Negatif (Benchmark Kausal)
```bash
python tools/benchmarking/behavioral_harness.py \
  -c checkpoints/tinystories_gpu/checkpoint-best \
  -t tools/benchmarking/test_cases.json \
  --device cuda
```

### B. Bedah Dinamika Layer Otak (Logit Lens)
```bash
python tools/interpretability/logit_lens.py \
  -c checkpoints/tinystories_gpu/checkpoint-best \
  -p "Tom has a red ball and a blue ball. He gives the red ball to Anna. Tom still has a" \
  --device cuda
```

### C. Causal Activation Patching (Intervensi Langsung)
```bash
python tools/interpretability/activation_patching.py \
  -c checkpoints/tinystories_gpu/checkpoint-best \
  --clean-prompt "Tom has a red ball and a blue ball. He gives the red ball to Anna. Tom still has a" \
  --corrupt-prompt "Tom has a red ball and a green ball. He gives the red ball to Anna. Tom still has a" \
  --target " blue" \
  --foil " green" \
  --patch-pos 7 \
  --device cuda
```

### D. Evaluasi Benchmark Paper Resmi (44 Prompts + Perplexity)
```bash
python experiments/eval.py \
  -c checkpoints/tinystories_gpu/checkpoint-best \
  -p experiments/eval_prompts.yaml \
  --eval-data experiments/eval.npy \
  --max-prompts 5 \
  --device cuda
```

### E. Generate Cerita Bebas (CLI Kilat)
```bash
kilat generate \
  -c checkpoints/tinystories_gpu/checkpoint-best \
  -p "Once upon a time, there was a little girl named Lily who" \
  --max-new-tokens 60 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.2 \
  --do-sample \
  --device cuda
```
