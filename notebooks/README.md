# ⚡ Kilat Interactive Tutorial Notebooks

This directory contains self-contained, step-by-step English tutorials runnable locally or on a free Google Colab T4 GPU:

| # | Notebook | Topics Covered |
| :---: | :--- | :--- |
| **1** | [`kilat_general_pretraining_colab.ipynb`](./kilat_general_pretraining_colab.ipynb) | **Pre-Training From Scratch & Hugging Face Export**: Configuring Macro-Synergy (`Global Decay` + `Latent MLA`), running the 6-Gate Pre-Flight Assurance Pipeline (`G0`–`G6`), training from scratch via native `KilatTrainer` or `AutoModelForCausalLM.from_config()`, and exporting a standalone Hub bundle (`export_to_hf`). |
| **2** | [`kilat_peft_lora_example.ipynb`](./kilat_peft_lora_example.ipynb) | **Parameter-Efficient Fine-Tuning (PEFT LoRA)**: Wrapping `KilatForCausalLM` with `get_kilat_lora_config()`, protecting `log_lambda` recurrence parameters, running 6-Gate Assurance directly on `PeftModel`, and merging weights via `merge_and_unload()`. |
| **3** | [`kilat_sft_colab.ipynb`](./kilat_sft_colab.ipynb) | **Post-Training Alignment (SFT & DPO)**: Preparing models with `prepare_kilat_for_post_training()` (`use_cache=False`, non-reentrant gradient checkpointing), masking user prompts with `-100` (`KilatCompletionOnlyCollator`), and aligning preferences via Direct Preference Optimization (`compute_dpo_preference_loss`). |
