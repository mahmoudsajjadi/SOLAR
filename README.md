# SOLAR: Communication-Efficient Model Adaptation via Subspace-Oriented Lightweight Approximation

This repository provides two modular pipelines for evaluating our SOLAR method for efficient model adaptation:

- **`llm/llm.py`** — Instruction tuning and language generation using LLaMA or GPT-2 models.
- **`vit/vit.py`** — Few-shot image classification using ViT-based models with LoRA+SOLAR compression.

**SOLAR** applies a randomized subspace projection and sparsity-aware reconstruction to compress PEFT adapters after training — achieving extreme communication and storage efficiency without significant performance loss.

---

## 📁 Project Structure

```
.
├── vit/
│   ├── vit.py               # Vision Transformer + LoRA + SOLAR
│   ├── requirements.txt     # Vision-specific dependencies
│   └── data/                # Includes data_loader.py for dataset setup
├── llm/
│   ├── llm.py               # GPT-2/LLaMA instruction tuning with SOLAR
│   ├── requirements.txt     # LLM-specific dependencies
└── README.md
```

---

## Setup Instructions

Install dependencies in separate virtual environments:

### ViT Setup

```bash
cd vit
python3 -m venv vit_env
source vit_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> ✅ Datasets: Supports CIFAR-10, CIFAR-100, SUN397, TinyImageNet, etc.  
> ✅ Models: ViT-base (via Hugging Face) or timm-based ViTs  
> ✅ Parameters:
> - `--lora_rank` (e.g. 4)
> - `--retain_params` (e.g. 0.25 means retain 25% of basis)
> - `--num_random_basis` (e.g. 4000)

---

### LLM Setup

```bash
cd llm
python3 -m venv llm_env
source llm_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> ✅ Models: GPT-2, LLaMA  
> ✅ Datasets: Alpaca, E2E NLG, etc.  
> ✅ Evaluation: METEOR via Hugging Face `evaluate`  
> ✅ Compression Parameters:
> - `--retain_params`: percentage (e.g., 0.4 = 40% of weights kept)
> - `--num_random_basis`: number of random basis vectors (e.g. 1000, 4000)

---

## Logging & Monitoring

All experiments support [Weights & Biases](https://wandb.ai) for real-time logging.

Enable it with:

```bash
wandb login
```

Logs include:
- Accuracy, METEOR
- FLOPs, compression stats
- Training & evaluation runtime
- Subspace similarity heatmaps

---
