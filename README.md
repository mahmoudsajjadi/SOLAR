# SOLAR: Communication-Efficient Model Adaptation via Subspace-Oriented Lightweight Approximation

This repository provides two modular pipelines for evaluating our SOLAR method for efficient model adaptation:

- **`llm/llm.py`** — Instruction tuning and language generation using LLaMA or GPT-2 models.
- **`vit/vit.py`** — Few-shot image classification using ViT-based models with LoRA+SOLAR compression.

**SOLAR** applies a randomized subspace projection and sparsity-aware reconstruction to compress LoRA adapters after training — achieving extreme communication and storage efficiency without significant performance loss.


## 🔧 Configuration Notes

> **Important**:  
> - **Update model paths** in `MODEL_DATA_CONFIG` inside `llm/llm.py` to match your local or Hugging Face-hosted models.  
> - **Insert your personal `wandb` token** in the `wandb.login()` call if logging is enabled. For anonymized submissions, remove or comment this out.  

```python
# In llm.py:
login(token='YOUR_WANDB_TOKEN_HERE', add_to_git_credential=False)
```

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
> ✅ Evaluation: BLEU, METEOR, ROUGE-L via Hugging Face `evaluate`  
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

---

## 📄 Related Research & Citations

If you find this repository useful, please also check out our recent peer-reviewed publications and benchmarks in distributed learning, multi-agent systems, and smart energy grids:

- **[Sustainable Smart Grid Resilience and Renewable Integration Using Multi-Agent Reinforcement Learning](https://www.mdpi.com/2071-1050/18/20/10574)** — *Seyed Mahmoud Sajjadi Mohammadabadi, Sayed Mohammad Sajjadi*, *MDPI Sustainability*, Vol. 18, Issue 20, 10574 (2026). [[MDPI Open Access](https://www.mdpi.com/2071-1050/18/20/10574)] • [[DOI: 10.3390/su182010574](https://doi.org/10.3390/su182010574)] • [[Code & Benchmarks](https://github.com/mahmoudsajjadi/Smart-Grid-Resilience-MARL)] • [[Interactive Page & BibTeX](https://mahmoudsajjadi.github.io/papers/sustainable-smart-grid-resilience-marl.html)]
- **[Speed Up Federated Learning in Heterogeneous Environments: A Dynamic Tiering Approach](https://doi.org/10.1109/JIOT.2024.3468511)** — *Seyed Mahmoud Sajjadi Mohammadabadi, Syed Zawad, Feng Yan, Lei Yang*, *IEEE Internet of Things Journal*, Vol. 12, No. 5, pp. 5026–5035 (2025). [[DOI](https://doi.org/10.1109/JIOT.2024.3468511)] • [[Code](https://github.com/mahmoudsajjadi/DTFL)]
- **[Communication-Efficient Training Workload Balancing for Decentralized Multi-Agent Learning](https://doi.org/10.1109/ICDCS60910.2024.00062)** — *Seyed Mahmoud Sajjadi Mohammadabadi, Lei Yang, Feng Yan, Junshan Zhang*, *IEEE ICDCS* (2024). [[DOI](https://doi.org/10.1109/ICDCS60910.2024.00062)] • [[Code](https://github.com/mahmoudsajjadi/ComDML)]
- **[Towards Distributed Learning of PMU Data: A Federated Learning Based Event Classification Approach](https://doi.org/10.1109/PESGM52003.2023.10252874)** — *Seyed Mahmoud Sajjadi Mohammadabadi, Yunchuan Liu, Abraham Canafe, Lei Yang*, *IEEE PESGM* (2023). [[DOI](https://doi.org/10.1109/PESGM52003.2023.10252874)]

```bibtex
@article{sajjadi2026sustainable,
  title     = {Sustainable Smart Grid Resilience and Renewable Integration Using Multi-Agent Reinforcement Learning},
  author    = {Sajjadi Mohammadabadi, Seyed Mahmoud and Sajjadi, Sayed Mohammad},
  journal   = {Sustainability},
  volume    = {18},
  number    = {20},
  pages     = {10574},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/su182010574},
  url       = {https://www.mdpi.com/2071-1050/18/20/10574}
}
```

> 🏛️ Explore all 21 publications and 1-click BibTeX citations at the [**Academic Research Hub**](https://mahmoudsajjadi.github.io/research-hub.html) and [**Google Scholar**](https://scholar.google.com/citations?user=tkhCR5IAAAAJ&hl=en).