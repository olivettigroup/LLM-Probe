# LLM-Probe: Probing Materials Intelligence in Large Language Models

Code and data for the paper: **"Probing Materials Intelligence in LLMs: From Latent Embeddings to Reliable Predictions"**

Vineeth Venugopal, Soroush Mahjoubi, Elsa Olivetti
*Massachusetts Institute of Technology*

---

## Overview

This repository contains notebooks for fine-tuning and evaluating large language models on materials science tasks:

- **Bandgap Prediction** - Regression (eV)
- **Dielectric Constant Prediction** - Regression
- **Crystal System Classification** - 7-class classification
- **MatKG Link Prediction** - Knowledge graph completion

## Datasets

Available on Hugging Face:

| Dataset | Task | Link |
|---------|------|------|
| Bandgap | Regression | [vinven7/materials-bandgap-prediction](https://huggingface.co/datasets/vinven7/materials-bandgap-prediction) |
| Dielectric | Regression | [vinven7/materials-dielectric-prediction](https://huggingface.co/datasets/vinven7/materials-dielectric-prediction) |
| Crystal System | Classification | [vinven7/materials-crystal-system-classification](https://huggingface.co/datasets/vinven7/materials-crystal-system-classification) |
| MatKG | Link Prediction | [vinven7/matkg-link-prediction](https://huggingface.co/datasets/vinven7/matkg-link-prediction) |

## Fine-tuned Models

LoRA adapters available on Hugging Face ([vinven7](https://huggingface.co/vinven7)):

| Base Model | Bandgap | Dielectric | Crystal System | MatKG |
|------------|---------|------------|----------------|-------|
| Llama-2-7B | [link](https://huggingface.co/vinven7/Llama2-ft-BandGap) | [link](https://huggingface.co/vinven7/Llama2-ft-Dielectric) | [link](https://huggingface.co/vinven7/Llama2-ft-CrystalStructure) | [link](https://huggingface.co/vinven7/Llama2-ft-MatKG) |
| Llama-3-8B | [link](https://huggingface.co/vinven7/Llama3-ft-BandGap) | [link](https://huggingface.co/vinven7/Llama3-ft-Dielectric) | [link](https://huggingface.co/vinven7/Llama3-ft-CrystalStructure) | [link](https://huggingface.co/vinven7/Llama3-ft-MatKG) |
| Mistral-7B | [link](https://huggingface.co/vinven7/Mistral-ft-BandGap) | [link](https://huggingface.co/vinven7/Mistral-ft-Dielectric) | [link](https://huggingface.co/vinven7/Mistral-ft-CrystalStructure) | [link](https://huggingface.co/vinven7/Mistral-ft-MatKG) |
| Mixtral-8x7B | [link](https://huggingface.co/vinven7/Mixtral-ft-BandGap) | [link](https://huggingface.co/vinven7/Mixtral-ft-Dielectric) | [link](https://huggingface.co/vinven7/Mixtral-ft-CrystalStructure) | [link](https://huggingface.co/vinven7/Mixtral-ft-MatKG) |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_Predibase_Finetuning_Inference.ipynb` | Fine-tuning open-source LLMs (Llama, Mistral, Gemma) via Predibase |
| `02_GPT_Finetuning_Inference.ipynb` | Fine-tuning GPT models via OpenAI API |
| `03_Embedding_Extraction.ipynb` | Layer-wise embedding extraction and probe training |
| `04_Results_Analysis.ipynb` | Aggregate results and compute metrics |
| `05_Figures.ipynb` | Generate manuscript figures |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Fine-tuning with Predibase

```python
from predibase import Predibase

pb = Predibase(api_token="YOUR_TOKEN")
adapter = pb.adapters.create(
    config="path/to/config.yaml",
    dataset="path/to/data.csv"
)
```

### Loading fine-tuned adapters

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(base_model, "vinven7/Llama2-ft-BandGap")
```

### Embedding extraction

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    output_hidden_states=True
)

with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden_dim)
```

## Citation

```bibtex
@article{venugopal2026probing,
  title={Probing Materials Intelligence in LLMs: From Latent Embeddings to Reliable Predictions},
  author={Venugopal, Vineeth and Mahjoubi, Soroush and Olivetti, Elsa},
  journal={},
  year={2026}
}
```

## Related Work

- [MatKG: An autonomously generated knowledge graph in Materials Science](https://www.nature.com/articles/s41597-024-03039-z) (Scientific Data, 2024)

## License

MIT License
