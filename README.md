# TriEval 🔍

**A Lightweight Distributed Pipeline for Evaluating Large Language Models Across Bias, Toxicity, and Truthfulness**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Models: Llama3 | Mistral | Gemma2 | Claude](https://img.shields.io/badge/models-Llama3%20%7C%20Mistral%20%7C%20Gemma2%20%7C%20Claude-green)]()

---

## Overview

TriEval is an open-source evaluation pipeline that simultaneously assesses **bias**, **toxicity**, and **truthfulness** across multiple large language models (LLMs) in a single reproducible run.

Unlike existing frameworks such as HELM or DecodingTrust, DistEval is designed to run on **consumer-grade hardware** (tested on Apple M-series CPU) without requiring a GPU or expensive cloud infrastructure. A full evaluation across 4 models costs under $2 in API credits and completes in approximately 20 minutes.

This pipeline was developed as part of the research paper:

> **DistEval: A Lightweight Distributed Pipeline for Scalable Evaluation of Large Language Models Across Bias, Toxicity, and Truthfulness**
> Akshatha Srikantha, Independent Researcher, Los Angeles, CA
> 2025

---

## Features

- **Multi-dimensional evaluation** — bias, toxicity, and truthfulness in one unified pipeline
- **Open + closed source models** — Llama 3, Mistral 7B, Gemma 2 (via Ollama) and Claude Haiku (via API)
- **No GPU required** — runs fully on CPU using 4-bit quantized models
- **Judge-LLM scoring** — uses Claude as an evaluator, no dependency on Perspective API
- **Statistical analysis** — mean, standard deviation, 95% confidence intervals, t-tests, Cohen's d
- **Reproducible** — fixed prompts, versioned models, deterministic parsing
- **Open source** — MIT licensed, fully documented

---

## Models Evaluated

| Model | Parameters | Type | Inference |
|-------|-----------|------|-----------|
| Llama 3 8B | 8B | Open-source | Local (Ollama) |
| Mistral 7B | 7B | Open-source | Local (Ollama) |
| Gemma 2 9B | 9B | Open-source | Local (Ollama) |
| Claude Haiku | N/A | Closed-source | Anthropic API |

---

## Evaluation Dimensions

### 1. Toxicity
- 50 adversarial prompts designed to elicit harmful responses
- Scored 0.0 (harmless) to 1.0 (extremely toxic) using Claude-as-judge
- Refusal responses automatically scored as 0.05 (safe behaviour)

### 2. Truthfulness
- 100 questions sampled from the TruthfulQA dataset
- Multiple choice format with ground truth labels
- Accuracy reported as proportion of correctly answered questions

### 3. Bias
- 25 paired prompt sets across 5 demographic dimensions:
  - Gender, Race, Religion, Age, Nationality
- Each pair scored for sentiment differential and professionalism differential
- Bias detection rate reported per model and per dimension

---

## Results

### Toxicity (lower = better)
| Model | Avg Score |
|-------|-----------|
| Claude Haiku | 0.050 |
| Gemma 2 9B | 0.060 |
| Llama 3 8B | 0.060 |
| Mistral 7B | 0.085 |

### Truthfulness (higher = better)
| Model | Accuracy |
|-------|----------|
| Gemma 2 9B | 83.3% |
| Mistral 7B | 63.3% |
| Llama 3 8B | 40.0% |
| Claude Haiku | 0.0%* |

*Claude Haiku scored 0% due to prompt format sensitivity — see paper for full discussion.

---

## Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed on your machine
- Anthropic API key

### Setup

```bash
# Clone the repository
git clone https://github.com/physics-vibes15/disteval.git
cd disteval

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install ollama anthropic python-dotenv datasets pandas numpy matplotlib seaborn scipy

# Pull open-source models via Ollama
ollama pull llama3
ollama pull mistral
ollama pull gemma2
```

### API Key Setup

DistEval requires an Anthropic API key to access Claude Haiku for closed-source model evaluation and for the judge-LLM scoring functions. You can obtain an API key by creating an account at [console.anthropic.com](https://console.anthropic.com). Once obtained, store your key securely as an environment variable in your local development environment. Never commit API keys to version control.

---

## Usage

### Basic evaluation (v1)

Run `evaluate.py` after setting up your environment and ensuring Ollama is running with the required models downloaded. This version evaluates all four models across 10 toxicity prompts, 30 TruthfulQA questions, and 5 bias prompt pairs. Results are saved automatically as CSV files in the project directory.

### Journal-quality evaluation with statistical analysis (v2)

Run `evaluate_v2.py` for a more comprehensive evaluation suitable for research publication. This version uses 50 toxicity prompts, 100 TruthfulQA questions, and 25 bias prompt pairs across 5 demographic dimensions. It additionally computes mean, standard deviation, 95% confidence intervals, pairwise t-tests, and Cohen's d effect sizes for each model comparison.

### Output files
| File | Contents |
|------|----------|
| `results_toxicity.csv` | Per-prompt toxicity scores per model |
| `results_truthfulness.csv` | Per-question accuracy per model |
| `results_bias.csv` | Per-pair bias scores per model |
| `results_summary.csv` | Aggregated summary across all dimensions |

---

## Repository Structure

```
disteval/
├── evaluate.py              # Basic evaluation pipeline (v1)
├── evaluate_v2.py           # Journal-quality pipeline with statistics (v2)
├── results_toxicity.csv     # Toxicity experiment results
├── results_truthfulness.csv # Truthfulness experiment results
├── results_bias.csv         # Bias experiment results
├── results_summary.csv      # Summary results
├── .env                     # API keys (not committed)
├── LICENSE                  # MIT License
└── README.md                # This file
```

---

## Citation

If you use DistEval in your research, please cite:

```bibtex
@article{srikantha2025disteval,
  title={DistEval: A Lightweight Distributed Pipeline for Scalable Evaluation 
         of Large Language Models Across Bias, Toxicity, and Truthfulness},
  author={Srikantha, Akshatha},
  year={2025},
  url={https://github.com/physics-vibes15/disteval}
}
```

---

## Author

**Akshatha Srikantha**
Independent Researcher | Los Angeles, CA
ORCID: [0009-0005-3753-9848](https://orcid.org/0009-0005-3753-9848)
Email: sakshatha99@gmail.com

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds on foundational evaluation frameworks including TruthfulQA, RealToxicityPrompts, HELM, and DecodingTrust. Open-source model inference is powered by [Ollama](https://ollama.com). Closed-source model evaluation uses the [Anthropic API](https://anthropic.com).
