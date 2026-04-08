# The Edge-Device Dilemma: Trade-Offs in Speed, Size, and Robustness for Fake News Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


> ⚠️ **Citation Notice:** This repository contains the official implementation
> of the manuscript *"The Edge-Device Dilemma: Trade-Offs in Speed, Size, and
> Robustness for Fake News Detection"*.
> If you use this code or build upon this work, please cite our paper using the
> BibTeX entry provided at the bottom of this README.

---

## Overview

This is the official implementation of:

**"The Edge-Device Dilemma: Trade-Offs in Speed, Size, and Robustness for Fake News Detection"**  submitted for consideration on Sadhana, Springer. 

*Maitreyee Ganguly, Paramita Dey*  
*Department of Information Technology, Government College of Engineering & Ceramic Technology, Kolkata, India*

We study whether a fake news detection model can be simultaneously **fast**, **small**, and **robust to domain shift** — the *Edge-Device Dilemma*. An empirical multi-phase study compares five architectural paradigms across two misinformation datasets, and proposes a **Dynamic Inference Cascading** framework that navigates the efficiency–robustness trade-off in practice.

### Key Findings

| Model | In-Domain Acc. | Zero-Shot Acc. | Latency | Size |
|---|---|---|---|---|
| **Hashing-SVC** | 99.79% | 51–54% | **0.14 ms** | **2 MB** |
| TF-IDF-SVC | 99.80% | 51–55% | 0.22 ms | 76 MB |
| fastText | 99.79% | 52–56% | 0.08 ms | 384 MB |
| **Hybrid Super-Vector** | 99.10% | **65–69%** | 4.82 ms | 14.5 MB |
| BERT | 99.82% | 48–50% | 125 ms | 440 MB |

---

## Architecture

```
Raw Text Input
      │
      ▼
┌─────────────────────────────┐
│  Low-Latency Preprocessor   │   §3.2
│  Regex · Normalise · Tokenise│
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  v_lex (P1/P2)    v_style (P3)
  HashingVec       VADER · textstat
  TF-IDF           readability · ratios
  fastText         (13-dim vector)
       │                │
       └───────┬────────┘
               ▼
      v_final = v_lex ⊕ v_style     §3.3.3
      Hybrid Super-Vector
               │
               ▼
     ┌──────────────────┐
     │   LinearSVC      │
     │   Classifier     │
     └────────┬─────────┘
              │
     ┌────────┴────────────────────┐
     │  Dynamic Inference Cascade  │   §6.1
     │                             │
     │  |score| ≥ τ  → Tier-1     │   fast path   (0.14 ms)
     │  |score| < τ  → Tier-2     │   robust path (4.82 ms)
     └─────────────────────────────┘
```

---

## Repository Structure

```
edge-device-dilemma/
│
├── README.md
├── requirements.txt
├── gitignore
├── __init__.py
│
├── config.py          ← All hyperparameters and dataset paths
├── preprocess.py      ← §3.2: Low-latency preprocessing pipeline
├── dataset.py         ← §3.3: Feature engineering & model builders
│                              (SuperVectorFeaturizer, Hashing-SVC,
│                               TF-IDF-SVC, fastText, Super-Vector)
│
├── train.py           ← §3.4.1 / §4.2: Phase I in-domain efficiency benchmarking
├── evaluate.py        ← §3.4.2 / §4.4: Phase II bidirectional zero-shot robustness
├── ablation.py        ← §4.7:          Ablation study (Table 9)
├── visualize.py       ← §4–6:          All paper figures (Figs 3–6, Table 9)
└── inference.py       ← §6.1–6.2:      Dynamic Inference Cascading + BERT baseline
```

---

## Requirements

### System Requirements

- Python >= 3.8
- CPU sufficient for all models except BERT
- GPU with >= 4 GB VRAM recommended for `inference.py` in `bert` mode

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Maitreyee12aug/edge-device-dilemma.git
cd edge-device-dilemma

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; [nltk.download(r, quiet=True) for r in ('stopwords','punkt','punkt_tab')]"
```

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | >= 1.2.0 | Hashing-SVC, TF-IDF-SVC, LinearSVC |
| fasttext-wheel | >= 0.9.2 | fastText (Paradigm 2) |
| vaderSentiment | >= 3.3.2 | Sentiment features (Super-Vector) |
| textstat | >= 0.7.3 | Readability features (Super-Vector) |
| torch | >= 2.0.0 | BERT baseline |
| transformers | >= 4.30.0 | bert-base-uncased |

---

## Datasets

| Dataset | Role | Size | Source |
|---|---|---|---|
| **ISOT Fake News** (`concatenated_dataset.csv`) | Political corpus — in-domain source | 44,898 articles | [UVic ISOT](https://onlineacademiccommunity.uvic.ca/isot/) |
| **CONSTRAINT 2021** (`Constraint_Train.csv`) | COVID-19 corpus — zero-shot target | 6,420 tweets | [CONSTRAINT 2021](https://constraint-shared-task-2021.github.io/) |

Place both CSV files in the repo root directory. Both require columns `tweet` (text) and `label` (`real` / `fake`). See `config.py` to adjust column names.

**ISOT preparation:**
```python
import pandas as pd
fake = pd.read_csv("Fake.csv"); fake["label"] = "fake"
real = pd.read_csv("True.csv"); real["label"] = "real"
df   = pd.concat([fake, real]).rename(columns={"text": "tweet"})
df[["tweet", "label"]].to_csv("concatenated_dataset.csv", index=False)
```

---

## Usage

### Step 1 — Verify Data Loading
```bash
python preprocess.py
```

### Step 2 — Phase I: In-Domain Efficiency Benchmarking (Tables 6 & 7)
```bash
python train.py
```

### Step 3 — Phase II: Bidirectional Zero-Shot Robustness (Table 8, Figure 5)
```bash
python evaluate.py
```

### Step 4 — Ablation Study (Table 9)
```bash
python ablation.py
```

### Step 5 — Generate All Figures (Figures 3–6)
```bash
python visualize.py
```

### Step 6 — Dynamic Inference Cascading (Figure 8, Table 10)
```bash
python inference.py          # MODE = "cascade" in inference.py
```

### Step 7 — BERT Baseline (GPU recommended)
```bash
# Edit inference.py: set MODE = "bert"
python inference.py
```

---

## Hyperparameters

All hyperparameters are defined in `config.py`.

| Parameter | Value | Paper Reference |
|---|---|---|
| Hashing n_features | 2^18 = 262,144 | §3.3.1-B |
| fastText lr | 1.0 | §3.3.2 |
| fastText dim | 100 | §3.3.2 |
| Super-Vector hash features | 2^14 | §3.3.3 |
| Super-Vector TF-IDF vocab | 2,000 | §3.3.3 |
| BERT model | bert-base-uncased | §3.3.4 |
| BERT epochs | 2 | §3.3.4 |
| Cascade threshold τ | 0.3 | §6.1 |
| Latency warmup iters | 1,000 | §3.4.1 |
| Latency measurement iters | 10,000 | §3.4.1 |

---

## Citation

If you use this code in your research, please cite our paper.

---

## Contact

For questions or issues, please open a GitHub Issue or contact:

- **Maitreyee Ganguly** — maitreyee12aug@gmail.com  
- **Paramita Dey** — dey.paramita77@gmail.com

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/)
- [CONSTRAINT 2021 Shared Task](https://constraint-shared-task-2021.github.io/)
- [fastText](https://fasttext.cc/)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
