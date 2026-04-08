# The Edge-Device Dilemma: Trade-Offs in Speed, Size, and Robustness for Fake News Detection

> **Ganguly, M. & Dey, P.** — Department of Information Technology, Government College of Engineering & Ceramic Technology  


---

## Overview

This repository contains the full source code for the paper **"The Edge-Device Dilemma"**, an empirical study that quantifies the trade-offs between inference efficiency, model size, and zero-shot cross-domain generalization for fake news detection on resource-constrained edge devices.

The paper asks: *can a model be simultaneously fast, small, and robust to domain shift?* The answer is — not without deliberate architectural choices. This repo reproduces all experiments and figures from the paper.

---

## Key Findings

| Model | In-Domain Acc. | Cross-Domain Acc. | Latency | Size |
|---|---|---|---|---|
| **Hashing-SVC** | 99.79% | 51–54% | **0.14 ms** | **2 MB** |
| TF-IDF + SVC | 99.80% | 51–55% | 0.22 ms | 76 MB |
| fastText | 99.79% | 52–56% | 0.08 ms | 384 MB |
| **Hybrid Super-Vector** | 99.10% | **65–69%** | 4.82 ms | 14.5 MB |
| BERT | 99.82% | 48–50% | 125 ms | 440 MB |

**Proposed framework:** *Dynamic Inference Cascading* — a two-tier system that uses Hashing-SVC for high-confidence inputs and falls back to the Hybrid Super-Vector when domain shift is detected, achieving 88% greater efficiency than static BERT while recovering accuracy during drift events.

---

## Repository Structure

```
edge-device-dilemma/
│
├── README.md                          # This file
├── requirements.txt                   # All Python dependencies
├── setup.sh                           # One-command environment setup
│
├── configs/
│   └── config.yaml                    # All dataset paths and hyperparameters
│
├── data/
│   └── README_data.md                 # Dataset download & preparation instructions
│
├── src/                               # Core library (importable modules)
│   ├── preprocessing/
│   │   └── text_cleaner.py            # Preprocessing pipeline
│   ├── features/
│   │   └── feature_engineering.py     # Super-Vector featurizer & stylistic features
│   ├── models/
│   │   ├── lightweight_models.py      # Hashing-SVC, TF-IDF, fastText wrappers
│   │   ├── bert_model.py              # BERT fine-tuning & evaluation
│   │   └── dynamic_cascade.py         # Dynamic Inference Cascading framework
│   └── evaluation/
│       └── metrics.py                 # Accuracy, F1, latency, footprint utilities
│
├── scripts/                           # End-to-end runnable experiment scripts
│   ├── 01_phase1_efficiency.py        # Phase I: In-domain efficiency benchmarking
│   ├── 02_phase2_robustness.py        # Phase II: Bidirectional zero-shot transfer
│   ├── 03_ablation_study.py           # Ablation: Lexical vs. Stylistic vs. Super-Vector
│   ├── 04_dynamic_cascade.py          # Dynamic Inference Cascading simulation
│   └── 05_bert_experiment.py          # BERT training + cross-domain evaluation
│
├── notebooks/
│   └── full_pipeline_walkthrough.ipynb  # Google Colab-ready notebook
│
└── results/
    └── figures/                       # Generated plots (saved by scripts)
```

---

## Datasets

Two publicly available datasets are used. See [`data/README_data.md`](data/README_data.md) for download links and preparation steps.

| Dataset | Role | Size | Source |
|---|---|---|---|
| **ISOT Fake News** (`concatenated_dataset.csv`) | Primary in-domain (Political) | 44,898 articles | [UVic ISOT](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) |
| **CONSTRAINT 2021** (`Constraint_Train.csv`) | Zero-shot target (COVID-19) | 6,420 tweets | [CONSTRAINT 2021 Shared Task](https://constraint-shared-task-2021.github.io/) |

Both datasets use binary labels: `real` → `0`, `fake` → `1`.

---

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/edge-device-dilemma.git
cd edge-device-dilemma
bash setup.sh
```

### 2. Add your data

Place your datasets in the `data/` directory:

```
data/
  concatenated_dataset.csv   # Political corpus (ISOT)
  Constraint_Train.csv       # COVID-19 corpus (CONSTRAINT 2021)
```

Confirm column names match what is set in [`configs/config.yaml`](configs/config.yaml).

### 3. Run experiments

```bash
# Phase I: In-domain efficiency (Hashing, TF-IDF, fastText, Super-Vector)
python scripts/01_phase1_efficiency.py

# Phase II: Bidirectional zero-shot robustness
python scripts/02_phase2_robustness.py

# Ablation study (lexical vs. stylistic vs. combined)
python scripts/03_ablation_study.py

# Dynamic Inference Cascading simulation
python scripts/04_dynamic_cascade.py

# BERT baseline (GPU strongly recommended)
python scripts/05_bert_experiment.py
```

All plots are saved to `results/figures/`.

### 4. Google Colab

Open [`notebooks/full_pipeline_walkthrough.ipynb`](notebooks/full_pipeline_walkthrough.ipynb) in Colab for a GPU-enabled walkthrough of the full pipeline.

---

## Reproducing the Paper's Results

All hyperparameters used in the paper are stored in `configs/config.yaml`. The scripts read from this file directly — no code changes needed.

| Script | Paper Section | Expected Runtime (CPU) |
|---|---|---|
| `01_phase1_efficiency.py` | §3.4.1, §4.2 | ~5–15 min |
| `02_phase2_robustness.py` | §3.4.2, §4.4 | ~10–25 min |
| `03_ablation_study.py` | §4.7 | ~5–10 min |
| `04_dynamic_cascade.py` | §6.1, §6.2 | ~5 min |
| `05_bert_experiment.py` | §3.3.4, §4.2 | ~2–8 hrs (CPU) / ~30 min (GPU) |

---



---

## License

This code is released for academic reproducibility. Please cite the paper if you use this work.
