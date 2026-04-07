"""
scripts/05_bert_experiment.py

BERT Baseline Experiment (paper §3.3.4, §4.2, §4.4).

Fine-tunes bert-base-uncased on the Political corpus (source domain) and
evaluates both in-domain accuracy and zero-shot cross-domain accuracy on
the COVID-19 corpus.

Confirms the paper's finding that BERT collapses to near-random-chance
(≈48%) on the unseen target domain despite 99.8% in-domain accuracy,
validating the "Semantic Overfitting" analysis in §5.3.

GPU STRONGLY RECOMMENDED. CPU training will take several hours.
"""

import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.bert_model import run_bert_experiment
from src.preprocessing.text_cleaner import preprocess_series

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_dataset(path, text_col, label_col, real_val, fake_val):
    df = pd.read_csv(path).dropna()
    df = df.rename(columns={text_col: "tweet", label_col: "label"})
    df["label_num"] = df["label"].map({real_val: 0, fake_val: 1})
    df = df.dropna(subset=["label_num", "tweet"])
    df["label_num"] = df["label_num"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dcfg = cfg["data"]

    print("Loading Political corpus (source / training domain)...")
    df_pol = load_dataset(
        dcfg["political_csv"], dcfg["political_text_col"],
        dcfg["political_label_col"], dcfg["political_real_val"], dcfg["political_fake_val"],
    )

    print("Loading COVID-19 corpus (target / zero-shot domain)...")
    df_cov = load_dataset(
        dcfg["covid_csv"], dcfg["covid_text_col"],
        dcfg["covid_label_col"], dcfg["covid_real_val"], dcfg["covid_fake_val"],
    )

    print(f"\nSource (Political): {len(df_pol)} samples")
    print(f"Target (COVID-19):  {len(df_cov)} samples")

    # Run experiment
    results = run_bert_experiment(
        src_texts=df_pol["tweet"].tolist(),
        src_labels=df_pol["label_num"].tolist(),
        tgt_texts=df_cov["tweet"].tolist(),
        tgt_labels=df_cov["label_num"].tolist(),
        cfg=cfg,
    )

    # Save summary
    summary = pd.DataFrame([{
        "model":           "BERT (bert-base-uncased)",
        "source_accuracy": results["source_accuracy"],
        "source_f1":       results["source_f1"],
        "target_accuracy": results["target_accuracy"],
        "target_f1":       results["target_f1"],
        "delta":           results["delta"],
        "rho_pct":         results["rho_pct"],
        "latency_ms":      results["avg_latency_ms"],
        "size_mb":         results["model_size_mb"],
    }])

    out_csv = os.path.join("results", "bert_results.csv")
    os.makedirs("results", exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")
