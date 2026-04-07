"""
scripts/02_phase2_robustness.py

Phase II: Bidirectional Zero-Shot Robustness Testing (paper §3.4.2, §4.4).

Two cross-domain experiments:
  Exp A (Political → COVID-19): Train on large Political corpus;
         evaluate zero-shot on unseen COVID-19 misinformation.
  Exp B (COVID-19 → Political): Train on low-resource COVID-19;
         evaluate on broad Political corpus.

Measures Absolute Decay (Δ) and Relative Decay Ratio (ρ) per Eqs. 6–7.
Reproduces Table 8 and Figure 5 (Generalization Gap) from the paper.
"""

import os
import sys

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import (
    build_results_table,
    compute_robustness_decay,
    evaluate_model,
    print_results_table,
)
from src.models.lightweight_models import (
    build_hashing_svc,
    build_super_vector_model,
    build_tfidf_svc,
    predict_fasttext,
    train_fasttext,
)
from src.preprocessing.text_cleaner import preprocess_series

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

OUT_DIR = cfg["output"]["figures_dir"]
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_dataset(path, text_col, label_col, real_val, fake_val):
    df = pd.read_csv(path).dropna()
    df = df.rename(columns={text_col: "tweet", label_col: "label"})
    df["label_num"] = df["label"].map({real_val: 0, fake_val: 1})
    df = df.dropna(subset=["label_num", "tweet"])
    df["label_num"] = df["label_num"].astype(int)
    df["cleaned_tweet"] = preprocess_series(df["tweet"])
    return df


# ---------------------------------------------------------------------------
# One cross-domain experiment
# ---------------------------------------------------------------------------

def run_zero_shot_experiment(
    df_train, df_test, exp_name: str, cfg: dict
) -> list:
    """
    Train all models on df_train; evaluate zero-shot on df_test.

    Returns list of result dicts (one per model).
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  Train: {len(df_train)} samples  |  Test: {len(df_test)} samples")
    print(f"{'='*60}")

    X_train_raw     = df_train["tweet"]
    X_train_cleaned = df_train["cleaned_tweet"]
    y_train         = df_train["label_num"]

    X_test_raw     = df_test["tweet"]
    X_test_cleaned = df_test["cleaned_tweet"]
    y_test         = df_test["label_num"]

    results = []

    def _record(name, y_pred, src_acc):
        m = evaluate_model(y_test, y_pred)
        decay = compute_robustness_decay(src_acc, m["accuracy"])
        results.append({
            "model":           name,
            "source_accuracy": src_acc,
            "target_accuracy": m["accuracy"],
            "target_f1":       m["f1"],
            "delta":           decay["delta"],
            "rho_pct":         decay["rho_pct"],
        })
        print(f"  {name:<28} src={src_acc:.2%}  tgt={m['accuracy']:.2%}  "
              f"Δ={decay['delta']:.2%}  [ρ={decay['rho_pct']:.1f}%]")

    # --- TF-IDF ---
    print("\n[1] TF-IDF-SVC")
    m = build_tfidf_svc(cfg)
    m.fit(X_train_cleaned, y_train)
    src_acc = evaluate_model(y_train, m.predict(X_train_cleaned))["accuracy"]
    _record("TF-IDF", m.predict(X_test_cleaned), src_acc)

    # --- Hashing-SVC ---
    print("[2] Hashing-SVC")
    m = build_hashing_svc(cfg)
    m.fit(X_train_cleaned, y_train)
    src_acc = evaluate_model(y_train, m.predict(X_train_cleaned))["accuracy"]
    _record("Hashing-SVC", m.predict(X_test_cleaned), src_acc)

    # --- fastText ---
    print("[3] fastText")
    ft = train_fasttext(X_train_cleaned, y_train, cfg)
    src_acc = evaluate_model(y_train, predict_fasttext(ft, X_train_cleaned))["accuracy"]
    _record("fastText", predict_fasttext(ft, X_test_cleaned), src_acc)

    # --- Super-Vector ---
    print("[4] Hybrid Super-Vector")
    featurizer, classifier = build_super_vector_model(cfg)
    X_tr_sv = featurizer.fit_transform(X_train_raw, X_train_cleaned)
    classifier.fit(X_tr_sv, y_train)
    src_acc = evaluate_model(y_train, classifier.predict(X_tr_sv))["accuracy"]
    X_te_sv = featurizer.transform(X_test_raw, X_test_cleaned)
    _record("Hybrid Super-Vector", classifier.predict(X_te_sv), src_acc)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dcfg = cfg["data"]

    print("Loading datasets...")
    df_pol = load_dataset(
        dcfg["political_csv"], dcfg["political_text_col"],
        dcfg["political_label_col"], dcfg["political_real_val"], dcfg["political_fake_val"],
    )
    df_cov = load_dataset(
        dcfg["covid_csv"], dcfg["covid_text_col"],
        dcfg["covid_label_col"], dcfg["covid_real_val"], dcfg["covid_fake_val"],
    )

    # Exp A: Political → COVID-19
    results_a = run_zero_shot_experiment(
        df_pol, df_cov, "Exp A: Political → COVID-19", cfg
    )

    # Exp B: COVID-19 → Political
    results_b = run_zero_shot_experiment(
        df_cov, df_pol, "Exp B: COVID-19 → Political", cfg
    )

    df_a = build_results_table(results_a)
    df_b = build_results_table(results_b)

    print_results_table(df_a, "Exp A Results — Political → COVID-19 (Table 8, left)")
    print_results_table(df_b, "Exp B Results — COVID-19 → Political (Table 8, right)")

    # --- Figure 5: Generalisation Gap ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, df_r, title in [
        (axes[0], df_a.reset_index(), "Exp A: Political → COVID-19"),
        (axes[1], df_b.reset_index(), "Exp B: COVID-19 → Political"),
    ]:
        df_r_sorted = df_r.sort_values("target_accuracy")
        ax.barh(df_r_sorted["model"], df_r_sorted["source_accuracy"],
                alpha=0.4, label="Source Acc", color="steelblue")
        ax.barh(df_r_sorted["model"], df_r_sorted["target_accuracy"],
                alpha=0.9, label="Target Acc (Zero-Shot)", color="coral")
        ax.set_xlabel("Accuracy")
        ax.set_title(title)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, label="Random Chance")
        ax.legend(fontsize=9)

    plt.suptitle("Figure 5: Generalisation Gap — Bidirectional Zero-Shot Transfer", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "figure5_generalisation_gap.png")
    plt.savefig(out, dpi=cfg["output"]["plot_dpi"])
    print(f"\nFigure saved → {out}")
