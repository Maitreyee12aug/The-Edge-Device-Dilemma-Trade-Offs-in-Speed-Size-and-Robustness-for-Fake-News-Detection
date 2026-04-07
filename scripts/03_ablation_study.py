"""
scripts/03_ablation_study.py

Ablation Study (paper §4.7, Table 9).

Evaluates three feature configurations of the Super-Vector model under
strict zero-shot cross-domain transfer to isolate the contribution of
lexical vs. stylistic features:

  Config 1 — Lexical Only  : HashingVectorizer (v_lex)
  Config 2 — Stylistic Only: 14-dim stylometric feature vector (v_style)
  Config 3 — Super-Vector  : v_final = v_lex ⊕ v_style  [full model]

All configurations use the same LinearSVC classifier.
Reproduces Table 9 from the paper.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.feature_engineering import (
    SuperVectorFeaturizer,
    get_stylistic_features,
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
# Feature builders
# ---------------------------------------------------------------------------

def build_lexical_features(X_cleaned_train, X_cleaned_test, cfg):
    """HashingVectorizer features (v_lex only)."""
    hcfg = cfg["models"]["super_vector"]
    vec = HashingVectorizer(
        n_features=hcfg["n_hash_features"],
        ngram_range=(1, 2),
    )
    return vec.transform(X_cleaned_train), vec.transform(X_cleaned_test)


def build_stylistic_features(X_raw_train, X_raw_test):
    """Scaled 13-dim stylistic features (v_style only)."""
    X_tr = np.array([get_stylistic_features(t) for t in X_raw_train])
    X_te = np.array([get_stylistic_features(t) for t in X_raw_test])
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te)


def build_super_vector_features(X_raw_train, X_cleaned_train,
                                 X_raw_test, X_cleaned_test, cfg):
    """Full Super-Vector (v_lex ⊕ v_style)."""
    hcfg = cfg["models"]["super_vector"]
    fz = SuperVectorFeaturizer(
        n_hash_features=hcfg["n_hash_features"],
        n_tfidf_features=hcfg["n_tfidf_features"],
    )
    X_tr = fz.fit_transform(X_raw_train, X_cleaned_train)
    X_te = fz.transform(X_raw_test, X_cleaned_test)
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Run one direction of the ablation
# ---------------------------------------------------------------------------

def run_ablation_direction(df_train, df_test, direction: str, cfg: dict) -> list:
    print(f"\n  Direction: {direction}")
    svcfg = cfg["models"]["super_vector"]

    X_tr_raw = df_train["tweet"]
    X_tr_cln = df_train["cleaned_tweet"]
    y_tr = df_train["label_num"]

    X_te_raw = df_test["tweet"]
    X_te_cln = df_test["cleaned_tweet"]
    y_te = df_test["label_num"]

    svc_kwargs = dict(
        C=svcfg["svc_C"], random_state=42,
        dual=svcfg["svc_dual"], max_iter=svcfg["svc_max_iter"],
    )

    results = []

    # Config 1: Lexical only
    print("    [1/3] Lexical only (HashingVectorizer)...")
    X_tr, X_te = build_lexical_features(X_tr_cln, X_te_cln, cfg)
    clf = LinearSVC(**svc_kwargs)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    results.append({"config": "Lexical only", "direction": direction, "accuracy": acc})
    print(f"      Accuracy: {acc:.4f}")

    # Config 2: Stylistic only
    print("    [2/3] Stylistic only...")
    X_tr_s, X_te_s = build_stylistic_features(X_tr_raw, X_te_raw)
    clf = LinearSVC(**svc_kwargs)
    clf.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te_s))
    results.append({"config": "Stylistic only", "direction": direction, "accuracy": acc})
    print(f"      Accuracy: {acc:.4f}")

    # Config 3: Super-Vector (full)
    print("    [3/3] Super-Vector (Lexical + Stylistic)...")
    X_tr_sv, X_te_sv = build_super_vector_features(
        X_tr_raw, X_tr_cln, X_te_raw, X_te_cln, cfg
    )
    clf = LinearSVC(**svc_kwargs)
    clf.fit(X_tr_sv, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te_sv))
    results.append({"config": "Super-Vector", "direction": direction, "accuracy": acc})
    print(f"      Accuracy: {acc:.4f}")

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

    print("\nRunning ablation study (Table 9)...")
    results_a = run_ablation_direction(df_pol, df_cov, "Pol → COV", cfg)
    results_b = run_ablation_direction(df_cov, df_pol, "COV → Pol", cfg)

    all_results = results_a + results_b
    df_abl = pd.DataFrame(all_results)
    df_pivot = df_abl.pivot(index="config", columns="direction", values="accuracy")
    df_pivot["Average"] = df_pivot.mean(axis=1)

    print("\n" + "=" * 55)
    print("  ABLATION STUDY RESULTS (Table 9)")
    print("=" * 55)
    try:
        print(df_pivot.to_markdown(floatfmt=".4f"))
    except ImportError:
        print(df_pivot.to_string(float_format="{:.4f}".format))

    # --- Plot ---
    df_plot = df_abl.copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_plot, x="config", y="accuracy", hue="direction",
                palette=["steelblue", "coral"])
    plt.title("Ablation Study: Feature Configuration vs. Zero-Shot Accuracy (Table 9)")
    plt.xlabel("Feature Configuration")
    plt.ylabel("Zero-Shot Accuracy")
    plt.ylim(0.45, 0.80)
    plt.legend(title="Transfer Direction")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "table9_ablation_study.png")
    plt.savefig(out, dpi=cfg["output"]["plot_dpi"])
    print(f"\nFigure saved → {out}")
