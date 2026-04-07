"""
scripts/01_phase1_efficiency.py

Phase I: In-Domain Efficiency Benchmarking (paper §3.4.1, §4.2).

Trains and benchmarks Hashing-SVC, TF-IDF-SVC, fastText, and the Hybrid
Super-Vector on each dataset independently, measuring:
  - In-domain accuracy and F1
  - Serialised model size (MB)
  - Single-sample inference latency (ms)

Reproduces Tables 6 and 7 and Figure 4 from the paper.
"""

import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import (
    build_results_table,
    evaluate_model,
    get_model_size_mb,
    print_results_table,
)
from src.models.lightweight_models import (
    build_hashing_svc,
    build_super_vector_model,
    build_tfidf_svc,
    get_fasttext_size_mb,
    predict_fasttext,
    train_fasttext,
)
from src.preprocessing.text_cleaner import preprocess_series

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

OUT_DIR = cfg["output"]["figures_dir"]
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: run one dataset benchmark
# ---------------------------------------------------------------------------

def benchmark_on_dataset(df, dataset_name: str, cfg: dict) -> list:
    """Train all paradigms on df and return a list of result dicts."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking on: {dataset_name}")
    print(f"{'='*60}")

    from sklearn.model_selection import train_test_split
    split_cfg = cfg["split"]

    df_train, df_test = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_seed"],
        stratify=df["label_num"],
    )

    X_train_raw     = df_train["tweet"]
    X_train_cleaned = df_train["cleaned_tweet"]
    y_train         = df_train["label_num"]

    X_test_raw     = df_test["tweet"]
    X_test_cleaned = df_test["cleaned_tweet"]
    y_test         = df_test["label_num"]

    results = []

    # --- P1a: Hashing-SVC ---
    print("\n[P1a] Hashing-SVC")
    import time
    model_hash = build_hashing_svc(cfg)
    model_hash.fit(X_train_cleaned, y_train)
    y_pred = model_hash.predict(X_test_cleaned)

    t0 = time.perf_counter()
    for _ in range(1000): model_hash.predict([X_test_cleaned.iloc[0]])
    lat = (time.perf_counter() - t0) / 1000 * 1000  # ms

    metrics = evaluate_model(y_test, y_pred)
    results.append({
        "model":      "Hashing-SVC",
        "accuracy":   metrics["accuracy"],
        "f1":         metrics["f1"],
        "latency_ms": lat,
        "size_mb":    get_model_size_mb(model_hash, "sklearn"),
    })
    print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
          f"Lat={lat:.3f} ms  Size={results[-1]['size_mb']:.2f} MB")

    # --- P1b: TF-IDF-SVC ---
    print("\n[P1b] TF-IDF-SVC")
    model_tfidf = build_tfidf_svc(cfg)
    model_tfidf.fit(X_train_cleaned, y_train)
    y_pred = model_tfidf.predict(X_test_cleaned)

    t0 = time.perf_counter()
    for _ in range(1000): model_tfidf.predict([X_test_cleaned.iloc[0]])
    lat = (time.perf_counter() - t0) / 1000 * 1000

    metrics = evaluate_model(y_test, y_pred)
    results.append({
        "model":      "TF-IDF-SVC",
        "accuracy":   metrics["accuracy"],
        "f1":         metrics["f1"],
        "latency_ms": lat,
        "size_mb":    get_model_size_mb(model_tfidf, "sklearn"),
    })
    print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
          f"Lat={lat:.3f} ms  Size={results[-1]['size_mb']:.2f} MB")

    # --- P2: fastText ---
    print("\n[P2] fastText")
    model_ft = train_fasttext(X_train_cleaned, y_train, cfg)
    y_pred = predict_fasttext(model_ft, X_test_cleaned)

    t0 = time.perf_counter()
    for _ in range(1000): predict_fasttext(model_ft, [X_test_cleaned.iloc[0]])
    lat = (time.perf_counter() - t0) / 1000 * 1000

    metrics = evaluate_model(y_test, y_pred)
    results.append({
        "model":      "fastText",
        "accuracy":   metrics["accuracy"],
        "f1":         metrics["f1"],
        "latency_ms": lat,
        "size_mb":    get_fasttext_size_mb(model_ft),
    })
    print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
          f"Lat={lat:.3f} ms  Size={results[-1]['size_mb']:.2f} MB")

    # --- P3: Hybrid Super-Vector ---
    print("\n[P3] Hybrid Super-Vector")
    featurizer, classifier = build_super_vector_model(cfg)
    X_train_sv = featurizer.fit_transform(X_train_raw, X_train_cleaned)
    classifier.fit(X_train_sv, y_train)

    X_test_sv = featurizer.transform(X_test_raw, X_test_cleaned)
    y_pred = classifier.predict(X_test_sv)

    t0 = time.perf_counter()
    for _ in range(200):
        xr = X_test_raw.iloc[0:1]
        xc = X_test_cleaned.iloc[0:1]
        classifier.predict(featurizer.transform(xr, xc))
    lat = (time.perf_counter() - t0) / 200 * 1000

    metrics = evaluate_model(y_test, y_pred)
    import joblib, tempfile
    sv_size = get_model_size_mb(classifier, "sklearn")

    results.append({
        "model":      "Hybrid Super-Vector",
        "accuracy":   metrics["accuracy"],
        "f1":         metrics["f1"],
        "latency_ms": lat,
        "size_mb":    sv_size,
    })
    print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
          f"Lat={lat:.3f} ms  Size={results[-1]['size_mb']:.2f} MB")

    return results


# ---------------------------------------------------------------------------
# Load and preprocess datasets
# ---------------------------------------------------------------------------

def load_dataset(path, text_col, label_col, real_val, fake_val):
    import pandas as pd
    df = pd.read_csv(path).dropna()
    df = df.rename(columns={text_col: "tweet", label_col: "label"})
    df["label_num"] = df["label"].map({real_val: 0, fake_val: 1})
    df = df.dropna(subset=["label_num", "tweet"])
    df["label_num"] = df["label_num"].astype(int)
    df["cleaned_tweet"] = preprocess_series(df["tweet"])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dcfg = cfg["data"]

    print("Loading Political corpus...")
    df_pol = load_dataset(
        dcfg["political_csv"], dcfg["political_text_col"],
        dcfg["political_label_col"], dcfg["political_real_val"], dcfg["political_fake_val"],
    )

    print("Loading COVID-19 corpus...")
    df_cov = load_dataset(
        dcfg["covid_csv"], dcfg["covid_text_col"],
        dcfg["covid_label_col"], dcfg["covid_real_val"], dcfg["covid_fake_val"],
    )

    results_pol = benchmark_on_dataset(df_pol, "Political Corpus (ISOT)", cfg)
    results_cov = benchmark_on_dataset(df_cov, "COVID-19 Corpus (CONSTRAINT)", cfg)

    df_pol_res = build_results_table(results_pol)
    df_cov_res = build_results_table(results_cov)

    print_results_table(df_pol_res, "Phase I-A: In-Domain Efficiency — Political Corpus (Table 6)")
    print_results_table(df_cov_res, "Phase I-B: In-Domain Efficiency — COVID-19 Corpus (Table 7)")

    # --- Figure 4: Accuracy vs Latency vs Size ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df_r, title in [
        (axes[0], df_pol_res.reset_index(), "Political"),
        (axes[1], df_cov_res.reset_index(), "COVID-19"),
    ]:
        sns.scatterplot(data=df_r, x="latency_ms", y="accuracy",
                        size="size_mb", hue="model", ax=ax,
                        sizes=(100, 800), palette="viridis", alpha=0.85)
        ax.set_title(f"Efficiency Frontier — {title}")
        ax.set_xlabel("Latency (ms/doc)")
        ax.set_ylabel("Accuracy")
        ax.set_xscale("log")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "figure4_efficiency_frontier.png")
    plt.savefig(out, dpi=cfg["output"]["plot_dpi"])
    print(f"\nFigure saved → {out}")
