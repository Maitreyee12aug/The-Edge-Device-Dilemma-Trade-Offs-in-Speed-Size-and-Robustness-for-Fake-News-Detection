"""
scripts/04_dynamic_cascade.py

Dynamic Inference Cascading simulation (paper §6.1, §6.2, Figure 8).

Simulates a 1,000-sample data stream that undergoes a sudden domain shift
at sample 500 (Phase A: Political in-domain → Phase B: COVID-19 drift).

Demonstrates that the two-tier cascade system:
  - Achieves ≈7,142 QPS (Tier-1 only) under stable conditions
  - Automatically escalates to Tier-2 (Super-Vector) when lexical
    confidence drops below threshold τ
  - Recovers accuracy to ≈93%+ after drift, vs. ≈51% for static Hashing-SVC

Reproduces Figure 8 and Table 10 from the paper.
"""

import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dynamic_cascade import DynamicInferenceCascade, run_stream_simulation
from src.models.lightweight_models import build_hashing_svc, build_super_vector_model
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
# Throughput helpers
# ---------------------------------------------------------------------------

def measure_throughput(predict_fn, samples, n=1000) -> float:
    """Return queries per second for a batch predict function."""
    t0 = time.perf_counter()
    for _ in range(n):
        predict_fn([samples[_ % len(samples)]])
    return n / (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dcfg  = cfg["data"]
    ccfg  = cfg["cascade"]
    tau   = ccfg["confidence_threshold"]

    print("Loading datasets...")
    df_pol = load_dataset(
        dcfg["political_csv"], dcfg["political_text_col"],
        dcfg["political_label_col"], dcfg["political_real_val"], dcfg["political_fake_val"],
    )
    df_cov = load_dataset(
        dcfg["covid_csv"], dcfg["covid_text_col"],
        dcfg["covid_label_col"], dcfg["covid_real_val"], dcfg["covid_fake_val"],
    )

    # --- Phase A: 500 in-domain Political samples ---
    phase_a = df_pol.sample(n=500, random_state=42)
    # --- Phase B: 500 drift COVID-19 samples ---
    phase_b = df_cov.sample(n=min(500, len(df_cov)), random_state=42)

    # ---------------------------------------------------------------------------
    # Train Tier-1 (Hashing-SVC) on Political training split
    # ---------------------------------------------------------------------------
    print("\n[1/3] Training Tier-1 Hashing-SVC on Political training data...")
    df_pol_train = df_pol.drop(phase_a.index)
    tier1 = build_hashing_svc(cfg)
    tier1.fit(df_pol_train["cleaned_tweet"], df_pol_train["label_num"])
    print(f"  Tier-1 trained on {len(df_pol_train)} samples.")

    # ---------------------------------------------------------------------------
    # Train Tier-2 (Hybrid Super-Vector) on Political training split
    # ---------------------------------------------------------------------------
    print("\n[2/3] Training Tier-2 Hybrid Super-Vector...")
    featurizer, tier2_clf = build_super_vector_model(cfg)
    X_tr_sv = featurizer.fit_transform(
        df_pol_train["tweet"], df_pol_train["cleaned_tweet"]
    )
    tier2_clf.fit(X_tr_sv, df_pol_train["label_num"])
    print(f"  Tier-2 trained.")

    # ---------------------------------------------------------------------------
    # Build cascade and run simulation
    # ---------------------------------------------------------------------------
    print(f"\n[3/3] Running stream simulation (τ = {tau})...")
    cascade = DynamicInferenceCascade(
        tier1_pipeline=tier1,
        tier2_featurizer=featurizer,
        tier2_classifier=tier2_clf,
        confidence_threshold=tau,
    )

    sim = run_stream_simulation(
        cascade=cascade,
        X_phase_a_cleaned=phase_a["cleaned_tweet"],
        y_phase_a=phase_a["label_num"],
        X_phase_b_cleaned=phase_b["cleaned_tweet"],
        y_phase_b=phase_b["label_num"],
        X_phase_a_raw=phase_a["tweet"],
        X_phase_b_raw=phase_b["tweet"],
        window=50,
    )

    # ---------------------------------------------------------------------------
    # Static Hashing-SVC baseline (no escalation)
    # ---------------------------------------------------------------------------
    X_all_cleaned = list(phase_a["cleaned_tweet"]) + list(phase_b["cleaned_tweet"])
    y_all         = list(phase_a["label_num"])     + list(phase_b["label_num"])
    y_static      = tier1.predict(X_all_cleaned)

    static_rolling = []
    buf = []
    for pred, true in zip(y_static, y_all):
        buf.append(int(pred == true))
        if len(buf) > 50: buf.pop(0)
        static_rolling.append(sum(buf) / len(buf))

    # ---------------------------------------------------------------------------
    # Throughput table (Table 10)
    # ---------------------------------------------------------------------------
    print("\nMeasuring throughput...")
    sample_pool = list(phase_a["cleaned_tweet"])
    tier1_qps   = measure_throughput(tier1.predict, sample_pool)

    tier2_qps_list = []
    for txt in sample_pool[:200]:
        t0 = time.perf_counter()
        featurizer.transform([txt], [txt])
        tier2_qps_list.append(1.0 / (time.perf_counter() - t0))
    tier2_qps = float(np.mean(tier2_qps_list))

    print("\n" + "=" * 50)
    print("  Table 10: Cascaded System Performance")
    print("=" * 50)
    print(f"  Tier-1 (Hashing-SVC) throughput : {tier1_qps:>8,.0f} QPS")
    print(f"  Tier-2 (Super-Vector) throughput: {tier2_qps:>8,.0f} QPS")
    print(f"  Tier-1 routing ratio             : {cascade.tier1_ratio:.1%}")
    print(f"  Mean system latency              : {sim['mean_latency_ms']:.3f} ms")
    print("=" * 50)

    # ---------------------------------------------------------------------------
    # Figure 8: Dynamic Adaptation Trajectory
    # ---------------------------------------------------------------------------
    n_a = sim["n_phase_a"]
    n_b = sim["n_phase_b"]
    n_total = n_a + n_b

    fig, ax1 = plt.subplots(figsize=(14, 6))

    x = range(n_total)
    ax1.plot(x, sim["rolling_accuracy"], color="steelblue", linewidth=2,
             label="Cascade (Tier-1 + Tier-2)")
    ax1.plot(x, static_rolling, color="firebrick", linewidth=1.5,
             linestyle="--", label="Static Hashing-SVC (Baseline)")

    ax1.axvline(n_a, color="black", linestyle=":", linewidth=1.2)
    ax1.text(n_a + 5, 0.45, "Domain Shift\n(COVID-19)", fontsize=10, color="black")

    ax1.fill_betweenx([0, 1], 0, n_a, alpha=0.05, color="green", label="In-Domain (Political)")
    ax1.fill_betweenx([0, 1], n_a, n_total, alpha=0.05, color="red",  label="Drift (COVID-19)")

    ax1.set_xlabel("Sample Stream Index", fontsize=12)
    ax1.set_ylabel("Rolling Accuracy (window=50)", fontsize=12)
    ax1.set_ylim(0.35, 1.05)
    ax1.set_title("Figure 8: Dynamic Inference Cascading — Adaptation Trajectory", fontsize=14)
    ax1.legend(fontsize=10, loc="lower left")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "figure8_dynamic_cascade.png")
    plt.savefig(out, dpi=cfg["output"]["plot_dpi"])
    print(f"\nFigure 8 saved → {out}")
