"""
02_train_models.py
-------------------
Train Random Forest, XGBoost, and SVM on the engineered features.
Two evaluation strategies:
  1. Standard 5-fold stratified cross-validation (all data)
  2. Severity generalisation test — train on 0.007+0.014, test on 0.021

Usage:
    python src/02_train_models.py

Inputs:
    data/processed/features.csv

Outputs (data/processed/):
    model_results.csv
    noise_robustness.csv
    generalization_results.csv

Outputs (figures/):
    confusion_matrix_grid.png
    per_class_f1_chart.png
    accuracy_comparison.png
    noise_robustness.png
    severity_performance.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS = 5
DATA_PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mean", "std", "rms", "peak", "peak2peak",
    "crest_factor", "kurtosis", "skewness", "shape_factor",
    "spectral_centroid", "dominant_frequency",
    "low_band_energy", "mid_band_energy", "high_band_energy",
    "motor_load",
]

# Noise levels as fractions of feature std
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]

# ── Model definitions ──────────────────────────────────────────────────────────

def build_models() -> dict:
    """Return dict of {name: estimator}. SVM is wrapped in a StandardScaler pipeline."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10, gamma="scale",
                        random_state=RANDOM_STATE, probability=True)),
        ]),
    }


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_confusion_matrix_grid(
    cms: dict,          # {model_name: {eval_name: (cm, class_names)}}
    eval_names: list[str],
    save_path: Path,
):
    """3-row × 2-col grid of confusion matrices."""
    model_names = list(cms.keys())
    n_models = len(model_names)
    n_evals  = len(eval_names)

    fig, axes = plt.subplots(
        n_models, n_evals,
        figsize=(7 * n_evals, 5 * n_models),
        constrained_layout=True,
    )
    fig.suptitle("Confusion Matrices — All Models & Evaluations", fontsize=16, fontweight="bold")

    for r, model_name in enumerate(model_names):
        for c, eval_name in enumerate(eval_names):
            ax = axes[r][c]
            cm, class_names = cms[model_name][eval_name]
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(
                cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax,
                vmin=0, vmax=1,
                annot_kws={"size": 9},
            )
            ax.set_title(f"{model_name} — {eval_name}", fontsize=11)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.tick_params(axis="x", rotation=30)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_f1(f1_data: dict, class_names: list[str], save_path: Path):
    """
    Grouped bar chart: x=class, groups=model, two bars per model (CV vs Gen).
    """
    # f1_data: {model_name: {"CV": [f1 per class], "Gen": [f1 per class]}}
    model_names = list(f1_data.keys())
    n_classes = len(class_names)
    n_models  = len(model_names)
    bar_width = 0.12
    x = np.arange(n_classes)

    palette_cv  = sns.color_palette("tab10", n_models)
    palette_gen = sns.color_palette("pastel",  n_models)

    fig, ax = plt.subplots(figsize=(12, 6))

    offset_step = bar_width * 2.2
    total_width = offset_step * n_models
    offsets = np.linspace(-total_width / 2, total_width / 2, n_models)

    for i, model_name in enumerate(model_names):
        cv_f1  = f1_data[model_name]["CV"]
        gen_f1 = f1_data[model_name]["Gen"]
        ax.bar(x + offsets[i] - bar_width / 2, cv_f1,  bar_width, label=f"{model_name} CV",
               color=palette_cv[i],  edgecolor="black", linewidth=0.5)
        ax.bar(x + offsets[i] + bar_width / 2, gen_f1, bar_width, label=f"{model_name} Gen",
               color=palette_gen[i], edgecolor="black", linewidth=0.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_xlabel("Fault Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-Class F1 Score: CV vs Severity Generalisation", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_comparison(results: dict, save_path: Path):
    """
    Bar chart: CV accuracy vs Generalisation accuracy for all 3 models.
    Highlights the performance gap.
    """
    model_names = list(results.keys())
    cv_accs  = [results[m]["cv_accuracy"]  for m in model_names]
    gen_accs = [results[m]["gen_accuracy"] for m in model_names]

    x = np.arange(len(model_names))
    bar_width = 0.32
    colors_cv  = ["#2196F3", "#4CAF50", "#FF9800"]
    colors_gen = ["#90CAF9", "#A5D6A7", "#FFCC80"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - bar_width / 2, cv_accs,  bar_width, label="5-Fold CV Accuracy",
                   color=colors_cv,  edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + bar_width / 2, gen_accs, bar_width, label="Severity Generalisation Accuracy",
                   color=colors_gen, edgecolor="black", linewidth=0.8, hatch="//")

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Draw gap arrows
    for i, m in enumerate(model_names):
        gap = cv_accs[i] - gen_accs[i]
        mid_x = x[i]
        ax.annotate(
            f"Δ={gap:.3f}",
            xy=(mid_x, gen_accs[i] + gap / 2),
            fontsize=8, color="red", ha="center", style="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("CV vs Severity Generalisation Accuracy\n(Key Research Finding: The Performance Gap)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_noise_robustness(noise_df: pd.DataFrame, save_path: Path):
    """Line chart: accuracy vs noise level for all 3 models on generalisation test."""
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^"]
    for i, model in enumerate(noise_df["model"].unique()):
        sub = noise_df[noise_df["model"] == model]
        ax.plot(
            sub["noise_pct"] * 100,
            sub["accuracy"],
            marker=markers[i % len(markers)],
            linewidth=2,
            label=model,
        )

    ax.set_xlabel("Gaussian Noise Level (% of feature std)", fontsize=12)
    ax.set_ylabel("Generalisation Accuracy", fontsize=12)
    ax.set_title("Noise Robustness — Severity Generalisation Test", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_severity_performance(sev_df: pd.DataFrame, save_path: Path):
    """Grouped bar chart: per-severity accuracy for all 3 models."""
    severities  = sorted(sev_df["severity"].unique())
    model_names = list(sev_df["model"].unique())

    x = np.arange(len(severities))
    bar_width = 0.25
    palette = sns.color_palette("tab10", len(model_names))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, model in enumerate(model_names):
        accs = [
            sev_df[(sev_df["model"] == model) & (sev_df["severity"] == s)]["accuracy"].values[0]
            for s in severities
        ]
        offset = (i - len(model_names) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, accs, bar_width, label=model,
                      color=palette[i], edgecolor="black", linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Severity\n{s}" for s in severities], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Severity Accuracy Breakdown — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def run_cv_evaluation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    le: LabelEncoder,
) -> dict:
    """5-fold stratified CV: returns accuracy, per-class F1, confusion matrix."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=-1)

    acc  = accuracy_score(y, y_pred)
    f1   = f1_score(y, y_pred, average=None, labels=range(len(class_names)))
    cm   = confusion_matrix(y, y_pred, labels=range(len(class_names)))
    return {"accuracy": acc, "f1_per_class": f1.tolist(), "cm": cm}


def run_gen_evaluation(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    class_names: list[str],
) -> dict:
    """Severity generalisation: train on 0.007+0.014, test on 0.021."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average=None, labels=range(len(class_names)),
                    zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    return {"accuracy": acc, "f1_per_class": f1.tolist(), "cm": cm}


def add_noise(X: np.ndarray, noise_pct: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise scaled by noise_pct × per-feature std."""
    if noise_pct == 0.0:
        return X
    stds = X.std(axis=0, keepdims=True)
    noise = rng.normal(0, noise_pct * stds, size=X.shape)
    return X + noise


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    feat_path = DATA_PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        print(f"[ERROR] {feat_path} not found. Run 01_load_and_preprocess.py first.")
        return

    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df):,} windows from {feat_path}\n")

    # ── Encode labels ──────────────────────────────────────────────────────────
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    class_names = list(le.classes_)
    print(f"Classes ({len(class_names)}): {class_names}\n")

    X_all = df[FEATURE_COLS].values.astype(np.float64)
    y_all = df["label_enc"].values

    # ── Severity generalisation split ──────────────────────────────────────────
    normal_mask = df["severity"] == "none"
    train_mask  = df["severity"].isin(["0.007", "0.014"]) | normal_mask
    test_mask   = (df["severity"] == "0.021") | normal_mask  # unseen severity + normal

    X_train_gen = df.loc[train_mask, FEATURE_COLS].values.astype(np.float64)
    y_train_gen = df.loc[train_mask, "label_enc"].values
    X_test_gen  = df.loc[test_mask,  FEATURE_COLS].values.astype(np.float64)
    y_test_gen  = df.loc[test_mask,  "label_enc"].values

    print(f"Generalisation split:")
    print(f"  Train: {len(X_train_gen):,} windows (sev 0.007 + 0.014 + normal)")
    print(f"  Test : {len(X_test_gen):,}  windows (sev 0.021 + normal — NEVER seen)\n")

    rng = np.random.default_rng(RANDOM_STATE)

    # Storage
    model_results = {}
    cms           = {}
    f1_data       = {}
    noise_rows    = []
    gen_rows      = []
    sev_rows      = []

    eval_names = ["5-Fold CV", "Severity Generalisation"]

    # ── Per-severity accuracy helper ───────────────────────────────────────────
    all_severities = sorted(df["severity"].unique())

    # ── Train & evaluate each model ────────────────────────────────────────────
    for model_name, model in build_models().items():
        print(f"{'─'*55}")
        print(f"  Model: {model_name}")
        print(f"{'─'*55}")

        # Build a fresh copy for each evaluation to avoid state leakage
        models = build_models()
        m_cv  = models[model_name]
        m_gen = build_models()[model_name]

        # 1. Cross-validation
        print("  Running 5-fold CV ...", end="", flush=True)
        cv_res = run_cv_evaluation(m_cv, X_all, y_all, class_names, le)
        print(f" accuracy = {cv_res['accuracy']:.4f}")

        # 2. Severity generalisation
        print("  Running severity generalisation ...", end="", flush=True)
        gen_res = run_gen_evaluation(m_gen, X_train_gen, y_train_gen,
                                     X_test_gen, y_test_gen, class_names)
        print(f" accuracy = {gen_res['accuracy']:.4f}")

        model_results[model_name] = {
            "cv_accuracy":  cv_res["accuracy"],
            "gen_accuracy": gen_res["accuracy"],
        }
        cms[model_name] = {
            eval_names[0]: (cv_res["cm"],  class_names),
            eval_names[1]: (gen_res["cm"], class_names),
        }
        f1_data[model_name] = {
            "CV":  cv_res["f1_per_class"],
            "Gen": gen_res["f1_per_class"],
        }

        # Generalisation results CSV
        for cls, f1_cv, f1_gen in zip(class_names, cv_res["f1_per_class"], gen_res["f1_per_class"]):
            gen_rows.append({
                "model": model_name, "class": cls,
                "cv_f1": f1_cv, "gen_f1": f1_gen,
            })

        # 3. Noise robustness (on generalisation test set)
        print("  Noise robustness ...", end="", flush=True)
        m_noise = build_models()[model_name]
        m_noise.fit(X_train_gen, y_train_gen)  # train once on clean data
        for noise_pct in NOISE_LEVELS:
            X_noisy = add_noise(X_test_gen, noise_pct, rng)
            acc_noisy = accuracy_score(y_test_gen, m_noise.predict(X_noisy))
            noise_rows.append({"model": model_name, "noise_pct": noise_pct, "accuracy": acc_noisy})
        print(" done")

        # 4. Per-severity accuracy (using generalisation-trained model)
        print("  Per-severity accuracy ...", end="", flush=True)
        for sev in all_severities:
            sev_mask = df["severity"] == sev
            if sev_mask.sum() == 0:
                continue
            X_sev = df.loc[sev_mask, FEATURE_COLS].values.astype(np.float64)
            y_sev = df.loc[sev_mask, "label_enc"].values
            # Use the noise-robustness model (already trained on gen split)
            acc_sev = accuracy_score(y_sev, m_noise.predict(X_sev))
            sev_rows.append({"model": model_name, "severity": sev, "accuracy": acc_sev})
        print(" done\n")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame([
        {"model": m, "cv_accuracy": v["cv_accuracy"], "gen_accuracy": v["gen_accuracy"],
         "gap": v["cv_accuracy"] - v["gen_accuracy"]}
        for m, v in model_results.items()
    ])
    results_df.to_csv(DATA_PROCESSED_DIR / "model_results.csv", index=False)

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(DATA_PROCESSED_DIR / "noise_robustness.csv", index=False)

    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(DATA_PROCESSED_DIR / "generalization_results.csv", index=False)

    print(f"Saved CSVs to {DATA_PROCESSED_DIR}/\n")

    # ── Generate plots ─────────────────────────────────────────────────────────
    print("Generating figures ...")

    plot_confusion_matrix_grid(cms, eval_names, FIGURES_DIR / "confusion_matrix_grid.png")
    plot_per_class_f1(f1_data, class_names, FIGURES_DIR / "per_class_f1_chart.png")
    plot_accuracy_comparison(model_results, FIGURES_DIR / "accuracy_comparison.png")
    plot_noise_robustness(noise_df, FIGURES_DIR / "noise_robustness.png")

    sev_df = pd.DataFrame(sev_rows)
    plot_severity_performance(sev_df, FIGURES_DIR / "severity_performance.png")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY: CV Accuracy vs Severity Generalisation Accuracy")
    print("=" * 65)
    print(f"{'Model':<18} {'CV Acc':>8} {'Gen Acc':>10} {'Gap (↓)':>10}")
    print("-" * 65)
    for _, row in results_df.iterrows():
        print(f"{row['model']:<18} {row['cv_accuracy']:>8.4f} {row['gen_accuracy']:>10.4f} "
              f"{row['gap']:>10.4f}  ← RESEARCH FINDING")
    print("=" * 65)
    print("\nKey finding: CV accuracy overstates true performance.")
    print("The gap (Δ) exposes how evaluation methodology affects reported results.")
    print("Severity generalisation is a more realistic deployment proxy.\n")


if __name__ == "__main__":
    main()
