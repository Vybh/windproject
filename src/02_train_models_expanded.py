"""
02_train_models_expanded.py
----------------------------
Train Random Forest, XGBoost, SVM (RBF), and KNN on features_expanded.csv.
Evaluation: Stratified 5-Fold CV across all 7 classes.

Usage:
    python src/02_train_models_expanded.py

Inputs:
    data/processed/features_expanded.csv

Outputs (data/processed/):
    model_results_expanded.csv
    noise_robustness_expanded.csv

Outputs (figures/):
    confusion_matrix_grid_expanded.png
    per_class_f1_expanded.png
    accuracy_comparison_expanded.png
    noise_robustness_expanded.png
    bearing_location_accuracy_expanded.png
    fault_size_f1_expanded.png
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS      = 5
DATA_PROCESSED_DIR = Path("data/processed")
FIGURES_DIR        = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mean", "std", "rms", "peak", "peak2peak",
    "crest_factor", "kurtosis", "skewness", "shape_factor",
    "spectral_centroid", "dominant_frequency",
    "low_band_energy", "mid_band_energy", "high_band_energy",
    "motor_load",
]

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]


# ── Model definitions ──────────────────────────────────────────────────────────

def build_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, eval_metric="mlogloss",
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def add_noise(X: np.ndarray, noise_pct: float, rng: np.random.Generator) -> np.ndarray:
    if noise_pct == 0.0:
        return X
    stds  = X.std(axis=0, keepdims=True)
    noise = rng.normal(0, noise_pct * stds, size=X.shape)
    return X + noise


def run_cv(model, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    """5-fold stratified CV; returns accuracy, per-class F1 list, confusion matrix."""
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=-1)
    acc    = accuracy_score(y, y_pred)
    f1     = f1_score(y, y_pred, average=None, labels=range(n_classes), zero_division=0)
    cm     = confusion_matrix(y, y_pred, labels=range(n_classes))
    return {"accuracy": acc, "f1_per_class": f1.tolist(), "cm": cm, "y_pred": y_pred}


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_confusion_matrix_grid(cms: dict, class_names: list[str], save_path: Path):
    """2×2 grid of normalised confusion matrices, one per model."""
    model_names = list(cms.keys())          # 4 models
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle("Confusion Matrices — Expanded Dataset (5-Fold CV)", fontsize=15, fontweight="bold")

    for idx, model_name in enumerate(model_names):
        ax  = axes[idx // 2][idx % 2]
        cm  = cms[model_name]
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", ax=ax, vmin=0, vmax=1, annot_kws={"size": 8},
        )
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_f1(f1_data: dict, class_names: list[str], save_path: Path):
    """Grouped bar chart: x=class, one bar per model."""
    model_names = list(f1_data.keys())
    n_classes   = len(class_names)
    n_models    = len(model_names)
    bar_width   = 0.18
    x           = np.arange(n_classes)
    palette     = sns.color_palette("tab10", n_models)

    offsets = np.linspace(
        -(n_models - 1) * bar_width / 2,
         (n_models - 1) * bar_width / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_name in enumerate(model_names):
        bars = ax.bar(
            x + offsets[i], f1_data[model_name], bar_width,
            label=model_name, color=palette[i], edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class F1 Score — All Models (5-Fold CV)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_comparison(results: dict, save_path: Path):
    """Simple bar chart of overall CV accuracy for all 4 models."""
    model_names = list(results.keys())
    accs        = [results[m]["cv_accuracy"] for m in model_names]
    palette     = sns.color_palette("tab10", len(model_names))

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(model_names, accs, color=palette, edgecolor="black", linewidth=0.8, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f"{acc:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Accuracy (5-Fold CV)", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Overall Accuracy Comparison — Expanded Dataset", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_noise_robustness(noise_df: pd.DataFrame, save_path: Path):
    """Line chart: CV accuracy vs noise level for all 4 models."""
    markers = ["o", "s", "^", "D"]
    palette = sns.color_palette("tab10", len(noise_df["model"].unique()))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, model in enumerate(noise_df["model"].unique()):
        sub = noise_df[noise_df["model"] == model].sort_values("noise_pct")
        ax.plot(
            sub["noise_pct"] * 100,
            sub["accuracy"],
            marker=markers[i % len(markers)],
            linewidth=2,
            color=palette[i],
            label=model,
        )

    ax.set_xlabel("Gaussian Noise Level (% of feature std)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Noise Robustness — Expanded Dataset", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bearing_location_accuracy(bl_data: dict, save_path: Path):
    """Side-by-side bar chart: DE vs FE accuracy per model."""
    model_names = list(bl_data.keys())
    de_accs = [bl_data[m]["DE"] for m in model_names]
    fe_accs = [bl_data[m]["FE"] for m in model_names]

    x         = np.arange(len(model_names))
    bar_width = 0.32
    fig, ax   = plt.subplots(figsize=(10, 6))

    bars_de = ax.bar(x - bar_width / 2, de_accs, bar_width, label="DE (Drive-End)",
                     color="#2196F3", edgecolor="black", linewidth=0.8)
    bars_fe = ax.bar(x + bar_width / 2, fe_accs, bar_width, label="FE (Fan-End)",
                     color="#FF9800", edgecolor="black", linewidth=0.8)

    for bar in bars_de + bars_fe:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Accuracy (5-Fold CV)", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Bearing Location Accuracy: DE vs FE — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fault_size_f1(fs_data: dict, fault_sizes: list[str], save_path: Path):
    """Grouped bar chart: F1 per fault size (x-axis) per model (bars)."""
    model_names = list(fs_data.keys())
    n_sizes     = len(fault_sizes)
    n_models    = len(model_names)
    bar_width   = 0.18
    x           = np.arange(n_sizes)
    palette     = sns.color_palette("tab10", n_models)
    offsets     = np.linspace(
        -(n_models - 1) * bar_width / 2,
         (n_models - 1) * bar_width / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(model_names):
        f1_vals = [fs_data[model_name].get(sz, 0.0) for sz in fault_sizes]
        ax.bar(
            x + offsets[i], f1_vals, bar_width,
            label=model_name, color=palette[i], edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"none\n(normal)" if s == "none" else f'"{s}"' for s in fault_sizes],
        fontsize=10,
    )
    ax.set_xlabel("Fault Size (inches)", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("F1 Score by Fault Size — All Models (5-Fold CV)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    feat_path = DATA_PROCESSED_DIR / "features_expanded.csv"
    if not feat_path.exists():
        print(f"[ERROR] {feat_path} not found. Run save_features_expanded.py first.")
        return

    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df):,} windows from {feat_path}\n")

    # ── Encode labels ──────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_all       = le.fit_transform(df["label"].values)
    class_names = list(le.classes_)
    n_classes   = len(class_names)
    X_all       = df[FEATURE_COLS].values.astype(np.float64)
    print(f"Classes ({n_classes}): {class_names}\n")

    # ── 80/20 stratified split for noise robustness ────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_STATE,
    )
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Storage ────────────────────────────────────────────────────────────────
    all_results  = {}
    cms          = {}
    f1_data      = {}          # {model: [f1 per class]}
    noise_rows   = []
    bearing_data = {}          # {model: {DE: acc, FE: acc}}
    fs_f1_data   = {}          # {model: {fault_size: macro_f1}}

    fault_sizes = sorted(df["fault_size"].astype(str).unique(),
                         key=lambda s: (s == "none", s))

    # ── Per-model evaluation ───────────────────────────────────────────────────
    for model_name, model in build_models().items():
        print(f"{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        # 1. Cross-validation (full dataset)
        print("  Running 5-fold CV ...", end="", flush=True)
        cv_res = run_cv(model, X_all, y_all, n_classes)
        print(f" accuracy = {cv_res['accuracy']:.4f}")

        y_pred_cv = cv_res["y_pred"]

        all_results[model_name] = {"cv_accuracy": cv_res["accuracy"]}
        cms[model_name]         = cv_res["cm"]
        f1_data[model_name]     = cv_res["f1_per_class"]

        # 2. Bearing location accuracy from CV predictions
        bearing_data[model_name] = {}
        for loc in ["DE", "FE"]:
            mask = df["bearing_location"].values == loc
            if mask.sum() == 0:
                bearing_data[model_name][loc] = 0.0
                continue
            bearing_data[model_name][loc] = accuracy_score(y_all[mask], y_pred_cv[mask])

        # 3. Fault-size macro F1 from CV predictions
        fs_f1_data[model_name] = {}
        for sz in fault_sizes:
            mask = df["fault_size"].astype(str).values == sz
            if mask.sum() == 0:
                continue
            fs_f1_data[model_name][sz] = f1_score(
                y_all[mask], y_pred_cv[mask],
                average="macro", zero_division=0,
            )

        # 4. Noise robustness (train on 80%, evaluate on noisy 20%)
        print("  Noise robustness ...", end="", flush=True)
        m_noise = build_models()[model_name]
        m_noise.fit(X_tr, y_tr)
        for noise_pct in NOISE_LEVELS:
            X_noisy = add_noise(X_te, noise_pct, rng)
            acc_n = accuracy_score(y_te, m_noise.predict(X_noisy))
            noise_rows.append({"model": model_name, "noise_pct": noise_pct, "accuracy": acc_n})
        print(" done\n")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame([
        {"model": m, "cv_accuracy": v["cv_accuracy"]}
        for m, v in all_results.items()
    ])
    results_df.to_csv(DATA_PROCESSED_DIR / "model_results_expanded.csv", index=False)
    print(f"Saved: {DATA_PROCESSED_DIR / 'model_results_expanded.csv'}")

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(DATA_PROCESSED_DIR / "noise_robustness_expanded.csv", index=False)
    print(f"Saved: {DATA_PROCESSED_DIR / 'noise_robustness_expanded.csv'}\n")

    # ── Generate plots ─────────────────────────────────────────────────────────
    print("Generating figures ...")

    plot_confusion_matrix_grid(
        cms, class_names,
        FIGURES_DIR / "confusion_matrix_grid_expanded.png",
    )
    plot_per_class_f1(
        f1_data, class_names,
        FIGURES_DIR / "per_class_f1_expanded.png",
    )
    plot_accuracy_comparison(
        all_results,
        FIGURES_DIR / "accuracy_comparison_expanded.png",
    )
    plot_noise_robustness(
        noise_df,
        FIGURES_DIR / "noise_robustness_expanded.png",
    )
    plot_bearing_location_accuracy(
        bearing_data,
        FIGURES_DIR / "bearing_location_accuracy_expanded.png",
    )
    plot_fault_size_f1(
        fs_f1_data, fault_sizes,
        FIGURES_DIR / "fault_size_f1_expanded.png",
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    width = 72
    print("\n" + "=" * width)
    print("EXPANDED DATASET — MODEL SUMMARY")
    print("=" * width)
    print(f"{'Model':<16} {'Acc':>7}  {'Best Class':<24}  {'Worst Class':<24}  {'DE>FE':>6}  {'Hardest Size':>13}")
    print("-" * width)

    for model_name in all_results:
        acc     = all_results[model_name]["cv_accuracy"]
        f1_vals = f1_data[model_name]

        best_idx  = int(np.argmax(f1_vals))
        worst_idx = int(np.argmin(f1_vals))
        best_cls  = class_names[best_idx]
        worst_cls = class_names[worst_idx]

        de_acc    = bearing_data[model_name]["DE"]
        fe_acc    = bearing_data[model_name]["FE"]
        de_better = "YES" if de_acc >= fe_acc else "NO"

        if fs_f1_data[model_name]:
            hardest_sz = min(fs_f1_data[model_name], key=fs_f1_data[model_name].get)
        else:
            hardest_sz = "N/A"

        print(
            f"{model_name:<16} {acc:>7.4f}  {best_cls:<24}  {worst_cls:<24}  {de_better:>6}  {hardest_sz:>13}"
        )

    print("=" * width)

    print("\nBearing location accuracy detail:")
    print(f"  {'Model':<16}  {'DE Acc':>8}  {'FE Acc':>8}")
    print(f"  {'-'*40}")
    for m in all_results:
        print(f"  {m:<16}  {bearing_data[m]['DE']:>8.4f}  {bearing_data[m]['FE']:>8.4f}")

    print("\nFault-size macro F1 detail:")
    header = f"  {'Model':<16}" + "".join(f"  {sz:>8}" for sz in fault_sizes)
    print(header)
    print(f"  {'-'*(16 + 10 * len(fault_sizes))}")
    for m in all_results:
        row = f"  {m:<16}" + "".join(
            f"  {fs_f1_data[m].get(sz, 0.0):>8.4f}" for sz in fault_sizes
        )
        print(row)

    print()


if __name__ == "__main__":
    main()
