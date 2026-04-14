"""
06_expanded_shap.py
--------------------
Extended SHAP explainability analysis covering all four model types.

GAP 4 IMPLEMENTED: SHAP Explainer Heterogeneity Acknowledged and Fixed.

EXPLAINER METHODS NOTE:
  • Random Forest    — TreeExplainer   (marginal, exact)       — 500-sample subset
  • XGBoost          — PermutationExplainer (marginal, approx) — 500-sample subset
  • SVM (RBF kernel) — KernelExplainer (Shapley kernel approx) — 100-sample subset
  • 1D-CNN           — GradientExplainer (gradient × input)    — 200-sample subset

IMPORTANT: These explainers compute fundamentally different quantities.
  - TreeExplainer and PermutationExplainer use marginal (interventional) conditioning.
  - KernelExplainer uses conditional expectations approximated via k-means background.
  - GradientExplainer uses integrated gradients (not Shapley-equivalent for all models).
  Direct comparison of raw SHAP magnitudes across these explainer types is
  methodologically invalid. Cross-model comparisons MUST use rank-based consensus.

Per the paper: "Cross-model SHAP comparisons are conducted via feature rank consensus
rather than magnitude comparison, as different explainer classes compute non-equivalent
quantities." (GAP 4 fix — the four-model raw-magnitude bar chart is caveated below.)

Usage:
    python src/06_expanded_shap.py

Inputs:
    data/processed/features_expanded.csv   (real rows only: is_synthetic == False)
    models/cnn_best_fold.keras

Outputs (figures/):
    shap_rf_summary.png
    shap_rf_bar.png
    shap_xgb_summary.png
    shap_xgb_bar.png
    shap_svm_summary.png
    shap_svm_bar.png
    shap_cnn_bar.png
    shap_four_model_comparison.png   ← CAVEATED: raw magnitudes, do not compare directly
    shap_consensus_heatmap.png       ← VALID: rank-based, explainer-agnostic

Outputs (data/processed/):
    shap_consensus.csv
    shap_methods_note.txt   ← Explainer heterogeneity note for paper
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# ── Reproducibility ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.random.set_seed(RANDOM_STATE)

# ── Configuration ───────────────────────────────────────────────────────────────
RF_SHAP_N       = 500   # stratified sample for RF TreeExplainer
XGB_SHAP_N      = 500   # stratified sample for XGBoost TreeExplainer
SVM_SHAP_N      = 100   # stratified sample for KernelExplainer (slow)
SVM_BG_CLUSTERS = 50    # k-means clusters for KernelExplainer background
CNN_BG_N        = 100   # random background samples for GradientExplainer
CNN_SHAP_N      = 200   # stratified sample for GradientExplainer
TOP_N_CONSENSUS = 10    # top-N features in consensus heatmap
TOP_K           = 5     # top-K for per-model summary

DATA_PROC_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
FIGURES_DIR   = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mean", "std", "rms", "peak", "peak2peak",
    "crest_factor", "kurtosis", "skewness", "shape_factor",
    "spectral_centroid", "dominant_frequency",
    "low_band_energy", "mid_band_energy", "high_band_energy",
    "motor_load",
]

FEATURE_DISPLAY = {
    "mean":               "Mean",
    "std":                "Std Dev",
    "rms":                "RMS",
    "peak":               "Peak",
    "peak2peak":          "Peak-to-Peak",
    "crest_factor":       "Crest Factor",
    "kurtosis":           "Kurtosis",
    "skewness":           "Skewness",
    "shape_factor":       "Shape Factor",
    "spectral_centroid":  "Spectral Centroid",
    "dominant_frequency": "Dominant Freq",
    "low_band_energy":    "Low Band Energy",
    "mid_band_energy":    "Mid Band Energy",
    "high_band_energy":   "High Band Energy",
    "motor_load":         "Motor Load",
}
DISPLAY_NAMES = [FEATURE_DISPLAY[f] for f in FEATURE_COLS]


# ── Helpers ─────────────────────────────────────────────────────────────────────

def stratified_sample(X: np.ndarray, y: np.ndarray, n: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_sub, y_sub) — stratified sample of size min(n, len(X))."""
    n = min(n, len(X))
    classes, counts = np.unique(y, return_counts=True)
    fracs  = counts / counts.sum()
    chosen = []
    for cls, frac in zip(classes, fracs):
        cls_idx  = np.where(y == cls)[0]
        k        = max(1, int(round(frac * n)))
        k        = min(k, len(cls_idx))
        chosen.append(rng.choice(cls_idx, size=k, replace=False))
    idx = np.concatenate(chosen)
    # trim/pad to exactly n if rounding drifts
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return X[idx], y[idx], idx


def tree_shap_3d(model, X: np.ndarray) -> np.ndarray:
    """
    TreeExplainer → (n_samples, n_features, n_classes).
    Handles both list-of-arrays (older shap) and 3-D array (newer shap).
    """
    exp = shap.TreeExplainer(model)
    sv  = exp.shap_values(X)
    if isinstance(sv, list):
        return np.stack(sv, axis=-1)          # list of (n,f) → (n,f,c)
    return sv                                  # already (n,f,c) in newer shap


def permutation_shap_3d(predict_fn, X_bg: np.ndarray, X: np.ndarray,
                         n_classes: int) -> np.ndarray:
    """
    PermutationExplainer fallback for models incompatible with TreeExplainer.
    Returns (n_samples, n_features, n_classes).
    """
    exp = shap.PermutationExplainer(predict_fn, X_bg)
    sv  = exp(X)
    vals = sv.values  # (n_samples, n_features, n_classes)
    return vals


def mean_abs_global(shap_3d: np.ndarray) -> np.ndarray:
    """Global mean |SHAP| over samples and classes → (n_features,)."""
    return np.abs(shap_3d).mean(axis=(0, 2))


def feature_ranks(mean_abs: np.ndarray) -> np.ndarray:
    """Rank array: rank 1 = most important. Returns (n_features,) int array."""
    order = np.argsort(mean_abs)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


# ── Plot utilities ───────────────────────────────────────────────────────────────

def plot_beeswarm(shap_values: np.ndarray, X_data: np.ndarray,
                  title: str, save_path: Path):
    """
    Beeswarm summary plot (mean over classes when multi-class).
    shap_values: (n_samples, n_features, n_classes) or (n_samples, n_features)
    """
    if shap_values.ndim == 3:
        sv2d = shap_values.mean(axis=2)   # (n_samples, n_features)
    else:
        sv2d = shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv2d, X_data, feature_names=DISPLAY_NAMES, show=False, plot_size=None)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_bar(mean_abs: np.ndarray, title: str, save_path: Path,
                  color: str = "#2196F3", xlim: float | None = None):
    """Horizontal bar chart of global mean |SHAP|."""
    idx_s  = np.argsort(mean_abs)
    names_s = [DISPLAY_NAMES[i] for i in idx_s]
    vals_s  = mean_abs[idx_s]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(names_s, vals_s, color=color, edgecolor="black", lw=0.4)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    if xlim is not None:
        ax.set_xlim(0, xlim)
    for bar, val in zip(bars, vals_s):
        ax.text(val + vals_s.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_four_model_comparison(mean_abs_dict: dict, save_path: Path):
    """
    2×2 panel figure — each model's raw SHAP magnitudes.

    METHODOLOGICAL CAVEAT (GAP 4):
    These panels use raw mean |SHAP| values from DIFFERENT explainer types.
    Direct cross-panel magnitude comparison is NOT valid:
      RF → TreeExplainer, XGBoost → PermutationExplainer,
      SVM → KernelExplainer,    CNN → GradientExplainer.
    Use the rank-consensus heatmap for valid cross-model comparison.
    This figure is retained for within-model feature importance only.
    """
    model_names = list(mean_abs_dict.keys())
    colors = {
        "Random Forest": "#42A5F5",
        "XGBoost":       "#FF7043",
        "SVM":           "#66BB6A",
        "1D-CNN":        "#AB47BC",
    }
    explainer_labels = {
        "Random Forest": "TreeExplainer",
        "XGBoost":       "PermutationExplainer",
        "SVM":           "KernelExplainer",
        "1D-CNN":        "GradientExplainer",
    }

    # Use per-model x-axis (NOT shared global_max) to avoid misleading comparisons
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        "SHAP Feature Importance per Model (Within-Model Ranking Only)\n"
        "⚠ Cross-model magnitude comparison is INVALID — see rank-consensus heatmap",
        fontsize=13, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for i, mname in enumerate(model_names[:4]):
        ax      = axes_flat[i]
        vals    = mean_abs_dict[mname]
        idx_s   = np.argsort(vals)
        names_s = [DISPLAY_NAMES[j] for j in idx_s]
        vals_s  = vals[idx_s]
        col     = colors.get(mname, "#888888")
        explainer = explainer_labels.get(mname, "Unknown")

        ax.barh(names_s, vals_s, color=col, edgecolor="black", lw=0.4)
        ax.set_title(f"{mname}\n({explainer})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|  [within-model only]", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=9)
        # Add per-model caveat annotation
        ax.text(
            0.99, 0.01, f"Explainer: {explainer}\n(not comparable across panels)",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="gray", style="italic",
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    print(f"  NOTE: Cross-panel magnitude comparison is invalid (different explainers).")


def plot_consensus_heatmap(rank_df: pd.DataFrame, save_path: Path):
    """Heatmap: rank per feature × model. Green=1 (important), Red=15 (unimportant)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "SHAP Rank (1 = most important)"},
        annot_kws={"size": 9},
        vmin=1, vmax=15,
    )
    ax.set_title(
        f"Cross-Model SHAP Rank Comparison (Top {len(rank_df)} Features)\n"
        "Green = rank 1 (most important)   Red = rank 15 (least important)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    # ── Load real data ──────────────────────────────────────────────────────────
    data_path = DATA_PROC_DIR / "features_expanded.csv"
    if not data_path.exists():
        print(f"[ERROR] {data_path} not found. Run 02_train_models_expanded.py first.")
        return

    df = pd.read_csv(data_path)
    df_real = df[df["is_synthetic"] == False].copy()
    print(f"Loaded {len(df_real):,} real samples from {data_path}")

    le = LabelEncoder()
    df_real["label_enc"] = le.fit_transform(df_real["label"])
    class_names = list(le.classes_)
    n_classes   = len(class_names)
    print(f"Classes ({n_classes}): {class_names}\n")

    X = df_real[FEATURE_COLS].fillna(0).values.astype(np.float64)
    y = df_real["label_enc"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rng = np.random.default_rng(RANDOM_STATE)

    # ── Train RF, XGBoost, SVM on full real dataset ─────────────────────────────
    print("Training Random Forest on full real dataset ...")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X, y)

    print("Training XGBoost on full real dataset ...")
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_STATE, verbosity=0,
    )
    xgb.fit(X, y)

    print("Training SVM on full real dataset (scaled) ...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              random_state=RANDOM_STATE)
    svm.fit(X_scaled, y)
    print()

    # ── Stratified subsets ──────────────────────────────────────────────────────
    X_rf_s,  y_rf_s,  _  = stratified_sample(X,        y, RF_SHAP_N,  rng)
    X_xgb_s, y_xgb_s, _  = stratified_sample(X,        y, XGB_SHAP_N, rng)
    X_svm_s, y_svm_s, _  = stratified_sample(X_scaled, y, SVM_SHAP_N, rng)
    X_cnn_s, y_cnn_s, _  = stratified_sample(X_scaled, y, CNN_SHAP_N, rng)

    # ── TreeExplainer — RF ──────────────────────────────────────────────────────
    print(f"Computing TreeSHAP — Random Forest (n={len(X_rf_s)}) ...")
    shap_rf     = tree_shap_3d(rf, X_rf_s)
    mean_abs_rf = mean_abs_global(shap_rf)
    print(f"  RF SHAP shape: {shap_rf.shape}")

    plot_beeswarm(shap_rf, X_rf_s,
                  "Random Forest — SHAP Beeswarm Summary (TreeExplainer)",
                  FIGURES_DIR / "shap_rf_summary.png")
    plot_shap_bar(mean_abs_rf,
                  "Random Forest — Global SHAP Feature Importance (TreeExplainer)",
                  FIGURES_DIR / "shap_rf_bar.png", color="#42A5F5")

    # ── PermutationExplainer — XGBoost ─────────────────────────────────────────
    # Note: shap 0.49.x TreeExplainer is incompatible with XGBoost 3.x multi-class
    # models (base_score stored as vector). PermutationExplainer gives valid SHAP
    # values via marginal conditioning — slightly slower but algorithmically correct.
    print(f"\nComputing PermutationSHAP — XGBoost (n={len(X_xgb_s)}) ...")
    print("  [Note: PermutationExplainer used — XGBoost 3.x / shap 0.49 compatibility]")
    shap_xgb     = permutation_shap_3d(xgb.predict_proba, X[:200], X_xgb_s, n_classes)
    mean_abs_xgb = mean_abs_global(shap_xgb)
    print(f"  XGB SHAP shape: {shap_xgb.shape}")

    plot_beeswarm(shap_xgb, X_xgb_s,
                  "XGBoost — SHAP Beeswarm Summary (PermutationExplainer)",
                  FIGURES_DIR / "shap_xgb_summary.png")
    plot_shap_bar(mean_abs_xgb,
                  "XGBoost — Global SHAP Feature Importance (PermutationExplainer)",
                  FIGURES_DIR / "shap_xgb_bar.png", color="#FF7043")

    # ── KernelExplainer — SVM ───────────────────────────────────────────────────
    print(f"\nComputing KernelSHAP — SVM (background={SVM_BG_CLUSTERS} k-means clusters) ...")
    print(f"  Subset size: {len(X_svm_s)} samples — progress every 10 ...")

    background    = shap.kmeans(X_scaled, SVM_BG_CLUSTERS)
    svm_explainer = shap.KernelExplainer(svm.predict_proba, background)

    shap_svm_list: list[np.ndarray] = []
    BATCH = 10
    n_feat = len(FEATURE_COLS)
    for start in range(0, len(X_svm_s), BATCH):
        end  = min(start + BATCH, len(X_svm_s))
        bsz  = end - start
        sv   = svm_explainer.shap_values(X_svm_s[start:end], silent=True)
        # Normalize to (batch, n_features, n_classes) regardless of shap format:
        #   - Old shap: list of n_classes arrays each (batch, n_feat)  → stack axis=-1
        #   - New shap: list of batch arrays each (n_feat, n_classes)  → np.array then OK
        #   - Or: numpy array (n_classes, batch, n_feat)               → transpose
        arr = np.array(sv)
        if arr.ndim == 3:
            if arr.shape == (bsz, n_feat, n_classes):
                batch_sv = arr                              # already correct
            elif arr.shape == (n_classes, bsz, n_feat):
                batch_sv = arr.transpose(1, 2, 0)          # (n, f, c)
            elif arr.shape == (bsz, n_classes, n_feat):
                batch_sv = arr.transpose(0, 2, 1)          # (n, f, c)
            elif arr.shape == (n_feat, n_classes, bsz):
                batch_sv = arr.transpose(2, 0, 1)          # (n, f, c)
            else:
                # Fallback: stack as list of (n_feat, n_cls) per sample
                batch_sv = arr.reshape(bsz, n_feat, n_classes)
        else:
            # list of (bsz, n_feat) per class — classic shap format
            batch_sv = np.stack(list(sv), axis=-1)         # (batch, n_feat, n_cls)
        shap_svm_list.append(batch_sv)
        print(f"  SVM SHAP: {end}/{len(X_svm_s)} samples", end="\r", flush=True)

    shap_svm     = np.concatenate(shap_svm_list, axis=0)
    mean_abs_svm = mean_abs_global(shap_svm)
    print(f"\n  SVM SHAP shape: {shap_svm.shape}")

    plot_beeswarm(shap_svm, X_svm_s,
                  "SVM — SHAP Beeswarm Summary (KernelExplainer)",
                  FIGURES_DIR / "shap_svm_summary.png")
    plot_shap_bar(mean_abs_svm,
                  "SVM — Global SHAP Feature Importance (KernelExplainer)",
                  FIGURES_DIR / "shap_svm_bar.png", color="#66BB6A")

    # ── GradientExplainer — 1D-CNN ──────────────────────────────────────────────
    cnn_path    = MODELS_DIR / "cnn_best_fold.keras"
    shap_cnn     = None
    mean_abs_cnn = None

    if cnn_path.exists():
        print(f"\nComputing GradientSHAP — 1D-CNN (load: {cnn_path}) ...")
        print(f"  Background: {CNN_BG_N} random training samples")
        print(f"  Explain:    {len(X_cnn_s)} stratified samples")

        cnn_model = tf.keras.models.load_model(str(cnn_path))

        # Random background (different from SHAP subset)
        bg_idx    = rng.choice(len(X_scaled), size=CNN_BG_N, replace=False)
        X_bg_cnn  = X_scaled[bg_idx, :, np.newaxis].astype(np.float32)
        X_ex_cnn  = X_cnn_s[:, :, np.newaxis].astype(np.float32)

        cnn_explainer = shap.GradientExplainer(cnn_model, X_bg_cnn)
        sv_cnn_raw    = cnn_explainer.shap_values(X_ex_cnn)

        # sv_cnn_raw: list of (n_samples, n_features, 1) — one element per class
        if isinstance(sv_cnn_raw, list):
            sv_cnn_sq = [sv[:, :, 0] for sv in sv_cnn_raw]   # (n_samples, n_features)
            shap_cnn  = np.stack(sv_cnn_sq, axis=-1)           # (n_samples, n_features, n_classes)
        else:
            # already (n_samples, n_features, 1, n_classes) or similar
            shap_cnn = sv_cnn_raw.squeeze(axis=2) if sv_cnn_raw.ndim == 4 else sv_cnn_raw

        mean_abs_cnn = mean_abs_global(shap_cnn)
        print(f"  CNN SHAP shape: {shap_cnn.shape}")

        plot_shap_bar(mean_abs_cnn,
                      "1D-CNN — Global SHAP Feature Importance (GradientExplainer)",
                      FIGURES_DIR / "shap_cnn_bar.png", color="#AB47BC")
    else:
        print(f"\n[SKIP] CNN model not found at {cnn_path}. Run Script 05 first.")

    # ── Assemble mean_abs dict and rank arrays ───────────────────────────────────
    mean_abs_dict = {
        "Random Forest": mean_abs_rf,
        "XGBoost":       mean_abs_xgb,
        "SVM":           mean_abs_svm,
    }
    if mean_abs_cnn is not None:
        mean_abs_dict["1D-CNN"] = mean_abs_cnn

    rank_data = {m: feature_ranks(v) for m, v in mean_abs_dict.items()}

    # ── Cross-model consensus ────────────────────────────────────────────────────
    top5_sets      = {m: set(np.argsort(v)[::-1][:TOP_K]) for m, v in mean_abs_dict.items()}
    consensus_idx  = set.intersection(*top5_sets.values())

    # Average rank across models → top-N for heatmap
    avg_rank   = np.mean([rank_data[m] for m in rank_data], axis=0)
    top_n_idx  = np.argsort(avg_rank)[:TOP_N_CONSENSUS]

    # Build columns matching spec: RF_rank, XGBoost_rank, SVM_rank, CNN_rank, mean_rank
    col_map = {
        "Random Forest": "RF_rank",
        "XGBoost":       "XGBoost_rank",
        "SVM":           "SVM_rank",
        "1D-CNN":        "CNN_rank",
    }

    records = []
    for fi in top_n_idx:
        rec = {"feature": DISPLAY_NAMES[fi]}
        for m, col in col_map.items():
            if m in rank_data:
                rec[col] = int(rank_data[m][fi])
        rec["mean_rank"] = float(avg_rank[fi])
        records.append(rec)

    consensus_df   = pd.DataFrame(records).sort_values("mean_rank").reset_index(drop=True)
    consensus_path = DATA_PROC_DIR / "shap_consensus.csv"
    consensus_df.to_csv(consensus_path, index=False)
    print(f"\nConsensus CSV saved → {consensus_path}")

    # ── Consensus heatmap ────────────────────────────────────────────────────────
    heatmap_cols = [col_map[m] for m in rank_data if m in col_map]
    rank_df = pd.DataFrame(
        {col_map[m]: rank_data[m][top_n_idx] for m in rank_data if m in col_map},
        index=[DISPLAY_NAMES[i] for i in top_n_idx],
    )
    # Sort rows by avg rank (most important first)
    rank_df = rank_df.loc[[DISPLAY_NAMES[i] for i in top_n_idx]]

    plot_consensus_heatmap(rank_df, FIGURES_DIR / "shap_consensus_heatmap.png")

    # ── 4-panel comparison ───────────────────────────────────────────────────────
    plot_four_model_comparison(mean_abs_dict, FIGURES_DIR / "shap_four_model_comparison.png")

    # ── Final summary ────────────────────────────────────────────────────────────
    sep = "=" * 65

    print(f"\n{sep}")
    print("TOP-5 FEATURES PER MODEL:")
    print(sep)
    for mname, vals in mean_abs_dict.items():
        top5_idx = np.argsort(vals)[::-1][:TOP_K]
        names    = [DISPLAY_NAMES[i] for i in top5_idx]
        print(f"  {mname:<18}: {names}")

    print(f"\n{sep}")
    print(f"TOP-5 CONSENSUS FEATURES (lowest mean rank across all models):")
    print(sep)
    top5_consensus_df = consensus_df.head(TOP_K)
    for _, row in top5_consensus_df.iterrows():
        rank_str = "  ".join(f"{col}={int(row[col])}" for col in heatmap_cols if col in row)
        print(f"  {row['feature']:<22}  mean_rank={row['mean_rank']:.1f}  ({rank_str})")

    print(f"\n{sep}")
    if consensus_idx:
        names_consensus = sorted([DISPLAY_NAMES[i] for i in consensus_idx])
        print(f"FEATURES IN TOP-{TOP_K} FOR ALL {len(mean_abs_dict)} MODELS:")
        for name in names_consensus:
            print(f"  ✓ {name}")
    else:
        print(f"No single feature appears in top-{TOP_K} for ALL {len(mean_abs_dict)} models.")
        print("  (This is common — each model emphasises different aspects.)")
        print(f"  Use the consensus heatmap to identify the most consistently ranked features.")

    print(f"\n{sep}")
    print(f"All figures saved to {FIGURES_DIR}/")
    print(f"Consensus CSV   → {consensus_path}")

    # ── GAP 4: Write SHAP methods note for paper ─────────────────────────────
    methods_note_path = DATA_PROC_DIR / "shap_methods_note.txt"
    explainer_rows = [
        ("Random Forest", "TreeExplainer",         "Exact Shapley (marginal)"),
        ("XGBoost",       "PermutationExplainer",   "Marginal approximation"),
        ("SVM",           "KernelExplainer",        "Kernel-weighted approx"),
        ("1D-CNN",        "GradientExplainer",      "Gradient × input (approx)"),
    ]
    note_lines = [
        "SHAP EXPLAINER HETEROGENEITY — Methods Note for Paper (GAP 4)",
        "=" * 70,
        "",
        "Explainer types used per model:",
        "",
        f"  {'Model':<18}  {'Explainer Class':<25}  {'Computation Method':<35}",
        "  " + "-" * 80,
    ]
    for model_name, expl_class, method in explainer_rows:
        note_lines.append(f"  {model_name:<18}  {expl_class:<25}  {method:<35}")
    note_lines += [
        "",
        "CRITICAL NOTE ON CROSS-MODEL COMPARISON:",
        "  These explainer classes compute fundamentally different quantities.",
        "  TreeExplainer and PermutationExplainer use marginal (interventional)",
        "  conditioning. KernelExplainer uses conditional expectations approximated",
        "  via k-means background samples. GradientExplainer computes integrated",
        "  gradients, which are not Shapley-equivalent for non-linear models.",
        "",
        "  Consequence: Raw SHAP magnitude values from different explainers cannot",
        "  be directly compared. A feature with mean|SHAP|=0.5 from TreeExplainer",
        "  is NOT equivalent to mean|SHAP|=0.5 from KernelExplainer.",
        "",
        "PAPER SENTENCE (add to Methods/SHAP section):",
        '  "Cross-model SHAP comparisons are conducted via feature rank consensus',
        "  rather than magnitude comparison, as different explainer classes compute",
        '  non-equivalent quantities."',
        "",
        "FIGURES VALIDITY:",
        "  shap_consensus_heatmap.png   — VALID (rank-based, explainer-agnostic)",
        "  shap_rf_bar.png              — VALID (within RF model only)",
        "  shap_xgb_bar.png             — VALID (within XGB model only)",
        "  shap_svm_bar.png             — VALID (within SVM model only)",
        "  shap_cnn_bar.png             — VALID (within CNN model only)",
        "  shap_four_model_comparison.png — WITHIN-MODEL valid; CROSS-MODEL invalid.",
        "    → Caption must state: 'per-panel ranking is valid; cross-panel",
        "      magnitude comparison is not methodologically valid.'",
        "",
    ]
    with open(methods_note_path, "w") as f:
        f.write("\n".join(note_lines) + "\n")
    print(f"SHAP methods note → {methods_note_path}")

    print()
    print("Research insight:")
    print("  SHAP consensus uses RANK-BASED comparison (explainer-agnostic) to")
    print("  identify features that rank highly across ALL models regardless of")
    print("  explainer class. This is the methodologically valid approach.")
    print()
    print("  Within each model, per-model bar charts show valid feature importance.")
    print("  The four-model comparison panel is retained for within-model reference")
    print("  ONLY. Cross-panel magnitude comparisons must not be made in the paper.")


if __name__ == "__main__":
    main()
