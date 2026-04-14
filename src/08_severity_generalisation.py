"""
08_severity_generalisation.py
------------------------------
Severity Generalisation Experiment (GAP 3).

Two experiments:

1. Per-class severity generalisation gap:
   Train on fault_size in {"0.007", "0.014"} + normal,
   test on fault_size == "0.021".
   Run under two conditions:
     (a) Without VAE augmentation  (real data only)
     (b) With VAE augmentation     (synthetic added to training only)
   Reports per-class F1 on test set and generalisation gap = train_f1 - gen_f1.
   → Table 8, Figure 8

2. Noise crossover analysis:
   Iterate Gaussian noise 0% → 30% in 1% increments.
   Find exact noise level where SVM first surpasses XGBoost accuracy.
   Reports precise crossover threshold.

Inputs:
    data/processed/features_expanded.csv
    data/processed/features_augmented.csv

Outputs:
    data/processed/severity_gen_results.csv
    data/processed/noise_crossover_fine.csv
    paper/tables/table8_severity_gen.csv / .txt
    paper/figures/fig8_severity_gen_gap.pdf / .png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths & config ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR   = Path("data/processed")
TABLES_DIR = Path("paper/tables")
FIG_DIR    = Path("paper/figures")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mean", "std", "rms", "peak", "peak2peak",
    "crest_factor", "kurtosis", "skewness", "shape_factor",
    "spectral_centroid", "dominant_frequency",
    "low_band_energy", "mid_band_energy", "high_band_energy",
    "motor_load",
]
N_FOLDS = 5

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ── Model factory ────────────────────────────────────────────────────────────

def build_models():
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


# ── Generalisation experiment ────────────────────────────────────────────────

def run_generalisation(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    le: LabelEncoder,
    class_names: list,
    n_classes: int,
    use_synthetic: bool,
) -> list[dict]:
    """
    Train on 0.007" + 0.014" + normal; test on 0.021".
    Returns list of dicts: {condition, model, class, cv_f1, gen_f1, gap}.
    """
    condition = "real+vae" if use_synthetic else "real_only"

    # Training split: small severities + normal
    train_mask = (
        (df_real["fault_size"].isin(["0.007", "0.014"])) |
        (df_real["fault_size"] == "none")
    )
    test_mask  = df_real["fault_size"] == "0.021"

    df_train_r = df_real[train_mask].reset_index(drop=True)
    df_test    = df_real[test_mask].reset_index(drop=True)

    if df_test.empty:
        print(f"  [WARN] No 0.021\" samples found — check fault_size column values.")
        return []

    X_train_r = df_train_r[FEATURE_COLS].values.astype(np.float64)
    y_train_r = le.transform(df_train_r["label"].values)
    X_test    = df_test[FEATURE_COLS].values.astype(np.float64)
    y_test    = le.transform(df_test["label"].values)

    # Add synthetic to training only
    if use_synthetic and len(df_synth) > 0:
        X_train = np.concatenate([X_train_r, df_synth[FEATURE_COLS].values.astype(np.float64)])
        y_train = np.concatenate([y_train_r, le.transform(df_synth["label"].values)])
    else:
        X_train, y_train = X_train_r, y_train_r

    # Train-CV F1 (5-fold on training split only, real rows)
    skf = StratifiedKFold(n_splits=min(N_FOLDS, np.unique(y_train_r, return_counts=True)[1].min()),
                          shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_name, _ in build_models().items():
        clf = build_models()[model_name]
        try:
            y_pred_cv = cross_val_predict(
                build_models()[model_name], X_train_r, y_train_r,
                cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                n_jobs=-1,
            )
        except Exception:
            y_pred_cv = None

        # Train full model on all training data, evaluate on test set
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)

        # Per-class F1 on test (only classes present in test set)
        test_classes = np.unique(y_test)
        f1_per_class_test = f1_score(
            y_test, y_pred_test,
            average=None, labels=list(range(n_classes)), zero_division=0,
        )

        # Per-class F1 on training CV
        if y_pred_cv is not None:
            f1_per_class_train = f1_score(
                y_train_r, y_pred_cv,
                average=None, labels=list(range(n_classes)), zero_division=0,
            )
        else:
            f1_per_class_train = np.zeros(n_classes)

        for cls_idx in range(n_classes):
            cls_name = class_names[cls_idx]
            # Only report for classes in test set (must have 0.021" data)
            has_test_data = cls_idx in test_classes
            gen_f1  = float(f1_per_class_test[cls_idx])
            cv_f1   = float(f1_per_class_train[cls_idx])
            gap     = cv_f1 - gen_f1 if has_test_data else None
            rows.append({
                "condition":    condition,
                "model":        model_name,
                "class":        cls_name,
                "has_test_data": has_test_data,
                "cv_f1":        cv_f1,
                "gen_f1":       gen_f1 if has_test_data else None,
                "gap":          gap,
            })

        gen_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
        print(f"  [{condition}] {model_name:<16}  gen_macro_f1={gen_macro:.4f}  "
              f"(test_n={len(df_test)})")

    return rows


# ── Noise crossover analysis ─────────────────────────────────────────────────

def run_noise_crossover(df_real: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    """
    Iterate Gaussian noise 0% → 30% in 1% steps.
    Find the exact crossover where SVM accuracy first surpasses XGBoost accuracy.
    """
    print("\n" + "=" * 65)
    print("NOISE CROSSOVER: SVM vs XGBoost (0% to 30% in 1% steps)")
    print("=" * 65)

    X_all = df_real[FEATURE_COLS].values.astype(np.float64)
    y_all = le.transform(df_real["label"].values)

    # 80/20 split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_STATE,
    )

    # Train both models once on clean data
    svm_model = SVC(kernel="rbf", random_state=RANDOM_STATE)
    xgb_model = XGBClassifier(
        n_estimators=100, eval_metric="mlogloss",
        random_state=RANDOM_STATE, verbosity=0,
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    svm_model.fit(X_tr_s, y_tr)
    xgb_model.fit(X_tr, y_tr)
    print("  Models trained. Iterating noise levels ...")

    rng   = np.random.default_rng(RANDOM_STATE)
    rows  = []
    noise_levels = [i / 100.0 for i in range(31)]   # 0.00 to 0.30

    crossover_noise = None

    for noise_pct in noise_levels:
        noise_std = X_te_s.std(axis=0, keepdims=True)
        if noise_pct > 0.0:
            X_noisy_s = X_te_s + rng.normal(0, noise_pct * noise_std, size=X_te_s.shape)
            X_noisy   = X_te   + rng.normal(0, noise_pct * X_te.std(axis=0, keepdims=True),
                                             size=X_te.shape)
        else:
            X_noisy_s = X_te_s
            X_noisy   = X_te

        svm_acc = accuracy_score(y_te, svm_model.predict(X_noisy_s))
        xgb_acc = accuracy_score(y_te, xgb_model.predict(X_noisy))
        leader  = "SVM" if svm_acc >= xgb_acc else "XGBoost"

        rows.append({
            "noise_pct":   noise_pct,
            "svm_acc":     svm_acc,
            "xgboost_acc": xgb_acc,
            "leader":      leader,
        })

        if crossover_noise is None and svm_acc > xgb_acc:
            crossover_noise = noise_pct

        if int(noise_pct * 100) % 5 == 0:
            print(f"  noise={noise_pct:.2f}  SVM={svm_acc:.4f}  XGBoost={xgb_acc:.4f}  "
                  f"leader={leader}")

    df_cross = pd.DataFrame(rows)

    if crossover_noise is not None:
        print(f"\n  *** CROSSOVER POINT: noise = {crossover_noise:.2f} "
              f"({crossover_noise*100:.0f}% of feature std) ***")
        print(f"  At this level, SVM first surpasses XGBoost accuracy.")
        print(f"  Threshold recommendation: prefer SVM over XGBoost when")
        print(f"  Gaussian noise exceeds {crossover_noise*100:.0f}% of feature standard deviation.")
    else:
        print("\n  No crossover found in range 0–30%. XGBoost dominates throughout.")

    df_cross["crossover_threshold"] = crossover_noise

    out_path = DATA_DIR / "noise_crossover_fine.csv"
    df_cross.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return df_cross, crossover_noise


# ── Figure 8: Severity generalisation gap ───────────────────────────────────

def plot_fig8(gen_df: pd.DataFrame, class_names: list, crossover_noise: float):
    """
    Figure 8: Two-panel figure.
    Left:  Per-class generalisation gap (bars) for each model, real_only condition.
    Right: Noise crossover curve (SVM vs XGBoost).
    """
    cross_path = DATA_DIR / "noise_crossover_fine.csv"
    if cross_path.exists():
        df_cross = pd.read_csv(cross_path)
    else:
        df_cross = None

    models      = ["Random Forest", "XGBoost", "SVM", "KNN"]
    real_df     = gen_df[(gen_df["condition"] == "real_only") & gen_df["has_test_data"]]
    fault_classes = sorted([c for c in class_names if c != "normal"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: per-class generalisation gap heatmap
    ax = axes[0]
    gap_matrix = np.full((len(fault_classes), len(models)), np.nan)
    for r, cls_name in enumerate(fault_classes):
        for c, model_name in enumerate(models):
            sub = real_df[(real_df["class"] == cls_name) & (real_df["model"] == model_name)]
            if not sub.empty and sub.iloc[0]["gap"] is not None:
                gap_matrix[r, c] = sub.iloc[0]["gap"]

    im = ax.imshow(gap_matrix, cmap="RdYlGn_r", vmin=-0.05, vmax=0.30, aspect="auto")
    plt.colorbar(im, ax=ax, label="Generalisation Gap (Train F1 − Test F1)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(fault_classes)))
    ax.set_yticklabels(fault_classes, fontsize=9)
    ax.set_title(
        "Fig. 8a: Per-Class Severity Generalisation Gap\n"
        "(Train: 0.007\" + 0.014\", Test: 0.021\")",
        fontsize=10, fontweight="bold",
    )
    for r in range(len(fault_classes)):
        for c in range(len(models)):
            v = gap_matrix[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if v < 0.15 else "white")

    # Right panel: noise crossover
    ax2 = axes[1]
    if df_cross is not None:
        noise_pcts = df_cross["noise_pct"] * 100
        ax2.plot(noise_pcts, df_cross["svm_acc"],    label="SVM",     linewidth=2,
                 marker="s", markevery=5, markersize=4, color="#FF7043")
        ax2.plot(noise_pcts, df_cross["xgboost_acc"], label="XGBoost", linewidth=2,
                 marker="o", markevery=5, markersize=4, color="#42A5F5")
        if crossover_noise is not None:
            ax2.axvline(crossover_noise * 100, color="black", linestyle="--",
                        linewidth=1.5, label=f"Crossover = {crossover_noise*100:.0f}%")
        ax2.set_xlabel("Gaussian Noise Level (% of feature std)", fontsize=10)
        ax2.set_ylabel("Accuracy", fontsize=10)
        ax2.set_title("Fig. 8b: SVM vs XGBoost Noise Robustness\n(Crossover Threshold Detection)",
                      fontsize=10, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Crossover data not available", ha="center", va="center",
                 transform=ax2.transAxes)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"fig8_severity_gen_gap.{ext}"
        plt.savefig(path, dpi=300)
        print(f"  Saved: {path}")
    plt.close()


# ── Table 8 ──────────────────────────────────────────────────────────────────

def save_table8(gen_df: pd.DataFrame):
    """Table 8: Per-class severity generalisation gap (real_only condition)."""
    models        = ["Random Forest", "XGBoost", "SVM", "KNN"]
    real_df       = gen_df[gen_df["condition"] == "real_only"]

    # Build per-class × per-model gap table
    records = []
    all_classes = sorted(real_df["class"].unique())
    for cls_name in all_classes:
        rec = {"Class": cls_name}
        for model_name in models:
            sub = real_df[(real_df["class"] == cls_name) & (real_df["model"] == model_name)]
            if not sub.empty:
                row = sub.iloc[0]
                if row["has_test_data"] and row["gap"] is not None:
                    rec[f"{model_name} Gap"] = f"{row['gap']:.4f}"
                else:
                    rec[f"{model_name} Gap"] = "N/A"
            else:
                rec[f"{model_name} Gap"] = "N/A"
        records.append(rec)

    table_df = pd.DataFrame(records)

    csv_path = TABLES_DIR / "table8_severity_gen.csv"
    txt_path = TABLES_DIR / "table8_severity_gen.txt"
    table_df.to_csv(csv_path, index=False)

    col_widths = [max(len(str(c)), table_df[c].astype(str).map(len).max()) + 2
                  for c in table_df.columns]
    header = "  ".join(str(c).ljust(w) for c, w in zip(table_df.columns, col_widths))
    sep    = "  ".join("-" * w for w in col_widths)
    lines  = [
        "TABLE 8 — Per-Class Severity Generalisation Gap",
        "Train: fault_size ∈ {0.007\", 0.014\"} + normal",
        "Test:  fault_size = 0.021\" (unseen severity)",
        "Gap = (5-fold CV F1 on training split) − (F1 on test set)",
        "N/A = class has no 0.021\" test samples in this dataset",
        "",
        header, sep,
    ]
    for _, r in table_df.iterrows():
        lines.append("  ".join(str(v).ljust(w) for v, w in zip(r, col_widths)))
    lines.append("")
    lines.append("Note: Large positive gap indicates a class that does not")
    lines.append("generalise well to unseen fault severity. These classes are")
    lines.append("candidates for additional training data at intermediate severities.")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nTable 8 saved → {csv_path}")
    print(f"              → {txt_path}")
    return table_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    feat_path = DATA_DIR / "features_expanded.csv"
    aug_path  = DATA_DIR / "features_augmented.csv"

    if not feat_path.exists():
        print(f"[ERROR] {feat_path} not found. Run 01_load_and_preprocess.py first.")
        return

    df_all = pd.read_csv(feat_path)
    if "is_synthetic" in df_all.columns:
        df_real = df_all[df_all["is_synthetic"] == False].reset_index(drop=True)
    else:
        df_real = df_all

    print(f"Loaded {len(df_real):,} real samples.")
    print(f"Fault sizes present: {sorted(df_real['fault_size'].unique())}\n")

    le = LabelEncoder()
    le.fit(df_real["label"].values)
    class_names = list(le.classes_)
    n_classes   = len(class_names)
    print(f"Classes ({n_classes}): {class_names}\n")

    # Load synthetic data for condition (b)
    df_synth = pd.DataFrame()
    if aug_path.exists():
        df_aug = pd.read_csv(aug_path)
        if "is_synthetic" in df_aug.columns:
            df_synth = df_aug[df_aug["is_synthetic"] == True].reset_index(drop=True)
            # Clean synthetic data
            df_synth = df_synth.dropna(subset=FEATURE_COLS).reset_index(drop=True)
            for col in FEATURE_COLS:
                df_synth[col] = df_synth[col].clip(
                    lower=df_real[col].min(), upper=df_real[col].max()
                )
        print(f"Loaded {len(df_synth):,} synthetic samples for condition (b).")
    else:
        print("[INFO] features_augmented.csv not found — only running condition (a).")

    # ── Experiment 1: Severity generalisation ─────────────────────────────────
    print("\n" + "=" * 65)
    print("SEVERITY GENERALISATION — Condition (a): Real data only")
    print("=" * 65)
    rows_a = run_generalisation(df_real, df_synth, le, class_names, n_classes,
                                use_synthetic=False)

    if len(df_synth) > 0:
        print("\n" + "=" * 65)
        print("SEVERITY GENERALISATION — Condition (b): Real + VAE synthetic")
        print("=" * 65)
        rows_b = run_generalisation(df_real, df_synth, le, class_names, n_classes,
                                    use_synthetic=True)
    else:
        rows_b = []

    all_gen_rows = rows_a + rows_b
    gen_df       = pd.DataFrame(all_gen_rows)

    out_path = DATA_DIR / "severity_gen_results.csv"
    gen_df.to_csv(out_path, index=False)
    print(f"\nSeverity gen results → {out_path}")

    # ── Experiment 2: Noise crossover ─────────────────────────────────────────
    df_cross, crossover_noise = run_noise_crossover(df_real, le)

    # ── Table 8 and Figure 8 ──────────────────────────────────────────────────
    table8 = save_table8(gen_df)
    plot_fig8(gen_df, class_names, crossover_noise)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SEVERITY GENERALISATION SUMMARY")
    print("=" * 65)

    real_only_df = gen_df[
        (gen_df["condition"] == "real_only") & (gen_df["has_test_data"] == True)
    ]
    if not real_only_df.empty:
        top_gaps = real_only_df.dropna(subset=["gap"]).nlargest(5, "gap")
        print("  Top-5 class × model combinations with largest generalisation gap:")
        for _, r in top_gaps.iterrows():
            print(f"    {r['model']:<16} {r['class']:<25} gap={r['gap']:.4f}  "
                  f"(cv_f1={r['cv_f1']:.4f} → gen_f1={r['gen_f1']:.4f})")

    if crossover_noise is not None:
        print(f"\n  SVM vs XGBoost crossover: {crossover_noise*100:.0f}% noise level")
        print(f"  → Recommendation: use SVM when feature noise exceeds "
              f"{crossover_noise*100:.0f}% of std")
    else:
        print("\n  No crossover found (XGBoost dominates up to 30% noise)")

    print()

    # VAE impact on generalisation
    if len(rows_b) > 0:
        vae_df = gen_df[gen_df["condition"] == "real+vae"].dropna(subset=["gap"])
        real_df2 = gen_df[gen_df["condition"] == "real_only"].dropna(subset=["gap"])
        avg_gap_real = real_df2["gap"].mean()
        avg_gap_vae  = vae_df["gap"].mean()
        improvement  = avg_gap_real - avg_gap_vae
        print(f"  VAE impact on generalisation gap:")
        print(f"    Real-only avg gap : {avg_gap_real:.4f}")
        print(f"    VAE-aug  avg gap  : {avg_gap_vae:.4f}")
        print(f"    Reduction         : {improvement:+.4f}")
        if improvement > 0.01:
            print("    → VAE augmentation reduces severity generalisation gap.")
        else:
            print("    → VAE augmentation does not significantly reduce severity gap.")
    print("=" * 65)


if __name__ == "__main__":
    main()
