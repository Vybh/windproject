"""
07_vae_ablation.py
-------------------
VAE Augmentation Ablation Study (GAP 2).

Trains RF, XGBoost, SVM, KNN, and 1D-CNN under two conditions and compares:
  (a) Real data only, class-imbalanced  (features_expanded.csv, is_synthetic==False)
  (b) Real + VAE synthetic, balanced    (features_augmented.csv)

5-fold stratified CV; for CNN, test folds are always real-only rows, and
synthetic rows are appended to every training fold (same protocol as Script 05).

Outputs
-------
data/processed/vae_ablation_results.csv
    columns: condition, model, cv_accuracy, macro_f1
paper/tables/table7_vae_ablation.csv / .txt
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RANDOM_STATE = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

import tensorflow as tf
tf.random.set_seed(RANDOM_STATE)
tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/processed")
TABLES_DIR = Path("paper/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mean", "std", "rms", "peak", "peak2peak",
    "crest_factor", "kurtosis", "skewness", "shape_factor",
    "spectral_centroid", "dominant_frequency",
    "low_band_energy", "mid_band_energy", "high_band_energy",
    "motor_load",
]
N_FEATURES = len(FEATURE_COLS)
N_FOLDS    = 5


# ── Model builders ─────────────────────────────────────────────────────────────

def build_classical():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
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


def build_cnn(n_classes: int) -> keras.Model:
    """Same 1D-CNN as Script 05: (15, 1) tabular feature input."""
    inp = keras.Input(shape=(N_FEATURES, 1))
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


# ── CV helpers ─────────────────────────────────────────────────────────────────

def run_classical_cv(
    model_name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    condition: str,
) -> dict:
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=-1)
    acc    = accuracy_score(y, y_pred)
    f1     = f1_score(y, y_pred, average="macro", zero_division=0)
    print(f"    [{condition}] {model_name:<16}  acc={acc:.4f}  macro_f1={f1:.4f}")
    return {"condition": condition, "model": model_name, "cv_accuracy": acc, "macro_f1": f1}


def run_cnn_cv(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    n_classes: int,
    condition: str,
) -> dict:
    """5-fold CV; test folds are always real-only."""
    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []
    fold_f1s  = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real)):
        X_tr_r, y_tr_r = X_real[train_idx], y_real[train_idx]
        X_te,   y_te   = X_real[test_idx],  y_real[test_idx]

        if condition == "real_plus_synthetic" and len(X_synth) > 0:
            X_train = np.concatenate([X_tr_r, X_synth], axis=0)
            y_train = np.concatenate([y_tr_r, y_synth], axis=0)
        else:
            X_train, y_train = X_tr_r, y_tr_r

        shuf    = np.random.permutation(len(X_train))
        X_train, y_train = X_train[shuf], y_train[shuf]

        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_te_s    = scaler.transform(X_te)
        X_train_r = X_train_s.reshape(-1, N_FEATURES, 1)
        X_te_r    = X_te_s.reshape(-1, N_FEATURES, 1)

        model = build_cnn(n_classes)
        es    = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        )
        model.fit(
            X_train_r, y_train,
            validation_data=(X_te_r, y_te),
            epochs=80, batch_size=32, callbacks=[es], verbose=0,
        )
        _, acc = model.evaluate(X_te_r, y_te, verbose=0)
        y_pred = np.argmax(model.predict(X_te_r, verbose=0), axis=1)
        f1     = f1_score(y_te, y_pred, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"      fold {fold_idx+1}: acc={acc:.4f}  macro_f1={f1:.4f}", flush=True)

    mean_acc = float(np.mean(fold_accs))
    mean_f1  = float(np.mean(fold_f1s))
    print(f"    [{condition}] 1D-CNN          acc={mean_acc:.4f}  macro_f1={mean_f1:.4f}")
    return {"condition": condition, "model": "1D-CNN", "cv_accuracy": mean_acc, "macro_f1": mean_f1}


# ── Table writer ──────────────────────────────────────────────────────────────

def save_table7(df: pd.DataFrame):
    """Write Table 7 as both CSV and formatted TXT for the paper."""
    # Pivot: rows = model, cols = (condition × metric)
    pivot = df.pivot_table(
        index="model",
        columns="condition",
        values=["cv_accuracy", "macro_f1"],
        aggfunc="first",
    )
    # Flatten multi-index columns
    pivot.columns = [f"{m}_{c}" for m, c in pivot.columns]
    pivot = pivot.reset_index()

    # Rename for readability
    col_rename = {
        "cv_accuracy_real_only":           "Real-Only Acc",
        "cv_accuracy_real_plus_synthetic": "VAE-Aug Acc",
        "macro_f1_real_only":              "Real-Only F1",
        "macro_f1_real_plus_synthetic":    "VAE-Aug F1",
    }
    pivot = pivot.rename(columns=col_rename)
    pivot = pivot.rename(columns={"model": "Model"})

    # Sort by canonical order
    order = ["Random Forest", "XGBoost", "SVM", "KNN", "1D-CNN"]
    pivot["sort"] = pivot["Model"].apply(lambda x: order.index(x) if x in order else 99)
    pivot = pivot.sort_values("sort").drop(columns="sort").reset_index(drop=True)

    # Add improvement column
    if "Real-Only Acc" in pivot.columns and "VAE-Aug Acc" in pivot.columns:
        pivot["Acc Δ"] = (pivot["VAE-Aug Acc"] - pivot["Real-Only Acc"]).map("{:+.4f}".format)
    if "Real-Only F1" in pivot.columns and "VAE-Aug F1" in pivot.columns:
        pivot["F1 Δ"] = (pivot["VAE-Aug F1"] - pivot["Real-Only F1"]).map("{:+.4f}".format)

    # Round float columns
    for col in ["Real-Only Acc", "VAE-Aug Acc", "Real-Only F1", "VAE-Aug F1"]:
        if col in pivot.columns:
            pivot[col] = pivot[col].map("{:.4f}".format)

    csv_path = TABLES_DIR / "table7_vae_ablation.csv"
    txt_path = TABLES_DIR / "table7_vae_ablation.txt"
    pivot.to_csv(csv_path, index=False)

    col_widths = [max(len(str(c)), pivot[c].astype(str).map(len).max()) + 2
                  for c in pivot.columns]
    header = "  ".join(str(c).ljust(w) for c, w in zip(pivot.columns, col_widths))
    sep    = "  ".join("-" * w for w in col_widths)
    rows   = [
        "TABLE 7 — VAE Augmentation Ablation Study",
        "Condition (a): Real data only (class-imbalanced)",
        "Condition (b): Real + VAE synthetic (class-balanced)",
        "Evaluation: 5-fold stratified CV; test folds are always real-only",
        "",
        header, sep,
    ]
    for _, r in pivot.iterrows():
        rows.append("  ".join(str(v).ljust(w) for v, w in zip(r, col_widths)))
    rows.append("")
    rows.append("Note: Positive Δ indicates VAE augmentation improved the metric.")
    rows.append("If improvement is minimal (<0.005), VAE augmentation primarily")
    rows.append("serves a data-balancing function rather than a performance boost.")
    with open(txt_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nTable 7 saved → {csv_path}")
    print(f"              → {txt_path}")
    return pivot


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load datasets ──────────────────────────────────────────────────────────
    feat_path = DATA_DIR / "features_expanded.csv"
    aug_path  = DATA_DIR / "features_augmented.csv"

    if not feat_path.exists():
        print(f"[ERROR] {feat_path} not found. Run 01_load_and_preprocess.py first.")
        return
    if not aug_path.exists():
        print(f"[ERROR] {aug_path} not found. Run 04_vae_augmentation.py first.")
        return

    df_real  = pd.read_csv(feat_path)
    df_aug   = pd.read_csv(aug_path)

    # Real rows only
    if "is_synthetic" in df_real.columns:
        df_real = df_real[df_real["is_synthetic"] == False].reset_index(drop=True)
    if "is_synthetic" in df_aug.columns:
        df_synth = df_aug[df_aug["is_synthetic"] == True].reset_index(drop=True)
        df_real2 = df_aug[df_aug["is_synthetic"] == False].reset_index(drop=True)
    else:
        df_synth = pd.DataFrame()
        df_real2 = df_aug

    print(f"Real rows      : {len(df_real):,}")
    print(f"Synthetic rows : {len(df_synth):,}")
    print()

    le = LabelEncoder()
    le.fit(df_real["label"].values)
    class_names = list(le.classes_)
    n_classes   = len(class_names)
    print(f"Classes ({n_classes}): {class_names}\n")

    X_real  = df_real[FEATURE_COLS].values.astype(np.float64)
    y_real  = le.transform(df_real["label"].values)

    # Clean synthetic data (same as Script 05)
    if len(df_synth) > 0:
        df_synth = df_synth.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        for col in FEATURE_COLS:
            col_min = df_real[col].min()
            col_max = df_real[col].max()
            df_synth[col] = df_synth[col].clip(lower=col_min, upper=col_max)
        X_synth = df_synth[FEATURE_COLS].values.astype(np.float64)
        y_synth = le.transform(df_synth["label"].values)
    else:
        X_synth = np.zeros((0, N_FEATURES))
        y_synth = np.array([], dtype=int)

    # ── Classical models ───────────────────────────────────────────────────────
    all_rows = []

    print("=" * 65)
    print("CLASSICAL MODELS — Condition (a): Real data only")
    print("=" * 65)
    for name, model in build_classical().items():
        row = run_classical_cv(name, model, X_real, y_real, n_classes, "real_only")
        all_rows.append(row)

    print("\n" + "=" * 65)
    print("CLASSICAL MODELS — Condition (b): Real + VAE synthetic")
    print("=" * 65)
    # For condition b, X_full = real + synthetic (balanced), labels combined
    if len(X_synth) > 0:
        X_full = np.concatenate([X_real, X_synth], axis=0)
        y_full = np.concatenate([y_real, y_synth], axis=0)
        # CV must test on real-only: use a custom CV loop
        skf_b = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for name, _ in build_classical().items():
            fold_accs = []
            fold_f1s  = []
            for train_idx, test_idx in skf_b.split(X_real, y_real):
                X_tr_r, y_tr_r = X_real[train_idx], y_real[train_idx]
                X_te,   y_te   = X_real[test_idx],  y_real[test_idx]
                X_tr = np.concatenate([X_tr_r, X_synth], axis=0)
                y_tr = np.concatenate([y_tr_r, y_synth], axis=0)
                clf  = build_classical()[name]
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                fold_accs.append(accuracy_score(y_te, y_pred))
                fold_f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
            acc  = float(np.mean(fold_accs))
            f1   = float(np.mean(fold_f1s))
            row  = {"condition": "real_plus_synthetic", "model": name,
                    "cv_accuracy": acc, "macro_f1": f1}
            all_rows.append(row)
            print(f"    [real_plus_synthetic] {name:<16}  acc={acc:.4f}  macro_f1={f1:.4f}")
    else:
        print("  [WARNING] No synthetic data available — copying real-only results.")
        for row in [r for r in all_rows if r["condition"] == "real_only"]:
            all_rows.append({**row, "condition": "real_plus_synthetic"})

    # ── 1D-CNN ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1D-CNN — Condition (a): Real data only")
    print("=" * 65)
    row_a = run_cnn_cv(
        X_real, y_real, np.zeros((0, N_FEATURES)), np.array([], dtype=int),
        n_classes, "real_only",
    )
    all_rows.append(row_a)

    print("\n" + "=" * 65)
    print("1D-CNN — Condition (b): Real + VAE synthetic")
    print("=" * 65)
    row_b = run_cnn_cv(
        X_real, y_real, X_synth, y_synth, n_classes, "real_plus_synthetic",
    )
    all_rows.append(row_b)

    # ── Save results ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    out_path   = DATA_DIR / "vae_ablation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VAE ABLATION SUMMARY")
    print("=" * 72)
    print(f"  {'Model':<16} {'Real Acc':>10} {'VAE Acc':>10} {'Δ Acc':>8}  "
          f"{'Real F1':>10} {'VAE F1':>10} {'Δ F1':>8}")
    print("  " + "-" * 70)
    order = ["Random Forest", "XGBoost", "SVM", "KNN", "1D-CNN"]
    for model_name in order:
        a_row = next((r for r in all_rows
                      if r["model"] == model_name and r["condition"] == "real_only"), None)
        b_row = next((r for r in all_rows
                      if r["model"] == model_name and r["condition"] == "real_plus_synthetic"), None)
        if a_row is None or b_row is None:
            continue
        da = b_row["cv_accuracy"] - a_row["cv_accuracy"]
        df1 = b_row["macro_f1"]   - a_row["macro_f1"]
        print(f"  {model_name:<16} {a_row['cv_accuracy']:>10.4f} {b_row['cv_accuracy']:>10.4f} "
              f"{da:>+8.4f}  {a_row['macro_f1']:>10.4f} {b_row['macro_f1']:>10.4f} {df1:>+8.4f}")

    print()
    print("Interpretation:")
    print("  • Positive Δ confirms VAE augmentation improved accuracy/F1.")
    print("  • Small Δ (<0.005) means VAE primarily balanced the dataset but")
    print("    did not provide statistically meaningful accuracy gains.")
    print("  • Claim must be softened accordingly in the paper.")
    print("=" * 72)

    # ── Table 7 ──────────────────────────────────────────────────────────────
    save_table7(results_df)


if __name__ == "__main__":
    main()
