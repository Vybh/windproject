"""
05_dl_comparison.py
--------------------
Train deep learning models on VAE-augmented dataset and compare
against classical ML results from Script 02.

GAP 1 IMPLEMENTED: Two CNN experiments are run:
  (a) Feature-based CNN (15-feature tabular vector reshaped to (15,1))
      This shows tree ensembles beat CNNs on pre-engineered tabular features.
  (b) Raw-signal CNN (1024-point raw vibration windows, no feature engineering)
      This is the real DL vs ML question: can end-to-end DL match XGBoost?

Critical rule — train on real+synthetic, test on real only:
  StratifiedKFold on real rows only; all synthetic rows added to every
  training fold. Test folds contain only is_synthetic==False rows.

NOTE: The feature-based CNN result (a) does NOT generalise to the claim
"deep learning is inferior" — it is specifically inferior when applied to
pre-summarised tabular features. Experiment (b) addresses the real question.

Inputs:
    data/processed/features_augmented.csv
    data/processed/model_results_expanded.csv
    data/processed/noise_robustness_expanded.csv
    data/raw/DE_12k/*.mat  (for raw-signal CNN)
    data/raw/FE_12k/*.mat  (for raw-signal CNN)
    data/raw/normal/*.mat  (for raw-signal CNN)

Outputs (data/processed/):
    dl_results.csv
    raw_cnn_results.csv
    autoencoder_results.csv

Outputs (figures/):
    cnn_confusion_matrix.png
    cnn_training_history.png
    dl_accuracy_comparison.png
    dl_per_class_f1.png
    dl_noise_robustness.png
    autoencoder_violin.png
    raw_cnn_vs_xgboost.png
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

# ── Seeds ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

import tensorflow as tf
tf.random.set_seed(RANDOM_STATE)

# Force CPU — TensorFlow Metal (Apple Silicon) has silent gradient bugs
# with BatchNormalization that prevent Conv1D models from learning.
tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from tensorflow import keras
from tensorflow.keras import layers

# ── Paths & constants ────────────────────────────────────────────────────────
DATA_DIR    = Path("data/processed")
FIGURES_DIR = Path("figures")
MODELS_DIR  = Path("models")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mean", "std", "rms", "peak", "peak2peak",
    "crest_factor", "kurtosis", "skewness", "shape_factor",
    "spectral_centroid", "dominant_frequency",
    "low_band_energy", "mid_band_energy", "high_band_energy",
    "motor_load",
]
N_FEATURES   = len(FEATURE_COLS)   # 15
N_CLASSES    = 7
N_FOLDS      = 5
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]


# ── Model builders ───────────────────────────────────────────────────────────

def build_cnn(n_classes: int = N_CLASSES) -> keras.Model:
    """1D-CNN as specified."""
    inp = keras.Input(shape=(N_FEATURES, 1))
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_autoencoder(n_features: int = N_FEATURES) -> keras.Model:
    """Dense autoencoder: 15→8→4→8→15."""
    inp = keras.Input(shape=(n_features,))
    x   = layers.Dense(8, activation="relu")(inp)
    x   = layers.Dense(4, activation="relu")(x)
    x   = layers.Dense(8, activation="relu")(x)
    out = layers.Dense(n_features, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def build_classical_models() -> dict:
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def add_noise(X: np.ndarray, noise_pct: float, rng: np.random.Generator) -> np.ndarray:
    if noise_pct == 0.0:
        return X
    stds  = X.std(axis=0, keepdims=True)
    noise = rng.normal(0, noise_pct * stds, size=X.shape)
    return X + noise


def reconstruction_error(autoencoder: keras.Model, X: np.ndarray) -> np.ndarray:
    X_pred = autoencoder.predict(X, verbose=0)
    return np.mean((X - X_pred) ** 2, axis=1)


def _pad_seq(seq: list, target_len: int) -> list:
    """Extend sequence to target_len by repeating the last value."""
    return seq + [seq[-1]] * (target_len - len(seq))


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_cnn_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax, vmin=0, vmax=1, annot_kws={"size": 9},
    )
    ax.set_title("1D-CNN Confusion Matrix (Summed Across 5 Folds, Normalised)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cnn_training_history(histories: list, save_path: Path):
    max_len = max(len(h["loss"]) for h in histories)

    loss_mat     = np.array([_pad_seq(h["loss"],         max_len) for h in histories])
    val_loss_mat = np.array([_pad_seq(h["val_loss"],     max_len) for h in histories])
    acc_mat      = np.array([_pad_seq(h["accuracy"],     max_len) for h in histories])
    val_acc_mat  = np.array([_pad_seq(h["val_accuracy"], max_len) for h in histories])

    epochs = np.arange(1, max_len + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mat, val_mat, ax, metric in [
        (loss_mat, val_loss_mat, ax1, "Loss"),
        (acc_mat,  val_acc_mat,  ax2, "Accuracy"),
    ]:
        m_tr  = mat.mean(axis=0);     s_tr  = mat.std(axis=0)
        m_val = val_mat.mean(axis=0); s_val = val_mat.std(axis=0)

        ax.plot(epochs, m_tr,  color="#2196F3", linewidth=2, label="Train (mean)")
        ax.fill_between(epochs, m_tr - s_tr, m_tr + s_tr, alpha=0.2, color="#2196F3")
        ax.plot(epochs, m_val, color="#FF5722", linewidth=2, label="Val (mean)")
        ax.fill_between(epochs, m_val - s_val, m_val + s_val, alpha=0.2, color="#FF5722")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"CNN {metric} — Mean ± Std (5 Folds)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparative_accuracy(classical_acc: dict, cnn_acc: float, save_path: Path):
    all_acc     = {**classical_acc, "1D-CNN": cnn_acc}
    model_names = list(all_acc.keys())
    accs        = [all_acc[m] for m in model_names]
    palette     = sns.color_palette("tab10", len(model_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, accs, color=palette,
                  edgecolor="black", linewidth=0.8, width=0.55)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f"{acc:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Accuracy (5-Fold CV)", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Accuracy Comparison: Classical ML vs 1D-CNN", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparative_f1(f1_data: dict, class_names: list, save_path: Path):
    model_names = list(f1_data.keys())
    n_classes   = len(class_names)
    n_models    = len(model_names)
    bar_width   = 0.14
    x           = np.arange(n_classes)
    palette     = sns.color_palette("tab10", n_models)
    offsets     = np.linspace(
        -(n_models - 1) * bar_width / 2,
         (n_models - 1) * bar_width / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, name in enumerate(model_names):
        ax.bar(
            x + offsets[i], f1_data[name], bar_width,
            label=name, color=palette[i], edgecolor="black", linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_title("Per-Class F1 Score: Classical ML vs 1D-CNN", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_noise_robustness(noise_df: pd.DataFrame, save_path: Path):
    models  = noise_df["model"].unique()
    markers = ["o", "s", "^", "D", "v"]
    palette = sns.color_palette("tab10", len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        sub = noise_df[noise_df["model"] == model].sort_values("noise_pct")
        ax.plot(
            sub["noise_pct"] * 100, sub["accuracy"],
            marker=markers[i % len(markers)],
            linewidth=2, color=palette[i], label=model,
        )
    ax.set_xlabel("Gaussian Noise Level (% of feature std)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Noise Robustness: Classical ML vs 1D-CNN", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_autoencoder_violin(
    recon_err: np.ndarray,
    y_labels: np.ndarray,
    class_names: list,
    threshold: float,
    save_path: Path,
):
    rows = [
        {"class": class_names[cls_idx], "recon_error": err}
        for cls_idx in range(len(class_names))
        for err in recon_err[y_labels == cls_idx]
    ]
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=plot_df, x="class", y="recon_error",
        order=class_names, palette="tab10",
        inner="quartile", ax=ax,
    )
    ax.axhline(
        threshold, color="red", linestyle="--", linewidth=1.8,
        label=f"Threshold = {threshold:.5f}",
    )
    ax.set_xlabel("Fault Class", fontsize=12)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_title("Autoencoder Reconstruction Error by Class", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    # ── Load & encode ─────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "features_augmented.csv")
    print(f"Loaded {len(df):,} rows from features_augmented.csv")

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    class_names = list(le.classes_)
    n_classes   = len(class_names)
    print(f"Classes ({n_classes}): {class_names}\n")

    real_df  = df[df["is_synthetic"] == False].reset_index(drop=True)
    synth_df = df[df["is_synthetic"] == True].reset_index(drop=True)
    print(f"Real rows : {len(real_df):,}")
    print(f"Synthetic : {len(synth_df):,}\n")

    # ── Clean synthetic data ──────────────────────────────────────────────────
    # VAE can decode out-of-range latent points, producing NaN values and
    # physically impossible feature values (negative std, rms, energy, etc.).
    # NaN anywhere in a training batch silently kills gradient flow.
    n_before = len(synth_df)
    synth_df = synth_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    n_after = len(synth_df)
    print(f"Synthetic NaN rows dropped: {n_before - n_after} "
          f"({n_before} → {n_after})")

    # Clip remaining synthetic features to the real data's observed [min, max].
    # This removes physically impossible values without discarding entire rows.
    for col in FEATURE_COLS:
        col_min = real_df[col].min()
        col_max = real_df[col].max()
        synth_df[col] = synth_df[col].clip(lower=col_min, upper=col_max)

    X_real  = real_df[FEATURE_COLS].values.astype(np.float64)
    y_real  = real_df["label_enc"].values
    X_synth = synth_df[FEATURE_COLS].values.astype(np.float64)
    y_synth = synth_df["label_enc"].values

    # ── 1D-CNN — 5-Fold CV ────────────────────────────────────────────────────
    print("=" * 65)
    print("1D-CNN — 5-Fold Cross-Validation (test on real only)")
    print("=" * 65)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accs      = []
    fold_val_accs  = []
    fold_cms       = []
    fold_histories = []
    fold_f1s       = []
    best_fold_idx  = -1
    best_val_acc   = -1.0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}", flush=True)

        X_tr_real = X_real[train_idx]
        y_tr_real = y_real[train_idx]
        X_te      = X_real[test_idx]
        y_te      = y_real[test_idx]

        # Combine: real train fold + ALL synthetic
        X_train = np.concatenate([X_tr_real, X_synth], axis=0)
        y_train = np.concatenate([y_tr_real, y_synth], axis=0)
        shuf    = np.random.permutation(len(X_train))
        X_train, y_train = X_train[shuf], y_train[shuf]

        # Scale — fit on training fold only
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_te_s    = scaler.transform(X_te)

        # Reshape for Conv1D: (n, 15, 1)
        X_train_r = X_train_s.reshape(-1, N_FEATURES, 1)
        X_te_r    = X_te_s.reshape(-1, N_FEATURES, 1)

        model = build_cnn(n_classes)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True,
        )
        history = model.fit(
            X_train_r, y_train,
            validation_data=(X_te_r, y_te),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        fold_histories.append(history.history)

        # Evaluate
        _, test_acc = model.evaluate(X_te_r, y_te, verbose=0)
        y_pred = np.argmax(model.predict(X_te_r, verbose=0), axis=1)
        cm  = confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))
        f1  = f1_score(y_te, y_pred, average=None,
                       labels=list(range(n_classes)), zero_division=0)

        fold_accs.append(test_acc)
        fold_cms.append(cm)
        fold_f1s.append(f1)

        max_val_acc = max(history.history["val_accuracy"])
        fold_val_accs.append(max_val_acc)
        print(f"    test_acc={test_acc:.4f}  best_val_acc={max_val_acc:.4f}"
              f"  epochs_run={len(history.history['loss'])}", flush=True)

        if max_val_acc > best_val_acc:
            best_val_acc  = max_val_acc
            best_fold_idx = fold_idx
            model.save(str(MODELS_DIR / "cnn_best_fold.keras"))
            print(f"    --> New best fold ({fold_idx + 1}), model saved.", flush=True)

    mean_acc      = float(np.mean(fold_accs))
    std_acc       = float(np.std(fold_accs))
    best_fold_acc = fold_accs[best_fold_idx]
    avg_cm        = np.sum(fold_cms, axis=0)
    avg_f1        = np.mean(fold_f1s, axis=0)

    print(f"\nCNN CV summary:")
    print(f"  Mean accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Best fold ({best_fold_idx + 1}): {best_fold_acc:.4f}")

    # Save dl_results.csv
    dl_rows = [{"fold": i + 1, "test_accuracy": a} for i, a in enumerate(fold_accs)]
    dl_rows.append({"fold": "mean", "test_accuracy": mean_acc})
    dl_rows.append({"fold": "std",  "test_accuracy": std_acc})
    pd.DataFrame(dl_rows).to_csv(DATA_DIR / "dl_results.csv", index=False)
    print(f"  Saved: {DATA_DIR / 'dl_results.csv'}")

    # ── Autoencoder — anomaly detection (standalone) ──────────────────────────
    print("\n" + "=" * 65)
    print("Autoencoder Anomaly Detector (standalone)")
    print("=" * 65)

    normal_enc  = int(le.transform(["normal"])[0])
    normal_mask = y_real == normal_enc
    X_normal    = X_real[normal_mask]

    # 80% train, 20% for threshold calibration
    X_ae_train, X_ae_thresh = train_test_split(
        X_normal, test_size=0.20, random_state=RANDOM_STATE,
    )
    scaler_ae     = StandardScaler()
    X_ae_train_s  = scaler_ae.fit_transform(X_ae_train)
    X_ae_thresh_s = scaler_ae.transform(X_ae_thresh)

    ae = build_autoencoder()
    ae.fit(
        X_ae_train_s, X_ae_train_s,
        epochs=100, batch_size=32, verbose=0,
    )
    print("  Autoencoder trained on normal class (real) samples.")

    # Threshold: mean + 2*std of reconstruction error on held-out 20%
    err_thresh = reconstruction_error(ae, X_ae_thresh_s)
    threshold  = float(err_thresh.mean() + 2.0 * err_thresh.std())
    print(f"  Threshold = {threshold:.6f}  "
          f"(mean={err_thresh.mean():.6f}, 2*std={2*err_thresh.std():.6f})")

    # Evaluate on ALL real samples
    X_real_s      = scaler_ae.transform(X_real)
    recon_err_all = reconstruction_error(ae, X_real_s)
    is_anomaly    = recon_err_all > threshold

    ae_rows = []
    print("\n  Per-class results:")
    for cls_idx, cls_name in enumerate(class_names):
        mask      = y_real == cls_idx
        n_total   = int(mask.sum())
        n_flagged = int(is_anomaly[mask].sum())
        rate      = n_flagged / n_total if n_total > 0 else 0.0

        if cls_name == "normal":
            ae_rows.append({
                "class": cls_name,
                "n_samples": n_total,
                "n_detected": n_flagged,
                "detection_rate": rate,
                "false_positive_rate": rate,
            })
            print(f"    {cls_name:<22}  FPR = {rate:.4f}  ({n_flagged}/{n_total})")
        else:
            ae_rows.append({
                "class": cls_name,
                "n_samples": n_total,
                "n_detected": n_flagged,
                "detection_rate": rate,
                "false_positive_rate": None,
            })
            print(f"    {cls_name:<22}  DR  = {rate:.4f}  ({n_flagged}/{n_total})")

    ae_df = pd.DataFrame(ae_rows)
    ae_df.to_csv(DATA_DIR / "autoencoder_results.csv", index=False)
    print(f"\n  Saved: {DATA_DIR / 'autoencoder_results.csv'}")

    # ── Classical per-class F1 (on real rows, for comparison chart) ───────────
    print("\n" + "=" * 65)
    print("Classical Models — Per-Class F1 on Real Data (for F1 chart)")
    print("=" * 65)

    skf_cl     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    classical_f1 = {}
    for name, clf in build_classical_models().items():
        print(f"  {name} ...", end="", flush=True)
        y_pred_cv = cross_val_predict(clf, X_real, y_real, cv=skf_cl, n_jobs=-1)
        f1 = f1_score(y_real, y_pred_cv, average=None,
                      labels=list(range(n_classes)), zero_division=0)
        classical_f1[name] = f1
        print(f" done  (macro F1 = {f1.mean():.4f})")

    # Load classical overall accuracies
    cls_acc_df  = pd.read_csv(DATA_DIR / "model_results_expanded.csv")
    classical_acc = dict(zip(cls_acc_df["model"], cls_acc_df["cv_accuracy"]))

    # ── CNN noise robustness ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CNN Noise Robustness")
    print("=" * 65)

    rng = np.random.default_rng(RANDOM_STATE)

    # 80/20 stratified split on real rows (mirrors Script 02 approach)
    X_tr_r, X_te_nr, y_tr_r, y_te_nr = train_test_split(
        X_real, y_real, test_size=0.20, stratify=y_real, random_state=RANDOM_STATE,
    )
    # Add synthetic to training split
    X_tr_nr = np.concatenate([X_tr_r, X_synth], axis=0)
    y_tr_nr = np.concatenate([y_tr_r, y_synth], axis=0)
    shuf    = np.random.permutation(len(X_tr_nr))
    X_tr_nr, y_tr_nr = X_tr_nr[shuf], y_tr_nr[shuf]

    scaler_nr  = StandardScaler()
    X_tr_nr_s  = scaler_nr.fit_transform(X_tr_nr)
    X_te_nr_s  = scaler_nr.transform(X_te_nr)

    cnn_nr = build_cnn(n_classes)
    es_nr  = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True,
    )
    cnn_nr.fit(
        X_tr_nr_s.reshape(-1, N_FEATURES, 1), y_tr_nr,
        validation_data=(X_te_nr_s.reshape(-1, N_FEATURES, 1), y_te_nr),
        epochs=100, batch_size=32, callbacks=[es_nr], verbose=0,
    )
    print("  CNN noise-robustness model trained.")

    cnn_noise_rows = []
    for noise_pct in NOISE_LEVELS:
        X_noisy = add_noise(X_te_nr_s, noise_pct, rng)
        y_pred_nr = np.argmax(
            cnn_nr.predict(X_noisy.reshape(-1, N_FEATURES, 1), verbose=0), axis=1
        )
        acc_n = accuracy_score(y_te_nr, y_pred_nr)
        cnn_noise_rows.append({"model": "1D-CNN", "noise_pct": noise_pct, "accuracy": acc_n})
        print(f"    noise={noise_pct:.2f}  acc={acc_n:.4f}")

    noise_df_classical = pd.read_csv(DATA_DIR / "noise_robustness_expanded.csv")
    noise_df_all = pd.concat(
        [noise_df_classical, pd.DataFrame(cnn_noise_rows)], ignore_index=True,
    )

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Generating figures ...")
    print("=" * 65)

    plot_cnn_confusion_matrix(
        avg_cm, class_names,
        FIGURES_DIR / "cnn_confusion_matrix.png",
    )
    plot_cnn_training_history(
        fold_histories,
        FIGURES_DIR / "cnn_training_history.png",
    )
    plot_comparative_accuracy(
        classical_acc, mean_acc,
        FIGURES_DIR / "dl_accuracy_comparison.png",
    )
    plot_comparative_f1(
        {**classical_f1, "1D-CNN": avg_f1}, class_names,
        FIGURES_DIR / "dl_per_class_f1.png",
    )
    plot_noise_robustness(
        noise_df_all,
        FIGURES_DIR / "dl_noise_robustness.png",
    )
    plot_autoencoder_violin(
        recon_err_all, y_real, class_names, threshold,
        FIGURES_DIR / "autoencoder_violin.png",
    )

    # ── Raw-signal CNN experiment (GAP 1) ────────────────────────────────────
    xgb_acc_for_raw = classical_acc.get("XGBoost", float("nan"))
    raw_cnn_result  = run_raw_cnn(xgb_acc_for_raw)

    # ── Final summary ─────────────────────────────────────────────────────────
    width = 65
    print("\n" + "=" * width)
    print("FINAL SUMMARY")
    print("=" * width)
    print(f"  Feature-based CNN  : {mean_acc:.4f} ± {std_acc:.4f}  (input=15-feat tabular)")
    print(f"  CNN Best Fold Acc  : {best_fold_acc:.4f}  (fold {best_fold_idx + 1})")
    xgb_acc_v = classical_acc.get("XGBoost", float("nan"))
    delta   = mean_acc - xgb_acc_v
    sign    = "+" if delta >= 0 else ""
    print(f"  Feature CNN vs XGB : {mean_acc:.4f} vs {xgb_acc_v:.4f}  ({sign}{delta:+.4f})")
    if raw_cnn_result:
        raw_acc = raw_cnn_result["cv_accuracy"]
        delta_r = raw_acc - xgb_acc_v
        print(f"  Raw-signal CNN     : {raw_acc:.4f}  (input=1024-pt raw window)")
        print(f"  Raw CNN vs XGB     : {raw_acc:.4f} vs {xgb_acc_v:.4f}  ({delta_r:+.4f})")
    print()
    print("  NOTE: Feature-based CNN on pre-engineered tabular features is NOT")
    print("  a fair DL vs ML comparison. See raw-signal CNN result above for")
    print("  the genuine end-to-end deep learning comparison.")
    print()
    print("  Autoencoder Results:")
    for _, row in ae_df.iterrows():
        cls = row["class"]
        if cls == "normal":
            print(f"    {cls:<22}  FPR = {row['false_positive_rate']:.4f}")
        else:
            print(f"    {cls:<22}  DR  = {row['detection_rate']:.4f}")
    print("=" * width)


# ── Raw-signal CNN (GAP 1) ───────────────────────────────────────────────────
#
# Loads raw 1024-point vibration windows directly from .mat files.
# No feature engineering — the CNN learns directly from the signal.
# Architecture: (1024, 1) input → Conv1D stack → GlobalAvgPool → Dense.
# This is the fair DL comparison: end-to-end learning vs XGBoost+feature-eng.

import re as _re
import scipy.io as _sio

# CWRU file → (label, fault_size) lookups (subset: ball faults + inner/outer + normal)
_DE_MAP: dict[int, tuple[str, str]] = {
    # normal (DE channel)
    97: ("normal","none"), 98: ("normal","none"),
    99: ("normal","none"), 100: ("normal","none"),
    # DE inner race
    105:("DE_inner_race","0.007"),106:("DE_inner_race","0.007"),
    107:("DE_inner_race","0.007"),108:("DE_inner_race","0.007"),
    169:("DE_inner_race","0.014"),170:("DE_inner_race","0.014"),
    171:("DE_inner_race","0.014"),172:("DE_inner_race","0.014"),
    209:("DE_inner_race","0.021"),210:("DE_inner_race","0.021"),
    211:("DE_inner_race","0.021"),212:("DE_inner_race","0.021"),
    # DE ball
    118:("DE_ball","0.007"),119:("DE_ball","0.007"),
    120:("DE_ball","0.007"),121:("DE_ball","0.007"),
    185:("DE_ball","0.014"),186:("DE_ball","0.014"),
    187:("DE_ball","0.014"),188:("DE_ball","0.014"),
    222:("DE_ball","0.021"),223:("DE_ball","0.021"),
    224:("DE_ball","0.021"),225:("DE_ball","0.021"),
    # DE outer race
    130:("DE_outer_race","0.007"),131:("DE_outer_race","0.007"),
    132:("DE_outer_race","0.007"),133:("DE_outer_race","0.007"),
    197:("DE_outer_race","0.014"),198:("DE_outer_race","0.014"),
    199:("DE_outer_race","0.014"),200:("DE_outer_race","0.014"),
    234:("DE_outer_race","0.021"),235:("DE_outer_race","0.021"),
    236:("DE_outer_race","0.021"),237:("DE_outer_race","0.021"),
}
_FE_MAP: dict[int, tuple[str, str]] = {
    # FE inner race
    278:("FE_inner_race","0.007"),279:("FE_inner_race","0.007"),
    280:("FE_inner_race","0.007"),281:("FE_inner_race","0.007"),
    282:("FE_inner_race","0.014"),283:("FE_inner_race","0.014"),
    284:("FE_inner_race","0.014"),285:("FE_inner_race","0.014"),
    286:("FE_inner_race","0.021"),287:("FE_inner_race","0.021"),
    288:("FE_inner_race","0.021"),289:("FE_inner_race","0.021"),
    # FE ball
    290:("FE_ball","0.007"),291:("FE_ball","0.007"),
    292:("FE_ball","0.007"),293:("FE_ball","0.007"),
    # FE outer race
    294:("FE_outer_race","0.007"),295:("FE_outer_race","0.007"),
    296:("FE_outer_race","0.007"),297:("FE_outer_race","0.007"),
    298:("FE_outer_race","0.014"),299:("FE_outer_race","0.014"),
    300:("FE_outer_race","0.014"),301:("FE_outer_race","0.014"),
    302:("FE_outer_race","0.021"),305:("FE_outer_race","0.021"),
    306:("FE_outer_race","0.021"),307:("FE_outer_race","0.021"),
}

RAW_WINDOW = 1024


def _find_mat_key(mat: dict, sub: str):
    return next((k for k in mat if not k.startswith("__") and sub in k), None)


def load_raw_windows() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load raw 1024-sample vibration windows from CWRU .mat files.
    Returns (X, y, class_names) where X.shape = (n_windows, 1024).
    This is the input for the raw-signal CNN — no feature engineering.
    """
    base    = Path("data/raw")
    all_X   = []
    all_y   = []
    labels  = []

    def _process_dir(mat_dir: Path, file_map: dict, channel: str, is_normal: bool):
        for mat_path in sorted(mat_dir.glob("*.mat")):
            try:
                num = int(mat_path.stem)
            except ValueError:
                continue
            if num not in file_map:
                # Normal dir: try both channels
                if is_normal:
                    pass
                else:
                    continue

            if is_normal:
                label, fault_size = "normal", "none"
                channels = [("DE_time", "DE"), ("FE_time", "FE")]
            else:
                if num not in file_map:
                    continue
                label, fault_size = file_map[num]
                channels = [(channel, "")]

            try:
                mat = _sio.loadmat(str(mat_path))
            except Exception:
                continue

            for ch, _bl in channels:
                sig_key = _find_mat_key(mat, ch)
                if sig_key is None:
                    continue
                signal = mat[sig_key].flatten().astype(np.float32)
                n_win  = len(signal) // RAW_WINDOW
                for i in range(n_win):
                    w = signal[i * RAW_WINDOW : (i + 1) * RAW_WINDOW]
                    all_X.append(w)
                    all_y.append(label)

    # Normal baseline
    norm_dir = base / "normal"
    if norm_dir.exists():
        for mat_path in sorted(norm_dir.glob("*.mat")):
            try:
                mat = _sio.loadmat(str(mat_path))
            except Exception:
                continue
            for ch in ("DE_time", "FE_time"):
                sig_key = _find_mat_key(mat, ch)
                if sig_key is None:
                    continue
                signal = mat[sig_key].flatten().astype(np.float32)
                n_win  = len(signal) // RAW_WINDOW
                for i in range(n_win):
                    w = signal[i * RAW_WINDOW : (i + 1) * RAW_WINDOW]
                    all_X.append(w)
                    all_y.append("normal")

    # DE fault files
    de_dir = base / "DE_12k"
    if de_dir.exists():
        for mat_path in sorted(de_dir.glob("*.mat")):
            try:
                num = int(mat_path.stem)
            except ValueError:
                continue
            if num not in _DE_MAP:
                continue
            label, _ = _DE_MAP[num]
            try:
                mat = _sio.loadmat(str(mat_path))
            except Exception:
                continue
            sig_key = _find_mat_key(mat, "DE_time")
            if sig_key is None:
                continue
            signal = mat[sig_key].flatten().astype(np.float32)
            n_win  = len(signal) // RAW_WINDOW
            for i in range(n_win):
                w = signal[i * RAW_WINDOW : (i + 1) * RAW_WINDOW]
                all_X.append(w)
                all_y.append(label)

    # FE fault files
    fe_dir = base / "FE_12k"
    if fe_dir.exists():
        for mat_path in sorted(fe_dir.glob("*.mat")):
            try:
                num = int(mat_path.stem)
            except ValueError:
                continue
            if num not in _FE_MAP:
                continue
            label, _ = _FE_MAP[num]
            try:
                mat = _sio.loadmat(str(mat_path))
            except Exception:
                continue
            sig_key = _find_mat_key(mat, "FE_time")
            if sig_key is None:
                continue
            signal = mat[sig_key].flatten().astype(np.float32)
            n_win  = len(signal) // RAW_WINDOW
            for i in range(n_win):
                w = signal[i * RAW_WINDOW : (i + 1) * RAW_WINDOW]
                all_X.append(w)
                all_y.append(label)

    if not all_X:
        return np.array([]), np.array([]), []

    X = np.stack(all_X)   # (n, 1024)
    le_raw = LabelEncoder()
    y = le_raw.fit_transform(all_y)
    class_names = list(le_raw.classes_)
    return X, y, class_names


def build_raw_cnn(n_classes: int, window_size: int = RAW_WINDOW) -> keras.Model:
    """
    Raw-signal 1D-CNN: input (window_size, 1), learns directly from vibration.
    Architecture designed for vibration fault classification at 12 kHz.
    """
    inp = keras.Input(shape=(window_size, 1))
    # Block 1: wide kernel to capture low-frequency modulation
    x = layers.Conv1D(32, kernel_size=64, strides=2, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    # Block 2
    x = layers.Conv1D(64, kernel_size=32, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    # Block 3
    x = layers.Conv1D(128, kernel_size=16, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def run_raw_cnn(xgb_acc: float) -> dict:
    """
    Train raw-signal CNN with 5-fold CV. Compare to XGBoost (feature-based).
    Returns result dict.
    """
    print("\n" + "=" * 65)
    print("RAW-SIGNAL CNN — End-to-End Deep Learning (GAP 1)")
    print("=" * 65)
    print("  Loading raw 1024-point vibration windows from .mat files ...")

    X_raw, y_raw, cn = load_raw_windows()
    if len(X_raw) == 0:
        print("  [SKIP] Raw .mat files not accessible. Skipping raw-signal CNN.")
        return {}

    n_classes_raw = len(cn)
    print(f"  Loaded {len(X_raw):,} windows  Classes ({n_classes_raw}): {cn}")
    print(f"  Class distribution:")
    for cls_idx, cls_name in enumerate(cn):
        n = int((y_raw == cls_idx).sum())
        print(f"    {cls_name:<25}: {n:>5,}")

    # Normalise raw windows to zero-mean, unit-variance per window
    mu  = X_raw.mean(axis=1, keepdims=True)
    sig = X_raw.std(axis=1, keepdims=True) + 1e-8
    X_norm = ((X_raw - mu) / sig).astype(np.float32)

    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_norm, y_raw)):
        X_tr = X_norm[tr_idx, :, np.newaxis]
        y_tr = y_raw[tr_idx]
        X_te = X_norm[te_idx, :, np.newaxis]
        y_te = y_raw[te_idx]

        model = build_raw_cnn(n_classes_raw)
        es    = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True,
        )
        model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=60, batch_size=64, callbacks=[es], verbose=0,
        )
        _, acc = model.evaluate(X_te, y_te, verbose=0)
        fold_accs.append(acc)
        print(f"    Fold {fold_idx+1}: acc={acc:.4f}", flush=True)

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    print(f"\n  Raw-signal CNN: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  XGBoost (feature-based): {xgb_acc:.4f}")
    delta = mean_acc - xgb_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  Gap (Raw CNN − XGBoost): {sign}{delta:.4f}")

    result = {
        "model": "Raw-Signal CNN",
        "cv_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "n_classes": n_classes_raw,
        "n_windows": len(X_raw),
        "notes": f"input=(1024,1), no feature engineering, {N_FOLDS}-fold CV",
    }

    # Save
    rows = [{"fold": i+1, "test_accuracy": a} for i, a in enumerate(fold_accs)]
    rows.append({"fold": "mean", "test_accuracy": mean_acc})
    rows.append({"fold": "std",  "test_accuracy": std_acc})
    pd.DataFrame(rows).to_csv(DATA_DIR / "raw_cnn_results.csv", index=False)
    print(f"  Saved → {DATA_DIR / 'raw_cnn_results.csv'}")

    # Figure: raw CNN vs XGBoost vs feature-based CNN comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    models  = ["XGBoost\n(Feature-based)", "Feature-based CNN\n(15→(15,1))",
               "Raw-Signal CNN\n(1024-point window)"]
    # We'll load feature-based CNN acc later; use placeholder if not set
    accs    = [xgb_acc, float("nan"), mean_acc]

    # Load feature-based CNN from dl_results.csv
    dl_csv = DATA_DIR / "dl_results.csv"
    if dl_csv.exists():
        dl_df = pd.read_csv(dl_csv)
        feat_cnn_row = dl_df[dl_df["fold"] == "mean"]
        if not feat_cnn_row.empty:
            accs[1] = float(feat_cnn_row["test_accuracy"].values[0])

    palette = ["#42A5F5", "#AB47BC", "#FF7043"]
    bars    = ax.bar(models, accs, color=palette, edgecolor="black", linewidth=0.8, width=0.5)
    for bar, acc in zip(bars, accs):
        if not np.isnan(acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{acc:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (5-Fold CV)", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "GAP 1: Feature-based vs Raw-Signal CNN vs XGBoost\n"
        "(Feature CNN on tabular input ≠ fair DL comparison)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    # Add annotation distinguishing the two CNN experiments
    ax.annotate(
        "Feature-based CNN applies Conv1D\nto pre-summarised features (not raw signals)",
        xy=(0.55, accs[1] - 0.05 if not np.isnan(accs[1]) else 0.85),
        xytext=(0.2, 0.6),
        xycoords=("data", "data"),
        textcoords=("data", "data"),
        fontsize=8, color="#AB47BC",
        arrowprops=dict(arrowstyle="->", color="#AB47BC"),
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_cnn_vs_xgboost.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {FIGURES_DIR / 'raw_cnn_vs_xgboost.png'}")

    # Print interpretation
    print("\n  Interpretation:")
    print("  The feature-based CNN operates on 15 pre-engineered statistics —")
    print("  this is NOT a fair test of deep learning vs classical ML. The")
    print("  raw-signal CNN above is the correct comparison: can end-to-end DL")
    print("  match XGBoost when features are NOT pre-computed?")
    if mean_acc >= xgb_acc - 0.02:
        print(f"  → Raw-signal CNN ({mean_acc:.4f}) is competitive with XGBoost ({xgb_acc:.4f}).")
        print("    The gap between feature-based CNN and XGBoost does NOT mean DL is inferior.")
    else:
        print(f"  → Even raw-signal CNN ({mean_acc:.4f}) is below XGBoost ({xgb_acc:.4f}).")
        print("    This confirms XGBoost + feature engineering outperforms end-to-end DL")
        print("    on this dataset size. More data may be needed to close the gap.")

    return result


if __name__ == "__main__":
    main()
