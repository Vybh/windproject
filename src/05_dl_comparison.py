"""
05_dl_comparison.py
--------------------
Train deep learning models on VAE-augmented dataset and compare
against classical ML results from Script 02.

Critical rule — train on real+synthetic, test on real only:
  StratifiedKFold on real rows only; all synthetic rows added to every
  training fold. Test folds contain only is_synthetic==False rows.

Inputs:
    data/processed/features_augmented.csv
    data/processed/model_results_expanded.csv
    data/processed/noise_robustness_expanded.csv

Outputs (data/processed/):
    dl_results.csv
    autoencoder_results.csv

Outputs (figures/):
    cnn_confusion_matrix.png
    cnn_training_history.png
    dl_accuracy_comparison.png
    dl_per_class_f1.png
    dl_noise_robustness.png
    autoencoder_violin.png
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

    # ── Final summary ─────────────────────────────────────────────────────────
    width = 65
    print("\n" + "=" * width)
    print("FINAL SUMMARY")
    print("=" * width)
    print(f"  CNN Mean Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  CNN Best Fold Acc  : {best_fold_acc:.4f}  (fold {best_fold_idx + 1})")
    xgb_acc = classical_acc.get("XGBoost", float("nan"))
    delta   = mean_acc - xgb_acc
    sign    = "+" if delta >= 0 else ""
    print(f"  CNN vs XGBoost     : {mean_acc:.4f} vs {xgb_acc:.4f}  ({sign}{delta:+.4f})")
    print()
    print("  Autoencoder Results:")
    for _, row in ae_df.iterrows():
        cls = row["class"]
        if cls == "normal":
            print(f"    {cls:<22}  FPR = {row['false_positive_rate']:.4f}")
        else:
            print(f"    {cls:<22}  DR  = {row['detection_rate']:.4f}")
    print("=" * width)


if __name__ == "__main__":
    main()
