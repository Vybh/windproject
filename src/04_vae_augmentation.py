"""
04_vae_augmentation.py
-----------------------
Variational Autoencoder (VAE) for class-conditional synthetic sample
generation.  Trains one VAE per class (with its own StandardScaler),
then up-samples every minority class until all 7 classes reach the
majority count of 2,477 windows.

Architecture (per-class VAE on 15-feature normalised vectors):
  Encoder : Dense(64, relu) → Dense(32, relu) → [mu, log_var] (latent=16)
  Reparam : z = mu + ε·exp(0.5·log_var)
  Decoder : Dense(32, relu) → Dense(64, relu) → Dense(15, linear)
  Loss    : MSE reconstruction + β·KL  (β = 0.001, beta-VAE style)

Usage:
    python src/04_vae_augmentation.py

Inputs:
    data/processed/features_expanded.csv

Outputs:
    data/processed/features_augmented.csv     (real + synthetic, is_synthetic col)
    models/vae_class_{label}/encoder.keras    (one Keras model per class)
    models/vae_class_{label}/decoder.keras
    figures/vae_class_distribution.png
    figures/vae_tsne.png
    figures/vae_training_loss.png
    figures/vae_feature_kde.png
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
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"]      = str(RANDOM_STATE)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
tf.random.set_seed(RANDOM_STATE)

from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# ── Config ─────────────────────────────────────────────────────────────────────
LATENT_DIM     = 16
BETA           = 0.001   # KL weight (low → prioritise reconstruction)
EPOCHS         = 200
BATCH_SIZE     = 32
LR             = 1e-3
MAJORITY_COUNT = 2477    # FE_outer_race count — target for all classes

DATA_PROC_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
FIGURES_DIR   = Path("figures")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mean", "std", "rms", "peak", "peak2peak",
    "crest_factor", "kurtosis", "skewness", "shape_factor",
    "spectral_centroid", "dominant_frequency",
    "low_band_energy", "mid_band_energy", "high_band_energy",
    "motor_load",
]
N_FEATURES = len(FEATURE_COLS)


# ── VAE building blocks ────────────────────────────────────────────────────────

class Sampling(layers.Layer):
    """Reparameterisation: z = mu + ε · exp(0.5 · log_var)."""
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu), seed=RANDOM_STATE)
        return mu + eps * tf.exp(0.5 * log_var)


def build_vae(n_features: int = N_FEATURES, latent_dim: int = LATENT_DIM):
    """Build encoder and decoder sub-models. Returns (encoder, decoder)."""
    # Encoder
    enc_input = keras.Input(shape=(n_features,), name="encoder_input")
    x         = layers.Dense(64, activation="relu")(enc_input)
    x         = layers.Dense(32, activation="relu")(x)
    mu        = layers.Dense(latent_dim, name="mu")(x)
    log_var   = layers.Dense(latent_dim, name="log_var")(x)
    z         = Sampling(name="z")([mu, log_var])
    encoder   = Model(enc_input, [mu, log_var, z], name="encoder")

    # Decoder
    dec_input  = keras.Input(shape=(latent_dim,), name="decoder_input")
    x          = layers.Dense(32, activation="relu")(dec_input)
    x          = layers.Dense(64, activation="relu")(x)
    dec_output = layers.Dense(n_features, activation="linear", name="decoder_output")(x)
    decoder    = Model(dec_input, dec_output, name="decoder")

    return encoder, decoder


class VAE(Model):
    """Keras 3-compatible VAE: custom train_step replaces add_loss/add_metric."""
    def __init__(self, encoder, decoder, beta=BETA, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x, training=True)
            reconstruction = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - reconstruction), axis=-1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mu) - tf.exp(log_var)
            )
            total_loss = recon_loss + self.beta * kl_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


def train_class_vae(X_norm: np.ndarray, class_name: str) -> tuple:
    """
    Train a VAE on a single class's normalised feature matrix.
    Returns (encoder, decoder, history_dict).
    """
    encoder, decoder = build_vae()
    vae = VAE(encoder, decoder, beta=BETA, name="vae")
    vae.compile(optimizer=optimizers.Adam(learning_rate=LR))

    print(f"  Training VAE for '{class_name}'  ({len(X_norm):,} samples) ...")
    hist = vae.fit(
        X_norm, X_norm,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=True,
    )
    history = {
        "total_loss": hist.history["loss"],
        "recon_loss": hist.history.get("recon_loss", []),
        "kl_loss":    hist.history.get("kl_loss",    []),
    }
    return encoder, decoder, history


def generate_samples(
    decoder: Model,
    n_samples: int,
    latent_dim: int = LATENT_DIM,
) -> np.ndarray:
    """Sample z ~ N(0,I), decode to normalised feature space."""
    rng = np.random.default_rng(RANDOM_STATE)
    z   = rng.normal(size=(n_samples, latent_dim)).astype(np.float32)
    return decoder.predict(z, verbose=0)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    original_counts: dict,
    final_counts: dict,
    save_path: Path,
) -> None:
    """Side-by-side bar chart: class distribution before vs after augmentation."""
    classes   = sorted(original_counts.keys())
    orig_vals = [original_counts[c] for c in classes]
    aug_vals  = [final_counts[c]    for c in classes]

    x         = np.arange(len(classes))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - bar_width / 2, orig_vals, bar_width,
                   label="Original", color="#42A5F5", edgecolor="black", lw=0.6)
    bars2 = ax.bar(x + bar_width / 2, aug_vals,  bar_width,
                   label="After VAE Augmentation", color="#66BB6A", edgecolor="black", lw=0.6)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("Class Distribution: Before vs After VAE Augmentation",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne(
    df_aug: pd.DataFrame,
    class_names: list,
    save_path: Path,
    tsne_cap: int = 4000,
) -> None:
    """
    t-SNE with one subplot per class.
    Each subplot shows real samples (blue) and synthetic samples (orange).
    """
    print("  Running t-SNE (may take a moment) ...")

    # Global scaler for t-SNE normalisation only
    global_scaler = StandardScaler()
    X_all = global_scaler.fit_transform(
        df_aug[FEATURE_COLS].fillna(0).values.astype(np.float64)
    )
    labels_arr = df_aug["label"].values
    is_syn_arr = df_aug["is_synthetic"].values.astype(bool)

    # Subsample for speed
    rng = np.random.default_rng(RANDOM_STATE)
    n   = len(df_aug)
    idx = rng.choice(n, min(n, tsne_cap), replace=False)

    X_2d = TSNE(
        n_components=2, random_state=RANDOM_STATE,
        perplexity=30, max_iter=1000,
    ).fit_transform(X_all[idx])

    labs = labels_arr[idx]
    syns = is_syn_arr[idx]

    n_cls = len(class_names)
    ncols = 4
    nrows = (n_cls + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()
    fig.suptitle("t-SNE: Real vs Synthetic Samples per Class",
                 fontsize=13, fontweight="bold")

    for i, cls in enumerate(class_names):
        ax        = axes_flat[i]
        mask_real = (labs == cls) & (~syns)
        mask_syn  = (labs == cls) & (syns)

        if mask_real.any():
            ax.scatter(
                X_2d[mask_real, 0], X_2d[mask_real, 1],
                c="blue", s=12, alpha=0.55, label="Real",
            )
        if mask_syn.any():
            ax.scatter(
                X_2d[mask_syn, 0], X_2d[mask_syn, 1],
                c="orange", s=9, alpha=0.45, label="Synthetic",
            )
        ax.set_title(cls, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, markerscale=1.5)
        ax.set_xlabel("Dim 1", fontsize=7)
        ax.set_ylabel("Dim 2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(linestyle="--", alpha=0.3)

    for j in range(n_cls, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_loss(
    history_dict: dict,
    save_path: Path,
) -> None:
    """One subplot per class: total loss, reconstruction loss, KL loss over epochs."""
    classes   = sorted(history_dict.keys())
    n_cls     = len(classes)
    ncols     = min(4, n_cls)
    nrows     = (n_cls + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten() if n_cls > 1 else [axes]
    fig.suptitle("VAE Training Loss Curves per Class",
                 fontsize=13, fontweight="bold")

    for i, cls in enumerate(classes):
        ax    = axes_flat[i]
        hist  = history_dict[cls]
        epoch = np.arange(1, len(hist["total_loss"]) + 1)

        ax.plot(epoch, hist["total_loss"], label="Total",
                color="#333333", lw=1.5)
        if hist["recon_loss"]:
            ax.plot(epoch, hist["recon_loss"], label="Reconstruction",
                    color="#2196F3", lw=1.2, ls="--")
        if hist["kl_loss"]:
            ax.plot(epoch, hist["kl_loss"], label="KL",
                    color="#FF7043", lw=1.2, ls=":")

        ax.set_title(cls, fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss",  fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=7)

    for j in range(n_cls, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_kde(
    df_aug: pd.DataFrame,
    class_names: list,
    save_path: Path,
) -> None:
    """
    KDE comparison of real vs synthetic samples for three representative features
    (rms, kurtosis, spectral_centroid), one row per class.
    Classes with no synthetic samples show only the real distribution.
    """
    rep_features = ["rms", "kurtosis", "spectral_centroid"]
    n_cls  = len(class_names)
    n_feat = len(rep_features)

    fig, axes = plt.subplots(
        n_cls, n_feat,
        figsize=(5 * n_feat, 3 * n_cls),
        constrained_layout=True,
    )
    if n_cls == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Feature Distribution: Real vs Synthetic per Class (KDE)",
        fontsize=13, fontweight="bold",
    )

    for r, cls in enumerate(class_names):
        df_real = df_aug[(df_aug["label"] == cls) & (~df_aug["is_synthetic"].astype(bool))]
        df_syn  = df_aug[(df_aug["label"] == cls) & (df_aug["is_synthetic"].astype(bool))]

        for c, feat in enumerate(rep_features):
            ax = axes[r][c]

            real_vals = df_real[feat].dropna()
            syn_vals  = df_syn[feat].dropna()

            if len(real_vals) > 1:
                sns.kdeplot(real_vals, ax=ax, label="Real",
                            color="blue", fill=True, alpha=0.4, linewidth=1.2)
            if len(syn_vals) > 1:
                sns.kdeplot(syn_vals, ax=ax, label="Synthetic",
                            color="orange", fill=True, alpha=0.4, linewidth=1.2)

            if r == 0:
                ax.set_title(feat, fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(cls, fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    feat_path = DATA_PROC_DIR / "features_expanded.csv"
    if not feat_path.exists():
        print(f"[ERROR] {feat_path} not found. Run save_features_expanded.py first.")
        return

    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df):,} windows from {feat_path}")

    class_names     = sorted(df["label"].unique())
    class_indices   = {cls: np.where(df["label"].values == cls)[0] for cls in class_names}
    original_counts = {cls: len(idx) for cls, idx in class_indices.items()}

    print(f"Classes: {class_names}\n")
    print(f"Majority count target: {MAJORITY_COUNT:,}  (FE_outer_race)\n")

    # ── Per-class VAE training and synthesis ───────────────────────────────────
    synthetic_rows:    list[dict]        = []
    history_per_class: dict[str, dict]   = {}
    class_scalers:     dict[str, StandardScaler] = {}

    for cls in class_names:
        idx       = class_indices[cls]
        X_raw     = df[FEATURE_COLS].values[idx].astype(np.float64)
        n_needed  = MAJORITY_COUNT - len(idx)

        # Fit scaler on this class's data only
        scaler    = StandardScaler()
        X_norm    = scaler.fit_transform(X_raw).astype(np.float32)
        class_scalers[cls] = scaler

        # Train VAE
        encoder, decoder, history = train_class_vae(X_norm, cls)
        history_per_class[cls]    = history

        # Save encoder + decoder
        safe_name = cls.replace(" ", "_")
        save_dir  = MODELS_DIR / f"vae_class_{safe_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        encoder.save(str(save_dir / "encoder.keras"))
        decoder.save(str(save_dir / "decoder.keras"))
        print(f"  Saved model → {save_dir}")

        # Generate synthetic samples if this class is below majority
        if n_needed > 0:
            synth_norm = generate_samples(decoder, n_needed)
            # Inverse-transform using this class's own scaler
            synth_orig = scaler.inverse_transform(synth_norm)

            for row_vals in synth_orig:
                row = dict(zip(FEATURE_COLS, row_vals.tolist()))
                row["label"] = cls
                if cls.startswith("DE"):
                    row["bearing_location"] = "DE"
                elif cls.startswith("FE"):
                    row["bearing_location"] = "FE"
                else:
                    row["bearing_location"] = ""
                row["fault_size"]   = ""
                row["is_synthetic"] = True
                synthetic_rows.append(row)

            print(f"  Generated {n_needed:,} synthetic samples for '{cls}'\n")
        else:
            print(f"  '{cls}' is already majority — no synthesis needed.\n")

    # ── Build augmented DataFrame ──────────────────────────────────────────────
    real_cols = FEATURE_COLS + ["label", "bearing_location", "fault_size"]
    df_real   = df[real_cols].copy()
    df_real["is_synthetic"] = False

    df_syn = pd.DataFrame(synthetic_rows) if synthetic_rows else pd.DataFrame(columns=df_real.columns)
    df_aug = pd.concat([df_real, df_syn], ignore_index=True)

    out_aug = DATA_PROC_DIR / "features_augmented.csv"
    df_aug.to_csv(out_aug, index=False)
    print(f"Saved augmented dataset → {out_aug}  ({len(df_aug):,} total rows)\n")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("Generating figures ...")

    final_counts = df_aug.groupby("label").size().to_dict()

    plot_class_distribution(
        original_counts, final_counts,
        FIGURES_DIR / "vae_class_distribution.png",
    )

    plot_tsne(
        df_aug, class_names,
        FIGURES_DIR / "vae_tsne.png",
    )

    plot_training_loss(
        history_per_class,
        FIGURES_DIR / "vae_training_loss.png",
    )

    plot_feature_kde(
        df_aug, class_names,
        FIGURES_DIR / "vae_feature_kde.png",
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    width = 72
    print()
    print("=" * width)
    print("VAE AUGMENTATION SUMMARY")
    print("=" * width)
    print(f"{'Class':<25}  {'Original':>10}  {'Synthetic Added':>16}  {'Final':>8}")
    print("-" * width)

    all_final_ok = True
    for cls in class_names:
        orig  = original_counts[cls]
        final = final_counts.get(cls, orig)
        added = final - orig
        check = "✓" if final == MAJORITY_COUNT else "✗"
        if final != MAJORITY_COUNT:
            all_final_ok = False
        print(f"{cls:<25}  {orig:>10,}  {added:>16,}  {final:>8,}  {check}")

    print("-" * width)
    total_orig = sum(original_counts.values())
    total_syn  = len(df_syn)
    total_aug  = len(df_aug)
    print(f"{'TOTAL':<25}  {total_orig:>10,}  {total_syn:>16,}  {total_aug:>8,}")
    print("=" * width)

    print()
    if all_final_ok:
        print(f"  All 7 classes have exactly {MAJORITY_COUNT:,} samples.")
    else:
        print("  WARNING: Not all classes reached the target count.")
    print(f"  features_augmented.csv saved to {out_aug}")
    print(f"  'is_synthetic' column present: {'is_synthetic' in df_aug.columns}")
    print()
    print(f"  Figures saved to {FIGURES_DIR}/")
    print(f"  VAE models  saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
