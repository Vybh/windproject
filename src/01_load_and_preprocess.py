"""
01_load_and_preprocess.py
--------------------------
Load raw CWRU bearing fault .mat files, extract vibration signals,
engineer time/frequency domain features, and save to CSV.

Usage:
    python src/01_load_and_preprocess.py

Outputs:
    data/processed/features.csv
"""

import os
import warnings
import re
import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import kurtosis, skew
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
WINDOW_SIZE = 1024
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
SAMPLING_RATE = 12000  # Hz (CWRU standard drive-end channel)

# ── CWRU file-number → fault label/severity mapping ───────────────────────────
# Each group of 4 files corresponds to motor loads 0HP, 1HP, 2HP, 3HP
FILE_MAPPING = {
    # (start_num, end_num): (fault_type, severity)
    (97,  100): ("normal",     "none"),
    (105, 108): ("inner_race", "0.007"),
    (169, 172): ("inner_race", "0.014"),
    (209, 212): ("inner_race", "0.021"),
    (118, 121): ("ball",       "0.007"),
    (185, 188): ("ball",       "0.014"),
    (222, 225): ("ball",       "0.021"),
    (130, 133): ("outer_race", "0.007"),
    (197, 200): ("outer_race", "0.014"),
    (234, 237): ("outer_race", "0.021"),
}

# Build flat lookup: file_number → (label, severity, motor_load_hp)
FILE_LOOKUP: dict[int, tuple[str, str, int]] = {}
for (start, end), (label, severity) in FILE_MAPPING.items():
    for i, num in enumerate(range(start, end + 1)):
        FILE_LOOKUP[num] = (label, severity, i)  # i = 0,1,2,3 → 0HP–3HP


# ── Feature extraction helpers ────────────────────────────────────────────────

def extract_file_number(filename: str) -> int | None:
    """Parse the numeric identifier from a CWRU .mat filename, e.g. '97.mat' → 97."""
    match = re.search(r"(\d+)", Path(filename).stem)
    return int(match.group(1)) if match else None


def detect_de_channel(mat_data: dict) -> np.ndarray | None:
    """
    Find the drive-end vibration channel across all known CWRU key variants.
    Returns a 1-D float64 array or None if not found.
    """
    candidates = [k for k in mat_data if "DE_time" in k and not k.startswith("__")]
    if not candidates:
        return None
    key = candidates[0]
    signal = mat_data[key].flatten().astype(np.float64)
    return signal


def time_domain_features(window: np.ndarray) -> dict:
    """Compute 9 time-domain statistical features from a signal window."""
    rms = np.sqrt(np.mean(window ** 2))
    peak = np.max(np.abs(window))
    return {
        "mean":         np.mean(window),
        "std":          np.std(window),
        "rms":          rms,
        "peak":         peak,
        "peak2peak":    np.ptp(window),
        "crest_factor": peak / (rms + 1e-12),
        "kurtosis":     float(kurtosis(window, fisher=True)),
        "skewness":     float(skew(window)),
        "shape_factor": rms / (np.mean(np.abs(window)) + 1e-12),
    }


def freq_domain_features(window: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """
    Compute 5 frequency-domain features via FFT.
    Band boundaries (Hz): low <1000, mid 1000–3000, high >3000.
    """
    n = len(window)
    fft_mag = np.abs(np.fft.rfft(window * np.hanning(n))) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total_energy = np.sum(fft_mag ** 2) + 1e-12
    spectral_centroid = np.sum(freqs * fft_mag ** 2) / total_energy

    dominant_idx = np.argmax(fft_mag)
    dominant_frequency = freqs[dominant_idx]

    low_mask  = freqs < 1000
    mid_mask  = (freqs >= 1000) & (freqs < 3000)
    high_mask = freqs >= 3000

    return {
        "spectral_centroid":  spectral_centroid,
        "dominant_frequency": dominant_frequency,
        "low_band_energy":    np.sum(fft_mag[low_mask]  ** 2) / total_energy,
        "mid_band_energy":    np.sum(fft_mag[mid_mask]  ** 2) / total_energy,
        "high_band_energy":   np.sum(fft_mag[high_mask] ** 2) / total_energy,
    }


def extract_features_from_signal(
    signal: np.ndarray,
    label: str,
    severity: str,
    motor_load: int,
    window_size: int = WINDOW_SIZE,
) -> list[dict]:
    """Segment signal into non-overlapping windows and extract all features."""
    n_windows = len(signal) // window_size
    rows = []
    for i in range(n_windows):
        w = signal[i * window_size : (i + 1) * window_size]
        feat = {}
        feat.update(time_domain_features(w))
        feat.update(freq_domain_features(w))
        feat["motor_load"] = motor_load
        feat["label"]      = label
        feat["severity"]   = severity
        rows.append(feat)
    return rows


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(DATA_RAW_DIR.glob("*.mat"))
    if not mat_files:
        print(f"[ERROR] No .mat files found in {DATA_RAW_DIR.resolve()}.")
        print("        Download CWRU bearing data and place .mat files there.")
        return

    print(f"Found {len(mat_files)} .mat file(s) in {DATA_RAW_DIR}\n")

    all_rows = []
    skipped = []

    for mat_path in mat_files:
        file_num = extract_file_number(mat_path.name)
        if file_num is None:
            warnings.warn(f"Cannot parse file number from '{mat_path.name}' — skipping.")
            skipped.append(mat_path.name)
            continue

        if file_num not in FILE_LOOKUP:
            warnings.warn(f"File number {file_num} ('{mat_path.name}') not in mapping — skipping.")
            skipped.append(mat_path.name)
            continue

        label, severity, motor_load = FILE_LOOKUP[file_num]

        try:
            mat_data = scipy.io.loadmat(str(mat_path))
        except Exception as exc:
            warnings.warn(f"Failed to load '{mat_path.name}': {exc} — skipping.")
            skipped.append(mat_path.name)
            continue

        signal = detect_de_channel(mat_data)
        if signal is None:
            warnings.warn(f"No DE_time channel found in '{mat_path.name}' — skipping.")
            skipped.append(mat_path.name)
            continue

        rows = extract_features_from_signal(signal, label, severity, motor_load)
        all_rows.extend(rows)
        print(f"  [{file_num:>3}] {mat_path.name:<30}  label={label:<12} sev={severity}  "
              f"load={motor_load}HP  windows={len(rows)}")

    if not all_rows:
        print("\n[ERROR] No features extracted. Check your data/raw/ directory.")
        return

    # ── Build DataFrame with canonical column order ────────────────────────────
    feature_cols = [
        "mean", "std", "rms", "peak", "peak2peak",
        "crest_factor", "kurtosis", "skewness", "shape_factor",
        "spectral_centroid", "dominant_frequency",
        "low_band_energy", "mid_band_energy", "high_band_energy",
        "motor_load",
    ]
    meta_cols = ["label", "severity"]
    df = pd.DataFrame(all_rows)[feature_cols + meta_cols]

    out_path = DATA_PROCESSED_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} windows → {out_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: Window counts per class and severity")
    print("=" * 60)
    summary = (
        df.groupby(["label", "severity"])
        .size()
        .reset_index(name="windows")
        .sort_values(["label", "severity"])
    )
    print(summary.to_string(index=False))
    print("=" * 60)
    print(f"Total windows : {len(df):,}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Classes       : {sorted(df['label'].unique())}")
    if skipped:
        print(f"\nSkipped files ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()
