"""
01_load_and_preprocess.py  (V2 — Expanded)
------------------------------------------
Load raw CWRU bearing fault .mat files from the reorganised directory
structure, extract vibration signals (both Drive-End and Fan-End channels),
engineer 15 time/frequency domain features, and save two CSVs:

  data/processed/features.csv          ← original flat-raw output (backward compat)
  data/processed/features_expanded.csv ← 7-class, multi-location expanded output

Directory layout expected for the expanded run:
  data/raw/
    DE_12k/   ← 12 kHz Drive-End fault files
    FE_12k/   ← 12 kHz Fan-End fault files
    normal/   ← normal baseline files (contain both DE_time + FE_time)

If DE_12k / FE_12k / normal/ are absent, the script falls back to the
original flat data/raw/*.mat scan and only writes features.csv.

Usage:
    python src/01_load_and_preprocess.py

Outputs:
    data/processed/features.csv          (flat-mode or always)
    data/processed/features_expanded.csv (expanded-mode, 7 classes)
"""

import os
import warnings
import re
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
from scipy.stats import kurtosis, skew

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
WINDOW_SIZE    = 1024
SAMPLING_RATE  = 12_000          # Hz — CWRU 12k sample rate
DATA_RAW_DIR   = Path("data/raw")
DATA_PROC_DIR  = Path("data/processed")

# ── CWRU file-number → (fault_type, fault_size_str) mappings ──────────────────
#
# Drive-End 12k (channel: DE_time)
# Each group of 4 = motor loads 0 HP, 1 HP, 2 HP, 3 HP
DE_FILE_MAPPING = {
    # Normal baseline — the normal files live in data/raw/normal/ but are also
    # listed here so both DE and FE channel extractions pick them up.
    (97,  100): ("normal",     "none"),
    # Inner race
    (105, 108): ("inner_race", "0.007"),
    (169, 172): ("inner_race", "0.014"),
    (209, 212): ("inner_race", "0.021"),
    # Ball
    (118, 121): ("ball",       "0.007"),
    (185, 188): ("ball",       "0.014"),
    (222, 225): ("ball",       "0.021"),
    # Outer race
    (130, 133): ("outer_race", "0.007"),
    (197, 200): ("outer_race", "0.014"),
    (234, 237): ("outer_race", "0.021"),
    # 0.028" severity — update file numbers if you have these files
    # (3001, 3004): ("inner_race", "0.028"),
    # (3005, 3008): ("ball",       "0.028"),
    # (3009, 3012): ("outer_race", "0.028"),
}

# Fan-End 12k (channel: FE_time)
# File numbers from the CWRU public dataset — verify against your download.
# NOTE: Normal files (97-100) already listed in DE_FILE_MAPPING; the loader
#       will re-use them for FE channel extraction when scanning normal/.
FE_FILE_MAPPING = {
    # Inner race
    (278, 281): ("inner_race", "0.007"),
    (282, 285): ("inner_race", "0.014"),
    (286, 289): ("inner_race", "0.021"),
    (290, 293): ("inner_race", "0.028"),
    # Ball
    (270, 273): ("ball",       "0.007"),
    (274, 277): ("ball",       "0.014"),
    # Outer race
    (294, 297): ("outer_race", "0.007"),
    (298, 301): ("outer_race", "0.014"),
    (302, 305): ("outer_race", "0.021"),
    (306, 309): ("outer_race", "0.028"),
}

# ── Build flat lookups: file_number → (fault_type, fault_size, motor_load) ────
def _build_lookup(mapping: dict) -> dict[int, tuple[str, str, int]]:
    lut: dict[int, tuple[str, str, int]] = {}
    for (start, end), (label, severity) in mapping.items():
        for load_idx, num in enumerate(range(start, end + 1)):
            lut[num] = (label, severity, load_idx)  # load_idx 0-3 → 0-3 HP
    return lut

DE_LOOKUP = _build_lookup(DE_FILE_MAPPING)
FE_LOOKUP = _build_lookup(FE_FILE_MAPPING)

# Normal file numbers share the same files for both DE and FE
NORMAL_FILE_LOOKUP: dict[int, tuple[str, str, int]] = {}
for (start, end), (label, severity) in DE_FILE_MAPPING.items():
    if label == "normal":
        for load_idx, num in enumerate(range(start, end + 1)):
            NORMAL_FILE_LOOKUP[num] = (label, severity, load_idx)


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_file_number(filename: str) -> int | None:
    """Parse the leading integer from a CWRU filename, e.g. '105.mat' → 105."""
    match = re.search(r"(\d+)", Path(filename).stem)
    return int(match.group(1)) if match else None


def detect_channel(mat_data: dict, preferred: str) -> tuple[np.ndarray | None, str | None]:
    """
    Return (signal, channel_key) for the preferred channel ('DE_time' or 'FE_time').
    Falls back to the other channel if the preferred one is absent.
    """
    candidates = {
        k: mat_data[k] for k in mat_data
        if not k.startswith("__") and isinstance(mat_data[k], np.ndarray)
    }
    # Try preferred
    for key in candidates:
        if preferred in key:
            return candidates[key].flatten().astype(np.float64), key
    # Try fallback
    fallback = "FE_time" if preferred == "DE_time" else "DE_time"
    for key in candidates:
        if fallback in key:
            return candidates[key].flatten().astype(np.float64), key
    return None, None


# ── Feature extraction ─────────────────────────────────────────────────────────

def time_domain_features(window: np.ndarray) -> dict:
    """9 time-domain statistical features."""
    rms  = np.sqrt(np.mean(window ** 2))
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
    """5 frequency-domain features via FFT (bands: <1 kHz, 1-3 kHz, >3 kHz)."""
    n       = len(window)
    fft_mag = np.abs(np.fft.rfft(window * np.hanning(n))) / n
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)

    total_energy      = np.sum(fft_mag ** 2) + 1e-12
    spectral_centroid = np.sum(freqs * fft_mag ** 2) / total_energy
    dominant_freq     = freqs[np.argmax(fft_mag)]

    low_mask  = freqs < 1_000
    mid_mask  = (freqs >= 1_000) & (freqs < 3_000)
    high_mask = freqs >= 3_000

    return {
        "spectral_centroid":  spectral_centroid,
        "dominant_frequency": dominant_freq,
        "low_band_energy":    np.sum(fft_mag[low_mask]  ** 2) / total_energy,
        "mid_band_energy":    np.sum(fft_mag[mid_mask]  ** 2) / total_energy,
        "high_band_energy":   np.sum(fft_mag[high_mask] ** 2) / total_energy,
    }


def extract_features(
    signal: np.ndarray,
    label: str,
    fault_size: str,
    motor_load: int,
    bearing_location: str,           # "DE" or "FE"
    window_size: int = WINDOW_SIZE,
) -> list[dict]:
    """Segment signal into non-overlapping windows and extract all 15 features."""
    n_windows = len(signal) // window_size
    rows = []
    for i in range(n_windows):
        w    = signal[i * window_size : (i + 1) * window_size]
        feat: dict = {}
        feat.update(time_domain_features(w))
        feat.update(freq_domain_features(w))
        feat["motor_load"]       = motor_load
        feat["bearing_location"] = bearing_location
        feat["fault_size"]       = fault_size
        feat["label"]            = label
        rows.append(feat)
    return rows


# ── Canonical column order ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "mean", "std", "rms", "peak", "peak2peak",
    "crest_factor", "kurtosis", "skewness", "shape_factor",
    "spectral_centroid", "dominant_frequency",
    "low_band_energy", "mid_band_energy", "high_band_energy",
    "motor_load",
]
META_COLS_ORIG     = ["label", "severity"]
META_COLS_EXPANDED = ["label", "bearing_location", "fault_size"]


# ── Per-directory loader ───────────────────────────────────────────────────────

def load_directory(
    mat_dir: Path,
    file_lookup: dict[int, tuple[str, str, int]],
    preferred_channel: str,
    bearing_location: str,
    is_normal_dir: bool = False,
) -> tuple[list[dict], list[str]]:
    """
    Scan mat_dir for .mat files, look up metadata, extract features.

    Parameters
    ----------
    mat_dir            : directory containing .mat files
    file_lookup        : {file_num: (fault_type, fault_size, motor_load)}
    preferred_channel  : 'DE_time' or 'FE_time'
    bearing_location   : 'DE' or 'FE'
    is_normal_dir      : if True, both DE and FE channels are extracted in one pass

    Returns
    -------
    (all_rows, skipped_names)
    """
    mat_files  = sorted(mat_dir.glob("*.mat"))
    all_rows   = []
    skipped    = []

    for mat_path in mat_files:
        file_num = extract_file_number(mat_path.name)
        if file_num is None:
            warnings.warn(f"Cannot parse number from '{mat_path.name}' — skipping.")
            skipped.append(mat_path.name)
            continue

        if file_num not in file_lookup:
            warnings.warn(f"File {file_num} ('{mat_path.name}') not in mapping — skipping.")
            skipped.append(mat_path.name)
            continue

        fault_type, fault_size, motor_load = file_lookup[file_num]

        try:
            mat_data = scipy.io.loadmat(str(mat_path))
        except Exception as exc:
            warnings.warn(f"Failed to load '{mat_path.name}': {exc} — skipping.")
            skipped.append(mat_path.name)
            continue

        if is_normal_dir:
            # Extract both DE and FE channels from the same normal file
            channels_to_extract = [("DE_time", "DE"), ("FE_time", "FE")]
        else:
            channels_to_extract = [(preferred_channel, bearing_location)]

        for ch_pref, bl in channels_to_extract:
            signal, found_key = detect_channel(mat_data, ch_pref)
            if signal is None:
                if not is_normal_dir:
                    warnings.warn(f"No {ch_pref} in '{mat_path.name}' — skipping.")
                    skipped.append(mat_path.name)
                continue

            # 7-class label: "normal" stays "normal"; faults are prefixed with location
            if fault_type == "normal":
                class_label = "normal"
            else:
                class_label = f"{bl}_{fault_type}"

            rows = extract_features(signal, class_label, fault_size, motor_load, bl)
            all_rows.extend(rows)
            print(
                f"  [{file_num:>4}] {mat_path.name:<20}  ch={found_key:<15}  "
                f"label={class_label:<20}  size={fault_size:<6}  "
                f"load={motor_load}HP  windows={len(rows)}"
            )

    return all_rows, skipped


# ── Flat-mode loader (backward compat → features.csv) ─────────────────────────

def load_flat(raw_dir: Path) -> list[dict]:
    """
    Original flat scan used by V1. Produces features with 'label' and 'severity'
    columns (backward compatible with scripts 02 and 03).
    """
    mat_files = sorted(raw_dir.glob("*.mat"))
    all_rows: list[dict] = []
    skipped: list[str] = []

    for mat_path in mat_files:
        file_num = extract_file_number(mat_path.name)
        if file_num is None or file_num not in DE_LOOKUP:
            skipped.append(mat_path.name)
            continue
        label, severity, motor_load = DE_LOOKUP[file_num]
        try:
            mat_data = scipy.io.loadmat(str(mat_path))
        except Exception:
            skipped.append(mat_path.name)
            continue
        signal, _ = detect_channel(mat_data, "DE_time")
        if signal is None:
            skipped.append(mat_path.name)
            continue
        n_windows = len(signal) // WINDOW_SIZE
        for i in range(n_windows):
            w    = signal[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
            feat: dict = {}
            feat.update(time_domain_features(w))
            feat.update(freq_domain_features(w))
            feat["motor_load"] = motor_load
            feat["label"]      = label
            feat["severity"]   = severity
            all_rows.append(feat)
    return all_rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

    de_dir     = DATA_RAW_DIR / "DE_12k"
    fe_dir     = DATA_RAW_DIR / "FE_12k"
    normal_dir = DATA_RAW_DIR / "normal"

    expanded_mode = de_dir.exists() and (fe_dir.exists() or normal_dir.exists())

    # ── FLAT MODE — produce features.csv (V1 backward compat) ─────────────────
    flat_mat_files = list(DATA_RAW_DIR.glob("*.mat"))
    if flat_mat_files:
        print(f"\n{'='*65}")
        print("FLAT MODE: found .mat files directly in data/raw/")
        print(f"{'='*65}")
        print(f"  {len(flat_mat_files)} files found — generating features.csv (V1 output)")
        flat_rows = load_flat(DATA_RAW_DIR)
        if flat_rows:
            df_flat = pd.DataFrame(flat_rows)[FEATURE_COLS + META_COLS_ORIG]
            out_flat = DATA_PROC_DIR / "features.csv"
            df_flat.to_csv(out_flat, index=False)
            print(f"  Saved {len(df_flat):,} windows → {out_flat}")

    # ── EXPANDED MODE — produce features_expanded.csv (V2 multi-location) ─────
    if not expanded_mode:
        print("\n[INFO] Expanded-mode directories (DE_12k/, FE_12k/, normal/) not found.")
        print("       To run expanded mode, reorganise your raw files:")
        print("         mv data/raw/*.mat data/raw/DE_12k/   (for drive-end files)")
        print("         # place fan-end .mat files in data/raw/FE_12k/")
        print("         # place normal baseline .mat files in data/raw/normal/")
        return

    print(f"\n{'='*65}")
    print("EXPANDED MODE: scanning DE_12k/, FE_12k/, normal/")
    print(f"{'='*65}\n")

    all_rows: list[dict] = []
    all_skipped: list[str] = []

    # 1. Drive-end fault files
    if de_dir.exists() and any(de_dir.glob("*.mat")):
        print(f"[DE_12k] Scanning {de_dir} ...")
        rows, skipped = load_directory(
            de_dir, DE_LOOKUP, preferred_channel="DE_time",
            bearing_location="DE", is_normal_dir=False,
        )
        all_rows.extend(rows)
        all_skipped.extend(skipped)
    else:
        print(f"[DE_12k] Directory empty or missing — skipping.")

    # 2. Fan-end fault files
    if fe_dir.exists() and any(fe_dir.glob("*.mat")):
        print(f"\n[FE_12k] Scanning {fe_dir} ...")
        rows, skipped = load_directory(
            fe_dir, FE_LOOKUP, preferred_channel="FE_time",
            bearing_location="FE", is_normal_dir=False,
        )
        all_rows.extend(rows)
        all_skipped.extend(skipped)
    else:
        print(f"[FE_12k] Directory empty or missing — skipping.")

    # 3. Normal baseline files (extract both DE and FE channels)
    if normal_dir.exists() and any(normal_dir.glob("*.mat")):
        print(f"\n[normal] Scanning {normal_dir} (both DE+FE channels) ...")
        rows, skipped = load_directory(
            normal_dir, NORMAL_FILE_LOOKUP, preferred_channel="DE_time",
            bearing_location="DE", is_normal_dir=True,
        )
        all_rows.extend(rows)
        all_skipped.extend(skipped)
    else:
        print(f"[normal] Directory empty or missing — skipping.")

    if not all_rows:
        print("\n[ERROR] No features extracted in expanded mode.")
        print("        Ensure .mat files are placed in the correct subdirectories.")
        return

    # ── Build expanded DataFrame ───────────────────────────────────────────────
    df_exp = pd.DataFrame(all_rows)[FEATURE_COLS + META_COLS_EXPANDED]

    out_exp = DATA_PROC_DIR / "features_expanded.csv"
    df_exp.to_csv(out_exp, index=False)
    print(f"\nSaved {len(df_exp):,} windows → {out_exp}")

    # ── Class distribution summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION SUMMARY (features_expanded.csv)")
    print("=" * 70)
    summary = (
        df_exp.groupby(["label", "bearing_location", "fault_size"])
        .size()
        .reset_index(name="windows")
        .sort_values(["label", "fault_size"])
    )
    print(summary.to_string(index=False))

    print("\n" + "-" * 70)
    print("TOTAL PER CLASS:")
    per_class = df_exp.groupby("label").size().sort_values(ascending=False)
    for cls, cnt in per_class.items():
        print(f"  {cls:<25}  {cnt:>6,} windows")

    print("-" * 70)
    print(f"Total windows  : {len(df_exp):,}")
    print(f"Total features : {len(FEATURE_COLS)}")
    print(f"Classes ({len(per_class)})    : {sorted(df_exp['label'].unique())}")
    if all_skipped:
        print(f"\nSkipped files ({len(all_skipped)}): {all_skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
