"""
load_cwru_files.py
------------------
Single function that scans the three CWRU bearing-fault directories,
loads every .mat file, extracts a DE or FE vibration signal, engineers
15 features per 1024-sample window, and returns a tidy DataFrame.

Directory layout expected:
    data/raw/normal/   – 4 baseline files (both DE + FE channels extracted)
    data/raw/DE_12k/   – drive-end fault files (numeric IDs only on disk)
    data/raw/FE_12k/   – fan-end fault files  (numeric IDs only on disk)

Usage:
    from src.load_cwru_files import load_cwru_files
    df = load_cwru_files()
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import kurtosis, skew

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 1024
SAMPLING_RATE = 12_000          # Hz

# Motor load mapping: nominal RPM → integer load level (0–3 HP)
_RPM_TARGETS = [1797, 1772, 1750, 1730]

# ── Fault maps ────────────────────────────────────────────────────────────────
# Values are (label, fault_size_inches_str).
# Do NOT add file numbers beyond those listed here.

FAULT_MAP: dict[int, tuple[str, str]] = {
    # ── DE inner race ─────────────────────────────────────────────────────────
    105: ('DE_inner_race', '0.007'), 106: ('DE_inner_race', '0.007'),
    107: ('DE_inner_race', '0.007'), 108: ('DE_inner_race', '0.007'),
    169: ('DE_inner_race', '0.014'), 170: ('DE_inner_race', '0.014'),
    171: ('DE_inner_race', '0.014'), 172: ('DE_inner_race', '0.014'),
    209: ('DE_inner_race', '0.021'), 210: ('DE_inner_race', '0.021'),
    211: ('DE_inner_race', '0.021'), 212: ('DE_inner_race', '0.021'),
    # ── DE ball ───────────────────────────────────────────────────────────────
    118: ('DE_ball', '0.007'), 119: ('DE_ball', '0.007'),
    120: ('DE_ball', '0.007'), 121: ('DE_ball', '0.007'),
    185: ('DE_ball', '0.014'), 186: ('DE_ball', '0.014'),
    187: ('DE_ball', '0.014'), 188: ('DE_ball', '0.014'),
    222: ('DE_ball', '0.021'), 223: ('DE_ball', '0.021'),
    224: ('DE_ball', '0.021'), 225: ('DE_ball', '0.021'),
    # ── DE outer race ─────────────────────────────────────────────────────────
    130: ('DE_outer_race', '0.007'), 131: ('DE_outer_race', '0.007'),
    132: ('DE_outer_race', '0.007'), 133: ('DE_outer_race', '0.007'),
    197: ('DE_outer_race', '0.014'), 198: ('DE_outer_race', '0.014'),
    199: ('DE_outer_race', '0.014'), 200: ('DE_outer_race', '0.014'),
    234: ('DE_outer_race', '0.021'), 235: ('DE_outer_race', '0.021'),
    236: ('DE_outer_race', '0.021'), 237: ('DE_outer_race', '0.021'),
}

FE_FAULT_MAP: dict[int, tuple[str, str]] = {
    # ── FE inner race (270–281) ───────────────────────────────────────────────
    270: ('FE_inner_race', '0.007'), 271: ('FE_inner_race', '0.007'),
    272: ('FE_inner_race', '0.007'), 273: ('FE_inner_race', '0.007'),
    274: ('FE_inner_race', '0.014'), 275: ('FE_inner_race', '0.014'),
    276: ('FE_inner_race', '0.014'), 277: ('FE_inner_race', '0.014'),
    278: ('FE_inner_race', '0.021'), 279: ('FE_inner_race', '0.021'),
    280: ('FE_inner_race', '0.021'), 281: ('FE_inner_race', '0.021'),
    # ── FE ball (282–293) ─────────────────────────────────────────────────────
    282: ('FE_ball', '0.007'), 283: ('FE_ball', '0.007'),
    284: ('FE_ball', '0.007'), 285: ('FE_ball', '0.007'),
    286: ('FE_ball', '0.014'), 287: ('FE_ball', '0.014'),
    288: ('FE_ball', '0.014'), 289: ('FE_ball', '0.014'),
    290: ('FE_ball', '0.021'), 291: ('FE_ball', '0.021'),
    292: ('FE_ball', '0.021'), 293: ('FE_ball', '0.021'),
    # ── FE outer race (294–318, various positions) ────────────────────────────
    294: ('FE_outer_race', '0.007'), 295: ('FE_outer_race', '0.007'),
    296: ('FE_outer_race', '0.007'), 297: ('FE_outer_race', '0.007'),
    298: ('FE_outer_race', '0.014'), 299: ('FE_outer_race', '0.014'),
    300: ('FE_outer_race', '0.014'), 301: ('FE_outer_race', '0.014'),
    302: ('FE_outer_race', '0.021'), 305: ('FE_outer_race', '0.021'),
    306: ('FE_outer_race', '0.021'), 307: ('FE_outer_race', '0.021'),
    309: ('FE_outer_race', '0.021'), 310: ('FE_outer_race', '0.021'),
    311: ('FE_outer_race', '0.021'), 312: ('FE_outer_race', '0.021'),
    313: ('FE_outer_race', '0.021'), 315: ('FE_outer_race', '0.028'),
    316: ('FE_outer_race', '0.028'), 317: ('FE_outer_race', '0.028'),
    318: ('FE_outer_race', '0.028'),
}

# ── DE 0.028" severity files (no RPM key; load from last digit of stem) ──────
DE_028_MAP: dict[int, str] = {
    3001: 'DE_inner_race', 3002: 'DE_inner_race',
    3003: 'DE_inner_race', 3004: 'DE_inner_race',
    3005: 'DE_ball',       3006: 'DE_ball',
    3007: 'DE_ball',       3008: 'DE_ball',
}

# ── Non-numeric DE filename regex (e.g. IR028_0.mat) ─────────────────────────
_NONNUMERIC_RE = re.compile(r'^(IR|B|OR)(\d{3})_(\d)$')
_FAULT_CODE: dict[str, str] = {
    'IR': 'DE_inner_race',
    'B':  'DE_ball',
    'OR': 'DE_outer_race',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rpm_to_motor_load(rpm: float) -> int:
    """Map a measured RPM to the nearest standard motor-load index (0–3)."""
    return int(min(range(4), key=lambda i: abs(_RPM_TARGETS[i] - rpm)))


def _find_key(mat: dict, substring: str) -> str | None:
    """Return the first non-dunder key whose name contains *substring*."""
    return next(
        (k for k in mat if not k.startswith('__') and substring in k),
        None,
    )


def _time_features(w: np.ndarray) -> dict:
    rms  = np.sqrt(np.mean(w ** 2))
    peak = np.max(np.abs(w))
    return {
        'mean':         float(np.mean(w)),
        'std':          float(np.std(w)),
        'rms':          float(rms),
        'peak':         float(peak),
        'peak2peak':    float(np.ptp(w)),
        'crest_factor': float(peak / (rms + 1e-12)),
        'kurtosis':     float(kurtosis(w, fisher=True)),
        'skewness':     float(skew(w)),
        'shape_factor': float(rms / (np.mean(np.abs(w)) + 1e-12)),
    }


def _freq_features(w: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    n       = len(w)
    mag     = np.abs(np.fft.rfft(w * np.hanning(n))) / n
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)
    energy  = np.sum(mag ** 2) + 1e-12
    return {
        'spectral_centroid':  float(np.sum(freqs * mag ** 2) / energy),
        'dominant_frequency': float(freqs[np.argmax(mag)]),
        'low_band_energy':    float(np.sum(mag[freqs < 1_000]      ** 2) / energy),
        'mid_band_energy':    float(np.sum(mag[(freqs >= 1_000) & (freqs < 3_000)] ** 2) / energy),
        'high_band_energy':   float(np.sum(mag[freqs >= 3_000]     ** 2) / energy),
    }


def _segment(
    signal: np.ndarray,
    filename: str,
    label: str,
    bearing_location: str,
    fault_size: str,
    motor_load: int,
) -> list[dict]:
    """Slice *signal* into non-overlapping 1024-sample windows; return feature rows."""
    n_win = len(signal) // WINDOW_SIZE
    rows  = []
    for i in range(n_win):
        w   = signal[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
        row = {
            'filename':         filename,
            'label':            label,
            'bearing_location': bearing_location,
            'fault_size':       fault_size,
            'motor_load':       motor_load,
        }
        row.update(_time_features(w))
        row.update(_freq_features(w))
        rows.append(row)
    return rows


# ── Main loader ───────────────────────────────────────────────────────────────

def load_cwru_files() -> pd.DataFrame:
    """
    Scan data/raw/normal/, data/raw/DE_12k/, and data/raw/FE_12k/.
    Load every recognised .mat file, extract the relevant vibration channel,
    engineer 15 features per 1024-sample window, and return a DataFrame.

    Columns
    -------
    filename, label, bearing_location, fault_size, motor_load,
    mean, std, rms, peak, peak2peak, crest_factor, kurtosis, skewness,
    shape_factor, spectral_centroid, dominant_frequency,
    low_band_energy, mid_band_energy, high_band_energy

    Prints a class distribution summary; never raises on bad/unrecognised files.
    """
    base = Path('data/raw')
    all_rows: list[dict] = []

    # ── 1. normal/ ────────────────────────────────────────────────────────────
    normal_dir = base / 'normal'
    for mat_path in sorted(normal_dir.glob('*.mat')):
        try:
            m = scipy.io.loadmat(str(mat_path))
        except Exception:
            continue

        rpm_key = _find_key(m, 'RPM')
        if rpm_key is None:
            continue
        motor_load = _rpm_to_motor_load(float(np.squeeze(m[rpm_key])))

        # Extract both DE and FE channels from every normal file
        for ch_suffix, bl in [('DE_time', 'DE'), ('FE_time', 'FE')]:
            sig_key = _find_key(m, ch_suffix)
            if sig_key is None:
                continue
            signal = m[sig_key].flatten().astype(np.float64)
            all_rows.extend(
                _segment(signal, mat_path.name, 'normal', bl, 'none', motor_load)
            )

    # ── 2. DE_12k/ ────────────────────────────────────────────────────────────
    de_dir = base / 'DE_12k'
    for mat_path in sorted(de_dir.glob('*.mat')):
        stem = mat_path.stem

        # Non-numeric filenames (e.g. IR028_0.mat) – fault info from filename
        nn = _NONNUMERIC_RE.match(stem)
        if nn:
            fault_code, size_digits, load_digit = nn.groups()
            label      = _FAULT_CODE[fault_code]
            fault_size = f'0.{size_digits}'
            motor_load = int(load_digit)
            try:
                m = scipy.io.loadmat(str(mat_path))
            except Exception:
                continue
            sig_key = _find_key(m, 'DE_time')
            if sig_key is None:
                continue
            signal = m[sig_key].flatten().astype(np.float64)
            all_rows.extend(
                _segment(signal, mat_path.name, label, 'DE', fault_size, motor_load)
            )
            continue

        # Numeric filenames – fault info from FAULT_MAP or DE_028_MAP
        try:
            file_num = int(stem)
        except ValueError:
            continue

        if file_num in DE_028_MAP:
            # 3001-3008: no RPM key; motor load cycles 0-3 within each group of 4
            label      = DE_028_MAP[file_num]
            fault_size = '0.028'
            motor_load = (file_num - 1) % 4
            try:
                m = scipy.io.loadmat(str(mat_path))
            except Exception:
                continue
            sig_key = _find_key(m, 'DE_time')
            if sig_key is None:
                continue
            signal = m[sig_key].flatten().astype(np.float64)
            all_rows.extend(
                _segment(signal, mat_path.name, label, 'DE', fault_size, motor_load)
            )
            continue

        if file_num not in FAULT_MAP:
            continue  # silently skip any other unmapped files

        label, fault_size = FAULT_MAP[file_num]
        try:
            m = scipy.io.loadmat(str(mat_path))
        except Exception:
            continue

        rpm_key = _find_key(m, 'RPM')
        if rpm_key is None:
            continue
        motor_load = _rpm_to_motor_load(float(np.squeeze(m[rpm_key])))

        sig_key = _find_key(m, 'DE_time')
        if sig_key is None:
            continue
        signal = m[sig_key].flatten().astype(np.float64)
        all_rows.extend(
            _segment(signal, mat_path.name, label, 'DE', fault_size, motor_load)
        )

    # ── 3. FE_12k/ ────────────────────────────────────────────────────────────
    fe_dir = base / 'FE_12k'
    for mat_path in sorted(fe_dir.glob('*.mat')):
        try:
            file_num = int(mat_path.stem)
        except ValueError:
            continue
        if file_num not in FE_FAULT_MAP:
            continue

        label, fault_size = FE_FAULT_MAP[file_num]
        try:
            m = scipy.io.loadmat(str(mat_path))
        except Exception:
            continue

        rpm_key = _find_key(m, 'RPM')
        if rpm_key is None:
            continue
        motor_load = _rpm_to_motor_load(float(np.squeeze(m[rpm_key])))

        sig_key = _find_key(m, 'FE_time')
        if sig_key is None:
            continue
        signal = m[sig_key].flatten().astype(np.float64)
        all_rows.extend(
            _segment(signal, mat_path.name, label, 'FE', fault_size, motor_load)
        )

    # ── Build DataFrame ───────────────────────────────────────────────────────
    col_order = [
        'filename', 'label', 'bearing_location', 'fault_size', 'motor_load',
        'mean', 'std', 'rms', 'peak', 'peak2peak', 'crest_factor',
        'kurtosis', 'skewness', 'shape_factor',
        'spectral_centroid', 'dominant_frequency',
        'low_band_energy', 'mid_band_energy', 'high_band_energy',
    ]
    df = pd.DataFrame(all_rows, columns=col_order)

    # ── Class distribution summary ────────────────────────────────────────────
    per_class = df.groupby('label').size().sort_values(ascending=False)
    width     = 70

    print('=' * width)
    print('CLASS DISTRIBUTION')
    print('=' * width)
    for cls, cnt in per_class.items():
        print(f'  {cls:<30}  {cnt:>7,} windows')
    print('-' * width)
    print(f'  {"Total windows":<30}  {len(df):>7,}')
    print(f'  {"Classes":<30}  {len(per_class):>7}')
    print('=' * width)

    return df
