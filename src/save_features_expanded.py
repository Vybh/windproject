"""
save_features_expanded.py
--------------------------
Loads all CWRU bearing-fault windows via load_cwru_files(), adds the
is_synthetic flag (False for all real-signal windows), and writes the
tidy feature table to data/processed/features_expanded.csv.

Usage:
    python src/save_features_expanded.py
"""

from pathlib import Path

import pandas as pd

from load_cwru_files import load_cwru_files

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_cwru_files()

# ── Add is_synthetic flag ─────────────────────────────────────────────────────
df.insert(df.columns.get_loc('motor_load') + 1, 'is_synthetic', False)

# ── Column order ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'mean', 'std', 'rms', 'peak', 'peak2peak', 'crest_factor',
    'kurtosis', 'skewness', 'shape_factor',
    'spectral_centroid', 'dominant_frequency',
    'low_band_energy', 'mid_band_energy', 'high_band_energy',
]
META_COLS = ['filename', 'label', 'bearing_location', 'fault_size', 'motor_load', 'is_synthetic']
df = df[META_COLS + FEATURE_COLS]

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path('data/processed/features_expanded.csv')
df.to_csv(out_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
width = 70

print()
print('=' * width)
print('WINDOWS PER CLASS')
print('=' * width)
for cls, cnt in df.groupby('label').size().sort_values(ascending=False).items():
    print(f'  {cls:<30}  {cnt:>7,}')

print()
print('=' * width)
print('WINDOWS PER FAULT SIZE')
print('=' * width)
for sz, cnt in df.groupby('fault_size').size().sort_index().items():
    print(f'  {sz:<30}  {cnt:>7,}')

print()
print('=' * width)
print('WINDOWS PER BEARING LOCATION')
print('=' * width)
for loc, cnt in df.groupby('bearing_location').size().sort_values(ascending=False).items():
    print(f'  {loc:<30}  {cnt:>7,}')

print()
print('=' * width)
print(f'CSV saved  →  {out_path}')
print(f'Shape      :  {df.shape[0]:,} rows × {df.shape[1]} columns')
print('=' * width)
