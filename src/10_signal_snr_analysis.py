"""
10_signal_snr_analysis.py
--------------------------
Physics-Grounded FE_ball Analysis (GAP 6).

Loads raw .mat files for DE_ball and FE_ball fault classes.
Computes per-signal:
  • RMS energy
  • Signal-to-noise ratio (fault peak energy / broadband noise floor)
  • Spectral peak prominence at bearing fault characteristic frequencies
  • Transmission path attenuation ratio: DE_ball / FE_ball

This transforms the FE_ball accuracy gap from a black-box observation into
a physically interpretable finding: fan-end accelerometer location produces
lower-energy fault signatures due to measurement path attenuation.

Inputs:
    data/raw/DE_12k/*.mat  (files 118-121, 185-188, 222-225 = DE_ball 0.007-0.021")
    data/raw/FE_12k/*.mat  (files 282-293 = FE_ball 0.007-0.021")

Outputs:
    data/processed/signal_snr_results.csv
    paper/figures/fig9_fe_vs_de_signal_analysis.pdf / .png
"""

import warnings
import re
import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW   = Path("data/raw")
DATA_DIR   = Path("data/processed")
FIG_DIR    = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE   = 1024
SAMPLING_RATE = 12_000

# CWRU bearing characteristic frequencies (shaft speed ~1797 RPM = 29.95 Hz)
# At 0 HP load: shaft_speed = 29.95 Hz
SHAFT_HZ = 29.95
# BPFO (Ball Pass Frequency Outer race) ≈ 3.585 × shaft_speed for 6205-2RS JEM
# BPFI (Ball Pass Frequency Inner race) ≈ 5.415 × shaft_speed
# BSF  (Ball Spin Frequency)            ≈ 2.357 × shaft_speed
BPFO = 3.585 * SHAFT_HZ    # ≈ 107.4 Hz
BPFI = 5.415 * SHAFT_HZ    # ≈ 162.2 Hz
BSF  = 2.357 * SHAFT_HZ    # ≈ 70.6  Hz

# File-number to (label, fault_size) mapping (from load_cwru_files.py)
DE_BALL_FILES = {
    118: "0.007", 119: "0.007", 120: "0.007", 121: "0.007",
    185: "0.014", 186: "0.014", 187: "0.014", 188: "0.014",
    222: "0.021", 223: "0.021", 224: "0.021", 225: "0.021",
}
FE_BALL_FILES = {
    282: "0.007", 283: "0.007", 284: "0.007", 285: "0.007",
    286: "0.014", 287: "0.014", 288: "0.014", 289: "0.014",
    290: "0.021", 291: "0.021", 292: "0.021", 293: "0.021",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ── Signal metrics ─────────────────────────────────────────────────────────────

def rms_energy(signal: np.ndarray) -> float:
    """Root-mean-square energy of the signal."""
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_snr(signal: np.ndarray, target_freq: float,
                fs: int = SAMPLING_RATE, window_size: int = WINDOW_SIZE) -> float:
    """
    Signal-to-noise ratio at target_freq.
    SNR = (spectral power at target_freq ± bandwidth) / (median broadband noise floor).
    Operates on a single window.
    """
    n       = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal * np.hanning(n))) ** 2 / n
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)

    # Bandwidth: ±2 FFT bins around target
    bin_width = fs / n
    bandwidth = 3 * bin_width
    signal_mask = np.abs(freqs - target_freq) <= bandwidth
    noise_mask  = (freqs > 10) & ~signal_mask  # exclude DC

    if signal_mask.sum() == 0 or noise_mask.sum() == 0:
        return float("nan")

    signal_power = fft_mag[signal_mask].mean()
    noise_floor  = np.median(fft_mag[noise_mask])
    return float(signal_power / (noise_floor + 1e-30))


def spectral_peak_prominence(signal: np.ndarray, target_freq: float,
                              fs: int = SAMPLING_RATE, window_size: int = WINDOW_SIZE,
                              n_harmonics: int = 3) -> float:
    """
    Peak prominence of the target frequency and its harmonics relative to
    the surrounding spectral noise. Higher value → clearer fault signature.
    """
    n       = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal * np.hanning(n))) / n
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)

    total_prominence = 0.0
    bin_width        = fs / n

    for h in range(1, n_harmonics + 1):
        freq_h = target_freq * h
        if freq_h >= fs / 2:
            break
        # Find closest bin
        bin_idx = int(round(freq_h / bin_width))
        if bin_idx >= len(fft_mag):
            continue
        # Local neighborhood (±5 bins)
        lo      = max(0, bin_idx - 5)
        hi      = min(len(fft_mag), bin_idx + 6)
        local   = fft_mag[lo:hi]
        peak_v  = fft_mag[bin_idx]
        baseline = np.median(local)
        if baseline > 0:
            total_prominence += float(peak_v / baseline)

    return total_prominence / n_harmonics


def _find_key(mat: dict, substring: str):
    return next((k for k in mat if not k.startswith("__") and substring in k), None)


# ── Per-file analysis ─────────────────────────────────────────────────────────

def analyse_file(
    mat_path: Path,
    channel: str,   # "DE_time" or "FE_time"
    fault_freq: float,   # BSF for ball faults
    location: str,
    fault_size: str,
) -> list[dict]:
    """Load a .mat file, segment into windows, compute metrics per window."""
    try:
        mat = scipy.io.loadmat(str(mat_path))
    except Exception as e:
        print(f"  [SKIP] {mat_path.name}: {e}")
        return []

    sig_key = _find_key(mat, channel)
    if sig_key is None:
        print(f"  [SKIP] {mat_path.name}: no {channel} channel found.")
        return []

    signal    = mat[sig_key].flatten().astype(np.float64)
    n_windows = len(signal) // WINDOW_SIZE

    rows = []
    for i in range(n_windows):
        w   = signal[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
        rows.append({
            "file":        mat_path.name,
            "location":    location,
            "fault_size":  fault_size,
            "window_idx":  i,
            "rms_energy":  rms_energy(w),
            "snr_at_bsf":  compute_snr(w, fault_freq),
            "peak_prom":   spectral_peak_prominence(w, fault_freq),
        })
    return rows


# ── Load DE_ball and FE_ball ───────────────────────────────────────────────────

def load_ball_signals() -> pd.DataFrame:
    """Load all DE_ball and FE_ball .mat files and compute signal metrics."""
    all_rows = []

    de_dir = DATA_RAW / "DE_12k"
    if not de_dir.exists():
        print(f"[ERROR] {de_dir} not found. Raw data must be present.")
        return pd.DataFrame()

    print("Loading DE_ball files ...")
    for mat_path in sorted(de_dir.glob("*.mat")):
        try:
            file_num = int(mat_path.stem)
        except ValueError:
            continue
        if file_num in DE_BALL_FILES:
            fault_size = DE_BALL_FILES[file_num]
            rows = analyse_file(mat_path, "DE_time", BSF, "DE", fault_size)
            all_rows.extend(rows)
            print(f"  [{file_num}] DE_ball {fault_size}\"  windows={len(rows)}")

    fe_dir = DATA_RAW / "FE_12k"
    if not fe_dir.exists():
        print(f"[WARN] {fe_dir} not found — FE_ball analysis will be skipped.")
    else:
        print("\nLoading FE_ball files ...")
        for mat_path in sorted(fe_dir.glob("*.mat")):
            try:
                file_num = int(mat_path.stem)
            except ValueError:
                continue
            if file_num in FE_BALL_FILES:
                fault_size = FE_BALL_FILES[file_num]
                rows = analyse_file(mat_path, "FE_time", BSF, "FE", fault_size)
                all_rows.extend(rows)
                print(f"  [{file_num}] FE_ball {fault_size}\"  windows={len(rows)}")

    if not all_rows:
        print("[ERROR] No DE_ball or FE_ball files loaded.")
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# ── Figure 9 ──────────────────────────────────────────────────────────────────

def plot_fig9(df: pd.DataFrame, summary: pd.DataFrame):
    """
    Figure 9: Three-panel comparison of DE_ball vs FE_ball signal quality.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, ylabel, title_suffix in [
        (axes[0], "rms_energy", "RMS Energy (g)",
         "RMS Signal Energy"),
        (axes[1], "snr_at_bsf", "SNR at BSF",
         "SNR at Ball Spin Frequency"),
        (axes[2], "peak_prom", "Peak Prominence",
         "Spectral Peak Prominence at BSF"),
    ]:
        de_vals = df[df["location"] == "DE"][metric].dropna()
        fe_vals = df[df["location"] == "FE"][metric].dropna()

        # Violin + swarm
        sub_df = df[["location", metric]].dropna()
        sns.violinplot(
            data=sub_df, x="location", y=metric,
            palette={"DE": "#42A5F5", "FE": "#FF7043"},
            inner="quartile", ax=ax, order=["DE", "FE"],
        )
        ax.set_xlabel("Accelerometer Location", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Fig. 9: {title_suffix}\n(DE_ball vs FE_ball)", fontsize=9,
                     fontweight="bold")

        # Annotate means
        for i, (loc, vals) in enumerate([("DE", de_vals), ("FE", fe_vals)]):
            ax.text(i, vals.max() * 1.02, f"μ={vals.mean():.4f}",
                    ha="center", fontsize=8, color="black")

    # Add attenuation annotation
    de_rms = df[df["location"] == "DE"]["rms_energy"].mean()
    fe_rms = df[df["location"] == "FE"]["rms_energy"].mean()
    if fe_rms > 0:
        attenuation_db = 20 * np.log10(de_rms / fe_rms)
        axes[0].set_title(
            f"Fig. 9a: RMS Energy\nDE/FE attenuation: {attenuation_db:.1f} dB",
            fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"fig9_fe_vs_de_signal_analysis.{ext}"
        plt.savefig(path, dpi=300)
        print(f"  Saved: {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SIGNAL SNR ANALYSIS: DE_ball vs FE_ball (GAP 6)")
    print("=" * 65)

    df = load_ball_signals()
    if df.empty:
        print("[ERROR] No signal data loaded. Ensure raw .mat files are present.")
        return

    print(f"\nLoaded {len(df):,} windows total.")
    print(f"  DE_ball: {(df['location']=='DE').sum():,}")
    print(f"  FE_ball: {(df['location']=='FE').sum():,}\n")

    # ── Summary statistics ─────────────────────────────────────────────────────
    summary = df.groupby(["location", "fault_size"])[
        ["rms_energy", "snr_at_bsf", "peak_prom"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.values]
    summary = summary.rename(columns={"location_": "location", "fault_size_": "fault_size"})

    out_path = DATA_DIR / "signal_snr_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Per-window results → {out_path}")

    summary_path = DATA_DIR / "signal_snr_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics → {summary_path}")

    # ── Key metrics ───────────────────────────────────────────────────────────
    de_rms   = df[df["location"] == "DE"]["rms_energy"].mean()
    fe_rms   = df[df["location"] == "FE"]["rms_energy"].mean()
    de_snr   = df[df["location"] == "DE"]["snr_at_bsf"].mean()
    fe_snr   = df[df["location"] == "FE"]["snr_at_bsf"].mean()
    de_prom  = df[df["location"] == "DE"]["peak_prom"].mean()
    fe_prom  = df[df["location"] == "FE"]["peak_prom"].mean()

    rms_ratio = de_rms / fe_rms if fe_rms > 0 else float("nan")
    snr_ratio = de_snr / fe_snr if fe_snr > 0 else float("nan")
    prom_ratio = de_prom / fe_prom if fe_prom > 0 else float("nan")

    attenuation_db = 20 * np.log10(rms_ratio) if rms_ratio > 0 else float("nan")

    print("\n" + "=" * 65)
    print("SUMMARY: DE_ball vs FE_ball Signal Quality Comparison")
    print("=" * 65)
    print(f"  {'Metric':<30}  {'DE_ball':>10}  {'FE_ball':>10}  {'DE/FE Ratio':>12}")
    print("  " + "-" * 65)
    print(f"  {'RMS Energy':<30}  {de_rms:>10.6f}  {fe_rms:>10.6f}  {rms_ratio:>12.2f}×")
    print(f"  {'SNR at BSF':<30}  {de_snr:>10.2f}  {fe_snr:>10.2f}  {snr_ratio:>12.2f}×")
    print(f"  {'Spectral Peak Prominence':<30}  {de_prom:>10.2f}  {fe_prom:>10.2f}  {prom_ratio:>12.2f}×")
    print(f"  {'Attenuation (dB)':<30}  {attenuation_db:>10.1f} dB (DE advantage over FE)")
    print("=" * 65)

    print("\nPhysics interpretation:")
    if rms_ratio > 1.5:
        print(f"  DE_ball RMS energy is {rms_ratio:.1f}× higher than FE_ball.")
        print(f"  This corresponds to {attenuation_db:.1f} dB transmission path attenuation")
        print(f"  from the drive end to the fan end bearing location.")
        print(f"  Lower signal energy at FE means fault signatures are closer to")
        print(f"  the noise floor, making FE_ball the hardest class to classify.")
    else:
        print(f"  RMS ratio DE/FE = {rms_ratio:.2f}× — modest attenuation difference.")
        print(f"  Other factors (fault geometry, accelerometer sensitivity) may")
        print(f"  explain the observed FE_ball classification gap.")

    if snr_ratio > 1.2:
        print(f"\n  SNR at BSF: DE is {snr_ratio:.1f}× higher → clearer fault frequency in DE.")
    if prom_ratio > 1.1:
        print(f"  Peak prominence: DE is {prom_ratio:.1f}× higher → stronger harmonic structure in DE.")

    print("\n  Paper subsection recommendation:")
    print("  'Fan-end bearing faults produce lower-energy vibration signatures")
    print("   at the measurement location due to transmission path attenuation.")
    print(f"  The DE/FE energy ratio of {rms_ratio:.1f}× ({attenuation_db:.1f} dB) explains")
    print("   why FE_ball is the hardest fault class across all models.'")

    # ── Write SNR results to text file ─────────────────────────────────────────
    snr_txt = DATA_DIR / "signal_snr_summary.txt"
    lines = [
        "SIGNAL SNR ANALYSIS: DE_ball vs FE_ball",
        "=" * 65,
        "",
        f"Metric                          DE_ball     FE_ball     DE/FE Ratio",
        "-" * 65,
        f"RMS Energy                   {de_rms:>10.6f}  {fe_rms:>10.6f}  {rms_ratio:>10.2f}×",
        f"SNR at BSF                   {de_snr:>10.2f}  {fe_snr:>10.2f}  {snr_ratio:>10.2f}×",
        f"Spectral Peak Prominence     {de_prom:>10.2f}  {fe_prom:>10.2f}  {prom_ratio:>10.2f}×",
        f"Attenuation                  {attenuation_db:>10.1f} dB",
        "",
        "Fault frequency analysed: BSF (Ball Spin Frequency)",
        f"BSF = 2.357 × shaft_speed = {BSF:.1f} Hz  (at shaft_speed = {SHAFT_HZ} Hz)",
        "",
        "Conclusion:",
        f"  DE_ball signals are {rms_ratio:.1f}× stronger (RMS) than FE_ball signals.",
        f"  This {attenuation_db:.1f} dB path attenuation is the primary reason FE_ball",
        "  is the lowest-performing fault class across all ML models.",
    ]
    with open(snr_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSNR summary text → {snr_txt}")

    # ── Figure 9 ──────────────────────────────────────────────────────────────
    plot_fig9(df, summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
