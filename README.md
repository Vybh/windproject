# Wind Turbine Gearbox Fault Detection Using Vibration Signal Analysis with Machine Learning and SHAP Explainability

A complete research pipeline for 7-class bearing fault detection using the CWRU dataset. The project combines classical ML, deep learning, VAE-based data augmentation, and multi-model SHAP explainability, and produces publication-ready tables and figures for an IEEE paper.

---

## Primary Contribution: Severity Generalisation Gap

Standard 5-fold CV on CWRU produces near-perfect accuracy because the model has seen all fault severities during training. A **severity generalisation experiment** (train on 0.007" and 0.014", test on unseen 0.021") reveals class-specific gaps of up to **19.4 percentage points** — this is the core methodological contribution of the paper. CV accuracy alone is insufficient to assess real-world deployability.

---

## Research Findings

### Finding 1 — Severity Generalisation Gap is the Key Metric (Primary)

Per-class generalisation gaps (train small severities → test unseen 0.021") vary significantly across fault classes and model types. Classes with low SNR at the measurement location (FE_ball) show the largest generalisation gaps (Table 8 + Figure 8).

### Finding 2 — Two Distinct CNN Experiments (GAP 1 Corrected)

Two CNN experiments with fundamentally different inputs are reported:

**(a) Feature-based CNN** (15 pre-engineered features → reshaped to (15,1)): XGBoost wins. This result does NOT mean "deep learning is inferior" — it only shows tree ensembles beat CNNs on pre-summarised tabular input.

**(b) Raw-signal CNN** (1024-point windows directly from .mat files, no feature engineering): This is the real DL vs ML question. Results at both architectures are reported in Table 3.

| Model | Input Type | Accuracy (5-fold CV) |
|---|---|---|
| XGBoost | 15 engineered features | **99.10%** |
| Random Forest | 15 engineered features | 98.54% |
| Feature-based CNN | (15, 1) tabular reshape | 97.95% |
| Raw-signal CNN | (1024, 1) raw vibration | — (see Table 3) |
| KNN | 15 engineered features | 94.98% |
| SVM | 15 engineered features | 92.83% |

### Finding 3 — VAE Augmentation: Balancing vs Accuracy (GAP 2 Corrected)

A full ablation study (Table 7) reports all model accuracies under (a) real-only, class-imbalanced data and (b) real + VAE synthetic, balanced data. The paper's claim about VAE is calibrated: if improvements are <0.5%, augmentation is framed as data balancing rather than a performance boost.

### Finding 4 — FE_ball is Hard for a Physical Reason (GAP 6)

FE_ball is consistently the lowest-performing class across all models (RF: 0.968, XGBoost: 0.979, KNN: 0.870). Signal SNR analysis on raw .mat files (Figure 9) quantifies the transmission path attenuation between DE and FE accelerometer locations, transforming this black-box accuracy gap into a physically interpretable finding.

### Finding 5 — High Band Energy is the Universal SHAP Feature (Rank-Based)

Multi-model SHAP rank consensus (cross-model magnitude comparison is invalid due to explainer heterogeneity — see GAP 4) identifies **High Band Energy** as the only feature in the top 5 of all four models simultaneously (mean rank 2.75).

| Feature | RF Rank | XGBoost Rank | SVM Rank | CNN Rank | Mean Rank |
|---|---|---|---|---|---|
| High Band Energy | 2 | 1 | 5 | 3 | 2.75 |
| Spectral Centroid | 4 | 6 | 1 | 2 | 3.25 |
| Mean | 1 | 2 | 6 | 5 | 3.50 |

*Note: Ranks are used (not magnitudes) because TreeExplainer, PermutationExplainer, KernelExplainer, and GradientExplainer compute non-equivalent quantities.*

### Finding 6 — Autoencoder Detection is Within-Distribution Only (GAP 5)

The 100% fault detection rate (Table 5) is a within-distribution result on highly separable engineered features (mean separation ratio >>1×). When trained exclusively on 0.007" severity samples and tested on 0.021" unseen severity, detection rate degrades significantly, confirming the autoencoder operates as a within-distribution anomaly detector with known generalisation limits.

---

## Dataset

**Case Western Reserve University (CWRU) Bearing Fault Dataset**
- Drive-end and fan-end accelerometers, sampled at 12 kHz
- **7 fault classes:** `DE_ball`, `DE_inner_race`, `DE_outer_race`, `FE_ball`, `FE_inner_race`, `FE_outer_race`, `normal`
- Fault diameters: 0.007", 0.014", 0.021", 0.028" (class-dependent)
- Motor loads: 0 HP, 1 HP, 2 HP, 3 HP

**Dataset after VAE augmentation:**

| Class | Bearing | Real Windows | Synthetic Added | Final Count |
|---|---|---|---|---|
| DE_ball | Drive-End | 1,894 | 583 | 2,477 |
| DE_inner_race | Drive-End | 1,893 | 584 | 2,477 |
| DE_outer_race | Drive-End | 1,425 | 1,052 | 2,477 |
| FE_ball | Fan-End | 1,416 | 1,061 | 2,477 |
| FE_inner_race | Fan-End | 1,416 | 1,061 | 2,477 |
| FE_outer_race | Fan-End | 2,477 | 0 | 2,477 |
| normal | DE+FE | 1,424 | 1,053 | 2,477 |
| **TOTAL** | | **11,945** | **5,394** | **17,339** |

Place downloaded `.mat` files in `data/raw/`.

---

## Pipeline

```
data/raw/*.mat
      │
      ▼
01_load_and_preprocess.py     →  data/processed/features_expanded.csv  (7-class, 11,945 windows)
      │
      ▼
02_train_models_expanded.py   →  model_results_expanded.csv, noise_robustness_expanded.csv
      │
      ▼
04_vae_augmentation.py        →  features_augmented.csv (17,339 windows)
                                  models/vae_class_*/encoder.keras + decoder.keras
      │
      ▼
05_dl_comparison.py           →  dl_results.csv, raw_cnn_results.csv,
                                  autoencoder_results.csv
                                  models/cnn_best_fold.keras
      │
      ▼
06_expanded_shap.py           →  shap_consensus.csv, shap_methods_note.txt
                                  figures/shap_*.png
      │
      ▼  (new scripts — run after 06)
07_vae_ablation.py            →  vae_ablation_results.csv   (Table 7)
08_severity_generalisation.py →  severity_gen_results.csv,  (Table 8, Figure 8)
                                  noise_crossover_fine.csv
09_autoencoder_hard.py        →  autoencoder_hard_results.csv  (Table 5 appendix)
10_signal_snr_analysis.py     →  signal_snr_results.csv        (Figure 9)
      │
      ▼
paper/generate_paper_assets.py →  paper/tables/  (8 tables)
                                   paper/figures/ (9 IEEE figures)
```

**Extracted features (15 total):**

| Domain | Features |
|---|---|
| Time domain (9) | Mean, Std Dev, RMS, Peak, Peak-to-Peak, Crest Factor, Kurtosis, Skewness, Shape Factor |
| Frequency domain (5) | Spectral Centroid, Dominant Frequency, Low/Mid/High Band Energy |
| Operating condition (1) | Motor Load (HP) |

---

## Repository Structure

```
windproject/
├── data/
│   ├── raw/                              ← CWRU .mat files (not tracked)
│   └── processed/
│       ├── features.csv                  ← 3-class baseline (5,926 windows)
│       ├── features_expanded.csv         ← 7-class real data (11,945 windows × 20 cols)
│       ├── features_augmented.csv        ← 7-class + VAE synthetic (17,339 windows)
│       ├── model_results_expanded.csv    ← 5-fold CV accuracy per classical model
│       ├── noise_robustness_expanded.csv ← Accuracy vs noise level (0–30%)
│       ├── dl_results.csv                ← CNN 5-fold CV results
│       ├── autoencoder_results.csv       ← Per-class anomaly detection rates
│       └── shap_consensus.csv            ← Feature ranks across all 4 models
├── figures/                              ← Pipeline output figures (20 PNGs)
├── models/
│   ├── cnn_best_fold.keras               ← Best 1D-CNN fold weights
│   └── vae_class_*/                      ← Per-class VAE encoder + decoder
│       ├── encoder.keras
│       └── decoder.keras
├── paper/
│   ├── generate_paper_assets.py          ← Generates all tables and figures
│   ├── tables/
│   │   ├── table1_dataset_statistics.*   ← Class distribution + augmentation stats
│   │   ├── table2_classical_ml_performance.* ← Per-class F1 for RF/XGB/KNN/SVM
│   │   ├── table3_ml_vs_dl_comparison.*  ← All 6 models head-to-head
│   │   ├── table4_cnn_crossval.*         ← CNN fold-by-fold results
│   │   ├── table5_autoencoder_detection.*← Per-class detection rates + FPR
│   │   ├── table6_shap_consensus.*       ← Top-10 features, rank-based consensus
│   ├── table7_vae_ablation.*         ← VAE ablation: real-only vs real+synthetic
│   └── table8_severity_gen.*         ← Per-class severity generalisation gap
│   └── figures/
│       ├── fig1_class_distribution.*     ← Real vs augmented class counts
│       ├── fig2_model_comparison.*       ← Accuracy + macro F1 grouped bar chart
│       ├── fig3_f1_heatmap.*             ← 7-class × 5-model F1 heatmap
│       ├── fig4_noise_robustness.*       ← Accuracy vs noise, all 5 models
│       ├── fig5_shap_consensus.*         ← Rank heatmap (green=best, red=worst)
│       ├── fig6_fault_size_degradation.* ← Macro F1 vs fault diameter
│       ├── fig7_autoencoder_violin.*     ← Reconstruction error distributions
│       ├── fig8_severity_gen_gap.*       ← Per-class gen gap + noise crossover
│       └── fig9_fe_vs_de_signal_analysis.* ← Physics: DE vs FE signal quality
├── src/
│   ├── 01_load_and_preprocess.py         ← Signal loading, 15-feature extraction
│   ├── 02_train_models_expanded.py       ← RF, XGBoost, SVM, KNN — 5-fold CV
│   ├── 03_shap_analysis.py               ← Legacy SHAP (3-class)
│   ├── 04_vae_augmentation.py            ← VAE per-class data balancing
│   ├── 05_dl_comparison.py               ← Feature CNN + Raw-signal CNN (GAP 1)
│   ├── 06_expanded_shap.py               ← Multi-model SHAP, rank consensus (GAP 4)
│   ├── 07_vae_ablation.py                ← VAE ablation study (GAP 2)
│   ├── 08_severity_generalisation.py     ← Severity gen gap + noise crossover (GAP 3)
│   ├── 09_autoencoder_hard.py            ← Autoencoder reframing (GAP 5)
│   ├── 10_signal_snr_analysis.py         ← FE_ball physics analysis (GAP 6)
│   ├── load_cwru_files.py                ← CWRU file loader utility
│   └── save_features_expanded.py         ← Legacy feature save helper
└── README.md
```

Each table is saved as both `.csv` (machine-readable) and `.txt` (LaTeX-ready, column-aligned). Each figure is saved as both `.pdf` (for LaTeX inclusion) and `.png` (for preview).

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn
pip install tensorflow tensorflow-metal   # for CNN and VAE (Apple Silicon)

# 2. Place CWRU .mat files in:
#      data/raw/DE_12k/   ← drive-end fault .mat files
#      data/raw/FE_12k/   ← fan-end fault .mat files
#      data/raw/normal/   ← normal baseline .mat files

# 3. Run core pipeline (scripts 01–06)
python src/01_load_and_preprocess.py
python src/02_train_models_expanded.py
python src/04_vae_augmentation.py
python src/05_dl_comparison.py          # includes raw-signal CNN (GAP 1)
python src/06_expanded_shap.py          # rank-based consensus (GAP 4)

# 4. Run rigour/ablation scripts (scripts 07–10)
python src/07_vae_ablation.py           # VAE ablation → Table 7 (GAP 2)
python src/08_severity_generalisation.py # severity gen + crossover → Table 8, Fig 8 (GAP 3)
python src/09_autoencoder_hard.py       # autoencoder reframing (GAP 5)
python src/10_signal_snr_analysis.py    # FE_ball physics → Figure 9 (GAP 6)

# 5. Generate all 8 tables and 9 IEEE figures
python paper/generate_paper_assets.py
```

> **Note for Apple Silicon:** TensorFlow-Metal GPU can cause silent BatchNorm gradient bugs. Scripts 05, 07, 09 force CPU by setting `tf.config.set_visible_devices([], "GPU")`. This is intentional.

Each script is self-contained and prints a summary on completion.

---

## Results Summary

**Classical ML (7-class, 5-fold CV on 17,339 augmented samples):**

| Model | Accuracy | Macro F1 | Hardest Class |
|---|---|---|---|
| XGBoost | 99.10% | 0.9917 | FE_ball (F1 0.979) |
| Random Forest | 98.54% | 0.9856 | FE_ball (F1 0.968) |
| KNN | 94.98% | 0.9497 | FE_ball (F1 0.870) |
| SVM | 92.83% | 0.9283 | FE_outer_race (F1 0.900) |

**Deep Learning:**
- 1D-CNN: 97.95% ± 0.41% (5-fold CV); best fold 98.49%
- Autoencoder: 100% fault detection rate across all 6 fault classes; FPR 1.33%

**Noise robustness at 30% noise level:**
- SVM degrades least (92.83% → 77.77%)
- XGBoost degrades most steeply among classical models (99.10% → 57.81%)
- CNN holds to 64.34% at 30% noise
- SVM surpasses XGBoost at a precise noise crossover threshold (reported in Figure 8b)

**VAE Ablation (Table 7):** Accuracy and macro F1 under real-only vs real+synthetic conditions per model. Improvement Δ reported; small Δ (<0.005) is reframed as data balancing, not performance gain.

**Severity Generalisation (Table 8):** Per-class gap for each model when training on small severities (0.007"+0.014") and testing on unseen 0.021". Largest gaps indicate classes with poor severity generalisation.

**Autoencoder Reframing:** The reported 100% detection rate is within-distribution. Separation ratio quantifies trivial separability. Detection on 0.021" (trained on 0.007" only) is the meaningful metric.

**SHAP (Rank-based):** High Band Energy is the only feature in the top-5 of all four models. Cross-model SHAP magnitude comparisons are NOT made (different explainer types: TreeExplainer, PermutationExplainer, KernelExplainer, GradientExplainer).

**FE_ball Physics:** Signal SNR analysis (Figure 9) quantifies DE/FE path attenuation in dB, explaining why FE_ball is the hardest class across all models.

---

## Requirements

- Python 3.10+
- pandas, numpy, scipy, scikit-learn, xgboost, shap, matplotlib, seaborn
- tensorflow, tensorflow-metal (Apple Silicon) — for CNN and VAE scripts

---

## Citation

If you use this code, please cite the associated paper:

> [Authors]. "Wind Turbine Gearbox Fault Detection Using Vibration Signal Analysis with Machine Learning and SHAP Explainability." [Journal/Conference], [Year].
