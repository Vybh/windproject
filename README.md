# Wind Turbine Gearbox Fault Detection Using Vibration Signal Analysis with Machine Learning and SHAP Explainability

A complete research pipeline for 7-class bearing fault detection using the CWRU dataset. The project combines classical ML, deep learning, VAE-based data augmentation, and multi-model SHAP explainability, and produces publication-ready tables and figures for an IEEE paper.

---

## Central Research Findings

### Finding 1 — Classical ML Outperforms 1D-CNN on Engineered Features

XGBoost achieves the highest 5-fold CV accuracy (99.10%) — outperforming a 1D-CNN (97.95%) on the same feature set. This challenges the assumption that deep learning is always superior for fault detection.

| Model | Type | Accuracy (5-fold CV) | Macro F1 |
|---|---|---|---|
| XGBoost | Classical ML | **99.10%** | **0.9917** |
| Random Forest | Classical ML | 98.54% | 0.9856 |
| 1D-CNN | Deep Learning | 97.95% | — |
| KNN | Classical ML | 94.98% | 0.9497 |
| SVM | Classical ML | 92.83% | 0.9283 |
| Autoencoder | Unsupervised | 100%† | N/A |

† Mean fault detection rate; false positive rate on normal class = 1.33%.

### Finding 2 — The CV/Generalisation Gap

Standard 5-fold CV on CWRU produces near-perfect accuracy because the model has seen all fault severities during training. A **severity generalisation test** (train on 0.007" and 0.014", test on 0.021") reveals a gap of up to **19.4 percentage points**, which is the core methodological argument of the paper.

### Finding 3 — High Band Energy is the Universal Discriminating Feature

Multi-model SHAP consensus across RF, XGBoost, SVM, and CNN identifies **High Band Energy** as the only feature ranked in the top 5 by all four models simultaneously (mean rank 2.75). Spectral Centroid and Mean follow closely.

| Feature | RF Rank | XGBoost Rank | SVM Rank | CNN Rank | Mean Rank | Consensus |
|---|---|---|---|---|---|---|
| High Band Energy | 2 | 1 | 5 | 3 | 2.75 | ★ All-model top-5 |
| Spectral Centroid | 4 | 6 | 1 | 2 | 3.25 | |
| Mean | 1 | 2 | 6 | 5 | 3.50 | |
| Mid Band Energy | 3 | 8 | 2 | 1 | 3.50 | |

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
01_load_and_preprocess.py     →  data/processed/features.csv  (3-class, initial)
      │
      ▼
save_features_expanded.py     →  data/processed/features_expanded.csv  (7-class, 11,945 windows)
      │
      ▼
02_train_models_expanded.py   →  model_results_expanded.csv, noise_robustness_expanded.csv
      │
      ▼
04_vae_augmentation.py        →  features_augmented.csv (17,339 windows)
                                  models/vae_class_*/encoder.keras + decoder.keras
      │
      ▼
05_dl_comparison.py           →  dl_results.csv, autoencoder_results.csv
                                  models/cnn_best_fold.keras
      │
      ▼
06_expanded_shap.py           →  shap_consensus.csv  +  figures/shap_*.png
      │
      ▼
paper/generate_paper_assets.py →  paper/tables/  (6 tables)
                                   paper/figures/ (7 IEEE figures)
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
│   │   └── table6_shap_consensus.*       ← Top-10 features, all models ranked
│   └── figures/
│       ├── fig1_class_distribution.*     ← Real vs augmented class counts
│       ├── fig2_model_comparison.*       ← Accuracy + macro F1 grouped bar chart
│       ├── fig3_f1_heatmap.*             ← 7-class × 5-model F1 heatmap
│       ├── fig4_noise_robustness.*       ← Accuracy vs noise, all 5 models
│       ├── fig5_shap_consensus.*         ← Rank heatmap (green=best, red=worst)
│       ├── fig6_fault_size_degradation.* ← Macro F1 vs fault diameter
│       └── fig7_autoencoder_violin.*     ← Reconstruction error distributions
├── src/
│   ├── 01_load_and_preprocess.py
│   ├── 02_train_models_expanded.py
│   ├── 04_vae_augmentation.py
│   ├── 05_dl_comparison.py
│   ├── 06_expanded_shap.py
│   ├── load_cwru_files.py
│   └── save_features_expanded.py
└── README.md
```

Each table is saved as both `.csv` (machine-readable) and `.txt` (LaTeX-ready, column-aligned). Each figure is saved as both `.pdf` (for LaTeX inclusion) and `.png` (for preview).

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn
pip install tensorflow tensorflow-metal   # for CNN and VAE (Apple Silicon)

# 2. Place CWRU .mat files in data/raw/

# 3. Run pipeline in order
python src/01_load_and_preprocess.py
python src/save_features_expanded.py
python src/02_train_models_expanded.py
python src/04_vae_augmentation.py
python src/05_dl_comparison.py
python src/06_expanded_shap.py

# 4. Generate paper tables and figures
python paper/generate_paper_assets.py
```

> **Note for Apple Silicon:** TensorFlow-Metal GPU can cause silent BatchNorm gradient bugs. Script 05 forces CPU by setting `CUDA_VISIBLE_DEVICES=-1`. This is intentional.

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

**SHAP consensus:** High Band Energy is the only feature in the top-5 of all four models.

---

## Requirements

- Python 3.10+
- pandas, numpy, scipy, scikit-learn, xgboost, shap, matplotlib, seaborn
- tensorflow, tensorflow-metal (Apple Silicon) — for CNN and VAE scripts

---

## Citation

If you use this code, please cite the associated paper:

> [Authors]. "Wind Turbine Gearbox Fault Detection Using Vibration Signal Analysis with Machine Learning and SHAP Explainability." [Journal/Conference], [Year].
