# Wind Turbine Gearbox Fault Detection Using Vibration Signal Analysis with Machine Learning and SHAP Explainability

A complete research pipeline for bearing fault detection using the CWRU dataset. The project makes two central contributions: exposing a systematic flaw in how fault detection models are evaluated, and using SHAP to explain what the models actually learn.

---

## Central Research Findings

### Finding 1 — The CV/Generalisation Gap

Standard 5-fold cross-validation on the CWRU dataset produces near-perfect accuracy. This is widely reported in the literature, but it is misleading — the model has seen all fault severities during training, so it is not being tested on anything truly new.

This project introduces a **severity generalisation test**: train on fault severities 0.007" and 0.014", then test on 0.021" (a severity the model has never encountered). This simulates real-world deployment where fault severity is unknown.

| Model | CV Accuracy | Generalisation Accuracy | Gap |
|---|---|---|---|
| Random Forest | 99.61% | 88.93% | **−10.68%** |
| XGBoost | 99.65% | 80.23% | **−19.41%** |
| SVM | 99.87% | 88.58% | **−11.29%** |

The gap — up to **19.4 percentage points** — is the central argument of the paper: **evaluation methodology matters as much as model choice**.

### Finding 2 — Severity-Dependent Feature Importance (SHAP)

TreeSHAP analysis reveals that which vibration features matter most changes as the fault worsens. The three features that shift most from severity 0.007" → 0.021":

| Feature | Severity 0.007" | Severity 0.021" | Δ | Direction |
|---|---|---|---|---|
| Shape Factor | 0.485 | 0.714 | **+0.229** | ↑ increases with severity |
| High Band Energy | 0.784 | 0.633 | **−0.151** | ↓ decreases with severity |
| Mean | 0.409 | 0.536 | **+0.127** | ↑ increases with severity |

As faults worsen, the signal shifts from high-frequency spectral content toward time-domain amplitude statistics. This has implications for which sensors and features are most useful for early vs. advanced fault detection.

### Top Discriminating Features Per Fault Class (XGBoost SHAP)

| Fault Class | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| Ball | Mid Band Energy | Mean | Std Dev |
| Inner Race | Peak-to-Peak | Low Band Energy | High Band Energy |
| Outer Race | Peak-to-Peak | Mid Band Energy | Spectral Centroid |
| Normal | High Band Energy | Spectral Centroid | Std Dev |

---

## Dataset

**Case Western Reserve University (CWRU) Bearing Fault Dataset**
- Drive-end accelerometer, sampled at 12 kHz (`DE_time` channel)
- 4 fault classes: `normal`, `inner_race`, `ball`, `outer_race`
- 3 fault severities: 0.007", 0.014", 0.021" (fault diameter)
- 4 motor loads: 0 HP, 1 HP, 2 HP, 3 HP

**Required file numbers:**

| Fault Type | 0.007" | 0.014" | 0.021" |
|---|---|---|---|
| Normal | 97–100 | — | — |
| Inner Race | 105–108 | 169–172 | 209–212 |
| Ball | 118–121 | 185–188 | 222–225 |
| Outer Race | 130–133 | 197–200 | 234–237 |

Place downloaded `.mat` files in `data/raw/`.

---

## Pipeline

```
data/raw/*.mat
      │
      ▼
01_load_and_preprocess.py   →  data/processed/features.csv  (5,926 windows × 15 features)
      │
      ▼
02_train_models.py          →  figures/ (5 plots)  +  data/processed/ (3 CSVs)
      │
      ▼
03_shap_analysis.py         →  figures/ (5 SHAP plots)  +  console research findings
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
│   ├── raw/                        ← CWRU .mat files (not tracked)
│   └── processed/
│       ├── features.csv            ← 5,926 windows × 15 features
│       ├── model_results.csv       ← CV vs generalisation accuracy per model
│       ├── generalization_results.csv ← Per-class F1 for both evaluations
│       └── noise_robustness.csv    ← Accuracy under Gaussian noise
├── figures/
│   ├── confusion_matrix_grid.png   ← 3×2 grid (3 models × 2 evaluations)
│   ├── per_class_f1_chart.png      ← Per-class F1: CV vs generalisation
│   ├── accuracy_comparison.png     ← The performance gap — key figure
│   ├── noise_robustness.png        ← Accuracy under 0–30% Gaussian noise
│   ├── severity_performance.png    ← Per-severity accuracy breakdown
│   ├── shap_rf_importance.png      ← RF global SHAP importance
│   ├── shap_xgb_importance.png     ← XGBoost global SHAP importance
│   ├── shap_importance_comparison.png ← RF vs XGBoost side-by-side
│   ├── shap_xgb_class_heatmap.png  ← Per-class SHAP heatmap
│   └── shap_severity_impact.png    ← Feature importance shift by severity
├── src/
│   ├── 01_load_and_preprocess.py
│   ├── 02_train_models.py
│   └── 03_shap_analysis.py
└── README.md
```

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn

# 2. Place CWRU .mat files in data/raw/

# 3. Run in order
python src/01_load_and_preprocess.py
python src/02_train_models.py
python src/03_shap_analysis.py
```

Each script is self-contained and prints a summary report on completion.

---

## Results Summary

**Data:** 5,926 windows extracted from 40 `.mat` files, 1,024 samples each, zero overlap.
Window distribution: ~473–476 windows per fault/severity group (well-balanced), 1,656 normal windows.

**Models:** Random Forest (200 trees), XGBoost (200 estimators, depth 6), SVM (RBF kernel, C=10, StandardScaler pipeline). All trained with `RANDOM_STATE = 42`.

**Evaluation 1 — 5-Fold Stratified CV:** All three models exceed 99.6% accuracy, consistent with published CWRU benchmarks.

**Evaluation 2 — Severity Generalisation:** Performance drops to 80–89%, with XGBoost showing the largest degradation (−19.4%). This is the realistic deployment scenario.

**Noise robustness:** All models were additionally evaluated under Gaussian noise at 0%, 5%, 10%, 20%, and 30% of feature standard deviation on the generalisation test set.

**SHAP:** TreeExplainer applied to a 2,000-sample representative subset. Peak-to-Peak amplitude and mid-band energy dominate for structural faults (inner/outer race, ball). Shape Factor shows the largest severity-driven shift, rising 47% from 0.007" to 0.021" severity.

---

## Requirements

- Python 3.10+
- pandas, numpy, scipy, scikit-learn, xgboost, shap, matplotlib, seaborn

---

## Citation

If you use this code, please cite the associated paper:

> [Authors]. "Wind Turbine Gearbox Fault Detection Using Vibration Signal Analysis with Machine Learning and SHAP Explainability." [Journal/Conference], [Year].
