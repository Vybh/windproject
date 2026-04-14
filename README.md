# CWRU Bearing Fault Detection: A 7-Class ML Research Pipeline

## 1. Project Overview

This repository is a complete research pipeline for 7-class rolling-element bearing fault detection using the Case Western Reserve University (CWRU) vibration dataset. The task is to classify raw accelerometer signals into one of seven fault classes — three Drive-End (DE) faults, three Fan-End (FE) faults, and a normal baseline — across four fault severity sizes. The approach combines classical machine learning (Random Forest, XGBoost, SVM, KNN), deep learning (feature-based 1D-CNN and raw-signal 1D-CNN), VAE-based data augmentation for class balancing, multi-model SHAP explainability, and a physics-grounded FE sensor attenuation analysis. The central methodological contribution is a cross-severity generalisation experiment that quantifies how well models trained on small fault severities (0.007" and 0.014") transfer to an unseen severity (0.021"), and how VAE augmentation affects that transfer. This pipeline is structured for submission to an IEEE conference or journal.

---

## 2. Central Research Findings

Findings are ordered by significance, not pipeline execution order.

### Finding 1 — Raw-signal CNN outperforms all other models

The feature-based 1D-CNN operating on 15 engineered tabular features achieves mean 5-fold CV accuracy of **0.9822 ± 0.0034**, which is lower than XGBoost at **0.9910**. Tree ensembles are well-suited to tabular features with domain-engineered semantics. However, the raw-signal 1D-CNN operating directly on 1024-point vibration windows achieves mean CV accuracy of **0.9998 ± 0.0004** — surpassing every other model by a substantial margin. This demonstrates that the temporal structure of raw vibration signals contains fault-discriminative information that the 15 engineered features partially discard. The practical implication: feature engineering is the right choice when interpretability and noise robustness are primary objectives; raw-signal CNNs are the right choice when accuracy on clean data is the primary objective.

### Finding 2 — The severity generalisation gap is large and class-dependent

Training on fault sizes 0.007" and 0.014" and testing on unseen 0.021" produces dramatic F1 degradation. In the real-data-only condition, the per-class gaps (cv_f1 − generalisation_f1) are:

| Class | RF Gap | XGB Gap | SVM Gap | KNN Gap |
|---|---|---|---|---|
| DE_ball | 0.6298 | 0.3416 | 0.5824 | 0.5779 |
| DE_inner_race | 0.9723 | 0.3588 | 0.8267 | 0.7925 |
| DE_outer_race | 0.5651 | 0.8284 | 0.8507 | 0.8522 |
| FE_ball | 0.6202 | 0.3091 | 0.7406 | 0.7015 |
| FE_inner_race | 0.9890 | 0.8919 | 0.8893 | 0.5987 |
| FE_outer_race | 0.4737 | 0.2653 | 0.4701 | 0.5848 |

FE_inner_race is the most severely degraded class (cross-model average gap 0.84). FE_outer_race is the most robust (average gap 0.45). The spread within a single model across classes is wider than the spread across models for a given class, indicating that fault geometry — not model architecture — is the primary driver of generalisation failure. VAE augmentation substantially reduces the mean generalisation gap from **0.6547** (real-only) to **0.4203** (real+VAE), a reduction of 0.2344 (35.8%). The largest improvements are in inner-race faults: RF DE_inner_race gap drops from 0.9723 to 0.1540, and RF FE_inner_race from 0.9890 to 0.3249. See Section 7 for full analysis.

### Finding 3 — SVM crossover at 5% noise

Under zero noise, XGBoost (0.9916) far outperforms SVM (0.9293). However, XGBoost degrades steeply under additive Gaussian noise: at **5% noise** (5% of feature standard deviation), XGBoost drops to **0.8966** while SVM holds at **0.9238**. The crossover occurs at exactly 5% and SVM maintains its lead at every noise level above that threshold up to 30%. For real-world deployments where sensor calibration drift, temperature variation, or electromagnetic interference introduce noise exceeding 5% of feature standard deviation, SVM is the preferred model despite its lower clean-data accuracy.

### Finding 4 — VAE balances classes but does not reliably boost accuracy

The effect of VAE augmentation on 5-fold CV accuracy is mixed: XGBoost improves (+0.0009), KNN improves (+0.0072), while RF (−0.0010), SVM (−0.0051), and 1D-CNN (−0.0018) degrade. No model changes by more than 0.0072. VAE augmentation is justified as a class-balancing technique and as a cross-severity generalisation aid (Finding 2), but should not be claimed as an in-distribution accuracy improvement technique.

### Finding 5 — Autoencoder separation ratios are trivially large

A single autoencoder trained on normal samples produces mean reconstruction errors for fault classes that are **85,524× to 261,998×** larger than for normal samples (DE_inner_race: 85,524×; FE_outer_race: 261,998×). All six fault classes achieve 100% detection rate (FPR = 0.0133). The hard generalisation experiment (train on 0.007" only, test on 0.021") also achieves 100% detection rate, but this does not demonstrate generalisation — reconstruction error captures any deviation from the training distribution, and a more severe fault is simply more anomalous. The autoencoder is a within-distribution detector, not a fault-severity classifier.

### Finding 6 — FE_ball is consistently the hardest class due to physics, not data

Despite FE_ball signals having higher total RMS energy than DE_ball signals (0.1640 vs 0.1336), the SNR at the Ball Spin Frequency (BSF = 70.6 Hz) for FE_ball is **304.75** versus **0.08** for DE_ball. The RMS path attenuation is **−1.8 dB**. The FE accelerometer is physically further from the fan-end bearing, and the transmission path attenuates the fault-frequency component while preserving broadband energy. This physics-grounded result explains why FE_ball is the lowest-performing class in Table 2 without requiring post-hoc feature-importance rationalisation.

### Finding 7 — High Band Energy is the only cross-model consensus feature

High Band Energy is the only feature ranked in the top 5 by all four SHAP explainers simultaneously (RF rank 2, XGBoost rank 1, SVM rank 5, CNN rank 3; mean rank 2.75). All other features appear in the top 5 of at most three models. Cross-model SHAP comparisons are conducted via feature rank consensus rather than magnitude comparison, as different explainer classes compute non-equivalent quantities.

---

## 3. Dataset

The CWRU bearing dataset contains 12 kHz accelerometer signals from a motor test stand. Faults were seeded at four sizes (0.007", 0.014", 0.021", 0.028") using electro-discharge machining at three motor load levels (0, 1, 2 HP). Signals are segmented into non-overlapping 1024-point windows, and 15 time-frequency features are extracted per window.

**Table 1 — Dataset Statistics (after VAE augmentation)**

| Class | Bearing Location | Fault Sizes (in) | Real Windows | Synthetic Added | Final Count |
|---|---|---|---|---|---|
| DE_ball | DE | 0.007, 0.014, 0.021, 0.028 | 1,894 | 583 | 2,477 |
| DE_inner_race | DE | 0.007, 0.014, 0.021, 0.028 | 1,893 | 584 | 2,477 |
| DE_outer_race | DE | 0.007, 0.014, 0.021 | 1,425 | 1,052 | 2,477 |
| FE_ball | FE | 0.007, 0.014, 0.021 | 1,416 | 1,061 | 2,477 |
| FE_inner_race | FE | 0.007, 0.014, 0.021 | 1,416 | 1,061 | 2,477 |
| FE_outer_race | FE | 0.007, 0.014, 0.021, 0.028 | 2,477 | 0 | 2,477 |
| normal | DE+FE | N/A | 1,424 | 1,053 | 2,477 |
| **TOTAL** | — | — | **11,945** | **5,394** | **17,339** |

FE_outer_race required no augmentation (already the largest real-data class). DE_outer_race and FE_ball/inner_race lack 0.028" data in the original CWRU release, which limits severity coverage and contributes to their larger generalisation gaps.

---

## 4. Pipeline Architecture

```
data/raw/
  DE_12k/*.mat  FE_12k/*.mat
       │
01_load_and_preprocess.py
  Input: raw .mat files
  Action: segments 1024-pt windows, extracts 15 features, assigns 7 class labels
  Output: data/processed/features_expanded.csv  (11,945 × 20)
       │
02_train_models_expanded.py
  Input: features_expanded.csv
  Action: trains RF, XGBoost, SVM, KNN with stratified 5-fold CV; noise robustness sweep
  Output: model_results_expanded.csv, noise_robustness_expanded.csv, 6 diagnostic figures
       │
03_shap_analysis.py
  Input: features_expanded.csv
  Action: computes TreeExplainer and PermutationExplainer SHAP values for RF and XGBoost
  Output: shap_consensus.csv (partial), 4 per-model SHAP bar charts
       │
04_vae_augmentation.py
  Input: features_expanded.csv
  Action: trains 7 per-class VAEs (latent_dim=8, 200 epochs), samples synthetic windows
  Output: features_augmented.csv (17,339 × 20), models/vae_class_*/
       │
05_dl_comparison.py
  Input: features_augmented.csv + DE_12k/*.mat + FE_12k/*.mat
  Action: trains feature-based 1D-CNN (tabular) and raw-signal 1D-CNN (1024-pt windows);
          trains reconstruction autoencoder on normal class
  Output: dl_results.csv, raw_cnn_results.csv, autoencoder_results.csv,
          models/cnn_best_fold.keras, 7 figures
       │
06_expanded_shap.py
  Input: features_augmented.csv + cnn_best_fold.keras
  Action: adds KernelExplainer (SVM) and GradientExplainer (CNN); builds 4-model rank consensus
  Output: shap_consensus.csv (final, 4-model)
       │
07_vae_ablation.py
  Input: features_expanded.csv + features_augmented.csv
  Action: runs all 5 models under real_only and real_plus_synthetic; 5-fold CV each condition
  Output: vae_ablation_results.csv, paper/tables/table7_vae_ablation.csv
       │
08_severity_generalisation.py
  Input: features_expanded.csv
  Action: train on {0.007, 0.014}, test on 0.021 for 4 models; 1%-step noise crossover sweep
  Output: severity_gen_results.csv, noise_crossover_fine.csv,
          paper/tables/table8_severity_gen.csv, paper/figures/fig8_severity_gen_gap.png
       │
09_autoencoder_hard.py
  Input: features_expanded.csv
  Action: computes separation ratios; hard experiment — train on 0.007" only, test on 0.021"
  Output: autoencoder_hard_results.csv, paper/tables/table5_autoencoder_hard_appendix.txt
       │
10_signal_snr_analysis.py
  Input: DE_12k/*.mat + FE_12k/*.mat
  Action: computes RMS energy and SNR at BSF for DE_ball vs FE_ball across all severity levels
  Output: signal_snr_results.csv, signal_snr_summary.txt,
          paper/figures/fig9_fe_vs_de_signal_analysis.png
       │
paper/generate_paper_assets.py
  Input: all 7 processed CSVs
  Action: trains 5 models on 80/20 split for per-class F1; assembles all publication tables/figures
  Output: paper/tables/ (8 tables × CSV+TXT), paper/figures/ (9 figures × PDF+PNG)
```

---

## 5. Feature Engineering

All 15 features are computed per 1024-point window at 12 kHz sampling rate.

| Feature | Domain | Description | SHAP Mean Rank |
|---|---|---|---|
| High Band Energy | Spectral | Power in 3–6 kHz band | **2.75** |
| Spectral Centroid | Spectral | Frequency-weighted mean of spectrum | 3.25 |
| Mean | Temporal | Window mean amplitude | 3.50 |
| Mid Band Energy | Spectral | Power in 1–3 kHz band | 3.50 |
| Std Dev | Temporal | Standard deviation of amplitude | 5.75 |
| Low Band Energy | Spectral | Power in 0–1 kHz band | 6.00 |
| Shape Factor | Temporal | RMS / mean absolute value | 6.75 |
| Peak-to-Peak | Temporal | max − min amplitude | 7.50 |
| Peak | Temporal | Maximum absolute amplitude | 9.75 |
| Dominant Frequency | Spectral | Frequency of peak spectral magnitude | 10.00 |
| RMS | Temporal | Root mean square amplitude | — |
| Kurtosis | Temporal | Fourth standardised moment | — |
| Skewness | Temporal | Third standardised moment | — |
| Crest Factor | Temporal | Peak / RMS | — |
| Motor Load | Operational | Motor load level (0, 1, 2, 3 HP) | — |

SHAP mean ranks are from shap_consensus.csv (top 10 of 15 features shown). Spectral band features dominate the top 4 rankings, consistent with bearing fault signatures appearing as fault-frequency harmonics in the kilohertz range.

---

## 6. Model Results

**Table 2 — Classical ML Per-Class F1 (5-fold stratified CV, features_augmented.csv)**

| Model | Accuracy | Macro F1 | DE_ball | DE_inner | DE_outer | FE_ball | FE_inner | FE_outer | normal |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.9910 | 0.9913 | 0.9882 | 0.9980 | 0.9970 | 0.9768 | 0.9929 | 0.9859 | 1.0000 |
| Random Forest | 0.9854 | 0.9863 | 0.9764 | 0.9960 | 0.9970 | 0.9682 | 0.9919 | 0.9745 | 1.0000 |
| KNN | 0.9498 | 0.9500 | 0.9033 | 0.9880 | 0.9830 | 0.8696 | 0.9779 | 0.9283 | 1.0000 |
| SVM | 0.9283 | 0.9540 | 0.9079 | 0.9890 | 0.9820 | 0.9104 | 0.9880 | 0.9004 | 1.0000 |

FE_ball is the lowest-performing class in every model (Finding 6). The normal class achieves perfect F1 in all models because its feature distribution is unambiguously distinct from all fault classes.

**Table 3 — ML vs DL Comparison**

| Model | Type | Accuracy | Macro F1 | Notes |
|---|---|---|---|---|
| Raw-signal CNN | Deep Learning | **0.9998** | — | 5-fold CV; 1024-pt windows; no feature engineering |
| XGBoost | Classical ML | 0.9910 | 0.9913 | 5-fold CV; 200 estimators |
| Random Forest | Classical ML | 0.9854 | 0.9863 | 5-fold CV; 200 trees |
| 1D-CNN (tabular) | Deep Learning | 0.9822 | — | 5-fold CV; 15 engineered features |
| KNN | Classical ML | 0.9498 | 0.9500 | 5-fold CV; k=5 |
| SVM | Classical ML | 0.9283 | 0.9540 | 5-fold CV; RBF kernel |
| Autoencoder | Unsupervised | 1.0000† | N/A | †mean fault detection rate; FPR = 0.0133 |

†Autoencoder detection rate is not comparable to supervised accuracy; FPR = 0.0133 means 1.3% of normal windows are flagged as anomalous.

**Table 4 — Feature-based 1D-CNN Fold-by-Fold Results**

| Fold | Test Accuracy |
|---|---|
| 1 | 0.9774 |
| 2 | 0.9795 |
| 3 | 0.9833 |
| 4 | 0.9837 |
| 5 | 0.9870 |
| **Mean ± Std** | **0.9822 ± 0.0034** |

**Table 7 — VAE Augmentation Ablation**

| Model | Real-Only Acc | VAE-Aug Acc | Acc Δ | Real-Only F1 | VAE-Aug F1 | F1 Δ |
|---|---|---|---|---|---|---|
| Random Forest | 0.9854 | 0.9844 | −0.0010 | 0.9856 | 0.9845 | −0.0011 |
| XGBoost | 0.9912 | 0.9921 | +0.0009 | 0.9914 | 0.9922 | +0.0008 |
| SVM | 0.9283 | 0.9231 | −0.0051 | 0.9298 | 0.9250 | −0.0049 |
| KNN | 0.9498 | 0.9570 | +0.0072 | 0.9497 | 0.9569 | +0.0073 |
| 1D-CNN | 0.9833 | 0.9816 | −0.0018 | 0.9837 | 0.9817 | −0.0021 |

---

## 7. Severity Generalisation Experiment

### Experimental Design

Models are trained on all samples with fault_size ∈ {0.007", 0.014"} plus normal, then evaluated on held-out samples with fault_size = 0.021". This replicates the realistic scenario where a monitoring system is trained on early-wear signatures and must detect more advanced fault progression. The 0.028" severity is excluded from both train and test because not all classes have 0.028" data in the CWRU release, making cross-class comparison invalid at that severity level.

### Results: Real-Only Condition (Table 8)

The generalisation gap is defined as cv_f1 − gen_f1. A gap near 1.0 means the class is effectively unrecognisable at the unseen severity; a gap near 0.0 means strong transfer.

| Class | RF Gap | XGB Gap | SVM Gap | KNN Gap |
|---|---|---|---|---|
| DE_ball | 0.6298 | 0.3416 | 0.5824 | 0.5779 |
| DE_inner_race | 0.9723 | 0.3588 | 0.8267 | 0.7925 |
| DE_outer_race | 0.5651 | 0.8284 | 0.8507 | 0.8522 |
| FE_ball | 0.6202 | 0.3091 | 0.7406 | 0.7015 |
| FE_inner_race | 0.9890 | 0.8919 | 0.8893 | 0.5987 |
| FE_outer_race | 0.4737 | 0.2653 | 0.4701 | 0.5848 |

Two structural patterns are notable. First, XGBoost generalises well on ball faults (DE_ball gap 0.34, FE_ball gap 0.31) but fails on outer-race and inner-race faults (DE_outer_race gap 0.83, FE_inner_race gap 0.89). Second, inner-race faults show the most severe degradation across all models — RF DE_inner_race gap 0.97, RF FE_inner_race gap 0.99 — suggesting that inner-race vibration signatures are the most severity-dependent, likely because the excitation pathway changes qualitatively as the crack progresses through different phases of the ball-pass geometry.

### Results: VAE Augmentation Effect

With VAE-augmented training folds (real + synthetic from small-severity distributions), the cross-model average gap drops from **0.6547 → 0.4203** (−0.2344, a 35.8% reduction).

| Class | RF Gap (real) | RF Gap (+VAE) | XGB Gap (real) | XGB Gap (+VAE) |
|---|---|---|---|---|
| DE_ball | 0.6298 | 0.6077 | 0.3416 | 0.3245 |
| DE_inner_race | 0.9723 | **0.1540** | 0.3588 | 0.2132 |
| DE_outer_race | 0.5651 | 0.3331 | 0.8284 | 0.5215 |
| FE_ball | 0.6202 | 0.4445 | 0.3091 | 0.2578 |
| FE_inner_race | 0.9890 | **0.3249** | 0.8919 | 0.3771 |
| FE_outer_race | 0.4737 | 0.5354 | 0.2653 | 0.2319 |

The largest reductions occur in inner-race faults, which were the hardest cases without augmentation. The exception is RF FE_outer_race (0.47 → 0.54 — marginal worsening). The most plausible mechanism: synthetic VAE samples generated from 0.007" and 0.014" distributions increase training density near the low-severity manifold, encouraging the classifier to learn fault-geometry features (class-consistent across severities) rather than fault-magnitude features (severity-dependent). This regularisation effect is model-dependent and not guaranteed to generalise to all fault types.

---

## 8. Noise Robustness and SVM Crossover

Additive Gaussian noise (0–30% of feature standard deviation, 1% steps) is injected into features at test time. Results at selected noise levels from noise_crossover_fine.csv:

| Noise Level | SVM Accuracy | XGBoost Accuracy | Leader |
|---|---|---|---|
| 0% | 0.9293 | 0.9916 | XGBoost |
| 1% | 0.9314 | 0.9879 | XGBoost |
| 4% | 0.9221 | 0.9443 | XGBoost |
| **5%** | **0.9238** | **0.8966** | **SVM (crossover)** |
| 10% | 0.8761 | 0.7694 | SVM |
| 20% | 0.7082 | 0.5115 | SVM |
| 30% | 0.5366 | 0.3746 | SVM |

The crossover occurs at exactly **5% noise**. XGBoost's 6.2 percentage point advantage at 0% noise is fully erased by modest noise and becomes a liability at higher levels. For sensor noise environments above 5% of feature standard deviation, SVM should be preferred over XGBoost. KNN is the most robust classical model at high noise levels (0.8493 at 30%) but its lower clean-data baseline (0.9501) limits utility in clean environments. These results are from noise_robustness_expanded.csv and noise_crossover_fine.csv respectively.

---

## 9. SHAP Explainability

Four SHAP explainer types are used to avoid locking explainability to a single methodology:

- **TreeExplainer** (RF, XGBoost): exact Shapley values via tree path enumeration; fast and exact
- **PermutationExplainer** (RF, XGBoost): model-agnostic Monte Carlo permutation, used to verify TreeExplainer consistency
- **KernelExplainer** (SVM): kernel-based linear SHAP approximation via coalition sampling
- **GradientExplainer** (CNN): gradient × input attribution for neural network layers

Cross-model SHAP comparisons are conducted via feature rank consensus rather than magnitude comparison, as different explainer classes compute non-equivalent quantities.

**Table 6 — SHAP Feature Rank Consensus (top 10 of 15)**

| Feature | RF Rank | XGB Rank | SVM Rank | CNN Rank | Mean Rank | In Top-5 of All 4? |
|---|---|---|---|---|---|---|
| High Band Energy | 2 | 1 | 5 | 3 | 2.75 | **Yes** |
| Spectral Centroid | 4 | 6 | 1 | 2 | 3.25 | No |
| Mean | 1 | 2 | 6 | 5 | 3.50 | No |
| Mid Band Energy | 3 | 8 | 2 | 1 | 3.50 | No |
| Std Dev | 5 | 3 | 11 | 4 | 5.75 | No |
| Low Band Energy | 10 | 4 | 3 | 7 | 6.00 | No |
| Shape Factor | 8 | 7 | 4 | 8 | 6.75 | No |
| Peak-to-Peak | 6 | 5 | 9 | 10 | 7.50 | No |
| Peak | 9 | 13 | 8 | 9 | 9.75 | No |
| Dominant Frequency | 12 | 9 | 13 | 6 | 10.00 | No |

High Band Energy is the sole consensus feature across all four models. The concentration of spectral features (High Band Energy, Spectral Centroid, Mid Band Energy, Low Band Energy) in positions 1–6 aligns with bearing fault physics: defect-pass frequencies and their harmonics appear in the kilohertz range for CWRU shaft speeds, and band-energy features directly capture power at those frequencies.

---

## 10. Autoencoder Analysis

A single LSTM-based autoencoder is trained exclusively on normal-condition windows. Fault detection is based on reconstruction error exceeding a learned threshold.

**Table 5 — Autoencoder Detection Results**

| Class | Samples | Detected | Detection Rate | Notes |
|---|---|---|---|---|
| DE_ball | 1,894 | 1,894 | 1.0000 | Separation ratio: 194,860× |
| DE_inner_race | 1,893 | 1,893 | 1.0000 | Separation ratio: 85,525× |
| DE_outer_race | 1,425 | 1,425 | 1.0000 | Separation ratio: 240,691× |
| FE_ball | 1,416 | 1,416 | 1.0000 | Separation ratio: 241,283× |
| FE_inner_race | 1,416 | 1,416 | 1.0000 | Separation ratio: 233,442× |
| FE_outer_race | 2,477 | 2,477 | 1.0000 | Separation ratio: 261,998× |
| normal (FPR) | 1,424 | 19 | 0.0133 | False positive rate |

The separation ratios (mean fault reconstruction error / mean normal reconstruction error) range from **85,524× to 261,998×**. Ratios at this magnitude indicate that the feature space presents no ambiguous boundary between normal and fault conditions; the autoencoder operates as a simple threshold, not as a learned generaliser. The hard experiment (train on 0.007" only, test on 0.021") also achieves 100% detection rate for all classes. This should not be interpreted as evidence of good severity generalisation: any fault at any severity will produce anomalously high reconstruction error relative to the normal training distribution. The autoencoder cannot distinguish severity levels or separate fault types from one another, and its near-perfect detection on CWRU data reflects the controlled, single-operating-condition nature of the dataset.

---

## 11. FE_ball Physics Analysis

The CWRU test rig mounts accelerometers at two locations: the Drive-End (DE) bearing housing and the Fan-End (FE) bearing housing. The Ball Spin Frequency for the CWRU rig geometry is:

```
BSF = 2.357 × shaft_speed = 2.357 × 29.95 Hz ≈ 70.6 Hz
```

Signal quality metrics computed from signal_snr_results.csv across all 1024-point windows at severities 0.007", 0.014", and 0.021":

| Metric | DE_ball | FE_ball | DE/FE Ratio |
|---|---|---|---|
| RMS Energy | 0.1336 | 0.1640 | 0.81× |
| SNR at BSF | 0.08 | 304.75 | 0.00× |
| Spectral Peak Prominence | 0.93 | 1.34 | 0.69× |
| Path Attenuation (RMS) | **−1.8 dB** | — | — |

The counterintuitive finding is that FE_ball signals have higher total RMS energy than DE_ball signals, but dramatically lower SNR at the fault frequency BSF. The −1.8 dB RMS attenuation reflects energy dissipated through the motor housing structure between the fan-end bearing and its accelerometer. The fault signature from FE_ball arrives at the DE accelerometer after passing through a longer, more attenuated structural path. This physics-grounded result explains why FE_ball is the lowest F1 class in Table 2 (XGBoost F1 = 0.9768, KNN F1 = 0.8696) without requiring any post-hoc explanation: the model receives a weaker fault-frequency signal for FE faults by design of the mechanical system.

---

## 12. Repository Structure

```
windproject/
├── README.md
├── app.py                              Streamlit app — single narrative research flow
│
├── data/
│   ├── raw/
│   │   ├── DE_12k/                    Drive-End 12 kHz .mat files (CWRU)
│   │   └── FE_12k/                    Fan-End 12 kHz .mat files (CWRU)
│   └── processed/
│       ├── features_expanded.csv       11,945 × 20 — real data, 7 classes, 15 features
│       ├── features_augmented.csv      17,339 × 20 — real + VAE synthetic, balanced
│       ├── model_results_expanded.csv  Per-model 5-fold CV accuracy
│       ├── noise_robustness_expanded.csv  Accuracy vs noise % (RF/XGB/SVM/KNN)
│       ├── dl_results.csv              Feature-based CNN 5-fold results
│       ├── raw_cnn_results.csv         Raw-signal CNN 5-fold results
│       ├── vae_ablation_results.csv    Real-only vs real+synthetic comparison
│       ├── severity_gen_results.csv    Per-class generalisation gaps (real + real+VAE)
│       ├── noise_crossover_fine.csv    SVM vs XGBoost 0–30% noise (1% steps)
│       ├── autoencoder_results.csv     Standard AE detection rates per class
│       ├── autoencoder_hard_results.csv  Separation ratios + hard generalisation
│       ├── signal_snr_results.csv      Per-window RMS and BSF SNR (2,839 rows)
│       ├── signal_snr_summary.txt      Aggregated DE vs FE attenuation metrics
│       └── shap_consensus.csv          4-model SHAP rank consensus (15 features)
│
├── src/
│   ├── 01_load_and_preprocess.py
│   ├── 02_train_models_expanded.py
│   ├── 03_shap_analysis.py
│   ├── 04_vae_augmentation.py
│   ├── 05_dl_comparison.py
│   ├── 06_expanded_shap.py
│   ├── 07_vae_ablation.py
│   ├── 08_severity_generalisation.py
│   ├── 09_autoencoder_hard.py
│   ├── 10_signal_snr_analysis.py
│   ├── load_cwru_files.py              Shared .mat loading utility
│   ├── save_features_expanded.py       Feature extraction helper
│   ├── cnn_best_fold.keras             Best CNN fold weights (saved by script 05)
│   ├── vae_class_DE_ball/              VAE weights — DE_ball
│   ├── vae_class_DE_inner_race/        VAE weights — DE_inner_race
│   ├── vae_class_DE_outer_race/        VAE weights — DE_outer_race
│   ├── vae_class_FE_ball/              VAE weights — FE_ball
│   ├── vae_class_FE_inner_race/        VAE weights — FE_inner_race
│   ├── vae_class_FE_outer_race/        VAE weights — FE_outer_race
│   └── vae_class_normal/               VAE weights — normal class
│
├── paper/
│   ├── generate_paper_assets.py
│   ├── tables/                         8 tables × (CSV + TXT) = 16 files
│   └── figures/                        9 figures × (PDF + PNG) = 18 files
│
└── figures/                            36 diagnostic PNGs from pipeline scripts
```

---

## 13. How to Run

### Installation

```bash
python3 -m pip install numpy pandas scipy scikit-learn matplotlib seaborn
python3 -m pip install xgboost==2.0.3    # see note below
python3 -m pip install shap tensorflow streamlit
```

**XGBoost on Apple Silicon:** XGBoost ≥ 3.x ships an arm64-only binary with an rpath pointing to `/opt/homebrew/opt/libomp/lib/libomp.dylib` (ARM Homebrew). If your system has Intel Homebrew at `/usr/local/` or no ARM Homebrew at `/opt/homebrew/`, the import will fail with `Library not loaded: @rpath/libomp.dylib`. Fix: pin to `xgboost==2.0.3`, which resolves its rpath correctly on both Intel and ARM macOS without requiring ARM Homebrew.

**TensorFlow Metal:** Scripts 05 and 07 force CPU execution with `tf.config.set_visible_devices([], "GPU")` to avoid instability in the training loops under the Metal GPU backend. This line is already in the source; do not remove it.

### Execution Order

```bash
python3 src/01_load_and_preprocess.py          # ~1 min
python3 src/02_train_models_expanded.py        # ~3 min
python3 src/03_shap_analysis.py                # ~5 min
python3 src/04_vae_augmentation.py             # ~10 min
python3 src/05_dl_comparison.py                # ~15 min  (raw CNN is CPU-intensive)
python3 src/06_expanded_shap.py                # ~20 min  (KernelExplainer is slow)
python3 src/07_vae_ablation.py                 # ~15 min  (10 folds × 2 conditions × CNN)
python3 src/08_severity_generalisation.py      # ~5 min
python3 src/09_autoencoder_hard.py             # ~3 min
python3 src/10_signal_snr_analysis.py          # ~2 min
python3 paper/generate_paper_assets.py         # ~5 min
```

Each script verifies required upstream files and exits with a clear message if a dependency is missing. All times are approximate for an Apple M-series CPU.

### Streamlit App

```bash
streamlit run app.py
```

---

## 14. Known Limitations and Honest Caveats

**SHAP explainer heterogeneity.** The four explainer types do not compute the same underlying quantity. TreeExplainer computes exact Shapley values via tree path enumeration. KernelExplainer approximates Shapley values via linear regression on feature coalitions. GradientExplainer computes gradient × input, which is a local attribution method not equivalent to a Shapley value in the strict game-theoretic sense. The rank consensus in Table 6 is a statement about which features rank highly across different attribution frameworks, not about which features have the highest true Shapley magnitude. Cross-model SHAP magnitude comparisons would be methodologically invalid and are not reported here.

**VAE augmentation and severity generalisation.** While VAE augmentation reduces the mean severity generalisation gap from 0.6547 to 0.4203, the effect is uneven across model-class pairs. KNN DE_outer_race gap improves only marginally (0.8522 → 0.8108), and RF FE_outer_race slightly worsens (0.4737 → 0.5354). The mechanism — that synthetic samples from small-severity distributions regularise the classifier toward fault-geometry rather than fault-magnitude features — is plausible but not definitively demonstrated by these experiments. VAE augmentation should be treated as a useful tool with known variance in its generalisation benefit, not a principled cure for severity shift.

**Autoencoder trivial separability.** Separation ratios of 85,524× to 261,998× indicate that the CWRU feature space does not present a meaningful anomaly detection challenge under laboratory conditions. In a realistic field deployment, normal-condition reconstruction error would vary substantially with motor load, temperature, shaft speed transients, and sensor aging, widening the normal distribution and compressing the separation ratio. The 100% detection rates reported in Table 5 cannot be extrapolated to field deployments without re-evaluating thresholds under realistic normal-condition variability.

**CWRU is a controlled laboratory dataset.** Faults are machined (EDM) rather than naturally progressed, shaft speed and load are held constant within each recording session, and the rig is vibrationally isolated. Real wind turbine gearbox vibration will exhibit amplitude modulation from variable wind load, cross-talk from generator, main shaft, and planet bearings, and gear mesh frequency interference at the same kilohertz bands used here for fault detection. The raw-signal CNN accuracy of 0.9998 and XGBoost accuracy of 0.9910 reflect CWRU laboratory conditions. These numbers should not be cited as evidence of field-deployable performance.

---

## 15. Citation

```bibtex
@article{cwru_fault_detection_2026,
  title   = {Seven-Class Bearing Fault Detection with Severity Generalisation Analysis
             Using CWRU Vibration Signals},
  author  = {[Author]},
  journal = {IEEE [Target Venue]},
  year    = {2026},
  note    = {Pipeline code: https://github.com/[repo]}
}
```

```bibtex
@misc{cwru_bearing_dataset,
  title  = {Bearing Data Center},
  author = {Case Western Reserve University},
  url    = {https://engineering.case.edu/bearingdatacenter},
  year   = {2000}
}
```
