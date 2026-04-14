"""
app.py — Streamlit research narrative
Wind Turbine Gearbox Fault Detection
Single-flow story: Problem → Data → Augmentation → Models → Explainability → Conclusion
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bearing Fault Detection — Research Paper",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tighten vertical spacing */
  .block-container { padding-top: 2rem; padding-bottom: 3rem; }
  /* Section dividers */
  .section-label {
      font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
      text-transform: uppercase; color: #888; margin-bottom: 0.2rem;
  }
  .section-title {
      font-size: 1.55rem; font-weight: 700; margin-bottom: 0.5rem;
      line-height: 1.25;
  }
  .section-body {
      font-size: 0.97rem; color: #444; line-height: 1.65;
      max-width: 820px;
  }
  /* Step badge */
  .step-badge {
      display: inline-block;
      background: #1f77b4; color: white;
      font-size: 0.72rem; font-weight: 700;
      padding: 2px 10px; border-radius: 99px;
      margin-bottom: 6px;
  }
  /* Finding callout */
  .finding-box {
      border-left: 4px solid #1f77b4;
      background: #f0f6ff;
      padding: 0.9rem 1.1rem;
      border-radius: 0 6px 6px 0;
      margin-bottom: 0.8rem;
  }
  .finding-title { font-weight: 700; margin-bottom: 0.25rem; }
  hr.thin { border: none; border-top: 1px solid #e0e0e0; margin: 2.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA     = Path("data/processed")
PFIGS    = Path("paper/figures")
FIGS     = Path("figures")

# ── Helpers ────────────────────────────────────────────────────────────────────
def hex_rgba(h: str, a: float = 0.45) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def load_img(p: Path):
    return Image.open(p) if p.exists() else None

PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_family="Arial, sans-serif",
    margin=dict(t=30, b=20, l=10, r=10),
)

MODEL_COLORS = {
    "Random Forest": "#2166ac",
    "XGBoost":       "#d6604d",
    "KNN":           "#1a7837",
    "SVM":           "#762a83",
    "1D-CNN":        "#8c510a",
}
MODEL_MARKERS = {
    "Random Forest": "circle",
    "XGBoost":       "square",
    "KNN":           "triangle-up",
    "SVM":           "diamond",
    "1D-CNN":        "cross",
}
CLASS_ORDER = ["DE_ball","DE_inner_race","DE_outer_race",
               "FE_ball","FE_inner_race","FE_outer_race","normal"]

# ── Load CSVs (all cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_features_exp():
    return pd.read_csv(DATA / "features_expanded.csv")

@st.cache_data
def load_features_aug():
    return pd.read_csv(DATA / "features_augmented.csv")

@st.cache_data
def load_model_results():
    return pd.read_csv(DATA / "model_results_expanded.csv")

@st.cache_data
def load_noise():
    return pd.read_csv(DATA / "noise_robustness_expanded.csv")

@st.cache_data
def load_dl():
    return pd.read_csv(DATA / "dl_results.csv")

@st.cache_data
def load_ae():
    return pd.read_csv(DATA / "autoencoder_results.csv")

@st.cache_data
def load_shap():
    return pd.read_csv(DATA / "shap_consensus.csv")

@st.cache_data
def load_table2():
    return pd.read_csv(Path("paper/tables/table2_classical_ml_performance.csv"))

@st.cache_data
def load_table3():
    return pd.read_csv(Path("paper/tables/table3_ml_vs_dl_comparison.csv"))

feat_exp = load_features_exp()
feat_aug = load_features_aug()
model_res = load_model_results()
noise_df  = load_noise()
dl_df     = load_dl()
ae_df     = load_ae()
shap_df   = load_shap()
t2        = load_table2()
t3        = load_table3()

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 2.2rem 0 1.2rem 0;">
  <div style="font-size:0.8rem; font-weight:700; letter-spacing:0.15em;
              text-transform:uppercase; color:#888; margin-bottom:0.4rem;">
    IEEE Research Paper · CWRU Bearing Fault Dataset
  </div>
  <div style="font-size:2.1rem; font-weight:800; line-height:1.2; margin-bottom:0.7rem;">
    Wind Turbine Gearbox Fault Detection<br>
    <span style="color:#1f77b4;">Using Vibration Signals, VAE Augmentation & SHAP</span>
  </div>
  <div style="font-size:1.0rem; color:#555; max-width:760px; line-height:1.6;">
    This page walks through the full research paper — from the problem with how fault
    detection benchmarks are commonly reported, to the expanded 7-class pipeline,
    deep learning comparison, and multi-model SHAP explainability.
  </div>
</div>
""", unsafe_allow_html=True)

# Key headline numbers
cnn_mean = dl_df[dl_df["fold"].astype(str).str.isnumeric()]["test_accuracy"].astype(float).mean()
best_classical = model_res["cv_accuracy"].max()
ae_fault_dr = ae_df[ae_df["class"] != "normal"]["detection_rate"].mean()
top_feature = shap_df.sort_values("mean_rank").iloc[0]["feature"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best Classical ML", f"{best_classical:.2%}", help="XGBoost 5-fold CV")
k2.metric("1D-CNN Accuracy",   f"{cnn_mean:.2%}",        help="5-fold CV mean")
k3.metric("Autoencoder DR",    f"{ae_fault_dr:.0%}",     help="Mean fault detection rate")
k4.metric("Top SHAP Feature",  top_feature,               help="Consensus across RF/XGB/SVM/CNN")

st.markdown('<hr class="thin">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — THE PROBLEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">Section 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">The Problem with CWRU Benchmarks</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-body">
Most published results on the CWRU bearing fault dataset report 5-fold cross-validation
accuracy above 99%. That number is real — but it is <strong>misleading</strong>.<br><br>
Standard CV mixes all fault severities (0.007", 0.014", 0.021", 0.028") across training
and test folds. The model always sees every severity during training. It is never asked
to diagnose a severity it hasn't encountered before.<br><br>
Real-world deployment doesn't work that way. A bearing fault starts small and grows.
The model you deploy must generalise to severities it was never trained on.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_p1, col_p2 = st.columns([1, 1], gap="large")

with col_p1:
    st.markdown("""
    <div class="finding-box">
      <div class="finding-title">🔴 What the literature reports</div>
      5-fold CV on the CWRU dataset routinely exceeds 99% accuracy for RF, XGBoost, and SVM.
      These results are cited as evidence that the problem is "solved."
    </div>
    <div class="finding-box" style="border-color:#e05c2a; background:#fff5f0;">
      <div class="finding-title">⚠️ What this actually tests</div>
      A model that has memorised all four fault diameters. When evaluated on the
      same severity distribution it trained on, of course it performs well.
    </div>
    <div class="finding-box" style="border-color:#2ca02c; background:#f0fff0;">
      <div class="finding-title">✅ What this paper tests instead</div>
      A <strong>severity generalisation test</strong>: train on 0.007" + 0.014",
      test on 0.021". The model must generalise to a fault diameter it has never seen.
      Gaps of up to <strong>19 percentage points</strong> emerge.
    </div>
    """, unsafe_allow_html=True)

with col_p2:
    # CV gap chart (original 3-model comparison from model_results.csv)
    try:
        orig = pd.read_csv(DATA / "model_results.csv")
        fig_gap = go.Figure()
        for _, row in orig.iterrows():
            c = {"Random Forest": "#2166ac", "XGBoost": "#d6604d", "SVM": "#762a83"}.get(row["model"], "#999")
            fig_gap.add_trace(go.Bar(
                x=["5-Fold CV", "Severity\nGeneralisation"],
                y=[row["cv_accuracy"], row["gen_accuracy"]],
                name=row["model"],
                marker_color=[c, hex_rgba(c, 0.5)],
                text=[f"{row['cv_accuracy']:.1%}", f"{row['gen_accuracy']:.1%}"],
                textposition="outside",
            ))
        fig_gap.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.12], tickformat=".0%", title="Accuracy"),
            height=320, **PLOTLY_LAYOUT,
            legend=dict(orientation="h", y=1.1),
            title_text="The CV–Generalisation Gap",
        )
        st.plotly_chart(fig_gap, use_container_width=True)
        st.caption("Train on 0.007\" + 0.014\", test on 0.021\". Gaps range from −11% to −19%.")
    except FileNotFoundError:
        st.info("model_results.csv not found — run 02_train_models.py.")

st.markdown('<hr class="thin">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET & FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">Section 2</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dataset & Feature Extraction</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-body">
The CWRU bearing dataset contains raw vibration signals from drive-end and fan-end
accelerometers at 12 kHz. This project expands the standard 3-class setup to
<strong>7 fault classes</strong> — separating Drive-End (DE) and Fan-End (FE) faults,
which prior work collapses into a single label and thereby hides meaningful differences
in how faults at different locations present in the signal.<br><br>
Each raw signal is segmented into non-overlapping 1,024-sample windows.
From each window, 15 features are extracted across three domains.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_d1, col_d2 = st.columns([1.1, 0.9], gap="large")

with col_d1:
    # Feature domain table
    feat_table = pd.DataFrame({
        "Domain": ["Time domain", "Time domain", "Time domain", "Time domain", "Time domain",
                   "Time domain", "Time domain", "Time domain", "Time domain",
                   "Frequency", "Frequency", "Frequency", "Frequency", "Frequency",
                   "Operating"],
        "Feature": ["Mean", "Std Dev", "RMS", "Peak", "Peak-to-Peak",
                    "Crest Factor", "Kurtosis", "Skewness", "Shape Factor",
                    "Spectral Centroid", "Dominant Frequency",
                    "Low Band Energy", "Mid Band Energy", "High Band Energy",
                    "Motor Load (HP)"],
    })
    st.markdown("**15 extracted features:**")
    st.dataframe(feat_table, hide_index=True, use_container_width=True, height=260)

with col_d2:
    # Raw class counts from features_expanded
    counts = feat_exp.groupby("label").size().reindex(CLASS_ORDER).reset_index()
    counts.columns = ["Class", "Windows"]
    counts["Location"] = counts["Class"].apply(
        lambda c: "Drive-End" if c.startswith("DE") else ("Fan-End" if c.startswith("FE") else "Normal"))
    color_map = {"Drive-End": "#2166ac", "Fan-End": "#d6604d", "Normal": "#888888"}
    fig_cnt = px.bar(counts, y="Class", x="Windows", color="Location",
                     color_discrete_map=color_map, orientation="h",
                     text="Windows", title="Real windows per class (before augmentation)")
    fig_cnt.update_traces(textposition="outside")
    fig_cnt.update_layout(height=320, showlegend=True, **PLOTLY_LAYOUT,
                          yaxis=dict(categoryorder="array", categoryarray=CLASS_ORDER[::-1]))
    st.plotly_chart(fig_cnt, use_container_width=True)

# Interactive feature explorer
st.markdown("**Explore feature distributions interactively:**")
FCOLS = ["mean","std","rms","peak","peak2peak","crest_factor","kurtosis",
         "skewness","shape_factor","spectral_centroid","dominant_frequency",
         "low_band_energy","mid_band_energy","high_band_energy"]
sel_feat = st.selectbox("Select a feature", FCOLS,
                         index=FCOLS.index("kurtosis"), key="feat_select")
loc_colors = {c: ("#2166ac" if c.startswith("DE") else
                  ("#d6604d" if c.startswith("FE") else "#888888"))
              for c in CLASS_ORDER}
sample = feat_exp.sample(min(3000, len(feat_exp)), random_state=42)
fig_box = px.box(sample, x="label", y=sel_feat, color="label",
                 color_discrete_map=loc_colors,
                 category_orders={"label": CLASS_ORDER},
                 labels={"label": "Fault Class", sel_feat: sel_feat.replace("_"," ").title()},
                 title=f"{sel_feat.replace('_',' ').title()} by fault class")
fig_box.update_layout(showlegend=False, height=360, **PLOTLY_LAYOUT)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown('<hr class="thin">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CLASS IMBALANCE & VAE AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">Section 3</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Class Imbalance & VAE Augmentation</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-body">
The 7-class expansion reveals a significant imbalance. Fan-End faults and Drive-End
outer-race faults have fewer data recordings — FE classes have only 1,416 real windows
each, while FE_outer_race has 2,477. Training on imbalanced data biases classifiers
toward majority classes, inflating overall accuracy while degrading detection of the
rarer fault types that matter most in deployment.<br><br>
To address this, a <strong>Variational Autoencoder (VAE)</strong> is trained independently
for each under-represented class on its real samples. The VAE learns the latent
distribution of each fault signature and generates synthetic windows to balance all
classes to 2,477 samples — the natural maximum.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_v1, col_v2 = st.columns([1, 1], gap="large")

with col_v1:
    real_counts = feat_exp.groupby("label").size().reindex(CLASS_ORDER)
    aug_counts  = feat_aug.groupby("label").size().reindex(CLASS_ORDER)
    synth       = aug_counts - real_counts

    fig_aug = go.Figure()
    bar_colors = ["#2166ac" if c.startswith("DE") else
                  ("#d6604d" if c.startswith("FE") else "#888888")
                  for c in CLASS_ORDER]
    fig_aug.add_trace(go.Bar(
        name="Real windows", y=CLASS_ORDER, x=real_counts.values,
        orientation="h", marker_color=bar_colors, opacity=0.9,
        text=real_counts.values, textposition="inside",
    ))
    fig_aug.add_trace(go.Bar(
        name="VAE synthetic", y=CLASS_ORDER, x=synth.values,
        orientation="h", marker_color=[hex_rgba(c, 0.55) for c in bar_colors],
        marker_pattern_shape="/",
        text=[f"+{v}" if v > 0 else "" for v in synth.values], textposition="inside",
    ))
    fig_aug.update_layout(
        barmode="stack", height=350, **PLOTLY_LAYOUT,
        xaxis_title="Windows", title="Before and after VAE augmentation",
        legend=dict(orientation="h", y=1.08),
        yaxis=dict(categoryorder="array", categoryarray=CLASS_ORDER[::-1]),
    )
    st.plotly_chart(fig_aug, use_container_width=True)

with col_v2:
    st.markdown("**VAE design:**")
    vae_info = pd.DataFrame({
        "Component": ["Encoder input", "Latent space", "Decoder output",
                      "Training data", "Loss function", "Classes augmented"],
        "Detail": ["14 vibration features", "4-dimensional (z-mean, z-log-var)",
                   "14 reconstructed features", "Real windows per class only",
                   "Reconstruction + KL divergence", "6 of 7 (FE_outer_race at max)"],
    })
    st.dataframe(vae_info, hide_index=True, use_container_width=True)

    st.markdown("**Augmentation summary:**")
    aug_summary = pd.DataFrame({
        "Class":           CLASS_ORDER,
        "Real":            real_counts.values,
        "Synthetic Added": synth.values,
        "Final":           aug_counts.values,
    })
    st.dataframe(aug_summary, hide_index=True, use_container_width=True, height=230)

vae_img = load_img(FIGS / "vae_tsne.png")
if vae_img:
    st.markdown("**t-SNE of the augmented feature space** — synthetic samples (lighter) "
                "fill gaps in the real data manifold without collapsing class boundaries:")
    st.image(vae_img, use_container_width=True)

st.markdown('<hr class="thin">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">Section 4</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Comparison: Classical ML vs Deep Learning</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-body">
Five models are evaluated on the augmented 7-class dataset using 5-fold stratified
cross-validation. The result that challenges prevailing assumptions:
<strong>XGBoost outperforms the 1D-CNN</strong> on engineered vibration features.
This is not a CNN failure — the CNN achieves 97.95% — but it demonstrates that
hand-crafted frequency and time-domain features carry enough discriminative information
that a well-tuned gradient-boosted tree captures it more efficiently than a deep network.<br><br>
An <strong>unsupervised autoencoder</strong> trained only on normal-class samples achieves
100% fault detection across all six fault types, with only a 1.33% false positive rate —
making it a strong candidate for deployment scenarios where labelled fault data is scarce.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Top-level metrics row
all_models = ["Random Forest", "XGBoost", "KNN", "SVM", "1D-CNN"]
cv_accs = {row["model"]: row["cv_accuracy"] for _, row in model_res.iterrows()}
cv_accs["1D-CNN"] = cnn_mean
m1, m2, m3, m4, m5 = st.columns(5)
for col, mname in zip([m1,m2,m3,m4,m5], all_models):
    col.metric(mname, f"{cv_accs[mname]:.2%}")

st.markdown("<br>", unsafe_allow_html=True)

tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs([
    "📊 Accuracy & F1", "🟦 Per-Class F1 Heatmap", "🔊 Noise Robustness", "🤖 Autoencoder"
])

with tab_m1:
    col_m1, col_m2 = st.columns([1.1, 0.9], gap="large")
    with col_m1:
        accs = [cv_accs[m] for m in all_models]
        mf1s = t2.set_index("Model")["Macro F1"].reindex(["Random Forest","XGBoost","KNN","SVM"]).tolist()
        # CNN macro F1 from table3
        cnn_f1_row = t3[t3["Model"] == "1D-CNN"]["Macro F1"].values
        mf1s.append(float(cnn_f1_row[0]) if len(cnn_f1_row) and cnn_f1_row[0] != "N/A" else np.nan)

        fig_bar = go.Figure()
        x = list(range(len(all_models)))
        fig_bar.add_trace(go.Bar(
            x=all_models, y=accs, name="5-fold CV Accuracy",
            marker_color=[MODEL_COLORS[m] for m in all_models],
            text=[f"{v:.3f}" for v in accs], textposition="outside",
            width=0.35, offset=-0.18,
        ))
        fig_bar.add_trace(go.Bar(
            x=all_models, y=[v if not np.isnan(v) else 0 for v in mf1s],
            name="Macro F1",
            marker_color=[hex_rgba(MODEL_COLORS[m], 0.65) for m in all_models],
            text=[f"{v:.3f}" if not np.isnan(v) else "—" for v in mf1s],
            textposition="outside",
            width=0.35, offset=0.18,
        ))
        fig_bar.update_layout(
            barmode="overlay",
            yaxis=dict(range=[0.88, 1.03], title="Score", tickformat=".2f"),
            height=380, **PLOTLY_LAYOUT,
            legend=dict(orientation="h", y=1.1),
            title_text="All models — Accuracy vs Macro F1 (5-fold CV)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_m2:
        st.markdown("**Full comparison table:**")
        t3_display = t3.copy()
        st.dataframe(t3_display, hide_index=True, use_container_width=True, height=260)
        st.markdown("""
        <div style="font-size:0.85rem; color:#666; margin-top:0.5rem;">
        † Autoencoder accuracy = mean fault detection rate.<br>
        Classical ML uses 5-fold CV; CNN uses 5-fold CV on augmented data.
        </div>
        """, unsafe_allow_html=True)

with tab_m2:
    st.markdown("""
    The heatmap below shows per-class F1 score for each of the five models.
    Rows are the 7 fault classes; columns are models. Darker blue = higher F1.
    FE_ball is consistently the hardest class across all models — Fan-End ball
    faults produce a subtler vibration signature than Drive-End equivalents.
    """)
    fig_heat = load_img(PFIGS / "fig3_f1_heatmap.png")
    if fig_heat:
        st.image(fig_heat, use_container_width=True)
    else:
        # Build interactively from table2
        cl_models = ["Random Forest","XGBoost","KNN","SVM"]
        hm = t2.set_index("Model")[CLASS_ORDER].reindex(cl_models).values.astype(float)
        fig_hm = px.imshow(hm, x=CLASS_ORDER, y=cl_models,
                           color_continuous_scale="Blues", zmin=0.85, zmax=1.0,
                           text_auto=".3f", aspect="auto",
                           title="Per-Class F1 (Classical ML)")
        fig_hm.update_layout(height=280, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_hm, use_container_width=True)

with tab_m3:
    st.markdown("""
    Models are evaluated on the test split with increasing levels of Gaussian noise
    added to every feature (scaled as % of each feature's standard deviation).
    This tests whether learned representations are fragile or truly signal-grounded.
    """)

    noise_models = noise_df["model"].unique().tolist()
    # CNN noise data (from pipeline obs)
    CNN_NOISE_PTS = {0.0:0.9812, 0.05:0.9535, 0.10:0.8794, 0.20:0.7560, 0.30:0.6434}

    fig_noise = go.Figure()
    for m in noise_models:
        sub = noise_df[noise_df["model"]==m].sort_values("noise_pct")
        fig_noise.add_trace(go.Scatter(
            x=(sub["noise_pct"]*100).tolist(),
            y=sub["accuracy"].tolist(),
            mode="lines+markers",
            name=m,
            line=dict(color=MODEL_COLORS.get(m,"#999"), width=2.2),
            marker=dict(size=7, symbol=MODEL_MARKERS.get(m,"circle")),
        ))
    # CNN line
    cx = [k*100 for k in sorted(CNN_NOISE_PTS)]
    cy = [CNN_NOISE_PTS[k] for k in sorted(CNN_NOISE_PTS)]
    fig_noise.add_trace(go.Scatter(
        x=cx, y=cy, mode="lines+markers", name="1D-CNN",
        line=dict(color=MODEL_COLORS["1D-CNN"], width=2.2, dash="dot"),
        marker=dict(size=7, symbol="cross"),
    ))
    fig_noise.update_layout(
        xaxis=dict(title="Gaussian Noise Level (%)", ticksuffix="%", tickvals=[0,5,10,20,30]),
        yaxis=dict(title="Accuracy", tickformat=".0%", range=[0.5, 1.02]),
        height=420, **PLOTLY_LAYOUT,
        legend=dict(orientation="h", y=1.08),
        title_text="Noise Robustness — All 5 Models",
    )
    noise_sel = st.slider("Highlight noise level (%)", 0, 30, 10, 5, key="noise_slider")
    fig_noise.add_vline(x=noise_sel, line_dash="dash", line_color="#aaa",
                        annotation_text=f"{noise_sel}%", annotation_position="top left")
    st.plotly_chart(fig_noise, use_container_width=True)

    st.markdown("""
    **Key takeaway:** SVM degrades most gracefully (−15 pp at 30% noise) because its
    max-margin boundary is more noise-tolerant. XGBoost has the steepest classical ML
    drop (−41 pp). The 1D-CNN falls from 98.1% to 64.3% at 30% noise — competitive
    with RF but not robust enough for high-noise industrial environments.
    """)

with tab_m4:
    col_ae1, col_ae2 = st.columns([1, 1], gap="large")
    with col_ae1:
        st.markdown("""
        The autoencoder is trained **only on normal-class samples** using a
        reconstruction objective. At inference, it measures how well it can
        reconstruct a new window. Fault samples are structurally different from
        the training distribution and produce high reconstruction error —
        they are flagged as anomalies.

        This requires **zero fault labels during training**, making it deployable
        on machinery where fault data hasn't been collected yet.
        """)
        ae_display = ae_df.copy()
        ae_display["detection_rate"] = ae_display["detection_rate"].map("{:.1%}".format)
        ae_display.loc[ae_display["false_positive_rate"].notna(), "false_positive_rate"] = \
            ae_df.loc[ae_df["false_positive_rate"].notna(), "false_positive_rate"].map("{:.2%}".format)
        ae_display.columns = ["Class", "Total", "Detected", "Detection Rate", "FPR"]
        st.dataframe(ae_display, hide_index=True, use_container_width=True)

    with col_ae2:
        ae_violin = load_img(PFIGS / "fig7_autoencoder_violin.png")
        if ae_violin:
            st.image(ae_violin, use_container_width=True,
                     caption="Reconstruction error by class. Red dashed line = anomaly threshold. "
                             "All fault classes sit above it; normal class mostly below.")

st.markdown('<hr class="thin">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FAULT SIZE DEGRADATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">Section 5</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Does Performance Degrade with Fault Severity?</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-body">
Using a hold-one-size-out evaluation — train on all fault sizes except one, test on
the held-out size — we measure whether models struggle with any particular fault
diameter. The critical boundary is between 0.021" and 0.028":
at 0.028" fault diameter, the vibration signature changes character as the damage
becomes macroscopic, and all models show measurable F1 degradation.
</div>
""", unsafe_allow_html=True)

fs_img = load_img(PFIGS / "fig6_fault_size_degradation.png")
if fs_img:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(fs_img, use_container_width=True,
             caption="Hold-one-size-out macro F1 per model. "
                     "Dashed line marks the severity threshold between 0.021\" and 0.028\".")
else:
    st.info("Run paper/generate_paper_assets.py to generate fig6_fault_size_degradation.png")

st.markdown('<hr class="thin">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">Section 6</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">SHAP Explainability — What Do the Models Actually Learn?</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-body">
High accuracy answers <em>whether</em> a fault is present. It doesn't answer
<em>why</em> the model made that decision, which features are actually diagnostic,
or whether different models are learning the same signal patterns or exploiting
different artifacts.<br><br>
SHAP (SHapley Additive exPlanations) is applied to all four models —
Random Forest, XGBoost, SVM, and CNN — and the results are compared in a
<strong>consensus ranking</strong>. Features that are important across all four
models are genuinely signal-grounded, not model-specific artifacts.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_s1, col_s2 = st.columns([1, 1], gap="large")

with col_s1:
    # Interactive SHAP consensus table
    top_n = st.slider("Show top N features", 5, 10, 10, key="shap_top")
    shap_top = shap_df.sort_values("mean_rank").head(top_n).copy()
    shap_top["All-model top-5?"] = shap_top.apply(
        lambda r: "★" if all(r[c] <= 5 for c in ["RF_rank","XGBoost_rank","SVM_rank","CNN_rank"])
                  else "", axis=1)
    shap_display = shap_top[["feature","RF_rank","XGBoost_rank","SVM_rank",
                              "CNN_rank","mean_rank","All-model top-5?"]].copy()
    shap_display.columns = ["Feature","RF","XGB","SVM","CNN","Mean Rank","Consensus"]
    shap_display["Mean Rank"] = shap_display["Mean Rank"].round(2)
    st.markdown(f"**Top {top_n} features by consensus rank:**")
    st.dataframe(shap_display, hide_index=True, use_container_width=True)
    st.markdown("""
    <div style="font-size:0.85rem; color:#666;">
    ★ = ranked in top 5 by <em>all four</em> models simultaneously.<br>
    Only High Band Energy earns this distinction.
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    shap_img = load_img(PFIGS / "fig5_shap_consensus.png")
    if shap_img:
        st.image(shap_img, use_container_width=True,
                 caption="Rank heatmap — dark green = rank 1 (most important), "
                         "dark red = rank 15 (least important). "
                         "Columns are the four models; rows are the top 10 features.")

st.markdown("<br>", unsafe_allow_html=True)

# Individual model SHAP plots in an expander
with st.expander("View individual model SHAP bar charts"):
    sc1, sc2 = st.columns(2)
    with sc1:
        img_rf = load_img(FIGS / "shap_rf_bar.png")
        if img_rf: st.image(img_rf, caption="Random Forest — top feature importances", use_container_width=True)
        img_svm = load_img(FIGS / "shap_svm_bar.png")
        if img_svm: st.image(img_svm, caption="SVM — top feature importances", use_container_width=True)
    with sc2:
        img_xgb = load_img(FIGS / "shap_xgb_bar.png")
        if img_xgb: st.image(img_xgb, caption="XGBoost — top feature importances", use_container_width=True)
        img_cnn = load_img(FIGS / "shap_cnn_bar.png")
        if img_cnn: st.image(img_cnn, caption="1D-CNN — top feature importances", use_container_width=True)

st.markdown('<hr class="thin">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CONCLUSIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">Section 7</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Conclusions & Key Contributions</div>', unsafe_allow_html=True)

cc1, cc2 = st.columns(2, gap="large")

with cc1:
    st.markdown("""
    <div class="finding-box">
      <div class="finding-title">📌 C1 — Benchmark Methodology</div>
      Standard 5-fold CV on CWRU overstates real-world performance by 11–19 percentage
      points. Severity generalisation evaluation is the appropriate test for deployment readiness.
    </div>
    <div class="finding-box" style="border-color:#d6604d; background:#fff5f0;">
      <div class="finding-title">📌 C2 — XGBoost Beats CNN on Engineered Features</div>
      On 15 hand-crafted vibration features, XGBoost (99.10%) outperforms a 1D-CNN (97.95%).
      Deep learning does not automatically win when the feature engineering is sound.
    </div>
    <div class="finding-box" style="border-color:#1a7837; background:#f0fff4;">
      <div class="finding-title">📌 C3 — Unsupervised Anomaly Detection Works</div>
      An autoencoder trained only on normal data achieves 100% fault detection with
      1.33% false positives — viable for cold-start deployment with no fault labels.
    </div>
    """, unsafe_allow_html=True)

with cc2:
    st.markdown("""
    <div class="finding-box" style="border-color:#762a83; background:#faf0ff;">
      <div class="finding-title">📌 C4 — VAE Augmentation Improves Minority Classes</div>
      Class-conditional VAE augmentation balances the 7-class dataset from
      1,416–2,477 real samples to 2,477 per class, improving F1 on FE faults
      by an average of 4–6 percentage points.
    </div>
    <div class="finding-box" style="border-color:#8c510a; background:#fdf6e3;">
      <div class="finding-title">📌 C5 — High Band Energy is the Universal Discriminator</div>
      Across RF, XGBoost, SVM, and CNN, High Band Energy is the only feature
      ranked in the top 5 by all four models. It is the most signal-grounded,
      model-agnostic diagnostic feature in this dataset.
    </div>
    <div class="finding-box" style="border-color:#888; background:#f8f8f8;">
      <div class="finding-title">📌 C6 — Severity Threshold at 0.028"</div>
      F1 degrades measurably at 0.028" fault diameter across all models — a
      natural "severity threshold" with implications for maintenance scheduling.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Final summary numbers
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Real windows",    "11,945")
s2.metric("After VAE aug.",  "17,339")
s3.metric("Fault classes",   "7")
s4.metric("Best accuracy",   f"{best_classical:.2%}", help="XGBoost 5-fold CV")
s5.metric("SHAP models",     "4", help="RF · XGBoost · SVM · CNN")

st.markdown('<hr class="thin">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:0.82rem; color:#aaa; padding:0.5rem 0 1rem 0;">
  CWRU Bearing Fault Dataset · 7-class · VAE Augmentation · 5-fold CV ·
  IEEE paper pipeline &nbsp;|&nbsp; Vybh/windproject
</div>
""", unsafe_allow_html=True)
