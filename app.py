"""
app.py — Streamlit interactive dashboard
Wind Turbine Gearbox Fault Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Turbine Fault Detection",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/processed")
FIGURES_DIR = Path("figures")

# ── Colour helpers ────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.53) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,alpha)' for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Colour palette ─────────────────────────────────────────────────────────────
FAULT_COLORS = {
    "normal":     "#4CAF50",
    "inner_race": "#2196F3",
    "ball":       "#FF9800",
    "outer_race": "#E91E63",
}
MODEL_COLORS = {
    "Random Forest": "#2196F3",
    "XGBoost":       "#FF7043",
    "SVM":           "#7E57C2",
}

# ── Load data (cached) ─────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    return pd.read_csv(DATA_DIR / "features.csv")

@st.cache_data
def load_model_results():
    return pd.read_csv(DATA_DIR / "model_results.csv")

@st.cache_data
def load_noise():
    return pd.read_csv(DATA_DIR / "noise_robustness.csv")

@st.cache_data
def load_gen():
    return pd.read_csv(DATA_DIR / "generalization_results.csv")

def load_fig(name):
    p = FIGURES_DIR / name
    return Image.open(p) if p.exists() else None

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌬️ Wind Fault Detection")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "📊  Data Explorer",
            "🤖  Model Results",
            "🔊  Noise Robustness",
            "🔍  SHAP Explainability",
        ],
    )
    st.markdown("---")
    st.markdown(
        """
        **Dataset:** CWRU Bearing
        **Windows:** 5,926 × 1,024 samples
        **Features:** 15 (time + freq)
        **Models:** RF · XGBoost · SVM
        **RANDOM_STATE:** 42
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.title("🌬️ Wind Turbine Gearbox Fault Detection")
    st.markdown(
        "##### Vibration Signal Analysis · Machine Learning · SHAP Explainability"
    )
    st.markdown("---")

    # ── Hero metric cards ──────────────────────────────────────────────────────
    try:
        results = load_model_results()
        best_cv  = results["cv_accuracy"].max()
        best_gen = results["gen_accuracy"].max()
        max_gap  = results["gap"].max()
        df_feat  = load_features()
        n_windows = len(df_feat)
    except FileNotFoundError:
        best_cv = best_gen = max_gap = None
        n_windows = 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Windows", f"{n_windows:,}", help="1,024-sample non-overlapping segments")
    if best_cv:
        c2.metric("Best CV Accuracy",  f"{best_cv:.2%}",  help="5-fold stratified cross-validation")
        c3.metric("Best Gen Accuracy", f"{best_gen:.2%}", help="Trained on 0.007\"+0.014\", tested on 0.021\"")
        c4.metric("Max CV–Gen Gap",    f"{max_gap:.2%}",  delta=f"-{max_gap:.2%}", delta_color="inverse",
                  help="The central research finding")

    st.markdown("---")

    # ── Two-column layout: findings + pipeline ─────────────────────────────────
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("🎯 Central Research Contributions")

        st.error(
            "**Finding 1 — The CV / Generalisation Gap**\n\n"
            "Standard cross-validation on CWRU achieves ~99.6–99.9% accuracy. "
            "But when models are tested on a *fault severity they have never seen* (0.021\"), "
            "accuracy drops to **80–89%** — a gap of up to **19 percentage points**.\n\n"
            "This exposes a systematic flaw in how fault detection benchmarks are reported."
        )

        st.success(
            "**Finding 2 — Severity-Dependent Feature Importance (SHAP)**\n\n"
            "As faults worsen (0.007\" → 0.021\"), the most important features *change*:\n"
            "- **Shape Factor** importance ↑ 47%\n"
            "- **High Band Energy** importance ↓ 19%\n\n"
            "Fault progression shifts signal energy from high-frequency spectral content "
            "toward time-domain amplitude statistics."
        )

        st.info(
            "**Conclusion**\n\n"
            "Evaluation methodology matters as much as model choice. "
            "SHAP provides the *why* behind predictions, not just the *what*."
        )

    with right:
        st.subheader("🔄 Pipeline")
        st.markdown(
            """
```
 data/raw/*.mat  (40 CWRU files)
        │
        ▼
 01_load_and_preprocess.py
   └─ 5,926 windows × 15 features
        │
        ▼
 02_train_models.py
   ├─ 5-fold CV  → ~99.6% accuracy
   └─ Severity generalisation → 80–89%
        │
        ▼
 03_shap_analysis.py
   ├─ TreeSHAP on RF + XGBoost
   └─ Per-class & per-severity importance
```
            """
        )
        st.subheader("📁 Fault Classes")
        fault_info = pd.DataFrame({
            "Fault":    ["Normal", "Inner Race", "Ball", "Outer Race"],
            "Severity": ["—", "0.007\" / 0.014\" / 0.021\""] * 2,
            "Windows":  ["1,656", "~1,422", "~1,423", "~1,425"],
        })
        st.dataframe(fault_info, hide_index=True, use_container_width=True)

    # ── Key gap table ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 The Performance Gap — At a Glance")
    if best_cv:
        fig = go.Figure()
        models = results["model"].tolist()
        cv_vals  = results["cv_accuracy"].tolist()
        gen_vals = results["gen_accuracy"].tolist()
        colors = [MODEL_COLORS.get(m, "#999") for m in models]

        fig.add_trace(go.Bar(
            name="5-Fold CV",
            x=models, y=cv_vals,
            marker_color=colors,
            text=[f"{v:.2%}" for v in cv_vals],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Severity Generalisation",
            x=models, y=gen_vals,
            marker_color=[hex_to_rgba(c) for c in colors],
            marker_pattern_shape="/",
            text=[f"{v:.2%}" for v in gen_vals],
            textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.12], tickformat=".0%", title="Accuracy"),
            xaxis_title="Model",
            legend=dict(orientation="h", y=1.12),
            height=400,
            margin=dict(t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("Interactively explore the 5,926 extracted vibration feature windows.")

    try:
        df = load_features()
    except FileNotFoundError:
        st.error("Run `01_load_and_preprocess.py` first to generate features.csv.")
        st.stop()

    st.markdown("---")

    # ── Sidebar filters ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        sel_faults = st.multiselect(
            "Fault classes",
            df["label"].unique().tolist(),
            default=df["label"].unique().tolist(),
        )
        sel_sevs = st.multiselect(
            "Severities",
            sorted(df["severity"].unique().tolist()),
            default=sorted(df["severity"].unique().tolist()),
        )

    mask = df["label"].isin(sel_faults) & df["severity"].isin(sel_sevs)
    dff = df[mask]
    st.caption(f"Showing {len(dff):,} of {len(df):,} windows")

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 Distribution", "📈 Feature Scatter", "🌡️ Correlation", "🗂️ Raw Table"
    ])

    FEATURE_COLS = [
        "mean", "std", "rms", "peak", "peak2peak",
        "crest_factor", "kurtosis", "skewness", "shape_factor",
        "spectral_centroid", "dominant_frequency",
        "low_band_energy", "mid_band_energy", "high_band_energy",
        "motor_load",
    ]

    with tab1:
        feat = st.selectbox("Feature", FEATURE_COLS, index=FEATURE_COLS.index("kurtosis"))
        fig = px.histogram(
            dff, x=feat, color="label",
            color_discrete_map=FAULT_COLORS,
            barmode="overlay", opacity=0.65, nbins=60,
            labels={feat: feat.replace("_", " ").title(), "label": "Fault Class"},
            title=f"Distribution of {feat.replace('_', ' ').title()} by Fault Class",
        )
        fig.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Box plot view**")
        fig2 = px.box(
            dff, x="label", y=feat, color="label",
            color_discrete_map=FAULT_COLORS,
            points=False,
            labels={"label": "Fault Class", feat: feat.replace("_", " ").title()},
        )
        fig2.update_layout(height=380, showlegend=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        x_feat = c1.selectbox("X axis", FEATURE_COLS, index=FEATURE_COLS.index("kurtosis"))
        y_feat = c2.selectbox("Y axis", FEATURE_COLS, index=FEATURE_COLS.index("rms"))
        sample = dff.sample(min(1500, len(dff)), random_state=42)
        fig = px.scatter(
            sample, x=x_feat, y=y_feat,
            color="label", symbol="severity",
            color_discrete_map=FAULT_COLORS,
            opacity=0.65,
            labels={
                x_feat: x_feat.replace("_", " ").title(),
                y_feat: y_feat.replace("_", " ").title(),
                "label": "Fault Class",
            },
            title=f"{x_feat.replace('_',' ').title()} vs {y_feat.replace('_',' ').title()}",
            hover_data=["label", "severity", "motor_load"],
        )
        fig.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr = dff[FEATURE_COLS].corr()
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        fig.update_layout(height=560, margin=dict(t=40))
        fig.update_traces(textfont_size=8)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.dataframe(
            dff.head(500).style.background_gradient(subset=FEATURE_COLS, cmap="Blues"),
            use_container_width=True,
            height=450,
        )
        st.caption("Showing first 500 rows of filtered selection.")

    # ── Window counts ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Window Count by Class × Severity")
    pivot = dff.groupby(["label", "severity"]).size().reset_index(name="windows")
    fig = px.bar(
        pivot, x="severity", y="windows", color="label",
        color_discrete_map=FAULT_COLORS,
        barmode="group",
        text="windows",
        labels={"severity": "Fault Severity", "windows": "Window Count", "label": "Fault Class"},
        title="Extracted Windows per Class and Severity",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Results":
    st.title("🤖 Model Results")
    st.markdown(
        "Three models, two evaluation strategies. The gap between them is the finding."
    )

    try:
        results  = load_model_results()
        gen_df   = load_gen()
    except FileNotFoundError:
        st.error("Run `02_train_models.py` first.")
        st.stop()

    st.markdown("---")

    # ── Summary metric cards ───────────────────────────────────────────────────
    cols = st.columns(3)
    for i, row in results.iterrows():
        with cols[i]:
            st.markdown(f"#### {row['model']}")
            st.metric("CV Accuracy",             f"{row['cv_accuracy']:.2%}")
            st.metric("Generalisation Accuracy", f"{row['gen_accuracy']:.2%}",
                      delta=f"-{row['gap']:.2%}", delta_color="inverse")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Accuracy Gap", "📋 Per-Class F1", "🟦 Confusion Matrices", "📈 Severity Breakdown"
    ])

    with tab1:
        st.subheader("CV vs Severity Generalisation Accuracy")
        fig = go.Figure()
        for _, row in results.iterrows():
            color = MODEL_COLORS.get(row["model"], "#999")
            fig.add_trace(go.Bar(
                x=["5-Fold CV", "Severity Generalisation"],
                y=[row["cv_accuracy"], row["gen_accuracy"]],
                name=row["model"],
                marker_color=[color, hex_to_rgba(color)],
                text=[f"{row['cv_accuracy']:.2%}", f"{row['gen_accuracy']:.2%}"],
                textposition="outside",
            ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.13], tickformat=".0%", title="Accuracy"),
            height=460,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Gap table**")
        styled = results.copy()
        styled["cv_accuracy"]  = styled["cv_accuracy"].map("{:.2%}".format)
        styled["gen_accuracy"] = styled["gen_accuracy"].map("{:.2%}".format)
        styled["gap"]          = styled["gap"].map("{:.2%}".format)
        styled.columns         = ["Model", "CV Accuracy", "Gen Accuracy", "Gap ↓"]
        st.dataframe(styled, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Per-Class F1 Score: CV vs Generalisation")
        classes = gen_df["class"].unique().tolist()
        sel_model = st.selectbox("Select model", results["model"].tolist())
        sub = gen_df[gen_df["model"] == sel_model]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sub["class"], y=sub["cv_f1"],
            name="5-Fold CV", marker_color=MODEL_COLORS.get(sel_model, "#999"),
            text=[f"{v:.3f}" for v in sub["cv_f1"]], textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=sub["class"], y=sub["gen_f1"],
            name="Severity Gen", marker_color=hex_to_rgba(MODEL_COLORS.get(sel_model, "#4CAF50")),
            marker_pattern_shape="/",
            text=[f"{v:.3f}" for v in sub["gen_f1"]], textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.12], title="F1 Score"),
            xaxis_title="Fault Class",
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**All models side-by-side**")
        img = load_fig("per_class_f1_chart.png")
        if img:
            st.image(img, use_container_width=True)

    with tab3:
        st.subheader("Confusion Matrices (3 × 2 Grid)")
        img = load_fig("confusion_matrix_grid.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.warning("confusion_matrix_grid.png not found.")

    with tab4:
        st.subheader("Per-Severity Accuracy Breakdown")
        img = load_fig("severity_performance.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.warning("severity_performance.png not found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — NOISE ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔊  Noise Robustness":
    st.title("🔊 Noise Robustness")
    st.markdown(
        "How well do models hold up when Gaussian noise is injected into the feature vector? "
        "Noise is scaled as a **percentage of each feature's standard deviation** on the "
        "severity generalisation test set."
    )

    try:
        noise_df = load_noise()
    except FileNotFoundError:
        st.error("Run `02_train_models.py` first.")
        st.stop()

    st.markdown("---")

    # ── Interactive noise slider ───────────────────────────────────────────────
    noise_pct_sel = st.slider(
        "Highlight noise level (%)",
        min_value=0, max_value=30, value=10, step=5,
        help="Vertical marker will be drawn at this noise level",
    )

    fig = go.Figure()
    for model, color in MODEL_COLORS.items():
        sub = noise_df[noise_df["model"] == model].sort_values("noise_pct")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["noise_pct"] * 100,
            y=sub["accuracy"],
            mode="lines+markers",
            name=model,
            line=dict(color=color, width=2.5),
            marker=dict(size=8),
            hovertemplate=f"<b>{model}</b><br>Noise: %{{x:.0f}}%<br>Accuracy: %{{y:.2%}}<extra></extra>",
        ))

    fig.add_vline(
        x=noise_pct_sel,
        line_dash="dash", line_color="gray",
        annotation_text=f"{noise_pct_sel}% noise",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(title="Gaussian Noise Level (% of feature std)", ticksuffix="%"),
        yaxis=dict(title="Generalisation Accuracy", tickformat=".0%", range=[0, 1.05]),
        height=480,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Table at selected noise level ──────────────────────────────────────────
    noise_target = noise_pct_sel / 100
    closest = noise_df.groupby("model").apply(
        lambda g: g.iloc[(g["noise_pct"] - noise_target).abs().argsort()[:1]]
    ).reset_index(drop=True)
    closest["accuracy"] = closest["accuracy"].map("{:.2%}".format)
    closest["noise_pct"] = closest["noise_pct"].map("{:.0%}".format)
    closest.columns = ["Model", "Noise Level", "Accuracy"]
    st.markdown(f"**Accuracy at ~{noise_pct_sel}% noise:**")
    st.dataframe(closest, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Static Figure")
    img = load_fig("noise_robustness.png")
    if img:
        st.image(img, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  SHAP Explainability":
    st.title("🔍 SHAP Explainability")
    st.markdown(
        "TreeSHAP applied to Random Forest and XGBoost trained on the full dataset "
        "(2,000-sample representative subset). These plots answer *why* the models predict what they do."
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Global Importance", "⚖️ RF vs XGBoost", "🟥 Class Heatmap", "📉 Severity Shift"
    ])

    with tab1:
        st.subheader("Global Feature Importance (Mean |SHAP|)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Random Forest**")
            img = load_fig("shap_rf_importance.png")
            if img:
                st.image(img, use_container_width=True)
        with c2:
            st.markdown("**XGBoost**")
            img = load_fig("shap_xgb_importance.png")
            if img:
                st.image(img, use_container_width=True)

        st.info(
            "Features are ranked by their mean absolute SHAP value across all samples and classes. "
            "Higher = more influential in model decisions globally."
        )

    with tab2:
        st.subheader("RF vs XGBoost — Side-by-Side Comparison")
        img = load_fig("shap_importance_comparison.png")
        if img:
            st.image(img, use_container_width=True)

        st.markdown("""
**Key observations:**
- Both models broadly agree on top features, validating the signal pipeline.
- XGBoost places heavier weight on frequency-domain features (mid/high band energy).
- RF distributes importance more evenly — characteristic of ensemble averaging.
        """)

    with tab3:
        st.subheader("Per-Class SHAP Heatmap (XGBoost)")
        st.markdown(
            "Each cell shows mean |SHAP| for a feature–class pair. "
            "Darker = more important for distinguishing that fault class."
        )
        img = load_fig("shap_xgb_class_heatmap.png")
        if img:
            st.image(img, use_container_width=True)

        st.markdown("**Top discriminating features per fault class:**")
        top_features = pd.DataFrame({
            "Fault Class":  ["Ball", "Inner Race", "Outer Race", "Normal"],
            "Rank 1":       ["Mid Band Energy", "Peak-to-Peak", "Peak-to-Peak", "High Band Energy"],
            "Rank 2":       ["Mean",            "Low Band Energy", "Mid Band Energy", "Spectral Centroid"],
            "Rank 3":       ["Std Dev",          "High Band Energy", "Spectral Centroid", "Std Dev"],
        })
        st.dataframe(top_features, hide_index=True, use_container_width=True)

    with tab4:
        st.subheader("Feature Importance Shift Across Severity (XGBoost)")
        st.markdown(
            "The central SHAP finding: feature importance is **not constant** — "
            "it changes as fault severity increases."
        )
        img = load_fig("shap_severity_impact.png")
        if img:
            st.image(img, use_container_width=True)

        st.markdown("---")
        st.markdown("**Top 3 features with the largest importance shift (0.007\" → 0.021\"):**")
        shift_df = pd.DataFrame({
            "Feature":      ["Shape Factor", "High Band Energy", "Mean"],
            "Sev 0.007\"":  [0.485, 0.784, 0.409],
            "Sev 0.021\"":  [0.714, 0.633, 0.536],
            "Δ":            ["+0.229 ↑", "−0.151 ↓", "+0.127 ↑"],
            "Interpretation": [
                "Time-domain amplitude becomes more diagnostic at high severity",
                "High-freq energy dissipates as fault grows",
                "Mean signal level rises with fault size",
            ],
        })
        st.dataframe(shift_df, hide_index=True, use_container_width=True)

        st.success(
            "**Research insight:** As faults worsen, the vibration signal shifts from "
            "high-frequency spectral content toward time-domain amplitude statistics. "
            "This implies that different sensor strategies may be optimal at different "
            "stages of fault progression."
        )
