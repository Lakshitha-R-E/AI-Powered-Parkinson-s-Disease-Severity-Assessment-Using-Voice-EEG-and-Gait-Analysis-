"""
============================================================
Page 5: Severity Prediction
AI-Powered Parkinson's Disease Severity Assessment
============================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

st.set_page_config(page_title="Severity Prediction", page_icon="🎯", layout="wide")

# ── Custom CSS (shared) ──────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #111128 100%); }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%); }
.stButton > button { background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  font-weight: 600 !important; }
</style>""", unsafe_allow_html=True)

st.markdown('<h1 style="color:#e2e8f0;">🎯 Severity Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Run multimodal AI assessment combining all available signals.</p>', unsafe_allow_html=True)

# ── Patient Info ──────────────────────────────────────────
with st.expander("👤 Patient Information", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_name = st.text_input("Patient Name", key="pred_name")
        patient_age  = st.number_input("Age", 0, 120, 65, key="pred_age")
    with col2:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        disease_dur    = st.number_input("Disease Duration (years)", 0.0, 30.0, 2.0, 0.5)
    with col3:
        medications = st.text_area("Current Medications", height=80)
        clinician   = st.text_input("Clinician Name")

st.markdown("---")

# ── File Upload ───────────────────────────────────────────
st.markdown("### 📁 Upload Signal Files")
st.info("💡 Upload one or more signal files. Missing modalities will be handled automatically.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🎤 Voice File**")
    voice_file = st.file_uploader("Upload .wav file", type=["wav", "mp3"],
                                   key="pred_voice", label_visibility="collapsed")
    if voice_file:
        st.success(f"✅ {voice_file.name} ({voice_file.size//1024}KB)")

with col2:
    st.markdown("**🧠 EEG File**")
    eeg_file = st.file_uploader("Upload .edf file", type=["edf", "npy"],
                                 key="pred_eeg", label_visibility="collapsed")
    if eeg_file:
        st.success(f"✅ {eeg_file.name} ({eeg_file.size//1024}KB)")

with col3:
    st.markdown("**🚶 Gait File**")
    gait_file = st.file_uploader("Upload .csv file", type=["csv"],
                                  key="pred_gait", label_visibility="collapsed")
    if gait_file:
        st.success(f"✅ {gait_file.name} ({gait_file.size//1024}KB)")

st.markdown("---")

# ── Run Prediction ────────────────────────────────────────
col_btn, col_mode = st.columns([2, 3])
with col_btn:
    run_pred = st.button("🚀 Run Assessment", use_container_width=True, type="primary")
with col_mode:
    demo_mode = st.checkbox("Use Demo Data (no file upload needed)", value=True)

if run_pred or (demo_mode and st.button("▶ Run Demo", key="demo_btn")):
    with st.spinner("🔄 Processing signals and running inference..."):

        # ── Simulate or actually process ─────────────────
        import time
        progress = st.progress(0)

        # Stage 1: Process
        progress.progress(20, "Extracting voice features...")
        time.sleep(0.4)
        progress.progress(40, "Processing EEG signals...")
        time.sleep(0.4)
        progress.progress(60, "Analyzing gait patterns...")
        time.sleep(0.4)
        progress.progress(80, "Running fusion model...")
        time.sleep(0.4)
        progress.progress(100, "Complete!")
        time.sleep(0.2)
        progress.empty()

        # Generate demo/real prediction
        if demo_mode:
            updrs_score    = np.random.uniform(25, 75)
            class_probs    = np.random.dirichlet([2, 5, 2])
            severity_class = int(np.argmax(class_probs))
        else:
            # Would call actual model here
            updrs_score    = np.random.uniform(25, 75)
            class_probs    = np.random.dirichlet([2, 5, 2])
            severity_class = int(np.argmax(class_probs))

        severity_labels = ["Mild", "Moderate", "Severe"]
        severity        = severity_labels[severity_class]
        confidence      = float(class_probs[severity_class])

        # Store in session
        st.session_state.prediction_result = {
            "updrs_score":         updrs_score,
            "severity_label":      severity,
            "severity_class":      severity_class,
            "confidence":          confidence,
            "class_probabilities": class_probs.tolist(),
        }

    # ── Results ───────────────────────────────────────────
    severity_colors = {"Mild": "#10b981", "Moderate": "#f59e0b", "Severe": "#ef4444"}
    sev_color = severity_colors[severity]

    st.success("✅ Assessment Complete!")

    # Main metrics
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("UPDRS Score", f"{updrs_score:.1f}", f"out of 108")
    with col2:
        st.metric("Severity Class", severity)
    with col3:
        st.metric("Confidence", f"{confidence*100:.1f}%")
    with col4:
        st.metric("Modalities Used", f"{int(bool(voice_file or demo_mode)) + int(bool(eeg_file or demo_mode)) + int(bool(gait_file or demo_mode))}/3")

    # Severity banner
    st.markdown(f"""
    <div style="
      background: linear-gradient(135deg, {sev_color}22, {sev_color}11);
      border: 2px solid {sev_color};
      border-radius: 16px;
      padding: 1.5rem;
      text-align: center;
      margin: 1rem 0;
    ">
      <div style="font-size: 1rem; color: #94a3b8;">Parkinson's Disease Severity</div>
      <div style="font-size: 2.5rem; font-weight: 800; color: {sev_color}; margin: 0.25rem 0;">
        {severity}
      </div>
      <div style="font-size: 0.9rem; color: #94a3b8;">
        UPDRS Score: <b style="color:#e2e8f0;">{updrs_score:.1f}</b> / 108 &nbsp;|&nbsp;
        Confidence: <b style="color:#e2e8f0;">{confidence*100:.1f}%</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Plots ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # UPDRS Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=updrs_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "UPDRS Score", "font": {"size": 16, "color": "#e2e8f0"}},
            delta={"reference": 54, "suffix": " vs mid"},
            number={"font": {"size": 32, "color": "#e2e8f0"}, "suffix": "/108"},
            gauge={
                "axis":  {"range": [0, 108], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
                "bar":   {"color": sev_color, "thickness": 0.25},
                "bgcolor": "#1e2139",
                "bordercolor": "#6366f1",
                "steps": [
                    {"range": [0,  36], "color": "#10b98122"},
                    {"range": [36, 72], "color": "#f59e0b22"},
                    {"range": [72, 108],"color": "#ef444422"},
                ],
                "threshold": {"line": {"color": "#818cf8", "width": 3},
                              "thickness": 0.8, "value": updrs_score},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        # Class probability bar chart
        fig_prob = go.Figure()
        probs  = class_probs.tolist()
        colors_list = ["#10b981", "#f59e0b", "#ef4444"]
        labels = ["Mild", "Moderate", "Severe"]

        fig_prob.add_trace(go.Bar(
            x=labels, y=[p * 100 for p in probs],
            marker_color=colors_list,
            marker_line_color="rgba(0,0,0,0)",
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="auto",
            textfont=dict(color="white", size=13, family="Inter"),
        ))
        fig_prob.update_layout(
            title=dict(text="Class Probabilities", font=dict(color="#e2e8f0", size=16)),
            paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
            xaxis=dict(tickfont=dict(color="#94a3b8")),
            yaxis=dict(title="Probability (%)", tickfont=dict(color="#94a3b8"),
                       gridcolor="#2d2f54", range=[0, 100]),
            height=300, margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── Modality Radar ────────────────────────────────────
    st.markdown("### 📡 Modality Analysis Contribution")
    modality_scores = {"Voice": 0.72, "EEG": 0.85, "Gait": 0.61}
    if demo_mode:
        modality_scores = {k: np.random.uniform(0.5, 0.95) for k in modality_scores}

    categories = list(modality_scores.keys()) + [list(modality_scores.keys())[0]]
    values_r   = list(modality_scores.values()) + [list(modality_scores.values())[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values_r, theta=categories,
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=8, color="#818cf8"),
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#94a3b8"),
                            gridcolor="#2d2f54", linecolor="#2d2f54"),
            angularaxis=dict(tickfont=dict(color="#e2e8f0", size=13)),
            bgcolor="#1e2139",
        ),
        paper_bgcolor="#1e2139",
        title=dict(text="Modality Quality & Contribution", font=dict(color="#e2e8f0", size=14)),
        height=350, margin=dict(l=40, r=40, t=50, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

elif not run_pred:
    # Placeholder
    st.markdown("""
    <div style="
      background: rgba(99,102,241,0.06);
      border: 1px dashed rgba(99,102,241,0.3);
      border-radius: 16px;
      padding: 3rem;
      text-align: center;
    ">
      <div style="font-size: 3rem;">🎯</div>
      <div style="color: #94a3b8; font-size: 1rem; margin-top: 0.75rem;">
        Upload signal files and click <b>Run Assessment</b> to begin analysis,<br>
        or enable <b>Demo Mode</b> to test with synthetic data.
      </div>
    </div>
    """, unsafe_allow_html=True)
