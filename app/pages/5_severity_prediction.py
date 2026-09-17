"""
============================================================
Page 5: Severity Prediction & Parkinson's Diagnosis
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Combines Voice, EEG, and Gait signals to:
  1. Determine whether Parkinson's is DETECTED or NOT DETECTED
  2. Estimate total UPDRS motor severity score (0 - 108)
  3. Classify clinical severity stage (Mild / Moderate / Severe)
  4. Provide per-modality biomarker insights
============================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.processing.unified_loader import (
    get_available_sample_files,
    load_and_process_voice,
    load_and_process_eeg,
    load_and_process_gait,
    predict_parkinsons,
)

st.set_page_config(page_title="Severity Prediction", page_icon="🎯", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #111128 100%); }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%); }
.stButton > button { background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  font-weight: 600 !important; }
</style>""", unsafe_allow_html=True)

st.markdown('<h1 style="color:#e2e8f0;">🎯 Multimodal Parkinson\'s Disease Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">End-to-end deep learning system predicting Parkinson\'s detection status, UPDRS score, and severity level.</p>', unsafe_allow_html=True)

# Patient Demographics
with st.expander("👤 Patient Demographics & Record", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_name = st.text_input("Patient Name", value="Patient-001", key="pred_name")
        patient_age  = st.number_input("Age", 20, 100, 68, key="pred_age")
    with col2:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        disease_dur    = st.number_input("Disease History / Tremor Duration (years)", 0.0, 30.0, 2.5, 0.5)
    with col3:
        clinician   = st.text_input("Consulting Neurologist / Clinician", value="Dr. Specialist")
        medications = st.text_input("Current Dopaminergic Therapy", value="Levodopa/Carbidopa (100/25mg)")

st.markdown("---")

# File Selection & Upload
samples = get_available_sample_files()

tab_upload, tab_sample = st.tabs(["📤 Upload Signal Files", "📁 Choose from Real Dataset Samples"])

voice_target = None
eeg_target   = None
gait_target  = None

with tab_upload:
    st.info("💡 Supports all medical signal formats: Voice (.wav, .mp3), EEG (.set, .edf, .npy, .csv, .txt), and Gait (.txt, .csv, .tsv).")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🎤 Voice Recording**")
        v_up = st.file_uploader("Voice (.wav, .mp3)", type=["wav", "mp3", "ogg"], key="up_v")
        if v_up:
            st.success(f"✅ {v_up.name} ({v_up.size//1024} KB)")
            voice_target = v_up
            
    with col2:
        st.markdown("**🧠 EEG Signal**")
        e_up = st.file_uploader("EEG (.set, .edf, .npy, .csv, .txt)", type=["set", "edf", "npy", "csv", "tsv", "txt"], key="up_e")
        if e_up:
            st.success(f"✅ {e_up.name} ({e_up.size//1024} KB)")
            eeg_target = e_up
            
    with col3:
        st.markdown("**🚶 Gait Sensor Signal**")
        g_up = st.file_uploader("Gait (.txt, .csv, .tsv, .npy)", type=["txt", "csv", "tsv", "npy"], key="up_g")
        if g_up:
            st.success(f"✅ {g_up.name} ({g_up.size//1024} KB)")
            gait_target = g_up

with tab_sample:
    st.markdown("⚡ *Quick 1-Click Testing using real clinical files included in your project:*")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🎤 Select Voice Sample**")
        v_choices = ["None"] + list(samples["voice"].keys())
        v_sel = st.selectbox("Voice Sample", v_choices, index=1 if len(v_choices)>1 else 0)
        if v_sel != "None":
            voice_target = samples["voice"][v_sel]
            st.caption(f"📁 Path: {Path(voice_target).name}")
            
    with col2:
        st.markdown("**🧠 Select EEG Sample**")
        e_choices = ["None"] + list(samples["eeg"].keys())
        e_sel = st.selectbox("EEG Sample", e_choices, index=1 if len(e_choices)>1 else 0)
        if e_sel != "None":
            eeg_target = samples["eeg"][e_sel]
            st.caption(f"📁 Path: {Path(eeg_target).name}")
            
    with col3:
        st.markdown("**🚶 Select Gait Sample**")
        g_choices = ["None"] + list(samples["gait"].keys())
        g_sel = st.selectbox("Gait Sample", g_choices, index=1 if len(g_choices)>1 else 0)
        if g_sel != "None":
            gait_target = samples["gait"][g_sel]
            st.caption(f"📁 Path: {Path(gait_target).name}")

st.markdown("---")

# Run Assessment Button
col_btn, col_demo = st.columns([2, 3])
with col_btn:
    run_pred = st.button("🚀 Analyze All & Predict Parkinson's", type="primary", use_container_width=True)
with col_demo:
    use_synth = st.checkbox("Generate Synthetic Demo Signals if no files uploaded", value=False)

if run_pred:
    has_inputs = any([voice_target is not None, eeg_target is not None, gait_target is not None])
    
    if not has_inputs and not use_synth:
        st.warning("⚠️ Please upload at least one signal file or select a sample from the dataset.")
    else:
        with st.spinner("🔄 Processing multimodal signals and evaluating deep neural network..."):
            progress = st.progress(0)
            
            # Step 1: Voice
            voice_vec = None
            voice_meta = {}
            if voice_target is not None:
                progress.progress(25, "🎤 Extracting acoustic dysphonia, jitter, shimmer, and MFCC features...")
                voice_vec, voice_meta, _, _ = load_and_process_voice(voice_target)
            elif use_synth:
                voice_vec = np.random.randn(200).astype(np.float32)

            # Step 2: EEG
            eeg_vec = None
            eeg_meta = {}
            if eeg_target is not None:
                progress.progress(50, "🧠 Processing EEG brain rhythm band powers and connectivity...")
                eeg_vec, eeg_meta, _, _ = load_and_process_eeg(eeg_target)
            elif use_synth:
                eeg_vec = np.random.randn(150).astype(np.float32)

            # Step 3: Gait
            gait_vec = None
            gait_meta = {}
            if gait_target is not None:
                progress.progress(75, "🚶 Analyzing IMU gait cycle, cadence, stride asymmetry, and freeze index...")
                gait_vec, gait_meta, _, _, _ = load_and_process_gait(gait_target)
            elif use_synth:
                gait_vec = np.random.randn(40).astype(np.float32)

            # Step 4: Model Forward Pass
            progress.progress(90, "🤖 Running Transformer Cross-Modal Fusion Model...")
            result = predict_parkinsons(voice_vec=voice_vec, eeg_vec=eeg_vec, gait_vec=gait_vec)
            progress.progress(100, "✅ Inference complete!")
            progress.empty()

            # Save in session state for reports
            st.session_state["prediction_result"] = result
            st.session_state["patient_info"] = {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "duration": disease_dur,
                "clinician": clinician,
                "medications": medications
            }

        # Display Diagnostic Results
        is_pd = result["is_parkinsons"]
        diag_title = result["diagnosis_title"]
        badge_color = "#ef4444" if is_pd else "#10b981"
        badge_bg    = "rgba(239, 68, 68, 0.15)" if is_pd else "rgba(16, 185, 129, 0.15)"
        border_col  = "#ef4444" if is_pd else "#10b981"
        icon_status = "🔴" if is_pd else "🟢"

        st.markdown(f"""
        <div style="
          background: {badge_bg};
          border: 2px solid {border_col};
          border-radius: 16px;
          padding: 1.8rem;
          text-align: center;
          margin: 1.5rem 0;
          box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        ">
          <div style="font-size: 1.1rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
            Diagnostic Finding
          </div>
          <div style="font-size: 2.3rem; font-weight: 800; color: {badge_color}; margin: 0.5rem 0;">
            {icon_status} {diag_title}
          </div>
          <div style="font-size: 1.1rem; color: #e2e8f0;">
            Estimated UPDRS Motor Score: <b style="color:{badge_color}; font-size: 1.4rem;">{result['updrs_score']:.1f}</b> / 108 &nbsp;|&nbsp;
            Severity Stage: <b style="color:{badge_color}; font-size: 1.2rem;">{result['severity_label']}</b> &nbsp;|&nbsp;
            Confidence: <b style="color:#e2e8f0;">{result['confidence']*100:.1f}%</b>
          </div>
          <div style="margin-top: 0.8rem; font-size: 0.95rem; color: #cbd5e1; max-width: 800px; margin-left: auto; margin-right: auto;">
            <i>{result['clinical_advice']}</i>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Key Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Parkinson's Status", "POSITIVE" if is_pd else "NEGATIVE", delta="Abnormal" if is_pd else "Healthy Range", delta_color="inverse" if is_pd else "normal")
        with col2:
            st.metric("Total UPDRS Score", f"{result['updrs_score']:.1f}", "Scale: 0 - 108")
        with col3:
            st.metric("Severity Level", result["severity_label"], f"{result['confidence']*100:.1f}% Conf.")
        with col4:
            st.metric("Modalities Evaluated", f"{result['modalities_used']}/3", "Active fusion inputs")

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            # UPDRS Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["updrs_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "UPDRS Severity Scale (0 = Normal, 108 = Extreme)", "font": {"size": 14, "color": "#e2e8f0"}},
                number={"font": {"size": 34, "color": "#e2e8f0"}, "suffix": " / 108"},
                gauge={
                    "axis": {"range": [0, 108], "tickcolor": "#94a3b8"},
                    "bar": {"color": badge_color, "thickness": 0.28},
                    "bgcolor": "#1e2139",
                    "steps": [
                        {"range": [0, 25], "color": "rgba(16, 185, 129, 0.25)"},
                        {"range": [25, 60], "color": "rgba(245, 158, 11, 0.25)"},
                        {"range": [60, 108], "color": "rgba(239, 68, 68, 0.25)"},
                    ],
                    "threshold": {"line": {"color": "#818cf8", "width": 3}, "thickness": 0.8, "value": result["updrs_score"]}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="#1e2139", plot_bgcolor="#1e2139", height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            # Class Probabilities
            probs = result["class_probabilities"]
            labels = ["Mild / Early", "Moderate", "Severe"]
            colors = ["#10b981", "#f59e0b", "#ef4444"]
            
            fig_bar = go.Figure(go.Bar(
                x=labels,
                y=[p * 100 for p in probs],
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="auto",
                textfont=dict(color="white", size=13)
            ))
            fig_bar.update_layout(
                title=dict(text="Severity Stage Probability Distribution", font=dict(color="#e2e8f0", size=14)),
                paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(tickfont=dict(color="#94a3b8")),
                yaxis=dict(title="Probability (%)", tickfont=dict(color="#94a3b8"), gridcolor="#2d2f54", range=[0, 100]),
                height=280, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Modality Attention & Signal Breakdown
        st.markdown("### 🔬 Multi-Signal Biomarker Findings")
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.markdown("""
            <div style="background: #1e2139; border-radius: 12px; padding: 1.2rem; border-left: 4px solid #6366f1;">
              <h4 style="color:#818cf8; margin-top:0;">🎤 Voice Modality</h4>
              <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0.4rem;">
                <b>Acoustic Perturbations:</b> Analyzes micro-tremors, vocal cord rigidity, and airflow instability.<br>
                <b>Key Markers:</b> Jitter (frequency perturbation), Shimmer (amplitude variation), HNR (noise ratio).
              </p>
            </div>
            """, unsafe_allow_html=True)
            if voice_meta:
                v_df = pd.DataFrame(list(voice_meta.items())[:6], columns=["Feature", "Value"])
                v_df["Value"] = v_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
                st.dataframe(v_df, hide_index=True, use_container_width=True)

        with col_m2:
            st.markdown("""
            <div style="background: #1e2139; border-radius: 12px; padding: 1.2rem; border-left: 4px solid #06b6d4;">
              <h4 style="color:#38bdf8; margin-top:0;">🧠 EEG Modality</h4>
              <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0.4rem;">
                <b>Cortical Rhythms:</b> Detects basal ganglia-thalamocortical disruption and background slowing.<br>
                <b>Key Markers:</b> Increased Theta/Delta power, diminished Alpha peak frequency, altered coherence.
              </p>
            </div>
            """, unsafe_allow_html=True)
            if eeg_meta:
                e_df = pd.DataFrame(list(eeg_meta.items())[:6], columns=["Feature", "Value"])
                e_df["Value"] = e_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
                st.dataframe(e_df, hide_index=True, use_container_width=True)

        with col_m3:
            st.markdown("""
            <div style="background: #1e2139; border-radius: 12px; padding: 1.2rem; border-left: 4px solid #f59e0b;">
              <h4 style="color:#fbbf24; margin-top:0;">🚶 Gait Modality</h4>
              <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0.4rem;">
                <b>Kinematics:</b> Identifies bradykinesia, festination, and bilateral movement asymmetry.<br>
                <b>Key Markers:</b> Stride variability (CoV), reduced cadence, Freeze-of-Gait (FOG) index.
              </p>
            </div>
            """, unsafe_allow_html=True)
            if gait_meta:
                g_df = pd.DataFrame(list(gait_meta.items())[:6], columns=["Feature", "Value"])
                g_df["Value"] = g_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
                st.dataframe(g_df, hide_index=True, use_container_width=True)

        # Cross-Modal Attention Weights
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📡 Cross-Modal Attention & Modality Contributions")
        w = result["modality_weights"]
        categories = ["Voice", "EEG", "Gait", "Voice"]
        weights_r  = [w["Voice"], w["EEG"], w["Gait"], w["Voice"]]
        
        fig_radar = go.Figure(go.Scatterpolar(
            r=weights_r, theta=categories, fill="toself",
            fillcolor="rgba(99, 102, 241, 0.25)",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(size=8, color="#818cf8")
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#94a3b8"), gridcolor="#2d2f54"),
                angularaxis=dict(tickfont=dict(color="#e2e8f0", size=13)),
                bgcolor="#1e2139"
            ),
            paper_bgcolor="#1e2139", height=320, margin=dict(l=40, r=40, t=30, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Next Action Links
        st.info("📋 You can now proceed to **Page 6: Explainable AI** for SHAP / Attention maps or **Page 7: Report Generation** to download the official PDF medical report.")

elif not run_pred:
    st.markdown("""
    <div style="
      background: rgba(99,102,241,0.06);
      border: 1px dashed rgba(99,102,241,0.3);
      border-radius: 16px;
      padding: 3rem;
      text-align: center;
      margin-top: 1rem;
    ">
      <div style="font-size: 3rem;">🎯</div>
      <h3 style="color:#e2e8f0; margin-top:0.5rem;">Ready to Analyze Parkinson's Disease</h3>
      <p style="color: #94a3b8; font-size: 1rem; max-width: 650px; margin: 0 auto;">
        Upload medical signal files (.wav, .set, .edf, .txt) or pick a sample from your local dataset above, then click <b>Analyze All &amp; Predict Parkinson's</b>.
      </p>
    </div>
    """, unsafe_allow_html=True)
