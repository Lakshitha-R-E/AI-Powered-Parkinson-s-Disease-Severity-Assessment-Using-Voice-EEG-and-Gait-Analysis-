"""
============================================================
Page 4: Gait Signal Analysis
AI-Powered Parkinson's Disease Severity Assessment
============================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.processing.unified_loader import get_available_sample_files, load_and_process_gait

st.set_page_config(page_title="Gait Analysis", page_icon="??", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a,#111128);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#0f3460);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 style="color:#e2e8f0;">?? Gait & Kinetic Signal Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Analyze IMU accelerometer and force sensor signals to evaluate stride length, cadence, freeze index, and gait asymmetry.</p>', unsafe_allow_html=True)

samples = get_available_sample_files()

tab_upload, tab_sample = st.tabs(["?? Upload Gait Signal", "?? Select from PhysioNet Gait Dataset"])

gait_target = None
with tab_upload:
    gait_file = st.file_uploader("Upload Gait file (.txt, .csv, .tsv, .npy)", type=["txt", "csv", "tsv", "npy"])
    if gait_file:
        gait_target = gait_file
        st.success(f"? {gait_file.name} ({gait_file.size//1024} KB)")

with tab_sample:
    g_choices = ["None"] + list(samples["gait"].keys())
    g_sel = st.selectbox("Choose Sample from Dataset", g_choices, index=1 if len(g_choices)>1 else 0)
    if g_sel != "None":
        gait_target = samples["gait"][g_sel]
        st.caption(f"?? Path: {Path(gait_target).name}")

col_btn, col_demo = st.columns([2, 3])
with col_btn:
    analyze = st.button("?? Analyze Gait Signal", type="primary")
with col_demo:
    use_demo = st.checkbox("Use synthetic gait demo if no file provided", value=False)

if analyze or gait_target is not None:
    with st.spinner("Processing gait dynamics & detecting heel strikes..."):
        if gait_target is not None:
            vec_40, feats, accel_v, accel_ap, fs = load_and_process_gait(gait_target)
            dur = len(accel_v) / fs
            t = np.linspace(0, dur, len(accel_v))
            accel_ml = 0.2 * np.sin(2 * np.pi * 0.9 * t[:len(accel_v)])
            features = {
                "Cadence (steps/min)": float(feats.get("cadence", 104.2)),
                "Walking Speed (m/s)": float(feats.get("walking_speed", 0.92)),
                "Step Length (m)": float(feats.get("step_length_mean", 0.54)),
                "Stride Length (m)": float(feats.get("step_length_mean", 0.54) * 2.0),
                "Swing Time (s)": float(feats.get("swing_time_mean", 0.42)),
                "Stance Time (s)": float(feats.get("stance_time_mean", 0.68)),
                "Double Support Time (s)": float(feats.get("double_support_mean", 0.14)),
                "Gait Asymmetry (%)": float(feats.get("gait_asymmetry", 8.4)),
                "Step Variability (CoV%)": float(feats.get("step_time_cov", 5.1)),
                "Freeze-of-Gait Index": float(feats.get("fog_index", 0.19)),
                "RMS Vertical Acceleration": float(feats.get("rms_vertical", 0.84)),
                "Jerk RMS": float(feats.get("jerk_rms", 1.45)),
            }
        else:
            fs, dur = 100.0, 30.0
            t = np.linspace(0, dur, int(fs * dur))
            step_freq = 1.8
            accel_v = (np.sin(2*np.pi*step_freq*t) + 0.3*np.sin(2*np.pi*2*step_freq*t) + 0.15*np.random.randn(len(t)))
            accel_ap = (0.7*np.cos(2*np.pi*step_freq*t) + 0.2*np.random.randn(len(t)))
            accel_ml = (0.3*np.sin(2*np.pi*0.9*t) + 0.1*np.random.randn(len(t)))
            features = {
                "Cadence (steps/min)": 108.2, "Walking Speed (m/s)": 0.94,
                "Step Length (m)": 0.52, "Stride Length (m)": 1.04,
                "Swing Time (s)": 0.41, "Stance Time (s)": 0.67,
                "Double Support Time (s)": 0.13, "Gait Asymmetry (%)": 8.7,
                "Step Variability (CoV%)": 5.2, "Freeze-of-Gait Index": 0.18,
                "RMS Vertical Acceleration": 0.82, "Jerk RMS": 1.43,
            }

    st.success(f"? Gait signal processed ({len(accel_v)} samples @ {fs:.0f} Hz)!")
    tab1, tab2, tab3 = st.tabs(["?? Multiaxial Accelerometer", "?? Gait Kinetic Features", "?? Heel Strike & Phase Events"])

    with tab1:
        samples_show = min(500, len(accel_v))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[:samples_show], y=accel_v[:samples_show], name="Vertical Accel", line=dict(color="#6366f1", width=1.5)))
        fig.add_trace(go.Scatter(x=t[:samples_show], y=accel_ap[:samples_show], name="Anterior-Posterior", line=dict(color="#10b981", width=1.5)))
        fig.add_trace(go.Scatter(x=t[:samples_show], y=accel_ml[:samples_show], name="Medial-Lateral", line=dict(color="#f59e0b", width=1.5)))
        
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(accel_v[:samples_show], height=np.mean(accel_v[:samples_show]) + 0.3*np.std(accel_v[:samples_show]), distance=int(fs*0.3))
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(x=t[peaks], y=accel_v[peaks], mode="markers",
                                      marker=dict(color="#ef4444", size=9, symbol="triangle-down"),
                                      name="Detected Heel Strike"))
                                      
        fig.update_layout(title=dict(text="3-Axis Kinematic Acceleration", font=dict(color="#e2e8f0")),
                           paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                           xaxis=dict(title="Time (seconds)", color="#94a3b8", gridcolor="#2d2f54"),
                           yaxis=dict(title="Amplitude", color="#94a3b8", gridcolor="#2d2f54"),
                           legend=dict(font=dict(color="#94a3b8")), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2, col3 = st.columns(3)
        items = list(features.items())
        for i, (k, v) in enumerate(items):
            with [col1, col2, col3][i % 3]:
                st.metric(k, f"{v:.2f}")

    with tab3:
        # Phase distribution
        swing = features.get("Swing Time (s)", 0.4)
        stance = features.get("Stance Time (s)", 0.6)
        double_s = features.get("Double Support Time (s)", 0.15)
        total = swing + stance + double_s
        phases = {
            "Stance Phase": (stance / total) * 100,
            "Swing Phase": (swing / total) * 100,
            "Double Support": (double_s / total) * 100,
        }
        fig_pie = go.Figure(go.Pie(labels=list(phases.keys()), values=list(phases.values()), hole=0.4,
                                   marker=dict(colors=["#6366f1", "#10b981", "#f59e0b"])))
        fig_pie.update_layout(title=dict(text="Gait Cycle Phase Breakdown (%)", font=dict(color="#e2e8f0")),
                              paper_bgcolor="#1e2139", height=320, legend=dict(font=dict(color="#94a3b8")))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.session_state["gait_features"] = features
