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

st.set_page_config(page_title="Gait Analysis", page_icon="🚶", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a,#111128);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#0f3460);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 style="color:#e2e8f0;">🚶 Gait &amp; Kinetic Signal Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Analyze IMU accelerometer and force sensor signals to evaluate stride length, cadence, freeze index, and gait asymmetry.</p>', unsafe_allow_html=True)

samples = get_available_sample_files()

tab_upload, tab_sample = st.tabs(["📤 Upload Gait Signal", "📁 Select from PhysioNet Gait Dataset"])

gait_target = None
with tab_upload:
    gait_file = st.file_uploader("Upload Gait file (.txt, .csv, .tsv, .npy)", type=["txt", "csv", "tsv", "npy"])
    if gait_file:
        gait_target = gait_file
        st.success(f"✅ {gait_file.name} ({gait_file.size//1024} KB)")

with tab_sample:
    g_choices = ["-- Select a sample --"] + list(samples["gait"].keys())
    g_sel = st.selectbox("Choose Sample from Dataset", g_choices, index=0)
    if g_sel != "-- Select a sample --":
        gait_target = samples["gait"][g_sel]
        st.caption(f"📁 Selected: `{Path(gait_target).name}`")

st.markdown("---")
analyze = st.button("🔬 Analyze Gait Signal", type="primary", use_container_width=False)

if analyze:
    if gait_target is None:
        st.warning("⚠️ Please upload a gait file or select a sample from the dataset first.")
    else:
        with st.spinner("Processing gait dynamics & detecting heel strikes..."):
            vec_40, feats, accel_v, accel_ap, fs = load_and_process_gait(gait_target)
            dur = len(accel_v) / fs
            t = np.linspace(0, dur, len(accel_v))
            # Medial-lateral: derive from signal or use a phase-shifted version
            accel_ml = np.diff(np.concatenate([[0], accel_ap])) * 0.5
            accel_ml = accel_ml[:len(accel_v)]
            features = {
                "Cadence (steps/min)":       float(feats.get("cadence", 0.0)),
                "Walking Speed (m/s)":       float(feats.get("walking_speed", 0.0)),
                "Step Length (m)":           float(feats.get("step_length_mean", 0.0)),
                "Stride Length (m)":         float(feats.get("step_length_mean", 0.0) * 2.0),
                "Swing Time (s)":            float(feats.get("swing_time_mean", 0.0)),
                "Stance Time (s)":           float(feats.get("stance_time_mean", 0.0)),
                "Double Support Time (s)":   float(feats.get("double_support_mean", 0.0)),
                "Gait Asymmetry (%)":        float(feats.get("gait_asymmetry", 0.0)),
                "Step Variability (CoV%)":   float(feats.get("step_time_cov", 0.0)),
                "Freeze-of-Gait Index":      float(feats.get("fog_index", 0.0)),
                "RMS Vertical Acceleration": float(feats.get("rms_vertical", 0.0)),
                "Jerk RMS":                  float(feats.get("jerk_rms", 0.0)),
            }

        fname = Path(gait_target).name if isinstance(gait_target, str) else gait_file.name if gait_file else "gait"
        st.success(f"✅ Gait signal processed: {len(accel_v)} samples @ {fs:.0f} Hz, duration {dur:.1f}s — {fname}")

        tab1, tab2, tab3 = st.tabs([
            "📈 Multiaxial Accelerometer",
            "📊 Gait Kinetic Features",
            "⏱️ Heel Strike & Phase Events"
        ])

        with tab1:
            samples_show = min(2000, len(accel_v))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t[:samples_show], y=accel_v[:samples_show],
                                     name="Vertical Accel", line=dict(color="#6366f1", width=1.5)))
            fig.add_trace(go.Scatter(x=t[:samples_show], y=accel_ap[:samples_show],
                                     name="Anterior-Posterior", line=dict(color="#10b981", width=1.5)))
            fig.add_trace(go.Scatter(x=t[:samples_show], y=accel_ml[:samples_show],
                                     name="Medial-Lateral (derived)", line=dict(color="#f59e0b", width=1.5)))

            from scipy.signal import find_peaks
            peaks, props = find_peaks(
                accel_v[:samples_show],
                height=np.mean(accel_v[:samples_show]) + 0.3 * np.std(accel_v[:samples_show]),
                distance=int(fs * 0.25)
            )
            if len(peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=t[peaks], y=accel_v[peaks], mode="markers",
                    marker=dict(color="#ef4444", size=9, symbol="triangle-down"),
                    name=f"Heel Strike ({len(peaks)} detected)"
                ))

            fig.update_layout(
                title=dict(text=f"3-Axis Kinematic Acceleration — {fname}", font=dict(color="#e2e8f0")),
                paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(title="Time (seconds)", color="#94a3b8", gridcolor="#2d2f54"),
                yaxis=dict(title="Amplitude (g)", color="#94a3b8", gridcolor="#2d2f54"),
                legend=dict(font=dict(color="#94a3b8")), height=420
            )
            st.plotly_chart(fig, use_container_width=True)

            # Frequency spectrum of vertical accel
            from scipy.signal import welch
            freqs_w, psd = welch(accel_v, fs=fs, nperseg=min(512, len(accel_v)//2))
            mask = freqs_w <= 20
            fig_freq = go.Figure(go.Scatter(
                x=freqs_w[mask], y=psd[mask], mode="lines", fill="tozeroy",
                fillcolor="rgba(99,102,241,0.2)", line=dict(color="#6366f1", width=2)
            ))
            fig_freq.add_vrect(x0=0.5, x1=3.0, fillcolor="#10b981", opacity=0.1,
                               annotation_text="Walking Band", annotation_position="top left",
                               annotation=dict(font=dict(color="#10b981", size=10)))
            fig_freq.update_layout(
                title=dict(text="Power Spectrum of Vertical Acceleration", font=dict(color="#e2e8f0")),
                paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(title="Frequency (Hz)", color="#94a3b8", gridcolor="#2d2f54"),
                yaxis=dict(title="Power", color="#94a3b8", gridcolor="#2d2f54"),
                height=280
            )
            st.plotly_chart(fig_freq, use_container_width=True)

        with tab2:
            col1, col2, col3 = st.columns(3)
            items = list(features.items())
            for i, (k, v) in enumerate(items):
                with [col1, col2, col3][i % 3]:
                    st.metric(k, f"{v:.3f}" if v < 10 else f"{v:.1f}")

            # Bar chart of key features
            key_feats = ["Cadence (steps/min)", "Gait Asymmetry (%)", "Step Variability (CoV%)",
                         "Freeze-of-Gait Index", "RMS Vertical Acceleration"]
            key_vals = [features[k] for k in key_feats]
            key_colors = ["#6366f1", "#f59e0b", "#8b5cf6", "#ef4444", "#10b981"]
            fig_bar = go.Figure(go.Bar(
                x=key_feats, y=key_vals, marker_color=key_colors,
                text=[f"{v:.2f}" for v in key_vals], textposition="auto",
                textfont=dict(color="white")
            ))
            fig_bar.update_layout(
                title=dict(text="Key Gait Biomarkers", font=dict(color="#e2e8f0")),
                paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(color="#94a3b8", tickangle=-20),
                yaxis=dict(color="#94a3b8", gridcolor="#2d2f54"),
                height=320
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            swing = features.get("Swing Time (s)", 0.4) or 0.4
            stance = features.get("Stance Time (s)", 0.6) or 0.6
            double_s = features.get("Double Support Time (s)", 0.15) or 0.15
            total = swing + stance + double_s
            phases = {
                "Stance Phase": (stance / total) * 100,
                "Swing Phase": (swing / total) * 100,
                "Double Support": (double_s / total) * 100,
            }
            col_pie, col_detail = st.columns(2)
            with col_pie:
                fig_pie = go.Figure(go.Pie(
                    labels=list(phases.keys()), values=list(phases.values()),
                    hole=0.45, marker=dict(colors=["#6366f1", "#10b981", "#f59e0b"])
                ))
                fig_pie.update_layout(
                    title=dict(text="Gait Cycle Phase Breakdown (%)", font=dict(color="#e2e8f0")),
                    paper_bgcolor="#1e2139", height=340,
                    legend=dict(font=dict(color="#94a3b8"))
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_detail:
                # Step interval analysis from peaks
                from scipy.signal import find_peaks
                peaks_all, _ = find_peaks(
                    accel_v,
                    height=np.mean(accel_v) + 0.3 * np.std(accel_v),
                    distance=int(fs * 0.25)
                )
                if len(peaks_all) > 2:
                    step_intervals = np.diff(peaks_all) / fs
                    fig_hist = go.Figure(go.Histogram(
                        x=step_intervals, nbinsx=20,
                        marker_color="#8b5cf6", opacity=0.8
                    ))
                    fig_hist.update_layout(
                        title=dict(text=f"Step Interval Distribution ({len(peaks_all)} steps detected)", font=dict(color="#e2e8f0")),
                        paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                        xaxis=dict(title="Step Interval (s)", color="#94a3b8", gridcolor="#2d2f54"),
                        yaxis=dict(title="Count", color="#94a3b8", gridcolor="#2d2f54"),
                        height=340
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("Not enough heel strike events detected to plot distribution.")

        st.session_state["gait_features"] = features
        st.session_state["gait_vec"] = vec_40

else:
    st.markdown("""
    <div style="
      background: rgba(99,102,241,0.06);
      border: 1px dashed rgba(99,102,241,0.3);
      border-radius: 16px;
      padding: 3rem;
      text-align: center;
      margin-top: 1rem;
    ">
      <div style="font-size: 3rem;">🚶</div>
      <h3 style="color:#e2e8f0; margin-top:0.5rem;">Upload or Select a Gait File</h3>
      <p style="color: #94a3b8; font-size: 1rem; max-width: 600px; margin: 0 auto;">
        Upload a <b>.txt / .csv / .tsv / .npy</b> gait file or choose a sample from the PhysioNet dataset above,
        then click <b>Analyze Gait Signal</b> to visualize accelerometry and stride metrics.
      </p>
    </div>
    """, unsafe_allow_html=True)
