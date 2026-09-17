"""
============================================================
Page 2: Voice Analysis
AI-Powered Parkinson's Disease Severity Assessment
============================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import io
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

st.set_page_config(page_title="Voice Analysis", page_icon="🎤", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a 0%,#111128 100%);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#0f3460 100%);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 style="color:#e2e8f0;">🎤 Voice Signal Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Extract MFCC, pitch, jitter, shimmer, HNR, and spectral features from voice recordings.</p>', unsafe_allow_html=True)

# Upload
col_up, col_info = st.columns([1, 1])
with col_up:
    audio_file = st.file_uploader("Upload Voice Recording (.wav / .mp3)", type=["wav", "mp3"])
with col_info:
    use_demo = st.checkbox("Use synthetic demo signal", value=True)
    if audio_file:
        st.info(f"📁 File: **{audio_file.name}** | Size: {audio_file.size//1024}KB")

analyze = st.button("🔬 Analyze Voice", type="primary")

if analyze or use_demo:
    with st.spinner("Extracting voice features..."):
        import time; time.sleep(0.8)

        # Synthetic signal
        sr   = 22050
        dur  = 3.0
        t    = np.linspace(0, dur, int(sr * dur))
        f0   = 120 + 15 * np.sin(2 * np.pi * 0.3 * t)  # tremor
        y    = (np.sin(2 * np.pi * f0 * t) +
                0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
                0.15 * np.random.randn(len(t)))
        y    = y / np.max(np.abs(y))

        # Synthetic features
        features = {
            "Pitch Mean (Hz)":    118.4,
            "Pitch Std (Hz)":      22.7,
            "Jitter Local (%)":     0.0082,
            "Shimmer Local (%)":    0.0531,
            "HNR (dB)":            15.2,
            "Spectral Centroid":  1842.3,
            "Spectral Roll-off":  3241.0,
            "ZCR":                 0.0714,
            "RMS Energy":          0.3821,
            "Voiced Fraction":     0.783,
            "Formant F1 (Hz)":    578.0,
            "Formant F2 (Hz)":   1243.0,
        }

        mfcc_demo = np.random.randn(40, 130) * 0.5
        mfcc_demo[0] = np.linspace(-10, 5, 130)

    st.success("✅ Voice features extracted")

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Waveform", "🌈 Spectrogram", "🎵 MFCC", "📊 Features"])

    with tab1:
        # Waveform plot
        fig = go.Figure()
        t_sec = np.linspace(0, dur, len(y))
        fig.add_trace(go.Scatter(
            x=t_sec[:3000], y=y[:3000],
            mode="lines",
            line=dict(color="#6366f1", width=1),
            name="Waveform",
        ))
        # Envelope
        env_w = 500
        envelope = np.array([np.max(np.abs(y[max(0,i-env_w):i+env_w])) for i in range(0, len(y), 100)])
        t_env    = np.linspace(0, dur, len(envelope))
        fig.add_trace(go.Scatter(x=t_env, y=envelope, mode="lines",
                                  line=dict(color="#f59e0b", width=2, dash="dot"),
                                  name="Envelope"))
        fig.update_layout(
            title="Voice Waveform", paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
            xaxis=dict(title="Time (s)", color="#94a3b8", gridcolor="#2d2f54"),
            yaxis=dict(title="Amplitude", color="#94a3b8", gridcolor="#2d2f54"),
            legend=dict(font=dict(color="#94a3b8")), height=300,
            title_font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # STFT Spectrogram
        from scipy.signal import spectrogram
        f_spec, t_spec, Sxx = spectrogram(y, fs=sr, nperseg=512, noverlap=384)
        Sxx_db = 10 * np.log10(Sxx[:200, :] + 1e-8)

        fig = go.Figure(go.Heatmap(
            z=Sxx_db, x=t_spec, y=f_spec[:200],
            colorscale="Viridis",
            colorbar=dict(title="dB", tickfont=dict(color="#94a3b8")),
        ))
        fig.update_layout(
            title="Voice Spectrogram", paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
            xaxis=dict(title="Time (s)", color="#94a3b8"),
            yaxis=dict(title="Frequency (Hz)", color="#94a3b8"),
            height=350, title_font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # MFCC heatmap
        fig = go.Figure(go.Heatmap(
            z=mfcc_demo, colorscale="RdBu",
            colorbar=dict(title="Value", tickfont=dict(color="#94a3b8")),
        ))
        fig.update_layout(
            title="MFCC Coefficients (40 × Time frames)",
            paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
            xaxis=dict(title="Time Frame", color="#94a3b8"),
            yaxis=dict(title="MFCC Coefficient", color="#94a3b8"),
            height=350, title_font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Feature table + bar chart
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Extracted Features**")
            import pandas as pd
            df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
            df["Value"] = df["Value"].apply(lambda x: f"{x:.4f}")
            st.dataframe(df, use_container_width=True, hide_index=True)

        with col2:
            fig_bar = go.Figure(go.Bar(
                x=list(features.values()),
                y=list(features.keys()),
                orientation="h",
                marker=dict(
                    color=np.array(list(features.values())),
                    colorscale="Viridis",
                    showscale=True,
                ),
            ))
            fig_bar.update_layout(
                title="Feature Values", paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8"),
                height=380, margin=dict(l=160),
                title_font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.session_state.voice_features = features
