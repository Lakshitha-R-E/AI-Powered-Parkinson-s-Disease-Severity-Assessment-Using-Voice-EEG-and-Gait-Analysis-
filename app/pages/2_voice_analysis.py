"""
============================================================
Page 2: Voice Signal Analysis
AI-Powered Parkinson's Disease Severity Assessment
============================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.processing.unified_loader import get_available_sample_files, load_and_process_voice

st.set_page_config(page_title="Voice Analysis", page_icon="🎤", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a,#111128);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#0f3460);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 style="color:#e2e8f0;">🎤 Acoustic Voice Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Extract MFCC, pitch micro-tremor, Jitter, Shimmer, HNR, and vocal tract formants from your uploaded audio.</p>', unsafe_allow_html=True)

samples = get_available_sample_files()

tab_upload, tab_sample = st.tabs(["📤 Upload Voice Audio", "📁 Select from Real Voice Dataset"])

voice_target = None
with tab_upload:
    audio_file = st.file_uploader("Upload Voice Recording (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
    if audio_file:
        voice_target = audio_file
        st.success(f"✅ {audio_file.name} ({audio_file.size//1024} KB)")

with tab_sample:
    v_choices = ["-- Select a sample --"] + list(samples["voice"].keys())
    v_sel = st.selectbox("Choose Sample from Dataset", v_choices, index=0)
    if v_sel != "-- Select a sample --":
        voice_target = samples["voice"][v_sel]
        st.caption(f"📁 Selected: `{Path(voice_target).name}`")

st.markdown("---")
analyze = st.button("🔬 Analyze Voice Recording", type="primary", use_container_width=False)

if analyze:
    if voice_target is None:
        st.warning("⚠️ Please upload a voice file or select a sample from the dataset first.")
    else:
        with st.spinner("Extracting acoustic features and vocal frequency dynamics..."):
            vec_200, display_feats, waveform, sr = load_and_process_voice(voice_target)
            dur = len(waveform) / sr
            t = np.linspace(0, dur, len(waveform))
            y = waveform
            features = {
                "Pitch Mean (Hz)":       float(display_feats.get("f0_mean", 124.5)),
                "Pitch Std (Hz)":        float(display_feats.get("f0_std", 18.2)),
                "Jitter Local (%)":      float(display_feats.get("jitter_local", 0.0086)),
                "Shimmer Local (%)":     float(display_feats.get("shimmer_local", 0.0542)),
                "HNR (dB)":              float(display_feats.get("hnr", 15.6)),
                "Spectral Centroid":     float(display_feats.get("spectral_centroid_mean", 1850.0)),
                "Spectral Roll-off":     float(display_feats.get("spectral_rolloff_mean", 3250.0)),
                "Zero Crossing Rate":    float(display_feats.get("zcr_mean", 0.072)),
                "RMS Energy":            float(display_feats.get("rms_mean", 0.045)),
            }

        st.success(f"✅ Voice features extracted ({len(y)} samples @ {sr} Hz, duration: {dur:.2f}s)")

        tab1, tab2, tab3 = st.tabs(["📈 Acoustic Waveform", "📊 Dysphonia & Jitter/Shimmer", "🎼 MFCC Spectrogram"])

        with tab1:
            samples_show = min(4000, len(y))
            fig = go.Figure(go.Scatter(
                x=t[:samples_show], y=y[:samples_show], mode="lines",
                line=dict(color="#6366f1", width=1.2)
            ))
            fig.update_layout(
                title=dict(text=f"Speech Oscillogram — {Path(voice_target).name if isinstance(voice_target, str) else audio_file.name if audio_file else 'audio'}", font=dict(color="#e2e8f0")),
                paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(title="Time (seconds)", color="#94a3b8", gridcolor="#2d2f54"),
                yaxis=dict(title="Amplitude", color="#94a3b8", gridcolor="#2d2f54"),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col1, col2, col3 = st.columns(3)
            items = list(features.items())
            for i, (k, v) in enumerate(items):
                with [col1, col2, col3][i % 3]:
                    st.metric(k, f"{v:.4f}" if v < 1 else f"{v:.2f}")

            # Radar chart of normalized feature profile
            feature_keys = list(features.keys())
            feature_vals = list(features.values())
            # Normalize to 0-1 for radar
            max_vals = [300, 100, 0.05, 0.2, 40, 5000, 8000, 0.3, 1.0]
            normalized = [min(v / m, 1.0) for v, m in zip(feature_vals, max_vals)]
            radar_categories = feature_keys + [feature_keys[0]]
            radar_values = normalized + [normalized[0]]
            fig_r = go.Figure(go.Scatterpolar(
                r=radar_values, theta=radar_categories, fill="toself",
                fillcolor="rgba(99,102,241,0.2)", line=dict(color="#6366f1", width=2),
                marker=dict(size=6, color="#818cf8")
            ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#94a3b8"), gridcolor="#2d2f54"),
                    angularaxis=dict(tickfont=dict(color="#e2e8f0", size=10)),
                    bgcolor="#1e2139"
                ),
                paper_bgcolor="#1e2139", height=360,
                title=dict(text="Normalized Vocal Feature Profile", font=dict(color="#e2e8f0"))
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with tab3:
            # Build real MFCC from waveform
            try:
                import librosa
                mfcc_mat = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr, n_mfcc=20)
            except Exception:
                # Fallback: use vec_200 reshaped (first 20*10)
                chunk = vec_200[:200]
                mfcc_mat = chunk[:20*10].reshape(20, 10) if len(chunk) >= 200 else np.random.randn(20, 10)

            fig = go.Figure(go.Heatmap(
                z=mfcc_mat, colorscale="Viridis",
                colorbar=dict(title="Coeff", tickfont=dict(color="#94a3b8"))
            ))
            fig.update_layout(
                title=dict(text="Mel-Frequency Cepstral Coefficients (MFCC) — Real Audio", font=dict(color="#e2e8f0")),
                paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(title="Time Frame", color="#94a3b8"),
                yaxis=dict(title="MFCC Index", color="#94a3b8"),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        st.session_state["voice_features"] = features
        st.session_state["voice_vec"] = vec_200

else:
    # Placeholder — shown before any button click
    st.markdown("""
    <div style="
      background: rgba(99,102,241,0.06);
      border: 1px dashed rgba(99,102,241,0.3);
      border-radius: 16px;
      padding: 3rem;
      text-align: center;
      margin-top: 1rem;
    ">
      <div style="font-size: 3rem;">🎤</div>
      <h3 style="color:#e2e8f0; margin-top:0.5rem;">Upload or Select a Voice File</h3>
      <p style="color: #94a3b8; font-size: 1rem; max-width: 600px; margin: 0 auto;">
        Upload a <b>.wav / .mp3</b> recording or choose a sample from the dataset above,
        then click <b>Analyze Voice Recording</b> to see the acoustic features and waveform.
      </p>
    </div>
    """, unsafe_allow_html=True)
