"""
============================================================
Page 3: EEG Signal Analysis
AI-Powered Parkinson's Disease Severity Assessment
============================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.processing.unified_loader import get_available_sample_files, load_and_process_eeg

st.set_page_config(page_title="EEG Analysis", page_icon="🧠", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a,#111128);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#0f3460);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 style="color:#e2e8f0;">🧠 EEG Signal Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Extract frequency band powers, spectral density, coherence, and cortical dynamics from your EEG signal.</p>', unsafe_allow_html=True)

samples = get_available_sample_files()

tab_upload, tab_sample = st.tabs(["📤 Upload EEG Signal", "📁 Select from PRED+CT EEG Dataset"])

eeg_target = None
with tab_upload:
    eeg_file = st.file_uploader("Upload EEG file (.set, .edf, .npy, .csv, .tsv, .txt)", type=["set", "edf", "npy", "csv", "tsv", "txt"])
    if eeg_file:
        eeg_target = eeg_file
        st.success(f"✅ {eeg_file.name} ({eeg_file.size//1024} KB)")

with tab_sample:
    e_choices = ["-- Select a sample --"] + list(samples["eeg"].keys())
    e_sel = st.selectbox("Choose Sample from Dataset", e_choices, index=0)
    if e_sel != "-- Select a sample --":
        eeg_target = samples["eeg"][e_sel]
        st.caption(f"📁 Selected: `{Path(eeg_target).name}`")

st.markdown("---")
analyze = st.button("🔬 Analyze EEG Signal", type="primary", use_container_width=False)

if analyze:
    if eeg_target is None:
        st.warning("⚠️ Please upload an EEG file or select a sample from the dataset first.")
    else:
        with st.spinner("Processing EEG signals & calculating spectral bands..."):
            vec_150, feats, ch_data, fs = load_and_process_eeg(eeg_target)
            n_ch = ch_data.shape[0]
            dur = ch_data.shape[1] / fs
            t = np.linspace(0, dur, ch_data.shape[1])
            eeg = ch_data

            bands = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 40)}

            # Features dict has per-channel keys like delta_ch00, theta_ch01 etc.
            # Aggregate by averaging across all channels for each band
            band_powers = {}
            for b in bands:
                key_prefix = b.lower() + "_ch"
                ch_vals = [v for k, v in feats.items()
                           if k.startswith(key_prefix) and isinstance(v, (int, float, np.floating))]
                if ch_vals:
                    band_powers[b] = float(np.mean(ch_vals))
                else:
                    band_powers[b] = float(feats.get(b.lower() + "_power", 0.0))

            # If still all zero, compute directly from EEG signal via Welch PSD
            if sum(band_powers.values()) < 1e-6:
                from scipy.signal import welch
                for b, (lo, hi) in bands.items():
                    freqs_w, psd = welch(eeg[0], fs=fs, nperseg=min(256, eeg.shape[1]//2))
                    idx = np.where((freqs_w >= lo) & (freqs_w < hi))[0]
                    band_powers[b] = float(np.mean(psd[idx])) if len(idx) > 0 else 0.01

        fname = Path(eeg_target).name if isinstance(eeg_target, str) else eeg_file.name if eeg_file else "eeg"
        st.success(f"✅ EEG signals processed: {n_ch} channels @ {fs:.0f} Hz, duration {dur:.1f}s — {fname}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Raw EEG Channels",
            "🔵 Frequency Band Powers",
            "🔗 Functional Connectivity (PLV)",
            "📊 Coherence & Spectrum"
        ])

        with tab1:
            fig = go.Figure()
            n_show = min(8, n_ch)
            offset = 0
            samples_show = min(1000, eeg.shape[1])
            colors = ["#6366f1","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899","#14b8a6"]
            for i in range(n_show):
                sig_slice = eeg[i, :samples_show]
                denom = np.std(sig_slice) if np.std(sig_slice) > 1e-6 else 1.0
                sig = (sig_slice - np.mean(sig_slice)) / denom * 0.4
                fig.add_trace(go.Scatter(
                    x=t[:samples_show], y=sig + offset, mode="lines",
                    line=dict(width=1.2, color=colors[i % len(colors)]),
                    name=f"CH{i+1}"
                ))
                offset += 1.2
            fig.update_layout(
                title=dict(text=f"EEG Multichannel Waveforms — {fname} (showing {n_show}/{n_ch} channels)", font=dict(color="#e2e8f0")),
                paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(title="Time (seconds)", color="#94a3b8", gridcolor="#2d2f54"),
                yaxis=dict(title="Channels (offset)", color="#94a3b8", showticklabels=False),
                legend=dict(font=dict(color="#94a3b8")), height=420
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            band_colors = {"Delta":"#6366f1","Theta":"#8b5cf6","Alpha":"#06b6d4","Beta":"#10b981","Gamma":"#f59e0b"}
            total_p = sum(band_powers.values()) if sum(band_powers.values()) > 0 else 1.0
            rel_powers = {k: (v / total_p) * 100 for k, v in band_powers.items()}

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Bar(
                    x=list(rel_powers.keys()), y=list(rel_powers.values()),
                    marker_color=list(band_colors.values()),
                    text=[f"{v:.1f}%" for v in rel_powers.values()],
                    textposition="auto", textfont=dict(color="white")
                ))
                fig.update_layout(
                    title=dict(text="Relative Frequency Band Power Distribution", font=dict(color="#e2e8f0")),
                    paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                    yaxis=dict(title="Relative Power (%)", color="#94a3b8", gridcolor="#2d2f54"),
                    xaxis=dict(color="#94a3b8"), height=320
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure(go.Pie(
                    labels=list(rel_powers.keys()), values=list(rel_powers.values()),
                    marker_colors=list(band_colors.values()), hole=0.35
                ))
                fig.update_layout(
                    paper_bgcolor="#1e2139", height=320,
                    legend=dict(font=dict(color="#94a3b8")),
                    title=dict(text="Band Power Proportion", font=dict(color="#e2e8f0"))
                )
                st.plotly_chart(fig, use_container_width=True)

            # Show raw band power values
            st.markdown("#### Absolute Band Power Values")
            bp_col1, bp_col2, bp_col3, bp_col4, bp_col5 = st.columns(5)
            for col, (band, pwr) in zip([bp_col1, bp_col2, bp_col3, bp_col4, bp_col5], band_powers.items()):
                with col:
                    st.metric(band, f"{pwr:.4f}")

        with tab3:
            # Compute real PLV from actual EEG channels
            from scipy.signal import hilbert, butter, filtfilt
            plv_dim = min(16, n_ch)
            plv_mat = np.zeros((plv_dim, plv_dim))
            # Alpha band PLV
            b_filt, a_filt = butter(4, [8/(fs/2), 13/(fs/2)], btype="band")
            alpha_signals = []
            for i in range(plv_dim):
                try:
                    filt = filtfilt(b_filt, a_filt, eeg[i])
                    phase = np.angle(hilbert(filt))
                    alpha_signals.append(phase)
                except Exception:
                    alpha_signals.append(np.zeros(eeg.shape[1]))
            for i in range(plv_dim):
                for j in range(plv_dim):
                    if i == j:
                        plv_mat[i, j] = 1.0
                    else:
                        phase_diff = alpha_signals[i] - alpha_signals[j]
                        plv_mat[i, j] = abs(np.mean(np.exp(1j * phase_diff)))

            fig = go.Figure(go.Heatmap(
                z=plv_mat, colorscale="Viridis",
                colorbar=dict(title="PLV", tickfont=dict(color="#94a3b8")),
                zmin=0, zmax=1
            ))
            fig.update_layout(
                title=dict(text=f"Alpha-Band Phase Locking Value (PLV) — {plv_dim}x{plv_dim} channels", font=dict(color="#e2e8f0")),
                paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                xaxis=dict(title="Channel", color="#94a3b8"),
                yaxis=dict(title="Channel", color="#94a3b8"),
                height=420
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            from scipy.signal import coherence as sig_coherence, welch
            col_l, col_r = st.columns(2)
            with col_l:
                if eeg.shape[0] >= 2:
                    freqs_c, coh = sig_coherence(eeg[0], eeg[1], fs=fs, nperseg=min(256, eeg.shape[1]//2))
                    mask = freqs_c <= 50
                    fig = go.Figure(go.Scatter(
                        x=freqs_c[mask], y=coh[mask], mode="lines",
                        line=dict(color="#06b6d4", width=2)
                    ))
                    for name, (lo, hi) in bands.items():
                        bc = {"Delta":"#6366f1","Theta":"#8b5cf6","Alpha":"#06b6d4","Beta":"#10b981","Gamma":"#f59e0b"}[name]
                        fig.add_vrect(x0=lo, x1=hi, fillcolor=bc, opacity=0.08,
                                      annotation_text=name, annotation_position="top left",
                                      annotation=dict(font=dict(color=bc, size=10)))
                    fig.update_layout(
                        title=dict(text="Coherence CH1 ↔ CH2", font=dict(color="#e2e8f0")),
                        paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                        xaxis=dict(title="Frequency (Hz)", color="#94a3b8", gridcolor="#2d2f54"),
                        yaxis=dict(title="Coherence (0-1)", color="#94a3b8", gridcolor="#2d2f54"),
                        height=360
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col_r:
                freqs_w, psd = welch(eeg[0], fs=fs, nperseg=min(256, eeg.shape[1]//2))
                mask = freqs_w <= 50
                fig = go.Figure(go.Scatter(
                    x=freqs_w[mask], y=10*np.log10(psd[mask]+1e-12), mode="lines",
                    line=dict(color="#8b5cf6", width=2), fill="tozeroy",
                    fillcolor="rgba(139,92,246,0.15)"
                ))
                fig.update_layout(
                    title=dict(text="Power Spectral Density — CH1 (dB)", font=dict(color="#e2e8f0")),
                    paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                    xaxis=dict(title="Frequency (Hz)", color="#94a3b8", gridcolor="#2d2f54"),
                    yaxis=dict(title="PSD (dB/Hz)", color="#94a3b8", gridcolor="#2d2f54"),
                    height=360
                )
                st.plotly_chart(fig, use_container_width=True)

        st.session_state["eeg_features"] = band_powers
        st.session_state["eeg_vec"] = vec_150

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
      <div style="font-size: 3rem;">🧠</div>
      <h3 style="color:#e2e8f0; margin-top:0.5rem;">Upload or Select an EEG File</h3>
      <p style="color: #94a3b8; font-size: 1rem; max-width: 600px; margin: 0 auto;">
        Upload a <b>.set / .edf / .npy / .csv / .txt</b> file or choose a sample from the PRED+CT dataset above,
        then click <b>Analyze EEG Signal</b> to visualize brain rhythms and band powers.
      </p>
    </div>
    """, unsafe_allow_html=True)
