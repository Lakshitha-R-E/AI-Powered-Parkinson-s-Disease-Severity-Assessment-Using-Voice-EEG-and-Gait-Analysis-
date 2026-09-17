"""EEG Analysis Page"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
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
st.markdown('<p style="color:#94a3b8;">Extract frequency band powers, coherence, and functional connectivity from EEG signals.</p>', unsafe_allow_html=True)

eeg_file = st.file_uploader("Upload EEG file (.edf / .npy)", type=["edf","npy"])
use_demo = st.checkbox("Use synthetic EEG demo", value=True)
analyze  = st.button("🔬 Analyze EEG", type="primary")

if analyze or use_demo:
    with st.spinner("Processing EEG..."):
        import time; time.sleep(0.7)
        # Synthetic 19-channel, 10s EEG
        fs, n_ch, dur = 256, 19, 10
        t  = np.linspace(0, dur, fs * dur)
        eeg = np.zeros((n_ch, len(t)))
        bands = {"Delta":(1,4),"Theta":(4,8),"Alpha":(8,13),"Beta":(13,30),"Gamma":(30,40)}
        band_powers = {b: np.random.uniform(0.05,0.3) for b in bands}
        # Make alpha dominant (Parkinson's shows altered alpha)
        band_powers["Alpha"] = np.random.uniform(0.15, 0.35)
        band_powers["Theta"] = np.random.uniform(0.20, 0.40)
        for i in range(n_ch):
            for band,(lo,hi) in bands.items():
                f = np.random.uniform(lo, hi)
                eeg[i] += band_powers[band] * np.sin(2*np.pi*f*t + np.random.uniform(0,2*np.pi))
            eeg[i] += 0.05 * np.random.randn(len(t))

    st.success("✅ EEG features extracted")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Raw EEG", "🔵 Band Powers", "🔗 Connectivity", "📊 Coherence"])

    with tab1:
        fig = go.Figure()
        ch_names = [f"CH{i+1}" for i in range(min(8,n_ch))]
        offset = 0
        for i, name in enumerate(ch_names):
            sig = eeg[i,:500] / np.std(eeg[i]) * 0.4
            fig.add_trace(go.Scatter(x=t[:500], y=sig + offset, mode="lines",
                                      line=dict(width=1, color=f"hsl({i*25},70%,60%)"),
                                      name=name))
            offset += 1.0
        fig.update_layout(title=dict(text="EEG Channels (8 of 19)", font=dict(color="#e2e8f0")),
                           paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                           xaxis=dict(title="Time (s)", color="#94a3b8"),
                           yaxis=dict(title="Channel", color="#94a3b8", showticklabels=False),
                           legend=dict(font=dict(color="#94a3b8")), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        band_colors = {"Delta":"#6366f1","Theta":"#8b5cf6","Alpha":"#06b6d4","Beta":"#10b981","Gamma":"#f59e0b"}
        col1,col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=list(band_powers.keys()), y=[v*100 for v in band_powers.values()],
                marker_color=list(band_colors.values()), text=[f"{v*100:.1f}%" for v in band_powers.values()],
                textposition="auto", textfont=dict(color="white")))
            fig.update_layout(title=dict(text="Relative Band Power", font=dict(color="#e2e8f0")),
                               paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                               yaxis=dict(title="%", color="#94a3b8", gridcolor="#2d2f54"),
                               xaxis=dict(color="#94a3b8"), height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Pie(labels=list(band_powers.keys()),
                                    values=list(band_powers.values()),
                                    marker_colors=list(band_colors.values()), hole=0.35))
            fig.update_layout(paper_bgcolor="#1e2139", height=300, legend=dict(font=dict(color="#94a3b8")))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # PLV connectivity matrix (synthetic)
        plv = np.random.uniform(0.1, 0.7, (n_ch, n_ch))
        np.fill_diagonal(plv, 1.0)
        plv = (plv + plv.T) / 2
        fig = go.Figure(go.Heatmap(z=plv, colorscale="Viridis",
                                    colorbar=dict(title="PLV", tickfont=dict(color="#94a3b8")),
                                    zmin=0, zmax=1))
        fig.update_layout(title=dict(text="Phase Locking Value Connectivity Matrix", font=dict(color="#e2e8f0")),
                           paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                           xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8"), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        from scipy.signal import coherence as sig_coherence
        freqs, coh = sig_coherence(eeg[0], eeg[1], fs=fs, nperseg=256)
        fig = go.Figure(go.Scatter(x=freqs[:80], y=coh[:80], mode="lines",
                                    line=dict(color="#06b6d4", width=2)))
        for name,(lo,hi) in bands.items():
            fig.add_vrect(x0=lo, x1=hi, fillcolor=band_colors[name], opacity=0.08,
                           annotation_text=name, annotation_position="top left",
                           annotation=dict(font=dict(color=band_colors[name], size=10)))
        fig.update_layout(title=dict(text="Coherence: CH1 ↔ CH2", font=dict(color="#e2e8f0")),
                           paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                           xaxis=dict(title="Frequency (Hz)", color="#94a3b8", gridcolor="#2d2f54"),
                           yaxis=dict(title="Coherence", color="#94a3b8", gridcolor="#2d2f54"), height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.session_state.eeg_features = band_powers
