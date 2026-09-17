"""Explainable AI Page"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(page_title="Explainable AI", page_icon="🔍", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a,#111128);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#0f3460);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)
st.markdown('<h1 style="color:#e2e8f0;">🔍 Explainable AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Understand what drives the model predictions using SHAP, attention weights, and feature importance.</p>', unsafe_allow_html=True)

np.random.seed(7)

# ── SHAP Feature Importance ───────────────────────────────
st.markdown("### 📊 SHAP Feature Importance")
tab1, tab2, tab3, tab4 = st.tabs(["🎤 Voice", "🧠 EEG", "🚶 Gait", "🔗 Fusion Attention"])

def _shap_bar(names, values, color, title):
    idx = np.argsort(values)[::-1][:12]
    fig = go.Figure(go.Bar(
        x=values[idx], y=[names[i] for i in idx],
        orientation="h",
        marker=dict(color=values[idx], colorscale=color, showscale=True,
                    colorbar=dict(title="|SHAP|", tickfont=dict(color="#94a3b8"))),
    ))
    fig.update_layout(title=dict(text=title, font=dict(color="#e2e8f0")),
                       paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                       xaxis=dict(title="Mean |SHAP|", color="#94a3b8", gridcolor="#2d2f54"),
                       yaxis=dict(color="#94a3b8"),
                       height=400, margin=dict(l=200))
    return fig

with tab1:
    v_names = np.array(["Jitter Local","Shimmer Local","HNR Mean","Pitch Mean","Pitch Std",
                          "MFCC_00","MFCC_01","MFCC_02","Spectral Centroid","ZCR","RMS Energy",
                          "Formant F1","Formant F2","MFCC Delta 00","Voiced Fraction"])
    v_vals  = np.abs(np.random.exponential(0.12, len(v_names)))
    v_vals[0], v_vals[1], v_vals[2] = 0.42, 0.38, 0.31
    st.plotly_chart(_shap_bar(v_names, v_vals, "Reds", "Voice SHAP Feature Importance"), use_container_width=True)

with tab2:
    e_names = np.array(["Theta Power","Alpha Power","Delta Power","Beta Power","Gamma Power",
                          "Theta/Alpha Ratio","Spectral Entropy","PLV Alpha","PLV Beta",
                          "EEG Kurtosis","Coherence Delta","PLV Theta","Theta/Beta Ratio"])
    e_vals  = np.abs(np.random.exponential(0.1, len(e_names)))
    e_vals[0], e_vals[1], e_vals[2] = 0.38, 0.34, 0.29
    st.plotly_chart(_shap_bar(e_names, e_vals, "Blues", "EEG SHAP Feature Importance"), use_container_width=True)

with tab3:
    g_names = np.array(["Gait Asymmetry","Step Variability","Freeze Index","Cadence",
                          "Walking Speed","Stride Length","Jerk RMS","Swing Time",
                          "Double Support","Step Length","Stance Time","RMS Vertical"])
    g_vals  = np.abs(np.random.exponential(0.11, len(g_names)))
    g_vals[0], g_vals[1], g_vals[2] = 0.41, 0.35, 0.33
    st.plotly_chart(_shap_bar(g_names, g_vals, "Greens", "Gait SHAP Feature Importance"), use_container_width=True)

with tab4:
    # Fusion cross-modal attention matrix
    attn_mat = np.array([
        [0.72, 0.18, 0.10],
        [0.22, 0.65, 0.13],
        [0.15, 0.21, 0.64],
    ]) + 0.03 * np.random.randn(3,3)
    attn_mat = np.clip(attn_mat, 0, 1)
    labels = ["Voice", "EEG", "Gait"]
    fig = go.Figure(go.Heatmap(z=attn_mat, x=labels, y=labels,
                                colorscale="YlOrRd", zmin=0, zmax=1,
                                colorbar=dict(title="Attention", tickfont=dict(color="#94a3b8")),
                                text=[[f"{attn_mat[i,j]:.2f}" for j in range(3)] for i in range(3)],
                                texttemplate="%{text}", textfont=dict(size=16, color="white")))
    fig.update_layout(title=dict(text="Cross-Modal Fusion Attention Matrix", font=dict(color="#e2e8f0")),
                       paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                       xaxis=dict(title="Key Modality", color="#94a3b8", tickfont=dict(size=13, color="#e2e8f0")),
                       yaxis=dict(title="Query Modality", color="#94a3b8", tickfont=dict(size=13, color="#e2e8f0")),
                       height=360)
    st.plotly_chart(fig, use_container_width=True)

# ── SHAP Waterfall (sample) ───────────────────────────────
st.markdown("### 💧 Sample Prediction Explanation (Waterfall)")
col1, col2 = st.columns(2)
with col1:
    feats = ["Jitter Local↑","Shimmer↑","HNR↓","Theta Power↑","Gait Asymmetry↑",
              "MFCC_00↓","Freeze Index↑","Cadence↓","Step Var↑","Alpha/Beta↑"]
    shap_v = np.array([+0.38,+0.31,-0.25,+0.28,+0.41,-0.18,+0.33,-0.22,+0.27,+0.19])
    colors = ["#ef4444" if v > 0 else "#10b981" for v in shap_v]
    fig = go.Figure(go.Bar(x=shap_v, y=feats, orientation="h",
                            marker_color=colors))
    base = 0.45
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig.add_vline(x=base, line_color="#818cf8", line_width=1.5, line_dash="dot",
                   annotation_text=f"Base={base}", annotation_font=dict(color="#818cf8"))
    fig.update_layout(title=dict(text="SHAP Waterfall — Severe Prediction", font=dict(color="#e2e8f0")),
                       paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                       xaxis=dict(title="SHAP Value", color="#94a3b8", gridcolor="#2d2f54"),
                       yaxis=dict(color="#94a3b8"), height=400, margin=dict(l=160))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Grad-CAM simulation
    n_feats = 40
    cam = np.zeros(n_feats)
    cam[2:5] = [0.9, 0.8, 0.75]  # Jitter/Shimmer region
    cam[10:14] = [0.7, 0.85, 0.6, 0.55]
    cam += 0.1 * np.random.rand(n_feats)
    cam /= cam.max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n_feats), y=cam, fill="tozeroy",
                              fillcolor="rgba(239,68,68,0.3)",
                              line=dict(color="#ef4444", width=2), name="Grad-CAM"))
    fig.update_layout(title=dict(text="Grad-CAM — Voice Encoder Activations (Severe)", font=dict(color="#e2e8f0")),
                       paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                       xaxis=dict(title="Feature Index", color="#94a3b8", gridcolor="#2d2f54"),
                       yaxis=dict(title="Activation", color="#94a3b8", gridcolor="#2d2f54"),
                       height=400)
    st.plotly_chart(fig, use_container_width=True)
