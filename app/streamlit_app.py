"""
============================================================
Streamlit Main Application
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Multi-page Streamlit app with:
  1. Dashboard
  2. Voice Analysis
  3. EEG Analysis
  4. Gait Analysis
  5. Severity Prediction
  6. Explainable AI
  7. Report Generation
============================================================
"""

import streamlit as st

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's AI Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/parkinsons-ai",
        "Report a bug":"mailto:support@parkinsons-ai.local",
        "About":       "## AI-Powered Parkinson's Severity Assessment\nFinal Year Research Project",
    },
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Import Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Hide default Streamlit menu */
  #MainMenu, footer { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
  }
  [data-testid="stSidebar"] * { color: white !important; }
  [data-testid="stSidebar"] .stRadio > label { color: white !important; }

  /* Main background */
  .stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #111128 100%);
  }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #1e2139, #252847);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.3);
  }

  /* Severity badges */
  .badge-mild {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
  }
  .badge-moderate {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
  }
  .badge-severe {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
  }

  /* Section headers */
  .section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #6366f1;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Upload zones */
  [data-testid="stFileUploader"] {
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background: rgba(99, 102, 241, 0.05) !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
  }

  /* Metrics */
  [data-testid="stMetric"] {
    background: rgba(99, 102, 241, 0.08);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(99, 102, 241, 0.15);
  }
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.85rem !important; }
  [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(30, 33, 57, 0.6);
    border-radius: 10px;
    padding: 0.3rem;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
  }

  /* Progress bars */
  .stProgress > div > div > div { background: linear-gradient(90deg, #6366f1, #818cf8) !important; }

  /* DataFrames */
  .dataframe { background: #1e2139 !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "voice_features"    not in st.session_state:
    st.session_state.voice_features    = None
if "eeg_features"      not in st.session_state:
    st.session_state.eeg_features      = None
if "gait_features"     not in st.session_state:
    st.session_state.gait_features     = None
if "patient_info"      not in st.session_state:
    st.session_state.patient_info      = {}


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
      <div style="font-size: 3rem;">🧠</div>
      <div style="font-size: 1.1rem; font-weight: 700; color: #818cf8;">Parkinson's AI</div>
      <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">
        Multimodal Severity Assessment
      </div>
    </div>
    <hr style="border-color: rgba(99,102,241,0.2); margin: 0.75rem 0;">
    """, unsafe_allow_html=True)

    # Patient quick info
    if st.session_state.patient_info:
        pi = st.session_state.patient_info
        st.markdown(f"""
        <div style="background: rgba(99,102,241,0.1); border-radius: 10px; padding: 0.75rem; margin-bottom: 1rem;">
          <div style="font-size: 0.8rem; color: #64748b;">Current Patient</div>
          <div style="font-weight: 600;">{pi.get('name', 'Unknown')}</div>
          <div style="font-size: 0.8rem; color: #94a3b8;">Age: {pi.get('age', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Status indicators
    st.markdown("**📊 Analysis Status**")
    col1, col2, col3 = st.columns(3)
    with col1:
        icon = "✅" if st.session_state.voice_features else "⬜"
        st.markdown(f"<div style='text-align:center; font-size:0.75rem;'>{icon}<br>Voice</div>",
                    unsafe_allow_html=True)
    with col2:
        icon = "✅" if st.session_state.eeg_features else "⬜"
        st.markdown(f"<div style='text-align:center; font-size:0.75rem;'>{icon}<br>EEG</div>",
                    unsafe_allow_html=True)
    with col3:
        icon = "✅" if st.session_state.gait_features else "⬜"
        st.markdown(f"<div style='text-align:center; font-size:0.75rem;'>{icon}<br>Gait</div>",
                    unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(99,102,241,0.2); margin: 0.75rem 0;'>",
                unsafe_allow_html=True)

    # Navigation hint
    st.markdown("""
    <div style="font-size:0.8rem; color:#64748b;">
    📌 <b>Navigation</b><br>
    Use the pages in the sidebar above to access each module.
    </div>
    """, unsafe_allow_html=True)


# ── Main Landing Page ──────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
  border: 1px solid rgba(99,102,241,0.2);
  border-radius: 20px;
  padding: 2.5rem;
  text-align: center;
  margin-bottom: 2rem;
">
  <div style="font-size: 4rem; margin-bottom: 0.5rem;">🧠</div>
  <h1 style="color: #e2e8f0; font-size: 2.2rem; font-weight: 800; margin: 0;">
    AI-Powered Parkinson's Disease<br>Severity Assessment
  </h1>
  <p style="color: #94a3b8; font-size: 1rem; margin-top: 0.75rem; max-width: 600px; margin-left: auto; margin-right: auto;">
    Multimodal deep learning system combining Voice, EEG, and Gait analysis
    for accurate UPDRS prediction and clinical severity classification.
  </p>
  <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
    <span style="background: rgba(16,185,129,0.2); color: #34d399; padding: 0.35rem 1rem; border-radius: 50px; font-size: 0.85rem; border: 1px solid rgba(16,185,129,0.3);">🎤 Voice Analysis</span>
    <span style="background: rgba(99,102,241,0.2); color: #818cf8; padding: 0.35rem 1rem; border-radius: 50px; font-size: 0.85rem; border: 1px solid rgba(99,102,241,0.3);">🧠 EEG Processing</span>
    <span style="background: rgba(245,158,11,0.2); color: #fbbf24; padding: 0.35rem 1rem; border-radius: 50px; font-size: 0.85rem; border: 1px solid rgba(245,158,11,0.3);">🚶 Gait Analysis</span>
    <span style="background: rgba(239,68,68,0.2); color: #f87171; padding: 0.35rem 1rem; border-radius: 50px; font-size: 0.85rem; border: 1px solid rgba(239,68,68,0.3);">📊 XAI Insights</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Quick stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Architecture", "Transformer Fusion", "CNN + BiLSTM")
with col2:
    st.metric("Modalities", "3 Signals", "Voice + EEG + Gait")
with col3:
    st.metric("UPDRS Scale", "0 – 108", "Regression output")
with col4:
    st.metric("Classes", "3 Severity", "Mild / Moderate / Severe")

st.markdown("<br>", unsafe_allow_html=True)

# Feature cards
st.markdown('<div class="section-title">🚀 System Capabilities</div>', unsafe_allow_html=True)

cols = st.columns(3)
cards = [
    ("🎤", "Voice Signal Processing",
     "MFCC • Jitter • Shimmer • HNR • Spectral features\nNoise reduction • VAD • Praat analysis"),
    ("🧠", "EEG Analysis",
     "Band powers (δ θ α β γ) • Spectral entropy\nPLV connectivity • ICA artifact removal"),
    ("🚶", "Gait Analysis",
     "Step length • Cadence • Asymmetry • Variability\nFreeze-of-Gait index • Jerk smoothness"),
    ("🔗", "Transformer Fusion",
     "Multi-head self-attention • Cross-modal attention\nModality dropout • Residual connections"),
    ("🔍", "Explainable AI",
     "SHAP feature importance • Grad-CAM activation\nAttention heatmaps • Modality contribution"),
    ("📄", "Clinical Reports",
     "PDF generation • Patient records • UPDRS trend\nTreatment recommendations • XAI plots"),
]

for i, (icon, title, desc) in enumerate(cards):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
          <div style="font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem;">{title}</div>
          <div style="font-size: 0.82rem; color: #94a3b8; white-space: pre-line;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.8rem; margin-top: 2rem; padding: 1rem;
     border-top: 1px solid rgba(99,102,241,0.15);">
  ⚠️ This system is for research purposes only. Not for clinical diagnosis without physician oversight.<br>
  Final Year Research Project — AI-Powered Parkinson's Disease Assessment
</div>
""", unsafe_allow_html=True)
