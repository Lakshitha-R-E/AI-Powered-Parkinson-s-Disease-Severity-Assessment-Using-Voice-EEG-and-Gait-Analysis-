"""Report Generation Page"""
import io
import numpy as np
import streamlit as st
from datetime import datetime
st.set_page_config(page_title="Report Generation", page_icon="📄", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a,#111128);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#0f3460);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)
st.markdown('<h1 style="color:#e2e8f0;">📄 Clinical Report Generation</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Generate a comprehensive PDF clinical report with all analysis results and recommendations.</p>', unsafe_allow_html=True)

# Patient info
st.markdown("### 👤 Report Details")
col1,col2,col3 = st.columns(3)
with col1:
    patient_name   = st.text_input("Patient Name",    value="John Doe")
    patient_id_str = st.text_input("Patient ID",      value="PAT-2024-001")
    age            = st.number_input("Age", 0, 120,   value=67)
with col2:
    gender     = st.selectbox("Gender", ["Male","Female","Other"])
    clinician  = st.text_input("Clinician", value="Dr. Smith")
    report_type= st.selectbox("Report Type", ["Full Report","Summary","Research"])
with col3:
    include_xai    = st.checkbox("Include XAI (SHAP/Attention)", value=True)
    include_recs   = st.checkbox("Include Recommendations",       value=True)
    notes          = st.text_area("Clinician Notes", height=80)

st.markdown("---")

# Prediction summary (from session or demo)
pred = st.session_state.get("prediction_result") or {
    "updrs_score": 47.3,
    "severity_label": "Moderate",
    "confidence": 0.81,
    "class_probabilities": [0.09, 0.81, 0.10],
}

severity = pred.get("severity_label","Moderate")
updrs    = pred.get("updrs_score", 47.3)
conf     = pred.get("confidence", 0.81)
probs    = pred.get("class_probabilities",[0.09,0.81,0.10])

sev_colors = {"Mild":"#10b981","Moderate":"#f59e0b","Severe":"#ef4444"}
sc = sev_colors.get(severity,"#6366f1")

st.markdown(f"""
<div style="background:linear-gradient(135deg,{sc}22,{sc}11);border:2px solid {sc};
     border-radius:14px;padding:1.2rem 1.5rem;display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-size:0.85rem;color:#94a3b8;">Prediction to Report</div>
    <div style="font-size:1.8rem;font-weight:800;color:{sc};">{severity}</div>
  </div>
  <div style="text-align:right;">
    <div style="color:#e2e8f0;font-size:1.2rem;font-weight:700;">UPDRS: {updrs:.1f}/108</div>
    <div style="color:#94a3b8;font-size:0.85rem;">Confidence: {conf*100:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    gen_report = st.button("📄 Generate PDF Report", type="primary", use_container_width=True)
with col_btn2:
    preview    = st.button("👁️ Preview Report Sections", use_container_width=True)

if gen_report:
    with st.spinner("Generating clinical PDF report..."):
        import time; time.sleep(1.2)
        try:
            import sys; sys.path.insert(0,".")
            from src.reports.report_generator import ClinicalReportGenerator
            patient_info = {
                "name": patient_name, "patient_id": patient_id_str,
                "age": str(age), "gender": gender, "clinician": clinician,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "report_id": f"RPT-{datetime.now().strftime('%Y%m%d-%H%M')}",
            }
            predictions = {
                "severity_label": severity, "updrs_score": updrs,
                "confidence": conf, "class_probabilities": probs,
            }
            analysis_results = {
                "voice": {k:round(v,4) for k,v in
                          (st.session_state.get("voice_features") or
                           {"pitch_mean":118.4,"jitter_local":0.0082,"shimmer_local":0.0531,"hnr_mean":15.2}).items()},
                "eeg": {k:round(float(v),4) for k,v in
                        (st.session_state.get("eeg_features") or
                         {"delta_power":0.21,"theta_power":0.28,"alpha_power":0.25}).items()},
                "gait": {k:round(float(v),4) for k,v in
                         (st.session_state.get("gait_features") or
                          {"cadence":108.2,"walking_speed":0.94,"gait_asymmetry":8.7}).items()},
            }
            gen = ClinicalReportGenerator()
            pdf_bytes = gen.generate_report_bytes(
                patient_info=patient_info,
                predictions=predictions,
                analysis_results=analysis_results,
            )
            st.success("✅ Report generated successfully!")
            st.download_button(
                "⬇️ Download PDF Report", data=pdf_bytes,
                file_name=f"parkinson_report_{patient_id_str}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf", use_container_width=True,
            )
        except Exception as e:
            st.error(f"Report generation error: {e}")
            st.info("💡 Install all dependencies: `pip install reportlab Pillow`")

if preview:
    st.markdown("### 📋 Report Sections Preview")
    sections = {
        "👤 Patient Information": f"Name: {patient_name} | Age: {age} | Gender: {gender} | Clinician: {clinician}",
        "📊 Assessment Results":  f"UPDRS: {updrs:.1f}/108 | Severity: **{severity}** | Confidence: {conf*100:.1f}%",
        "🎤 Voice Analysis":      "MFCC, Pitch, Jitter, Shimmer, HNR, Spectral features",
        "🧠 EEG Analysis":        "Delta/Theta/Alpha/Beta/Gamma powers, PLV connectivity, Spectral entropy",
        "🚶 Gait Analysis":       "Step length, Cadence, Asymmetry, Variability, Freeze index",
        "🔍 XAI Explanations":    "SHAP waterfall, Fusion attention matrix, Grad-CAM activations" if include_xai else "Not included",
        "💊 Recommendations":     f"Based on {severity} severity — see full report for details" if include_recs else "Not included",
        "⚠️ Disclaimer":          "AI-generated report — not a substitute for clinical diagnosis",
    }
    for sec, content in sections.items():
        with st.expander(sec):
            st.markdown(content)
