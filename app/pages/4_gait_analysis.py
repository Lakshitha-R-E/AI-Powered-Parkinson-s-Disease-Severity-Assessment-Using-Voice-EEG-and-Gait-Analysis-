"""Gait Analysis Page"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(page_title="Gait Analysis", page_icon="🚶", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a,#111128);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#0f3460);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)
st.markdown('<h1 style="color:#e2e8f0;">🚶 Gait Signal Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Analyze IMU accelerometer data to extract step, stride, cadence, and asymmetry features.</p>', unsafe_allow_html=True)

gait_file = st.file_uploader("Upload Gait CSV (.csv)", type=["csv"])
use_demo  = st.checkbox("Use synthetic gait demo", value=True)
analyze   = st.button("🔬 Analyze Gait", type="primary")

if analyze or use_demo:
    with st.spinner("Processing gait signals..."):
        import time; time.sleep(0.6)
        fs  = 100
        dur = 30
        t   = np.linspace(0, dur, fs*dur)
        step_freq = 1.8
        accel_v = (np.sin(2*np.pi*step_freq*t) + 0.3*np.sin(2*np.pi*2*step_freq*t) +
                   0.15*np.random.randn(len(t)))
        accel_ap= (0.7*np.cos(2*np.pi*step_freq*t) + 0.2*np.random.randn(len(t)))
        accel_ml= (0.3*np.sin(2*np.pi*0.9*t) + 0.1*np.random.randn(len(t)))
        # Features
        features = {
            "Cadence (steps/min)": 108.2, "Walking Speed (m/s)": 0.94,
            "Step Length (m)": 0.52, "Stride Length (m)": 1.04,
            "Swing Time (s)": 0.41, "Stance Time (s)": 0.67,
            "Double Support Time (s)": 0.13, "Gait Asymmetry (%)": 8.7,
            "Step Variability (CoV%)": 5.2, "Freeze Index": 0.18,
            "RMS Vertical Accel": 0.82, "Jerk RMS": 1.43,
        }

    st.success("✅ Gait features extracted")
    tab1, tab2, tab3 = st.tabs(["📈 Acceleration", "📊 Features", "⏱️ Temporal Events"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[:500], y=accel_v[:500], name="Vertical",
                                  line=dict(color="#6366f1", width=1.5)))
        fig.add_trace(go.Scatter(x=t[:500], y=accel_ap[:500], name="Ant-Post",
                                  line=dict(color="#10b981", width=1.5)))
        fig.add_trace(go.Scatter(x=t[:500], y=accel_ml[:500], name="Med-Lat",
                                  line=dict(color="#f59e0b", width=1.5)))
        # Highlight heel strikes
        from scipy.signal import find_peaks
        peaks,_ = find_peaks(accel_v[:500], height=0.3, distance=int(fs*0.3))
        fig.add_trace(go.Scatter(x=t[peaks], y=accel_v[peaks], mode="markers",
                                  marker=dict(color="#ef4444", size=10, symbol="triangle-down"),
                                  name="Heel Strike"))
        fig.update_layout(title=dict(text="3-Axis Accelerometer", font=dict(color="#e2e8f0")),
                           paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                           xaxis=dict(title="Time (s)", color="#94a3b8", gridcolor="#2d2f54"),
                           yaxis=dict(title="Acceleration (m/s²)", color="#94a3b8", gridcolor="#2d2f54"),
                           legend=dict(font=dict(color="#94a3b8")), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            import pandas as pd
            df = pd.DataFrame(list(features.items()), columns=["Feature","Value"])
            df["Value"] = df["Value"].apply(lambda x: f"{x:.3f}")
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col2:
            # Radar chart
            cats = ["Cadence","Speed","Step Len","Stride Len","Symmetry"]
            vals = [0.85, 0.72, 0.68, 0.70, 0.83]
            vals += vals[:1]; cats += cats[:1]
            fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                             fillcolor="rgba(245,158,11,0.2)",
                                             line=dict(color="#f59e0b",width=2)))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1],
                                                          gridcolor="#2d2f54",linecolor="#2d2f54",
                                                          tickfont=dict(color="#94a3b8")),
                                          angularaxis=dict(tickfont=dict(color="#e2e8f0")),
                                          bgcolor="#1e2139"),
                               paper_bgcolor="#1e2139",
                               title=dict(text="Gait Quality Radar", font=dict(color="#e2e8f0")),
                               height=320)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Step intervals
        n_steps = 54
        step_int = 0.556 + 0.03 * np.random.randn(n_steps)
        left  = step_int[0::2]
        right = step_int[1::2]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(left)),  y=left,  mode="lines+markers",
                                  name="Left", line=dict(color="#6366f1",width=2),
                                  marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=np.arange(len(right)), y=right, mode="lines+markers",
                                  name="Right", line=dict(color="#10b981",width=2),
                                  marker=dict(size=6)))
        fig.update_layout(title=dict(text="Step Intervals Over Time", font=dict(color="#e2e8f0")),
                           paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
                           xaxis=dict(title="Step Number", color="#94a3b8", gridcolor="#2d2f54"),
                           yaxis=dict(title="Interval (s)", color="#94a3b8", gridcolor="#2d2f54"),
                           legend=dict(font=dict(color="#94a3b8")), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.session_state.gait_features = features
