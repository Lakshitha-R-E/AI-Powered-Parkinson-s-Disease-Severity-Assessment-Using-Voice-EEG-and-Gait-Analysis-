"""
============================================================
Page 1: Dashboard
AI-Powered Parkinson's Disease Severity Assessment
============================================================
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0a1a 0%,#111128 100%);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#0f3460 100%);}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 style="color:#e2e8f0;">📊 Clinical Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;">Population-level statistics and system overview.</p>', unsafe_allow_html=True)

# ── Synthetic population stats ────────────────────────────
np.random.seed(42)
n_patients = 150
ages       = np.random.normal(68, 10, n_patients).clip(40, 90)
updrs_scores = np.clip(np.random.normal(45, 18, n_patients), 0, 108)
severity   = np.where(updrs_scores < 36, "Mild", np.where(updrs_scores < 72, "Moderate", "Severe"))
sex        = np.random.choice(["Male", "Female"], n_patients, p=[0.6, 0.4])

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Patients", n_patients)
with col2: st.metric("Mild Cases",     int(np.sum(severity == "Mild")))
with col3: st.metric("Moderate Cases", int(np.sum(severity == "Moderate")))
with col4: st.metric("Severe Cases",   int(np.sum(severity == "Severe")))
with col5: st.metric("Mean UPDRS",     f"{np.mean(updrs_scores):.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row 1 ──────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    # Severity pie
    counts = {s: int(np.sum(severity == s)) for s in ["Mild", "Moderate", "Severe"]}
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()), values=list(counts.values()),
        marker_colors=["#10b981", "#f59e0b", "#ef4444"],
        textinfo="label+percent", hole=0.4,
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        title=dict(text="Severity Distribution", font=dict(color="#e2e8f0")),
        paper_bgcolor="#1e2139", height=320, margin=dict(t=50, b=10),
        legend=dict(font=dict(color="#94a3b8")),
        annotations=[dict(text="Patients", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=13, color="#94a3b8"))],
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # UPDRS histogram
    fig = go.Figure(go.Histogram(
        x=updrs_scores, nbinsx=20,
        marker_color="#6366f1", opacity=0.85,
        name="UPDRS",
    ))
    fig.update_layout(
        title=dict(text="UPDRS Distribution", font=dict(color="#e2e8f0")),
        paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
        xaxis=dict(title="UPDRS Score", color="#94a3b8", gridcolor="#2d2f54"),
        yaxis=dict(title="Count", color="#94a3b8", gridcolor="#2d2f54"),
        height=320, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    # Age distribution by severity
    fig = go.Figure()
    for sev, color in [("Mild","#10b981"), ("Moderate","#f59e0b"), ("Severe","#ef4444")]:
        mask = severity == sev
        fig.add_trace(go.Box(y=ages[mask], name=sev, marker_color=color, boxmean=True))
    fig.update_layout(
        title=dict(text="Age by Severity", font=dict(color="#e2e8f0")),
        paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(title="Age", color="#94a3b8", gridcolor="#2d2f54"),
        height=320, margin=dict(t=50, b=40),
        legend=dict(font=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    # UPDRS vs Age scatter
    color_map = {"Mild": "#10b981", "Moderate": "#f59e0b", "Severe": "#ef4444"}
    colors_scatter = [color_map[s] for s in severity]
    fig = go.Figure(go.Scatter(
        x=ages, y=updrs_scores, mode="markers",
        marker=dict(color=colors_scatter, size=8, opacity=0.7,
                    line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
        text=[f"Severity: {s}" for s in severity],
        hovertemplate="Age: %{x:.0f}<br>UPDRS: %{y:.1f}<br>%{text}<extra></extra>",
    ))
    # Trendline
    z = np.polyfit(ages, updrs_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(40, 90, 100)
    fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines",
                              line=dict(color="#818cf8", dash="dash", width=2),
                              name="Trend"))
    fig.update_layout(
        title=dict(text="UPDRS Score vs Age", font=dict(color="#e2e8f0")),
        paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
        xaxis=dict(title="Age", color="#94a3b8", gridcolor="#2d2f54"),
        yaxis=dict(title="UPDRS Score", color="#94a3b8", gridcolor="#2d2f54"),
        height=350, margin=dict(t=50),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Prediction trend (simulated over 12 months)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    counts_monthly = np.random.randint(8, 20, 12)
    mild_m     = (counts_monthly * 0.4).astype(int)
    moderate_m = (counts_monthly * 0.45).astype(int)
    severe_m   = counts_monthly - mild_m - moderate_m

    fig = go.Figure()
    for data, name, color in [
        (mild_m,     "Mild",     "#10b981"),
        (moderate_m, "Moderate", "#f59e0b"),
        (severe_m,   "Severe",   "#ef4444"),
    ]:
        fig.add_trace(go.Bar(name=name, x=months, y=data, marker_color=color))

    fig.update_layout(
        title=dict(text="Monthly Assessments by Severity", font=dict(color="#e2e8f0")),
        paper_bgcolor="#1e2139", plot_bgcolor="#1e2139",
        barmode="stack",
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(title="Count", color="#94a3b8", gridcolor="#2d2f54"),
        legend=dict(font=dict(color="#94a3b8")),
        height=350, margin=dict(t=50),
    )
    st.plotly_chart(fig, use_container_width=True)
