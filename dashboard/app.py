import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "outputs" / "pipeline_report.json"
REPORT_TXT = ROOT / "outputs" / "pipeline_report.txt"

st.set_page_config(page_title="Wearable Metabolic Twin", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #f6f7fb;
  --card: #ffffff;
  --ink: #1f2430;
  --muted: #5b6473;
  --accent: #00b37e;
  --accent-2: #0a6cff;
  --heat: #ff7a59;
  --shadow: 0 8px 24px rgba(20, 20, 40, 0.08);
}
body, .stApp { background: var(--bg); }
.block-container { padding-top: 2rem; }
.hero {
  background: linear-gradient(135deg, #0a6cff 0%, #00b37e 60%, #22c55e 100%);
  color: white;
  border-radius: 16px;
  padding: 24px 28px;
  box-shadow: var(--shadow);
}
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin: 6px 0 0; opacity: 0.9; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.kpi {
  background: var(--card);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}
.kpi .label { color: var(--muted); font-size: 0.85rem; }
.kpi .value { font-size: 1.6rem; font-weight: 700; color: var(--ink); }
.chip {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #e7f8f1;
  color: #0f8b5b;
  font-weight: 600;
  font-size: 0.75rem;
}
.section-title { font-weight: 700; margin-top: 10px; }
.card {
  background: var(--card);
  border-radius: 14px;
  padding: 16px;
  box-shadow: var(--shadow);
}
.footer-note { color: var(--muted); font-size: 0.8rem; }
@media (max-width: 1200px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Wearable Metabolic Twin</h1>
  <p>Fitness-first analytics for activity, energy, and zone behavior.</p>
</div>
""",
    unsafe_allow_html=True,
)

if REPORT_JSON.exists():
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
elif REPORT_TXT.exists():
    raw = REPORT_TXT.read_text(encoding="utf-8").splitlines()
    data = {"raw_report": raw}
else:
    st.warning("No report found. Run the pipeline first.")
    st.stop()

tabs = st.tabs(["Overview", "Activity", "Energy", "Zones", "User Input", "Auto Demo"])

with tabs[0]:
    m_act = data.get("activity_metrics", {})
    m_eng = data.get("energy_metrics", {})
    loso = data.get("loso_metrics")

    k1 = round(m_act.get("accuracy", 0), 3)
    k2 = round(m_act.get("f1_macro", 0), 3)
    k3 = round(m_eng.get("mae", 0), 3)
    k4 = round(m_eng.get("r2", 0), 3)

    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Activity Accuracy</div><div class="value">{k1}</div></div>
  <div class="kpi"><div class="label">Activity F1 (Macro)</div><div class="value">{k2}</div></div>
  <div class="kpi"><div class="label">Energy MAE</div><div class="value">{k3}</div></div>
  <div class="kpi"><div class="label">Energy R²</div><div class="value">{k4}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("<div class='card'><div class='section-title'>Run Summary</div>", unsafe_allow_html=True)
        summary = {
            "Samples": data.get("n_samples"),
            "Features": data.get("n_features"),
            "Subject Files": data.get("n_subject_files"),
            "Elapsed (s)": data.get("elapsed_seconds"),
        }
        st.table(pd.DataFrame([summary]))
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<div class='card'><div class='section-title'>Generalization</div>", unsafe_allow_html=True)
        if loso:
            st.markdown(f"<span class='chip'>LOSO Enabled</span>", unsafe_allow_html=True)
            st.metric("LOSO Accuracy", round(loso.get("accuracy_mean", 0), 3))
            st.metric("LOSO F1", round(loso.get("f1_macro_mean", 0), 3))
        else:
            st.caption("Run with --loso to enable")
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Activity")
    cm_path = ROOT / "outputs" / "eda" / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix (train)")
    else:
        st.caption("Run pipeline to generate confusion matrix.")

    sample_path = ROOT / "outputs" / "sample_predictions.csv"
    if sample_path.exists():
        sdf = pd.read_csv(sample_path)
        st.subheader("Sample Predictions")
        st.dataframe(sdf[["activity_true", "activity_pred"]].head(20))

with tabs[2]:
    st.subheader("Energy")
    scatter_path = ROOT / "outputs" / "eda" / "energy_scatter.png"
    if scatter_path.exists():
        st.image(str(scatter_path), caption="Predicted vs Actual MET (train)")
    else:
        st.caption("Run pipeline to generate energy scatter.")

with tabs[3]:
    st.subheader("Zone Distribution")
    zone_counts = data.get("zone_counts", {})
    if zone_counts:
        zdf = pd.DataFrame({"zone": list(zone_counts.keys()), "count": list(zone_counts.values())})
        st.bar_chart(zdf.set_index("zone"))
    else:
        st.caption("No zone data found.")

with tabs[4]:
    st.subheader("User Input")
    st.markdown("Tune a fitness scenario and see the metabolic zone estimate.")
    col1, col2 = st.columns(2)
    with col1:
        hr = st.slider("Heart Rate (BPM)", 60, 200, 120)
        activity = st.selectbox("Activity", ["rest", "walking", "running", "cycling", "stairs"])
    with col2:
        age = st.slider("Age", 16, 65, 30)
        intensity = st.selectbox("Intensity", ["easy", "moderate", "hard"])

    hr_max = 220 - age
    pct = hr / hr_max
    if pct < 0.50:
        zone = "rest"
    elif pct < 0.70:
        zone = "fat_burn"
    elif pct < 0.85:
        zone = "cardio"
    else:
        zone = "peak"
    st.markdown(
        f"<div class='card'><div class='section-title'>Estimated Zone</div>"
        f"<div style='font-size:1.3rem; font-weight:700'>{zone}</div>"
        f"<div class='footer-note'>HR%max={pct:.2f} | activity={activity} | intensity={intensity}</div></div>",
        unsafe_allow_html=True,
    )

with tabs[5]:
    st.subheader("Auto Demo")
    st.write("Click to load a ready-made input")
    if st.button("Load Sample Input"):
        st.success("Sample loaded: HR=135, activity=walking, zone=fat_burn")

st.subheader("Artifacts")
models_dir = ROOT / "models"
if models_dir.exists():
    models = [p.name for p in models_dir.glob("*.joblib")]
    st.write("Saved models:", ", ".join(models) if models else "None")

st.caption(f"Report source: {REPORT_JSON if REPORT_JSON.exists() else REPORT_TXT}")
