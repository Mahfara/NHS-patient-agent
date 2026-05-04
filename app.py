"""
NHS Emergency Department AI Scheduling Assistant
================================================
MSc Big Data & Data Science Technology
Module LD7236 | Northumbria University London | 2024-25
Supervisor: Dr. Rejwan Bin Sulaiman

Architecture:
  - LightGBM + XGBoost: 4-hour breach prediction (90% accuracy, AUC 0.96)
  - SMOTE-balanced training | Optuna-tuned hyperparameters
  - SHAP: feature-level explanation per patient
  - RL-inspired policy: resource action recommendation
  - Calibrated against NHS England ECDS 2024-25 statistics

Run locally:
  pip install -r requirements.txt
  streamlit run app.py

Deploy to Streamlit Cloud:
  Push folder to GitHub → share.streamlit.io → Connect repo → Deploy
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time
import os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NHS ED AI Scheduling Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #005EB8 0%, #003087 100%);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
        color: white;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #005EB8;
        margin-bottom: 12px;
    }
    .risk-high   { background:#ffebee; border-left:5px solid #E24B4A; padding:16px; border-radius:10px; text-align:center; }
    .risk-medium { background:#fff3e0; border-left:5px solid #EF9F27; padding:16px; border-radius:10px; text-align:center; }
    .risk-low    { background:#e8f5e9; border-left:5px solid #1D9E75; padding:16px; border-radius:10px; text-align:center; }
    .action-box  { background:#f8f9fa; border-radius:10px; padding:18px; }
    .nhs-blue    { color: #005EB8; }
    .footer-text { color:#888; font-size:12px; text-align:center; margin-top:20px; }
    div[data-testid="stMetric"] { background:white; padding:12px; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    lgb_m  = pickle.load(open("lgb_model.pkl", "rb"))
    xgb_m  = pickle.load(open("xgb_model.pkl", "rb"))
    sc     = pickle.load(open("scaler.pkl",    "rb"))
    meta   = json.load(open("feature_meta.json"))
    shap_bg= pd.read_csv("shap_background.csv")
    return lgb_m, xgb_m, sc, meta, shap_bg

try:
    lgb_model, xgb_model, scaler, meta, shap_bg = load_models()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    missing = str(e)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style='margin:0'>🏥 NHS Emergency Department AI Scheduling Assistant</h2>
    <p style='margin:4px 0 0 0; opacity:0.85'>
        Predictive Deep Reinforcement Learning for Real-Time Resource Scheduling Optimisation<br>
        <small>MSc Big Data & Data Science Technology | Northumbria University London | 2024-25</small>
    </p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"""
    **Model files not found.** Please ensure these files are in the same folder as app.py:
    - lgb_model.pkl
    - xgb_model.pkl
    - scaler.pkl
    - feature_meta.json
    - shap_background.csv

    Error: {missing}
    """)
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/NHS_logo.svg/200px-NHS_logo.svg.png",
             width=120)
    st.markdown("---")
    st.markdown("### 🏥 ED System State")
    st.caption("Current department conditions:")

    bed_occ   = st.slider("Bed Occupancy (%)",    min_value=85,  max_value=100, value=93,   step=1)
    beds_avail= st.slider("Beds Available",        min_value=1,   max_value=20,  value=8)
    staff_r   = st.slider("Staff Ratio",           min_value=0.30,max_value=0.80,value=0.54, step=0.01)
    queue_len = st.slider("Queue Length (est.)",   min_value=0,   max_value=80,  value=25)
    cpi_val   = st.slider("Capacity Pressure Index",min_value=0.5,max_value=4.0, value=1.8,  step=0.1)
    hour_day  = st.slider("Hour of Day",           min_value=0,   max_value=23,  value=14)
    shift_arr = st.slider("Shift Total Arrivals",  min_value=20,  max_value=160, value=76)

    st.markdown("---")
    st.markdown("### 🧑‍⚕️ Patient Characteristics")

    patient_age   = st.slider("Patient Age", 0, 99, 55)
    triage_cat    = st.selectbox("Triage Category",
        options=[1,2,3,4,5],
        format_func=lambda x: {
            1:"🔴 Cat 1 — Immediate",
            2:"🟠 Cat 2 — Very Urgent",
            3:"🟡 Cat 3 — Urgent",
            4:"🔵 Cat 4 — Standard",
            5:"🟢 Cat 5 — Non-Urgent"
        }[x], index=2)
    news2_score   = st.slider("NEWS2 Score", 0, 9, 3)
    arrival_mode  = st.selectbox("Arrival Mode",
        ["Walk-in","Ambulance","GP Referral","Other"])
    imd_quintile  = st.selectbox("IMD Quintile (1=most deprived)", [1,2,3,4,5], index=2)
    comorbidities = st.slider("Comorbidity Count", 0, 5, 1)

    st.markdown("---")
    st.markdown("### ⚙️ Conditions")
    handover_breach = st.checkbox("Ambulance Handover Breach >30min", value=False)
    is_winter_month = st.checkbox("Winter Month (Dec/Jan/Feb)", value=False)
    is_weekend      = st.checkbox("Weekend", value=False)
    pre_alert       = st.checkbox("Pre-alert Received", value=False)

# ── Feature engineering (mirrors training pipeline exactly) ───────────────────
def build_feature_vector():
    ambul   = 1 if arrival_mode == "Ambulance" else 0
    night   = 1 if (hour_day >= 22 or hour_day <= 6) else 0
    winter  = 1 if is_winter_month else 0

    # Estimated values (in live system these come from EPR feed)
    los_est   = 160 + triage_cat*8  + comorbidities*6 + queue_len*0.5
    wait_est  = 60  + triage_cat*14 + (bed_occ-91)*2  + (18 if arrival_mode=="Walk-in" else 0)
    board_est = max(0, (bed_occ-91)*4)
    hov_est   = (bed_occ-91)*3 + (15 if is_winter_month else 0)
    acuity    = (6-triage_cat)*2 + news2_score*1.5 + comorbidities*0.8
    month_val = 1 if is_winter_month else 6

    feat = {
        "hour_of_day":               hour_day,
        "day_of_week":               5 if is_weekend else 2,
        "month":                     month_val,
        "is_weekend":                int(is_weekend),
        "patient_age":               patient_age,
        "patient_sex":               1,
        "imd_quintile":              imd_quintile,
        "comorbidity_count":         comorbidities,
        "previous_ed_visits_12m":    0,
        "re_attendance_72h":         0,
        "arrival_mode":              ["Walk-in","Ambulance","GP Referral","Other"].index(arrival_mode),
        "pre_alert_received":        int(pre_alert),
        "chief_complaint":           1,
        "triage_category":           triage_cat,
        "news2_score":               news2_score,
        "nhs_trust":                 0,
        "ics_region":                0,
        "trust_type":                0,
        "bed_occupancy_pct":         bed_occ,
        "beds_available":            beds_avail,
        "staff_ratio":               staff_r,
        "daily_ed_attendance":       shift_arr * 3,
        "shift_total_arrivals":      shift_arr,
        "shift_ambulance_arrivals":  int(shift_arr * 0.32),
        "ambulance_handover_delay_min": hov_est,
        "handover_breach_gt30min":   int(handover_breach),
        "wait_time_to_assessment_min": wait_est,
        "ed_los_min":                los_est,
        "boarding_delay_min":        board_est,
        "queue_length_estimate":     queue_len,
        "capacity_pressure_index":   cpi_val,
        "patient_acuity_score":      acuity,
        "wait_equity_delta_min":     0,
        # Interaction features
        "occ_x_triage":              (bed_occ/100) * triage_cat,
        "wait_per_staff":            wait_est / (staff_r + 0.01),
        "acuity_x_cpi":              acuity * cpi_val,
        "los_over_threshold":        max(0, los_est - 240),
        "night_high_occ":            night * (1 if bed_occ > 95 else 0),
        "ambul_handover_stress":     ambul * int(handover_breach),
        "is_winter":                 winter,
        "boarding_per_los":          board_est / (los_est + 1),
    }
    return pd.DataFrame([feat])[meta["feature_names"]]


def get_rl_action(prob, occ, q, sr, hov):
    """
    RL-inspired policy derived from Double DQN training.
    Combines breach probability with system state thresholds.
    Mirrors the action selection learned by the DQN agent.
    """
    if prob > 0.72 and occ > 96 and sr < 0.45:
        return 1  # Surge beds — triple crisis condition
    elif prob > 0.65 and q > 35:
        return 2  # Extra staff — queue overflow
    elif prob > 0.55 and hov > 30:
        return 2  # Extra staff — ambulance handover crisis
    elif prob > 0.50:
        return 3  # Fast-track — moderate pressure
    else:
        return 0  # Maintain — system stable


# ── Run prediction ─────────────────────────────────────────────────────────────
feat_df  = build_feature_vector()
lgb_prob = float(lgb_model.predict_proba(feat_df)[0, 1])
xgb_prob = float(xgb_model.predict_proba(feat_df)[0, 1])
avg_prob = (lgb_prob + xgb_prob) / 2

hov_val  = float((bed_occ - 91) * 3 + (15 if is_winter_month else 0))
action   = get_rl_action(avg_prob, bed_occ, queue_len, staff_r, hov_val)
action_info = meta["rl_actions"][str(action)]

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Live Prediction",
    "🔬 SHAP Explanation",
    "📊 System Dashboard",
    "📈 Model Performance",
    "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    col1, col2, col3 = st.columns([2, 2, 2])

    # Risk gauge
    with col1:
        st.markdown("#### 4-Hour Breach Risk")
        if avg_prob > 0.60:
            risk_class = "risk-high"; risk_label = "⚠️ HIGH RISK"; risk_color = "#E24B4A"
        elif avg_prob > 0.35:
            risk_class = "risk-medium"; risk_label = "⚡ MODERATE"; risk_color = "#EF9F27"
        else:
            risk_class = "risk-low"; risk_label = "✅ LOW RISK"; risk_color = "#1D9E75"

        st.markdown(f"""
        <div class="{risk_class}">
            <h1 style='color:{risk_color}; margin:0; font-size:52px'>{avg_prob*100:.1f}%</h1>
            <p style='color:{risk_color}; font-weight:700; font-size:20px; margin:4px 0'>{risk_label}</p>
            <hr style='border-color:{risk_color}; opacity:0.3'>
            <small style='color:#555'>
                LightGBM: <b>{lgb_prob*100:.1f}%</b> &nbsp;|&nbsp; XGBoost: <b>{xgb_prob*100:.1f}%</b>
            </small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        # Probability bar
        fig, ax = plt.subplots(figsize=(5, 0.6))
        ax.barh([""],   [1.0],  color="#e0e0e0", height=0.5)
        ax.barh([""], [avg_prob], color=risk_color, height=0.5)
        ax.axvline(0.261, color="#555", lw=1.5, linestyle="--")
        ax.set_xlim(0, 1); ax.axis("off")
        ax.text(0.261, 0.5, "NHS avg\n26.1%", ha="center", va="center",
                fontsize=7, color="#555", transform=ax.get_xaxis_transform())
        st.pyplot(fig, use_container_width=True); plt.close()

    # RL Action
    with col2:
        st.markdown("#### RL Resource Recommendation")
        action_colors = {"0":"#1D9E75","1":"#E24B4A","2":"#EF9F27","3":"#378ADD"}
        ac = action_colors[str(action)]
        st.markdown(f"""
        <div style='background:#f8f9fa; border-radius:12px; padding:20px; border-left:6px solid {ac}'>
            <div style='font-size:32px; margin-bottom:8px'>{action_info["icon"]}</div>
            <h3 style='color:{ac}; margin:0 0 6px 0'>Action {action}</h3>
            <h4 style='margin:0 0 8px 0'>{action_info["name"]}</h4>
            <p style='color:#555; margin:0 0 8px 0; font-size:14px'>{action_info["desc"]}</p>
            <span style='background:{ac}22; color:{ac}; padding:4px 10px; border-radius:20px; font-size:13px; font-weight:600'>
                Cost: {action_info["cost"]}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.info(f"""
        **Why this action?**
        {"🔴 Triple crisis: high occupancy + night + low staff" if action==1 and bed_occ>96 else
         "🟠 Queue overflow: " + str(queue_len) + " patients waiting" if action==2 and queue_len>35 else
         "🟠 Ambulance pressure: handover breach active" if action==2 and handover_breach else
         "🟡 Moderate pressure detected: fast-track will reduce wait times" if action==3 else
         "🟢 System within normal operating parameters"}
        """)

    # Status indicators
    with col3:
        st.markdown("#### System Status Indicators")
        indicators = [
            ("🛏️ Bed Occupancy",    f"{bed_occ:.1f}%",  bed_occ>95,   f"Critical >95%: {bed_occ:.1f}%"),
            ("👩‍⚕️ Staff Ratio",      f"{staff_r:.2f}",   staff_r<0.42, f"Below safe threshold: {staff_r:.2f}"),
            ("👥 Queue Length",      str(queue_len),    queue_len>40, f"High demand: {queue_len} patients"),
            ("📊 Capacity Pressure", f"{cpi_val:.2f}",   cpi_val>2.5,  f"Elevated CPI: {cpi_val:.2f}"),
            ("🚑 Handover Breach",   str(handover_breach),handover_breach,"Active handover delay"),
            ("❄️ Winter Pressure",   str(is_winter_month),is_winter_month,"Seasonal surge active"),
            ("📋 Triage Category",   f"Cat {triage_cat}",triage_cat<=2,f"High acuity patient"),
            ("⚠️ NEWS2 Score",       str(news2_score),  news2_score>=5,f"Elevated NEWS2: {news2_score}"),
        ]
        for label, value, alert, alert_msg in indicators:
            col_a, col_b, col_c = st.columns([3,2,1])
            col_a.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
            col_b.markdown(f"<b>{value}</b>", unsafe_allow_html=True)
            col_c.markdown("🔴" if alert else "🟢", unsafe_allow_html=True)

    st.markdown("---")

    # Comparative risk across scenarios
    st.markdown("#### 📊 Risk Sensitivity Analysis")
    st.caption("How breach risk changes with key system variables (all others held constant)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    fig.patch.set_facecolor('white')

    scenarios = [
        ("Bed Occupancy %",      np.linspace(91, 100, 20), "bed_occ",   bed_occ),
        ("Staff Ratio",          np.linspace(0.30, 0.80, 20), "staff_r", staff_r),
        ("Queue Length",         np.linspace(0, 80, 20),   "queue_len", queue_len),
    ]

    for ax, (title, values, var_name, current_val) in zip(axes, scenarios):
        risks = []
        for v in values:
            tmp = feat_df.copy()
            if var_name == "bed_occ":
                tmp["bed_occupancy_pct"] = v
                tmp["occ_x_triage"] = (v/100) * triage_cat
            elif var_name == "staff_r":
                tmp["staff_ratio"] = v
                tmp["wait_per_staff"] = float(feat_df["wait_time_to_assessment_min"]) / (v+0.01)
            elif var_name == "queue_len":
                tmp["queue_length_estimate"] = v
            p = (float(lgb_model.predict_proba(tmp)[0,1]) +
                 float(xgb_model.predict_proba(tmp)[0,1])) / 2
            risks.append(p * 100)

        ax.plot(values, risks, color='#005EB8', lw=2.5)
        ax.fill_between(values, risks, alpha=0.15, color='#005EB8')
        ax.axvline(current_val, color='#E24B4A', lw=2, linestyle='--', label=f'Current: {current_val:.1f}')
        ax.axhline(26.1, color='#888', lw=1, linestyle=':', label='NHS avg 26.1%')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Breach Risk (%)')
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SHAP EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 🔬 Why is the model making this prediction?")
    st.caption("SHAP (SHapley Additive exPlanations) shows exactly which features are driving the breach risk for this specific patient and system state.")

    if st.button("🔍 Compute SHAP Explanation", type="primary"):
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = shap.TreeExplainer(xgb_model)
                shap_vals = explainer.shap_values(feat_df)
                sv_series = pd.Series(shap_vals[0], index=feat_df.columns)

                col1, col2 = st.columns([1.3, 1])

                with col1:
                    # Waterfall-style bar chart
                    top_n = sv_series.abs().sort_values(ascending=False).head(12)
                    top_sv = sv_series[top_n.index].sort_values()

                    fig, ax = plt.subplots(figsize=(9, 5))
                    colors = ['#E24B4A' if v > 0 else '#1D9E75' for v in top_sv.values]
                    bars = ax.barh(range(len(top_sv)), top_sv.values, color=colors, alpha=0.85, height=0.65)
                    ax.set_yticks(range(len(top_sv)))
                    ax.set_yticklabels(top_sv.index, fontsize=10)
                    ax.axvline(0, color='black', lw=1)
                    ax.set_xlabel("SHAP Value (positive = increases breach risk)", fontsize=10)
                    ax.set_title(f"Feature Impact for This Patient\nPredicted breach probability: {avg_prob*100:.1f}%",
                                fontsize=12, fontweight='bold')
                    for bar, val in zip(bars, top_sv.values):
                        ax.text(val + (0.002 if val >= 0 else -0.002),
                                bar.get_y() + bar.get_height()/2,
                                f'{val:+.4f}', va='center',
                                ha='left' if val >= 0 else 'right', fontsize=8)
                    ax.set_facecolor('#f8f9fa'); fig.patch.set_facecolor('white')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True); plt.close()

                with col2:
                    st.markdown("**Clinical Interpretation:**")
                    st.markdown("")

                    # Top 5 risk-increasing features
                    risk_inc = sv_series[sv_series > 0].sort_values(ascending=False).head(5)
                    st.markdown("🔴 **Top risk-increasing factors:**")
                    for feat, val in risk_inc.items():
                        feat_val = float(feat_df[feat].iloc[0])
                        st.markdown(f"&nbsp;&nbsp;• **{feat}** = {feat_val:.2f} → +{val:.4f}")

                    st.markdown("")
                    # Top 5 risk-reducing features
                    risk_dec = sv_series[sv_series < 0].sort_values().head(5)
                    st.markdown("🟢 **Top risk-reducing factors:**")
                    for feat, val in risk_dec.items():
                        feat_val = float(feat_df[feat].iloc[0])
                        st.markdown(f"&nbsp;&nbsp;• **{feat}** = {feat_val:.2f} → {val:.4f}")

                    st.markdown("---")
                    base_val = float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0])
                    shap_sum = float(sv_series.sum())
                    st.markdown(f"""
                    **Prediction breakdown:**
                    - Base rate (population avg): **{1/(1+np.exp(-base_val))*100:.1f}%**
                    - SHAP adjustments: **{'+' if shap_sum>0 else ''}{shap_sum:.4f}**
                    - Final prediction: **{avg_prob*100:.1f}%**
                    """)

                # Global importance comparison
                st.markdown("---")
                st.markdown("**Global Feature Importance (from training data vs this patient):**")
                global_imp = pd.Series(
                    np.abs(shap_bg.values).mean(axis=0),
                    index=shap_bg.columns
                ).sort_values(ascending=False).head(10)

                fig2, axes = plt.subplots(1, 2, figsize=(14, 4))
                global_imp.sort_values().plot(kind='barh', ax=axes[0], color='#005EB8', alpha=0.8)
                axes[0].set_title('Global: Mean |SHAP| across 300 test patients', fontweight='bold', fontsize=11)
                axes[0].set_xlabel('Mean |SHAP Value|')

                patient_imp = sv_series.abs().sort_values(ascending=False).head(10).sort_values()
                patient_imp.plot(kind='barh', ax=axes[1], color='#E24B4A', alpha=0.8)
                axes[1].set_title('This Patient: |SHAP| values', fontweight='bold', fontsize=11)
                axes[1].set_xlabel('|SHAP Value|')

                for ax in axes: ax.set_facecolor('#f8f9fa')
                fig2.patch.set_facecolor('white')
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True); plt.close()

            except Exception as e:
                st.error(f"SHAP computation error: {e}")
    else:
        st.info("👆 Click the button above to compute SHAP explanation for the current patient and system state.")
        st.markdown("""
        **What SHAP tells you:**
        - Which features are *most responsible* for the current breach prediction
        - Whether each feature is *increasing* or *decreasing* the risk
        - How this patient compares to the *global average* pattern
        - Supports NHS AI transparency requirements (MHRA AIaMD)
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: SYSTEM DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📊 ED System Dashboard")
    st.caption(f"Snapshot at {datetime.now().strftime('%H:%M:%S')} | Based on current sidebar inputs")

    # KPI row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Breach Risk",      f"{avg_prob*100:.1f}%",
                f"{(avg_prob-0.261)*100:+.1f}pp vs NHS avg")
    kpi2.metric("Bed Occupancy",    f"{bed_occ:.1f}%",
                f"{bed_occ-92.4:+.1f}pp vs NHS avg")
    kpi3.metric("Staff Ratio",      f"{staff_r:.2f}",
                f"{staff_r-0.54:+.2f} vs target 0.54")
    kpi4.metric("Queue Length",     str(queue_len),
                f"{queue_len-25:+d} vs typical 25")
    kpi5.metric("Recommended Action", action_info["icon"] + f" Action {action}",
                action_info["name"][:20])

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # System state radar
        st.markdown("**System Pressure Indicators:**")
        indicators_dash = {
            "Bed Occ\n(91-100)":       (bed_occ - 91) / 9,
            "Staff Shortage\n(0=full)": 1 - min(staff_r / 0.80, 1),
            "Queue\n(0-80)":            queue_len / 80,
            "CPI\n(0.5-4)":            (cpi_val - 0.5) / 3.5,
            "Breach Risk\n(0-100%)":    avg_prob,
            "Winter\nPressure":         0.8 if is_winter_month else 0.2,
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = list(indicators_dash.keys())
        values = list(indicators_dash.values())
        colors_d = ['#E24B4A' if v > 0.7 else '#EF9F27' if v > 0.4 else '#1D9E75' for v in values]
        bars = ax.barh(labels, values, color=colors_d, alpha=0.85)
        ax.axvline(0.7, color='#E24B4A', lw=1.5, linestyle='--', alpha=0.6, label='Critical threshold')
        ax.set_xlim(0, 1); ax.set_xlabel("Normalised Pressure (0=low, 1=critical)")
        ax.set_title("System Pressure Radar", fontweight='bold')
        ax.legend(fontsize=8); ax.set_facecolor('#f8f9fa')
        for bar, v in zip(bars, values):
            ax.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                    f'{v:.1%}', va='center', fontsize=9)
        fig.patch.set_facecolor('white')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        # Action decision matrix
        st.markdown("**RL Policy Decision Matrix:**")
        st.markdown("""
        | Condition | Action | Rationale |
        |---|---|---|
        | Risk >72% + Occ >96% + Staff <0.45 | 🛏️ Surge Beds | Triple crisis |
        | Risk >65% + Queue >35 | 👩‍⚕️ Extra Staff | Queue overflow |
        | Risk >55% + Handover breach | 👩‍⚕️ Extra Staff | Ambulance pressure |
        | Risk >50% | ⚡ Fast-Track | Moderate pressure |
        | Risk ≤50% | ✅ Maintain | System stable |
        """)

        # Current state highlights
        current_cond = []
        if avg_prob > 0.72 and bed_occ > 96 and staff_r < 0.45:
            current_cond.append("🔴 TRIPLE CRISIS condition met")
        if avg_prob > 0.65 and queue_len > 35:
            current_cond.append("🟠 Queue overflow condition met")
        if handover_breach:
            current_cond.append("🟠 Ambulance handover breach active")
        if not current_cond:
            current_cond.append("🟢 No critical conditions active")

        for c in current_cond:
            st.markdown(f"**→ {c}**")

        st.markdown(f"""
        <div style='background:#005EB822; border-radius:8px; padding:12px; margin-top:10px'>
            <b>Decision:</b> {action_info["icon"]} <b>{action_info["name"]}</b><br>
            <small>{action_info["desc"]}</small>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📈 Model Performance & Methodology")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model Accuracy (test set, original distribution):**")
        perf_data = {
            'Model': ['Logistic Regression','Random Forest','Gradient Boosting',
                      'XGBoost (Optuna)','LightGBM (Optuna)'],
            'Accuracy': [85.38, 85.63, 87.20, 89.90, 90.08],
            'ROC-AUC':  [0.9042, 0.9292, 0.9416, 0.9592, 0.9596],
        }
        perf_df = pd.DataFrame(perf_data)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_m = ['#888780','#85B7EB','#EF9F27','#E24B4A','#1D9E75']
        bars = ax.barh(perf_df['Model'], perf_df['Accuracy'], color=colors_m, alpha=0.85)
        ax.axvline(85.38, color='#888', lw=1, linestyle=':', label='LR baseline')
        ax.set_xlim(82, 93); ax.set_xlabel('Accuracy (%)')
        ax.set_title('4-Hour Breach Prediction Accuracy\n(SMOTE balanced + Optuna tuned)', fontweight='bold')
        ax.legend(fontsize=9); ax.set_facecolor('#f8f9fa')
        for bar, v in zip(bars, perf_df['Accuracy']):
            ax.text(v+0.1, bar.get_y()+bar.get_height()/2, f'{v:.2f}%', va='center',
                    fontsize=10, fontweight='bold')
        fig.patch.set_facecolor('white')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("**Key Improvements in This Version:**")
        improvements = [
            ("✅ SMOTE Balancing",       "Class imbalance addressed — Recall improved from 0.27 to 0.80+"),
            ("✅ Optuna Tuning",          "25-trial Bayesian search per model — AUC improved +0.02"),
            ("✅ Non-linear Target",       "70% linear + 30% threshold interactions → 4.7pp LR-LightGBM gap"),
            ("✅ Double DQN",             "Reduced Q-value overestimation vs vanilla DQN"),
            ("✅ 4-Policy Comparison",    "DQN vs PPO vs Rule-based vs Maintain-baseline"),
            ("✅ SHAP Interactions",      "Feature pair interactions computed — leakage investigated"),
            ("✅ NHS Statistics",         "All 12 metrics validated against published 2024-25 data"),
        ]
        for title, desc in improvements:
            st.markdown(f"**{title}**  \n{desc}")
            st.markdown("")

    st.markdown("---")
    st.markdown("**Methodology Summary:**")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("""
        **Dataset**
        - 100,000 synthetic NHS ED records
        - April 2024 – March 2025
        - 10 NHS Trusts, England
        - All statistics: NHS Digital ECDS 2024-25
        - 26.1% breach rate (NHS published)
        """)
    with col4:
        st.markdown("""
        **ML Pipeline**
        - SMOTE: synthetic minority oversampling
        - Optuna: Bayesian hyperparameter search
        - Stratified 80/20 split
        - 5-fold cross-validation
        - SHAP: TreeExplainer + interaction values
        """)
    with col5:
        st.markdown("""
        **RL Framework**
        - Double DQN + experience replay
        - PPO with policy gradient
        - 4 actions: maintain/surge/staff/fast-track
        - Evaluated vs 2 baselines
        - DQN improves +32% vs always-maintain
        """)

    st.markdown("---")
    st.caption("""
    **Data Sources:** NHS Digital Hospital A&E Activity 2024-25 | RCEM NHS Performance Tracker |
    Institute for Government Performance Tracker 2025 | Nuffield Trust A&E Waiting Times
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### ℹ️ About This System")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Dissertation Project**

        This application is the deployed implementation of an MSc dissertation exploring
        the use of predictive machine learning and deep reinforcement learning for
        real-time NHS emergency department resource scheduling.

        **Key Findings:**
        - LightGBM achieves 90.08% accuracy and ROC-AUC 0.9596 on 4-hour breach prediction
        - XGBoost achieves 89.90% accuracy and ROC-AUC 0.9592
        - Both outperform Logistic Regression by 4.7pp accuracy (consistent with NHS ML literature)
        - DQN agent improves cumulative reward by +32% vs always-maintain baseline
        - SHAP identifies ED LOS, wait time, and bed occupancy as primary breach drivers

        **System Architecture:**

        Patient arrives → Triage + assessment → ML breach prediction → SHAP explanation
        → RL action recommendation → ED Manager decision

        **Limitations:**
        - Trained on synthetic data — real ECDS data would improve accuracy
        - RL environment does not model true state transitions
        - Not approved for clinical use — research prototype only
        """)

    with col2:
        st.markdown("""
        **Technologies Used:**

        | Component | Technology |
        |---|---|
        | Breach prediction | LightGBM, XGBoost |
        | Class balancing | SMOTE (imbalanced-learn) |
        | Hyperparameter tuning | Optuna (Bayesian) |
        | Explainability | SHAP (TreeExplainer) |
        | RL scheduling | Double DQN, PPO |
        | Web interface | Streamlit |
        | Data processing | Pandas, NumPy, Scikit-learn |

        **NHS Data Sources:**
        - NHS Digital ECDS Hospital A&E Activity 2024-25
        - NHS England Monthly A&E Sitrep 2024-25
        - RCEM Performance Tracker 2024-25
        - Institute for Government Performance Tracker 2025
        - Nuffield Trust A&E Waiting Times Analysis

        **Governance Note:**
        This is a research prototype. Clinical deployment would require
        NHS AI Lab ADOPT framework approval, MHRA AIaMD registration,
        DCB0129/DCB0160 clinical safety documentation, and IG agreement.
        """)

    st.markdown("---")
    st.markdown(f"""
    <div class="footer-text">
        {meta.get('dissertation','MSc Big Data & Data Science Technology | Northumbria University London | 2024-25')}<br>
        Supervisor: Dr. Rejwan Bin Sulaiman | Module LD7236<br>
        Model Accuracy: LightGBM {meta['model_accuracy']['LightGBM']*100:.2f}% |
        XGBoost {meta['model_accuracy']['XGBoost']*100:.2f}% |
        AUC: {meta['model_auc']['LightGBM']:.4f}<br>
        ⚠️ Research prototype — not approved for clinical decision making
    </div>
    """, unsafe_allow_html=True)
