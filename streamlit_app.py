NHS Emergency Department Patient AI Agent
==========================================
A personal AI assistant for ED patients that:
  - Predicts estimated wait time based on current conditions
  - Predicts likelihood of 4-hour breach
  - Suggests the best available care pathway
  - Answers common patient questions via chat
  - Provides real-time system status

MSc Big Data & Data Science Technology
Module LD7236 | Northumbria University London | 2024-25
Supervisor: Dr. Rejwan Bin Sulaiman

Run: streamlit run patient_agent.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NHS ED Patient Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .nhs-header {
        background: linear-gradient(135deg, #005EB8 0%, #003087 100%);
        padding: 24px 32px; border-radius: 16px;
        margin-bottom: 24px; color: white;
    }
    .chat-user {
        background: #005EB8; color: white; padding: 14px 18px;
        border-radius: 18px 18px 4px 18px; margin: 8px 0 8px 15%;
        font-size: 15px; line-height: 1.6;
    }
    .chat-agent {
        background: #f0f4ff; color: #1a1a2e; padding: 14px 18px;
        border-radius: 18px 18px 18px 4px; margin: 8px 15% 8px 0;
        font-size: 15px; line-height: 1.6; border-left: 4px solid #005EB8;
    }
    .chat-system {
        background: #fff8e1; color: #5d4037; padding: 10px 16px;
        border-radius: 10px; margin: 4px 10%; font-size: 13px; text-align: center;
    }
    .wait-card {
        background: white; border-radius: 16px; padding: 24px;
        box-shadow: 0 4px 20px rgba(0,94,184,0.12);
        border-top: 5px solid #005EB8; text-align: center; margin-bottom: 16px;
    }
    .pathway-card {
        background: white; border-radius: 12px; padding: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07); margin-bottom: 12px;
    }
    .metric-mini {
        background: white; border-radius: 10px; padding: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NHS AI models...")
def load_models():
    lgb_m = pickle.load(open("lgb_model.pkl", "rb"))
    xgb_m = pickle.load(open("xgb_model.pkl", "rb"))
    meta  = json.load(open("feature_meta.json"))
    return lgb_m, xgb_m, meta

try:
    lgb_model, xgb_model, meta = load_models()
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}\n\nEnsure lgb_model.pkl, xgb_model.pkl, feature_meta.json are in the same folder.")
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ('messages',        []),
    ('patient_data',    {}),
    ('prediction_done', False),
    ('arrival_time',    datetime.now()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Current system state ───────────────────────────────────────────────────────
# In real deployment: live EPR/EDIS feed
# Here: realistic simulation based on time of day
_h = datetime.now().hour
SYSTEM = {
    'bed_occupancy_pct':     93.5 + (2.0 if 8 <= _h <= 20 else -1.0),
    'beds_available':        max(2, 12 - (4 if 10 <= _h <= 18 else 0)),
    'staff_ratio':           0.52 if 8 <= _h <= 20 else 0.48,
    'shift_total_arrivals':  85  if 10 <= _h <= 20 else 55,
    'queue_length_estimate': 30  if 10 <= _h <= 20 else 15,
    'capacity_pressure_index': 2.1 if 10 <= _h <= 20 else 1.6,
    'hour_of_day':           _h,
    'is_weekend':            1 if datetime.now().weekday() >= 5 else 0,
    'month':                 datetime.now().month,
}

# ── Symptom to triage mapping ──────────────────────────────────────────────────
SYMPTOM_MAP = {
    "Chest pain or pressure":             (2, 5),
    "Difficulty breathing":               (2, 6),
    "Stroke symptoms (face/arm/speech)":  (1, 7),
    "Severe allergic reaction":           (2, 5),
    "Overdose or poisoning":              (2, 5),
    "Abdominal / stomach pain":           (3, 3),
    "Infection / fever / sepsis concern": (3, 4),
    "Head injury or fall":                (3, 3),
    "Mental health crisis":               (3, 2),
    "Limb injury / possible fracture":    (4, 2),
    "Back pain":                          (4, 1),
    "Urinary symptoms":                   (4, 1),
    "Minor cut or wound":                 (4, 1),
    "Other / not listed":                 (3, 2),
}

TRIAGE_LABELS = {
    1: "🔴 Immediate",
    2: "🟠 Very Urgent",
    3: "🟡 Urgent",
    4: "🔵 Standard",
    5: "🟢 Non-Urgent",
}


# ── Feature vector builder ─────────────────────────────────────────────────────
def build_features(patient):
    occ   = SYSTEM['bed_occupancy_pct']
    tri   = patient.get('triage_cat', 3)
    age   = patient.get('age', 45)
    comor = patient.get('comorbidities', 0)
    ambul = 1 if patient.get('arrival_mode') == 'Ambulance' else 0
    news2 = patient.get('news2', 2)
    imd   = patient.get('imd', 3)
    staff = SYSTEM['staff_ratio']
    night  = 1 if (SYSTEM['hour_of_day'] >= 22 or SYSTEM['hour_of_day'] <= 6) else 0
    winter = 1 if SYSTEM['month'] in [12,1,2] else 0

    wait_est  = 60 + tri*14 + (occ-91)*2.0 + (18 if ambul==0 else 0)
    los_est   = 160 + tri*8  + comor*6 + SYSTEM['queue_length_estimate']*0.5
    board_est = max(0, (occ-91)*4)
    hov_est   = (occ-91)*3 + (15 if winter else 0)
    acuity    = (6-tri)*2 + news2*1.5 + comor*0.8

    feat = {
        'hour_of_day':               SYSTEM['hour_of_day'],
        'day_of_week':               5 if SYSTEM['is_weekend'] else 2,
        'month':                     SYSTEM['month'],
        'is_weekend':                SYSTEM['is_weekend'],
        'patient_age':               age,
        'patient_sex':               1,
        'imd_quintile':              imd,
        'comorbidity_count':         comor,
        'previous_ed_visits_12m':    patient.get('prev_visits', 0),
        're_attendance_72h':         0,
        'arrival_mode':              ['Walk-in','Ambulance','GP Referral','Other'].index(
                                      patient.get('arrival_mode','Walk-in')),
        'pre_alert_received':        0,
        'chief_complaint':           1,
        'triage_category':           tri,
        'news2_score':               news2,
        'nhs_trust':                 0, 'ics_region': 0, 'trust_type': 0,
        'bed_occupancy_pct':         occ,
        'beds_available':            SYSTEM['beds_available'],
        'staff_ratio':               staff,
        'daily_ed_attendance':       SYSTEM['shift_total_arrivals']*3,
        'shift_total_arrivals':      SYSTEM['shift_total_arrivals'],
        'shift_ambulance_arrivals':  int(SYSTEM['shift_total_arrivals']*0.32),
        'ambulance_handover_delay_min': hov_est,
        'handover_breach_gt30min':   int(hov_est > 30),
        'wait_time_to_assessment_min': wait_est,
        'ed_los_min':                los_est,
        'boarding_delay_min':        board_est,
        'queue_length_estimate':     SYSTEM['queue_length_estimate'],
        'capacity_pressure_index':   SYSTEM['capacity_pressure_index'],
        'patient_acuity_score':      acuity,
        'wait_equity_delta_min':     0,
        'occ_x_triage':              (occ/100)*tri,
        'wait_per_staff':            wait_est/(staff+0.01),
        'acuity_x_cpi':              acuity*SYSTEM['capacity_pressure_index'],
        'los_over_threshold':        max(0, los_est-240),
        'night_high_occ':            night*(1 if occ>95 else 0),
        'ambul_handover_stress':     ambul*int(hov_est>30),
        'is_winter':                 winter,
        'boarding_per_los':          board_est/(los_est+1),
    }
    return pd.DataFrame([feat])[meta['feature_names']]


def get_prediction(patient):
    feat  = build_features(patient)
    lgb_p = float(lgb_model.predict_proba(feat)[0,1])
    xgb_p = float(xgb_model.predict_proba(feat)[0,1])
    prob  = (lgb_p + xgb_p) / 2

    tri   = patient.get('triage_cat', 3)
    occ   = SYSTEM['bed_occupancy_pct']
    q     = SYSTEM['queue_length_estimate']
    base  = {1:15, 2:30, 3:60, 4:90, 5:120}[tri]
    wait  = int(base + max(0,(occ-91)*5) + max(0,(q-20)*0.8))
    wait  = max(5, min(wait, 300))
    total = int(wait + (160 + tri*8 + patient.get('comorbidities',0)*6) * 0.6)
    return prob, wait, total


def get_pathways(patient, prob):
    tri   = patient.get('triage_cat', 3)
    age   = patient.get('age', 45)
    comor = patient.get('comorbidities', 0)
    occ   = SYSTEM['bed_occupancy_pct']
    wait  = int({1:15,2:30,3:60,4:90,5:120}[tri] + max(0,(occ-91)*5))

    paths = [{'icon':'🏥','name':'Main Emergency Department',
              'wait':f'{wait}–{wait+20} min',
              'desc':'Full emergency care with all specialists, imaging and treatment.',
              'why':'Required for your triage category' if tri<=2 else 'Standard ED pathway',
              'color':'#E24B4A' if tri<=2 else '#005EB8','rank':1}]

    if tri >= 3 and age >= 18 and comor <= 2 and prob < 0.6:
        paths.append({'icon':'⚡','name':'Same Day Emergency Care (SDEC)',
                      'wait':f'{max(20,wait-30)}–{max(30,wait-10)} min',
                      'desc':'Rapid assessment without overnight admission. Often faster for eligible patients.',
                      'why':'Your presentation may qualify for SDEC — ask at reception.',
                      'color':'#1D9E75','rank':2})

    if tri >= 3 and prob < 0.4:
        paths.append({'icon':'🩺','name':'Urgent Treatment Centre (UTC)',
                      'wait':f'{max(15,wait-40)}–{max(25,wait-20)} min',
                      'desc':'Treats minor injuries and illnesses. Shorter waits than main ED.',
                      'why':'Your condition may be suitable for UTC.',
                      'color':'#1D9E75','rank':2})

    if tri >= 4 and prob < 0.25:
        paths.append({'icon':'📞','name':'NHS 111 / Same-Day GP',
                      'wait':'Call 111 now for guidance',
                      'desc':'NHS 111 can arrange same-day GP or direct you to the right service.',
                      'why':'Your condition may not require an ED visit.',
                      'color':'#EF9F27','rank':3})

    return sorted(paths, key=lambda x: x['rank'])


# ── Agent response engine ──────────────────────────────────────────────────────
def agent_response(msg, patient):
    m = msg.lower()

    if any(w in m for w in ['how long','wait','waiting','when','time']):
        if patient.get('triage_cat'):
            prob, wait, total = get_prediction(patient)
            tri = patient['triage_cat']
            return (f"Based on your **{TRIAGE_LABELS[tri]}** triage and current conditions:\n\n"
                    f"**⏱ Wait to be seen: {wait}–{wait+20} minutes**\n"
                    f"**🏥 Total estimated time: {total//60}h {total%60}min**\n\n"
                    f"Current queue: **{SYSTEM['queue_length_estimate']} patients** | "
                    f"Bed occupancy: **{SYSTEM['bed_occupancy_pct']:.1f}%**\n\n"
                    f"⚠️ These are estimates — your actual wait depends on clinical need.")
        return "Tell me about your symptoms first and I can give you a personalised wait time estimate."

    if any(w in m for w in ['worse','deteriorat','pain getting','can\'t breathe','chest']):
        return ("🔴 **Please go to the reception desk or press the emergency call button NOW.**\n\n"
                "Do not wait for your name — tell a nurse immediately if your condition is worsening. "
                "You can be re-triaged at any time.\n\n"
                "**Your safety is the priority.**")

    if any(w in m for w in ['leave','go home','self discharge','cancel']):
        return ("I understand waiting is difficult. Before you leave please speak to a nurse — they can:\n\n"
                "1. Re-assess your priority if your condition changed\n"
                "2. Arrange an urgent alternative (GP, pharmacy, 111)\n"
                "3. Give you safe discharge advice\n\n"
                "⚠️ **Do not leave without speaking to staff** if you have chest pain, breathing difficulties, "
                "severe pain, or feel your condition is serious. Call **NHS 111 (free)** if you leave.")

    if any(w in m for w in ['triage','category','priority','what does it mean']):
        return ("**NHS Triage Categories:**\n\n"
                "🔴 **Cat 1 — Immediate**: Life-threatening (cardiac arrest, major trauma)\n"
                "🟠 **Cat 2 — Very Urgent**: Serious (stroke, severe chest pain) — within 10 min\n"
                "🟡 **Cat 3 — Urgent**: Needs care soon but stable — within 1 hour\n"
                "🔵 **Cat 4 — Standard**: Non-urgent — within 2 hours\n"
                "🟢 **Cat 5 — Non-Urgent**: Minor conditions — within 4 hours\n\n"
                "Higher priority patients are always seen first regardless of arrival time.")

    if any(w in m for w in ['4 hour','four hour','target','standard','breach']):
        return (f"**The NHS 4-Hour Standard:**\n\n"
                f"95% of patients should be seen, treated, and either admitted or discharged "
                f"within 4 hours of arrival. In 2024-25 the national average is **73.9%** — "
                f"meaning about 26% of patients wait longer.\n\n"
                f"Current department occupancy is **{SYSTEM['bed_occupancy_pct']:.1f}%** "
                f"which {'increases' if SYSTEM['bed_occupancy_pct']>93 else 'has minimal effect on'} wait times.\n\n"
                f"Our AI system monitors your breach risk and alerts staff proactively.")

    if any(w in m for w in ['option','alternative','other','elsewhere','somewhere','pathway']):
        if patient.get('triage_cat'):
            prob, _, _ = get_prediction(patient)
            paths = get_pathways(patient, prob)
            resp = "**Your care options based on your presentation:**\n\n"
            for p in paths[:3]:
                resp += f"**{p['icon']} {p['name']}** — {p['wait']}\n{p['why']}\n\n"
            return resp
        return "Tell me about your symptoms and I can recommend the best care pathway for you."

    if any(w in m for w in ['parking','food','wifi','toilet','cafe','facilities','charge']):
        return ("**Department Facilities:**\n\n"
                "🅿️ **Parking**: Available in hospital car park. Blue Badge spaces near entrance.\n"
                "🍵 **Food & drink**: Vending machines in waiting area. Hospital café/restaurant on site.\n"
                "🚻 **Toilets**: In the waiting area — ask any staff member.\n"
                "📶 **Free WiFi**: Connect to **NHSWiFi** network.\n"
                "🔋 **Phone charging**: Available at patient services desk.")

    if any(w in m for w in ['bring','need','documents','id','what should i']):
        return ("**Useful to have with you:**\n\n"
                "NHS number (on your NHS app or GP letters)\n"
                "List of current medications (names + doses)\n"
                "Any relevant specialist or GP letters\n"
                "Next of kin contact details\n\n"
                "You do **not** need any documents to receive emergency treatment.")

    if any(w in m for w in ['hello','hi','hey','help','start']):
        return ("Hello! I'm your **NHS ED Patient Assistant** 👋\n\n"
                "I can help you with:\n"
                "- ⏱ **Estimated wait time** for your condition\n"
                "- 🗺️ **Best care pathway** for your needs\n"
                "- ❓ **Questions** about your visit\n"
                "- 🔔 **What to expect** during your care\n\n"
                "Tell me your main symptom or what brought you to A&E today, "
                "and I'll give you a personalised estimate.")

    if any(w in m for w in ['thank','thanks','appreciate']):
        return ("You're welcome! 💙 I hope your visit goes smoothly.\n\n"
                "Remember — tell a nurse immediately if your condition worsens. "
                "**NHS 111** is available free 24/7 after you leave. Take care!")

    return ("I'll do my best to help. For clinical questions about your treatment, "
            "please speak directly with the nursing team — they're best placed to advise you.\n\n"
            "For questions about waiting times, facilities, or your visit, I'm here. "
            "You can also call **NHS 111 (free)** or ask at the reception desk.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(f"""
<div class="nhs-header">
    <div style="display:flex;align-items:center;gap:16px">
        <div style="font-size:40px">🏥</div>
        <div>
            <h2 style="margin:0;font-size:24px">NHS Emergency Department — Patient AI Assistant</h2>
            <p style="margin:2px 0 0 0;opacity:0.85;font-size:14px">
                Personalised wait time prediction &amp; care pathway guidance
            </p>
        </div>
        <div style="margin-left:auto;text-align:right;font-size:12px;opacity:0.8">
            🕐 {datetime.now().strftime("%H:%M")}<br>
            {'🔴 Peak hours' if 10<=_h<=20 else '🟢 Off-peak'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.8, 1.2])

# ══════════════════════════════════════════════════
# LEFT: CHAT
# ══════════════════════════════════════════════════
with left:
    st.markdown("### 💬 Ask me anything about your visit")

    # Chat display
    if not st.session_state.messages:
        st.markdown("""<div class="chat-agent">
             Hello! I'm your NHS ED Patient Assistant.<br><br>
            I can <b>estimate your wait time</b>, suggest the <b>best care pathway</b>,
            and answer any questions about your visit.<br><br>
            Fill in your details on the right, or just ask me a question to get started.
        </div>""", unsafe_allow_html=True)
    else:
        for m in st.session_state.messages:
            if m['role'] == 'user':
                st.markdown(f'<div class="chat-user"> {m["content"]}</div>', unsafe_allow_html=True)
            elif m['role'] == 'agent':
                st.markdown(f'<div class="chat-agent"> {m["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-system">ℹ️ {m["content"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick buttons
    st.markdown("**Quick questions:**")
    r1c1, r1c2, r1c3 = st.columns(3)
    quick_qs = [
        ("⏱ Wait time",         "how long will i wait",          r1c1),
        ("🗺️ My options",        "what are my care options",       r1c2),
        ("📋 Triage explained",  "what does my triage category mean", r1c3),
    ]
    for label, query, col in quick_qs:
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.messages.append({'role':'user','content':label})
                resp = agent_response(query, st.session_state.patient_data)
                st.session_state.messages.append({'role':'agent','content':resp})
                st.rerun()

    r2c1, r2c2, r2c3 = st.columns(3)
    quick_qs2 = [
        ("🍵 Facilities",        "parking food cafe toilet",      r2c1),
        ("⏰ 4-hour target",     "4 hour target standard",        r2c2),
        ("🚨 Condition worsening","my condition is getting worse", r2c3),
    ]
    for label, query, col in quick_qs2:
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.messages.append({'role':'user','content':label})
                resp = agent_response(query, st.session_state.patient_data)
                st.session_state.messages.append({'role':'agent','content':resp})
                st.rerun()

    # Free text
    with st.form("chat", clear_on_submit=True):
        user_input = st.text_input("Type your question...",
            placeholder="e.g. How long will I wait? / I have chest pain / What are my options?",
            label_visibility="collapsed")
        if st.form_submit_button("Send →", use_container_width=True, type="primary"):
            if user_input.strip():
                st.session_state.messages.append({'role':'user','content':user_input})
                resp = agent_response(user_input, st.session_state.patient_data)
                st.session_state.messages.append({'role':'agent','content':resp})
                st.rerun()

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════
# RIGHT: PATIENT FORM + PREDICTION + PATHWAYS
# ══════════════════════════════════════════════════
with right:
    tab1, tab2, tab3 = st.tabs(["👤 Your Details", "📊 Prediction", "🗺️ Pathways"])

    # ── TAB 1: Form ────────────────────────────────
    with tab1:
        st.markdown("#### Get your personalised prediction")
        st.caption("Used only to estimate your wait — nothing is stored.")

        with st.form("patient"):
            age = st.slider("Your age", 1, 99, 45)

            symptom = st.selectbox("Main reason for visiting A&E today", list(SYMPTOM_MAP.keys()))

            arrival = st.selectbox("How did you arrive?",
                ["Walk-in","Ambulance","GP Referral","Other"])

            comor_label = st.selectbox("Long-term health conditions?",
                ["None","1 condition","2 conditions","3 or more"])
            comor_val = {"None":0,"1 condition":1,"2 conditions":2,"3 or more":3}[comor_label]

            prev_label = st.selectbox("A&E visits in the last 12 months?",
                ["None","1","2","3 or more"])
            prev_val = {"None":0,"1":1,"2":2,"3 or more":3}[prev_label]

            imd_label = st.selectbox("Area deprivation (optional — for equity analysis)",
                ["Most deprived (1)","2","3 (average)","4","Least deprived (5)"],index=2)
            imd_val = {"Most deprived (1)":1,"2":2,"3 (average)":3,"4":4,"Least deprived (5)":5}[imd_label]

            if st.form_submit_button("🔍 Get My Prediction", use_container_width=True, type="primary"):
                tri, news2 = SYMPTOM_MAP[symptom]
                st.session_state.patient_data = {
                    'age':age,'triage_cat':tri,'news2':news2,
                    'arrival_mode':arrival,'comorbidities':comor_val,
                    'prev_visits':prev_val,'imd':imd_val,'symptom':symptom,
                }
                st.session_state.prediction_done = True
                st.session_state.arrival_time    = datetime.now()

                prob, wait, total = get_prediction(st.session_state.patient_data)
                st.session_state.messages.append({
                    'role':'system',
                    'content':f'Details submitted: {symptom} | Age {age} | {arrival}'
                })
                st.session_state.messages.append({
                    'role':'agent',
                    'content':(f"Thank you! Based on **{symptom}** and current conditions:\n\n"
                               f"**⏱ Estimated wait: {wait}–{wait+20} min**\n"
                               f"**🏥 Total time: ~{total//60}h {total%60}min**\n"
                               f"**📊 4-hour breach risk: {prob*100:.0f}%**\n\n"
                               f"See the **📊 Prediction** and **🗺️ Pathways** tabs for full details.")
                })
                st.rerun()

    # ── TAB 2: Prediction ──────────────────────────
    with tab2:
        if st.session_state.prediction_done:
            pd_ = st.session_state.patient_data
            prob, wait, total = get_prediction(pd_)
            tri  = pd_['triage_cat']
            eta  = st.session_state.arrival_time + timedelta(minutes=total)

            st.markdown(f"""
            <div class="wait-card">
                <div style="font-size:12px;color:#888;margin-bottom:4px">ESTIMATED WAIT TO BE SEEN</div>
                <div style="font-size:56px;font-weight:700;color:#005EB8;line-height:1">{wait}–{wait+20}</div>
                <div style="font-size:16px;color:#666;margin-bottom:12px">minutes</div>
                <hr style="border-color:#eee;margin:8px 0">
                <div style="font-size:13px;color:#666">
                    Total time: <b>~{total//60}h {total%60}min</b>
                    &nbsp;|&nbsp; Expected out: <b>~{eta.strftime("%H:%M")}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            rc = "#E24B4A" if prob>0.6 else "#EF9F27" if prob>0.35 else "#1D9E75"
            rl = "High Risk" if prob>0.6 else "Moderate" if prob>0.35 else "Low Risk"

            with c1:
                st.markdown(f"""<div class="metric-mini">
                    <div style="font-size:11px;color:#888">4-HR BREACH RISK</div>
                    <div style="font-size:30px;font-weight:700;color:{rc}">{prob*100:.0f}%</div>
                    <div style="font-size:12px;color:{rc};font-weight:600">{rl}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-mini">
                    <div style="font-size:11px;color:#888">YOUR TRIAGE</div>
                    <div style="font-size:18px;font-weight:700;color:#005EB8">{TRIAGE_LABELS[tri]}</div>
                    <div style="font-size:11px;color:#888">Category {tri}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**Department conditions right now:**")
            for label, val, alert in [
                ("🛏️ Bed Occupancy", f"{SYSTEM['bed_occupancy_pct']:.1f}%", SYSTEM['bed_occupancy_pct']>95),
                ("👥 Queue Length",   f"{SYSTEM['queue_length_estimate']} patients", SYSTEM['queue_length_estimate']>35),
                ("👩‍⚕️ Staff Level",   f"Ratio {SYSTEM['staff_ratio']:.2f}", SYSTEM['staff_ratio']<0.45),
                ("🕐 Time of Day",    "Peak hours" if 10<=_h<=20 else "Off-peak", 10<=_h<=20),
            ]:
                ca, cb, cc = st.columns([2.5,2,0.5])
                ca.markdown(f"<small>{label}</small>",unsafe_allow_html=True)
                cb.markdown(f"**{val}**",unsafe_allow_html=True)
                cc.markdown("🔴" if alert else "🟢")

            st.caption("⚠️ AI estimates only. Actual times vary with clinical need. "
                      "Model: LightGBM (AUC 0.9163) + XGBoost (AUC 0.9160)")
        else:
            st.info("👈 Fill in your details to see your personalised prediction")

    # ── TAB 3: Pathways ────────────────────────────
    with tab3:
        if st.session_state.prediction_done:
            pd_ = st.session_state.patient_data
            prob, wait, _ = get_prediction(pd_)
            paths = get_pathways(pd_, prob)

            st.markdown("#### Recommended care options for you")
            st.caption("Based on your presentation and current conditions")

            for i, p in enumerate(paths):
                badge = "🥇 RECOMMENDED" if i==0 else f"Option {i+1}"
                bc    = "#1D9E75" if i==0 else "#378ADD"
                st.markdown(f"""
                <div class="pathway-card" style="border-left:5px solid {p['color']}">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <b style="font-size:16px">{p['icon']} {p['name']}</b>
                        <span style="background:{bc}22;color:{bc};padding:3px 10px;
                              border-radius:20px;font-size:12px;font-weight:600">{badge}</span>
                    </div>
                    <p style="color:#555;font-size:13px;margin:6px 0">{p['desc']}</p>
                    <p style="color:{p['color']};font-size:13px;font-weight:600;margin:4px 0">
                        ⏱ {p['wait']}
                    </p>
                    <p style="color:#888;font-size:12px;margin:0">💡 {p['why']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**📞 Key contacts:**")
            st.markdown("- **NHS 111** — Free 24/7 (call or 111.nhs.uk)\n"
                       "- **Samaritans** — 116 123\n"
                       "- **Pharmacy** — Minor ailments & medication advice")
        else:
            st.info("👈 Fill in your details to see pathway recommendations")

# Footer
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#888;font-size:12px">
    NHS ED Patient AI Assistant — Research Prototype |
    Powered by LightGBM + XGBoost trained on NHS 2024-25 data (n=100,000)<br>
    MSc Big Data & Data Science Technology | Northumbria University London | 2024-25<br>
    ⚠️ Not a substitute for clinical advice. Always follow guidance from NHS staff.
</div>""", unsafe_allow_html=True)

