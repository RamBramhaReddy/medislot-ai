import streamlit as st
import base64
import os
from utils.ui import inject_custom_css

st.set_page_config(page_title="Medical Dashboard", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMAGE_PATH = os.path.join(BASE_DIR, 'assets/images/calendar.jpeg')
set_background(BG_IMAGE_PATH)
inject_custom_css()

#  Hero Section 

st.markdown("""
<div style="text-align: center; padding: 4rem 0;">
    <h1 style="font-size: 4rem; margin-bottom: 0.2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">🏥 MediSlot AI</h1>
    <h3 style="font-size: 1.6rem; font-weight: 600; margin-top: 0; background: -webkit-linear-gradient(45deg, #38bdf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Intelligent Patient No-Show Prediction System</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Problem Statement , Project Objective

st.markdown("""
<style>
.ps-card, .po-card {
    padding: 2.2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    color: #f8fafc;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: transform 0.35s ease, box-shadow 0.35s ease;
    position: relative;
    overflow: hidden;
}
.ps-card {
    background: rgba(15,23,42,0.52);
    border: 1px solid rgba(56,189,248,0.2);
    border-left: 6px solid #38bdf8;
    box-shadow: 0 8px 32px rgba(56,189,248,0.12), 0 2px 8px rgba(0,0,0,0.4);
}
.po-card {
    background: rgba(15,23,42,0.52);
    border: 1px solid rgba(236,72,153,0.2);
    border-left: 6px solid #ec4899;
    box-shadow: 0 8px 32px rgba(236,72,153,0.12), 0 2px 8px rgba(0,0,0,0.4);
}
.ps-card:hover {
    transform: translateY(-8px) scale(1.015);
    box-shadow: 0 20px 60px rgba(56,189,248,0.25), 0 4px 16px rgba(0,0,0,0.5);
}
.po-card:hover {
    transform: translateY(-8px) scale(1.015);
    box-shadow: 0 20px 60px rgba(236,72,153,0.25), 0 4px 16px rgba(0,0,0,0.5);
}
.home-card-title {
    margin-top: 0;
    margin-bottom: 1.5rem;
    font-size: 1.75rem;
    font-weight: 700;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 1rem;
}
.stat-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    backdrop-filter: blur(6px);
}
</style>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="ps-card">
    <h2 class="home-card-title">🔍 Problem Statement</h2>
    <p style="font-size:1.12rem; line-height:1.85;">
        Every day, public clinics lose <strong style="color:#38bdf8;">20–30% of appointment slots</strong>
        because patients simply don't show up — no call, no cancellation, just an empty chair.
    </p>
    <p style="font-size:1.12rem; line-height:1.85; margin-top:1rem;">
        The moment a booking is confirmed, the clinic commits resources — a doctor is scheduled,
        a room reserved, nurses on standby. When the patient doesn't arrive, all of that is wasted.
        Meanwhile, another patient who urgently needed that slot was turned away.
    </p>
    <p style="font-size:1.12rem; line-height:1.85; margin-top:1rem;">
        The current system is <strong style="color:#f87171;">completely reactive</strong> —
        staff discover a no-show only after the slot is already gone. By then it is too late
        to fill it, contact a waitlisted patient, or recover any value.
    </p>
    <div style="margin-top:1.8rem;">
        <span class="stat-badge" style="background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.3); color:#7dd3fc;">🇧🇷 Brazil: 20–30% no-show rate</span>
        <span class="stat-badge" style="background:rgba(248,113,113,0.15); border:1px solid rgba(248,113,113,0.3); color:#fca5a5;">🇺🇸 US: $150B lost yearly</span>
        <span class="stat-badge" style="background:rgba(234,179,8,0.15); border:1px solid rgba(234,179,8,0.3); color:#fde68a;">🇬🇧 NHS: 1M slots wasted/month</span>
        <span class="stat-badge" style="background:rgba(34,197,94,0.15); border:1px solid rgba(34,197,94,0.3); color:#86efac;">🇮🇳 India: up to 40% OPD no-shows</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="po-card">
    <h2 class="home-card-title">🎯 Project Objective</h2>
    <p style="font-size:1.12rem; line-height:1.85;">
        Build a <strong style="color:#f9a8d4;">Machine Learning Binary Classification Model</strong>
        that predicts — at least 1–2 days before the appointment — whether a specific patient
        is likely to miss their appointment.
    </p>
    <p style="font-size:1.12rem; line-height:1.85; margin-top:1rem;">
        The model takes in patient details and appointment information as input and outputs
        a <strong style="color:#f9a8d4;">probability score between 0 and 1</strong>.
        This score tells the clinic exactly how risky each appointment is before the day arrives.
    </p>
    <ul style="list-style:none; padding-left:0; margin-top:1.5rem;">
        <li style="margin-bottom:1rem; font-size:1.08rem; line-height:1.7;">🔴 <strong>Score &gt; 0.7</strong> — High Risk: Call personally, offer slot to waitlist</li>
        <li style="margin-bottom:1rem; font-size:1.08rem; line-height:1.7;">🟡 <strong>Score 0.4–0.7</strong> — Medium Risk: Send extra WhatsApp or SMS reminder</li>
        <li style="margin-bottom:1rem; font-size:1.08rem; line-height:1.7;">🟢 <strong>Score &lt; 0.4</strong> — Low Risk: Standard SMS is enough, no extra action</li>
    </ul>
    <p style="font-size:1.12rem; line-height:1.85; margin-top:1.5rem;">
        The goal is to shift the clinic from a <strong style="color:#f87171;">reactive</strong> system
        to a <strong style="color:#86efac;">proactive</strong> one — acting before the no-show happens.
    </p>
    </div>
    """, unsafe_allow_html=True)    