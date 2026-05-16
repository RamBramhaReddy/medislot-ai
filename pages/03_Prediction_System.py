import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import sys

st.set_page_config(page_title="MediSlot Intelligence Hub", page_icon="🧠", layout="wide")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ui import inject_custom_css

def set_background(image_path):
    import base64
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

# Setup Background  

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG_IMAGE_PATH = os.path.join(BASE_DIR, 'assets/images/appointments.jpeg')
set_background(BG_IMAGE_PATH)
inject_custom_css()

st.markdown("<h1 style='text-align: center; margin-bottom: 0.2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>🧠 MediSlot Intelligence Hub</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.35rem; font-weight: 600; margin-top: 0; margin-bottom: 3rem; background: -webkit-linear-gradient(45deg, #38bdf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Advanced predictive analytics for proactive healthcare scheduling.</p>", unsafe_allow_html=True)

MODEL_PATH = os.path.join(BASE_DIR, 'noshow_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'Medical Appointment.csv')

@st.cache_data
def get_neighbourhoods():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, usecols=['Neighbourhood'])
        counts = df['Neighbourhood'].value_counts()
        top = counts[counts >= 100].index.tolist()
        top.append('Other')
        return sorted(top)
    return ['Other']

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()
neighbourhoods = get_neighbourhoods()

if model is None:
    st.error("Model not found! Please run `ML.py` to train the model first.")
else:
    col_form, col_space, col_result = st.columns([1.5, 0.1, 1])
    
    with col_form:
        st.markdown("<h3 style='border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem; margin-bottom: 1rem;'>📝 Enter Patient Details</h3>", unsafe_allow_html=True)
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("👤 Age", min_value=0, max_value=120, value=30)
                gender = st.selectbox("🚻 Gender", ["Female", "Male"])
                neighbourhood = st.selectbox("📍 Neighbourhood", neighbourhoods)
                handicap = st.selectbox("♿ Handicap Level", [0, 1, 2, 3, 4])
                
            with c2:
                scheduled_date = st.date_input("📅 Scheduled Date", datetime.date.today(), help="The date the patient called to book.")
                scheduled_time = st.time_input("⏰ Scheduled Time", datetime.time(9, 0))
                appointment_date = st.date_input("🏥 Appointment Date", datetime.date.today() + datetime.timedelta(days=7), help="The actual date of the medical appointment.")
                prev_noshow = st.slider("📉 Previous No-Show Rate", 0.0, 1.0, 0.0, 0.05, help="In a real system, this is auto-calculated from the EHR database: (Past No-Shows / Total Past Appointments). For this demo, simulate their history.")
                
            st.markdown("<br>**Health Conditions & Programs**", unsafe_allow_html=True)
            c3, c4, c5, c6 = st.columns(4)
            with c3: scholarship = st.checkbox("Scholarship (Bolsa Família)")
            with c4: hypertension = st.checkbox("Hypertension")
            with c5: diabetes = st.checkbox("Diabetes")
            with c6: alcoholism = st.checkbox("Alcoholism")
            
            st.markdown("<br>", unsafe_allow_html=True)
            sms_received = st.checkbox("📱 Patient Received SMS Reminder")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("🚀 Run Prediction Analysis")

    with col_result:
        st.markdown("<h3 style='border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem; margin-bottom: 1rem;'>📊 Analysis Result</h3>", unsafe_allow_html=True)
        if submit:
            sched_dt = datetime.datetime.combine(scheduled_date, scheduled_time)
            app_dt = datetime.datetime.combine(appointment_date, datetime.time(0, 0))
            
            waiting_days = (app_dt.date() - sched_dt.date()).days
            if waiting_days < 0:
                st.error("Appointment date cannot be before scheduled date!")
            else:
                if waiting_days == 0 and sms_received:
                    sms_received = False
                    
                sched_hour = sched_dt.hour
                app_dow = app_dt.weekday()
                
                if age <= 12: age_group = 'Child'
                elif age <= 17: age_group = 'Teen'
                elif age <= 35: age_group = 'YoungAdult'
                elif age <= 55: age_group = 'Adult'
                else: age_group = 'Senior'
                
                has_condition = int(hypertension or diabetes or alcoholism)
                
                input_data = pd.DataFrame([{
                    'Age': age, 'WaitingDays': waiting_days, 'ScheduledHour': sched_hour,
                    'PreviousNoShowRate': prev_noshow, 'AppointmentDayOfWeek': app_dow,
                    'Neighbourhood': neighbourhood, 'AgeGroup': age_group,
                    'Gender': 1 if gender == 'Male' else 0, 'Scholarship': int(scholarship),
                    'Hypertension': int(hypertension), 'Diabetes': int(diabetes),
                    'Alcoholism': int(alcoholism), 'Handicap': handicap,
                    'SMS_received': int(sms_received), 'HasCondition': has_condition
                }])
                
                prob = model.predict_proba(input_data)[0, 1]
                pred = model.predict(input_data)[0]
                
                if pred == 1:
                    bg_color = "rgba(239, 68, 68, 0.15)"
                    border_color = "#ef4444"
                    icon = "⚠️"
                    text = "High Risk of No-Show"
                else:
                    bg_color = "rgba(34, 197, 94, 0.15)"
                    border_color = "#22c55e"
                    icon = "✅"
                    text = "Likely to Attend"
                    
                st.markdown(f"""
                <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 16px; padding: 3rem 2rem; text-align: center; margin-top: 0.5rem; backdrop-filter: blur(10px); box-shadow: 0 10px 30px {bg_color};">
                    <div style="font-size: 5rem; line-height: 1;">{icon}</div>
                    <h2 style="color: {border_color}; margin-top: 1rem; font-size: 2.2rem;">{text}</h2>
                    <p style="font-size: 1.2rem; color: #cbd5e1; margin-bottom: 1rem;">Probability of missing appointment:</p>
                    <h1 style="font-size: 4.5rem; margin: 0; color: {border_color};">{prob * 100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar
                
                st.markdown(f"""
                <div style="width: 100%; background-color: rgba(255,255,255,0.05); border-radius: 8px; margin-top: 2.5rem; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="width: {prob*100}%; height: 16px; background-color: {border_color}; border-radius: 8px; transition: width 1s ease-in-out; box-shadow: 0 0 10px {border_color};"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Actionable Recommendations

                st.markdown("<br><h4 style='color: #f8fafc; margin-bottom: 0.5rem; text-shadow: 1px 1px 0px #0ea5e9, 2px 2px 5px rgba(0, 0, 0, 0.6);'>💡 Prescriptive Actions</h4>", unsafe_allow_html=True)
                
                if prob >= 0.70:
                    rec_html = """<ul style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.6; margin-bottom: 0; padding-left: 20px;">
    <li style="margin-bottom: 8px;">📞 <b>Personal Call:</b> Have staff call directly to confirm.</li>
    <li style="margin-bottom: 8px;">📅 <b>Double Booking:</b> Consider double-booking this slot to prevent revenue loss.</li>
    <li>🚗 <b>Alternatives:</b> Offer a telehealth option or transportation voucher.</li>
</ul>"""
                elif prob >= 0.35:
                    rec_html = """<ul style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.6; margin-bottom: 0; padding-left: 20px;">
    <li style="margin-bottom: 8px;">💬 <b>WhatsApp:</b> Send a message requiring a direct reply "YES" or "NO" to confirm.</li>
    <li>📱 <b>Double SMS:</b> Ensure automated SMS reminders are sent 48hrs AND 24hrs prior.</li>
</ul>"""
                else:
                    rec_html = """<ul style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.6; margin-bottom: 0; padding-left: 20px;">
    <li>✅ <b>Automated Pipeline:</b> No manual intervention needed. Standard automated SMS reminder is sufficient.</li>
</ul>"""
                    
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 1.5rem; margin-top: 0.5rem; box-shadow: inset 0px 1px 1px rgba(255, 255, 255, 0.1), 0px 4px 15px rgba(0,0,0,0.5);">
                    {rec_html}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.4); border: 2px dashed rgba(255,255,255,0.1); border-radius: 16px; padding: 4rem 2rem; text-align: center; margin-top: 0.5rem; backdrop-filter: blur(10px);">
                <div style="font-size: 4rem; opacity: 0.5;">👆</div>
                <h3 style="color: #94a3b8; font-weight: 400; margin-top: 1.5rem; line-height: 1.5;">Fill out the patient form and run the analysis to view the prediction here.</h3>
            </div>
            """, unsafe_allow_html=True)
