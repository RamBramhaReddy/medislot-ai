import streamlit as st
import base64
import os
import sys

st.set_page_config(page_title="Business Problem", page_icon="💼", layout="wide")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ui import inject_custom_css

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG_IMAGE_PATH = os.path.join(BASE_DIR, 'assets/images/healthcare_bg.jpeg')
set_background(BG_IMAGE_PATH)
inject_custom_css()

st.markdown("""
<style>
.info-card, .pink-card, .yellow-card, .green-card {
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: #f8fafc;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    height: 100%;
    min-height: 480px;
}

.info-card:hover, .pink-card:hover, .yellow-card:hover, .green-card:hover {
    transform: translateY(-5px);
}

.info-card { background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(56, 189, 248, 0.2); border-left: 6px solid #38bdf8; }
.info-card:hover { box-shadow: 0 8px 32px 0 rgba(56, 189, 248, 0.2); }

.pink-card { background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(236, 72, 153, 0.2); border-left: 6px solid #ec4899; }
.pink-card:hover { box-shadow: 0 8px 32px 0 rgba(236, 72, 153, 0.2); }

.yellow-card { background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(234, 179, 8, 0.2); border-left: 6px solid #eab308; }
.yellow-card:hover { box-shadow: 0 8px 32px 0 rgba(234, 179, 8, 0.2); }

.green-card { background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(34, 197, 94, 0.2); border-left: 6px solid #22c55e; }
.green-card:hover { box-shadow: 0 8px 32px 0 rgba(34, 197, 94, 0.2); }

.card-title {
    margin-top: 0;
    margin-bottom: 1.5rem;
    font-size: 1.8rem;
    font-weight: 600;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 1rem;
}

ul { list-style-type: none; padding-left: 0; }
li { margin-bottom: 1rem; font-size: 1.1rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; margin-bottom: 3rem;'>🚑 Understanding the Missing Appointments Crisis</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
    <h2 class="card-title">💼 Business Problem</h2>
    <p style="font-size: 1.15rem; line-height: 1.8;">
    Public clinics waste <strong style="color:#38bdf8;">20–30% of appointment slots</strong> daily
    because patients don't show up without cancelling. Doctors wait, rooms sit empty, and other
    patients who needed those slots were turned away.
    <br><br>
    The clinic only finds out after the slot is gone — too late to call a waitlisted patient,
    too late to reassign the doctor, too late to recover any value.
    <br><br>
    The system is <strong style="color:#f87171;">completely reactive</strong>. Every patient gets
    the same generic reminder regardless of their risk level. There is no intelligence,
    no personalization, and no way to act before the problem happens.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="yellow-card">
    <h2 class="card-title">💡 The Healthcare Challenge</h2>
    <ul>
    <li>⏳ Large numbers of appointment slots wasted daily due to no-shows</li>
    <li>💰 Financial losses caused by underutilized medical resources</li>
    <li>👨‍⚕️ Idle doctor and staff time reducing operational productivity</li>
    <li>📋 Long waiting lists while reserved slots remain unused</li>
    <li>📉 Lack of proactive systems for personalized patient follow-ups</li>
    </ul>
    <p style="font-size: 1.15rem; margin-top: 2rem; font-weight: 600; color: #fcd34d;">
    Healthcare organizations must shift from reactive appointment management toward proactive
    and data-driven patient attendance prediction systems.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="pink-card">
    <h2 class="card-title">🌍 Background</h2>
    <p style="font-size: 1.15rem; line-height: 1.8;">
    In 2016, researchers collected <strong style="color:#f9a8d4;">110,527 real appointment records</strong>
    from public clinics in Vitoria, Brazil. The data revealed a clear pattern — certain types of patients
    consistently missed appointments at higher rates.
    <br><br>
    Patients with long waiting times, no chronic conditions, welfare recipients, and young adults
    were far more likely to miss their appointments than others.
    <br><br>
    This dataset became the foundation for building a predictive system that could identify
    high-risk patients <strong style="color:#f9a8d4;">before their appointment day</strong>,
    giving clinics time to act proactively — not reactively.
    </p>
    <ul style="margin-top:1rem;">
    <li>📩 Send the same generic reminder to every patient</li>
    <li>⚠️ Fail to identify patients with high no-show risk</li>
    <li>📉 Detect missed appointments only after the slot is wasted</li>
    <li>🚫 Lack personalized follow-up systems for vulnerable patients</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="green-card">
    <h2 class="card-title">🚧 Constraints & Considerations</h2>
    <ul>
    <li>🧪 <b style="color: #86efac;">Behavioral Uncertainty:</b> Patient attendance behavior can change due to personal, medical, or economic reasons.</li>
    <li>⚖️ <b style="color: #86efac;">Class Imbalance:</b> Most patients attend appointments while only a smaller percentage are no-shows, creating imbalance challenges for ML models.</li>
    <li>📊 <b style="color: #86efac;">Data Quality Issues:</b> Real-world healthcare datasets may contain missing values, incorrect ages, and inconsistent appointment records.</li>
    <li>🚀 <b style="color: #86efac;">Real-Time Prediction:</b> The system must generate fast predictions to support proactive clinic scheduling decisions.</li>
    <li>🩺 <b style="color: #86efac;">Decision Support:</b> Model predictions should assist healthcare staff in making smarter scheduling decisions, not replace human medical judgment.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)