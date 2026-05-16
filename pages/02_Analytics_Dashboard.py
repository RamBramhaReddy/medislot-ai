import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ui import inject_custom_css

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
plt.style.use('dark_background')
from cycler import cycler
plt.rcParams.update({
    "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),  
    "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),  
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    "axes.edgecolor": (1.0, 1.0, 1.0, 0.1),
    "axes.grid": True,
    "grid.color": (1.0, 1.0, 1.0, 0.15),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#e2e8f0",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "axes.prop_cycle": cycler(color=['#6366f1', '#f472b6', '#14b8a6', '#f59e0b', '#a78bfa'])
})

inject_custom_css()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSIGHTS_PATH = os.path.join(BASE_DIR, 'insights.json')
DATA_PATH = os.path.join(BASE_DIR, 'medical_appointment.csv')

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

set_background(os.path.join(BASE_DIR, 'assets/images/medical_dashboard.jpeg'))


with open(INSIGHTS_PATH, 'r') as f:
    insights = json.load(f)

# load & prepare data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap',
        'No-show': 'NoShow'
    }, inplace=True)

    df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    df['NoShow'] = df['NoShow'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

    df['WaitingDays'] = (
        df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()
    ).dt.days

    df = df[df['Age'] >= 0]
    df = df[df['WaitingDays'] >= 0]
    df.reset_index(drop=True, inplace=True)

    df['ScheduledHour']        = df['ScheduledDay'].dt.hour
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.dayofweek

    df = df.sort_values('AppointmentDay').reset_index(drop=True)
    df['PreviousNoShowRate'] = df.groupby('PatientId')['NoShow'].transform(
        lambda x: x.shift().expanding().mean()
    ).fillna(0)

    bins   = [-1, 12, 17, 35, 55, 120]
    labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

    df['HasCondition'] = (
        (df['Hypertension'] == 1) | (df['Diabetes'] == 1) | (df['Alcoholism'] == 1)
    ).astype(int)

    top = df['Neighbourhood'].value_counts()[df['Neighbourhood'].value_counts() >= 100].index
    df['Neighbourhood'] = df['Neighbourhood'].apply(lambda x: x if x in top else 'Other')

    df['WaitBucket'] = pd.cut(
        df['WaitingDays'],
        bins=[-1, 0, 7, 30, 60, 180],
        labels=['Same Day', '1-7', '8-30', '31-60', '60+']
    )

    return df

df = load_data()

def style_df(data):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    return data.style.set_properties(**{
        'background-color': 'rgba(15, 23, 42, 0.4)',
        'color': '#f8fafc',
        'border': '1px solid rgba(56, 189, 248, 0.15)',
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'rgba(15, 23, 42, 0.95)'), ('color', '#60a5fa'), ('border-bottom', '2px solid #38bdf8')]},
        {'selector': 'tr:hover td', 'props': [('background-color', 'rgba(56, 189, 248, 0.2) !important'), ('color', '#ffffff !important')]}
    ])

# page config 
st.markdown("<h2 style='text-align: center; font-size: 2.2rem; margin-bottom: 1rem; color: #f8fafc; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>🏥 Reducing Appointment Wastage Through Data-Driven Healthcare Analytics</h2>", unsafe_allow_html=True)
st.markdown('---')

section = st.sidebar.radio('Navigate', [
    "Dataset Intelligence",
    "Pattern Discovery",
    "Relationship Analysis",
    "Behavioral Insights",
    "Feature Correlation Map"
])



# 1 Data Overview

if section == 'Dataset Intelligence':
    st.header('Data Overview')

    st.subheader('Head')
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.dataframe(style_df(df.head()), use_container_width=True)

    st.subheader('Shape')
    st.write(f'**Rows:** {df.shape[0]:,}   |   **Columns:** {df.shape[1]}')

    st.subheader('Info')
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.notnull().sum().values,
        'Dtype': df.dtypes.astype(str).values
    })
    col_t3, col_t4 = st.columns([2, 1])
    with col_t3:
        st.dataframe(style_df(info_df), use_container_width=True)

    st.subheader('Describe')
    col_t5, col_t6 = st.columns([3, 1])
    with col_t5:
        st.dataframe(style_df(df.describe()), use_container_width=True)

    st.subheader('Null Check')
    col_t7, col_t8 = st.columns([1.5, 2])
    with col_t7:
        st.dataframe(style_df(df.isnull().sum().rename('Null Count')), use_container_width=True)

    st.subheader('Duplicates')
    st.write(f'**Duplicate rows:** {df.duplicated().sum()}')



# 2 — Univariate Analysis

elif section == 'Pattern Discovery':
    st.header('Univariate Analysis')

    col_type = st.selectbox('Column Type', ['Numeric', 'Categorical'])

    if col_type == 'Numeric':
        col = st.selectbox('Select Column', [
            'Age', 'WaitingDays', 'ScheduledHour',
            'AppointmentDayOfWeek', 'PreviousNoShowRate'
        ])

        fig, ax = plt.subplots(figsize=(10, 5))

        if col == 'AppointmentDayOfWeek':
            df[col].value_counts().sort_index().plot(
                kind='bar', ax=ax, color='#a78bfa', edgecolor=(1.0, 1.0, 1.0, 0.3), title=col)
            ax.set_xticks(range(7))
            ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
        else:
            df[col].plot(kind='hist', bins=30, ax=ax, edgecolor=(1.0, 1.0, 1.0, 0.3), title=col)

        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        key_map = {
            'Age': 'age',
            'WaitingDays': 'waiting_days',
            'ScheduledHour': 'scheduled_hour',
            'AppointmentDayOfWeek': 'day_of_week',
            'PreviousNoShowRate': 'previous_noshow_rate'
        }
        with st.expander('📌 Insight'):
            st.write(insights['univariate'][key_map[col]])

    elif col_type == 'Categorical':
        col = st.selectbox('Select Column', [
            'Gender', 'Scholarship', 'Hypertension', 'Diabetes',
            'Alcoholism', 'Handicap', 'SMS_received', 'NoShow',
            'AgeGroup', 'HasCondition'
        ])

        fig, ax = plt.subplots(figsize=(10, 5))
        df[col].value_counts().plot(
            kind='bar', ax=ax, color='#60a5fa', edgecolor=(1.0, 1.0, 1.0, 0.3), title=col)
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        key_map = {
            'Gender': 'gender', 'Scholarship': 'scholarship',
            'Hypertension': 'hypertension', 'Diabetes': 'diabetes',
            'Alcoholism': 'alcoholism', 'Handicap': 'handicap',
            'SMS_received': 'sms_received', 'NoShow': 'noshow',
            'AgeGroup': 'agegroup', 'HasCondition': 'hascondition'
        }
        with st.expander('📌 Insight'):
            st.write(insights['univariate'][key_map[col]])



# 3 — Bivariate Analysis

elif section == 'Relationship Analysis':
    st.header('Bivariate Analysis')

    biv_type = st.selectbox('Analysis Type', [
        'Numeric vs Numeric',
        'Numeric vs NoShow',
        'Categorical vs NoShow',
        'Neighbourhood No-show Rate',
        'NoShow by Waiting Days Bucket'
    ])

    #  Num vs Num

    if biv_type == 'Numeric vs Numeric':
        pair = st.selectbox('Select Pair', [
            'Age vs WaitingDays',
            'Age vs PreviousNoShowRate',
            'WaitingDays vs PreviousNoShowRate'
        ])

        fig, ax = plt.subplots(figsize=(10, 5))

        if pair == 'Age vs WaitingDays':
            ax.scatter(df['Age'], df['WaitingDays'], alpha=0.1, color='#60a5fa')
            ax.set_xlabel('Age')
            ax.set_ylabel('WaitingDays')
            key = 'age_vs_waitingdays'

        elif pair == 'Age vs PreviousNoShowRate':
            ax.scatter(df['Age'], df['PreviousNoShowRate'], alpha=0.1, color='#f472b6')
            ax.set_xlabel('Age')
            ax.set_ylabel('PreviousNoShowRate')
            key = 'age_vs_previousnoshowrate'

        elif pair == 'WaitingDays vs PreviousNoShowRate':
            ax.scatter(df['WaitingDays'], df['PreviousNoShowRate'], alpha=0.1, color='#34d399')
            ax.set_xlabel('WaitingDays')
            ax.set_ylabel('PreviousNoShowRate')
            key = 'waitingdays_vs_previousnoshowrate'

        ax.set_title(pair)
        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        with st.expander('📌 Insight'):
            st.write(insights['bivariate']['num_num'][key])

    #  Num vs NoShow
    elif biv_type == 'Numeric vs NoShow':
        col = st.selectbox('Select Column', [
            'Age', 'WaitingDays', 'ScheduledHour', 'PreviousNoShowRate'
        ])

        fig, ax = plt.subplots(figsize=(10, 5))
        df.groupby('NoShow')[col].plot(kind='hist', bins=30, alpha=0.5, ax=ax)
        ax.set_title(f'{col} by NoShow')
        ax.set_xlabel(col)
        ax.legend(['Showed Up', 'No Show'])
        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        key_map = {
            'Age': 'age_by_noshow',
            'WaitingDays': 'waitingdays_by_noshow',
            'ScheduledHour': 'scheduledhour_by_noshow',
            'PreviousNoShowRate': 'previousnoshowrate_by_noshow'
        }
        with st.expander('📌 Insight'):
            st.write(insights['bivariate']['num_cat'][key_map[col]])

    #  Cat vs NoShow
    
    elif biv_type == 'Categorical vs NoShow':
        col = st.selectbox('Select Column', [
            'Gender', 'Scholarship', 'Hypertension', 'Diabetes',
            'Alcoholism', 'SMS_received', 'HasCondition', 'AgeGroup'
        ])

        fig, ax = plt.subplots(figsize=(10, 5))
        ct = pd.crosstab(df[col], df['NoShow'])
        ct.plot(kind='bar', ax=ax, edgecolor=(1.0, 1.0, 1.0, 0.3), title=f'NoShow by {col}')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(['Showed Up', 'No Show'])
        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        key_map = {
            'Gender': 'gender_by_noshow',
            'Scholarship': 'scholarship_by_noshow',
            'Hypertension': 'hypertension_by_noshow',
            'Diabetes': 'diabetes_by_noshow',
            'Alcoholism': 'alcoholism_by_noshow',
            'SMS_received': 'sms_by_noshow',
            'HasCondition': 'hascondition_by_noshow',
            'AgeGroup': 'agegroup_by_noshow'
        }
        with st.expander('📌 Insight'):
            st.write(insights['bivariate']['cat_cat'][key_map[col]])

    #   Neighbourhood

    elif biv_type == 'Neighbourhood No-show Rate':
        neighbourhood_noshow = pd.crosstab(df['Neighbourhood'], df['NoShow'])
        neighbourhood_noshow['rate'] = neighbourhood_noshow[1] / neighbourhood_noshow.sum(axis=1) * 100
        valid    = df['Neighbourhood'].value_counts()[df['Neighbourhood'].value_counts() >= 50].index
        filtered = neighbourhood_noshow[neighbourhood_noshow.index.isin(valid)]

        fig, ax = plt.subplots(figsize=(10, 6))
        filtered['rate'].sort_values(ascending=True).tail(10).plot(
            kind='barh', ax=ax, color='#f472b6', edgecolor=(1.0, 1.0, 1.0, 0.3),
            title='Top 10 Neighbourhoods by No-show Rate'
        )
        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        with st.expander('📌 Insight'):
            st.write(insights['bivariate']['neighbourhood'])

    #  WaitBucket 
    elif biv_type == 'NoShow by Waiting Days Bucket':
        fig, ax = plt.subplots(figsize=(10, 5))
        pd.crosstab(df['WaitBucket'], df['NoShow']).plot(kind='bar', ax=ax, edgecolor=(1.0, 1.0, 1.0, 0.3))
        ax.set_xticklabels(['Same Day', '1-7', '8-30', '31-60', '60+'], rotation=0)
        ax.legend(['Showed Up', 'No Show'])
        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        with st.expander('📌 Insight'):
            st.write(insights['bivariate']['waitbucket'])



# 4 — Multivariate Analysis

elif section == 'Behavioral Insights':
    st.header('Multivariate Analysis')

    choice = st.selectbox('Select Analysis', [
        'SMS Received by Wait Bucket',
        'Scholarship by AgeGroup'
    ])

    if choice == 'SMS Received by Wait Bucket':
        ct = pd.crosstab(df['WaitBucket'], df['SMS_received'])
        fig, ax = plt.subplots(figsize=(10, 5))
        ct.div(ct.sum(axis=1), axis=0).plot(
            kind='bar', ax=ax, edgecolor=(1.0, 1.0, 1.0, 0.3), title='SMS Received by WaitBucket')
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        with st.expander('📌 Insight'):
            st.write(insights['multivariate']['sms_by_waitbucket'])

    elif choice == 'Scholarship by AgeGroup':
        fig, ax = plt.subplots(figsize=(10, 5))
        pd.crosstab(df['AgeGroup'], df['Scholarship']).plot(
            kind='bar', ax=ax, edgecolor=(1.0, 1.0, 1.0, 0.3), title='Scholarship by AgeGroup')
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        col_c1, col_c2 = st.columns([2.5, 1])
        with col_c1:
            st.pyplot(fig)

        with st.expander('📌 Insight'):
            st.write(insights['multivariate']['scholarship_by_agegroup'])



# 5 — Correlation Heatmap

elif section == 'Feature Correlation Map':
    st.header('Correlation Heatmap')

    num_cols = ['Age', 'WaitingDays', 'ScheduledHour', 'AppointmentDayOfWeek', 'PreviousNoShowRate']
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.tight_layout()
    col_c1, col_c2 = st.columns([2.5, 1])
    with col_c1:
        st.pyplot(fig)

    with st.expander('📌 Insight'):
        st.write(insights['correlation'])