import streamlit as st
import os

def inject_custom_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
