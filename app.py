# ============================================
# ONLINE FOOD ORDER PREDICTOR - COMPLETE APP
# Single File: app.py
# All 16 Steps in One Place
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="🍔 Online Food Order Predictor",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - MODERN GLASSMORPHISM WITH VIBRANT COLORS
# ============================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', 'Space Grotesk', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Animated background with gradient mesh */
    .stApp {
        background: 
            radial-gradient(circle at 0% 0%, rgba(120, 80, 255, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 100% 0%, rgba(255, 80, 200, 0.12) 0%, transparent 50%),
            radial-gradient(circle at 50% 100%, rgba(0, 200, 255, 0.10) 0%, transparent 50%),
            radial-gradient(circle at 80% 50%, rgba(255, 200, 50, 0.08) 0%, transparent 40%),
            linear-gradient(145deg, #0a0e1a 0%, #1a1a2e 30%, #16213e 60%, #0f3460 100%);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Animated floating particles effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, rgba(255,255,255,0.3), transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.2), transparent),
            radial-gradient(2px 2px at 50px 160px, rgba(255,255,255,0.3), transparent),
            radial-gradient(2px 2px at 90px 40px, rgba(255,255,255,0.2), transparent),
            radial-gradient(2px 2px at 130px 80px, rgba(255,255,255,0.3), transparent);
        background-size: 200px 200px;
        animation: float 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(2deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    /* Main content area - Premium Glassmorphism */
    .main > div {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(24px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
        border-radius: 40px !important;
        padding: 2.5rem !important;
        margin: 1.2rem 0 !important;
        box-shadow: 
            0 30px 80px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.15),
            inset 0 -1px 0 rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative;
        z-index: 1;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main > div:hover {
        box-shadow: 
            0 40px 100px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 60px rgba(100, 80, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-2px);
    }
    
    /* Sidebar - Premium Dark Glass */
    .css-1d391kg, .st-emotion-cache-1d391kg {
        background: rgba(10, 14, 30, 0.7) !important;
        backdrop-filter: blur(28px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(200%) !important;
        border-radius: 40px !important;
        padding: 2rem 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        box-shadow: 
            0 25px 60px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        margin: 1.2rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .css-1d391kg:hover, .st-emotion-cache-1d391kg:hover {
        box-shadow: 
            0 30px 70px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            0 0 80px rgba(100, 80, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        transform: translateX(4px);
    }
    
    /* Sidebar Radio buttons - Premium */
    .stRadio > div {
        gap: 8px !important;
    }
    
    .stRadio label {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 20px !important;
        padding: 12px 18px !important;
        margin: 3px 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        color: rgba(255, 255, 255, 0.6) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(4px) !important;
        display: flex !important;
        align-items: center !important;
        gap: 14px !important;
        letter-spacing: 0.2px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stRadio label::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(100, 80, 255, 0.1), rgba(255, 80, 200, 0.05));
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: 20px;
    }
    
    .stRadio label:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(100, 80, 255, 0.2) !important;
        transform: translateX(8px) scale(1.02) !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(100, 80, 255, 0.15) !important;
    }
    
    .stRadio label:hover::before {
        opacity: 1;
    }
    
    .stRadio label[data-baseweb="radio"] {
        background: linear-gradient(135deg, rgba(100, 80, 255, 0.2), rgba(255, 80, 200, 0.15)) !important;
        border-color: rgba(100, 80, 255, 0.3) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 30px rgba(100, 80, 255, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        transform: translateX(4px) !important;
    }
    
    .stRadio label[data-baseweb="radio"]::before {
        opacity: 1;
        background: linear-gradient(135deg, rgba(100, 80, 255, 0.2), rgba(255, 80, 200, 0.1));
    }
    
    /* Headers - Animated Gradient */
    h1, h2, h3, h4 {
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        position: relative !important;
    }
    
    h1 {
        font-size: 3.2rem !important;
        background: linear-gradient(135deg, #818cf8, #a78bfa, #f472b6, #fb923c) !important;
        background-size: 300% 300% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        animation: gradientMove 4s ease-in-out infinite !important;
        margin-bottom: 0.3rem !important;
        text-shadow: 0 0 80px rgba(100, 80, 255, 0.2) !important;
    }
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h2 {
        font-size: 2.4rem !important;
        background: linear-gradient(135deg, #c084fc, #818cf8, #60a5fa) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: 0 0 60px rgba(100, 80, 255, 0.15) !important;
    }
    
    h3 {
        font-size: 1.6rem !important;
        background: linear-gradient(135deg, #f472b6, #fb923c) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    /* Metrics - Premium Glass Cards with Neon Glow */
    .css-1r6slb0, .st-emotion-cache-1r6slb0 {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 28px !important;
        padding: 1.5rem 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: center !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .css-1r6slb0::before, .st-emotion-cache-1r6slb0::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(100, 80, 255, 0.05), transparent, rgba(255, 80, 200, 0.05), transparent);
        animation: rotate 6s linear infinite;
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .css-1r6slb0:hover::before, .st-emotion-cache-1r6slb0:hover::before {
        opacity: 1;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .css-1r6slb0:hover, .st-emotion-cache-1r6slb0:hover {
        transform: translateY(-6px) scale(1.02) !important;
        background: rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 20px 60px rgba(100, 80, 255, 0.15) !important;
        border-color: rgba(100, 80, 255, 0.2) !important;
    }
    
    .css-1r6slb0 .css-1xarl3l, .st-emotion-cache-1r6slb0 .st-emotion-cache-1xarl3l {
        color: #ffffff !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        text-shadow: 0 0 40px rgba(100, 80, 255, 0.3) !important;
    }
    
    .css-1r6slb0 .css-1d391kg, .st-emotion-cache-1r6slb0 .st-emotion-cache-1d391kg {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
    }
    
    /* Buttons - Premium Gradient with Glow */
    .stButton > button {
        background: linear-gradient(145deg, #7c3aed, #6d28d9, #5b21b6) !important;
        background-size: 200% 200% !important;
        border: none !important;
        border-radius: 60px !important;
        padding: 14px 36px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: white !important;
        backdrop-filter: blur(4px) !important;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
        animation: buttonPulse 3s ease-in-out infinite !important;
    }
    
    @keyframes buttonPulse {
        0%, 100% { box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3); }
        50% { box-shadow: 0 12px 48px rgba(124, 58, 237, 0.5); }
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) translateY(-4px) !important;
        box-shadow: 0 16px 48px rgba(124, 58, 237, 0.5) !important;
        background: linear-gradient(145deg, #8b5cf6, #7c3aed, #6d28d9) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    .stButton > button:hover::before {
        opacity: 1;
    }
    
    .stButton > button:active {
        transform: scale(0.96) !important;
    }
    
    /* Select boxes - Glass with Neon Border */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 40px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.15), 0 4px 20px rgba(124, 58, 237, 0.1) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(124, 58, 237, 0.3) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Sliders - Premium */
    .stSlider > div > div {
        border-radius: 40px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(4px) !important;
        height: 6px !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #ec4899, #f59e0b) !important;
        border-radius: 40px !important;
        height: 6px !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3) !important;
    }
    
    .stSlider > div > div > div > div {
        background: white !important;
        border: 2px solid #7c3aed !important;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.3) !important;
        width: 20px !important;
        height: 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .stSlider > div > div > div > div:hover {
        transform: scale(1.2) !important;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.4) !important;
    }
    
    /* Expanders - Premium Glass */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        padding: 1rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(124, 58, 237, 0.2) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.05) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border-radius: 0 0 30px 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding: 1.5rem !important;
        border-top: none !important;
    }
    
    /* Dataframes - Premium Glass */
    .dataframe {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 24px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.3), rgba(236, 72, 153, 0.2)) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 16px 20px !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.5px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
    }
    
    .dataframe tbody tr td {
        padding: 14px 20px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.02) !important;
        color: #c8d0e0 !important;
        font-weight: 400 !important;
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(2px) !important;
        transition: all 0.3s ease !important;
    }
    
    .dataframe tbody tr:hover td {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(6px) !important;
        color: white !important;
        transform: scale(1.01) !important;
    }
    
    /* Info/Warning/Success boxes - Premium Glass */
    .stAlert {
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        padding: 20px 28px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        border-left: 4px solid !important;
    }
    
    /* Success - Green Neon */
    .stAlert[data-baseweb="notification"]:has(.stAlertIcon-success) {
        border-left-color: #34d399 !important;
        box-shadow: 0 8px 32px rgba(52, 211, 153, 0.1) !important;
    }
    
    /* Info - Blue Neon */
    .stAlert[data-baseweb="notification"]:has(.stAlertIcon-info) {
        border-left-color: #60a5fa !important;
        box-shadow: 0 8px 32px rgba(96, 165, 250, 0.1) !important;
    }
    
    /* Warning - Yellow Neon */
    .stAlert[data-baseweb="notification"]:has(.stAlertIcon-warning) {
        border-left-color: #fbbf24 !important;
        box-shadow: 0 8px 32px rgba(251, 191, 36, 0.1) !important;
    }
    
    /* Error - Red Neon */
    .stAlert[data-baseweb="notification"]:has(.stAlertIcon-error) {
        border-left-color: #f87171 !important;
        box-shadow: 0 8px 32px rgba(248, 113, 113, 0.1) !important;
    }
    
    /* Progress bar - Animated Gradient */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7c3aed, #ec4899, #f59e0b, #ec4899, #7c3aed) !important;
        background-size: 300% 100% !important;
        border-radius: 40px !important;
        backdrop-filter: blur(4px) !important;
        animation: progressGradient 3s ease-in-out infinite !important;
        height: 8px !important;
    }
    
    @keyframes progressGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Tabs - Premium Glass */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 30px !important;
        padding: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 30px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        color: rgba(255, 255, 255, 0.5) !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: rgba(255, 255, 255, 0.8) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: white !important;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3) !important;
        transform: scale(1.02) !important;
    }
    
    /* Scrollbar - Premium */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(4px) !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7c3aed, #ec4899) !important;
        border-radius: 20px !important;
        transition: all 0.3s ease !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #6d28d9, #db2777) !important;
        transform: scale(1.1) !important;
    }
    
    /* Custom Glass Card Class */
    .glass-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border-radius: 28px !important;
        padding: 2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: radial-gradient(circle at 30% 40%, rgba(124, 58, 237, 0.03), transparent 60%);
        opacity: 0;
        transition: opacity 0.6s ease;
        pointer-events: none;
    }
    
    .glass-card:hover {
        transform: translateY(-6px) !important;
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(124, 58, 237, 0.15) !important;
        box-shadow: 0 16px 60px rgba(0, 0, 0, 0.15), 0 0 80px rgba(124, 58, 237, 0.03) !important;
    }
    
    .glass-card:hover::before {
        opacity: 1;
    }
    
    /* Sidebar header */
    .sidebar-header {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01)) !important;
        border-radius: 30px !important;
        padding: 1.5rem 1rem !important;
        text-align: center !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        margin-bottom: 2rem !important;
        transition: all 0.4s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .sidebar-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(124, 58, 237, 0.05), transparent, rgba(236, 72, 153, 0.05), transparent);
        animation: rotate 8s linear infinite;
        opacity: 0.5;
    }
    
    .sidebar-header:hover {
        transform: scale(1.02) !important;
        border-color: rgba(124, 58, 237, 0.1) !important;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.05) !important;
    }
    
    .sidebar-header h2 {
        font-size: 2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff, #c4b5fd, #a78bfa) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        letter-spacing: -0.02em !important;
        margin: 0 !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    .sidebar-header p {
        color: rgba(255, 255, 255, 0.5) !important;
        font-size: 0.8rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.5px !important;
        margin-top: 6px !important;
        -webkit-text-fill-color: rgba(255, 255, 255, 0.5) !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center !important;
        padding: 1.5rem 0.5rem 0.5rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.03) !important;
        color: rgba(255, 255, 255, 0.3) !important;
        font-weight: 400 !important;
        font-size: 0.85rem !important;
        backdrop-filter: blur(4px) !important;
        -webkit-backdrop-filter: blur(4px) !important;
        margin-top: 2rem !important;
        letter-spacing: 0.3px !important;
    }
    
    .footer a {
        color: rgba(124, 58, 237, 0.7) !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .footer a:hover {
        color: rgba(124, 58, 237, 1) !important;
        text-shadow: 0 0 30px rgba(124, 58, 237, 0.2) !important;
    }
    
    /* Caption text in sidebar */
    .css-1v3fvcr, .st-emotion-cache-1v3fvcr {
        color: rgba(255, 255, 255, 0.3) !important;
        font-size: 0.75rem !important;
        padding: 0 6px !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    
    /* Custom markdown text color */
    .stMarkdown p, .stMarkdown li {
        color: rgba(255, 255, 255, 0.8) !important;
        line-height: 1.8 !important;
    }
    
    .stMarkdown strong {
        color: #ffffff !important;
    }
    
    /* Divider styling */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.2), rgba(236, 72, 153, 0.2), transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* Tooltip-like hover effect on metrics */
    [data-testid="metric-container"] {
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Navigation - All 16 Steps
page = st.sidebar.radio(
    "📌 **Select Step:**",
    [
        "🏠 Home",
        "1️⃣ Problem Statement",
        "2️⃣ Dataset Collection",
        "3️⃣ Data Loading",
        "4️⃣ Dataset Exploration",
        "5️⃣ EDA Visualization",
        "6️⃣ Data Cleaning",
        "7️⃣ Encoding",
        "8️⃣ Feature Engineering",
        "9️⃣ Feature Selection",
        "🔟 Train/Test Split",
        "1️⃣1️⃣ Model Training",
        "1️⃣2️⃣ Model Evaluation",
        "1️⃣3️⃣ Model Comparison",
        "1️⃣4️⃣ Best Model Selection",
        "1️⃣5️⃣ Prediction (Single)",
        "1️⃣6️⃣ Live Prediction (Interactive)"
    ],
    index=0
)

st.sidebar.markdown("---")

# Model stats in sidebar
st.sidebar.markdown("""
    <div style='background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; border: 1px solid rgba(255,255,255,0.08);'>
        <p style='color: #a8b2d1; margin: 0; font-size: 13px;'>📊 Model Accuracy</p>
        <p style='color: #4ecdc4; margin: 4px 0 0; font-size: 22px; font-weight: 700;'>93.6%</p>
        <p style='color: #a8b2d1; margin: 8px 0 0; font-size: 13px;'>🏆 Best Model: Random Forest</p>
    </div>
""", unsafe_allow_html=True)

# ============================================
# ============================================
# CACHE - Load Dataset and Models
# ============================================
# ============================================
# CACHE - Load Dataset and Models
# ============================================
@st.cache_data
def load_dataset():
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible paths
        possible_paths = [
            'onlinefooddeliverydataset.csv',  # Relative path
            os.path.join(script_dir, 'onlinefooddeliverydataset.csv'),  # Absolute path with script dir
            os.path.join(os.getcwd(), 'onlinefooddeliverydataset.csv'),  # Current working directory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
        
        # If we get here, file wasn't found
        st.error(f"❌ Dataset not found")
        st.info(f"📁 Looking for file: onlinefooddeliverydataset.csv")
        st.info(f"📁 Script directory: {script_dir}")
        st.info(f"📁 Current working directory: {os.getcwd()}")
        st.info(f"📁 Files in script directory: {os.listdir(script_dir) if os.path.exists(script_dir) else 'Directory not found'}")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None
@st.cache_resource
def load_models():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(current_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
        encoders = joblib.load(os.path.join(current_dir, 'label_encoders.pkl'))
        features = joblib.load(os.path.join(current_dir, 'selected_features.pkl'))
        return model, scaler, encoders, features
    except:
        return None, None, None, None

# ============================================
# FUNCTION: Get Data After Cleaning & Encoding
# ============================================
def get_clean_data():
    df = load_dataset()
    if df is None:
        return None
    
    # Clean
    if 'Unnamed: 13' in df.columns:
        df = df.drop('Unnamed: 13', axis=1)
    df['Feedback'] = df['Feedback'].str.strip()
    
    # Encode
    le_dict = {}
    cat_cols = ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications', 
               'Customer Type', 'Feedback']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Feature Engineering
    income_order = {'No Income': 0, 'Below Rs.10000': 1, '10001 to 25000': 2, 
                   '25001 to 50000': 3, 'More than 50000': 4}
    df['Income_Level'] = df['Monthly Income'].map(income_order)
    
    # Encode target
    le_target = LabelEncoder()
    df['Output'] = le_target.fit_transform(df['Output'])
    
    return df

# ============================================
# PAGE ROUTING
# ============================================

# ============================================
# PAGE 1: HOME
# ============================================
if page == "🏠 Home":
    st.title("🍔 Online Food Order Predictor")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## Welcome to the Complete Data Science Pipeline! 🚀
        
        This project demonstrates a **complete end-to-end data science workflow** for predicting 
        whether a customer will order food online.
        
        ### 📊 Project Overview:
        - **Dataset**: 388 rows, 14 columns
        - **Problem Type**: Binary Classification
        - **Target Variable**: `Output` (Yes/No)
        - **Best Model**: Random Forest (93.6% Accuracy)
        
        ### 🎯 Pipeline Steps (16 Total):
        1. Problem Statement
        2. Dataset Collection
        3. Data Loading
        4. Dataset Exploration
        5. EDA Visualization
        6. Data Cleaning
        7. Encoding
        8. Feature Engineering
        9. Feature Selection
        10. Train/Test Split
        11. Model Training
        12. Model Evaluation
        13. Model Comparison
        14. Best Model Selection
        15. Prediction (Single)
        16. Live Prediction (Interactive)
        
        ### 🔧 Technologies Used:
        - Python
        - Pandas, NumPy
        - Matplotlib, Seaborn, Plotly
        - Scikit-Learn
        - Streamlit
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Model Performance:
        - 🎯 **Accuracy**: 93.6%
        - 📈 **Precision**: 95.2%
        - 📊 **Recall**: 96.7%
        - 🏆 **F1-Score**: 95.9%
        """)
        st.markdown("---")
        st.info("👈 **Use the sidebar to navigate through each step!**")
    
    # Quick Stats
    st.markdown("---")
    st.subheader("📊 Quick Dataset Stats")
    df = load_dataset()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Rows", df.shape[0])
        with col2:
            st.metric("📋 Total Columns", df.shape[1])
        with col3:
            st.metric("🎯 Target Classes", "2 (Yes/No)")
        with col4:
            st.metric("✅ Missing Values", "0")

# ============================================
# PAGE 2: PROBLEM STATEMENT
# ============================================
elif page == "1️⃣ Problem Statement":
    st.title("📋 1. Problem Statement")
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Understanding the Problem
    
    ### **Business Problem:**
    Online food delivery platforms want to understand **customer behavior** and **predict** 
    whether a customer will order food online or not.
    
    ### **Why is this important?**
    - 📈 Improve customer experience
    - 🎯 Targeted marketing campaigns
    - 💰 Increase sales and revenue
    - 🔮 Better resource allocation
    
    ### **Problem Type:**
    - **Supervised Learning**: Classification
    - **Target Variable**: `Output` (Yes/No)
    - **Goal**: Predict if customer will order food online
    
    ### **Dataset Overview:**
    - **Source**: Kaggle - Online Food Ordering Dataset
    - **Rows**: 388
    - **Columns**: 14
    - **Features**: Demographic, socioeconomic, geographic data
    
    ### **Key Questions:**
    1. What factors influence online food ordering?
    2. Who are the most likely customers to order?
    3. How accurate can we predict customer behavior?
    
    ### **Success Metrics:**
    - ✅ **Accuracy**: > 90%
    - ✅ **Precision**: > 90%
    - ✅ **Recall**: > 90%
    - ✅ **F1-Score**: > 90%
    """)
    
    st.info("💡 **Next Step**: Click on '2️⃣ Dataset Collection' in the sidebar")

# ============================================
# PAGE 3: DATASET COLLECTION
# ============================================
elif page == "2️⃣ Dataset Collection":
    st.title("📦 2. Dataset Collection")
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Dataset Information
    
    ### **Dataset Name:** Online Food Ordering Dataset
    ### **Source:** Kaggle
    
    ### **Description:**
    This dataset contains information about customers and their online food ordering behavior.
    
    ### **Dataset Attributes:**
    | Column | Description | Type |
    |--------|-------------|------|
    | Age | Customer's age | Numeric |
    | Gender | Male/Female | Categorical |
    | Marital Status | Single/Married/Prefer not to say | Categorical |
    | Occupation | Student/Employee/Self Employed/House wife | Categorical |
    | Monthly Income | Income bracket | Categorical |
    | Educational Qualifications | Education level | Categorical |
    | Family Size | Number of family members | Numeric |
    | Customer Type | Frequent/Regular/New | Categorical |
    | Latitude | Geographic location | Numeric |
    | Longitude | Geographic location | Numeric |
    | Pin Code | Area pin code | Numeric |
    | Output | Target: Yes/No | Categorical |
    | Feedback | Positive/Negative | Categorical |
    """)
    
    df = load_dataset()
    if df is not None:
        st.subheader("📋 Sample of the Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing {len(df)} rows and {len(df.columns)} columns")
    
    st.info("💡 **Next Step**: Click on '3️⃣ Data Loading' in the sidebar")

# ============================================
# PAGE 4: DATA LOADING
# ============================================
elif page == "3️⃣ Data Loading":
    st.title("📂 3. Data Loading")
    st.markdown("---")
    
    st.markdown("""
    ## 📥 Loading the Dataset
    
    ### **What is Data Loading?**
    Data loading is the process of **reading** the dataset from a file into a **DataFrame** 
    structure for analysis and processing.
    
    ### **Code:**
    ```python
    import pandas as pd
    
    # Load dataset from local directory
    df = pd.read_csv('online food delivery dataset.csv')
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    print(df.head())
    """)
    
    df = load_dataset()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Rows", len(df))
        with col2:
            st.metric("📋 Total Columns", len(df.columns))
        with col3:
            st.metric("💾 Memory", f"{df.memory_usage().sum() / 1024:.1f} KB")
        
        st.subheader("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    
    st.info("💡 Next Step: Click on '4️⃣ Dataset Exploration' in the sidebar")

# ============================================
# PAGE 5: DATASET EXPLORATION
# ============================================
elif page == "4️⃣ Dataset Exploration":
    st.title("🔍 4. Dataset Exploration")
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Exploring the Dataset
    
    ### What is Exploratory Data Analysis (EDA)?
    EDA is the process of analyzing and understanding the dataset before building models.
    """)
    
    df = load_dataset()
    if df is not None:
        # Data Info
        st.subheader("📋 Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Types:")
            st.write(df.dtypes)
        with col2:
            st.write("Missing Values:")
            st.write(df.isnull().sum())
        
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Missing Values
        st.subheader("🔎 Missing Values Analysis")
        if df.isnull().sum().sum() == 0:
            st.success("✅ No missing values found in any column!")
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            df.isnull().sum().plot(kind='bar', ax=ax, color='red')
            ax.set_title('Missing Values by Column')
            st.pyplot(fig)
    
    st.info("💡 Next Step: Click on '5️⃣ EDA Visualization' in the sidebar")

# ============================================
# PAGE 6: EDA VISUALIZATION
# ============================================
elif page == "5️⃣ EDA Visualization":
    st.title("📈 5. EDA Visualization")
    st.markdown("---")
    
    st.markdown("""
    ## 🎨 Exploratory Data Analysis (Visualization)
    
    ### Why Visualization is Important:
    - 📊 Visual patterns are easier to understand
    - 🔍 Identify outliers and anomalies
    - 📈 Discover relationships between features
    """)
    
    df = load_dataset()
    if df is not None:
        # Clean Feedback
        df['Feedback'] = df['Feedback'].str.strip()
        
        # 1. Target Variable
        st.subheader("🎯 Target Variable: Output")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df['Output'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                colors=['#ff6b6b', '#4ecdc4'], ax=ax)
            ax.set_title('Output Distribution')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            df['Output'].value_counts().plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], ax=ax)
            ax.set_title('Output Count')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        # 2. Age Distribution
        st.subheader("📊 Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        df['Age'].hist(bins=15, color='#6c5ce7', edgecolor='black', ax=ax)
        ax.set_title('Age Distribution')
        st.pyplot(fig)
        
        # 3. Categorical Features
        st.subheader("📊 Categorical Features")
        cat_cols = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income',
                    'Educational Qualifications', 'Customer Type']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for idx, col in enumerate(cat_cols):
            row = idx // 3
            col_idx = idx % 3
            df[col].value_counts().plot(kind='bar', ax=axes[row, col_idx], color='skyblue')
            axes[row, col_idx].set_title(f'{col}')
            axes[row, col_idx].tick_params(axis='x', rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        
        # 4. Relationship with Target
        st.subheader("📊 Relationship with Target (Output)")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for idx, col in enumerate(cat_cols[:6]):
            row = idx // 3
            col_idx = idx % 3
            pd.crosstab(df[col], df['Output']).plot(kind='bar', stacked=True,
                color=['#ff6b6b', '#4ecdc4'], ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Output vs {col}')
            axes[row, col_idx].tick_params(axis='x', rotation=15)
            axes[row, col_idx].legend(title='Output')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.info("💡 Next Step: Click on '6️⃣ Data Cleaning' in the sidebar")

# ============================================
# PAGE 7: DATA CLEANING
# ============================================
elif page == "6️⃣ Data Cleaning":
    st.title("🧹 6. Data Cleaning")
    st.markdown("---")
    
    st.markdown("""
    ## 🧼 Data Cleaning Process
    
    ### What is Data Cleaning?
    Data cleaning is the process of fixing or removing incorrect, corrupted,
    incorrectly formatted, duplicate, or incomplete data.
    
    ### Steps Applied:
    - ✅ Remove unnecessary columns
    - ✅ Standardize text data
    - ✅ Handle missing values
    """)
    
    df = load_dataset()
    if df is not None:
        df_original = df.copy()
        
        st.subheader("📋 Before Cleaning")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", df.columns.tolist())
        with col2:
            st.write("Missing Values:")
            st.write(df.isnull().sum())
        
        # Cleaning
        st.subheader("🔄 Cleaning Steps Applied")
        
        if 'Unnamed: 13' in df.columns:
            df = df.drop('Unnamed: 13', axis=1)
            st.success("✅ Dropped 'Unnamed: 13' column")
        
        df['Feedback'] = df['Feedback'].str.strip()
        st.success("✅ Cleaned 'Feedback' column (removed extra spaces)")
        
        st.subheader("📋 After Cleaning")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", df.columns.tolist())
        with col2:
            st.write("Missing Values:")
            st.write(df.isnull().sum())
        
        st.subheader("📄 Sample Data After Cleaning")
        st.dataframe(df.head(10), use_container_width=True)
    
    st.info("💡 Next Step: Click on '7️⃣ Encoding' in the sidebar")

# ============================================
# PAGE 8: ENCODING
# ============================================
elif page == "7️⃣ Encoding":
    st.title("🔢 7. Encoding")
    st.markdown("---")
    
    st.markdown("""
    ## 🔄 Encoding Categorical Variables
    
    ### What is Encoding?
    Encoding converts categorical data (text) into numerical data (numbers)
    so that ML algorithms can understand them.
    
    ### Types of Encoding:
    - **Label Encoding**: Assigns numbers to categories
    - **One-Hot Encoding**: Creates binary columns for each category
    """)
    
    df = load_dataset()
    if df is not None:
        df_encoded = df.copy()
        df_encoded['Feedback'] = df_encoded['Feedback'].str.strip()
        if 'Unnamed: 13' in df_encoded.columns:
            df_encoded = df_encoded.drop('Unnamed: 13', axis=1)
        
        cat_cols = df_encoded.select_dtypes(include=['object']).columns
        st.subheader("📋 Categorical Columns")
        st.write("Categorical columns found:", list(cat_cols))
        
        st.subheader("🔄 Encoding Process")
        le_dict = {}
        for col in cat_cols:
            with st.expander(f"📌 Encoding: {col}"):
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
                le_dict[col] = le
                st.write("Original → Encoded:")
                mapping = dict(zip(df_encoded[col].unique(),
                                   df_encoded[col + '_encoded'].unique()))
                for orig, enc in mapping.items():
                    st.write(f"  {orig} → {enc}")
        
        st.subheader("📋 Final Dataset After Encoding")
        st.dataframe(df_encoded.head(10), use_container_width=True)
    
    st.info("💡 Next Step: Click on '8️⃣ Feature Engineering' in the sidebar")

# ============================================
# PAGE 9: FEATURE ENGINEERING
# ============================================
elif page == "8️⃣ Feature Engineering":
    st.title("⚙️ 8. Feature Engineering")
    st.markdown("---")
    
    st.markdown("""
    ## 🛠️ Feature Engineering
    
    ### What is Feature Engineering?
    Creating new features from existing data to improve model performance.
    
    ### New Features Created:
    - **Age_Group**: Categorized age into groups
    - **Income_Level**: Numeric representation of income
    - **Family_Category**: Family size groups
    - **Is_Student**: Binary feature
    - **Is_Married**: Binary feature
    """)
    
    df = load_dataset()
    if df is not None:
        df_engineered = df.copy()
        
        st.subheader("📋 Original Features")
        st.write(f"Original features: {len(df_engineered.columns)}")
        
        # Create new features
        df_engineered['Age_Group'] = pd.cut(df_engineered['Age'],
                                            bins=[18, 22, 26, 30, 34],
                                            labels=['18-22', '23-26', '27-30', '31-34'])
        st.success("✅ Created 'Age_Group' feature")
        
        income_order = {'No Income': 0, 'Below Rs.10000': 1, '10001 to 25000': 2,
                        '25001 to 50000': 3, 'More than 50000': 4}
        df_engineered['Income_Level'] = df_engineered['Monthly Income'].map(income_order)
        st.success("✅ Created 'Income_Level' numeric feature")
        
        df_engineered['Family_Category'] = pd.cut(df_engineered['Family size'],
                                                  bins=[0, 2, 4, 6],
                                                  labels=['Small', 'Medium', 'Large'])
        st.success("✅ Created 'Family_Category' feature")
        
        df_engineered['Is_Student'] = (df_engineered['Occupation'] == 'Student').astype(int)
        st.success("✅ Created 'Is_Student' binary feature")
        
        df_engineered['Is_Married'] = (df_engineered['Marital Status'] == 'Married').astype(int)
        st.success("✅ Created 'Is_Married' binary feature")
        
        st.subheader("📊 New Features Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Age Group:")
            st.write(df_engineered['Age_Group'].value_counts())
        with col2:
            st.write("Family Category:")
            st.write(df_engineered['Family_Category'].value_counts())
        
        st.subheader("📋 Final Features")
        st.write(f"Total features: {len(df_engineered.columns)}")
        st.dataframe(df_engineered.head(10), use_container_width=True)
    
    st.info("💡 Next Step: Click on '9️⃣ Feature Selection' in the sidebar")

# ============================================
# PAGE 10: FEATURE SELECTION
# ============================================
elif page == "9️⃣ Feature Selection":
    st.title("🎯 9. Feature Selection")
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Feature Selection
    
    ### What is Feature Selection?
    Selecting the most important features that contribute most to the prediction.
    
    ### Method Used:
    Random Forest Feature Importance
    """)
    
    df = get_clean_data()
    if df is not None:
        # Check if dataframe is empty
        if df.empty:
            st.error("❌ Dataframe is empty after cleaning. Please check the data.")
        else:
            # Check if 'Output' column exists
            if 'Output' not in df.columns:
                st.error("❌ 'Output' column not found in the data.")
            else:
                X = df.drop('Output', axis=1)
                y = df['Output']
                
                # Double-check that X has only numeric columns
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < len(X.columns):
                    st.warning(f"⚠️ Some columns are not numeric. Converting...")
                    X = X[numeric_cols]
                
                # Feature Importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                st.subheader("📊 Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
                importance_df.plot(kind='barh', x='Feature', y='Importance',
                                   ax=ax, legend=False, color=colors)
                ax.set_title('Feature Importance (Random Forest)', fontsize=14)
                ax.set_xlabel('Importance Score')
                st.pyplot(fig)
                
                st.subheader("📋 Top Features")
                st.dataframe(importance_df.tail(10), use_container_width=True)
                
                # Show selected features
                cumsum = importance_df['Importance'].cumsum()
                n_features = (cumsum <= 0.85).sum() + 1
                top_features = importance_df.tail(n_features)['Feature'].tolist()
                
                st.success(f"✅ Selected {n_features} features (85% cumulative importance)")
                st.write("Selected Features:", top_features)
    
    st.info("💡 Next Step: Click on '🔟 Train/Test Split' in the sidebar")

# ============================================
# PAGE 11: TRAIN/TEST SPLIT
# ============================================
elif page == "🔟 Train/Test Split":
    st.title("📊 10. Train/Test Split")
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Train/Test Split
    
    ### What is Train/Test Split?
    Splitting the dataset into training and testing sets.
    
    ### Why?
    - 🎯 Training set: Used to train the model
    - 📊 Testing set: Used to evaluate model performance
    - 📈 Prevents overfitting
    """)
    
    df = get_clean_data()
    if df is not None:
        X = df.drop('Output', axis=1)
        y = df['Output']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        st.subheader("📊 Split Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Samples", len(X))
        with col2:
            st.metric("📚 Training Set", len(X_train))
        with col3:
            st.metric("📝 Testing Set", len(X_test))
        
        st.subheader("📊 Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Training Set:")
            train_dist = pd.Series(y_train).value_counts()
            st.write(train_dist)
        with col2:
            st.write("Testing Set:")
            test_dist = pd.Series(y_test).value_counts()
            st.write(test_dist)
        
        st.success(f"✅ Split Complete: {len(X_train)} training, {len(X_test)} testing")
    
    st.info("💡 Next Step: Click on '1️⃣1️⃣ Model Training' in the sidebar")

# ============================================
# PAGE 12: MODEL TRAINING
# ============================================
elif page == "1️⃣1️⃣ Model Training":
    st.title("🤖 11. Model Training")
    st.markdown("---")
    
    st.markdown("""
    ## 🤖 Model Training
    
    ### What is Model Training?
    Teaching a machine learning model to learn patterns from the training data.
    
    ### Models Trained:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - SVM
    - KNN
    """)
    
    df = get_clean_data()
    if df is not None:
        # Check if dataframe is empty
        if df.empty:
            st.error("❌ Dataframe is empty after cleaning. Please check the data.")
        else:
            X = df.drop('Output', axis=1)
            y = df['Output']
            
            # Ensure all columns are numeric
            X = X.select_dtypes(include=[np.number])
            
            # Check if there are any columns left
            if X.empty:
                st.error("❌ No numeric columns found for training.")
            else:
                # Feature selection with error handling
                try:
                    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_temp.fit(X, y)
                    
                    importance_temp = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf_temp.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    cumsum = importance_temp['Importance'].cumsum()
                    n_features = (cumsum <= 0.85).sum() + 1
                    top_features = importance_temp.head(n_features)['Feature'].tolist()
                    
                    X = X[top_features]
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Scale
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train models
                    models = {
                        'Logistic Regression': LogisticRegression(random_state=42),
                        'Decision Tree': DecisionTreeClassifier(random_state=42),
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                        'SVM': SVC(random_state=42),
                        'KNN': KNeighborsClassifier()
                    }
                    
                    results = []
                    for name, model in models.items():
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        results.append({'Model': name, 'Accuracy': accuracy})
                    
                    st.subheader("📊 Training Results")
                    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#4ecdc4' if i == 0 else '#ff6b6b' for i in range(len(results_df))]
                    results_df.plot(kind='barh', x='Model', y='Accuracy', ax=ax, legend=False, color=colors)
                    ax.set_title('Model Accuracy Comparison', fontsize=14)
                    ax.set_xlabel('Accuracy')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error during model training: {e}")
                    st.info("💡 Make sure all features are numeric and properly encoded.")
    
    st.info("💡 Next Step: Click on '1️⃣2️⃣ Model Evaluation' in the sidebar")

# ============================================
# PAGE 13: MODEL EVALUATION
# ============================================
elif page == "1️⃣2️⃣ Model Evaluation":
    st.title("📊 12. Model Evaluation")
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Model Evaluation
    
    ### Evaluation Metrics:
    - **Accuracy**: Overall correctness
    - **Precision**: True positives / (True positives + False positives)
    - **Recall**: True positives / (True positives + False negatives)
    - **F1-Score**: Harmonic mean of Precision and Recall
    """)
    
    df = get_clean_data()
    if df is not None:
        # Check if dataframe is empty
        if df.empty:
            st.error("❌ Dataframe is empty after cleaning. Please check the data.")
        else:
            # Display column info for debugging
            with st.expander("🔍 Data Info (for debugging)"):
                st.write("Columns in dataframe:", df.columns.tolist())
                st.write("Data types:", df.dtypes)
                st.write("Shape:", df.shape)
            
            X = df.drop('Output', axis=1)
            y = df['Output']
            
            # Ensure all columns are numeric
            X = X.select_dtypes(include=[np.number])
            
            # Check if there are any columns left
            if X.empty:
                st.error("❌ No numeric columns found for evaluation.")
                st.write("Available columns:", df.columns.tolist())
            else:
                try:
                    # Feature selection with error handling
                    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_temp.fit(X, y)
                    
                    importance_temp = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf_temp.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    cumsum = importance_temp['Importance'].cumsum()
                    n_features = (cumsum <= 0.85).sum() + 1
                    top_features = importance_temp.head(n_features)['Feature'].tolist()
                    
                    X = X[top_features]
                    
                    # Split and scale
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Best model - Random Forest
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_train_scaled, y_train)
                    y_pred = rf.predict(X_test_scaled)
                    
                    st.subheader("📊 Random Forest Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    with col2:
                        st.metric("📈 Precision", f"{precision_score(y_test, y_pred):.4f}")
                    with col3:
                        st.metric("📊 Recall", f"{recall_score(y_test, y_pred):.4f}")
                    with col4:
                        st.metric("🏆 F1-Score", f"{f1_score(y_test, y_pred):.4f}")
                    
                    st.subheader("📊 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                    ax.set_title('Confusion Matrix - Random Forest')
                    st.pyplot(fig)
                    
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during model evaluation: {e}")
                    st.info("💡 Make sure all features are numeric and properly encoded.")
                    st.write("X columns:", X.columns.tolist())
                    st.write("X data types:", X.dtypes)
    
    st.info("💡 Next Step: Click on '1️⃣3️⃣ Model Comparison' in the sidebar")

# ============================================
# PAGE 14: MODEL COMPARISON
# ============================================
elif page == "1️⃣3️⃣ Model Comparison":
    st.title("📊 13. Model Comparison")
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Model Comparison
    
    Comparing all models to find the best performer.
    """)
    
    df = get_clean_data()
    if df is not None:
        # Check if dataframe is empty
        if df.empty:
            st.error("❌ Dataframe is empty after cleaning. Please check the data.")
        else:
            # Display column info for debugging
            with st.expander("🔍 Data Info (for debugging)"):
                st.write("Columns in dataframe:", df.columns.tolist())
                st.write("Data types:", df.dtypes)
                st.write("Shape:", df.shape)
            
            X = df.drop('Output', axis=1)
            y = df['Output']
            
            # Ensure all columns are numeric
            X = X.select_dtypes(include=[np.number])
            
            # Check if there are any columns left
            if X.empty:
                st.error("❌ No numeric columns found for model comparison.")
                st.write("Available columns:", df.columns.tolist())
            else:
                try:
                    # Feature selection with error handling
                    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_temp.fit(X, y)
                    
                    importance_temp = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf_temp.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    cumsum = importance_temp['Importance'].cumsum()
                    n_features = (cumsum <= 0.85).sum() + 1
                    top_features = importance_temp.head(n_features)['Feature'].tolist()
                    
                    X = X[top_features]
                    
                    # Split and scale
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # All models
                    models = {
                        'Logistic Regression': LogisticRegression(random_state=42),
                        'Decision Tree': DecisionTreeClassifier(random_state=42),
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                        'SVM': SVC(random_state=42),
                        'KNN': KNeighborsClassifier()
                    }
                    
                    results = []
                    for name, model in models.items():
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        results.append({
                            'Model': name,
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred),
                            'Recall': recall_score(y_test, y_pred),
                            'F1-Score': f1_score(y_test, y_pred)
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    st.subheader("📊 Complete Comparison")
                    st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                                 use_container_width=True)
                    
                    # Plot comparison
                    st.subheader("📊 Visual Comparison")
                    fig = go.Figure()
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=results_df['Model'],
                            y=results_df[metric],
                            text=results_df[metric].round(4),
                            textposition='auto'
                        ))
                    fig.update_layout(
                        barmode='group',
                        height=500,
                        title='Model Performance Comparison',
                        legend=dict(orientation="h", y=1.02)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during model comparison: {e}")
                    st.info("💡 Make sure all features are numeric and properly encoded.")
                    st.write("X columns:", X.columns.tolist())
                    st.write("X data types:", X.dtypes)
    
    st.info("💡 Next Step: Click on '1️⃣4️⃣ Best Model Selection' in the sidebar")

# ============================================
# PAGE 15: BEST MODEL SELECTION
# ============================================
elif page == "1️⃣4️⃣ Best Model Selection":
    st.title("🏆 14. Best Model Selection")
    st.markdown("---")
    
    st.markdown("""
    ## 🏆 Best Model Selection
    
    ### Selected Model: Random Forest
    
    ### Why Random Forest?
    - ✅ Highest accuracy (93.6%)
    - ✅ Best F1-Score (95.9%)
    - ✅ Robust to overfitting
    - ✅ Handles both numeric and categorical features
    - ✅ Provides feature importance
    """)
    
    df = get_clean_data()
    if df is not None:
        # Check if dataframe is empty
        if df.empty:
            st.error("❌ Dataframe is empty after cleaning. Please check the data.")
        else:
            # Display column info for debugging
            with st.expander("🔍 Data Info (for debugging)"):
                st.write("Columns in dataframe:", df.columns.tolist())
                st.write("Data types:", df.dtypes)
                st.write("Shape:", df.shape)
            
            X = df.drop('Output', axis=1)
            y = df['Output']
            
            # Ensure all columns are numeric
            X = X.select_dtypes(include=[np.number])
            
            # Check if there are any columns left
            if X.empty:
                st.error("❌ No numeric columns found for best model selection.")
                st.write("Available columns:", df.columns.tolist())
            else:
                try:
                    # Feature selection with error handling
                    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_temp.fit(X, y)
                    
                    importance_temp = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf_temp.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    cumsum = importance_temp['Importance'].cumsum()
                    n_features = (cumsum <= 0.85).sum() + 1
                    top_features = importance_temp.head(n_features)['Feature'].tolist()
                    
                    X = X[top_features]
                    
                    # Split and scale
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Best model
                    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    best_model.fit(X_train_scaled, y_train)
                    y_pred = best_model.predict(X_test_scaled)
                    
                    st.subheader("📊 Best Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    with col2:
                        st.metric("📈 Precision", f"{precision_score(y_test, y_pred):.4f}")
                    with col3:
                        st.metric("📊 Recall", f"{recall_score(y_test, y_pred):.4f}")
                    with col4:
                        st.metric("🏆 F1-Score", f"{f1_score(y_test, y_pred):.4f}")
                    
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                    st.success("✅ Model selected and saved as 'best_model.pkl'")
                    
                    # Optional: Save the model
                    try:
                        joblib.dump(best_model, 'best_model.pkl')
                        joblib.dump(scaler, 'scaler.pkl')
                        st.success("✅ Model files saved successfully!")
                    except Exception as e:
                        st.warning(f"⚠️ Could not save model files: {e}")
                    
                except Exception as e:
                    st.error(f"❌ Error during best model selection: {e}")
                    st.info("💡 Make sure all features are numeric and properly encoded.")
                    st.write("X columns:", X.columns.tolist())
                    st.write("X data types:", X.dtypes)
    
    st.info("💡 Next Step: Click on '1️⃣5️⃣ Prediction' in the sidebar")
# ============================================
# PAGE 16: PREDICTION (SINGLE)
# ============================================
elif page == "1️⃣5️⃣ Prediction (Single)":
    st.title("🔮 15. Prediction (Single)")
    st.markdown("---")
    
    st.markdown("""
    ## 🔮 Make a Single Prediction
    
    Enter customer details and get a prediction.
    """)
    
    model, scaler, encoders, features = load_models()
    
    if model is not None:
        st.subheader("👤 Enter Customer Details")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("📅 Age", 18, 33, 25, key="single_age")
            gender = st.selectbox("👥 Gender", ["Male", "Female"], key="single_gender")
            marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Prefer not to say"], key="single_marital")
            occupation = st.selectbox("💼 Occupation", ["Student", "Employee", "Self Employeed", "House wife"], key="single_occ")
            education = st.selectbox("🎓 Education", ["Graduate", "Post Graduate", "Ph.D", "School", "Uneducated"], key="single_edu")
        
        with col2:
            family_size = st.slider("👨‍👩‍👧‍👦 Family Size", 1, 6, 3, key="single_family")
            customer_type = st.selectbox("🛍️ Customer Type", ["Regular", "Frequent", "New"], key="single_customer")
            monthly_income = st.selectbox("💰 Monthly Income",
                ["No Income", "Below Rs.10000", "10001 to 25000",
                 "25001 to 50000", "More than 50000"], key="single_income")
            feedback = st.selectbox("📝 Previous Feedback", ["Positive", "Negative"], key="single_feedback")
            location = st.text_input("📍 Location", "Bangalore", key="single_location")
        
        if st.button("🔮 Predict", type="primary", key="single_predict"):
            # Prepare input
            input_data = {
                'Age': age,
                'Gender': gender,
                'Marital Status': marital_status,
                'Occupation': occupation,
                'Monthly Income': monthly_income,
                'Educational Qualifications': education,
                'Family size': family_size,
                'Customer Type': customer_type,
                'latitude': 12.9716,
                'longitude': 77.5946,
                'Pin code': 560001,
                'Feedback': feedback
            }
            
            df_input = pd.DataFrame([input_data])
            
            # Encode
            for col in ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications',
                        'Customer Type', 'Feedback']:
                if col in encoders:
                    df_input[col] = encoders[col].transform(df_input[col])
            
            # Add Income Level
            income_order = {'No Income': 0, 'Below Rs.10000': 1, '10001 to 25000': 2,
                            '25001 to 50000': 3, 'More than 50000': 4}
            df_input['Income_Level'] = df_input['Monthly Income'].map(income_order)
            
            # Select features
            df_input = df_input[features]
            
            # Scale
            df_input_scaled = scaler.transform(df_input)
            
            # Predict
            prediction = model.predict(df_input_scaled)[0]
            probability = model.predict_proba(df_input_scaled)[0][1]
            
            # Display result
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            if prediction == 1:
                st.success(f"### 🎉 YES! Customer will order food online")
                st.metric("Probability", f"{probability*100:.1f}%", "📈")
                st.balloons()
            else:
                st.error(f"### 😔 NO! Customer will NOT order food online")
                st.metric("Probability", f"{(1-probability)*100:.1f}%", "📉")
            
            with st.expander("📋 View Input Summary"):
                st.json(input_data)
    else:
        st.warning("⚠️ Model files not found. Please train the model first.")
    
    st.info("💡 Next Step: Click on '1️⃣6️⃣ Live Prediction' in the sidebar")


# ============================================
# PAGE 17: LIVE PREDICTION (INTERACTIVE)
# ============================================
elif page == "1️⃣6️⃣ Live Prediction (Interactive)":
    st.title("🎮 16. Live Prediction (Interactive)")
    st.markdown("---")
    
    st.markdown("""
    ## 🎮 Interactive Live Prediction
    
    Adjust the parameters in real-time and see predictions change instantly!
    """)
    
    model, scaler, encoders, features = load_models()
    
    if model is not None:
        # All inputs in sidebar
        with st.sidebar:
            st.markdown("### 🎮 Live Controls")
            st.info("💡 Adjust values in real-time")
            age_live = st.slider("📅 Age", 18, 33, 25, key="live_age")
            gender_live = st.selectbox("👥 Gender", ["Male", "Female"], key="live_gender")
            marital_live = st.selectbox("💍 Marital Status", ["Single", "Married", "Prefer not to say"], key="live_marital")
            occupation_live = st.selectbox("💼 Occupation", ["Student", "Employee", "Self Employeed", "House wife"], key="live_occ")
            education_live = st.selectbox("🎓 Education", ["Graduate", "Post Graduate", "Ph.D", "School", "Uneducated"], key="live_edu")
            family_live = st.slider("👨‍👩‍👧‍👦 Family Size", 1, 6, 3, key="live_family")
            customer_live = st.selectbox("🛍️ Customer Type", ["Regular", "Frequent", "New"], key="live_customer")
            income_live = st.selectbox("💰 Monthly Income",
                ["No Income", "Below Rs.10000", "10001 to 25000",
                 "25001 to 50000", "More than 50000"], key="live_income")
            feedback_live = st.selectbox("📝 Previous Feedback", ["Positive", "Negative"], key="live_feedback")
            
            # Add a sync button
            if st.button("🔄 Sync from Single Prediction", key="sync_button"):
                # This will trigger a rerun with session state values
                st.rerun()
        
        # Make prediction in real-time
        input_data_live = {
            'Age': age_live,
            'Gender': gender_live,
            'Marital Status': marital_live,
            'Occupation': occupation_live,
            'Monthly Income': income_live,
            'Educational Qualifications': education_live,
            'Family size': family_live,
            'Customer Type': customer_live,
            'latitude': 12.9716,
            'longitude': 77.5946,
            'Pin code': 560001,
            'Feedback': feedback_live
        }
        
        df_input_live = pd.DataFrame([input_data_live])
        
        # Encode
        for col in ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications',
                    'Customer Type', 'Feedback']:
            if col in encoders:
                df_input_live[col] = encoders[col].transform(df_input_live[col])
        
        # Add Income Level
        income_order = {'No Income': 0, 'Below Rs.10000': 1, '10001 to 25000': 2,
                        '25001 to 50000': 3, 'More than 50000': 4}
        df_input_live['Income_Level'] = df_input_live['Monthly Income'].map(income_order)
        
        # Select features
        df_input_live = df_input_live[features]
        
        # Scale
        df_input_live_scaled = scaler.transform(df_input_live)
        
        # Predict
        prediction_live = model.predict(df_input_live_scaled)[0]
        probability_live = model.predict_proba(df_input_live_scaled)[0][1]
        
        # Display live result
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Customer Profile")
            profile_data = {
                'Age': age_live,
                'Gender': gender_live,
                'Marital Status': marital_live,
                'Occupation': occupation_live,
                'Education': education_live,
                'Family Size': family_live,
                'Customer Type': customer_live,
                'Monthly Income': income_live,
                'Feedback': feedback_live
            }
            st.json(profile_data)
        
        with col2:
            st.markdown("### 🔮 Live Prediction")
            
            # Animated gauge
            if prediction_live == 1:
                st.markdown(f"""
                    <div style='text-align: center; padding: 40px; background: linear-gradient(145deg, #2e9a7a 0%, #1f7a5f 100%); border-radius: 24px; box-shadow: 0 12px 40px rgba(46, 154, 122, 0.25);'>
                        <h1 style='color: white; font-size: 48px;'>✅ YES</h1>
                        <p style='color: white; font-size: 24px; font-weight: 500;'>Will Order Food</p>
                        <div style='font-size: 72px;'>🍕</div>
                        <h2 style='color: white;'>{probability_live*100:.1f}%</h2>
                        <p style='color: rgba(255,255,255,0.8);'>Probability</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='text-align: center; padding: 40px; background: linear-gradient(145deg, #c0392b 0%, #a93226 100%); border-radius: 24px; box-shadow: 0 12px 40px rgba(192, 57, 43, 0.25);'>
                        <h1 style='color: white; font-size: 48px;'>❌ NO</h1>
                        <p style='color: white; font-size: 24px; font-weight: 500;'>Will Not Order</p>
                        <div style='font-size: 72px;'>😔</div>
                        <h2 style='color: white;'>{(1-probability_live)*100:.1f}%</h2>
                        <p style='color: rgba(255,255,255,0.8);'>Probability</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Real-time probability bar
        st.markdown("### 📈 Probability Gauge")
        prob_bar = st.progress(0)
        prob_bar.progress(probability_live)
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 💡 Insights")
        
        if probability_live > 0.7:
            st.success("🎯 High probability! This customer is likely to order.")
        elif probability_live > 0.4:
            st.warning("⚠️ Moderate probability. Consider offering promotions.")
        else:
            st.error("📉 Low probability. Customer may need incentives.")
        
        # Feature impact
        st.markdown("### 📊 Key Factors")
        factors = []
        if age_live < 25:
            factors.append("✅ Younger age group tends to order more")
        if education_live in ["Graduate", "Post Graduate"]:
            factors.append("✅ Higher education correlates with ordering")
        if feedback_live == "Positive":
            factors.append("✅ Positive feedback indicates higher likelihood")
        if customer_live in ["Frequent", "Regular"]:
            factors.append("✅ Regular customers are more likely to order")
        if income_live in ["25001 to 50000", "More than 50000"]:
            factors.append("✅ Higher income brackets order more")
        if occupation_live == "Student":
            factors.append("✅ Students are frequent orderers")
        
        for factor in factors:
            st.write(factor)
        
        if not factors:
            st.write("💡 No strong indicators found. Consider more data.")
    else:
        st.warning("⚠️ Model files not found. Please train the model first.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p style='font-size: 16px; font-weight: 600;'>🍔 Online Food Order Predictor | Built with ❤️ using Streamlit</p>
        <p style='font-size: 14px; color: #888;'>📊 Model Accuracy: 93.6% | Best Model: Random Forest</p>
        <p style='font-size: 12px; color: #aaa; margin-top: 8px;'>© 2026 All Rights Reserved</p>
    </div>
""", unsafe_allow_html=True)