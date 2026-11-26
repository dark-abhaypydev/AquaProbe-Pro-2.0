import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import json
import os

# ==========================================
# 1. PAGE SETUP (FULL-SCREEN HARDWARE CONSOLE)
# ==========================================
st.set_page_config(
    page_title="AquaProbe Pro-9™ | Physicochemical Water Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Global Modern Hardware Console Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Outfit:wght@500;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Edge-to-edge full width styling */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    
    /* Hardware Header Container */
    .hardware-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 1.8rem 2.4rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }
    .hardware-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0284c7, #38bdf8, #0284c7);
    }
    .hardware-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    .hardware-header h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
    }
    .hardware-header p {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Input Sensor Grid Card */
    . {
        background: skyblue;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 2rem 2.5rem;
        box-shadow: 0 4px 25px -2px rgba(0,0,0,0.06);
        margin-bottom: 1.4rem;
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Primary Hardware Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.85rem 3rem;
        border-radius: 14px;
        border: none;
        box-shadow: 0 8px 25px -3px rgba(2, 132, 199, 0.45);
        transition: all 0.25s ease;
        margin: 0 auto;
        display: block;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        transform: translateY(-2px);
        box-shadow: 0 12px 30px -3px rgba(2, 132, 199, 0.55);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD TRAINED MODEL & 2D ANIME ASSETS
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(CURRENT_DIR, "water_quality_pipeline.joblib")
CLEAN_ANIME_IMG = os.path.join(CURRENT_DIR, "anime_clean_beach.jpg")
POLLUTED_ANIME_IMG = os.path.join(CURRENT_DIR, "anime_polluted_beach.jpg")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

@st.cache_data
def load_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

pipeline = load_model()
clean_anime_b64 = load_image_base64(CLEAN_ANIME_IMG)
polluted_anime_b64 = load_image_base64(POLLUTED_ANIME_IMG)

# ==========================================
# 3. CALIBRATED HARDWARE PROFILES (100% ENGLISH)
# ==========================================
PRESETS = {
    "🚰 Municipal Treated Tap Water (Safe)": {
        "ph": 7.40, "Hardness": 150.0, "Solids": 4500.0,
        "Chloramines": 2.8, "Sulfate": 190.0, "Conductivity": 320.0,
        "Organic_carbon": 6.0, "Trihalomethanes": 30.0, "Turbidity": 1.1,
        "description": "Standard municipal drinking water compliant with WHO guidelines."
    },
    "🧴 RO Purified Mineral Water (Pristine)": {
        "ph": 7.15, "Hardness": 70.0, "Solids": 1400.0,
        "Chloramines": 0.6, "Sulfate": 120.0, "Conductivity": 160.0,
        "Organic_carbon": 2.0, "Trihalomethanes": 10.0, "Turbidity": 0.4,
        "description": "Multi-stage purified water with optimal mineral equilibrium."
    },
    "🏖️ Coastal Ocean Seawater (Unsafe)": {
        "ph": 8.20, "Hardness": 320.0, "Solids": 24000.0,
        "Chloramines": 6.5, "Sulfate": 460.0, "Conductivity": 780.0,
        "Organic_carbon": 17.0, "Trihalomethanes": 75.0, "Turbidity": 4.8,
        "description": "High salinity, elevated dissolved solids, unfit for consumption."
    },
    "🏞️ Turbid River / Pond Water (Contaminated)": {
        "ph": 6.10, "Hardness": 250.0, "Solids": 15000.0,
        "Chloramines": 6.0, "Sulfate": 340.0, "Conductivity": 540.0,
        "Organic_carbon": 16.0, "Trihalomethanes": 70.0, "Turbidity": 5.2,
        "description": "High suspended silt and organic sediments requiring filtration."
    },
    "🌧️ Fresh Harvested Rainwater (Clean)": {
        "ph": 6.60, "Hardness": 110.0, "Solids": 6200.0,
        "Chloramines": 2.0, "Sulfate": 200.0, "Conductivity": 300.0,
        "Organic_carbon": 7.0, "Trihalomethanes": 25.0, "Turbidity": 2.1,
        "description": "Atmospheric precipitation with low mineral and chemical load."
    },
    "🏭 Industrial Chemical Effluent (Hazardous)": {
        "ph": 4.80, "Hardness": 340.0, "Solids": 24800.0,
        "Chloramines": 12.8, "Sulfate": 490.0, "Conductivity": 790.0,
        "Organic_carbon": 24.0, "Trihalomethanes": 120.0, "Turbidity": 7.0,
        "description": "Critical chemical pollution exceeding maximum allowable thresholds."
    }
}

# Session State Setup
if "page" not in st.session_state:
    st.session_state.page = "input"
if "active_preset" not in st.session_state:
    st.session_state.active_preset = "🚰 Municipal Treated Tap Water (Safe)"
    for k, v in PRESETS["🚰 Municipal Treated Tap Water (Safe)"].items():
        if k != "description":
            st.session_state[f"input_{k}"] = float(v)

def on_preset_selected():
    chosen = st.session_state.preset_dropdown
    st.session_state.active_preset = chosen
    for k, v in PRESETS[chosen].items():
        if k != "description":
            st.session_state[f"input_{k}"] = float(v)

# =========================================================================
# SCREEN 1: HARDWARE SENSOR INPUT CONSOLE (100% ENGLISH)
# =========================================================================
if st.session_state.page == "input":
    # Header with Hardware Branding
    st.markdown("""
    <div class="hardware-header">
        <div class="hardware-badge">HARDWARE SENSOR CONSOLE • MODEL: AP-9000 PRO</div>
        <h1>⚡ AquaProbe Pro-9™</h1>
        <p>High-Precision Physicochemical Water Quality Analyzer &amp; Environmental Diagnostics Console</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Preset Selector Box
    st.markdown("""
    <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:16px; padding:1.1rem 1.6rem; margin-bottom:1.5rem; max-width:1100px; margin-left:auto; margin-right:auto;">
        <span style="font-weight:700; color:#0f172a; font-size:1.05rem;">🔬 Calibrated Sensor Channel Preset:</span>
    </div>
    """, unsafe_allow_html=True)
    
    preset_col1, preset_col2, preset_col3 = st.columns([1, 4, 1])
    with preset_col2:
        st.selectbox(
            "Select Calibrated Sensor Profile",
            options=list(PRESETS.keys()),
            key="preset_dropdown",
            on_change=on_preset_selected,
            label_visibility="collapsed"
        )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # 3-Column 9 Sensor Channels
    st.markdown("<div class=''>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("<div class='section-title'>🧪 Channel 1–3: Physical & Acidity</div>", unsafe_allow_html=True)
        in_ph = st.slider("pH Sensor (Acidity / Alkalinity)", 0.0, 14.0, step=0.1, key="input_ph", help="Safe WHO drinking range: 6.5 – 8.5")
        in_hard = st.slider("Hardness Sensor (mg/L CaCO₃)", 50.0, 450.0, step=1.0, key="input_Hardness", help="Safe desirable limit: < 200 mg/L")
        in_sol = st.slider("Solids Sensor / TDS (ppm)", 300.0, 25000.0, step=100.0, key="input_Solids", help="Total dissolved mineral solids.")

    with col2:
        st.markdown("<div class='section-title'>🧴 Channel 4–6: Disinfectants & Salts</div>", unsafe_allow_html=True)
        in_chlor = st.slider("Chloramines Sensor (ppm)", 0.1, 15.0, step=0.1, key="input_Chloramines", help="EPA standard: ≤ 4.0 ppm")
        in_sulf = st.slider("Sulfate Sensor (mg/L)", 100.0, 600.0, step=5.0, key="input_Sulfate", help="WHO drinking limit: ≤ 250 mg/L")
        in_cond = st.slider("Conductivity Sensor (µS/cm)", 100.0, 900.0, step=5.0, key="input_Conductivity", help="Electrical conductivity of dissolved ions.")

    with col3:
        st.markdown("<div class='section-title'>🌫️ Channel 7–9: Organics & Optics</div>", unsafe_allow_html=True)
        in_org = st.slider("Organic Carbon - TOC (ppm)", 1.0, 30.0, step=0.5, key="input_Organic_carbon", help="Clean water target: < 10 ppm")
        in_thm = st.slider("Trihalomethanes - THMs (µg/L)", 0.0, 140.0, step=1.0, key="input_Trihalomethanes", help="EPA maximum limit: ≤ 80 µg/L")
        in_turb = st.slider("Turbidity Optical Sensor (NTU)", 0.1, 8.0, step=0.1, key="input_Turbidity", help="Cloudiness level. Pure drinking target: < 3.0 NTU")

    st.markdown("</div>", unsafe_allow_html=True)

    # Big Centered Hardware Action Button
    st.markdown("<div style='text-align: center; margin-top: 1rem;'>", unsafe_allow_html=True)
    if st.button("⚡ Execute Sensor Diagnostic & Launch Anime Simulation"):
        sample_data = {
            "ph": in_ph,
            "Hardness": in_hard,
            "Solids": in_sol,
            "Chloramines": in_chlor,
            "Sulfate": in_sulf,
            "Conductivity": in_cond,
            "Organic_carbon": in_org,
            "Trihalomethanes": in_thm,
            "Turbidity": in_turb
        }
        sample_df = pd.DataFrame([sample_data])
        
        # ML Pipeline Prediction
        if pipeline is not None:
            prob = float(pipeline.predict_proba(sample_df)[0][1])
        else:
            prob = 0.50
            
        is_potable = prob >= 0.50
        
        # Actionable Engineering Remediation Advice
        tips = []
        if in_sol > 10000:
            tips.append("High Dissolved Solids (TDS) 👉 Reverse Osmosis (RO) Membrane Filtration Required")
        if in_turb > 3.0:
            tips.append("Elevated Cloudiness / Turbidity 👉 Dual-Media Sediment Filtration & Boiling Required")
        if in_chlor > 4.0:
            tips.append("Excess Chlorine / Chloramines 👉 Granular Activated Carbon (GAC) Filter Needed")
        if in_ph < 6.5:
            tips.append("Acidic Water (pH < 6.5) 👉 Calcite Mineral Neutralization Required")
        elif in_ph > 8.5:
            tips.append("Alkaline Water (pH > 8.5) 👉 Acid Neutralization Balancing Required")
            
        if not tips:
            tips.append("All Parameters Compliant with WHO Drinking Water Guidelines! Safe for direct consumption.")
            
        st.session_state.result_data = {
            "prob_pct": round(prob * 100, 1),
            "is_potable": is_potable,
            "sample": sample_data,
            "tips": tips
        }
        st.session_state.page = "result"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# SCREEN 2: FULL-SCREEN HARDWARE 2D ANIME NATURE SIMULATION
# =========================================================================
else:
    res = st.session_state.get("result_data", {})
    prob_pct = res.get("prob_pct", 50.0)
    is_potable = res.get("is_potable", True)
    sample = res.get("sample", {})
    tips = res.get("tips", [])

    # Select 2D Anime Artwork Background
    bg_b64 = clean_anime_b64 if is_potable else polluted_anime_b64

    # Full-Screen 2D Anime Component with Hardware Telemetry HUD
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&family=Outfit:wght@600;700;800;900&family=JetBrains+Mono:wght@600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            body, html {{
                width: 100%;
                height: 100%;
                overflow: hidden;
                font-family: 'Plus Jakarta Sans', sans-serif;
                background: #020617;
            }}

            /* Full-Screen Anime Viewport */
            #anime-viewport {{
                position: relative;
                width: 100%;
                height: 100vh;
                min-height: 720px;
                overflow: hidden;
                background: #020617;
            }}

            /* 2D Anime Background Art with Slow Manga Camera Drift */
            .anime-bg {{
                position: absolute;
                top: -6%;
                left: -6%;
                width: 112%;
                height: 112%;
                background-image: url('data:image/jpeg;base64,{bg_b64}');
                background-size: cover;
                background-position: center;
                animation: animeCameraPan 26s ease-in-out infinite alternate;
                transform-origin: center center;
            }}

            @keyframes animeCameraPan {{
                0% {{
                    transform: scale(1.0) translate(0%, 0%);
                }}
                50% {{
                    transform: scale(1.05) translate(-2.8%, -1.5%);
                }}
                100% {{
                    transform: scale(1.02) translate(2.0%, -0.8%);
                }}
            }}

            /* Warm Golden Hour Anime Sunlight Rays */
            .anime-sunburst {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                mix-blend-mode: screen;
                opacity: {('0.75' if is_potable else '0.25')};
                background: radial-gradient(circle at 78% 22%, rgba(254, 240, 138, 0.85) 0%, rgba(251, 191, 36, 0.4) 30%, rgba(0, 0, 0, 0) 65%);
            }}

            /* Dystopian Smog / Acid Fog Overlay (If Unsafe) */
            .dystopian-haze {{
                display: {('none' if is_potable else 'block')};
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                background: linear-gradient(180deg, rgba(88, 28, 135, 0.25) 0%, rgba(180, 83, 9, 0.35) 50%, rgba(69, 26, 3, 0.55) 100%);
                mix-blend-mode: multiply;
            }}

            /* Canvas for Anime Water Sparkles & Light Shimmer */
            #anime-sparkle-canvas {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }}

            /* 2D Manga / Anime Seagulls in Flight */
            .anime-birds {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                overflow: hidden;
            }}

            .anime-bird {{
                position: absolute;
                width: 32px;
                height: 18px;
                opacity: 0.92;
                animation: animeBirdFlight 18s linear infinite;
            }}
            .anime-bird svg {{
                width: 100%;
                height: 100%;
                fill: #0f172a;
                animation: animeWingFlap 0.65s ease-in-out infinite alternate;
                transform-origin: center;
            }}

            @keyframes animeWingFlap {{
                0% {{ transform: scaleY(1); }}
                100% {{ transform: scaleY(0.28); }}
            }}

            @keyframes animeBirdFlight {{
                0% {{ transform: translate(-100px, 0px) scale(0.7); }}
                50% {{ transform: translate(700px, -40px) scale(0.95); }}
                100% {{ transform: translate(1400px, -70px) scale(1.15); }}
            }}

            /* Hardware Telemetry Header OSD */
            .hardware-telemetry {{
                position: absolute;
                top: 20px;
                left: 36px;
                color: rgba(255, 255, 255, 0.95);
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                background: rgba(15, 23, 42, 0.75);
                backdrop-filter: blur(14px);
                padding: 6px 18px;
                border-radius: 30px;
                border: 1px solid rgba(56, 189, 248, 0.35);
                display: flex;
                align-items: center;
                gap: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            }}
            .telemetry-dot {{
                width: 9px;
                height: 9px;
                background: #38bdf8;
                border-radius: 50%;
                box-shadow: 0 0 10px #38bdf8;
                animation: dotPulse 2s infinite;
            }}
            @keyframes dotPulse {{
                0%, 100% {{ opacity: 1; transform: scale(1); }}
                50% {{ opacity: 0.4; transform: scale(0.8); }}
            }}

            /* Full-Screen Floating HUD Cards */
            .anime-hud-layout {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                display: flex;
                justify-content: space-between;
                padding: 75px 36px 36px 36px;
            }}

            .hud-glass-card {{
                pointer-events: auto;
                background: rgba(15, 23, 42, 0.72);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border: 1px solid rgba(255, 255, 255, 0.24);
                border-radius: 24px;
                padding: 26px 28px;
                color: white;
                box-shadow: 0 25px 60px rgba(0,0,0,0.45);
                transition: transform 0.25s ease, box-shadow 0.25s ease;
            }}
            .hud-glass-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 30px 70px rgba(0,0,0,0.55);
            }}

            .hud-left {{
                width: 360px;
                align-self: flex-start;
            }}
            .hud-right {{
                width: 380px;
                align-self: flex-start;
            }}

            .status-badge {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 6px 16px;
                border-radius: 30px;
                font-family: 'Outfit', sans-serif;
                font-size: 0.95rem;
                font-weight: 800;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                margin-bottom: 14px;
            }}
            .badge-safe {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                box-shadow: 0 0 24px rgba(16, 185, 129, 0.65);
            }}
            .badge-danger {{
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                box-shadow: 0 0 24px rgba(239, 68, 68, 0.65);
            }}

            .card-title {{
                font-family: 'Outfit', sans-serif;
                font-size: 2.2rem;
                font-weight: 900;
                line-height: 1.1;
                margin-bottom: 8px;
            }}
            .card-desc {{
                font-size: 0.95rem;
                opacity: 0.92;
                line-height: 1.45;
                margin-bottom: 18px;
            }}

            .meter-track {{
                background: rgba(255, 255, 255, 0.16);
                border-radius: 12px;
                height: 14px;
                overflow: hidden;
                position: relative;
                margin-bottom: 10px;
            }}
            .meter-fill {{
                height: 100%;
                border-radius: 12px;
                transition: width 1.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            .fill-safe {{
                background: linear-gradient(90deg, #10b981, #34d399);
                box-shadow: 0 0 14px #34d399;
            }}
            .fill-danger {{
                background: linear-gradient(90deg, #ef4444, #f87171);
                box-shadow: 0 0 14px #f87171;
            }}

            .audit-row {{
                display: flex;
                justify-content: space-between;
                padding: 7px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.12);
                font-size: 0.92rem;
            }}
            .audit-name {{ opacity: 0.85; }}
            .audit-val {{ font-weight: 700; }}

            .remedy-box {{
                margin-top: 16px;
                background: rgba(255, 255, 255, 0.12);
                border-radius: 14px;
                padding: 14px 16px;
                font-size: 0.9rem;
                line-height: 1.4;
                border-left: 3.5px solid #38bdf8;
            }}

            .anime-tag {{
                font-size: 0.85rem;
                font-weight: 700;
                padding: 5px 12px;
                border-radius: 8px;
                display: inline-block;
                margin-top: 8px;
            }}
            .tag-safe {{ background: rgba(16, 185, 129, 0.28); color: #6ee7b7; border: 1px solid #10b981; }}
            .tag-danger {{ background: rgba(239, 68, 68, 0.28); color: #fca5a5; border: 1px solid #ef4444; }}
        </style>
    </head>
    <body>
        <div id="anime-viewport">
            <!-- 2D Anime Background Painting -->
            <div class="anime-bg"></div>

            <!-- Warm Shinkai Sunlight Rays / Dystopian Smog Overlay -->
            <div class="anime-sunburst"></div>
            <div class="dystopian-haze"></div>

            <!-- Sparkling Anime Caustics & Water Shimmer Canvas -->
            <canvas id="anime-sparkle-canvas"></canvas>

            <!-- 2D Anime Seagulls -->
            <div class="anime-birds">
                <div class="anime-bird" style="top: 90px; animation-delay: 0s;">
                    <svg viewBox="0 0 24 12"><path d="M0,6 Q6,0 12,6 Q18,0 24,6 Q18,3 12,8 Q6,3 0,6 Z"/></svg>
                </div>
                <div class="anime-bird" style="top: 120px; animation-delay: 5s;">
                    <svg viewBox="0 0 24 12"><path d="M0,6 Q6,0 12,6 Q18,0 24,6 Q18,3 12,8 Q6,3 0,6 Z"/></svg>
                </div>
                <div class="anime-bird" style="top: 65px; animation-delay: 9.5s;">
                    <svg viewBox="0 0 24 12"><path d="M0,6 Q6,0 12,6 Q18,0 24,6 Q18,3 12,8 Q6,3 0,6 Z"/></svg>
                </div>
                <div class="anime-bird" style="top: 150px; animation-delay: 14s;">
                    <svg viewBox="0 0 24 12"><path d="M0,6 Q6,0 12,6 Q18,0 24,6 Q18,3 12,8 Q6,3 0,6 Z"/></svg>
                </div>
            </div>

            <!-- Hardware Telemetry Header -->
            <div class="hardware-telemetry">
                <span class="telemetry-dot"></span>
                <span>AQUAPROBE PRO-9 SENSOR TELEMETRY &bull; GRID ACTIVE &bull; 60 FPS</span>
            </div>

            <!-- Floating Glassmorphic HUD -->
            <div class="anime-hud-layout">
                <!-- Left HUD Card -->
                <div class="hud-glass-card hud-left">
                    <div class="status-badge {('badge-safe' if is_potable else 'badge-danger')}">
                        {('✅ POTABLE (SAFE TO DRINK)' if is_potable else '🚫 HAZARDOUS (NOT POTABLE)')}
                    </div>
                    <div class="card-title">
                        {('Pristine Water' if is_potable else 'Toxic Effluent')}
                    </div>
                    <div class="card-desc">
                        {('AquaProbe optical spectrometry verifies crystal-clear river meeting ocean. Safe for human consumption.' if is_potable else 'Critical chemical contamination detected across sensor channels. River water is hazardous.')}
                    </div>
                    
                    <div style="display:flex; justify-content:space-between; font-size:0.88rem; font-weight:700; margin-bottom:5px;">
                        <span>Drinking Safety Probability:</span>
                        <span>{prob_pct}%</span>
                    </div>
                    <div class="meter-track">
                        <div class="meter-fill {('fill-safe' if is_potable else 'fill-danger')}" style="width: {prob_pct}%;"></div>
                    </div>

                    <div class="anime-tag {('tag-safe' if is_potable else 'tag-danger')}">
                        🌊 Sensor Visualization: {('Sparkling Anime Turquoise Water' if is_potable else 'Murky Muddy Water with Acidic Smog')}
                    </div>
                </div>

                <!-- Right HUD Card -->
                <div class="hud-glass-card hud-right">
                    <h3 style="font-family:'Outfit'; font-size:1.2rem; font-weight:800; margin-bottom:14px; color:#38bdf8;">
                        📋 Sensor Channel Telemetry
                    </h3>
                    <div class="audit-row">
                        <span class="audit-name">🧪 pH Channel</span>
                        <span class="audit-val">{sample.get('ph', 7.0):.1f}</span>
                    </div>
                    <div class="audit-row">
                        <span class="audit-name">🧂 Solids / TDS Channel</span>
                        <span class="audit-val">{sample.get('Solids', 5000):.0f} ppm</span>
                    </div>
                    <div class="audit-row">
                        <span class="audit-name">🌫️ Turbidity Sensor</span>
                        <span class="audit-val">{sample.get('Turbidity', 1.5):.1f} NTU</span>
                    </div>
                    <div class="audit-row">
                        <span class="audit-name">🧴 Chloramines Channel</span>
                        <span class="audit-val">{sample.get('Chloramines', 2.0):.1f} ppm</span>
                    </div>
                    <div class="audit-row">
                        <span class="audit-name">🧪 Sulfate Sensor</span>
                        <span class="audit-val">{sample.get('Sulfate', 200):.0f} mg/L</span>
                    </div>

                    <div class="remedy-box">
                        <b>💡 Actionable Recommendation:</b><br>
                        {"<br>".join(tips[:2])}
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Live 2D Anime Sparkle Stars & Shimmer Canvas Animation
            const canvas = document.getElementById('anime-sparkle-canvas');
            const ctx = canvas.getContext('2d');
            const isSafe = {str(is_potable).lower()};

            function resize() {{
                const viewport = document.getElementById('anime-viewport');
                canvas.width = viewport.clientWidth;
                canvas.height = viewport.clientHeight;
            }}
            window.addEventListener('resize', resize);
            resize();

            let time = 0;

            function drawAnimeSparkle(cx, cy, spikes, outerRadius, innerRadius, color) {{
                let rot = Math.PI / 2 * 3;
                let x = cx;
                let y = cy;
                const step = Math.PI / spikes;

                ctx.save();
                ctx.beginPath();
                ctx.moveTo(cx, cy - outerRadius);

                for (let i = 0; i < spikes; i++) {{
                    x = cx + Math.cos(rot) * outerRadius;
                    y = cy + Math.sin(rot) * outerRadius;
                    ctx.lineTo(x, y);
                    rot += step;

                    x = cx + Math.cos(rot) * innerRadius;
                    y = cy + Math.sin(rot) * innerRadius;
                    ctx.lineTo(x, y);
                    rot += step;
                }}
                ctx.lineTo(cx, cy - outerRadius);
                ctx.closePath();
                ctx.fillStyle = color;
                ctx.shadowColor = color;
                ctx.shadowBlur = 10;
                ctx.fill();
                ctx.restore();
            }}

            function animateAnimeScene() {{
                time += 0.04;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                const w = canvas.width;
                const h = canvas.height;
                const waterTop = h * 0.44;

                if (isSafe) {{
                    for (let i = 0; i < 22; i++) {{
                        const sx = (w * 0.32) + Math.sin(time * 1.6 + i * 2.2) * (w * 0.42);
                        const sy = waterTop + (i * 12) + Math.cos(time * 2.0 + i) * 8;
                        const sparkleSize = Math.abs(Math.sin(time * 3.5 + i * 1.5)) * 6.5 + 1.2;

                        drawAnimeSparkle(sx, sy, 4, sparkleSize, sparkleSize * 0.28, 'rgba(255, 255, 255, 0.95)');
                    }}

                    for (let p = 0; p < 15; p++) {{
                        const px = (w * 0.15 + (time * 25 + p * 80)) % w;
                        const py = (h * 0.25) + Math.sin(time * 1.2 + p) * 45;
                        const pRadius = Math.sin(time * 2 + p) * 1.8 + 2.2;

                        ctx.fillStyle = 'rgba(254, 240, 138, 0.65)';
                        ctx.shadowColor = '#fef08a';
                        ctx.shadowBlur = 8;
                        ctx.beginPath();
                        ctx.arc(px, py, pRadius, 0, Math.PI * 2);
                        ctx.fill();
                    }}
                }} else {{
                    for (let i = 0; i < 35; i++) {{
                        const dx = (w * 0.2 + (time * 15 + i * 50)) % w;
                        const dy = waterTop + (i * 9) + Math.cos(time * 0.8 + i) * 6;
                        
                        ctx.fillStyle = i % 2 === 0 ? 'rgba(30, 20, 10, 0.75)' : 'rgba(217, 119, 6, 0.6)';
                        ctx.beginPath();
                        ctx.arc(dx, dy, (i % 3) + 2.5, 0, Math.PI * 2);
                        ctx.fill();
                    }}
                }}

                requestAnimationFrame(animateAnimeScene);
            }}

            animateAnimeScene();
        </script>
    </body>
    </html>
    """

    # Full-Screen HTML5 Component (100% Height & Width)
    st.components.v1.html(html_code, height=750, scrolling=False)

    # Bottom Center "Test Another Water Sample" Action Button (100% English)
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns([1, 1.6, 1])
    with b_col2:
        if st.button("🔄 Recalibrate & Test New Water Sample", width="stretch"):
            st.session_state.page = "input"
            st.rerun()
