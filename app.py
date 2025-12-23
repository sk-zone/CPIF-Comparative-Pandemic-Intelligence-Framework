import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import warnings
import re

warnings.filterwarnings('ignore')

# ============================================
# 1. Page Config & Visual Theme (Refined)
# ============================================
st.set_page_config(page_title="Pakistan COVID-19 Intelligence Dashboard", layout="wide", page_icon="🇵🇰")

# Merging "Visual Theme": Dark/Transparent Polish
st.markdown("""
    <style>
    /* Executive Metric Cards */
    .metric-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        border-left: 5px solid #3b82f6; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 10px; 
    }
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        border-radius: 4px 4px 0px 0px; 
        gap: 1px; 
        padding-top: 10px; 
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: rgba(59, 130, 246, 0.1); 
        border-bottom: 2px solid #3b82f6; 
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🇵🇰 Executive Pandemic Intelligence Dashboard - Pakistan")
st.markdown("**Prepared By Studenta of Applied AI in Public Heatlhcare** | Advanced Insights, AI Projections & Scenario Calculator")

# ============================================
# 2. Robust Data Loading (The "Fix")
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please upload 'Refined + New entities.csv'")
        return pd.DataFrame()

    # ROBUST CLEANING (Replaces fragile .replace chain)
    df.columns = (df.columns.astype(str)
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_'))

    # Mapping columns to standard names
    col_map = {
        'grand_total_cases_till_date': 'cases',
        'clinic_total_numbers_recovered_and_discharged_so_far': 'recovered',
        'death_cumulative_total_deaths': 'deaths',
        'clinic_total_no_of_covid_patients_currently_admitted': 'admitted',
        'clinic_total_no_of_beds_allocated_for_covid_patients': 'beds_total',
        'clinic_total_on_oxygen': 'oxygen_patients',
        'clinic_total_no_of_beds_with_oxygen_facility_allocated_for_covid_patients': 'oxygen_beds_total',
        'clinic_total_no_of_patients_currently_on_ventilator': 'vent_patients',
        'clinic_total_no_of_ventilators_allocated_for_covid_patients': 'vents_total',
        'test_positivity_ratio': 'positivity',
        'grand_total_tests_conducted_till_date': 'tests',
        'oxygen_dependency_ratio': 'oxygen_dependency'
    }
    
    # Rename columns if they exist
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})

    # Numeric Conversion
    numeric_cols = list(col_map.values())
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace(',', '', regex=False)
                       .str.replace('%', '', regex=False)
                       .replace(['N/A', 'n/a', '-', '', 'nan'], '0'))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Date Parsing
    if 'date' in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by=["province", "date"]).reset_index(drop=True)

    # --- Refined Formulas ---
    # 1. Active Cases
    df["active_cases"] = (df["cases"] - df["recovered"] - df["deaths"]).clip(lower=0)
    
    # 2. Daily Diffs
    df["new_cases"] = df.groupby("province")["cases"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby("province")["deaths"].diff().fillna(0).clip(lower=0)
    df["new_recoveries"] = df.groupby("province")["recovered"].diff().fillna(0).clip(lower=0)
    
    # 3. 7-Day Moving Average (Smoother)
    df["new_cases_7da"] = df.groupby("province")["new_cases"].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    
    # 4. Rt Estimate (Growth Rate Proxy)
    shifted = df.groupby("province")["new_cases_7da"].shift(4).replace(0, 1)
    df["rt_estimate"] = (df["new_cases_7da"] / shifted).clip(0, 5).fillna(1.0)

    # 5. Rates
    df["recovery_rate"] = (df["recovered"] / df["cases"] * 100).fillna(0)
    df["fatality_rate"] = (df["deaths"] / df["cases"] * 100).fillna(0)
    
    # Fix Oxygen Dependency > 100%
    if 'oxygen_dependency' in df.columns:
         df.loc[df['oxygen_dependency'] > 100, 'oxygen_dependency'] = 100

    # Define Waves
    df["wave"] = "Other"
    df.loc[df["date"].between("2020-03-01", "2020-05-31"), "wave"] = "Wave 1 (Mar-May 2020)"
    df.loc[df["date"].between("2020-06-01", "2020-08-31"), "wave"] = "Wave 2 (Jun-Aug 2020)"
    df.loc[df["date"].between("2020-11-01", "2021-01-31"), "wave"] = "Wave 3 (Nov 2020-Jan 2021)"
    df.loc[df["date"].between("2021-04-01", "2021-06-30"), "wave"] = "Wave 4 (Delta)"
    df.loc[df["date"].between("2021-07-01", "2021-09-30"), "wave"] = "Wave 4 Peak"
    df.loc[df["date"].between("2022-01-01", "2022-03-31"), "wave"] = "Wave 5 (Omicron)"

    return df

df = load_data()

if df.empty:
    st.stop()

# ============================================
# 3. Sidebar
# ============================================
st.sidebar.header("Executive Controls")
provinces = ["All"] + sorted(df["province"].unique().tolist())
province = st.sidebar.selectbox("Select Province", provinces, index=0)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("Select Date Range", [max_date - timedelta(days=90), max_date], min_value=min_date, max_value=max_date)

# Filtering
mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
df_filtered = df[mask].copy()
if province != "All":
    df_filtered = df_filtered[df_filtered["province"] == province]

# ============================================
# 4. KPIs (Group's Layout, Refined Data)
# ============================================
latest = df_filtered.iloc[-1] if not df_filtered.empty else pd.Series()

# Safe Getters
def get_val(key, default=0): return latest.get(key, default)

st.header("Key Performance Indicators")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Cases", f"{int(get_val('cases')):,}", delta=f"{int(df_filtered['new_cases'].sum()):,} new")
c2.metric("Active Cases", f"{int(get_val('active_cases')):,}")
c3.metric("Recovered", f"{int(get_val('recovered')):,}", delta=f"{get_val('recovery_rate'):.1f}%")
c4.metric("Deaths", f"{int(get_val('deaths')):,}", delta=f"{get_val('fatality_rate'):.2f}%")
c5.metric("Positivity Rate", f"{get_val('positivity'):.2f}%")

c6, c7, c8 = st.columns(3)
bed_util = (get_val('admitted') / max(1, get_val('beds_total'))) * 100
vent_util = (get_val('vent_patients') / max(1, get_val('vents_total'))) * 100

c6.metric("New Cases (Today)", f"{int(get_val('new_cases')):,}")
c7.metric("Bed Utilization", f"{bed_util:.1f}%")
c8.metric("Ventilator Utilization", f"{vent_util:.1f}%")

# ============================================
# 5. Tabs
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Epidemic Curves",
    "Healthcare Capacity",
    "Comparisons & Trends",
    "AI Projections (Model Comparison)",
    "Province & Wave Comparison"
])

# --- Tab 1: Epidemic Curves ---
with tab1:
    st.subheader("Epidemic Curve Analysis")
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Daily New Cases (7-Day Avg)", "Cumulative Cases", "Daily Deaths", "Positivity Rate"))
    
    # Theme: Transparent background
    layout_args = dict(height=700, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases_7da"], name='7-Day Avg', line=dict(color='#3b82f6', width=3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases"], mode='markers', name='Daily', marker=dict(color='#3b82f6', opacity=0.3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["cases"], name='Total Cases', line=dict(color='#8b5cf6')), row=1, col=2)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_deaths"], name='Daily Deaths', line=dict(color='#ef4444')), row=2, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["positivity"], name='Positivity %', line=dict(color='#f59e0b')), row=2, col=2)

    fig1.update_layout(**layout_args)
    st.plotly_chart(fig1, use_container_width=True)

# --- Tab 2: Healthcare (Merged: Group's Bars + Refined Oxygen Area) ---
with tab2:
    st.subheader("Healthcare Capacity & Utilization")
    
    # 1. Group's Bar Chart Logic
    cap_data = pd.DataFrame({
        "Metric": ["General Beds", "Oxygen Beds", "Ventilators"],
        "Used": [get_val('admitted'), get_val('oxygen_patients'), get_val('vent_patients')],
        "Available": [
            max(0, get_val('beds_total') - get_val('admitted')),
            max(0, get_val('oxygen_beds_total') - get_val('oxygen_patients')),
            max(0, get_val('vents_total') - get_val('vent_patients'))
        ]
    })
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=cap_data["Metric"], x=cap_data["Used"], name="Used", orientation='h', marker_color='#ef4444'))
    fig2.add_trace(go.Bar(y=cap_data["Metric"], x=cap_data["Available"], name="Available", orientation='h', marker_color='#10b981'))
    fig2.update_layout(barmode='stack', title="Resource Limits", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)
    
    # 2. Refined Oxygen Dependency (Important!)
    if 'oxygen_dependency' in df_filtered.columns:
        st.subheader("Oxygen Dependency (Critical Stress Indicator)")
        fig_oxy = px.area(df_filtered, x='date', y='oxygen_dependency', title="Percentage of Admitted Patients Requiring Oxygen", color_discrete_sequence=['#10B981'])
        fig_oxy.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_oxy, use_container_width=True)

# --- Tab 3: Comparisons ---
with tab3:
    st.subheader("Cross-Province Comparisons")
    # Group's Heatmap Logic
    df_month = df.copy()
    df_month["month"] = df_month["date"].dt.to_period("M").astype(str)
    monthly = df_month.pivot_table(values="new_cases", index="province", columns="month", aggfunc="sum", fill_value=0)
    fig_heat = px.imshow(monthly, title="Monthly New Cases Heatmap")
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_heat, use_container_width=True)

# --- Tab 4: AI Projections (The Comparison) ---
with tab4:
    st.subheader("AI Model Comparison: Linear vs. Forest vs. Polynomial")
    
    if len(df_filtered) > 15:
        # Data Prep
        pred_data = df_filtered[["date", "new_cases_7da"]].dropna()
        pred_data["days"] = (pred_data["date"] - pred_data["date"].min()).dt.days
        
        X = pred_data[["days"]]
        y = pred_data["new_cases_7da"]

        # 1. Linear Regression (Group's)
        lin = LinearRegression().fit(X, y)
        
        # 2. Random Forest (Group's)
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        
        # 3. Polynomial Regression (Refined - "The SOTA")
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        poly_reg = LinearRegression().fit(X_poly, y)

        # Future Grid
        future_days = np.arange(X["days"].max() + 1, X["days"].max() + 31).reshape(-1, 1)
        future_dates = [pred_data["date"].max() + timedelta(days=i) for i in range(1, 31)]
        
        # Predict & CLIP NEGATIVES (Fix)
        lin_pred = np.maximum(0, lin.predict(future_days))
        rf_pred = np.maximum(0, rf.predict(future_days))
        poly_pred = np.maximum(0, poly_reg.predict(poly.transform(future_days)))

        # Plot Comparison
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=pred_data["date"], y=y, mode='lines', name='Historical Data', line=dict(color='gray')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=lin_pred, mode='lines', name='Linear (Simple)', line=dict(dash='dash', color='red')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=rf_pred, mode='lines', name='Random Forest (No Trend)', line=dict(dash='dot', color='orange')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=poly_pred, mode='lines', name='Polynomial (Curve Aware)', line=dict(width=4, color='#10B981'))) # Highlight this!

        fig_ai.update_layout(title="30-Day Forecast Comparison", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ai, use_container_width=True)
        
        st.caption("Note: Random Forest may flatline (plateau) because it cannot extrapolate trends outside training data. Polynomial Regression captures the wave shape best.")
    else:
        st.info("Insufficient data for AI modeling.")

    # Scenario Calculator (Refined with Rt)
    st.subheader("Policy Scenario Calculator")
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        current_active = int(latest.get('new_cases_7da', 100))
        # Use calculated Rt as default
        default_r0 = float(latest.get('rt_estimate', 1.1))
        r0 = st.slider("Baseline Rt (Reproduction Rate)", 0.5, 4.0, max(0.5, default_r0), 0.1)
        reduction = st.slider("Intervention Impact (%)", 0, 60, 20)
        days_sim = st.slider("Simulation Days", 30, 90, 60)
    
    with col_s2:
        # Simulation
        proj_base = []
        proj_int = []
        val_base = current_active
        val_int = current_active
        
        # Intervention Rt
        r_int = r0 * (1 - reduction/100)
        
        for d in range(days_sim):
            # Exp Growth Formula: N_t = N_0 * R^(t/4)
            val_base = current_active * (r0 ** (d/4))
            val_int = current_active * (r_int ** (d/4))
            proj_base.append(val_base)
            proj_int.append(val_int)
            
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(y=proj_base, name=f"Baseline (Rt={r0})", line=dict(color='red', dash='dot')))
        fig_sim.add_trace(go.Scatter(y=proj_int, name=f"With Intervention (Rt={r_int:.2f})", line=dict(color='green', width=3)))
        fig_sim.update_layout(title="Impact of Intervention", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sim, use_container_width=True)

# --- Tab 5: Wave Comparison ---
with tab5:
    st.header("Wave-Wise Historical Analysis")
    # Group's Aggregation Logic
    wave_stats = df.groupby(["wave", "province"]).agg({
        "new_cases": "sum",
        "deaths": "sum",
        "cases": "max",
        "positivity": "mean"
    }).reset_index()
    
    fig_cases = px.bar(wave_stats, x="wave", y="cases", color="province", barmode="group", title="Peak Cases by Wave")
    fig_cases.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cases, use_container_width=True)

# ============================================
# 6. Risk Assessment (Stylized)
# ============================================
st.header("Automated Risk Assessment")
risk_score = 0
alerts = []

# Refined Logic
pos = get_val('positivity')
occ = bed_util
fat = get_val('fatality_rate')

if pos > 10: risk_score += 30; alerts.append(f"🔴 CRITICAL: Positivity is {pos:.1f}% (High Spread)")
elif pos > 5: risk_score += 15; alerts.append(f"🟡 WARNING: Positivity is {pos:.1f}%")

if occ > 80: risk_score += 25; alerts.append(f"🔴 CRITICAL: Hospital Capacity at {occ:.1f}%")
elif occ > 60: risk_score += 10; alerts.append(f"🟡 WARNING: Hospital Capacity at {occ:.1f}%")

if fat > 2.5: risk_score += 20; alerts.append(f"🔴 CRITICAL: Fatality Rate {fat:.2f}% (High Severity)")

st.progress(min(risk_score, 100))
if risk_score > 50:
    st.error(f"High Risk Level ({risk_score}/100) - Immediate Action Required")
else:
    st.success(f"Manageable Risk Level ({risk_score}/100)")

for a in alerts: st.warning(a)

st.markdown("---")
st.caption(f"Framework v4.7 | Generated: {datetime.now().strftime('%Y-%m-%d')}")

