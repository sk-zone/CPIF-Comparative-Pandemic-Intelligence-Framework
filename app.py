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
# 1. Page Config & Visual Theme (The "Polish")
# ============================================
st.set_page_config(
    page_title="Pakistan COVID-19 Intelligence Dashboard", 
    layout="wide", 
    page_icon="🇵🇰"
)

# Custom CSS for Executive Polish (Transparent & Clean)
st.markdown("""
    <style>
    /* Metric Cards */
    .metric-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        border-left: 5px solid #3b82f6; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 10px; 
    }
    
    /* Tabs Styling - Elegant & Transparent */
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
        color: #3b82f6; 
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🇵🇰 Executive Pandemic Intelligence Dashboard - Pakistan")
st.markdown("**Prepared By Student of AI in Public Healthcare** | Advanced Insights, AI Projections & Scenario Calculator")

# ============================================
# 2. Data Loading & Formula Injection
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except:
        return pd.DataFrame() # Return empty if file missing

    # 1. Robust Column Cleaning (Regex)
    df.columns = (df.columns.astype(str)
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_'))

    df["province"] = df["province"].str.strip().str.title()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # 2. Key Numeric Columns (Conversion)
    numeric_cols = [
        'grand_total_cases_till_date',
        'clinic_total_numbers_recovered_and_discharged_so_far',
        'death_cumulative_total_deaths',
        'clinic_total_no_of_covid_patients_currently_admitted',
        'clinic_total_no_of_beds_allocated_for_covid_patients',
        'clinic_total_on_oxygen',
        'clinic_total_no_of_beds_with_oxygen_facility_allocated_for_covid_patients',
        'clinic_total_no_of_patients_currently_on_ventilator',
        'clinic_total_no_of_ventilators_allocated_for_covid_patients',
        'test_positivity_ratio',
        'grand_total_tests_conducted_till_date',
        'healtcare_stress_index', # Using cleaned name
        'oxygen_dependency_ratio'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (df[col]
                       .astype(str)
                       .str.replace(',', '', regex=False)
                       .str.replace('%', '', regex=False)
                       .str.strip()
                       .replace(['N/A', 'n/a', '-', '', 'Not Reported', 'nan'], '0'))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.sort_values(by=["province", "date"]).reset_index(drop=True)

    # 3. Derived Metrics (The Correct Formulas)
    df["active_cases"] = (df["grand_total_cases_till_date"] -
                          df["clinic_total_numbers_recovered_and_discharged_so_far"] -
                          df["death_cumulative_total_deaths"]).clip(lower=0)

    # Daily Diffs
    df["new_cases"] = df.groupby("province")["grand_total_cases_till_date"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby("province")["death_cumulative_total_deaths"].diff().fillna(0).clip(lower=0)
    df["new_recoveries"] = df.groupby("province")["clinic_total_numbers_recovered_and_discharged_so_far"].diff().fillna(0).clip(lower=0)
    
    # Rates
    df["recovery_rate"] = (df["clinic_total_numbers_recovered_and_discharged_so_far"] / df["grand_total_cases_till_date"] * 100).fillna(0)
    df["fatality_rate"] = (df["death_cumulative_total_deaths"] / df["grand_total_cases_till_date"] * 100).fillna(0)
    
    # Smoothers (7-Day Avg)
    df["new_cases_7da"] = df.groupby("province")["new_cases"].rolling(7).mean().reset_index(0, drop=True).fillna(0)

    # Rt (Reproduction Number) Proxy
    # Formula: (Cases Today / Cases 4 Days Ago)
    shifted_cases = df.groupby("province")["new_cases_7da"].shift(4).replace(0, 1)
    df["rt_estimate"] = (df["new_cases_7da"] / shifted_cases).fillna(1.0).clip(upper=4.0)

    # Healthcare Stress (If column missing, calculate it)
    if 'healtcare_stress_index' not in df.columns or df['healtcare_stress_index'].sum() == 0:
        # Formula: Vent Patients / Vent Capacity
        df['healtcare_stress_index'] = (df['clinic_total_no_of_patients_currently_on_ventilator'] / 
                                        df['clinic_total_no_of_ventilators_allocated_for_covid_patients'].replace(0, 1) * 100)

    # Oxygen Dependency (If column missing, calculate it)
    if 'oxygen_dependency_ratio' not in df.columns or df['oxygen_dependency_ratio'].sum() == 0:
        # Formula: Oxygen Patients / Total Admitted
        df['oxygen_dependency_ratio'] = (df['clinic_total_on_oxygen'] / 
                                         df['clinic_total_no_of_covid_patients_currently_admitted'].replace(0, 1) * 100).clip(upper=100)

    # Define Waves
    df["wave"] = "Other Periods"
    df.loc[df["date"].between("2020-03-01", "2020-05-31"), "wave"] = "Wave 1 (Mar-May 2020)"
    df.loc[df["date"].between("2020-06-01", "2020-08-31"), "wave"] = "Wave 2 (Jun-Aug 2020)"
    df.loc[df["date"].between("2020-11-01", "2021-01-31"), "wave"] = "Wave 3 (Nov 2020-Jan 2021)"
    df.loc[df["date"].between("2021-04-01", "2021-06-30"), "wave"] = "Wave 4 (Delta - Apr-Jun 2021)"
    df.loc[df["date"].between("2021-07-01", "2021-09-30"), "wave"] = "Wave 4 Peak (Jul-Sep 2021)"
    df.loc[df["date"].between("2022-01-01", "2022-03-31"), "wave"] = "Wave 5 (Omicron - Jan-Mar 2022)"

    return df

df = load_data()

if df.empty:
    st.error("Data file not found! Please upload 'Refined + New entities.csv'.")
    st.stop()

# ============================================
# Sidebar Controls
# ============================================
st.sidebar.header("Executive Controls")

provinces = ["All"] + sorted(df["province"].unique().tolist())
province = st.sidebar.selectbox("Select Province", provinces, index=0)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    [max_date - timedelta(days=90), max_date],
    min_value=min_date,
    max_value=max_date
)

# Filter logic
mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
df_filtered = df[mask].copy()
if province != "All":
    df_filtered = df_filtered[df_filtered["province"] == province]

# ============================================
# KPIs
# ============================================
latest = df_filtered.iloc[-1] if not df_filtered.empty else pd.Series()
prev = df_filtered.iloc[-8] if len(df_filtered) > 8 else latest

total_cases = int(latest.get("grand_total_cases_till_date", 0))
active_cases = int(latest.get("active_cases", 0))
total_deaths = int(latest.get("death_cumulative_total_deaths", 0))
rt_val = float(latest.get("rt_estimate", 1.0))
positivity = float(latest.get("test_positivity_ratio", 0))

st.header("Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Cases", f"{total_cases:,}", delta=f"{int(df_filtered['new_cases'].sum()):,} new")
col2.metric("Active Cases", f"{active_cases:,}")
col3.metric("Rt (Spread Speed)", f"{rt_val:.2f}", delta="Growing" if rt_val > 1 else "Shrinking", delta_color="inverse")
col4.metric("Deaths", f"{total_deaths:,}", delta=f"{int(latest.get('new_deaths', 0))} new")
col5.metric("Positivity Rate", f"{positivity:.2f}%", delta=f"{positivity - prev.get('test_positivity_ratio', 0):.2f}%", delta_color="inverse")

# ============================================
# Tabs
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Epidemic Curves",
    "🏥 Healthcare & Capacity",
    "📊 Comparisons & Trends",
    "🤖 AI Model Comparison",
    "🗓️ Wave Analysis"
])

# --- Tab 1: Epidemic Curves ---
with tab1:
    st.subheader("Epidemic Curve Analysis")
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Daily New Cases (7-Day Avg)", "Cumulative Cases", "Daily Deaths", "Positivity Rate"))

    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases_7da"], mode='lines', name='7-Day Avg', line=dict(color='#3b82f6', width=3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases"], mode='markers', name='Daily', marker=dict(color='#3b82f6', opacity=0.3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["grand_total_cases_till_date"], mode='lines', name='Total Cases', line=dict(color='#8b5cf6')), row=1, col=2)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_deaths"], mode='lines+markers', name='Daily Deaths', line=dict(color='#ef4444')), row=2, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["test_positivity_ratio"], mode='lines', name='Positivity %', line=dict(color='#f59e0b')), row=2, col=2)

    # Transparent Theme Applied
    fig1.update_layout(height=700, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

# --- Tab 2: Healthcare (Refined Formulas) ---
with tab2:
    st.subheader("Healthcare Capacity & Utilization")
    
    # Safe calc for utilization
    admitted = latest.get("clinic_total_no_of_covid_patients_currently_admitted", 0)
    beds_cap = latest.get("clinic_total_no_of_beds_allocated_for_covid_patients", 1)
    oxy_pts = latest.get("clinic_total_on_oxygen", 0)
    oxy_cap = latest.get("clinic_total_no_of_beds_with_oxygen_facility_allocated_for_covid_patients", 1)
    vent_pts = latest.get("clinic_total_no_of_patients_currently_on_ventilator", 0)
    vent_cap = latest.get("clinic_total_no_of_ventilators_allocated_for_covid_patients", 1)

    capacity_df = pd.DataFrame({
        "Metric": ["General Beds", "Oxygen Beds", "Ventilators"],
        "Used": [admitted, oxy_pts, vent_pts],
        "Available": [max(0, beds_cap - admitted), max(0, oxy_cap - oxy_pts), max(0, vent_cap - vent_pts)]
    })

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=capacity_df["Metric"], x=capacity_df["Used"], name="Used", orientation='h', marker_color='#ef4444'))
    fig2.add_trace(go.Bar(y=capacity_df["Metric"], x=capacity_df["Available"], name="Available", orientation='h', marker_color='#10b981'))
    fig2.update_layout(barmode='stack', title="Resource Utilization", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.metric("Healthcare Stress Index", f"{latest.get('healtcare_stress_index', 0):.1f}", help="Formula: Vent Patients / Vent Capacity")
    with col_h2:
        st.metric("Oxygen Dependency Ratio", f"{latest.get('oxygen_dependency_ratio', 0):.1f}%", help="Formula: Oxygen Patients / Total Admitted")

# --- Tab 3: Comparisons ---
with tab3:
    st.subheader("Cross-Province Comparisons")
    if province == "All":
        latest_prov = df.groupby("province").last().reset_index()
        fig_prov = make_subplots(rows=1, cols=3, subplot_titles=("Total Cases", "Active Cases", "Fatality Rate (%)"))
        fig_prov.add_trace(go.Bar(x=latest_prov["province"], y=latest_prov["grand_total_cases_till_date"], marker_color='#3b82f6'), row=1, col=1)
        fig_prov.add_trace(go.Bar(x=latest_prov["province"], y=latest_prov["active_cases"], marker_color='#f59e0b'), row=1, col=2)
        fig_prov.add_trace(go.Bar(x=latest_prov["province"], y=latest_prov["fatality_rate"], marker_color='#ef4444'), row=1, col=3)
        fig_prov.update_layout(height=400, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_prov, use_container_width=True)

# --- Tab 4: AI Projections (The Upgrade) ---
with tab4:
    st.subheader("🤖 AI Model Comparison")
    st.markdown("Comparing **Linear Regression**, **Random Forest**, and **Polynomial Regression (Degree 3)**.")
    
    if len(df_filtered) > 20:
        pred_data = df_filtered[["date", "new_cases_7da"]].dropna()
        pred_data["days"] = (pred_data["date"] - pred_data["date"].min()).dt.days

        X = pred_data[["days"]]
        y = pred_data["new_cases_7da"]

        # 1. Linear Regression
        lin = LinearRegression().fit(X, y)
        
        # 2. Random Forest (Standard)
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        
        # 3. Polynomial Regression (Degree 3) - The "Best Fit" for waves
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        poly_reg = LinearRegression().fit(X_poly, y)

        # Future Generation
        future_days_count = 30
        last_day = pred_data["days"].max()
        future_X = np.arange(last_day + 1, last_day + future_days_count + 1).reshape(-1, 1)
        future_dates = [pred_data["date"].max() + timedelta(days=i) for i in range(1, future_days_count + 1)]

        # Predictions (Clipped to 0 to avoid negatives)
        pred_lin = np.maximum(lin.predict(future_X), 0)
        pred_rf = np.maximum(rf.predict(future_X), 0)
        pred_poly = np.maximum(poly_reg.predict(poly.transform(future_X)), 0)

        # Plotting
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=pred_data["date"], y=y, mode='lines', name='Historical Data', line=dict(color='gray')))
        
        fig_ai.add_trace(go.Scatter(x=future_dates, y=pred_lin, mode='lines', name='Linear (Underfit)', line=dict(dash='dash', color='red')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=pred_rf, mode='lines', name='Random Forest (Flatlines)', line=dict(dash='dot', color='green')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=pred_poly, mode='lines', name='Polynomial Deg-3 (Best Fit)', line=dict(width=3, color='#F59E0B')))

        fig_ai.update_layout(title="30-Day Forecast Comparison", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ai, use_container_width=True)
        
        st.info("💡 **Insight:** Notice how Linear Regression fails to capture the curve, and Random Forest struggles to extrapolate trends it hasn't seen before. **Polynomial Regression** adapts best to the wave-like nature of pandemics.")

    else:
        st.info("Not enough data for projections.")

    st.subheader("Scenario Calculator (Rt Based)")
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.markdown("**Intervention Parameters**")
        r0_input = st.slider("Baseline Rt", 0.5, 3.0, float(max(0.5, rt_val)), 0.1)
        intervention_strength = st.slider("Intervention Impact (%)", 0, 50, 15)
        
        final_rt = r0_input * (1 - intervention_strength/100)
        st.metric("Effective Rt", f"{final_rt:.2f}", delta=f"-{intervention_strength}%")

    with col_s2:
        # Simple SIR-like projection based on Rt
        proj_days = 60
        start_cases = latest['new_cases_7da']
        # Approx serial interval = 4 days
        
        y_base = [start_cases * (r0_input ** (d/4)) for d in range(proj_days)]
        y_inter = [start_cases * (final_rt ** (d/4)) for d in range(proj_days)]
        dates_proj = [datetime.now() + timedelta(days=d) for d in range(proj_days)]
        
        fig_scen = go.Figure()
        fig_scen.add_trace(go.Scatter(x=dates_proj, y=y_base, name="No Action", line=dict(color='red', dash='dot')))
        fig_scen.add_trace(go.Scatter(x=dates_proj, y=y_inter, name="With Intervention", line=dict(color='green', width=3)))
        fig_scen.update_layout(title="Intervention Impact Projection", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scen, use_container_width=True)

# --- Tab 5: Wave Analysis ---
with tab5:
    st.header("Province-Wise & Wave-Wise Historical Comparison")

    wave_stats = df.groupby(["wave", "province"]).agg({
        "new_cases": "sum",
        "new_deaths": "sum",
        "grand_total_cases_till_date": "max",
        "test_positivity_ratio": "mean"
    }).reset_index()

    wave_stats.rename(columns={
        "new_cases": "Total New Cases",
        "new_deaths": "Total Deaths",
        "grand_total_cases_till_date": "Peak Cases",
        "test_positivity_ratio": "Avg Positivity %"
    }, inplace=True)

    fig_cases = px.bar(wave_stats, x="wave", y="Peak Cases", color="province", barmode="group", title="Peak Cases by Wave & Province")
    fig_cases.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cases, use_container_width=True)

# ============================================
# Risk Assessment (Using New Metrics)
# ============================================
st.markdown("---")
st.header("🛡️ Automated Risk Assessment")
risk_score = 0
alerts = []

# Using refined metrics for risk
if positivity > 10: risk_score += 30; alerts.append("🔴 CRITICAL: Positivity >10% (High Spread)")
elif positivity > 5: risk_score += 15; alerts.append("🟡 WARNING: Positivity >5%")

if latest.get('healtcare_stress_index', 0) > 80: risk_score += 25; alerts.append("🔴 CRITICAL: Ventilator Capacity Near Limit")
elif latest.get('healtcare_stress_index', 0) > 60: risk_score += 10; alerts.append("🟡 WARNING: High Ventilator Usage")

if rt_val > 1.2: risk_score += 20; alerts.append(f"🔴 CRITICAL: Rt {rt_val:.2f} (Exponential Growth)")

risk_level = "🟢 LOW" if risk_score < 30 else "🟡 MODERATE" if risk_score < 60 else "🔴 HIGH"
st.metric("Risk Level", risk_level, f"{risk_score}/100 Risk Score")

for a in alerts:
    st.warning(a)

st.caption(f"System v4.5 | Data Last Updated: {datetime.now().strftime('%Y-%m-%d')}")



