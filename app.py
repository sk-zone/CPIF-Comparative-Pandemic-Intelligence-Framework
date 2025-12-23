import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
import re

warnings.filterwarnings('ignore')

# ============================================
# 1. Configuration & Theme
# ============================================
st.set_page_config(
    page_title="CPIF | Pandemic Intelligence",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for Executive Polish & Dark/Light Compatibility
st.markdown("""
    <style>
    .metric-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        border-left: 5px solid #3b82f6; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 10px; 
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        border-radius: 4px 4px 0px 0px; 
        gap: 1px; 
        padding-top: 10px; 
        padding-bottom: 10px;
        color: #888;
    }
    .stTabs [aria-selected="true"] { 
        background-color: rgba(59, 130, 246, 0.1); 
        border-bottom: 2px solid #3b82f6; 
        color: #3b82f6; 
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ CPIF: Comparative Pandemic Intelligence Framework")

# ============================================
# 2. Data Loading & Engineering
# ============================================
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except FileNotFoundError:
        return pd.DataFrame(), {}

    # 1. Robust Column Cleaning
    df.columns = (df.columns.astype(str)
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_'))

    # 2. Display Mapping
    display_map = {
        'healtcare_stress_index': 'Healthcare Stress Index',
        'fetaility_ratio': 'Fatality Ratio',
        'clinic_total_no_of_covid_patients_currently_admitted': 'Hospital Admissions',
        'test_positivity_ratio': 'Test Positivity Rate (TPR)',
        'oxygen_dependency_ratio': 'Oxygen Dependency',
        'clinic_total_numbers_recovered_and_discharged_so_far': 'Total Recovered',
        'clinic_total_no_of_patients_currently_on_ventilator': 'Critical (Ventilator)',
        'clinic_total_no_of_beds_allocated_for_covid_patients': 'Total Beds Capacity',
        'clinic_total_no_of_ventilators_allocated_for_covid_patients': 'Total Vent Capacity',
        'grand_total_cases_till_date': 'Total Cases'
    }

    # 3. Numeric Conversion
    numeric_cols = [
        'grand_total_cases_till_date', 'death_cumulative_total_deaths',
        'clinic_total_numbers_recovered_and_discharged_so_far',
        'grand_total_tests_conducted_till_date', 'test_positivity_ratio',
        'clinic_total_no_of_covid_patients_currently_admitted',
        'clinic_total_no_of_patients_currently_on_ventilator',
        'clinic_total_on_oxygen', 'healtcare_stress_index', 
        'oxygen_dependency_ratio',
        'clinic_total_no_of_beds_allocated_for_covid_patients',
        'clinic_total_no_of_ventilators_allocated_for_covid_patients'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace(',', '', regex=False)
                       .str.replace('%', '', regex=False)
                       .replace(['N/A', 'n/a', '-', '', 'nan', 'Nan'], '0'))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Dates & Sorting
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by=['province', 'date'])
    else:
        return pd.DataFrame(), {}

    # 5. Feature Engineering: Waves Classification (Pakistan Context)
    def classify_wave(d):
        if d < pd.Timestamp("2020-05-31"): return "Wave 1 (Initial)"
        elif d < pd.Timestamp("2020-10-01"): return "Lull 1"
        elif d < pd.Timestamp("2021-01-31"): return "Wave 2 (Winter)"
        elif d < pd.Timestamp("2021-05-31"): return "Wave 3 (Spring)"
        elif d < pd.Timestamp("2021-10-01"): return "Wave 4 (Delta)"
        elif d > pd.Timestamp("2021-12-31"): return "Wave 5 (Omicron)"
        else: return "Inter-Wave"

    df['wave'] = df['date'].apply(classify_wave)

    # 6. Daily Diffs & Smoothing
    df['new_cases'] = df.groupby('province')['grand_total_cases_till_date'].diff().fillna(0).clip(lower=0)
    df['new_deaths'] = df.groupby('province')['death_cumulative_total_deaths'].diff().fillna(0).clip(lower=0)
    df['new_cases_7da'] = df.groupby('province')['new_cases'].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    
    # 7. Rt Proxy
    shifted_cases = df.groupby('province')['new_cases_7da'].shift(4).replace(0, 1)
    df['growth_factor'] = df['new_cases_7da'] / shifted_cases
    df['rt_estimate'] = df['growth_factor'].pow(1).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    return df, display_map

df_raw, col_map = load_and_clean_data()

if df_raw.empty:
    st.error("Data file not found. Please upload 'Refined + New entities.csv'.")
    st.stop()

# ============================================
# 3. Sidebar & Filters
# ============================================
st.sidebar.header("🔍 Controls")
provinces_list = ["All (National Aggregate)"] + sorted(df_raw['province'].unique().tolist())
selected_prov = st.sidebar.selectbox("Select Region", provinces_list, index=0)

# Aggregation Logic
if selected_prov == "All (National Aggregate)":
    numeric_cols_df = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    # Group by Date and Sum (preserve wave col via first/mode)
    df_agg = df_raw.groupby('date')[numeric_cols_df].sum().reset_index()
    
    # Re-attach Wave Info (Date is index)
    df_agg['wave'] = df_agg['date'].apply(lambda d: df_raw[df_raw['date'] == d]['wave'].iloc[0])
    
    # Re-calculate Rt & Positivity Proxy
    df_agg['new_cases_7da'] = df_agg['new_cases'].rolling(7).mean().fillna(0)
    shifted = df_agg['new_cases_7da'].shift(4).replace(0, 1)
    df_agg['rt_estimate'] = (df_agg['new_cases_7da'] / shifted).fillna(1.0)
    if 'test_positivity_ratio' in df_raw.columns:
        df_agg['test_positivity_ratio'] = df_raw.groupby('date')['test_positivity_ratio'].mean().values
    
    df_filtered = df_agg
else:
    df_filtered = df_raw[df_raw['province'] == selected_prov].copy()

# Date Filtering
min_d, max_d = df_filtered['date'].min(), df_filtered['date'].max()
dates = st.sidebar.date_input("Analysis Period", [max_d - timedelta(days=90), max_d], min_value=min_d, max_value=max_d)

if len(dates) == 2:
    start_d, end_d = dates
    df_filtered = df_filtered[(df_filtered['date'].dt.date >= start_d) & (df_filtered['date'].dt.date <= end_d)]

if df_filtered.empty:
    st.warning("⚠️ No data available for this date range.")
    st.stop()

# ============================================
# 4. Expanded Executive KPIs
# ============================================
latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-8] if len(df_filtered) > 8 else latest

# Calculate Utilizations
beds_used = latest.get('clinic_total_no_of_covid_patients_currently_admitted', 0)
beds_tot = latest.get('clinic_total_no_of_beds_allocated_for_covid_patients', 1)
bed_util = (beds_used / beds_tot * 100) if beds_tot > 0 else 0

vent_used = latest.get('clinic_total_no_of_patients_currently_on_ventilator', 0)
vent_tot = latest.get('clinic_total_no_of_ventilators_allocated_for_covid_patients', 1)
vent_util = (vent_used / vent_tot * 100) if vent_tot > 0 else 0

st.markdown("### 📊 Executive Summary")
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Total Confirmed Cases", f"{int(latest['grand_total_cases_till_date']):,}")
with k2: st.metric("Total Recovered", f"{int(latest['clinic_total_numbers_recovered_and_discharged_so_far']):,}")
with k3: st.metric("Bed Utilization", f"{bed_util:.1f}%", delta=f"{bed_util - ((prev.get('clinic_total_no_of_covid_patients_currently_admitted',0)/beds_tot)*100):.1f}%")
with k4: st.metric("Ventilator Utilization", f"{vent_util:.1f}%", delta=f"{vent_util - ((prev.get('clinic_total_no_of_patients_currently_on_ventilator',0)/vent_tot)*100):.1f}%")

k5, k6, k7, k8 = st.columns(4)
with k5: st.metric("New Cases (7d Avg)", f"{int(latest['new_cases_7da']):,}", delta=f"{int(latest['new_cases_7da'] - prev['new_cases_7da'])}")
with k6: st.metric("Positivity Rate", f"{latest.get('test_positivity_ratio', 0):.1f}%", delta_color="inverse")
with k7: st.metric("Rt (Spread)", f"{latest['rt_estimate']:.2f}", delta="Exp. Growth" if latest['rt_estimate']>1 else "Stable", delta_color="inverse")
with k8: st.metric("Total Deaths", f"{int(latest['death_cumulative_total_deaths']):,}")

# ============================================
# 5. Main Tabs
# ============================================
tabs = st.tabs(["📈 Key Trends", "🏥 Resource Intelligence", "🌊 Wave Analysis", "🤖 AI Projections", "🧮 Scenario Calculator"])

# --- Tab 1: Key Trends & Heatmap ---
with tabs[0]:
    st.subheader("Epidemic Trajectory")
    
    # 1. Main Trend Plot
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['new_cases'], name="Daily Cases", marker_color='rgba(59, 130, 246, 0.4)'), secondary_y=False)
    fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['new_cases_7da'], name="7-Day Trend", line=dict(color='#2563EB', width=3)), secondary_y=False)
    if 'test_positivity_ratio' in df_filtered.columns:
        fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['test_positivity_ratio'], name="Positivity %", line=dict(color='#DC2626', dash='dot')), secondary_y=True)
    
    fig_trend.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", y=1.1), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_trend, use_container_width=True)

    # 2. Monthly Heatmap
    st.subheader("Monthly Case Intensity")
    df_heat = df_filtered.copy()
    df_heat['Year'] = df_heat['date'].dt.year
    df_heat['Month'] = df_heat['date'].dt.month_name()
    # Group by Year/Month
    heatmap_data = df_heat.pivot_table(index='Year', columns='Month', values='new_cases', aggfunc='sum').fillna(0)
    # Reorder months
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    heatmap_data = heatmap_data.reindex(columns=months_order)
    
    fig_heat = px.imshow(heatmap_data, color_continuous_scale='Reds', aspect='auto')
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_heat, use_container_width=True)

# --- Tab 2: Resource Intelligence ---
with tabs[1]:
    st.subheader("Healthcare Capacity & Utilization")
    
    # 1. Resource Trends Plot
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['clinic_total_no_of_covid_patients_currently_admitted'], name="Total Admitted", fill='tozeroy', line=dict(color='#6366F1')))
    fig_res.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['clinic_total_no_of_patients_currently_on_ventilator'], name="On Ventilator", line=dict(color='#EF4444', width=2)))
    fig_res.update_layout(title="Admissions vs Critical Care Trend", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
    st.plotly_chart(fig_res, use_container_width=True)

    # 2. Detailed Resource Table
    st.subheader("Daily Resource Log")
    res_cols = ['date', 'clinic_total_no_of_covid_patients_currently_admitted', 'clinic_total_no_of_patients_currently_on_ventilator', 'oxygen_dependency_ratio', 'healtcare_stress_index']
    res_df = df_filtered[res_cols].copy()
    res_df.columns = ['Date', 'Admitted', 'On Ventilator', 'Oxygen Dep %', 'Stress Index']
    res_df['Date'] = res_df['Date'].dt.date
    st.dataframe(res_df.sort_values('Date', ascending=False), use_container_width=True, height=300)

# --- Tab 3: Wave Analysis (New) ---
with tabs[2]:
    st.subheader("Comparative Wave Analysis")
    
    # Group by Wave (Using the filtered data context or full raw for comparison)
    # Let's use Full Raw for this tab to show historical context regardless of date slider
    if selected_prov == "All (National Aggregate)":
        wave_df = df_raw.groupby(['wave', 'province'])[['new_cases', 'new_deaths']].sum().reset_index()
        wave_df = wave_df.groupby('wave')[['new_cases', 'new_deaths']].sum().reset_index()
    else:
        wave_df = df_raw[df_raw['province'] == selected_prov].groupby('wave')[['new_cases', 'new_deaths']].sum().reset_index()
    
    # Peak Cases Calculation
    peak_df = df_raw if selected_prov == "All (National Aggregate)" else df_raw[df_raw['province'] == selected_prov]
    peak_stats = peak_df.groupby('wave')['new_cases'].max().reset_index()
    peak_stats.columns = ['wave', 'peak_daily_cases']
    
    c_w1, c_w2 = st.columns(2)
    with c_w1:
        st.markdown("**Total Deaths by Wave**")
        fig_wd = px.bar(wave_df, x='wave', y='new_deaths', color='wave', title="Mortality Impact")
        fig_wd.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wd, use_container_width=True)
        
    with c_w2:
        st.markdown("**Peak Daily Cases by Wave**")
        fig_wp = px.bar(peak_stats, x='wave', y='peak_daily_cases', color='wave', title="Highest Infection Spike")
        fig_wp.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wp, use_container_width=True)

# --- Tab 4: AI Projections (Clipped) ---
with tabs[3]:
    st.subheader("Predictive Analytics (Clipped)")
    
    col_p1, col_p2 = st.columns([1, 4])
    with col_p1:
        forecast_days = st.slider("Forecast Horizon", 15, 90, 30)
        degree = st.selectbox("Polynomial Degree", [2, 3, 4], index=1)
        st.caption("Lower limit set to 0 to prevent negative case predictions.")
    
    with col_p2:
        pred_df = df_filtered[['date', 'new_cases_7da']].dropna()
        if len(pred_df) > 20:
            pred_df['days_idx'] = (pred_df['date'] - pred_df['date'].min()).dt.days
            
            poly = PolynomialFeatures(degree=degree)
            X = poly.fit_transform(pred_df[['days_idx']])
            y = pred_df['new_cases_7da']
            
            model = LinearRegression()
            model.fit(X, y)
            
            last_idx = pred_df['days_idx'].max()
            future_X = poly.transform(np.arange(last_idx + 1, last_idx + forecast_days + 1).reshape(-1, 1))
            
            # Predict and CLIP to avoid negatives
            preds = model.predict(future_X).clip(min=0)
            
            future_dates = [pred_df['date'].max() + timedelta(days=x) for x in range(1, forecast_days + 1)]
            
            fig_ai = go.Figure()
            fig_ai.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['new_cases_7da'], name="Historical", line=dict(color='gray')))
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", line=dict(color='#F59E0B', width=3, dash='dash')))
            
            # Uncertainty Area
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds*1.2, mode='lines', line=dict(width=0), showlegend=False))
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds*0.8, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)', name="Uncertainty Range"))
            
            fig_ai.update_layout(title="Trajectory Forecast", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_ai, use_container_width=True)
        else:
            st.info("Insufficient data for robust projection.")

# --- Tab 5: Dynamic Scenario Calculator ---
with tabs[4]:
    st.subheader("Dynamic Policy Simulator")
    
    c_sim1, c_sim2 = st.columns([1, 2])
    with c_sim1:
        st.markdown("**Simulation Parameters**")
        base_rt_sim = st.slider("Current Rt", 0.5, 3.0, float(max(0.5, latest['rt_estimate'])), 0.1)
        intervention_strength = st.slider("Intervention Strength (%)", 0, 100, 20)
        compliance = st.slider("Public Compliance (%)", 0, 100, 70)
        
        # Dynamic Calculation
        effective_reduction = (intervention_strength / 100) * (compliance / 100)
        final_rt_sim = max(0.1, base_rt_sim * (1 - effective_reduction))
        
        st.metric("Effective Rt", f"{final_rt_sim:.2f}", delta=f"-{effective_reduction*100:.1f}% Reduction")
        
    with c_sim2:
        sim_days = 60
        start_val = max(100, latest['new_cases_7da'])
        dates_sim = [datetime.now() + timedelta(days=x) for x in range(sim_days)]
        
        curve_base = [start_val * (base_rt_sim**(d/4)) for d in range(sim_days)]
        curve_int = [start_val * (final_rt_sim**(d/4)) for d in range(sim_days)]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=dates_sim, y=curve_base, name="Status Quo", line=dict(color='#ef4444', dash='dot')))
        fig_sim.add_trace(go.Scatter(x=dates_sim, y=curve_int, name="With Intervention", line=dict(color='#10b981', width=3)))
        
        fig_sim.update_layout(title="Impact Projection (60 Days)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
        st.plotly_chart(fig_sim, use_container_width=True)
        
        saved = sum(curve_base) - sum(curve_int)
        st.success(f"Projected to avert **{int(saved):,}** cases.")

st.markdown("---")
st.caption("Framework v4.0 | Sponsor: Development Synergies International ")


