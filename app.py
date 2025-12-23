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
# 1. Page Config & Visual Theme
# ============================================
st.set_page_config(page_title="Pakistan COVID-19 Intelligence Dashboard", layout="wide", page_icon="🇵🇰")

st.markdown("""
    <style>
    .metric-card { background-color: rgba(255, 255, 255, 0.05); border-left: 5px solid #3b82f6; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; }
    .stTabs [aria-selected="true"] { background-color: rgba(59, 130, 246, 0.1); border-bottom: 2px solid #3b82f6; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🇵🇰 Executive Pandemic Intelligence Dashboard")
st.markdown("**Prepared By Students of Applied AI in Public Heatlhcare** | Advanced Insights, AI Projections & Scenario Calculator")

# ============================================
# 2. Robust Data Loading
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    # Robust Cleaning
    df.columns = (df.columns.astype(str).str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_'))

    # Mapping
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
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})

    # Numeric Conversion
    for col in col_map.values():
        if col in df.columns:
            df[col] = (df[col].astype(str).str.replace(',', '').str.replace('%', '').replace(['N/A', 'nan', '-'], '0'))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Dates & Sorting
    if 'date' in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by=["province", "date"]).reset_index(drop=True)

    # Derived Metrics
    df["active_cases"] = (df["cases"] - df["recovered"] - df["deaths"]).clip(lower=0)
    df["new_cases"] = df.groupby("province")["cases"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby("province")["deaths"].diff().fillna(0).clip(lower=0)
    df["new_cases_7da"] = df.groupby("province")["new_cases"].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    
    # Rt Proxy
    shifted = df.groupby("province")["new_cases_7da"].shift(4).replace(0, 1)
    df["rt_estimate"] = (df["new_cases_7da"] / shifted).clip(0, 5).fillna(1.0)
    
    # Rates
    df["recovery_rate"] = (df["recovered"] / df["cases"] * 100).fillna(0)
    df["fatality_rate"] = (df["deaths"] / df["cases"] * 100).fillna(0)
    if 'oxygen_dependency' in df.columns:
         df.loc[df['oxygen_dependency'] > 100, 'oxygen_dependency'] = 100

    # Define Waves (Chronological Sorting Helper)
    df["wave"] = "Other"
    df.loc[df["date"].between("2020-03-01", "2020-05-31"), "wave"] = "1. Wave 1 (Spring 2020)"
    df.loc[df["date"].between("2020-06-01", "2020-08-31"), "wave"] = "2. Wave 2 (Summer 2020)"
    df.loc[df["date"].between("2020-11-01", "2021-01-31"), "wave"] = "3. Wave 3 (Winter 2020)"
    df.loc[df["date"].between("2021-04-01", "2021-06-30"), "wave"] = "4. Wave 4 (Delta)"
    df.loc[df["date"].between("2021-07-01", "2021-09-30"), "wave"] = "5. Wave 4 Peak"
    df.loc[df["date"].between("2022-01-01", "2022-03-31"), "wave"] = "6. Wave 5 (Omicron)"

    return df

df = load_data()
if df.empty: st.stop()

# ============================================
# 3. Sidebar
# ============================================
st.sidebar.header("Executive Controls")
provinces = ["All"] + sorted(df["province"].unique().tolist())
province = st.sidebar.selectbox("Select Province", provinces, index=0)

min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input("Date Range", [max_date - timedelta(days=90), max_date], min_value=min_date, max_value=max_date)

mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
df_filtered = df[mask].copy()
if province != "All":
    df_filtered = df_filtered[df_filtered["province"] == province]

# ============================================
# 4. KPIs
# ============================================
latest = df_filtered.iloc[-1] if not df_filtered.empty else pd.Series()
def get_val(key, default=0): return latest.get(key, default)

st.header("Key Performance Indicators")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Cases", f"{int(get_val('cases')):,}", delta=f"{int(df_filtered['new_cases'].sum()):,} new")
c2.metric("Active Cases", f"{int(get_val('active_cases')):,}")
c3.metric("Recovered", f"{int(get_val('recovered')):,}", delta=f"{get_val('recovery_rate'):.1f}%")
c4.metric("Deaths", f"{int(get_val('deaths')):,}", delta=f"{get_val('fatality_rate'):.2f}%")
c5.metric("Positivity Rate", f"{get_val('positivity'):.2f}%")

# ============================================
# 5. Tabs
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Epidemic Curves", "Healthcare Capacity", "Comparisons & Heatmaps", 
    "AI Projections (All Models)", "Wave Analysis"
])

layout_args = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

with tab1:
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Daily New Cases (7-Day Avg)", "Cumulative Cases", "Daily Deaths", "Positivity Rate"))
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases_7da"], name='7-Day Avg', line=dict(color='#3b82f6', width=3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases"], mode='markers', name='Daily', marker=dict(color='#3b82f6', opacity=0.3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["cases"], name='Total Cases', line=dict(color='#8b5cf6')), row=1, col=2)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_deaths"], name='Daily Deaths', line=dict(color='#ef4444')), row=2, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["positivity"], name='Positivity %', line=dict(color='#f59e0b')), row=2, col=2)
    fig1.update_layout(height=600, **layout_args)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Healthcare Capacity")
    # Group's Bar Chart Logic
    cap_data = pd.DataFrame({
        "Metric": ["General Beds", "Oxygen Beds", "Ventilators"],
        "Used": [get_val('admitted'), get_val('oxygen_patients'), get_val('vent_patients')],
        "Available": [max(0, get_val('beds_total') - get_val('admitted')), max(0, get_val('oxygen_beds_total') - get_val('oxygen_patients')), max(0, get_val('vents_total') - get_val('vent_patients'))]
    })
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=cap_data["Metric"], x=cap_data["Used"], name="Used", orientation='h', marker_color='#ef4444'))
    fig2.add_trace(go.Bar(y=cap_data["Metric"], x=cap_data["Available"], name="Available", orientation='h', marker_color='#10b981'))
    fig2.update_layout(barmode='stack', title="Resource Limits", height=300, **layout_args)
    st.plotly_chart(fig2, use_container_width=True)
    
    if 'oxygen_dependency' in df_filtered.columns:
        fig_oxy = px.area(df_filtered, x='date', y='oxygen_dependency', title="Oxygen Dependency (Critical Stress Indicator)", color_discrete_sequence=['#10B981'])
        fig_oxy.update_layout(height=300, **layout_args)
        st.plotly_chart(fig_oxy, use_container_width=True)

with tab3:
    st.subheader("Comparisons")
    # IMPROVED HEATMAP: High Contrast
    df_month = df.copy()
    df_month["month"] = df_month["date"].dt.to_period("M").astype(str)
    monthly = df_month.pivot_table(values="new_cases", index="province", columns="month", aggfunc="sum", fill_value=0)
    
    # Using 'Redor' scale for high distinction
    fig_heat = px.imshow(monthly, title="Monthly New Cases Heatmap", color_continuous_scale='Redor', aspect='auto')
    fig_heat.update_layout(**layout_args)
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.subheader("AI Projections: Model Face-Off")
    if len(df_filtered) > 20:
        pred_data = df_filtered[["date", "new_cases_7da"]].dropna()
        pred_data["days"] = (pred_data["date"] - pred_data["date"].min()).dt.days
        X = pred_data[["days"]]
        y = pred_data["new_cases_7da"]

        # Models
        lin = LinearRegression().fit(X, y)
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        poly = PolynomialFeatures(degree=3)
        poly_reg = LinearRegression().fit(poly.fit_transform(X), y)

        # Future
        future_days = 30
        future_X = np.arange(X["days"].max() + 1, X["days"].max() + future_days + 1).reshape(-1, 1)
        future_dates = [pred_data["date"].max() + timedelta(days=i) for i in range(1, future_days + 1)]

        # Predictions (Clipped to 0)
        p_lin = np.maximum(0, lin.predict(future_X))
        p_rf = np.maximum(0, rf.predict(future_X))
        p_poly = np.maximum(0, poly_reg.predict(poly.transform(future_X)))
        
        # Growth Scenarios (1%, 2%, 3%)
        last_val = y.iloc[-1]
        p_1 = [last_val * (1.01 ** i) for i in range(1, future_days + 1)]
        p_2 = [last_val * (1.02 ** i) for i in range(1, future_days + 1)]
        p_3 = [last_val * (1.03 ** i) for i in range(1, future_days + 1)]

        # Plotting All
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=pred_data["date"], y=y, name='Historical', line=dict(color='gray', width=2)))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=p_lin, name='Linear Reg', line=dict(dash='dash', color='red')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=p_rf, name='Random Forest', line=dict(dash='dot', color='orange')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=p_poly, name='Polynomial (SOTA)', line=dict(width=4, color='#10B981')))
        
        # Growth Curves
        fig_ai.add_trace(go.Scatter(x=future_dates, y=p_1, name='1% Growth', line=dict(color='#93C5FD')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=p_2, name='2% Growth', line=dict(color='#3B82F6')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=p_3, name='3% Growth', line=dict(color='#1E3A8A')))

        fig_ai.update_layout(title="Comprehensive Forecasting Model Comparison", height=600, **layout_args)
        st.plotly_chart(fig_ai, use_container_width=True)
    else:
        st.info("Insufficient data for projections.")

    # Calculator
    st.subheader("Scenario Calculator")
    c_s1, c_s2 = st.columns([1, 2])
    with c_s1:
        cur_active = int(latest.get('new_cases_7da', 100))
        r0 = st.slider("Baseline Rt", 0.5, 4.0, 1.2, 0.1)
        reduction = st.slider("Intervention Impact (%)", 0, 60, 20)
        sim_days = st.slider("Simulation Horizon (Days)", 30, 90, 60) # Kept as requested
    
    with c_s2:
        proj_base = [cur_active * (r0 ** (d/4)) for d in range(sim_days)]
        proj_int = [cur_active * ((r0 * (1-reduction/100)) ** (d/4)) for d in range(sim_days)]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(y=proj_base, name="Baseline", line=dict(color='red', dash='dot')))
        fig_sim.add_trace(go.Scatter(y=proj_int, name="With Intervention", line=dict(color='green', width=3)))
        fig_sim.update_layout(title=f"Impact over {sim_days} Days", **layout_args)
        st.plotly_chart(fig_sim, use_container_width=True)

with tab5:
    st.header("Wave-Wise Analysis (Chronological)")
    # Sort waves chronologically
    df_full = df.sort_values(by="wave")
    wave_stats = df_full.groupby(["wave", "province"]).agg({"new_cases": "sum", "deaths": "sum", "cases": "max"}).reset_index()
    
    fig_cases = px.bar(wave_stats, x="wave", y="cases", color="province", barmode="group", title="Peak Cases by Wave (Sorted)")
    fig_cases.update_layout(**layout_args)
    st.plotly_chart(fig_cases, use_container_width=True)
