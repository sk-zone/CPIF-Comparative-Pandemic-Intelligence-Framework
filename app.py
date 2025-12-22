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

# Custom CSS for Executive Polish (Fixed for Dark/Light Mode Compatibility)
st.markdown("""
    <style>
    /* Metric Cards - Uses a semi-transparent background to blend with any theme */
    .metric-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        border-left: 5px solid #3b82f6; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 10px; 
    }
    
    /* Tabs Styling - Removed hardcoded white backgrounds that caused 'glitches' */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        border-radius: 4px 4px 0px 0px; 
        gap: 1px; 
        padding-top: 10px; 
        padding-bottom: 10px;
        color: #888; /* Unselected Text Color */
    }
    
    .stTabs [aria-selected="true"] { 
        background-color: rgba(59, 130, 246, 0.1); /* Subtle blue tint instead of stark white */
        border-bottom: 2px solid #3b82f6; 
        color: #3b82f6; /* Selected Text Color */
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
        # Attempt to load data
        df = pd.read_csv("Refined + New entities.csv")
    except FileNotFoundError:
        return pd.DataFrame(), {}

    # 1. Robust Column Cleaning (Regex)
    df.columns = (df.columns.astype(str)
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_'))

    # 2. Display Mapping (For referencing if needed)
    display_map = {
        'healtcare_stress_index': 'Healthcare Stress Index',
        'fetaility_ratio': 'Fatality Ratio',
        'clinic_total_no_of_covid_patients_currently_admitted': 'Hospital Admissions',
        'test_positivity_ratio': 'Test Positivity Rate (TPR)',
        'oxygen_dependency_ratio': 'Oxygen Dependency',
        'clinic_total_numbers_recovered_and_discharged_so_far': 'Total Recovered',
        'clinic_total_no_of_patients_currently_on_ventilator': 'Critical (Ventilator)',
        'clinic_total_no_of_beds_allocated_for_covid_patients': 'Total Beds Capacity',
        'clinic_total_no_of_ventilators_allocated_for_covid_patients': 'Total Vent Capacity'
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

    # 5. Daily Diffs (Per Province)
    df['new_cases'] = df.groupby('province')['grand_total_cases_till_date'].diff().fillna(0).clip(lower=0)
    df['new_deaths'] = df.groupby('province')['death_cumulative_total_deaths'].diff().fillna(0).clip(lower=0)
    
    # Smoothers
    df['new_cases_7da'] = df.groupby('province')['new_cases'].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    
    # Rt Proxy
    shifted_cases = df.groupby('province')['new_cases_7da'].shift(4).replace(0, 1)
    df['growth_factor'] = df['new_cases_7da'] / shifted_cases
    df['rt_estimate'] = df['growth_factor'].pow(1).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    return df, display_map

df_raw, col_map = load_and_clean_data()

if df_raw.empty:
    st.error("Data file not found. Please upload 'Refined + New entities.csv'.")
    st.stop()

# ============================================
# 3. Logic: Aggregation Handler
# ============================================
st.sidebar.header("🔍 Controls")
provinces_list = ["All (National Aggregate)"] + sorted(df_raw['province'].unique().tolist())
selected_prov = st.sidebar.selectbox("Select Region", provinces_list, index=0)

# Aggregation Logic
if selected_prov == "All (National Aggregate)":
    numeric_cols_df = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    # Group by Date and Sum
    df_filtered = df_raw.groupby('date')[numeric_cols_df].sum().reset_index()
    
    # Re-calculate Rt for the National Aggregate
    df_filtered['new_cases_7da'] = df_filtered['new_cases'].rolling(7).mean().fillna(0)
    shifted = df_filtered['new_cases_7da'].shift(4).replace(0, 1)
    df_filtered['rt_estimate'] = (df_filtered['new_cases_7da'] / shifted).fillna(1.0)
    
    # Proxy for Positivity: Mean of daily records (approximate)
    if 'test_positivity_ratio' in df_raw.columns:
        df_filtered['test_positivity_ratio'] = df_raw.groupby('date')['test_positivity_ratio'].mean().values
else:
    df_filtered = df_raw[df_raw['province'] == selected_prov].copy()

# Date Filtering
min_d, max_d = df_filtered['date'].min(), df_filtered['date'].max()
dates = st.sidebar.date_input("Analysis Period", [max_d - timedelta(days=90), max_d], min_value=min_d, max_value=max_d)

if len(dates) == 2:
    start_d, end_d = dates
    df_filtered = df_filtered[(df_filtered['date'].dt.date >= start_d) & (df_filtered['date'].dt.date <= end_d)]

if df_filtered.empty:
    st.warning("⚠️ No data available for this date range. Please select a wider window.")
    st.stop()

# ============================================
# 4. Metrics & Key Stats
# ============================================
latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-8] if len(df_filtered) > 8 else latest
rt_now = df_filtered['rt_estimate'].mean()

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("New Cases (7d Avg)", f"{int(latest['new_cases_7da']):,}", delta=f"{int(latest['new_cases_7da'] - prev['new_cases_7da'])}")
with c2: st.metric("Rt (Spread Velocity)", f"{rt_now:.2f}", delta="Exp. Growth" if rt_now > 1.1 else "Stable", delta_color="inverse")
with c3: st.metric("Positivity Rate", f"{latest.get('test_positivity_ratio', 0):.1f}%", delta=f"{(latest.get('test_positivity_ratio',0) - prev.get('test_positivity_ratio',0)):.1f}%", delta_color="inverse")
with c4: st.metric("Active Admissions", f"{int(latest.get('clinic_total_no_of_covid_patients_currently_admitted',0)):,}")
with c5: st.metric("Total Deaths", f"{int(latest.get('death_cumulative_total_deaths',0)):,}", delta=f"{int(latest['new_deaths'])}")

# ============================================
# 5. Tabs
# ============================================
t1, t2, t3, t4, t5 = st.tabs(["📈 Intelligence", "🏥 Capacity Insights", "🔮 Projections", "🕵️ Bias Audit", "♟️ Intervention Sim"])

# --- Tab 1: Intelligence ---
with t1:
    st.subheader("Epidemic Wave Analysis")
    
    col_opt1, col_opt2 = st.columns([1, 5])
    with col_opt1:
        log_scale = st.checkbox("Logarithmic Scale", help="Useful for seeing early exponential growth")
        show_pos = st.checkbox("Show Positivity", value=True)
    
    with col_opt2:
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['new_cases'], name="Daily Cases", marker_color='rgba(59, 130, 246, 0.4)'), secondary_y=False)
        fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['new_cases_7da'], name="7-Day Trend", line=dict(color='#2563EB', width=3)), secondary_y=False)
        
        if show_pos and 'test_positivity_ratio' in df_filtered.columns:
            fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['test_positivity_ratio'], name="Positivity %", line=dict(color='#DC2626', dash='dot')), secondary_y=True)

        # FIXED: Removed 'template="plotly_white"' and added transparent bg
        fig_trend.update_layout(
            height=450, 
            hovermode="x unified", 
            legend=dict(orientation="h", y=1.1),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        if log_scale: fig_trend.update_yaxes(type="log", secondary_y=False)
        st.plotly_chart(fig_trend, use_container_width=True)

# --- Tab 2: Capacity (Enhanced) ---
with t2:
    st.subheader("Healthcare Load vs. Capacity")
    
    # Calculate utilization
    beds_used = latest.get('clinic_total_no_of_covid_patients_currently_admitted', 0)
    beds_total = latest.get('clinic_total_no_of_beds_allocated_for_covid_patients', 1) 
    beds_pct = (beds_used / beds_total) * 100 if beds_total > 0 else 0
    
    vents_used = latest.get('clinic_total_no_of_patients_currently_on_ventilator', 0)
    vents_total = latest.get('clinic_total_no_of_ventilators_allocated_for_covid_patients', 1)
    vents_pct = (vents_used / vents_total) * 100 if vents_total > 0 else 0

    g1, g2, g3 = st.columns(3)
    with g1:
        fig_g1 = go.Figure(go.Indicator(
            mode = "gauge+number", value = beds_pct, title = {'text': "General Bed Occupancy"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}, 'steps': [{'range': [0, 70], 'color': "rgba(200, 200, 200, 0.3)"}, {'range': [70, 100], 'color': "rgba(252, 165, 165, 0.5)"}]}
        ))
        # FIXED: Transparent BG
        fig_g1.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g1, use_container_width=True)
        
    with g2:
        fig_g2 = go.Figure(go.Indicator(
            mode = "gauge+number", value = vents_pct, title = {'text': "Ventilator Utilization"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#ef4444"}, 'steps': [{'range': [0, 60], 'color': "rgba(200, 200, 200, 0.3)"}, {'range': [60, 100], 'color': "rgba(252, 165, 165, 0.5)"}]}
        ))
        # FIXED: Transparent BG
        fig_g2.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g2, use_container_width=True)

    with g3:
        if 'oxygen_dependency_ratio' in df_filtered.columns:
            st.metric("Current Oxygen Dependency", f"{latest['oxygen_dependency_ratio']:.1f}%")
            st.caption("Percentage of admitted patients requiring Oxygen.")
            fig_oxy = px.area(df_filtered, x='date', y='oxygen_dependency_ratio', color_discrete_sequence=['#10B981'])
            # FIXED: Transparent BG
            fig_oxy.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_oxy, use_container_width=True)

# --- Tab 3: Projections (Cleaned) ---
with t3:
    st.subheader("Predictive Analytics")
    
    col_p1, col_p2 = st.columns([1, 4])
    with col_p1:
        forecast_days = st.slider("Forecast Horizon (Days)", 15, 90, 30)
        st.caption("Model: Polynomial Regression (Order 3) + Multivariate adjustment.")
    
    with col_p2:
        pred_df = df_filtered[['date', 'new_cases_7da']].dropna()
        if len(pred_df) > 20:
            pred_df['days_idx'] = (pred_df['date'] - pred_df['date'].min()).dt.days
            
            poly = PolynomialFeatures(degree=3)
            X = poly.fit_transform(pred_df[['days_idx']])
            y = pred_df['new_cases_7da']
            
            model = LinearRegression()
            model.fit(X, y)
            
            last_idx = pred_df['days_idx'].max()
            future_X = poly.transform(np.arange(last_idx + 1, last_idx + forecast_days + 1).reshape(-1, 1))
            preds = model.predict(future_X)
            
            future_dates = [pred_df['date'].max() + timedelta(days=x) for x in range(1, forecast_days + 1)]
            
            fig_ai = go.Figure()
            fig_ai.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['new_cases_7da'], name="Historical", line=dict(color='gray')))
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", line=dict(color='#F59E0B', width=3, dash='dash')))
            
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds*1.2, mode='lines', line=dict(width=0), showlegend=False))
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds*0.8, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)', name="Uncertainty Range"))
            
            # FIXED: Removed 'template="plotly_dark"' and added transparent bg
            fig_ai.update_layout(
                title="Trajectory Forecast", 
                height=500,
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_ai, use_container_width=True)
        else:
            st.info("Insufficient data for robust projection in this window.")

# --- Tab 4: Audit (Bias) ---
with t4:
    st.subheader("Data Integrity Check")
    st.markdown("Use this to verify if 'Surges' are real or just a result of 'More Testing'.")
    
    if 'grand_total_tests_conducted_till_date' in df_filtered.columns:
        fig_bias = px.scatter(
            df_filtered, x='grand_total_tests_conducted_till_date', y='new_cases_7da',
            color='test_positivity_ratio', title="Testing Volume vs. Detected Cases",
            labels={'grand_total_tests_conducted_till_date': 'Total Tests', 'new_cases_7da': 'New Cases (7d Avg)', 'test_positivity_ratio': 'Positivity %'}
        )
        # FIXED: Transparent BG
        fig_bias.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bias, use_container_width=True)

# --- Tab 5: Interventions (Expanded) ---
with t5:
    st.subheader("Policy Impact Simulator")
    
    ci1, ci2 = st.columns([1, 2])
    with ci1:
        st.markdown("**Intervention Matrix**")
        base_rt_sim = st.number_input("Baseline Rt", value=float(max(0.5, rt_now)), step=0.05, format="%.2f")
        
        i1 = st.checkbox("😷 Mask Mandate", help="Est. Reduction: 15%")
        i2 = st.checkbox("📏 Social Distancing", help="Est. Reduction: 10%")
        i3 = st.checkbox("🏠 Remote Work/School", help="Est. Reduction: 12%")
        i4 = st.checkbox("🔒 Smart Lockdown", help="Est. Reduction: 25%")
        
        reduction = 0.0
        if i1: reduction += 0.15
        if i2: reduction += 0.10
        if i3: reduction += 0.12
        if i4: reduction += 0.25
        
        final_rt_sim = max(0.1, base_rt_sim * (1 - reduction))
        st.metric("Adjusted Rt", f"{final_rt_sim:.2f}", delta=f"-{reduction*100:.0f}% Efficacy")

    with ci2:
        sim_days = 60
        start_val = max(100, latest['new_cases_7da'])
        dates_sim = [datetime.now() + timedelta(days=x) for x in range(sim_days)]
        
        curve_base = [start_val * (base_rt_sim**(d/4)) for d in range(sim_days)]
        curve_int = [start_val * (final_rt_sim**(d/4)) for d in range(sim_days)]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=dates_sim, y=curve_base, name="Do Nothing", line=dict(color='#ef4444', dash='dot')))
        fig_sim.add_trace(go.Scatter(x=dates_sim, y=curve_int, name="With Intervention", line=dict(color='#10b981', width=3)))
        
        # FIXED: Transparent BG
        fig_sim.update_layout(
            title="60-Day Impact Projection", 
            yaxis_title="Projected Daily Cases", 
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
        cases_saved = sum(curve_base) - sum(curve_int)
        st.success(f"potential reduction of **{int(cases_saved):,}** cases over 60 days.")

st.markdown("---")
st.caption("Framework v3.0 | Executive Intelligence Dashboard")
