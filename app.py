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

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    h1, h2, h3 { color: #0E1117; }
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
        st.error("Data file not found.")
        return pd.DataFrame(), {}

    # 1. Robust Column Cleaning
    df.columns = (df.columns.astype(str)
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_'))

    # 2. Display Mapping
    display_map = {
        'healtcare_stress_index': 'Healthcare Stress',
        'fetaility_ratio': 'Fatality %',
        'clinic_total_no_of_covid_patients_currently_admitted': 'Hospital Admissions',
        'test_positivity_ratio': 'Positivity Rate',
        'oxygen_dependency_ratio': 'Oxygen Dependency',
        'grand_total_cases_till_date': 'Cumulative Cases',
        'death_cumulative_total_deaths': 'Cumulative Deaths'
    }

    # 3. Numeric Conversion
    numeric_cols = [
        'grand_total_cases_till_date', 'death_cumulative_total_deaths',
        'clinic_total_numbers_recovered_and_discharged_so_far',
        'grand_total_tests_conducted_till_date', 'test_positivity_ratio',
        'clinic_total_no_of_covid_patients_currently_admitted',
        'clinic_total_no_of_patients_currently_on_ventilator',
        'clinic_total_on_oxygen',
        'clinic_total_no_of_beds_allocated_for_covid_patients', # Needed for utilization calc
        'healtcare_stress_index', 'oxygen_dependency_ratio'
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
    
    # 5. Derived Daily Metrics (Diff)
    df['new_cases'] = df.groupby('province')['grand_total_cases_till_date'].diff().fillna(0).clip(lower=0)
    df['new_deaths'] = df.groupby('province')['death_cumulative_total_deaths'].diff().fillna(0).clip(lower=0)
    
    # 6. Smoothing (7-Day Avg)
    df['new_cases_7da'] = df.groupby('province')['new_cases'].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    df['new_deaths_7da'] = df.groupby('province')['new_deaths'].rolling(7).mean().reset_index(0, drop=True).fillna(0)

    # 7. Rt & Intelligence Features
    shifted_cases = df.groupby('province')['new_cases_7da'].shift(4).replace(0, 1)
    df['growth_factor'] = df['new_cases_7da'] / shifted_cases
    df['rt_estimate'] = df['growth_factor'].pow(1).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    if 'oxygen_dependency_ratio' in df.columns:
        df.loc[df['oxygen_dependency_ratio'] > 100, 'oxygen_dependency_ratio'] = 100

    return df, display_map

df, col_map = load_and_clean_data()
if df.empty: st.stop()

# ============================================
# 3. Aggregation Logic (Fixing the "All < Punjab" Bug)
# ============================================
st.sidebar.header("🔍 Controls")
province_list = ["All"] + sorted(df['province'].unique().tolist())
province = st.sidebar.selectbox("Region / Province", province_list)

# Date Filter
min_d, max_d = df['date'].min(), df['date'].max()
date_range = st.sidebar.date_input("Analysis Period", [max_d - timedelta(days=90), max_d], min_value=min_d, max_value=max_d)

# Filter Data
mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
df_raw = df.loc[mask]

if province == "All":
    # CRITICAL FIX: Group by Date and SUM to get National Totals
    # We aggregate numeric columns by Sum, except Rates which we recalculate or average
    agg_funcs = {col: 'sum' for col in df_raw.select_dtypes(include=np.number).columns}
    # For rates, summing is wrong (e.g., sum of positivity). We will take weighted means later or just mean for simplicity here.
    # Overwrite rate aggregation to 'mean'
    for rate in ['test_positivity_ratio', 'healtcare_stress_index', 'oxygen_dependency_ratio', 'rt_estimate']:
        if rate in agg_funcs: agg_funcs[rate] = 'mean'
            
    df_filtered = df_raw.groupby('date').agg(agg_funcs).reset_index()
    df_filtered['province'] = 'National' # Label
else:
    df_filtered = df_raw[df_raw['province'] == province]

if df_filtered.empty:
    st.warning("No data found.")
    st.stop()

# ============================================
# 4. Executive Summary (KPIs)
# ============================================
latest = df_filtered.iloc[-1]
prev_idx = max(0, len(df_filtered) - 8)
prev = df_filtered.iloc[prev_idx]

st.markdown("### 📊 Executive Summary")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("New Cases (7-Day Avg)", f"{int(latest['new_cases_7da']):,}", delta=f"{int(latest['new_cases_7da'] - prev['new_cases_7da'])}")
k2.metric("Rt (Spread Velocity)", f"{latest['rt_estimate']:.2f}", delta="Expansive" if latest['rt_estimate'] > 1 else "Shrinking", delta_color="inverse")
k3.metric("Positivity Rate", f"{latest['test_positivity_ratio']:.1f}%", delta=f"{latest['test_positivity_ratio'] - prev['test_positivity_ratio']:.1f}%", delta_color="inverse")
k4.metric("Healthcare Stress", f"{latest['healtcare_stress_index']:.1f}", help="Utilization Index")
k5.metric("Total Deaths", f"{int(latest['death_cumulative_total_deaths']):,}", delta=f"{int(latest['new_deaths'])}")

# ============================================
# 5. Dashboard Tabs
# ============================================
tabs = st.tabs(["📈 Trends & Heatmaps", "🏥 Capacity Intelligence", "🤖 AI Projections", "🔬 Data Audit", "🧮 Scenario Calculator"])

# --- TAB 1: Trends & Heatmaps (Enriched) ---
with tabs[0]:
    col_t1, col_t2 = st.columns([3, 1])
    with col_t2:
        trend_metric = st.selectbox("Select Trend Metric", ["new_cases_7da", "new_deaths", "test_positivity_ratio"], format_func=lambda x: x.replace("_", " ").title())
    
    with col_t1:
        st.subheader("Epidemic Trajectory")
    
    # Main Trend Chart
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered[trend_metric], mode='lines', fill='tozeroy', 
                                   line=dict(color='#3b82f6'), name=trend_metric.replace("_", " ").title()))
    fig_trend.update_layout(height=400, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_trend, use_container_width=True)

    # The "More Stuff" - Monthly Heatmap
    st.subheader("🗓️ Monthly Intensity Heatmap")
    try:
        df_heat = df_raw.copy() # Use raw to show province breakdown even if "All" is selected
        df_heat['month'] = df_heat['date'].dt.strftime('%Y-%m')
        
        # Pivot: Rows=Province, Cols=Month, Val=New Cases
        heatmap_data = df_heat.pivot_table(index='province', columns='month', values='new_cases', aggfunc='sum', fill_value=0)
        
        fig_heat = px.imshow(heatmap_data, color_continuous_scale="Reds", aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception as e:
        st.info("Heatmap requires full province data.")

# --- TAB 2: Capacity Intelligence ---
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Oxygen Dependency")
        fig_oxy = px.area(df_filtered, x='date', y='oxygen_dependency_ratio', color_discrete_sequence=['#10B981'])
        fig_oxy.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_oxy, use_container_width=True)
        
    with c2:
        st.subheader("Critical Care Saturation")
        fig_cc = go.Figure()
        fig_cc.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['clinic_total_no_of_covid_patients_currently_admitted'], name="Admitted"))
        fig_cc.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['clinic_total_no_of_patients_currently_on_ventilator'], name="Ventilated", line=dict(color='red', width=2)))
        fig_cc.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_cc, use_container_width=True)

# --- TAB 3: AI Projections ---
with tabs[2]:
    st.subheader("🤖 Predictive Intelligence Engine")
    
    # Added Option: Choose what to predict
    proj_target = st.radio("Target Variable", ["New Cases", "Deaths"], horizontal=True)
    target_col = 'new_cases_7da' if proj_target == "New Cases" else 'new_deaths_7da'
    
    # Data Prep
    pred_df = df_filtered[['date', target_col]].dropna()
    pred_df['days_idx'] = (pred_df['date'] - pred_df['date'].min()).dt.days
    
    if len(pred_df) > 10:
        # Polynomial Regression (Hidden Complexity)
        poly = PolynomialFeatures(degree=3)
        X = poly.fit_transform(pred_df[['days_idx']])
        model = LinearRegression().fit(X, pred_df[target_col])
        
        # Future
        future_days = 30
        last_day = pred_df['days_idx'].max()
        future_X = poly.transform(np.arange(last_day+1, last_day+future_days+1).reshape(-1, 1))
        preds = model.predict(future_X)
        dates_future = [pred_df['date'].max() + timedelta(days=i) for i in range(1, future_days+1)]
        
        # Plot
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=pred_df['date'], y=pred_df[target_col], name="Historical", line=dict(color='gray')))
        fig_ai.add_trace(go.Scatter(x=dates_future, y=preds, name="AI Forecast", line=dict(color='#F59E0B', width=3, dash='dash')))
        
        # Uncertainty Bounds
        fig_ai.add_trace(go.Scatter(x=dates_future, y=preds*1.2, line=dict(width=0), showlegend=False))
        fig_ai.add_trace(go.Scatter(x=dates_future, y=preds*0.8, line=dict(width=0), fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)', name="Confidence Interval"))
        
        st.plotly_chart(fig_ai, use_container_width=True)
    else:
        st.warning("Insufficient data for AI modeling.")

# --- TAB 4: Data Audit ---
with tabs[3]:
    st.subheader("🕵️ Data Integrity Check")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Feature Correlation**")
        corr = df_filtered[['new_cases_7da', 'test_positivity_ratio', 'healtcare_stress_index', 'rt_estimate']].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
    with col_a2:
        st.markdown("**Testing Bias Analysis**")
        if 'grand_total_tests_conducted_till_date' in df_filtered.columns:
             fig_bias = px.scatter(df_filtered, x='grand_total_tests_conducted_till_date', y='new_cases_7da', 
                                   color='test_positivity_ratio', title="Testing Volume vs Detected Cases")
             st.plotly_chart(fig_bias, use_container_width=True)

# --- TAB 5: Scenario Calculator ---
with tabs[4]:
    st.subheader("🧮 Intervention Simulator")
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("**Parameters**")
        base_cases = int(latest['new_cases_7da']) if latest['new_cases_7da'] > 0 else 1000
        current_rt = st.slider("Current Rt", 0.5, 3.0, float(max(0.5, latest['rt_estimate'])))
        
        st.markdown("**Policy Interventions**")
        p1 = st.checkbox("Mask Mandate (-15%)")
        p2 = st.checkbox("Smart Lockdown (-25%)")
        p3 = st.checkbox("School Closure (-10%)")
        
        impact = 0.15*p1 + 0.25*p2 + 0.10*p3
        new_rt = current_rt * (1 - impact)
        
        st.metric("Projected Rt", f"{new_rt:.2f}", delta=f"-{impact*100:.0f}%", delta_color="inverse")

    with col_s2:
        days = 60
        x_days = list(range(days))
        y_base = [base_cases * (current_rt**(d/4)) for d in x_days]
        y_int = [base_cases * (new_rt**(d/4)) for d in x_days]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(y=y_base, name="Do Nothing", line=dict(color='red', dash='dot')))
        fig_sim.add_trace(go.Scatter(y=y_int, name="With Intervention", line=dict(color='green', width=3)))
        fig_sim.update_layout(title="60-Day Projection", xaxis_title="Days from Now", yaxis_title="Daily Cases")
        st.plotly_chart(fig_sim, use_container_width=True)
        
        averted = sum(y_base) - sum(y_int)
        st.success(f"Estimated cases averted: **{int(averted):,}**")

