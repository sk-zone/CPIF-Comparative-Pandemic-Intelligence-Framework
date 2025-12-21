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

# Custom CSS for "Pandemic Command Center" Theme
st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ CPIF: Comparative Pandemic Intelligence Framework")

# ============================================
# 2. Data Engine (Robust & Aggregated)
# ============================================
@st.cache_data
def load_and_process_data():
    # Load Data
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except FileNotFoundError:
        st.error("Data file 'Refined + New entities.csv' not found.")
        return pd.DataFrame(), {}

    # 1. Robust Column Cleaning (Regex)
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
        'new_cases_7da': 'New Cases (7-Day Avg)'
    }

    # 3. Type Conversion
    numeric_cols = [
        'grand_total_cases_till_date', 
        'death_cumulative_total_deaths',
        'clinic_total_numbers_recovered_and_discharged_so_far',
        'grand_total_tests_conducted_till_date', 
        'test_positivity_ratio',
        'clinic_total_no_of_covid_patients_currently_admitted',
        'clinic_total_no_of_patients_currently_on_ventilator',
        'clinic_total_on_oxygen',
        'healtcare_stress_index', 
        'oxygen_dependency_ratio'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace(',', '', regex=False)
                       .str.replace('%', '', regex=False)
                       .replace(['N/A', 'n/a', '-', '', 'nan', 'Nan'], '0'))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Date & Sort
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by=['province', 'date'])
    else:
        st.error("Column 'date' not found.")
        return pd.DataFrame(), {}

    # 5. Core Feature Engineering
    # Calculate Daily New Numbers from Cumulative
    df['new_cases'] = df.groupby('province')['grand_total_cases_till_date'].diff().fillna(0).clip(lower=0)
    df['new_deaths'] = df.groupby('province')['death_cumulative_total_deaths'].diff().fillna(0).clip(lower=0)
    
    # Smoothers
    df['new_cases_7da'] = df.groupby('province')['new_cases'].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    df['new_deaths_7da'] = df.groupby('province')['new_deaths'].rolling(7).mean().reset_index(0, drop=True).fillna(0)

    # Wave Definition (For Comparisons)
    def assign_wave(d):
        if d < pd.Timestamp("2020-05-31"): return "Wave 1"
        elif d < pd.Timestamp("2020-10-31"): return "Lull 1"
        elif d < pd.Timestamp("2021-02-28"): return "Wave 2"
        elif d < pd.Timestamp("2021-06-30"): return "Wave 3 (Alpha/Beta)"
        elif d < pd.Timestamp("2021-12-31"): return "Wave 4 (Delta)"
        elif d < pd.Timestamp("2022-03-31"): return "Wave 5 (Omicron)"
        else: return "Endemic Phase"
    
    df['wave'] = df['date'].apply(assign_wave)

    return df, display_map

df_raw, col_map = load_and_process_data()

if df_raw.empty:
    st.stop()

# ============================================
# 3. Sidebar Controls
# ============================================
st.sidebar.header("🔍 Intelligence Filters")

# Province Selection
provinces = ["All"] + sorted(df_raw['province'].unique().tolist())
province_sel = st.sidebar.selectbox("Region / Province", provinces, index=0)

# Date Selection
min_d, max_d = df_raw['date'].min().date(), df_raw['date'].max().date()
date_range = st.sidebar.date_input("Analysis Period", [max_d - timedelta(days=90), max_d], min_value=min_d, max_value=max_d)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_d, max_d

# ============================================
# 4. Aggregation Logic (The "Fix")
# ============================================
# Filter by Date first
mask_date = (df_raw['date'].dt.date >= start_date) & (df_raw['date'].dt.date <= end_date)
df_period = df_raw.loc[mask_date]

if province_sel == "All":
    # CRITICAL FIX: Aggregate SUM for "All"
    # We group by Date to get the national total for each day
    numeric_cols = df_period.select_dtypes(include=np.number).columns.tolist()
    df_agg = df_period.groupby('date')[numeric_cols].sum().reset_index()
    # Re-calculate averages/ratios for the aggregated data since summing ratios is wrong
    if 'grand_total_tests_conducted_till_date' in df_agg.columns and 'grand_total_cases_till_date' in df_agg.columns:
         # Simplified TPR re-calc for aggregation
         df_agg['test_positivity_ratio'] = df_period.groupby('date')['test_positivity_ratio'].mean().values # Mean of TPR is a safe proxy for visualization
    
    # Recalculate 7DA for the Aggregated Line
    df_agg['new_cases_7da'] = df_agg['new_cases'].rolling(7).mean().fillna(0)
    df_agg['new_deaths_7da'] = df_agg['new_deaths'].rolling(7).mean().fillna(0)
    
    # For Wave Analysis, we need the full dataset with provinces
    df_analysis = df_period.copy() 
    
else:
    # Filter by Province
    df_agg = df_period[df_period['province'] == province_sel].copy()
    df_analysis = df_agg.copy()

if df_agg.empty:
    st.warning("No data found for this selection.")
    st.stop()

# ============================================
# 5. KPI Dashboard
# ============================================
# Get Latest available day in the filtered range
latest = df_agg.iloc[-1]
# Get comparison (7 days ago within the range, or first day if range < 7)
prev_idx = max(0, len(df_agg) - 8)
prev = df_agg.iloc[prev_idx]

# Rt Estimation (Simple Growth Proxy)
try:
    current_cases_7da = max(1, latest['new_cases_7da'])
    past_cases_7da = max(1, prev['new_cases_7da'])
    rt_proxy = (current_cases_7da / past_cases_7da) ** (4/7) # 4-day serial interval
except:
    rt_proxy = 1.0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("New Cases (Daily)", f"{int(latest['new_cases']):,}", delta=f"{int(latest['new_cases'] - prev['new_cases'])}")
with c2:
    st.metric("Active Cases Trend", f"{int(latest['new_cases_7da']):,}", delta=f"{(latest['new_cases_7da'] - prev['new_cases_7da']):.1f} (7-Day Avg)")
with c3:
    st.metric("Rt (Growth Rate)", f"{rt_proxy:.2f}", delta="Accelerating" if rt_proxy > 1.1 else "Stable", delta_color="inverse")
with c4:
    # Handle missing columns gracefully
    tpr = latest.get('test_positivity_ratio', 0)
    prev_tpr = prev.get('test_positivity_ratio', 0)
    st.metric("Positivity Rate", f"{tpr:.2f}%", delta=f"{tpr - prev_tpr:.2f}%", delta_color="inverse")
with c5:
    deaths = latest.get('death_cumulative_total_deaths', 0)
    new_deaths = latest.get('new_deaths', 0)
    st.metric("Total Deaths", f"{int(deaths):,}", delta=f"+{int(new_deaths)}")

# ============================================
# 6. Main Tabs
# ============================================
tabs = st.tabs([
    "📈 Trend Analysis", 
    "🏥 Healthcare Capacity", 
    "🔮 AI Projections", 
    "📊 Comparative Intelligence",
    "🕵️ Data Audit",
    "🧮 Policy Calculator"
])

# --- TAB 1: Trend Analysis ---
with tabs[0]:
    col_opt1, col_opt2 = st.columns([1, 5])
    with col_opt1:
        st.markdown("##### View Options")
        log_scale = st.checkbox("Logarithmic Scale", help="Useful for seeing early exponential growth")
        show_ma = st.checkbox("Show 7-Day Avg", value=True)
    
    with col_opt2:
        # Main Epidemic Curve
        fig_main = go.Figure()
        fig_main.add_trace(go.Bar(x=df_agg['date'], y=df_agg['new_cases'], name="Daily Cases", marker_color='rgba(59, 130, 246, 0.4)'))
        if show_ma:
            fig_main.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['new_cases_7da'], name="7-Day Average", line=dict(color='#2563EB', width=3)))
        
        fig_main.update_layout(
            title=f"Epidemic Curve - {province_sel}",
            yaxis_type="log" if log_scale else "linear",
            template="plotly_white",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_main, use_container_width=True)

    # Sub-charts
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        # Deaths
        fig_d = px.line(df_agg, x='date', y='new_deaths_7da', title="Daily Deaths (7-Day Avg)", color_discrete_sequence=['#EF4444'])
        st.plotly_chart(fig_d, use_container_width=True)
    with c_sub2:
        # Positivity
        if 'test_positivity_ratio' in df_agg.columns:
            fig_p = px.line(df_agg, x='date', y='test_positivity_ratio', title="Test Positivity Rate (%)", color_discrete_sequence=['#F59E0B'])
            # Add threshold line
            fig_p.add_hline(y=5, line_dash="dot", annotation_text="WHO Threshold (5%)", annotation_position="bottom right")
            st.plotly_chart(fig_p, use_container_width=True)

# --- TAB 2: Healthcare Capacity ---
with tabs[1]:
    st.subheader("Resource Utilization & Stress")
    
    # Only show if columns exist
    stress_cols = ['clinic_total_no_of_covid_patients_currently_admitted', 'clinic_total_no_of_patients_currently_on_ventilator', 'clinic_total_on_oxygen']
    valid_cols = [c for c in stress_cols if c in df_agg.columns]
    
    if valid_cols:
        fig_res = go.Figure()
        if 'clinic_total_no_of_covid_patients_currently_admitted' in df_agg.columns:
            fig_res.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['clinic_total_no_of_covid_patients_currently_admitted'], name="Total Admitted", fill='tozeroy'))
        if 'clinic_total_on_oxygen' in df_agg.columns:
            fig_res.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['clinic_total_on_oxygen'], name="On Oxygen", line=dict(width=2)))
        if 'clinic_total_no_of_patients_currently_on_ventilator' in df_agg.columns:
            fig_res.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['clinic_total_no_of_patients_currently_on_ventilator'], name="Critical (Ventilator)", line=dict(color='red', width=2)))
        
        fig_res.update_layout(title="Hospital Load Breakdown", template="plotly_white", height=400)
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Healthcare resource data not available for this selection.")

# --- TAB 3: AI Projections ---
with tabs[2]:
    st.subheader("Projections")
    
    # Model Setup
    model_df = df_agg[['date', 'new_cases_7da']].dropna()
    if len(model_df) > 14:
        # Feature Engineering: Days since start
        model_df['days'] = (model_df['date'] - model_df['date'].min()).dt.days
        
        # Polynomial Features (Degree 3)
        poly = PolynomialFeatures(degree=3)
        X = poly.fit_transform(model_df[['days']])
        y = model_df['new_cases_7da']
        
        # Train
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict Future (30 Days)
        last_day = model_df['days'].max()
        future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
        future_X = poly.transform(future_days)
        predictions = model.predict(future_X)
        predictions = np.maximum(predictions, 0) # No negative cases
        
        # Dates
        last_date = model_df['date'].max()
        future_dates = [last_date + timedelta(days=int(x)) for x in range(1, 31)]
        
        # Plot
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=model_df['date'], y=model_df['new_cases_7da'], name="Historical Trend", line=dict(color='gray')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=predictions, name="Projected Trend", line=dict(color='#F59E0B', width=3, dash='dash')))
        
        # Confidence Interval (Simulated visual for UI)
        upper = predictions * 1.2
        lower = predictions * 0.8
        fig_ai.add_trace(go.Scatter(x=future_dates, y=upper, mode='lines', line=dict(width=0), showlegend=False))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)', name="Uncertainty Range"))
        
        fig_ai.update_layout(title="30-Day Trend Forecast", template="plotly_dark", height=500)
        st.plotly_chart(fig_ai, use_container_width=True)
        st.caption("Note: Projections assume current interaction patterns and testing rates persist.")
    else:
        st.warning("Insufficient data for reliable projections in this time range.")

# --- TAB 4: Comparative Intelligence (RESTORED & IMPROVED) ---
with tabs[3]:
    st.subheader("Historical & Regional Comparison")
    
    # 1. Wave Comparison (The "Old Stuff" Refined)
    # Use the full raw dataset for this to show all history
    wave_stats = df_raw.groupby('wave').agg({
        'new_cases': 'mean',
        'new_deaths': 'sum',
        'test_positivity_ratio': 'mean'
    }).reset_index()
    
    # Sort waves logically (not alphabetically)
    wave_order = ["Wave 1", "Lull 1", "Wave 2", "Wave 3 (Alpha/Beta)", "Wave 4 (Delta)", "Wave 5 (Omicron)", "Endemic Phase"]
    wave_stats['wave'] = pd.Categorical(wave_stats['wave'], categories=wave_order, ordered=True)
    wave_stats = wave_stats.sort_values('wave')
    
    c_comp1, c_comp2 = st.columns(2)
    with c_comp1:
        fig_w = px.bar(wave_stats, x='wave', y='new_cases', title="Average Daily Cases per Wave", color='new_cases', color_continuous_scale='Blues')
        st.plotly_chart(fig_w, use_container_width=True)
    with c_comp2:
        fig_wd = px.bar(wave_stats, x='wave', y='new_deaths', title="Total Deaths per Wave", color='new_deaths', color_continuous_scale='Reds')
        st.plotly_chart(fig_wd, use_container_width=True)
        
    # 2. Regional Heatmap (If All Selected)
    if province_sel == "All":
        st.subheader("Regional Intensity Map")
        # Pivot for heatmap
        # Group by Month and Province
        df_hm = df_raw.copy()
        df_hm['month'] = df_hm['date'].dt.to_period('M').astype(str)
        hm_data = df_hm.pivot_table(index='province', columns='month', values='new_cases', aggfunc='sum').fillna(0)
        
        fig_hm = px.imshow(hm_data, labels=dict(x="Month", y="Province", color="Total Cases"), title="Heatmap: Monthly Cases by Province")
        st.plotly_chart(fig_hm, use_container_width=True)

# --- TAB 5: Data Audit ---
with tabs[4]:
    st.subheader("Data Integrity")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Bias Check: Tests vs Cases**")
        if 'grand_total_tests_conducted_till_date' in df_agg.columns:
            fig_bias = px.scatter(df_agg, x='grand_total_tests_conducted_till_date', y='grand_total_cases_till_date', 
                                  color='test_positivity_ratio', title="Testing Volume vs Detected Cases")
            st.plotly_chart(fig_bias, use_container_width=True)
            st.caption("If points drift left but stay high, it indicates under-testing.")
            
    with col_a2:
        st.markdown("**Missing Data Report**")
        missing = df_agg.isnull().sum()
        st.dataframe(missing[missing > 0], use_container_width=True)

# --- TAB 6: Policy Calculator ---
with tabs[5]:
    st.subheader("Intervention Simulator")
    
    c_calc1, c_calc2 = st.columns([1, 2])
    with c_calc1:
        st.markdown("**Simulation Parameters**")
        sim_rt = st.slider("Base Rt", 0.5, 3.0, float(max(0.5, min(3.0, rt_proxy))), 0.1)
        sim_active = int(latest['new_cases_7da'])
        
        st.markdown("**Policy Bundle**")
        p_mask = st.checkbox("Strict Masking (-15%)")
        p_dist = st.checkbox("Social Distancing (-20%)")
        p_school = st.checkbox("Remote Schooling (-10%)")
        
        reduction = 0.0
        if p_mask: reduction += 0.15
        if p_dist: reduction += 0.20
        if p_school: reduction += 0.10
        
        final_rt = sim_rt * (1 - reduction)
        st.metric("Effective Rt", f"{final_rt:.2f}", delta=f"-{reduction*100:.0f}%")
        
    with c_calc2:
        days = 60
        x_days = list(range(days))
        y_base = [sim_active * (sim_rt ** (d/4)) for d in x_days]
        y_int = [sim_active * (final_rt ** (d/4)) for d in x_days]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=x_days, y=y_base, name="Do Nothing", line=dict(color='red', dash='dot')))
        fig_sim.add_trace(go.Scatter(x=x_days, y=y_int, name="With Intervention", line=dict(color='green', width=3)))
        fig_sim.update_layout(title="Projected Daily Cases (60 Days)", xaxis_title="Days from Now", yaxis_title="Daily Cases")
        st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("---")
st.caption(f"CPIF v3.0 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
