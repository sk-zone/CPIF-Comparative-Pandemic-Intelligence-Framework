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
    page_icon="🇵🇰",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional "Command Center" Look
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #007bff; }
    </style>
""", unsafe_allow_html=True)

st.title("🇵🇰 CPIF: Comparative Pandemic Intelligence Framework")
st.markdown("### National Command & Operation Center Dashboard")

# ============================================
# 2. Data Loading & Engineering (Robust)
# ============================================
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except FileNotFoundError:
        st.error("Data file 'Refined + New entities.csv' not found. Please upload it.")
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
        'death_cumulative_total_deaths': 'Total Deaths',
        'grand_total_cases_till_date': 'Total Cases'
    }

    # 3. Numeric Conversion
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

    # 5. Remove any pre-existing "All" or "Total" rows to prevent double counting
    # We will calculate "All" dynamically later.
    df = df[~df['province'].str.lower().isin(['all', 'total', 'pakistan'])]

    # 6. Feature Engineering
    # Daily counts from cumulative
    df['new_cases'] = df.groupby('province')['grand_total_cases_till_date'].diff().fillna(0).clip(lower=0)
    df['new_deaths'] = df.groupby('province')['death_cumulative_total_deaths'].diff().fillna(0).clip(lower=0)
    
    # 7-Day Moving Averages
    df['new_cases_7da'] = df.groupby('province')['new_cases'].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    df['new_deaths_7da'] = df.groupby('province')['new_deaths'].rolling(7).mean().reset_index(0, drop=True).fillna(0)

    # Rt Estimate (Growth based)
    shifted_cases = df.groupby('province')['new_cases_7da'].shift(4).replace(0, 1)
    df['growth_factor'] = df['new_cases_7da'] / shifted_cases
    df['rt_estimate'] = df['growth_factor'].pow(1).clip(0, 5) # Clip to remove insane spikes
    df['rt_estimate'] = df['rt_estimate'].fillna(1.0)
    
    # Fix Oxygen Ratio > 100% bug
    if 'oxygen_dependency_ratio' in df.columns:
        df.loc[df['oxygen_dependency_ratio'] > 100, 'oxygen_dependency_ratio'] = 100

    return df, display_map

df, col_map = load_and_clean_data()

if df.empty:
    st.stop()

# ============================================
# 3. Sidebar & Filtering
# ============================================
st.sidebar.header("🔍 Controls")
provinces_list = sorted(df['province'].unique().tolist())
selected_province = st.sidebar.selectbox("Select Region", ["All"] + provinces_list, index=0)

min_date, max_date = df['date'].min(), df['date'].max()
date_range = st.sidebar.date_input("Analysis Period", [max_date - timedelta(days=90), max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# --- Dynamic Data Aggregation (The Fix for "All < Punjab") ---
mask_date = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
df_time_slice = df.loc[mask_date]

if selected_province == "All":
    # Group by Date and Sum numerical columns to create a national view
    numeric_cols = df_time_slice.select_dtypes(include=np.number).columns
    df_filtered = df_time_slice.groupby('date')[numeric_cols].sum().reset_index()
    # Re-calculate averages for rates (cannot sum rates)
    if 'test_positivity_ratio' in df_time_slice.columns:
        df_filtered['test_positivity_ratio'] = df_time_slice.groupby('date')['test_positivity_ratio'].mean().reset_index(drop=True)
    if 'rt_estimate' in df_time_slice.columns:
        df_filtered['rt_estimate'] = df_time_slice.groupby('date')['rt_estimate'].mean().reset_index(drop=True)
else:
    df_filtered = df_time_slice[df_time_slice['province'] == selected_province]

if df_filtered.empty:
    st.warning("No data found for this selection.")
    st.stop()

# ============================================
# 4. KPI Dashboard (The "Stuff")
# ============================================
latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-8] if len(df_filtered) > 8 else latest

st.markdown("#### 📊 Situation Report")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("New Cases (7-Day Avg)", f"{int(latest.get('new_cases_7da', 0)):,}", 
            delta=f"{int(latest.get('new_cases_7da', 0) - prev.get('new_cases_7da', 0))}")
kpi2.metric("Effective Rt", f"{latest.get('rt_estimate', 1.0):.2f}", 
            delta="Expanding" if latest.get('rt_estimate', 1) > 1 else "Shrinking", delta_color="inverse")
kpi3.metric("Positivity Rate", f"{latest.get('test_positivity_ratio', 0):.2f}%",
            delta=f"{(latest.get('test_positivity_ratio', 0) - prev.get('test_positivity_ratio', 0)):.2f}%", delta_color="inverse")
kpi4.metric("Active Critical", f"{int(latest.get('clinic_total_no_of_covid_patients_currently_admitted', 0)):,}",
            help="Total Admitted Patients")
kpi5.metric("Deaths (Daily)", f"{int(latest.get('new_deaths_7da', 0))}", 
            delta=f"{int(latest.get('new_deaths_7da', 0) - prev.get('new_deaths_7da', 0))}")

# ============================================
# 5. Main Analysis Tabs
# ============================================
tabs = st.tabs([
    "📈 Intelligence & Trends", 
    "🏥 Healthcare Capacity", 
    "🌊 Wave Analysis", 
    "🤖 AI Projections", 
    "🕵️ Data Audit",
    "🧮 Scenario Calculator"
])

# --- TAB 1: Intelligence & Trends ---
with tabs[0]:
    st.subheader("Epidemic Trajectory & Regional Heatmap")
    
    col_t1, col_t2 = st.columns([2, 1])
    
    with col_t1:
        # Main Trend Line
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['new_cases'], name="Daily Cases", marker_color='rgba(59, 130, 246, 0.3)'), secondary_y=False)
        fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['new_cases_7da'], name="7-Day Trend", line=dict(color='#007bff', width=3)), secondary_y=False)
        
        if 'test_positivity_ratio' in df_filtered.columns:
            fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['test_positivity_ratio'], name="Positivity %", line=dict(color='#ffc107', dash='dot')), secondary_y=True)
            
        fig_trend.update_layout(title="Disease Spread Dynamics", template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
        fig_trend.update_yaxes(title_text="Cases", secondary_y=False)
        fig_trend.update_yaxes(title_text="Positivity %", secondary_y=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_t2:
        # Regional Comparison (If All selected)
        if selected_province == "All":
            # Get latest snapshot per province
            snapshot = df.loc[df['date'] == df['date'].max()].groupby('province')[['new_cases_7da', 'test_positivity_ratio']].sum().reset_index()
            fig_bar = px.bar(snapshot, x='new_cases_7da', y='province', orientation='h', title="Current Hotspots (Cases)", color='new_cases_7da', color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            # Gauge Charts for Province
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest.get('test_positivity_ratio', 0),
                title = {'text': "Positivity Risk"},
                gauge = {'axis': {'range': [None, 20]}, 'bar': {'color': "#ffc107"},
                         'steps': [{'range': [0, 5], 'color': "lightgreen"}, {'range': [5, 10], 'color': "orange"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

# --- TAB 2: Healthcare Capacity ---
with tabs[1]:
    st.subheader("System Stress Indicators")
    
    c1, c2 = st.columns(2)
    with c1:
        # Oxygen vs Vents
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered.get('clinic_total_on_oxygen', []), name="Patients on Oxygen", fill='tozeroy', line=dict(color='#17a2b8')))
        fig_cap.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered.get('clinic_total_no_of_patients_currently_on_ventilator', []), name="Critical (Ventilator)", line=dict(color='#dc3545', width=2)))
        fig_cap.update_layout(title="Critical Care Load (Oxygen vs. Ventilator)", template="plotly_white")
        st.plotly_chart(fig_cap, use_container_width=True)
        
    with c2:
        # Resource Saturation
        if 'healtcare_stress_index' in df_filtered.columns:
            fig_stress = px.line(df_filtered, x='date', y='healtcare_stress_index', title="Healthcare Stress Index", color_discrete_sequence=['#6610f2'])
            fig_stress.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Capacity Breach")
            st.plotly_chart(fig_stress, use_container_width=True)

# --- TAB 3: Wave Analysis (New Feature for "More Stuff") ---
with tabs[2]:
    st.subheader("🌊 Historical Wave Comparison")
    st.markdown("Comparative analysis of major infection waves.")
    
    # Define Wave Periods (Approximate for Pakistan)
    # Wave 1: Mar 20 - Jul 20
    # Wave 2: Oct 20 - Feb 21
    # Wave 3: Mar 21 - May 21
    # Wave 4: Jul 21 - Sep 21
    
    wave_data = []
    periods = {
        "Wave 1 (2020)": ("2020-03-01", "2020-07-31"),
        "Wave 2 (Winter)": ("2020-10-01", "2021-02-28"),
        "Wave 3 (Spring)": ("2021-03-15", "2021-05-31"),
        "Wave 4 (Delta)": ("2021-07-01", "2021-09-30")
    }
    
    for name, (start, end) in periods.items():
        mask_w = (df['date'] >= start) & (df['date'] <= end)
        if selected_province != "All":
            mask_w = mask_w & (df['province'] == selected_province)
        
        slice_w = df.loc[mask_w]
        if not slice_w.empty:
            peak_cases = slice_w['new_cases'].max()
            total_deaths = slice_w['new_deaths'].sum()
            avg_pos = slice_w['test_positivity_ratio'].mean() if 'test_positivity_ratio' in slice_w.columns else 0
            wave_data.append({"Wave": name, "Peak Cases": peak_cases, "Total Deaths": total_deaths, "Avg Positivity": avg_pos})
    
    if wave_data:
        wdf = pd.DataFrame(wave_data)
        
        cw1, cw2 = st.columns(2)
        with cw1:
            fig_wave_cases = px.bar(wdf, x='Wave', y='Peak Cases', title="Peak Severity by Wave", color='Peak Cases', color_continuous_scale='Reds')
            st.plotly_chart(fig_wave_cases, use_container_width=True)
        with cw2:
            fig_wave_pos = px.line(wdf, x='Wave', y='Avg Positivity', title="Avg Positivity Rate per Wave", markers=True)
            st.plotly_chart(fig_wave_pos, use_container_width=True)
            
        st.dataframe(wdf.style.format({"Peak Cases": "{:,.0f}", "Total Deaths": "{:,.0f}", "Avg Positivity": "{:.2f}%"}))

# --- TAB 4: AI Projections ---
with tabs[3]:
    st.subheader("🤖 Predictive Modeling")
    
    # Prepare Data
    model_df = df_filtered[['date', 'new_cases_7da', 'days_idx'] if 'days_idx' in df_filtered.columns else ['date', 'new_cases_7da']].dropna()
    model_df['days_idx'] = (model_df['date'] - model_df['date'].min()).dt.days
    
    if len(model_df) > 20:
        # Polynomial Regression (Degree 3)
        poly = PolynomialFeatures(degree=3)
        X = poly.fit_transform(model_df[['days_idx']])
        y = model_df['new_cases_7da']
        
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        
        # Forecast
        future_days = 30
        last_day = model_df['days_idx'].max()
        future_X = poly.transform(np.arange(last_day, last_day + future_days).reshape(-1, 1))
        preds = model.predict(future_X)
        
        # Plot
        dates_future = [model_df['date'].max() + timedelta(days=i) for i in range(future_days)]
        
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=model_df['date'], y=y, name="Historical", line=dict(color='gray')))
        fig_ai.add_trace(go.Scatter(x=dates_future, y=preds, name="AI Forecast", line=dict(color='#28a745', width=3, dash='dash')))
        
        fig_ai.update_layout(title=f"30-Day Forecast (Poly-Reg | Confidence: {r2:.2f})", template="plotly_white")
        st.plotly_chart(fig_ai, use_container_width=True)
        
        st.caption("Note: Predictions assume current intervention levels remain constant. Sudden policy changes (e.g., lockdowns) will alter this trajectory.")
    else:
        st.warning("Insufficient data for reliable AI projections.")

# --- TAB 5: Data Audit ---
with tabs[4]:
    st.subheader("🕵️ Data Integrity Check")
    
    c_audit1, c_audit2 = st.columns(2)
    with c_audit1:
        st.markdown("**Bias Detection (Tests vs Cases)**")
        if 'grand_total_tests_conducted_till_date' in df_filtered.columns:
            fig_bias = px.scatter(df_filtered, x='grand_total_tests_conducted_till_date', y='new_cases', 
                                  color='test_positivity_ratio', title="Testing Volume vs. Detected Cases")
            st.plotly_chart(fig_bias, use_container_width=True)
            
    with c_audit2:
        st.markdown("**Metric Correlations**")
        corr_cols = ['new_cases_7da', 'test_positivity_ratio', 'healtcare_stress_index', 'rt_estimate']
        valid_cols = [c for c in corr_cols if c in df_filtered.columns]
        if valid_cols:
            corr = df_filtered[valid_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 6: Scenario Calculator ---
with tabs[5]:
    st.subheader("🧮 Intervention Impact Calculator")
    
    c_calc1, c_calc2 = st.columns([1, 2])
    
    with c_calc1:
        st.markdown("### Policy Controls")
        current_rt = float(latest.get('rt_estimate', 1.0))
        base_rt = st.number_input("Baseline Rt", value=max(0.5, current_rt), format="%.2f")
        
        st.markdown("---")
        masking = st.checkbox("Strict Mask Mandate (-15%)")
        distancing = st.checkbox("Smart Lockdown (-25%)")
        vaccine_push = st.checkbox("Vaccination Drive (-10%)")
        
        impact = 0
        if masking: impact += 0.15
        if distancing: impact += 0.25
        if vaccine_push: impact += 0.10
        
        final_rt = base_rt * (1 - impact)
        st.metric("Adjusted Rt", f"{final_rt:.2f}", delta=f"-{impact*100:.0f}%")

    with c_calc2:
        # Projection Logic
        days = 60
        x_axis = [datetime.now() + timedelta(days=i) for i in range(days)]
        start_cases = latest.get('new_cases_7da', 1000)
        
        # Exponential Growth Formula: N_t = N_0 * R^(t/serial_interval)
        # Serial interval ~4 days
        y_base = [start_cases * (base_rt ** (i/4)) for i in range(days)]
        y_int = [start_cases * (final_rt ** (i/4)) for i in range(days)]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=x_axis, y=y_base, name="No Action", line=dict(color='#dc3545', dash='dot')))
        fig_sim.add_trace(go.Scatter(x=x_axis, y=y_int, name="With Intervention", line=dict(color='#28a745', width=3)))
        fig_sim.add_trace(go.Scatter(x=x_axis, y=[2000]*days, name="Healthcare Capacity", line=dict(color='black', dash='longdashdot')))
        
        fig_sim.update_layout(title="Projected Daily Cases", yaxis_title="Cases", template="plotly_white")
        st.plotly_chart(fig_sim, use_container_width=True)
        
        averted = sum(y_base) - sum(y_int)
        st.success(f"Potential Cases Averted: {int(averted):,}")

