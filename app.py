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

st.markdown("""
    <style>
    .metric-card { background-color: rgba(255, 255, 255, 0.05); border-left: 5px solid #3b82f6; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; color: #888; }
    .stTabs [aria-selected="true"] { background-color: rgba(59, 130, 246, 0.1); border-bottom: 2px solid #3b82f6; color: #3b82f6; font-weight: bold; }
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
    df.columns = (df.columns.astype(str).str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_'))

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
            df[col] = (df[col].astype(str).str.replace(',', '', regex=False).str.replace('%', '', regex=False).replace(['N/A', 'n/a', '-', '', 'nan', 'Nan'], '0'))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Dates & Sorting
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by=['province', 'date'])
    else:
        return pd.DataFrame(), {}

    # 5. Feature Engineering
    df['new_cases'] = df.groupby('province')['grand_total_cases_till_date'].diff().fillna(0).clip(lower=0)
    df['new_deaths'] = df.groupby('province')['death_cumulative_total_deaths'].diff().fillna(0).clip(lower=0)
    df['new_cases_7da'] = df.groupby('province')['new_cases'].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    
    # Rt Proxy
    shifted_cases = df.groupby('province')['new_cases_7da'].shift(4).replace(0, 1)
    df['growth_factor'] = df['new_cases_7da'] / shifted_cases
    df['rt_estimate'] = df['growth_factor'].pow(1).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Wave Definitions (Restoring Group's Request)
    df["wave"] = "Inter-Wave Period"
    df.loc[df["date"].between("2020-03-01", "2020-07-31"), "wave"] = "Wave 1 (Original)"
    df.loc[df["date"].between("2020-10-01", "2021-01-31"), "wave"] = "Wave 2 (Winter)"
    df.loc[df["date"].between("2021-03-01", "2021-05-31"), "wave"] = "Wave 3 (Alpha/Beta)"
    df.loc[df["date"].between("2021-07-01", "2021-10-31"), "wave"] = "Wave 4 (Delta)"
    df.loc[df["date"].between("2021-12-01", "2022-03-31"), "wave"] = "Wave 5 (Omicron)"
    
    return df, display_map

df_raw, col_map = load_and_clean_data()

if df_raw.empty:
    st.error("Data file not found. Please upload 'Refined + New entities.csv'.")
    st.stop()

# ============================================
# 3. Sidebar & Filtering
# ============================================
st.sidebar.header("🔍 Controls")
provinces_list = ["All (National Aggregate)"] + sorted(df_raw['province'].unique().tolist())
selected_prov = st.sidebar.selectbox("Select Region", provinces_list, index=0)

if selected_prov == "All (National Aggregate)":
    numeric_cols_df = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    # Need to keep 'wave' column which is string, so we group by date and wave first (if wave is consistent per date)
    # Simpler approach: Agg numbers, re-map waves by date
    df_filtered = df_raw.groupby('date')[numeric_cols_df].sum().reset_index()
    # Re-apply wave logic to aggregate
    df_filtered["wave"] = "Inter-Wave Period"
    df_filtered.loc[df_filtered["date"].between("2020-03-01", "2020-07-31"), "wave"] = "Wave 1 (Original)"
    df_filtered.loc[df_filtered["date"].between("2020-10-01", "2021-01-31"), "wave"] = "Wave 2 (Winter)"
    df_filtered.loc[df_filtered["date"].between("2021-03-01", "2021-05-31"), "wave"] = "Wave 3 (Alpha/Beta)"
    df_filtered.loc[df_filtered["date"].between("2021-07-01", "2021-10-31"), "wave"] = "Wave 4 (Delta)"
    df_filtered.loc[df_filtered["date"].between("2021-12-01", "2022-03-31"), "wave"] = "Wave 5 (Omicron)"
    
    # Re-calc Rt/Trends for Agg
    df_filtered['new_cases_7da'] = df_filtered['new_cases'].rolling(7).mean().fillna(0)
    shifted = df_filtered['new_cases_7da'].shift(4).replace(0, 1)
    df_filtered['rt_estimate'] = (df_filtered['new_cases_7da'] / shifted).fillna(1.0)
    if 'test_positivity_ratio' in df_raw.columns:
        df_filtered['test_positivity_ratio'] = df_raw.groupby('date')['test_positivity_ratio'].mean().values
else:
    df_filtered = df_raw[df_raw['province'] == selected_prov].copy()

min_d, max_d = df_filtered['date'].min(), df_filtered['date'].max()
dates = st.sidebar.date_input("Analysis Period", [max_d - timedelta(days=90), max_d], min_value=min_d, max_value=max_d)

if len(dates) == 2:
    start_d, end_d = dates
    df_filtered = df_filtered[(df_filtered['date'].dt.date >= start_d) & (df_filtered['date'].dt.date <= end_d)]

if df_filtered.empty:
    st.warning("⚠️ No data available for this date range.")
    st.stop()

# ============================================
# 4. Main Dashboard (Expanded KPIs)
# ============================================
latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-8] if len(df_filtered) > 8 else latest
rt_now = df_filtered['rt_estimate'].mean()

# utilization calcs
beds_used = latest.get('clinic_total_no_of_covid_patients_currently_admitted', 0)
beds_total = latest.get('clinic_total_no_of_beds_allocated_for_covid_patients', 1) 
beds_pct = (beds_used / beds_total) * 100 if beds_total > 0 else 0
vent_used = latest.get('clinic_total_no_of_patients_currently_on_ventilator', 0)
vent_total = latest.get('clinic_total_no_of_ventilators_allocated_for_covid_patients', 1)
vent_pct = (vent_used / vent_total) * 100 if vent_total > 0 else 0

st.markdown("#### 📊 Real-time Situation Report")
# Row 1: Infection Metrics
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Cases", f"{int(latest.get('grand_total_cases_till_date',0)):,}", delta=f"+{int(latest['new_cases'])}")
with c2: st.metric("Recovered", f"{int(latest.get('clinic_total_numbers_recovered_and_discharged_so_far',0)):,}")
with c3: st.metric("Active Cases (Est)", f"{int(latest.get('grand_total_cases_till_date',0) - latest.get('clinic_total_numbers_recovered_and_discharged_so_far',0) - latest.get('death_cumulative_total_deaths',0)):,}")
with c4: st.metric("Total Deaths", f"{int(latest.get('death_cumulative_total_deaths',0)):,}", delta=f"+{int(latest['new_deaths'])}")

# Row 2: Capacity & Intelligence
d1, d2, d3, d4 = st.columns(4)
with d1: st.metric("Bed Utilization", f"{beds_pct:.1f}%", delta="Critical" if beds_pct > 80 else "Stable", delta_color="inverse")
with d2: st.metric("Ventilator Utilization", f"{vent_pct:.1f}%", delta="Critical" if vent_pct > 70 else "Stable", delta_color="inverse")
with d3: st.metric("Rt (Spread Velocity)", f"{rt_now:.2f}", delta="Exp. Growth" if rt_now > 1.1 else "Contained", delta_color="inverse")
with d4: st.metric("Positivity Rate", f"{latest.get('test_positivity_ratio', 0):.1f}%", delta=f"{(latest.get('test_positivity_ratio',0) - prev.get('test_positivity_ratio',0)):.1f}%", delta_color="inverse")

# ============================================
# 5. Visual Tabs
# ============================================
t1, t2, t3, t4, t5 = st.tabs(["📈 Trends & Heatmaps", "🏥 Resource Intelligence", "📊 Wave Analysis", "🔮 AI Projections", "♟️ Dynamic Intervention"])

# --- Tab 1: Trends & Heatmaps ---
with t1:
    st.subheader("Epidemic Trajectory")
    
    # 1. Main Trend Line
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['new_cases'], name="Daily Cases", marker_color='rgba(59, 130, 246, 0.4)'), secondary_y=False)
    fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['new_cases_7da'], name="7-Day Trend", line=dict(color='#2563EB', width=3)), secondary_y=False)
    if 'test_positivity_ratio' in df_filtered.columns:
        fig_trend.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['test_positivity_ratio'], name="Positivity %", line=dict(color='#DC2626', dash='dot')), secondary_y=True)
    fig_trend.update_layout(height=400, hovermode="x unified", legend=dict(orientation="h", y=1.1), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_trend, use_container_width=True)

    # 2. Monthly Heatmap (New Addition)
    st.subheader("📅 Monthly Infection Heatmap")
    df_heat = df_raw.copy()
    if selected_prov != "All (National Aggregate)":
        df_heat = df_heat[df_heat['province'] == selected_prov]
    
    df_heat['YearMonth'] = df_heat['date'].dt.strftime('%Y-%m')
    heatmap_data = df_heat.pivot_table(index='province', columns='YearMonth', values='new_cases', aggfunc='sum', fill_value=0)
    
    fig_heat = px.imshow(heatmap_data, color_continuous_scale='Reds', aspect='auto', title=f"Monthly New Cases ({selected_prov})")
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_heat, use_container_width=True)

# --- Tab 2: Resource Intelligence ---
with t2:
    st.subheader("Healthcare Capacity Audit")
    
    # 1. Detailed Table (New Addition)
    res_cols = ['date', 'clinic_total_no_of_covid_patients_currently_admitted', 
                'clinic_total_no_of_patients_currently_on_ventilator', 'clinic_total_on_oxygen', 'oxygen_dependency_ratio']
    
    # Rename for display
    disp_cols = {
        'clinic_total_no_of_covid_patients_currently_admitted': 'Admitted',
        'clinic_total_no_of_patients_currently_on_ventilator': 'On Ventilator',
        'clinic_total_on_oxygen': 'On Oxygen',
        'oxygen_dependency_ratio': 'Oxygen Dependency %'
    }
    
    st.dataframe(df_filtered[res_cols].sort_values('date', ascending=False).rename(columns=disp_cols).style.background_gradient(cmap="Reds", subset=['Oxygen Dependency %']), use_container_width=True)

    # 2. Utilization Trend Plot (New Addition)
    st.subheader("Resource Stress Trends")
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['clinic_total_no_of_covid_patients_currently_admitted'], name="Total Admitted", fill='tozeroy', line=dict(color='#6366F1')))
    fig_res.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['clinic_total_no_of_patients_currently_on_ventilator'], name="Critical (Vent)", line=dict(color='#EF4444', width=2)))
    fig_res.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_res, use_container_width=True)

# --- Tab 3: Wave Analysis (New Addition) ---
with t3:
    st.subheader("🌊 Wave-wise Comparative Intelligence")
    
    # Aggregate data by Wave
    wave_stats = df_filtered.groupby('wave').agg({
        'new_cases': 'sum',
        'new_deaths': 'sum',
        'new_cases_7da': 'max' # Peak cases
    }).reset_index()
    
    # Filter out Inter-Wave if desired, or keep
    
    c_w1, c_w2 = st.columns(2)
    
    with c_w1:
        st.markdown("**Peak Cases by Wave**")
        fig_peak = px.bar(wave_stats, x='wave', y='new_cases_7da', color='wave', title="Peak Daily Cases (Intensity)")
        fig_peak.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_peak, use_container_width=True)
        
    with c_w2:
        st.markdown("**Total Deaths by Wave**")
        fig_death = px.bar(wave_stats, x='wave', y='new_deaths', color='wave', title="Total Mortality (Severity)")
        fig_death.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_death, use_container_width=True)

# --- Tab 4: AI Projections ---
with t4:
    st.subheader("Predictive Analytics")
    col_p1, col_p2 = st.columns([1, 4])
    with col_p1:
        forecast_days = st.slider("Forecast Horizon (Days)", 15, 90, 30)
    
    with col_p2:
        pred_df = df_filtered[['date', 'new_cases_7da']].dropna()
        if len(pred_df) > 20:
            pred_df['days_idx'] = (pred_df['date'] - pred_df['date'].min()).dt.days
            poly = PolynomialFeatures(degree=3)
            X = poly.fit_transform(pred_df[['days_idx']])
            y = pred_df['new_cases_7da']
            model = LinearRegression().fit(X, y)
            
            future_X = poly.transform(np.arange(pred_df['days_idx'].max() + 1, pred_df['days_idx'].max() + forecast_days + 1).reshape(-1, 1))
            preds = model.predict(future_X)
            future_dates = [pred_df['date'].max() + timedelta(days=x) for x in range(1, forecast_days + 1)]
            
            fig_ai = go.Figure()
            fig_ai.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['new_cases_7da'], name="Historical", line=dict(color='gray')))
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", line=dict(color='#F59E0B', width=3, dash='dash')))
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds*1.2, mode='lines', line=dict(width=0), showlegend=False))
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds*0.8, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)', name="Uncertainty"))
            fig_ai.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_ai, use_container_width=True)

# --- Tab 5: Dynamic Intervention (Enhanced) ---
with t5:
    st.subheader("♟️ Dynamic Policy Simulator")
    st.markdown("Adjust intervention intensity to see impact on Rt and projected cases.")
    
    ci1, ci2 = st.columns([1, 2])
    with ci1:
        base_rt_sim = st.number_input("Baseline Rt", value=float(max(0.5, rt_now)), step=0.05, format="%.2f")
        
        # Dynamic Sliders (No more tick cross)
        s_mask = st.slider("😷 Mask Mandate Intensity", 0, 100, 0, help="Max impact: 15% reduction")
        s_dist = st.slider("📏 Social Distancing Strictness", 0, 100, 0, help="Max impact: 10% reduction")
        s_school = st.slider("🏠 Remote Work/School %", 0, 100, 0, help="Max impact: 12% reduction")
        s_lock = st.slider("🔒 Lockdown Severity", 0, 100, 0, help="Max impact: 25% reduction")
        
        # Weighted Reduction Calculation
        reduction = (s_mask/100 * 0.15) + (s_dist/100 * 0.10) + (s_school/100 * 0.12) + (s_lock/100 * 0.25)
        
        final_rt_sim = max(0.1, base_rt_sim * (1 - reduction))
        st.metric("Adjusted Rt", f"{final_rt_sim:.2f}", delta=f"-{reduction*100:.1f}% Impact")

    with ci2:
        sim_days = 60
        start_val = max(100, latest['new_cases_7da'])
        dates_sim = [datetime.now() + timedelta(days=x) for x in range(sim_days)]
        
        curve_base = [start_val * (base_rt_sim**(d/4)) for d in range(sim_days)]
        curve_int = [start_val * (final_rt_sim**(d/4)) for d in range(sim_days)]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=dates_sim, y=curve_base, name="Status Quo", line=dict(color='#ef4444', dash='dot')))
        fig_sim.add_trace(go.Scatter(x=dates_sim, y=curve_int, name="With Intervention", line=dict(color='#10b981', width=3)))
        fig_sim.update_layout(title=f"Projected Cases Saved: {int(sum(curve_base) - sum(curve_int)):,}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("---")
st.caption("Framework v4.0 | CPIF")

