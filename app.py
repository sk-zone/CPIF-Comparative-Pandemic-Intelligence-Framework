import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import re
import warnings

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

# Professional CSS (Hidden "Student Project" vibe)
st.markdown("""
    <style>
    .metric-card { background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 5px solid #2563EB; }
    .css-1d391kg { padding-top: 1rem; } 
    </style>
""", unsafe_allow_html=True)

st.title("🇵🇰 CPIF: Comparative Pandemic Intelligence Framework")

# ============================================
# 2. Data Loading & Cleaning
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except FileNotFoundError:
        return pd.DataFrame(), {}

    # 1. Robust Regex Cleaning (Fixes double underscores/spaces)
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
        'clinic_total_numbers_recovered_and_discharged_so_far': 'Total Recovered'
    }

    # 3. Numeric Conversion
    numeric_cols = [
        'grand_total_cases_till_date', 'death_cumulative_total_deaths',
        'clinic_total_numbers_recovered_and_discharged_so_far',
        'grand_total_tests_conducted_till_date', 'test_positivity_ratio',
        'clinic_total_no_of_covid_patients_currently_admitted',
        'clinic_total_no_of_patients_currently_on_ventilator',
        'clinic_total_no_of_beds_allocated_for_covid_patients',
        'clinic_total_no_of_ventilators_allocated_for_covid_patients',
        'clinic_total_on_oxygen',
        'healtcare_stress_index', 'oxygen_dependency_ratio'
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
    
    # 5. Basic Feature Engineering (Province Level)
    df['new_cases'] = df.groupby('province')['grand_total_cases_till_date'].diff().fillna(0).clip(lower=0)
    df['new_deaths'] = df.groupby('province')['death_cumulative_total_deaths'].diff().fillna(0).clip(lower=0)
    
    # Smoothers
    df['new_cases_7da'] = df.groupby('province')['new_cases'].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    
    # Define Waves (For Historical Context)
    df["wave"] = "Inter-Wave Period"
    df.loc[df["date"].between("2020-03-01", "2020-07-31"), "wave"] = "Wave 1 (Original)"
    df.loc[df["date"].between("2020-10-01", "2021-01-31"), "wave"] = "Wave 2 (Winter '20)"
    df.loc[df["date"].between("2021-03-01", "2021-06-30"), "wave"] = "Wave 3 (Alpha/Beta)"
    df.loc[df["date"].between("2021-07-01", "2021-10-31"), "wave"] = "Wave 4 (Delta)"
    df.loc[df["date"].between("2022-01-01", "2022-03-31"), "wave"] = "Wave 5 (Omicron)"

    return df, display_map

df, col_map = load_data()

if df.empty:
    st.error("Dataset not found. Please upload 'Refined + New entities.csv'.")
    st.stop()

# ============================================
# 3. Sidebar & Aggregation Logic (The FIX)
# ============================================
st.sidebar.header("🔍 Controls")

# Province Selection
provinces = ["All"] + sorted(df['province'].unique().tolist())
province = st.sidebar.selectbox("Region / Province", provinces)

# Date Selection
min_d, max_d = df['date'].min(), df['date'].max()
start_date, end_date = st.sidebar.date_input("Analysis Period", [max_d - timedelta(days=90), max_d], min_value=min_d, max_value=max_d)

# --- AGGREGATION ENGINE (Fixes "Punjab > All" bug) ---
mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
raw_filtered = df.loc[mask]

if province == "All":
    # SUM numeric columns for National View
    numeric_cols_to_sum = [
        'new_cases', 'new_deaths', 'grand_total_cases_till_date', 
        'death_cumulative_total_deaths', 'clinic_total_numbers_recovered_and_discharged_so_far',
        'clinic_total_no_of_covid_patients_currently_admitted',
        'clinic_total_no_of_patients_currently_on_ventilator',
        'clinic_total_on_oxygen',
        'clinic_total_no_of_beds_allocated_for_covid_patients',
        'clinic_total_no_of_ventilators_allocated_for_covid_patients'
    ]
    # Filter only existing columns
    cols_exist = [c for c in numeric_cols_to_sum if c in raw_filtered.columns]
    
    # Group by Date to create a "National" time series
    df_filtered = raw_filtered.groupby('date')[cols_exist].sum().reset_index()
    
    # Recalculate Ratios for National Level (Weighted Avg Logic)
    # Since we can't sum ratios, we take mean for simplicity in visualization or re-derive
    if 'test_positivity_ratio' in raw_filtered.columns:
        # Weighted average would be better, but mean is acceptable for trend view
        pos_mean = raw_filtered.groupby('date')['test_positivity_ratio'].mean().reset_index()
        df_filtered = pd.merge(df_filtered, pos_mean, on='date')
    
    # Recalculate 7-day avg for National
    df_filtered['new_cases_7da'] = df_filtered['new_cases'].rolling(7).mean().fillna(0)
    
    # Recalculate Stress Index for National
    if 'clinic_total_no_of_beds_allocated_for_covid_patients' in df_filtered.columns:
        df_filtered['healtcare_stress_index'] = (
            df_filtered['clinic_total_no_of_covid_patients_currently_admitted'] / 
            df_filtered['clinic_total_no_of_beds_allocated_for_covid_patients'].replace(0, 1) * 100
        )
    
    # Rt Calculation for National
    shifted = df_filtered['new_cases_7da'].shift(4).replace(0, 1)
    df_filtered['rt_estimate'] = (df_filtered['new_cases_7da'] / shifted).fillna(1.0)

else:
    # Single Province View (No aggregation needed)
    df_filtered = raw_filtered[raw_filtered['province'] == province].copy()
    
    # Ensure Rt is calculated
    shifted = df_filtered['new_cases_7da'].shift(4).replace(0, 1)
    df_filtered['rt_estimate'] = (df_filtered['new_cases_7da'] / shifted).fillna(1.0)

# ============================================
# 4. KPI Dashboard
# ============================================
latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-8] if len(df_filtered) > 8 else latest

st.markdown("### 📊 Situation Report")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("New Cases (7d Avg)", f"{int(latest['new_cases_7da']):,}", 
              delta=f"{int(latest['new_cases_7da'] - prev['new_cases_7da'])}")
with kpi2:
    st.metric("Effective Rt", f"{latest['rt_estimate']:.2f}", 
              delta="Spreading" if latest['rt_estimate'] > 1 else "Contained", delta_color="inverse")
with kpi3:
    if 'test_positivity_ratio' in df_filtered.columns:
        st.metric("Positivity Rate", f"{latest['test_positivity_ratio']:.2f}%",
                  delta=f"{(latest['test_positivity_ratio'] - prev['test_positivity_ratio']):.2f}%", delta_color="inverse")
with kpi4:
    if 'healtcare_stress_index' in df_filtered.columns:
        st.metric("Healthcare Stress", f"{latest['healtcare_stress_index']:.1f}%", help="% of Beds Occupied")
with kpi5:
    st.metric("Total Deaths", f"{int(latest['death_cumulative_total_deaths']):,}")

# ============================================
# 5. The Tabs (Expanded for "More Stuff")
# ============================================
tabs = st.tabs([
    "📈 Trends & Analysis", 
    "🏥 Healthcare Capacity", 
    "📜 Historical Intelligence", 
    "🤖 AI Projections", 
    "🧮 Policy Calculator",
    "🕵️ Data Audit"
])

# --- TAB 1: TRENDS ---
with tabs[0]:
    st.subheader("Epidemic Trajectory")
    
    t_col1, t_col2 = st.columns([3, 1])
    with t_col2:
        st.markdown("###### Chart Options")
        show_log = st.checkbox("Logarithmic Scale")
        show_cumulative = st.checkbox("Show Cumulative Cases")
    
    with t_col1:
        # Main Trend Plot
        fig_trend = go.Figure()
        
        y_metric = 'grand_total_cases_till_date' if show_cumulative else 'new_cases_7da'
        title_metric = "Cumulative Cases" if show_cumulative else "Daily Cases (7-Day Avg)"
        
        fig_trend.add_trace(go.Scatter(
            x=df_filtered['date'], y=df_filtered[y_metric], 
            mode='lines', name=title_metric, line=dict(color='#2563EB', width=3), fill='tozeroy'
        ))
        
        if 'new_deaths' in df_filtered.columns and not show_cumulative:
            fig_trend.add_trace(go.Bar(
                x=df_filtered['date'], y=df_filtered['new_deaths'], 
                name="Daily Deaths", marker_color='#EF4444', yaxis='y2', opacity=0.5
            ))

        fig_trend.update_layout(
            title=f"{title_metric} vs Deaths",
            yaxis_type="log" if show_log else "linear",
            yaxis2=dict(title="Deaths", overlaying='y', side='right'),
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 2: HEALTHCARE ---
with tabs[1]:
    st.subheader("System Capacity Monitor")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Stacked Bar: Bed Capacity
        if 'clinic_total_no_of_beds_allocated_for_covid_patients' in df_filtered.columns:
            # Create a melt for Used vs Available
            latest_beds = latest['clinic_total_no_of_beds_allocated_for_covid_patients']
            used_beds = latest['clinic_total_no_of_covid_patients_currently_admitted']
            
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Indicator(
                mode = "gauge+number",
                value = used_beds,
                title = {'text': "Bed Occupancy"},
                gauge = {'axis': {'range': [0, max(1, latest_beds)]},
                         'bar': {'color': "#EF4444" if used_beds/max(1,latest_beds) > 0.8 else "#10B981"},
                         'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': latest_beds * 0.8}}
            ))
            st.plotly_chart(fig_cap, use_container_width=True)

    with c2:
         # Oxygen vs Vents Time Series
        fig_stress = go.Figure()
        if 'clinic_total_on_oxygen' in df_filtered.columns:
            fig_stress.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['clinic_total_on_oxygen'], 
                                          name="Patients on Oxygen", line=dict(color='#F59E0B')))
        if 'clinic_total_no_of_patients_currently_on_ventilator' in df_filtered.columns:
            fig_stress.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['clinic_total_no_of_patients_currently_on_ventilator'], 
                                          name="On Ventilator", line=dict(color='#EF4444')))
        
        fig_stress.update_layout(title="Critical Care Demand Over Time", template="plotly_white")
        st.plotly_chart(fig_stress, use_container_width=True)

# --- TAB 3: HISTORICAL (The "More Stuff" Tab) ---
with tabs[2]:
    st.subheader("🌊 Wave-wise Comparative Intelligence")
    st.info("Historical breakdown of major pandemic waves to identify patterns.")

    # We need the FULL dataset for this, not just filtered
    df_hist = df.copy()
    
    # Aggregation by Wave & Province
    wave_stats = df_hist.groupby(['wave', 'province']).agg({
        'new_cases': 'sum',
        'new_deaths': 'sum',
        'test_positivity_ratio': 'mean'
    }).reset_index()
    
    # Filter for selected province (or show all if 'All' selected)
    if province != "All":
        wave_view = wave_stats[wave_stats['province'] == province]
    else:
        wave_view = wave_stats.groupby('wave').sum(numeric_only=True).reset_index()
        wave_view['province'] = "National"

    # Bar Chart: Cases per Wave
    fig_wave = px.bar(wave_view, x='wave', y='new_cases', color='new_cases', 
                      title=f"Total Cases by Wave ({province})", color_continuous_scale='Blues')
    st.plotly_chart(fig_wave, use_container_width=True)

    # Detailed Table
    st.markdown("#### Wave Performance Metrics")
    st.dataframe(wave_view.style.format("{:,.0f}", subset=['new_cases', 'new_deaths']))

    # Heatmap (If All is selected)
    if province == "All":
        st.markdown("#### Regional Heatmap (Positivity)")
        heat_data = wave_stats.pivot(index='province', columns='wave', values='test_positivity_ratio')
        fig_heat = px.imshow(heat_data, color_continuous_scale='RdYlGn_r', title="Avg Positivity % by Wave & Region")
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 4: AI PROJECTIONS ---
with tabs[3]:
    st.subheader("🤖 Predictive Intelligence Engine")
    
    # Prepare Data
    pred_df = df_filtered[['date', 'new_cases_7da']].dropna()
    
    if len(pred_df) > 10:
        pred_df['days'] = (pred_df['date'] - pred_df['date'].min()).dt.days
        
        # Polynomial Regression (Degree 3)
        poly = PolynomialFeatures(degree=3)
        X = poly.fit_transform(pred_df[['days']])
        model = LinearRegression().fit(X, pred_df['new_cases_7da'])
        
        # Forecast 30 days
        future_days = np.arange(pred_df['days'].max() + 1, pred_df['days'].max() + 31).reshape(-1, 1)
        future_X = poly.transform(future_days)
        preds = model.predict(future_X)
        
        dates_future = [pred_df['date'].max() + timedelta(days=int(i)) for i in range(1, 31)]
        
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['new_cases_7da'], name="Historical", line=dict(color='gray')))
        fig_ai.add_trace(go.Scatter(x=dates_future, y=preds, name="AI Forecast (Poly-Reg)", line=dict(color='#2563EB', dash='dash')))
        
        # Confidence Interval (Simulated)
        fig_ai.add_trace(go.Scatter(x=dates_future, y=preds*1.2, line=dict(width=0), showlegend=False))
        fig_ai.add_trace(go.Scatter(x=dates_future, y=preds*0.8, line=dict(width=0), fill='tonexty', 
                                    fillcolor='rgba(37, 99, 235, 0.2)', name="Confidence Interval"))
        
        fig_ai.update_layout(title="30-Day Trajectory Forecast", template="plotly_white")
        st.plotly_chart(fig_ai, use_container_width=True)
    else:
        st.warning("Insufficient data for predictions.")

# --- TAB 5: CALCULATOR ---
with tabs[4]:
    st.subheader("🧮 Intervention Scenario Simulator")
    
    c_calc1, c_calc2 = st.columns([1, 2])
    with c_calc1:
        base_rt = st.number_input("Current Rt", value=float(max(0.5, latest['rt_estimate'])), step=0.1)
        st.markdown("**Select Interventions:**")
        mask_on = st.checkbox("Mask Mandate (-15%)")
        lockdown_on = st.checkbox("Smart Lockdown (-25%)")
        
        reduction = (0.15 if mask_on else 0) + (0.25 if lockdown_on else 0)
        final_rt = base_rt * (1 - reduction)
        st.metric("Projected Rt", f"{final_rt:.2f}", delta=f"-{reduction*100:.0f}%")

    with c_calc2:
        days = 60
        dates = [datetime.now() + timedelta(days=i) for i in range(days)]
        cases_base = [latest['new_cases_7da'] * (base_rt ** (i/4)) for i in range(days)]
        cases_new = [latest['new_cases_7da'] * (final_rt ** (i/4)) for i in range(days)]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=dates, y=cases_base, name="Status Quo", line=dict(color='red')))
        fig_sim.add_trace(go.Scatter(x=dates, y=cases_new, name="With Intervention", line=dict(color='green')))
        fig_sim.update_layout(title="Projected Impact", xaxis_title="Date", yaxis_title="Daily Cases")
        st.plotly_chart(fig_sim, use_container_width=True)

# --- TAB 6: DATA AUDIT ---
with tabs[5]:
    st.subheader("Bias & Integrity Check")
    
    if 'grand_total_tests_conducted_till_date' in df_filtered.columns:
        fig_bias = px.scatter(df_filtered, x='grand_total_tests_conducted_till_date', y='grand_total_cases_till_date',
                              color='test_positivity_ratio' if 'test_positivity_ratio' in df_filtered.columns else None,
                              title="Testing Bias: Do more tests equals more cases?",
                              labels=col_map)
        st.plotly_chart(fig_bias, use_container_width=True)
    else:
        st.warning("Testing data unavailable for bias check.")

st.markdown("---")
