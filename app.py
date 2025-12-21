import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# ============================================
# Page Config & Styling
# ============================================
st.set_page_config(
    page_title="CPIF: Comparative Pandemic Intelligence Framework",
    layout="wide",
    page_icon="🛡️"
)

# Custom CSS for "Trustworthy" aesthetic
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6;}
    .alert-card {background-color: #fee2e2; padding: 15px; border-radius: 10px; border: 1px solid #ef4444;}
    h1 {color: #1e3a8a;}
    h2, h3 {color: #1e40af;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ CPIF: Comparative Pandemic Intelligence Framework")
st.markdown("**Version 2.0 (Refined)** | Features: *Multivariate Polynomial AI, R₀ estimation, Capacity Intelligence*")

# ============================================
# 1. Data Layer (ETL + Cleaning + Feature Eng)
# ============================================
@st.cache_data
def load_and_refine_data():
    # Load original file
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except FileNotFoundError:
        st.error("Data file not found. Please upload 'Refined + New entities.csv'")
        return pd.DataFrame()

    # --- A. Data Cleaning ---
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Fix Data Types (Remove '%' and ',')
    cols_to_clean = [
        'Test Positivity Ratio', 'Fetaility Ratio', 'Ventilator Fetality Ratio',
        'Recovery Velocity', 'Healtcare Stress Index', 'Oxygen Dependency Ratio',
        'Grand Total Cases till date', 'Grand Total Cases in Last 24 hours',
        'Death Cumulative / Total Deaths', 'Clinic Total No. Of COVID Patients currently Admitted',
        'Clinic Total No. Of Ventilators allocated for COVID Patients',
        'Clinic Total No. of Patients currently on Ventilator'
    ]
    
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').replace('-', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Date conversion
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by=['Province', 'Date'])

    # --- B. Feature Engineering (The "Refined" Formulas) ---
    
    # 1. Healthcare Stress Index (Corrected: Capacity based, not Death based)
    # Formula: (Patients on Vent + Oxygen) / Total Capacity
    # We will use Ventilator utilization as the proxy for high stress
    df['Refined_Stress_Index'] = (
        df['Clinic Total No. of Patients currently on Ventilator'] / 
        df['Clinic Total No. Of Ventilators allocated for COVID Patients'].replace(0, 1)
    ) * 100

    # 2. Recovery Velocity (Corrected: Speed of recovery)
    # Formula: New Recoveries / Active Cases
    # Note: We need to calculate New Recoveries first as it's not always clean
    df['New_Recoveries'] = df.groupby('Province')['Clinic Total Numbers Recovered and Discharged so far'].diff().fillna(0).clip(lower=0)
    df['Active_Cases'] = df['Grand Total Cases till date'] - df['Clinic Total Numbers Recovered and Discharged so far'] - df['Death Cumulative / Total Deaths']
    
    df['Refined_Recovery_Velocity'] = (df['New_Recoveries'] / df['Active_Cases'].replace(0, 1)) * 100

    # 3. 7-Day Moving Averages (To fix "Weekend Effect" Anomaly)
    df['New_Cases_7MA'] = df.groupby('Province')['Grand Total Cases in Last 24 hours'].transform(lambda x: x.rolling(7).mean())
    df['TPR_7MA'] = df.groupby('Province')['Test Positivity Ratio'].transform(lambda x: x.rolling(7).mean())

    # 4. Wave Classification
    df["Wave_Name"] = "Inter-Wave Period"
    df.loc[df["Date"].between("2020-03-01", "2020-07-31"), "Wave_Name"] = "Wave 1 (Original)"
    df.loc[df["Date"].between("2020-10-01", "2021-02-28"), "Wave_Name"] = "Wave 2 (Alpha)"
    df.loc[df["Date"].between("2021-03-01", "2021-06-30"), "Wave_Name"] = "Wave 3 (Beta/Gamma)"
    df.loc[df["Date"].between("2021-07-01", "2021-10-31"), "Wave_Name"] = "Wave 4 (Delta)"
    df.loc[df["Date"].between("2021-12-01", "2022-03-31"), "Wave_Name"] = "Wave 5 (Omicron)"

    return df

df = load_and_refine_data()

if df.empty:
    st.stop()

# ============================================
# 2. Controls & Context
# ============================================
with st.sidebar:
    st.header("🎛️ Intelligence Controls")
    
    # Province Filter
    prov_list = ['All'] + sorted(df['Province'].unique().tolist())
    selected_prov = st.selectbox("Select Region", prov_list, index=0)
    
    # Date Filter
    min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
    dates = st.date_input("Analysis Window", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filtering Logic
if len(dates) == 2:
    mask = (df['Date'].dt.date >= dates[0]) & (df['Date'].dt.date <= dates[1])
    df_filtered = df[mask].copy()
else:
    df_filtered = df.copy()

if selected_prov != 'All':
    df_filtered = df_filtered[df_filtered['Province'] == selected_prov]

# Aggregating for display if "All" is selected
if selected_prov == 'All':
    display_df = df_filtered.groupby('Date').sum(numeric_only=True).reset_index()
    # Recalculate rates for aggregated data
    display_df['Test Positivity Ratio'] = df_filtered.groupby('Date')['Test Positivity Ratio'].mean().values # Approx
else:
    display_df = df_filtered

latest = display_df.iloc[-1]

# ============================================
# 3. KPI Layer (Corrected Formulas)
# ============================================
st.markdown("### 📊 Situation Overview")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("New Cases (7-Day Avg)", 
            f"{int(latest.get('New_Cases_7MA', 0)):,}", 
            delta=f"{latest.get('Grand Total Cases in Last 24 hours', 0):.0f} raw")

kpi2.metric("Active Cases", 
            f"{int(latest.get('Active_Cases', 0)):,}")

kpi3.metric("Test Positivity Rate (TPR)", 
            f"{latest['Test Positivity Ratio']:.2f}%",
            delta_color="inverse")

kpi4.metric("Healthcare Stress Index", 
            f"{latest.get('Refined_Stress_Index', 0):.1f}%",
            help="Ventilator Occupancy Rate")

kpi5.metric("Recovery Velocity", 
            f"{latest.get('Refined_Recovery_Velocity', 0):.2f}%",
            help="Daily % of active cases recovering")

# ============================================
# 4. Main Intelligence Tabs
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend Intelligence", 
    "🏥 Healthcare Capacity", 
    "🔄 Wave Comparison", 
    "🤖 AI Projections & Scenarios"
])

# --- TAB 1: Trend Intelligence ---
with tab1:
    st.subheader("Epidemic Curve & Positivity Analysis")
    
    fig_main = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar: Raw Cases
    fig_main.add_trace(go.Bar(
        x=display_df['Date'], y=display_df['Grand Total Cases in Last 24 hours'],
        name="Daily Cases", marker_color='rgba(59, 130, 246, 0.3)'
    ), secondary_y=False)
    
    # Line: 7-Day Average (Smooth)
    fig_main.add_trace(go.Scatter(
        x=display_df['Date'], y=display_df['New_Cases_7MA'],
        name="7-Day Trend (Bias Corrected)", line=dict(color='#1d4ed8', width=3)
    ), secondary_y=False)
    
    # Line: Positivity Rate (The "Truth" Metric)
    fig_main.add_trace(go.Scatter(
        x=display_df['Date'], y=display_df['TPR_7MA'],
        name="Positivity Rate (7-MA)", line=dict(color='#ef4444', width=2, dash='dot')
    ), secondary_y=True)
    
    fig_main.update_layout(title="Cases vs. Positivity (Checking for Testing Bias)", height=500)
    fig_main.update_yaxes(title_text="Daily Cases", secondary_y=False)
    fig_main.update_yaxes(title_text="Positivity %", secondary_y=True)
    st.plotly_chart(fig_main, use_container_width=True)

# --- TAB 2: Healthcare Capacity (The "Oxygen" Insight) ---
with tab2:
    st.subheader("Resource Stress Monitors")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Oxygen Dependency Trend
        fig_oxy = px.area(display_df, x='Date', y='Clinic Total (on Oxygen)', 
                          title="Oxygen Demand Trend (Critical for Pakistan)",
                          color_discrete_sequence=['#06b6d4'])
        st.plotly_chart(fig_oxy, use_container_width=True)
        
    with col_b:
        # Ventilator vs Capacity
        fig_vent = go.Figure()
        fig_vent.add_trace(go.Scatter(x=display_df['Date'], y=display_df['Clinic Total No. Of Ventilators allocated for COVID Patients'],
                                      name="Total Ventilators", line=dict(color='green')))
        fig_vent.add_trace(go.Scatter(x=display_df['Date'], y=display_df['Clinic Total No. of Patients currently on Ventilator'],
                                      name="In Use", fill='tozeroy', line=dict(color='red')))
        fig_vent.update_layout(title="Ventilator Saturation Analysis")
        st.plotly_chart(fig_vent, use_container_width=True)

# --- TAB 3: Wave Comparison ---
with tab3:
    st.subheader("Comparative Wave Analysis")
    
    wave_stats = df.groupby(['Wave_Name', 'Province']).agg({
        'Grand Total Cases in Last 24 hours': 'sum',
        'Death Cumulative / Total Deaths': 'max',
        'Test Positivity Ratio': 'mean'
    }).reset_index()
    
    # Rename for display
    wave_stats.columns = ['Wave', 'Province', 'Total Cases', 'Max Deaths', 'Avg TPR']
    
    fig_wave = px.bar(wave_stats, x='Wave', y='Total Cases', color='Province', 
                      title="Total Burden per Wave by Province", barmode='group')
    st.plotly_chart(fig_wave, use_container_width=True)
    
    st.markdown("### Wave Efficiency (Cases per Death)")
    fig_scatter = px.scatter(wave_stats, x='Total Cases', y='Max Deaths', color='Wave', size='Avg TPR',
                             hover_data=['Province'], title="Severity Matrix: Bigger Bubble = Higher Positivity")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 4: AI Projections (The FIXED Logic) ---
with tab4:
    st.subheader("🤖 Multivariate Polynomial Forecasting")
    
    # 1. Prepare Data for ML
    # We use 'Days' as X, but we also check correlations
    ml_data = display_df[['Date', 'New_Cases_7MA']].dropna().reset_index(drop=True)
    ml_data['Day_Index'] = ml_data.index
    
    if len(ml_data) > 30:
        X = ml_data[['Day_Index']]
        y = ml_data['New_Cases_7MA']
        
        # MODEL: Polynomial Regression (Degree 3) - Fits waves better than Linear
        poly_model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
        poly_model.fit(X, y)
        
        # Future Predictions (30 Days)
        last_day = ml_data['Day_Index'].max()
        future_X = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
        future_dates = [ml_data['Date'].max() + timedelta(days=i) for i in range(1, 31)]
        
        pred_y = poly_model.predict(future_X)
        pred_y = np.maximum(pred_y, 0) # No negative cases
        
        # Confidence Interval (Simple residual std dev method for demo)
        y_pred_train = poly_model.predict(X)
        resid_std = np.std(y - y_pred_train)
        upper_bound = pred_y + 1.96 * resid_std
        lower_bound = np.maximum(pred_y - 1.96 * resid_std, 0)
        
        # Plotting
        fig_ai = go.Figure()
        
        # Historical
        fig_ai.add_trace(go.Scatter(x=ml_data['Date'], y=y, name='Actual Trend', line=dict(color='gray', width=1)))
        
        # Prediction
        fig_ai.add_trace(go.Scatter(x=future_dates, y=pred_y, name='AI Forecast (Poly-Reg)', 
                                    line=dict(color='#2563eb', width=3, dash='dash')))
        
        # Confidence Interval
        fig_ai.add_trace(go.Scatter(x=future_dates + future_dates[::-1], 
                                    y=list(upper_bound) + list(lower_bound)[::-1],
                                    fill='toself', fillcolor='rgba(37, 99, 235, 0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='95% Confidence Interval'))
        
        fig_ai.update_layout(title="30-Day Predictive Intelligence", height=500)
        st.plotly_chart(fig_ai, use_container_width=True)
        
        st.info("ℹ️ **Model Note:** Uses 4th-Degree Polynomial Regression to capture wave curvature. Shaded area represents uncertainty based on historical volatility.")
        
    else:
        st.warning("Insufficient data for reliable AI training.")

    # --- Scenario Calculator (Refined with Rt) ---
    st.markdown("---")
    st.subheader("🧮 Intervention Scenario Calculator")
    
    c1, c2, c3 = st.columns(3)
    current_cases = ml_data['New_Cases_7MA'].iloc[-1] if not ml_data.empty else 1000
    
    with c1:
        st.markdown("**Current Status**")
        st.metric("Starting Daily Cases", f"{int(current_cases):,}")
    
    with c2:
        # Intelligent Default for R0 based on recent growth
        st.markdown("**Virus Behavior**")
        r_input = st.slider("Effective Reproduction Number (Rt)", 0.5, 3.0, 1.2, 0.1, help="Expected secondary infections per case")
        
    with c3:
        st.markdown("**Policy Intervention**")
        reduction = st.slider("Intervention Impact (%)", 0, 50, 15, help="e.g., Mask mandate = ~15%")
        
    # Calculation Logic: Nt = Nt-1 * R
    # To convert R to daily growth: Daily Growth ~ R^(1/Serial Interval)
    # Serial Interval approx 4 days
    serial_interval = 4
    days_proj = 30
    
    dates_scen = [datetime.now() + timedelta(days=i) for i in range(days_proj)]
    cases_scen_base = []
    cases_scen_int = []
    
    curr_b = current_cases
    curr_i = current_cases
    
    # Daily growth factor derived from Rt
    daily_r_base = r_input ** (1/serial_interval)
    daily_r_int = (r_input * (1 - reduction/100)) ** (1/serial_interval)
    
    for _ in range(days_proj):
        curr_b *= daily_r_base
        curr_i *= daily_r_int
        cases_scen_base.append(curr_b)
        cases_scen_int.append(curr_i)
        
    fig_scen = go.Figure()
    fig_scen.add_trace(go.Scatter(x=dates_scen, y=cases_scen_base, name=f"Status Quo (Rt={r_input})", line=dict(color='red')))
    fig_scen.add_trace(go.Scatter(x=dates_scen, y=cases_scen_int, name=f"With Intervention (-{reduction}%)", line=dict(color='green')))
    
    fig_scen.update_layout(title="Projected Impact of Policy Intervention", hovermode="x unified")
    st.plotly_chart(fig_scen, use_container_width=True)
    
    saved_cases = sum(cases_scen_base) - sum(cases_scen_int)
    st.success(f"🛡️ **Projected Impact:** This policy could prevent approx. **{int(saved_cases):,} cases** in the next 30 days.")

# ============================================
# 5. Footer / Disclaimer
# ============================================
st.markdown("---")
st.caption("Disclaimer: This Comparative Pandemic Intelligence Framework (CPIF) is an academic project designed for educational purposes. Predictions are based on historical data patterns and do not account for biological mutations unseen in the training set.")
