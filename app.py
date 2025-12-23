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
# 1. Configuration & Executive Theme
# ============================================
st.set_page_config(
    page_title="CPIF | Pakistan Pandemic Intelligence",
    layout="wide",
    page_icon="🇵🇰",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Executive Dashboard" Look
st.markdown("""
    <style>
    .metric-card { background-color: rgba(255, 255, 255, 0.05); border-left: 5px solid #3b82f6; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; color: #888; }
    .stTabs [aria-selected="true"] { background-color: rgba(59, 130, 246, 0.1); border-bottom: 2px solid #3b82f6; color: #3b82f6; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
    </style>
""", unsafe_allow_html=True)

st.title("🇵🇰 CPIF: Comparative Pandemic Intelligence Framework")
st.markdown("**Pendemic Decision Support System** | Historical Analysis • AI Projections • Scenario Planning")

# ============================================
# 2. Data Loading (Robust & Formula Enriched)
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Refined + New entities.csv")
    except:
        return pd.DataFrame()

    # 1. Robust Column Cleaning (Regex)
    df.columns = (df.columns.astype(str)
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_'))

    # 2. Key Numeric Columns to Clean
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
                       .replace(['N/A', 'n/a', '-', '', 'nan', 'Nan', 'Not Reported'], '0'))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Dates & Sorting
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df["province"] = df["province"].astype(str).str.strip().str.title()
        df = df.sort_values(by=["province", "date"]).reset_index(drop=True)

    # 4. Feature Engineering (The "Correct" Formulas)
    
    # Active Cases
    df["active_cases"] = (df["grand_total_cases_till_date"] -
                          df["clinic_total_numbers_recovered_and_discharged_so_far"] -
                          df["death_cumulative_total_deaths"]).clip(lower=0)

    # Daily Diffs
    df["new_cases"] = df.groupby("province")["grand_total_cases_till_date"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby("province")["death_cumulative_total_deaths"].diff().fillna(0).clip(lower=0)
    
    # 7-Day Rolling (Smoothing)
    df["new_cases_7da"] = df.groupby("province")["new_cases"].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    
    # Rates & Ratios
    df["recovery_rate"] = (df["clinic_total_numbers_recovered_and_discharged_so_far"] / df["grand_total_cases_till_date"] * 100).fillna(0)
    df["fatality_rate"] = (df["death_cumulative_total_deaths"] / df["grand_total_cases_till_date"] * 100).fillna(0)
    
    # Intelligence Metrics (Rt Proxy)
    shifted_cases = df.groupby("province")["new_cases_7da"].shift(4).replace(0, 1)
    df["rt_estimate"] = (df["new_cases_7da"] / shifted_cases).fillna(1.0).clip(upper=4.0)

    # Wave Definition (Pakistan Context)
    df["wave"] = "Inter-Wave Period"
    df.loc[df["date"].between("2020-03-01", "2020-07-31"), "wave"] = "Wave 1 (Original)"
    df.loc[df["date"].between("2020-10-01", "2021-01-31"), "wave"] = "Wave 2 (Winter 2020)"
    df.loc[df["date"].between("2021-03-01", "2021-05-31"), "wave"] = "Wave 3 (Alpha/Beta)"
    df.loc[df["date"].between("2021-07-01", "2021-09-30"), "wave"] = "Wave 4 (Delta)"
    df.loc[df["date"].between("2021-12-01", "2022-03-31"), "wave"] = "Wave 5 (Omicron)"

    return df

df = load_data()

if df.empty:
    st.error("Data file not found or empty. Please ensure 'Refined + New entities.csv' is in the folder.")
    st.stop()

# ============================================
# 3. Sidebar Controls
# ============================================
st.sidebar.header("🔍 Executive Controls")

provinces = ["All (National)"] + sorted(df["province"].unique().tolist())
province = st.sidebar.selectbox("Select Province", provinces, index=0)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Analysis Window",
    [max_date - timedelta(days=90), max_date],
    min_value=min_date,
    max_value=max_date
)

# Filtering Logic
if len(date_range) == 2:
    mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    df_filtered = df[mask].copy()
else:
    df_filtered = df.copy()

if province != "All (National)":
    df_filtered = df_filtered[df_filtered["province"] == province]
else:
    # Aggregation for National View
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns
    df_filtered = df_filtered.groupby("date")[numeric_cols].sum().reset_index()
    # Recalculate Rt for National
    df_filtered["new_cases_7da"] = df_filtered["new_cases"].rolling(7).mean().fillna(0)
    df_filtered["rt_estimate"] = (df_filtered["new_cases_7da"] / df_filtered["new_cases_7da"].shift(4).replace(0,1)).fillna(1.0)
    # Average the rates instead of sum
    df_filtered["test_positivity_ratio"] = df[mask].groupby("date")["test_positivity_ratio"].mean().values

# ============================================
# 4. KPI Section (Refined)
# ============================================
latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-8] if len(df_filtered) > 8 else latest

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active Cases", f"{int(latest['active_cases']):,}", delta=f"{int(latest['active_cases'] - prev['active_cases'])}")
col2.metric("Rt (Spread Velocity)", f"{latest['rt_estimate']:.2f}", delta="Exp. Growth" if latest['rt_estimate']>1 else "Contained", delta_color="inverse")
col3.metric("Positivity Rate", f"{latest.get('test_positivity_ratio',0):.2f}%", delta=f"{(latest.get('test_positivity_ratio',0)-prev.get('test_positivity_ratio',0)):.2f}%", delta_color="inverse")
col4.metric("Healthcare Stress", f"{latest.get('healtcare_stress_index',0):.1f}", help="Composite index of system load")
col5.metric("Total Deaths", f"{int(latest['death_cumulative_total_deaths']):,}", delta=f"{int(latest['new_deaths'])}")

# ============================================
# 5. Dashboard Tabs
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Epidemic Curves",
    "🏥 Healthcare & Capacity",
    "📊 Comparisons & Trends",
    "🤖 AI Model Comparison",
    "🌊 Wave Analysis"
])

# --- TAB 1: Epidemic Curves ---
with tab1:
    st.subheader("Epidemic Trajectory Analysis")
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Daily New Cases (7-Day Trend)", "Cumulative Cases", "Daily Deaths", "Positivity Rate"))

    # Plot 1: Cases
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases_7da"], mode='lines', name='7-Day Avg', line=dict(color='#3b82f6', width=3)), row=1, col=1)
    fig1.add_trace(go.Bar(x=df_filtered["date"], y=df_filtered["new_cases"], name='Daily Cases', marker=dict(color='rgba(59, 130, 246, 0.3)')), row=1, col=1)
    
    # Plot 2: Cumulative
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["grand_total_cases_till_date"], mode='lines', name='Total Cases', line=dict(color='#8b5cf6'), fill='tozeroy'), row=1, col=2)
    
    # Plot 3: Deaths
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_deaths"], mode='lines', name='Daily Deaths', line=dict(color='#ef4444')), row=2, col=1)
    
    # Plot 4: Positivity
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["test_positivity_ratio"], mode='lines', name='Positivity %', line=dict(color='#f59e0b', dash='dot')), row=2, col=2)

    fig1.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: Healthcare (Refined Formulas) ---
with tab2:
    st.subheader("Healthcare Capacity Intelligence")
    
    # Metrics Calculation
    beds_total = latest.get("clinic_total_no_of_beds_allocated_for_covid_patients", 1)
    beds_used = latest.get("clinic_total_no_of_covid_patients_currently_admitted", 0)
    vents_total = latest.get("clinic_total_no_of_ventilators_allocated_for_covid_patients", 1)
    vents_used = latest.get("clinic_total_no_of_patients_currently_on_ventilator", 0)
    
    # Avoid Div/0
    beds_total = 1 if beds_total == 0 else beds_total
    vents_total = 1 if vents_total == 0 else vents_total

    c1, c2 = st.columns(2)
    with c1:
        # Gauge Chart for Beds
        fig_bed = go.Figure(go.Indicator(
            mode = "gauge+number", value = (beds_used/beds_total)*100, title = {'text': "Bed Occupancy (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}, 'steps': [{'range': [0, 70], 'color': "rgba(200,200,200,0.3)"}, {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.5)"}]}
        ))
        fig_bed.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bed, use_container_width=True)

    with c2:
        # Oxygen Dependency Trend (The "Refined" Metric)
        if 'oxygen_dependency_ratio' in df_filtered.columns:
            st.metric("Oxygen Dependency Ratio", f"{latest['oxygen_dependency_ratio']:.1f}%", help="% of Admitted Patients needing Oxygen")
            fig_oxy = px.area(df_filtered, x='date', y='oxygen_dependency_ratio', title="Oxygen Dependency Trend", color_discrete_sequence=['#10B981'])
            fig_oxy.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_oxy, use_container_width=True)

# --- TAB 3: Comparisons ---
with tab3:
    st.subheader("Comparative Analysis")
    
    if province == "All (National)":
        # Aggregate view for provinces
        latest_prov = df.groupby("province").last().reset_index()
        fig_comp = px.bar(latest_prov, x="province", y="test_positivity_ratio", color="test_positivity_ratio", 
                          title="Current Positivity Rate by Province", color_continuous_scale="Reds")
        fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_comp, use_container_width=True)
    
    st.subheader("Multi-Metric Trends")
    trend_cols = ["new_cases_7da", "test_positivity_ratio", "active_cases"]
    fig_trends = px.line(df_filtered, x="date", y=trend_cols, title="Correlation: Cases vs Positivity vs Active")
    fig_trends.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
    st.plotly_chart(fig_trends, use_container_width=True)

# --- TAB 4: AI Projections (The Battle of Models) ---
with tab4:
    st.subheader("🤖 Predictive Modeling: Comparison")
    st.markdown("Comparing **Linear Regression** (Baseline), **Random Forest** (Non-Linear), and **Polynomial Regression** (Trend Aware).")

    if len(df_filtered) > 30:
        # Prepare Data
        pred_data = df_filtered[["date", "new_cases_7da"]].dropna()
        pred_data["days"] = (pred_data["date"] - pred_data["date"].min()).dt.days
        
        X = pred_data[["days"]]
        y = pred_data["new_cases_7da"]

        # 1. Linear Model
        lin_model = LinearRegression().fit(X, y)
        
        # 2. Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        
        # 3. Polynomial Model (Degree 3)
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression().fit(X_poly, y)

        # Future Frames
        future_days_count = 30
        last_day = pred_data["days"].max()
        future_X = np.arange(last_day + 1, last_day + future_days_count + 1).reshape(-1, 1)
        future_dates = [pred_data["date"].max() + timedelta(days=i) for i in range(1, future_days_count + 1)]

        # Predictions (With Clipping to prevent negatives)
        pred_lin = lin_model.predict(future_X).clip(min=0)
        pred_rf = rf_model.predict(future_X).clip(min=0) # RF fails to extrapolate usually
        pred_poly = poly_model.predict(poly.transform(future_X)).clip(min=0)

        # Visualization
        fig_ai = go.Figure()
        
        # History
        fig_ai.add_trace(go.Scatter(x=pred_data["date"], y=y, name='Historical Data', line=dict(color='gray', width=1)))
        
        # Models
        fig_ai.add_trace(go.Scatter(x=future_dates, y=pred_lin, name='Linear Regression', line=dict(color='blue', dash='dot')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=pred_rf, name='Random Forest', line=dict(color='green', dash='dot')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=pred_poly, name='Polynomial (Deg 3)', line=dict(color='orange', width=3)))

        fig_ai.update_layout(title=f"30-Day Forecast Comparison ({province})", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ai, use_container_width=True)
        
        st.info("""
        **Insights:**
        * **Linear Regression:** Assumes constant growth/decline. Often oversimplifies.
        * **Random Forest:** Good for history, but cannot predict "new highs" (cannot extrapolate). Note how it flatlines.
        * **Polynomial (Deg 3):** Best for capturing the 'bending' of a wave.
        """)
        
    else:
        st.warning("Not enough data points for robust modeling.")

# --- TAB 5: Wave Analysis (Group's Original Logic Refined) ---
with tab5:
    st.header("🌊 Historical Wave Analysis")
    st.markdown("Aggregated performance across major pandemic waves in Pakistan.")

    # Group by Wave and Province
    # We use the full dataset 'df' to show comparison even if a specific province is selected in sidebar, 
    # but let's respect the sidebar for consistency if desired. Here we show ALL for comparison context.
    
    wave_stats = df.groupby(["wave", "province"]).agg({
        "new_cases": "sum",
        "new_deaths": "sum",
        "grand_total_cases_till_date": "max",
        "test_positivity_ratio": "mean"
    }).reset_index()

    # Filter out "Inter-Wave" if it's too noisy
    wave_stats = wave_stats[wave_stats["wave"] != "Inter-Wave Period"]

    # Visuals
    c_w1, c_w2 = st.columns(2)
    with c_w1:
        fig_peak = px.bar(wave_stats, x="wave", y="new_cases", color="province", title="Total Cases by Wave", barmode="group")
        fig_peak.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_peak, use_container_width=True)
        
    with c_w2:
        fig_pos = px.line(wave_stats, x="wave", y="test_positivity_ratio", color="province", title="Average Positivity by Wave", markers=True)
        fig_pos.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pos, use_container_width=True)

# ============================================
# 6. Risk Assessment
# ============================================
st.markdown("---")
st.subheader("🚦 Real-Time Risk Assessment")

risk_score = 0
alerts = []
pos_val = latest.get("test_positivity_ratio", 0)
bed_val = (beds_used/beds_total)*100

if pos_val > 10: 
    risk_score += 30; alerts.append("🔴 CRITICAL: Positivity > 10%")
elif pos_val > 5: 
    risk_score += 15; alerts.append("🟡 WARNING: Positivity > 5%")
    
if bed_val > 80: 
    risk_score += 25; alerts.append("🔴 CRITICAL: Hospital Beds > 80% Full")
elif bed_val > 60: 
    risk_score += 10; alerts.append("🟡 WARNING: Hospital Beds > 60% Full")
    
if latest.get("rt_estimate", 0) > 1.2:
    risk_score += 20; alerts.append("🔴 CRITICAL: Rapid Spread (Rt > 1.2)")

risk_level = "🟢 LOW" if risk_score < 30 else "🟡 MODERATE" if risk_score < 60 else "🔴 HIGH"

c_risk1, c_risk2 = st.columns([1, 3])
with c_risk1:
    st.metric("Risk Level", risk_level, f"{risk_score}/100 Risk Score")
with c_risk2:
    if alerts:
        for a in alerts: st.warning(a)
    else:
        st.success("No critical alerts active.")

st.caption(f"Framework v4.5 | Last Updated: {datetime.now().strftime('%Y-%m-%d')}")
