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
# 1. Configuration & Design System
# ============================================
st.set_page_config(
    page_title="CPIF | Pandemic Intelligence",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- THEME COLORS ---
COLORS = {
    'primary': '#1E3A8A',      # Navy Blue (Official)
    'accent': '#3B82F6',       # Bright Blue
    'success': '#10B981',      # Emerald Green
    'warning': '#F59E0B',      # Amber
    'danger': '#EF4444',       # Red
    'background': '#F8FAFC',   # Light Gray
    'card': '#FFFFFF'          # White
}

# --- EXECUTIVE CSS ---
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-color: {COLORS['background']};
    }}
    
    /* Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: {COLORS['card']};
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid {COLORS['accent']};
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: #E2E8F0;
        border-radius: 5px 5px 0px 0px;
        font-weight: 600;
        padding: 0 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['card']};
        border-top: 3px solid {COLORS['primary']};
        color: {COLORS['primary']};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        font-family: 'Helvetica Neue', sans-serif;
        color: #0F172A;
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================
# 2. Data Pipeline
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
        'clinic_total_numbers_recovered_and_discharged_so_far': 'Total Recovered'
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

    # 5. Feature Engineering
    df['new_cases'] = df.groupby('province')['grand_total_cases_till_date'].diff().fillna(0).clip(lower=0)
    df['new_deaths'] = df.groupby('province')['death_cumulative_total_deaths'].diff().fillna(0).clip(lower=0)
    
    # Smoothers
    df['new_cases_7da'] = df.groupby('province')['new_cases'].rolling(7).mean().reset_index(0, drop=True).fillna(0)
    
    # Rt Proxy
    shifted_cases = df.groupby('province')['new_cases_7da'].shift(4).replace(0, 1)
    df['growth_factor'] = df['new_cases_7da'] / shifted_cases
    df['rt_estimate'] = df['growth_factor'].pow(1).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Cap Oxygen ratio
    if 'oxygen_dependency_ratio' in df.columns:
        df.loc[df['oxygen_dependency_ratio'] > 100, 'oxygen_dependency_ratio'] = 100
        
    return df, display_map

# Load Data
df_raw, col_map = load_and_clean_data()

if df_raw.empty:
    st.error("🚨 Data File Missing. Please upload 'Refined + New entities.csv'")
    st.stop()

# ============================================
# 3. Sidebar & Filtering
# ============================================
with st.sidebar:
    st.title("🛡️ CPIF Controls")
    st.markdown("---")
    
    # Region Selector
    provinces_list = ["All (National Aggregate)"] + sorted(df_raw['province'].unique().tolist())
    selected_prov = st.selectbox("Select Region", provinces_list, index=0)
    
    # Date Selector
    df_temp = df_raw.copy() # Temp for date finding
    min_d, max_d = df_temp['date'].min(), df_temp['date'].max()
    dates = st.date_input("Analysis Window", [max_d - timedelta(days=90), max_d], min_value=min_d, max_value=max_d)
    
    st.markdown("---")
    st.caption(f"📅 Data range: {min_d.date()} to {max_d.date()}")
    st.caption("Developed for MS Healthcare Intelligence")

# Apply Filters
if selected_prov == "All (National Aggregate)":
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    df_filtered = df_raw.groupby('date')[numeric_cols].sum().reset_index()
    
    # Recalc Rolling & Rt for Aggregate
    df_filtered['new_cases_7da'] = df_filtered['new_cases'].rolling(7).mean().fillna(0)
    shifted = df_filtered['new_cases_7da'].shift(4).replace(0, 1)
    df_filtered['rt_estimate'] = (df_filtered['new_cases_7da'] / shifted).fillna(1.0)
    # Average Positivity for national (Approximation)
    df_filtered['test_positivity_ratio'] = df_raw.groupby('date')['test_positivity_ratio'].mean().values
else:
    df_filtered = df_raw[df_raw['province'] == selected_prov].copy()

# Date Filter
if len(dates) == 2:
    start_d, end_d = dates
    df_filtered = df_filtered[(df_filtered['date'].dt.date >= start_d) & (df_filtered['date'].dt.date <= end_d)]

if df_filtered.empty:
    st.warning("⚠️ No data in selected range.")
    st.stop()

# ============================================
# 4. Header & KPI Cards
# ============================================
st.markdown(f"## 🇵🇰 Situation Report: {selected_prov}")
st.markdown(f"**Status:** {end_d.strftime('%B %d, %Y')}")

latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-8] if len(df_filtered) > 8 else latest
rt_now = df_filtered['rt_estimate'].mean()

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("New Cases (7d Avg)", f"{int(latest['new_cases_7da']):,}", delta=f"{int(latest['new_cases_7da'] - prev['new_cases_7da'])}")
with c2: st.metric("Rt (Spread)", f"{rt_now:.2f}", delta="Exp. Growth" if rt_now > 1.1 else "Contained", delta_color="inverse")
with c3: st.metric("Positivity Rate", f"{latest.get('test_positivity_ratio', 0):.1f}%", delta=f"{(latest.get('test_positivity_ratio',0) - prev.get('test_positivity_ratio',0)):.1f}%", delta_color="inverse")
with c4: st.metric("Active Admissions", f"{int(latest.get('clinic_total_no_of_covid_patients_currently_admitted',0)):,}")
with c5: st.metric("Total Deaths", f"{int(latest.get('death_cumulative_total_deaths',0)):,}", delta=f"{int(latest['new_deaths'])}", delta_color="inverse")

# ============================================
# 5. Main Analysis Tabs
# ============================================
t1, t2, t3, t4, t5 = st.tabs([
    "📈 Intelligence Layer", 
    "🏥 Capacity Insights", 
    "🔮 AI Projections", 
    "🕵️ Data Audit", 
    "♟️ Policy Sim"
])

# --- TAB 1: Intelligence ---
with t1:
    st.subheader("Epidemic Wave Trajectory")
    
    col_opt1, col_opt2 = st.columns([1, 4])
    with col_opt1:
        st.markdown("**Visualization Settings**")
        log_scale = st.checkbox("Logarithmic Scale", help="Useful for detecting early exponential growth phases.")
        show_pos = st.checkbox("Overlay Positivity %", value=True)
    
    with col_opt2:
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bars for Daily Cases
        fig_trend.add_trace(go.Bar(
            x=df_filtered['date'], y=df_filtered['new_cases'], 
            name="Daily Cases", marker_color='rgba(59, 130, 246, 0.3)'
        ), secondary_y=False)
        
        # Line for Trend
        fig_trend.add_trace(go.Scatter(
            x=df_filtered['date'], y=df_filtered['new_cases_7da'], 
            name="7-Day Trend", line=dict(color=COLORS['primary'], width=3)
        ), secondary_y=False)
        
        # Line for Positivity
        if show_pos and 'test_positivity_ratio' in df_filtered.columns:
            fig_trend.add_trace(go.Scatter(
                x=df_filtered['date'], y=df_filtered['test_positivity_ratio'], 
                name="Positivity %", line=dict(color=COLORS['danger'], dash='dot', width=2)
            ), secondary_y=True)

        fig_trend.update_layout(
            height=450, 
            template="plotly_white", 
            hovermode="x unified",
            margin=dict(t=10),
            legend=dict(orientation="h", y=1.1, x=0)
        )
        if log_scale: fig_trend.update_yaxes(type="log", secondary_y=False)
        st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 2: Capacity ---
with t2:
    st.subheader("Healthcare System Stress")
    
    # Utilization Calcs
    beds_used = latest.get('clinic_total_no_of_covid_patients_currently_admitted', 0)
    beds_total = latest.get('clinic_total_no_of_beds_allocated_for_covid_patients', 1) 
    beds_pct = (beds_used / beds_total) * 100 if beds_total > 0 else 0
    
    vents_used = latest.get('clinic_total_no_of_patients_currently_on_ventilator', 0)
    vents_total = latest.get('clinic_total_no_of_ventilators_allocated_for_covid_patients', 1)
    vents_pct = (vents_used / vents_total) * 100 if vents_total > 0 else 0

    col_g1, col_g2, col_g3 = st.columns(3)
    
    with col_g1:
        fig_g1 = go.Figure(go.Indicator(
            mode = "gauge+number", value = beds_pct, 
            title = {'text': "General Bed Occupancy", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100]}, 
                'bar': {'color': COLORS['accent']}, 
                'steps': [{'range': [0, 70], 'color': "#F1F5F9"}, {'range': [70, 100], 'color': "#FECACA"}]
            }
        ))
        fig_g1.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_g1, use_container_width=True)
        
    with col_g2:
        fig_g2 = go.Figure(go.Indicator(
            mode = "gauge+number", value = vents_pct, 
            title = {'text': "Ventilator Utilization", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100]}, 
                'bar': {'color': COLORS['danger']}, 
                'steps': [{'range': [0, 60], 'color': "#F1F5F9"}, {'range': [60, 100], 'color': "#FECACA"}]
            }
        ))
        fig_g2.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_g2, use_container_width=True)

    with col_g3:
        st.markdown("##### 🏥 Oxygen Dependency")
        oxy_val = latest.get('oxygen_dependency_ratio', 0)
        st.metric("Admitted on O2", f"{oxy_val:.1f}%", help="% of Admitted patients req Oxygen")
        
        if 'oxygen_dependency_ratio' in df_filtered.columns:
            fig_oxy = px.area(
                df_filtered, x='date', y='oxygen_dependency_ratio', 
                color_discrete_sequence=[COLORS['success']]
            )
            fig_oxy.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, template="plotly_white")
            fig_oxy.update_xaxes(showticklabels=False)
            fig_oxy.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_oxy, use_container_width=True)

# --- TAB 3: Projections ---
with t3:
    st.subheader("Predictive Analytics (Poly-Regression)")
    
    col_p1, col_p2 = st.columns([1, 4])
    with col_p1:
        st.info("Uses Polynomial Regression (Order 3) to model wave curvature.")
        forecast_days = st.slider("Forecast Horizon", 15, 90, 30)
    
    with col_p2:
        pred_df = df_filtered[['date', 'new_cases_7da']].dropna()
        if len(pred_df) > 20:
            pred_df['days_idx'] = (pred_df['date'] - pred_df['date'].min()).dt.days
            
            # Poly Model
            poly = PolynomialFeatures(degree=3)
            X = poly.fit_transform(pred_df[['days_idx']])
            y = pred_df['new_cases_7da']
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Future
            last_idx = pred_df['days_idx'].max()
            future_X = poly.transform(np.arange(last_idx + 1, last_idx + forecast_days + 1).reshape(-1, 1))
            preds = model.predict(future_X)
            
            # Dates
            future_dates = [pred_df['date'].max() + timedelta(days=x) for x in range(1, forecast_days + 1)]
            
            # Plot
            fig_ai = go.Figure()
            # Historical
            fig_ai.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['new_cases_7da'], name="Historical Data", line=dict(color='#94A3B8')))
            # Forecast
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds, name="AI Forecast", line=dict(color=COLORS['warning'], width=3, dash='dash')))
            
            # Confidence Interval
            fig_ai.add_trace(go.Scatter(x=future_dates, y=preds*1.2, mode='lines', line=dict(width=0), showlegend=False))
            fig_ai.add_trace(go.Scatter(
                x=future_dates, y=preds*0.8, mode='lines', line=dict(width=0), 
                fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)', name="Uncertainty Range"
            ))
            
            fig_ai.update_layout(title="Projected Trajectory", template="plotly_white", height=500, hovermode="x unified")
            st.plotly_chart(fig_ai, use_container_width=True)
        else:
            st.warning("Not enough data points for a reliable projection.")

# --- TAB 4: Audit ---
with t4:
    st.subheader("Data Integrity & Bias Check")
    col_a1, col_a2 = st.columns([2, 1])
    
    with col_a1:
        if 'grand_total_tests_conducted_till_date' in df_filtered.columns:
            fig_bias = px.scatter(
                df_filtered, x='grand_total_tests_conducted_till_date', y='new_cases_7da',
                color='test_positivity_ratio', title="Testing Volume vs. Cases Detected",
                color_continuous_scale="RdBu_r",
                labels={'grand_total_tests_conducted_till_date': 'Total Tests', 'new_cases_7da': 'New Cases', 'test_positivity_ratio': 'Positivity %'}
            )
            fig_bias.update_layout(template="plotly_white")
            st.plotly_chart(fig_bias, use_container_width=True)
    with col_a2:
        st.markdown("### How to read this:")
        st.markdown("""
        * **Linear Line:** If cases rise ONLY when tests rise, the surge might be artificial (testing bias).
        * **Color Shift:** If dots turn **Red** (High Positivity) while moving up, it is a **Real Surge**.
        """)

# --- TAB 5: Policy ---
with t5:
    st.subheader("Non-Pharmaceutical Intervention (NPI) Simulator")
    
    ci1, ci2 = st.columns([1, 2])
    with ci1:
        st.markdown("### ⚙️ Scenario Inputs")
        base_rt_sim = st.number_input("Baseline Rt", value=float(max(0.5, rt_now)), step=0.05, format="%.2f")
        
        st.markdown("**Select Interventions:**")
        i1 = st.checkbox("😷 Mask Mandate (-15%)")
        i2 = st.checkbox("📏 Social Distancing (-10%)")
        i3 = st.checkbox("🏠 Smart Lockdown (-25%)")
        
        reduction = 0.0
        if i1: reduction += 0.15
        if i2: reduction += 0.10
        if i3: reduction += 0.25
        
        final_rt_sim = max(0.1, base_rt_sim * (1 - reduction))
        st.metric("Adjusted Rt", f"{final_rt_sim:.2f}", delta=f"-{reduction*100:.0f}% Efficacy")

    with ci2:
        sim_days = 60
        start_val = max(100, latest['new_cases_7da'])
        dates_sim = [datetime.now() + timedelta(days=x) for x in range(sim_days)]
        
        # Exponential Growth Logic
        curve_base = [start_val * (base_rt_sim**(d/4)) for d in range(sim_days)]
        curve_int = [start_val * (final_rt_sim**(d/4)) for d in range(sim_days)]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=dates_sim, y=curve_base, name="Status Quo", line=dict(color=COLORS['danger'], dash='dot')))
        fig_sim.add_trace(go.Scatter(x=dates_sim, y=curve_int, name="With Intervention", line=dict(color=COLORS['success'], width=3)))
        
        fig_sim.update_layout(
            title="60-Day Impact Projection", 
            yaxis_title="Projected Daily Cases", 
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
        cases_saved = sum(curve_base) - sum(curve_int)
        st.success(f"🛡️ **Impact:** This strategy could prevent **{int(cases_saved):,}** cases over the next 60 days.")

st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #64748B;'>Framework v3.1 | Data Source: Official Records | {datetime.now().year}</div>", unsafe_allow_html=True)


