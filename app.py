import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Config
# ============================================
st.set_page_config(page_title="Pakistan COVID-19 Intelligence Dashboard", layout="wide", page_icon="🇵🇰")
st.title("🇵🇰 Executive Pandemic Intelligence Dashboard - Pakistan")
st.markdown("**Prepared By Student of AI in Public Heatlhcare** | Advanced Insights, AI Projections & Scenario Calculator")

# ============================================
# Load and Prepare Data
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv("Refined + New entities.csv")

    # Clean column names
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("/", "_")
                  .str.replace("(", "")
                  .str.replace(")", "")
                  .str.replace(".", "")
                  .str.replace("__", "_"))

    df["province"] = df["province"].str.strip().str.title()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Key numeric columns
    numeric_cols = [
        'grand_total_cases_till_date',
        'clinic_total_numbers_recovered_and_discharged_so_far',
        'death_cumulative__total_deaths',
        'clinic_total_no_of_covid_patients_currently_admitted',
        'clinic_total_no_of_beds_allocated_for_covid_patients',
        'clinic_total_on_oxygen',
        'clinic_total_no_of_beds_with_oxygen_facility_allocated_for_covid_patients',
        'clinic_total_no_of_patients_currently_on_ventilator',
        'clinic_total_no_of_ventilators_allocated_for_covid_patients',
        'test_positivity_ratio',
        'grand_total_tests_conducted_till_date'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (df[col]
                       .astype(str)
                       .str.replace(',', '', regex=False)
                       .str.strip()
                       .replace(['N/A', 'n/a', '-', '', 'Not Reported'], '0'))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.sort_values(by=["province", "date"]).reset_index(drop=True)

    # Derived metrics
    df["active_cases"] = (df["grand_total_cases_till_date"] -
                          df["clinic_total_numbers_recovered_and_discharged_so_far"] -
                          df["death_cumulative__total_deaths"]).clip(lower=0)

    df["new_cases"] = df.groupby("province")["grand_total_cases_till_date"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby("province")["death_cumulative__total_deaths"].diff().fillna(0).clip(lower=0)
    df["new_recoveries"] = df.groupby("province")["clinic_total_numbers_recovered_and_discharged_so_far"].diff().fillna(0).clip(lower=0)
    df["recovery_rate"] = (df["clinic_total_numbers_recovered_and_discharged_so_far"] / df["grand_total_cases_till_date"] * 100).fillna(0)
    df["fatality_rate"] = (df["death_cumulative__total_deaths"] / df["grand_total_cases_till_date"] * 100).fillna(0)
    df["new_cases_7da"] = df.groupby("province")["new_cases"].rolling(7).mean().reset_index(0, drop=True).fillna(0)

    # Define major COVID waves
    df["wave"] = "Other Periods"
    df.loc[df["date"].between("2020-03-01", "2020-05-31"), "wave"] = "Wave 1 (Mar-May 2020)"
    df.loc[df["date"].between("2020-06-01", "2020-08-31"), "wave"] = "Wave 2 (Jun-Aug 2020)"
    df.loc[df["date"].between("2020-11-01", "2021-01-31"), "wave"] = "Wave 3 (Nov 2020-Jan 2021)"
    df.loc[df["date"].between("2021-04-01", "2021-06-30"), "wave"] = "Wave 4 (Delta - Apr-Jun 2021)"
    df.loc[df["date"].between("2021-07-01", "2021-09-30"), "wave"] = "Wave 4 Peak (Jul-Sep 2021)"
    df.loc[df["date"].between("2022-01-01", "2022-03-31"), "wave"] = "Wave 5 (Omicron - Jan-Mar 2022)"

    return df

df = load_data()

# ============================================
# Sidebar Controls
# ============================================
st.sidebar.header("Executive Controls")

provinces = ["All"] + sorted(df["province"].unique().tolist())
province = st.sidebar.selectbox("Select Province", provinces, index=provinces.index("Punjab"))

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    [max_date - timedelta(days=90), max_date],
    min_value=min_date,
    max_value=max_date
)

# Filter data
mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
df_filtered = df[mask].copy()
if province != "All":
    df_filtered = df_filtered[df_filtered["province"] == province]

# ============================================
# Define 'latest' early
# ============================================
latest = df_filtered.iloc[-1] if not df_filtered.empty else pd.Series()

total_cases = int(latest.get("grand_total_cases_till_date", 0))
active_cases = int(latest.get("active_cases", 0))
total_deaths = int(latest.get("death_cumulative__total_deaths", 0))
total_recovered = int(latest.get("clinic_total_numbers_recovered_and_discharged_so_far", 0))
positivity = float(latest.get("test_positivity_ratio", 0))
new_cases_today = int(latest.get("new_cases", 0))
recovery_rate = float(latest.get("recovery_rate", 0))
fatality_rate = float(latest.get("fatality_rate", 0))
bed_util = (latest.get("clinic_total_no_of_covid_patients_currently_admitted", 0) /
            latest.get("clinic_total_no_of_beds_allocated_for_covid_patients", 1) * 100)
vent_util = (latest.get("clinic_total_no_of_patients_currently_on_ventilator", 0) /
             latest.get("clinic_total_no_of_ventilators_allocated_for_covid_patients", 1) * 100)

# ============================================
# Key Performance Indicators
# ============================================
st.header("Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Cases", f"{total_cases:,}", delta=f"{int(df_filtered['new_cases'].sum()):,} new")
col2.metric("Active Cases", f"{active_cases:,}")
col3.metric("Recovered", f"{total_recovered:,}", delta=f"{recovery_rate:.1f}%")
col4.metric("Deaths", f"{total_deaths:,}", delta=f"{fatality_rate:.2f}%")
col5.metric("Positivity Rate", f"{positivity:.2f}%")

col6, col7, col8 = st.columns(3)
col6.metric("New Cases Today", f"{new_cases_today:,}")
col7.metric("Bed Utilization", f"{bed_util:.1f}%")
col8.metric("Ventilator Utilization", f"{vent_util:.1f}%")

# ============================================
# Tabs
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Epidemic Curves",
    "Healthcare & Capacity",
    "Comparisons & Trends",
    "AI Projections & Calculator",
    "Province & Wave Comparison (All Time)"
])

with tab1:
    st.subheader("Epidemic Curve Analysis")
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Daily New Cases (7-Day Avg)", "Cumulative Cases", "Daily Deaths", "Positivity Rate"))

    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases_7da"], mode='lines', name='7-Day Avg', line=dict(color='#3b82f6', width=3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_cases"], mode='markers', name='Daily', marker=dict(color='#3b82f6', opacity=0.6)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["grand_total_cases_till_date"], mode='lines', name='Total Cases', line=dict(color='#8b5cf6')), row=1, col=2)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["new_deaths"], mode='lines+markers', name='Daily Deaths', line=dict(color='#ef4444')), row=2, col=1)
    fig1.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["test_positivity_ratio"], mode='lines', name='Positivity %', line=dict(color='#f59e0b')), row=2, col=2)

    fig1.update_layout(height=700, template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Healthcare Capacity & Utilization")
    capacity_df = pd.DataFrame({
        "Metric": ["General Beds", "Oxygen Beds", "Ventilators"],
        "Used": [
            latest.get("clinic_total_no_of_covid_patients_currently_admitted", 0),
            latest.get("clinic_total_on_oxygen", 0),
            latest.get("clinic_total_no_of_patients_currently_on_ventilator", 0)
        ],
        "Available": [
            latest.get("clinic_total_no_of_beds_allocated_for_covid_patients", 1) - latest.get("clinic_total_no_of_covid_patients_currently_admitted", 0),
            latest.get("clinic_total_no_of_beds_with_oxygen_facility_allocated_for_covid_patients", 1) - latest.get("clinic_total_on_oxygen", 0),
            latest.get("clinic_total_no_of_ventilators_allocated_for_covid_patients", 1) - latest.get("clinic_total_no_of_patients_currently_on_ventilator", 0)
        ]
    })

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=capacity_df["Metric"], x=capacity_df["Used"], name="Used", orientation='h', marker_color='#ef4444'))
    fig2.add_trace(go.Bar(y=capacity_df["Metric"], x=capacity_df["Available"], name="Available", orientation='h', marker_color='#10b981'))
    fig2.update_layout(barmode='stack', title="Resource Utilization", height=400, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(capacity_df)

    st.subheader("Utilization Trends")
    util_trend = df_filtered.melt(id_vars="date", value_vars=[
        "clinic_total_no_of_covid_patients_currently_admitted",
        "clinic_total_on_oxygen",
        "clinic_total_no_of_patients_currently_on_ventilator"
    ], var_name="Resource", value_name="Count")
    fig_util = px.line(util_trend, x="date", y="Count", color="Resource")
    st.plotly_chart(fig_util, use_container_width=True)

with tab3:
    st.subheader("Cross-Province Comparisons")
    if province == "All":
        latest_prov = df.groupby("province").last().reset_index()
        fig_prov = make_subplots(rows=1, cols=3, subplot_titles=("Total Cases", "Active Cases", "Fatality Rate (%)"))
        fig_prov.add_trace(go.Bar(x=latest_prov["province"], y=latest_prov["grand_total_cases_till_date"], marker_color='#3b82f6'), row=1, col=1)
        fig_prov.add_trace(go.Bar(x=latest_prov["province"], y=latest_prov["active_cases"], marker_color='#f59e0b'), row=1, col=2)
        fig_prov.add_trace(go.Bar(x=latest_prov["province"], y=latest_prov["fatality_rate"], marker_color='#ef4444'), row=1, col=3)
        fig_prov.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_prov, use_container_width=True)

        st.subheader("Province Comparison Table")
        comp_table = df.groupby("province").last()[["grand_total_cases_till_date", "active_cases", "death_cumulative__total_deaths", "test_positivity_ratio", "recovery_rate"]]
        comp_table.columns = ["Total Cases", "Active Cases", "Total Deaths", "Positivity %", "Recovery Rate %"]
        st.dataframe(comp_table.style.format("{:,.0f}", subset=["Total Cases", "Active Cases", "Total Deaths"])
                                   .format("{:.2f}%", subset=["Positivity %", "Recovery Rate %"]))

    st.subheader("All Key Trends (Automatically Displayed)")
    trend_cols = ["new_cases", "active_cases", "new_deaths", "new_recoveries", "test_positivity_ratio"]
    trend_names = ["New Cases", "Active Cases", "Daily Deaths", "Daily Recoveries", "Positivity Rate"]
    fig_trends = px.line(df_filtered, x="date", y=trend_cols, title="Key Trends Over Time")
    fig_trends.for_each_trace(lambda t: t.update(name=trend_names[trend_cols.index(t.name)]))
    st.plotly_chart(fig_trends, use_container_width=True)

    st.subheader("Monthly Cases Heatmap")
    df_month = df.copy()
    df_month["month"] = df_month["date"].dt.to_period("M").astype(str)
    monthly = df_month.pivot_table(values="new_cases", index="province", columns="month", aggfunc="sum", fill_value=0)
    fig_heat = px.imshow(monthly, title="Monthly New Cases by Province")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.subheader("AI Projections")
    if len(df_filtered) > 10:
        pred_data = df_filtered[["date", "new_cases"]].dropna()
        pred_data["days"] = (pred_data["date"] - pred_data["date"].min()).dt.days

        lin = LinearRegression().fit(pred_data[["days"]], pred_data["new_cases"])
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(pred_data[["days"]], pred_data["new_cases"])

        future_days = np.arange(pred_data["days"].max() + 1, pred_data["days"].max() + 31).reshape(-1, 1)
        future_dates = [pred_data["date"].max() + timedelta(days=i) for i in range(1, 31)]

        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=pred_data["date"], y=pred_data["new_cases"], mode='lines', name='Historical'))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=lin.predict(future_days), mode='lines', name='Linear', line=dict(dash='dash')))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=rf.predict(future_days), mode='lines', name='Random Forest', line=dict(dash='dot')))

        rates = [-3, -1, 2, 5, 8]
        current = pred_data["new_cases"].iloc[-1]
        for r in rates:
            proj = [current * (1 + r/100)**d for d in range(1, 31)]
            fig_ai.add_trace(go.Scatter(x=future_dates, y=proj, mode='lines', name=f'{r}% Growth'))

        fig_ai.update_layout(title="30-Day Projections", height=600)
        st.plotly_chart(fig_ai, use_container_width=True)
    else:
        st.info("Not enough data for projections.")

    st.subheader("Scenario Calculator")
    base_active = active_cases if active_cases > 0 else 1000
    current_active = st.number_input("Current Active Cases", value=base_active)
    r0 = st.slider("R0", 0.5, 4.0, 1.1, 0.05)
    reduction = st.slider("Intervention Reduction (%)", 0, 60, 20)
    start_day = st.slider("Intervention Start Day", 0, 60, 14)
    days = st.slider("Days", 30, 180, 90)

    proj = []
    cur = current_active
    for d in range(1, days + 1):
        r = r0 if d < start_day else r0 * (1 - reduction/100)
        cur *= r
        proj.append(cur)

    fig_scen = px.line(x=list(range(1, days + 1)), y=proj, title=f"R0 = {r0} → {r0*(1-reduction/100):.2f} after Day {start_day}")
    st.plotly_chart(fig_scen, use_container_width=True)
    st.write(f"**Projected in {days} days:** {int(proj[-1]):,}")

with tab5:
    st.header("Province-Wise & Wave-Wise Historical Comparison")

    df_full = df.copy()

    wave_stats = df_full.groupby(["wave", "province"]).agg({
        "new_cases": "sum",
        "new_deaths": "sum",
        "grand_total_cases_till_date": "max",
        "test_positivity_ratio": "mean"
    }).reset_index()

    wave_stats.rename(columns={
        "new_cases": "Total New Cases",
        "new_deaths": "Total Deaths",
        "grand_total_cases_till_date": "Peak Cases",
        "test_positivity_ratio": "Avg Positivity %"
    }, inplace=True)

    st.subheader("Summary Table by Wave & Province")
    # Safe display without complex styling on multi-index
    st.dataframe(wave_stats.style.format({
        "Total New Cases": "{:,.0f}",
        "Total Deaths": "{:,.0f}",
        "Peak Cases": "{:,.0f}",
        "Avg Positivity %": "{:.2f}%"
    }))

    st.subheader("Peak Cases by Wave")
    fig_cases = px.bar(wave_stats, x="wave", y="Peak Cases", color="province", barmode="group", title="Peak Cases by Wave & Province")
    st.plotly_chart(fig_cases, use_container_width=True)

    st.subheader("Total Deaths by Wave")
    fig_deaths = px.bar(wave_stats, x="wave", y="Total Deaths", color="province", barmode="group", title="Deaths by Wave & Province")
    st.plotly_chart(fig_deaths, use_container_width=True)

    st.subheader("Average Positivity Rate Heatmap")
    pos_pivot = wave_stats.pivot(index="province", columns="wave", values="Avg Positivity %")
    fig_pos = px.imshow(pos_pivot, color_continuous_scale="RdYlGn_r", title="Avg Positivity % by Wave & Province")
    st.plotly_chart(fig_pos, use_container_width=True)

# ============================================
# Risk & Recommendations
# ============================================
st.header("Risk Assessment")
risk_score = 0
alerts = []
if positivity > 10: risk_score += 30; alerts.append("🔴 CRITICAL: Positivity >10%")
elif positivity > 5: risk_score += 15; alerts.append("🟡 WARNING: Positivity >5%")
if bed_util > 80: risk_score += 25; alerts.append(f"🔴 CRITICAL: Bed Occupancy {bed_util:.1f}%")
elif bed_util > 60: risk_score += 10; alerts.append(f"🟡 WARNING: Bed Occupancy {bed_util:.1f}%")
if fatality_rate > 3: risk_score += 20; alerts.append(f"🔴 CRITICAL: Fatality {fatality_rate:.2f}%")

risk_level = "🟢 LOW" if risk_score < 30 else "🟡 MODERATE" if risk_score < 60 else "🔴 HIGH"
st.markdown(f"**Risk Level:** {risk_level} ({risk_score}/100)")

for a in alerts:
    st.warning(a)

st.success("Dashboard Ready for Executive Briefing")
st.caption(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')} | Data Source: Refined + New entities.csv")