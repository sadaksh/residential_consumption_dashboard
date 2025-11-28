import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
import numpy as np
from datetime import datetime, timedelta
import warnings
import re

warnings.filterwarnings('ignore')

# ================= CONFIG =================
DATA_FOLDER = "processed_apartments2"

# 5-min interval data → hours
INTERVAL_MINUTES = 5
INTERVAL_HOURS = INTERVAL_MINUTES / 60.0  # 5/60 = 0.0833 h

# ----------------- NAME NORMALIZATION (for matching) -----------------
def normalize_name(name: str) -> str:
    """
    Normalize apartment / column names so variants map to same key:
    - lowercased
    - remove spaces, hyphens, punctuation
    """
    if pd.isna(name):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


# ----------------- AC COLUMN DETECTION HELPER -----------------
FORBIDDEN_AC_SUBSTRINGS = [
    'washing', 'washer', 'wm', 'machine',
    'fridge', 'refrigerator',
    'geyser',
    'microwave', 'oven',
    'light', 'lighting',
    'fan', 'fans',
    'socket', 'plug'
]


def is_ac_column(name: str) -> bool:
    """
    Decide if a column represents an Air Conditioner.

    - Excludes known non-AC appliances (washing machine, fridge, lights, fans, etc.)
    - Accepts explicit 'air conditioner'
    - Accepts 'ac' / 'a/c' only as a standalone token
    """
    n = name.lower()

    # 1) Explicitly rule out obvious non-AC loads
    if any(f in n for f in FORBIDDEN_AC_SUBSTRINGS):
        return False

    # 2) Explicit "air conditioner"
    if 'air conditioner' in n:
        return True

    # 3) Token-based 'ac' detection
    clean = re.sub(r'[^a-z0-9]+', ' ', n)  # "Living_AC-1TR" → "living ac 1tr"
    tokens = clean.split()

    return 'ac' in tokens or 'a/c' in tokens


# ================= ENERGY COLUMN HELPER =================
def get_energy_columns(df: pd.DataFrame) -> list:
    """
    Return likely appliance / power columns (exclude meta and non-appliance columns).
    This version automatically excludes phase columns.
    """
    exclude_keys = ['timestamp', 'apartment', 'phase', 'date', 'time',
                    'b phase', 'r phase', 'y phase']

    cols = []
    for c in df.columns:
        cname = c.lower().strip()
        if not any(k in cname for k in exclude_keys):
            cols.append(c)
    return cols


# ================= LOAD DATA =================
@st.cache_data
def load_all_data():
    """
    Loads all CSVs from DATA_FOLDER.
    Each file corresponds to one apartment.
    Values represent instantaneous power in kW at 5-min intervals.
    Removes non-appliance phase columns (B, R, Y Phase).
    """
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    dfs = {}

    for f in all_files:
        try:
            df = pd.read_csv(f)
            apt_name = os.path.splitext(os.path.basename(f))[0]
            df["Apartment"] = apt_name

            # Remove phase columns before any calculations
            df = df[[c for c in df.columns
                     if not any(k in c.lower() for k in ['b phase', 'r phase', 'y phase'])]]

            # Convert timestamps
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

            # Clean and convert energy columns
            energy_cols = get_energy_columns(df)
            if energy_cols:
                df[energy_cols] = (
                    df[energy_cols]
                    .apply(lambda s: pd.to_numeric(
                        s.astype(str).str.replace(",", ""), errors="coerce"))
                    .abs()
                )

            dfs[apt_name] = df

        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
    return dfs


# ================= LOAD WEATHER DATA =================
@st.cache_data
def load_weather_data(data_folder="processed_apartments2"):
    """
    Loads indoor weather Excel files for specific apartments/zones.
    Returns dict with zone name → DataFrame.
    """
    weather_files = {
        "Living": [
            os.path.join(data_folder,
                         "Ambient Temperature & Humidity LakesideK502_Living Bedroom.xlsx"),
            "Ambient Temperature & Humidity LakesideK502_Living Bedroom.xlsx"
        ],
        "Master": [
            os.path.join(data_folder,
                         "Ambient Temperature & Humidity LakesideK502_Master Bedroom.xlsx"),
            "Ambient Temperature & Humidity LakesideK502_Master Bedroom.xlsx"
        ]
    }

    results = {}
    for zone, patterns in weather_files.items():
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        if not files:
            continue

        dfs = []
        for f in files:
            try:
                df = pd.read_excel(f)
                time_col = [c for c in df.columns
                            if 'time' in c.lower() or 'timestamp' in c.lower()]
                if not time_col:
                    continue
                df.rename(columns={time_col[0]: 'Timestamp'}, inplace=True)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                df = df[['Timestamp'] + numeric_cols]
                df['Source'] = os.path.basename(f)
                dfs.append(df)
            except Exception as e:
                st.warning(f"Could not load {zone} file {f}: {e}")

        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged = merged.groupby('Timestamp').mean(numeric_only=True).reset_index()
            rename_map = {c: c.strip().title() for c in merged.columns}
            merged.rename(columns=rename_map, inplace=True)
            results[zone] = merged
    return results


# ================= WEATHER CORRELATION (AC vs Weather) =================
def plot_weather_correlation(df_energy, df_weather):
    """
    Correlate AC instantaneous power (kW) with ambient conditions.
    Also computes daily AC energy (kWh) using 5-min intervals.
    """
    if df_weather.empty:
        st.warning("⚠️ No weather data available.")
        return

    merged = pd.merge_asof(
        df_energy.sort_values('Timestamp'),
        df_weather.sort_values('Timestamp'),
        on='Timestamp',
        tolerance=pd.Timedelta('15min'),
        direction='nearest'
    )

    # Robust AC column detection
    ac_cols = [c for c in merged.columns if is_ac_column(c)]

    if not ac_cols:
        st.warning("No AC column found.")
        return

    ac_col = ac_cols[0]

    temp_candidates = [c for c in merged.columns if 'temp' in c.lower()]
    hum_candidates = [c for c in merged.columns
                      if 'humid' in c.lower() or 'rh' in c.lower()]

    if not temp_candidates:
        st.warning("No temperature column found.")
        return

    temp_col = temp_candidates[0]
    color_col = hum_candidates[0] if hum_candidates else None

    st.success(
        f"✅ Using {ac_col} (AC), {temp_col} (Temperature)"
        f"{'' if color_col is None else f' and {color_col} (Humidity)'}"
    )

    # Scatter: AC power (kW) vs temperature
    fig = px.scatter(
        merged,
        x=temp_col,
        y=ac_col,
        color=color_col,
        trendline='ols',
        title=f"{ac_col} vs {temp_col}",
        labels={temp_col: "Temperature (°C)", ac_col: "AC Power (kW)"}
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

    # Daily aggregates: total AC energy (kWh/day) from kW time-series
    merged['Date'] = merged['Timestamp'].dt.date
    daily = merged.groupby('Date').agg({
        ac_col: 'sum',
        temp_col: 'mean'
    }).reset_index()

    # Convert summed kW readings → kWh using 5-min interval
    daily['AC Energy (kWh)'] = daily[ac_col] * INTERVAL_HOURS

    BASE_TEMP = 24
    daily['CDD'] = (daily[temp_col] - BASE_TEMP).clip(lower=0)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(
        go.Bar(x=daily['Date'], y=daily['CDD'],
               name='Cooling Degree Days', opacity=0.6),
        secondary_y=False
    )
    fig2.add_trace(
        go.Scatter(
            x=daily['Date'],
            y=daily['AC Energy (kWh)'],
            name='Daily AC Energy (kWh)',
            mode='lines+markers'
        ),
        secondary_y=True
    )
    fig2.update_layout(title="Daily AC Energy vs Cooling Degree Days",
                       xaxis_title="Date")
    st.plotly_chart(fig2, use_container_width=True)


# ================= OPERATION DURATION & ANOMALIES =================
def calculate_operation_duration_improved(df, appliance_col, threshold=0.001):
    """Calculate ON durations for an appliance (based on power > threshold)."""
    if appliance_col not in df.columns:
        return pd.DataFrame()

    data = df[['Timestamp', appliance_col]].dropna().sort_values('Timestamp')
    if data.empty:
        return pd.DataFrame()

    data['is_on'] = data[appliance_col] > threshold
    data['state_change'] = data['is_on'].ne(data['is_on'].shift())
    data['group_id'] = data['state_change'].cumsum()

    events = []
    for group_id, group in data.groupby('group_id'):
        if group['is_on'].iloc[0]:
            start_time = group['Timestamp'].iloc[0]
            end_time = group['Timestamp'].iloc[-1]
            duration_hours = (end_time - start_time).total_seconds() / 3600
            if duration_hours >= 0.083:  # ≥5 minutes
                events.append({
                    'appliance': appliance_col,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_hours': duration_hours,
                    'avg_power': group[appliance_col].mean(),
                    'max_power': group[appliance_col].max(),
                    'day_of_week': start_time.strftime('%A')
                })
    return pd.DataFrame(events)


def detect_on_off_events(df, appliance_col, threshold=0.01):
    """Detect ON/OFF transitions."""
    if appliance_col not in df.columns:
        return pd.DataFrame()
    data = df[['Timestamp', appliance_col]].dropna().sort_values('Timestamp')
    if data.empty:
        return pd.DataFrame()

    data['state'] = df[appliance_col] > threshold
    data['state_change'] = data['state'].ne(data['state'].shift())
    data['event_id'] = data['state_change'].cumsum()
    return data[data['state_change'] & data['state']]


def detect_anomalies_operation(df, appliance_col, expected_hours=None):
    """Detect appliances running longer/shorter than typical range."""
    anomalies = []
    if appliance_col not in df.columns:
        return anomalies

    operation_df = calculate_operation_duration_improved(df, appliance_col)
    if operation_df.empty:
        return anomalies

    if expected_hours is None:
        appliance_lower = appliance_col.lower()
        if 'ac' in appliance_lower:
            expected_hours = {'min': 1, 'max': 8}
        elif 'geyser' in appliance_lower:
            expected_hours = {'min': 0.5, 'max': 3}
        elif 'washing' in appliance_lower:
            expected_hours = {'min': 0.5, 'max': 2}
        else:
            expected_hours = {'min': 0.1, 'max': 24}

    long_ops = operation_df[operation_df['duration_hours'] > expected_hours['max']]
    short_ops = operation_df[operation_df['duration_hours'] < expected_hours['min']]

    for _, op in long_ops.iterrows():
        anomalies.append({
            'appliance': appliance_col,
            'type': 'Long Operation',
            'description': f"{op['duration_hours']:.2f}h > expected {expected_hours['max']}h",
            'start_time': op['start_time'],
            'severity': 'Medium'
        })
    for _, op in short_ops.iterrows():
        anomalies.append({
            'appliance': appliance_col,
            'type': 'Short Operation',
            'description': f"{op['duration_hours']:.2f}h < expected {expected_hours['min']}h",
            'start_time': op['start_time'],
            'severity': 'Low'
        })
    return anomalies


# ================= COMPARATIVE PLOTS: ALL APARTMENTS =================
def plot_hourly_consumption_all_apartments(all_dfs):
    """Plot average power (kW) by hour of day for all apartments."""
    hourly_data = []
    for apt_name, df in all_dfs.items():
        if 'Timestamp' not in df.columns:
            continue
        appliance_cols = get_energy_columns(df)
        if not appliance_cols:
            continue
        df['hour'] = df['Timestamp'].dt.hour
        hourly_avg = df.groupby('hour')[appliance_cols].mean().mean(axis=1)
        for hour, consumption in hourly_avg.items():
            hourly_data.append(
                {'Apartment': apt_name, 'Hour': hour,
                 'Average Power (kW)': consumption}
            )

    if not hourly_data:
        st.warning("No hourly data available")
        return

    hourly_df = pd.DataFrame(hourly_data)
    fig = px.scatter(
        hourly_df,
        x='Hour',
        y='Average Power (kW)',
        color='Apartment',
        title='Average Power by Hour of Day (kW)',
        labels={'Hour': 'Hour of Day', 'Average Power (kW)': 'Average Power (kW)'}
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)


def plot_weekday_weekend_comparison_all_apartments(all_dfs):
    """Compare average power (kW) weekdays vs weekends for each apartment."""
    comparison_data = []
    for apt_name, df in all_dfs.items():
        if 'Timestamp' not in df.columns:
            continue
        appliance_cols = get_energy_columns(df)
        if not appliance_cols:
            continue
        df['day_type'] = df['Timestamp'].dt.dayofweek.apply(
            lambda x: 'Weekend' if x >= 5 else 'Weekday')
        daily_mean = df.groupby('day_type')[appliance_cols].mean().mean(axis=1)
        for day_type, val in daily_mean.items():
            comparison_data.append(
                {'Apartment': apt_name, 'Day Type': day_type,
                 'Average Power (kW)': val}
            )

    if not comparison_data:
        st.warning("No weekday/weekend data available")
        return

    comp_df = pd.DataFrame(comparison_data)
    fig = px.scatter(
        comp_df, x='Day Type', y='Average Power (kW)', color='Apartment',
        title='Weekday vs Weekend - Average Power (kW)',
        labels={'Average Power (kW)': 'Average Power (kW)'}
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

def plot_epi_all_apartments(all_dfs, areas, start_date, end_date):
    """
    Plot *average daily* EPI (kWh/m²/day) for each apartment
    over the currently selected date range.

    - Uses the data already date-filtered in all_dfs
    - Skips apartments with no data, no appliance columns, or no area
    """
    # Number of days in selected period (avoid division by zero)
    period_days = max((end_date - start_date).days, 1)

    epi_rows = []

    for apt_name, df in all_dfs.items():
        if df is None or df.empty:
            continue
        if "Timestamp" not in df.columns:
            continue

        energy_cols = get_energy_columns(df)
        if not energy_cols:
            continue

        # Get apartment area (m²) using normalized name
        apt_key = normalize_name(apt_name)
        area_m2 = areas.get(apt_key)
        if not area_m2 or area_m2 <= 0:
            continue

        # Total kWh for the selected range
        total_kwh = df[energy_cols].sum(numeric_only=True).sum() * INTERVAL_HOURS
        if total_kwh <= 0:
            continue

        # Average daily EPI over the selected period
        epi_daily = total_kwh / (area_m2 * period_days)
        epi_rows.append({"Apartment": apt_name, "EPI (kWh/m²/day)": epi_daily})

    if not epi_rows:
        st.info("No EPI data available for the selected date range.")
        return

    epi_df = (
        pd.DataFrame(epi_rows)
        .sort_values("EPI (kWh/m²/day)", ascending=False)
    )

    fig = px.bar(
        epi_df,
        x="Apartment",
        y="EPI (kWh/m²/day)",
        title="Average Daily Energy Performance Index (EPI) – Selected Date Range",
        labels={
            "Apartment": "Apartment",
            "EPI (kWh/m²/day)": "Average Daily EPI (kWh/m²/day)",
        },
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ================= INDIVIDUAL APARTMENT PLOTS =================
def plot_energy_consumption_over_time(df, apartment_name, tab_name):
    """Plot instantaneous appliance power (kW) over time."""
    appliance_cols = get_energy_columns(df)
    if not appliance_cols:
        st.warning("No appliance data found for this apartment")
        return

    melt_df = df.melt(
        id_vars=['Timestamp'], value_vars=appliance_cols,
        var_name='Appliance', value_name='Power (kW)'
    ).dropna()

    fig = px.line(
        melt_df,
        x='Timestamp', y='Power (kW)', color='Appliance',
        title=f'{apartment_name} - Power Consumption Over Time (kW)',
        labels={'Power (kW)': 'Power (kW)', 'Timestamp': 'Time'}
    )
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"{apartment_name}_energy_time_{tab_name}")


def plot_appliance_wise_energy(df, apartment_name, tab_name):
    """Plot appliance-wise total energy (kWh)."""
    appliance_cols = get_energy_columns(df)
    if not appliance_cols:
        return

    energy_totals = [
        {
            'Appliance': col,
            'Total Energy (kWh)': df[col].sum() * INTERVAL_HOURS
        }
        for col in appliance_cols
    ]
    energy_df = pd.DataFrame(energy_totals).sort_values(
        'Total Energy (kWh)', ascending=False)

    fig = px.bar(
        energy_df,
        x='Appliance', y='Total Energy (kWh)', color='Total Energy (kWh)',
        title=f'{apartment_name} - Appliance-wise Total Energy (kWh)'
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"{apartment_name}_appliance_wise_{tab_name}")


def plot_weekday_weekend_comparison(df, apartment_name, tab_name):
    """Compare average power (kW) on weekdays vs weekends."""
    appliance_cols = get_energy_columns(df)
    if not appliance_cols or 'Timestamp' not in df.columns:
        return

    data = df.copy()
    data['day_type'] = data['Timestamp'].dt.dayofweek.apply(
        lambda x: 'Weekend' if x >= 5 else 'Weekday')
    comparison_data = []
    for col in appliance_cols:
        weekday_avg = data[data['day_type'] == 'Weekday'][col].mean()
        weekend_avg = data[data['day_type'] == 'Weekend'][col].mean()
        if not np.isnan(weekday_avg) and not np.isnan(weekend_avg):
            comparison_data.extend([
                {'Appliance': col, 'Day Type': 'Weekday',
                 'Average Power (kW)': weekday_avg},
                {'Appliance': col, 'Day Type': 'Weekend',
                 'Average Power (kW)': weekend_avg}
            ])
    comp_df = pd.DataFrame(comparison_data)
    if comp_df.empty:
        st.warning("No data available for weekday/weekend comparison")
        return

    fig = px.bar(
        comp_df, x='Appliance', y='Average Power (kW)', color='Day Type',
        barmode='group',
        title=f'{apartment_name} - Weekday vs Weekend Power (kW)'
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"{apartment_name}_weekday_weekend_{tab_name}")


def plot_ac_energy_time_bins(df, apartment_name, tab_name):
    """Plot AC power (kW) levels by hour of day."""
    # Use robust AC detection
    ac_cols = [c for c in df.columns if is_ac_column(c)]
    if not ac_cols:
        st.info("No AC data found for this apartment")
        return

    for i, ac_col in enumerate(ac_cols):
        ac_data = df[['Timestamp', ac_col]].copy()
        ac_data = ac_data[ac_data[ac_col] > 0.01]
        if ac_data.empty:
            continue

        ac_data['hour'] = ac_data['Timestamp'].dt.hour
        ac_data['power_bin'] = pd.cut(
            ac_data[ac_col],
            bins=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        pivot_data = ac_data.groupby(
            ['hour', 'power_bin']).size().unstack(fill_value=0)

        fig = px.imshow(
            pivot_data.T,
            title=f'{apartment_name} - {ac_col} Power Levels by Time of Day',
            labels=dict(x="Hour of Day", y="Power Level", color="Occurrences"),
            aspect="auto"
        )
        fig.update_layout(height=400)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"{apartment_name}_ac_bins_{i}_{tab_name}"
        )


def plot_operation_duration(df, apartment_name, tab_name):
    """Plot average duration of operation (hours)."""
    appliance_cols = get_energy_columns(df)
    if not appliance_cols:
        return

    duration_data = []
    for appliance in appliance_cols:
        operation_df = calculate_operation_duration_improved(df, appliance)
        if operation_df.empty:
            continue
        operation_df['week'] = operation_df['start_time'].dt.isocalendar().week
        operation_df['year'] = operation_df['start_time'].dt.year
        weekly_avg = operation_df.groupby(
            ['year', 'week'])['duration_hours'].mean().reset_index()
        weekly_avg['Appliance'] = appliance
        duration_data.append(weekly_avg)

    if not duration_data:
        st.info("No operation duration data available")
        return

    duration_df = pd.concat(duration_data)
    fig = px.line(
        duration_df, x='week', y='duration_hours', color='Appliance',
        title=f'{apartment_name} - Weekly Average Operation Duration (h)',
        labels={'duration_hours': 'Average Duration (h)', 'week': 'Week'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"{apartment_name}_operation_duration_{tab_name}")


def plot_hourly_profile(df, apartment_name, tab_name):
    """
    Plot average total power (kW) by hour of day for a single apartment.
    - Uses only rows with valid timestamps
    - Sums all appliance powers to get total kW at each timestamp
    - Averages that total by hour of day
    """
    if "Timestamp" not in df.columns or df.empty:
        st.info("No timestamped data available for hourly profile.")
        return

    data = df.dropna(subset=["Timestamp"]).copy()
    data["hour"] = data["Timestamp"].dt.hour

    appliance_cols = get_energy_columns(data)
    if not appliance_cols:
        st.info("No appliance columns found for hourly profile.")
        return

    # Total instantaneous power across all appliances
    data["Total Power (kW)"] = data[appliance_cols].sum(axis=1)

    # Average total power per hour
    hourly = (
        data.groupby("hour")["Total Power (kW)"]
        .mean()
        .reset_index()
        .sort_values("hour")
    )

    fig = px.line(
        hourly,
        x="hour",
        y="Total Power (kW)",
        markers=True,
        title=f"{apartment_name} – Average Power by Hour of Day (kW)",
        labels={"hour": "Hour of Day", "Total Power (kW)": "Average Power (kW)"}
    )
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{apartment_name}_hourly_profile_{tab_name}",
    )


# ================= APPLIANCE OPERATION HEATMAP =================
def plot_appliance_operation_heatmap(df, apartment_name, tab_name):
    """Create heatmap of appliance ON occurrences by hour/day."""
    appliance_cols = get_energy_columns(df)
    if not appliance_cols or 'Timestamp' not in df.columns:
        return

    # Focus on top 5 appliances by total energy (kWh)
    totals = {c: df[c].sum() * INTERVAL_HOURS for c in appliance_cols}
    top_appliances = sorted(totals, key=totals.get, reverse=True)[:5]

    for i, appliance in enumerate(top_appliances):
        data = df[['Timestamp', appliance]].copy()
        data = data[data[appliance] > 0.01]
        if data.empty:
            continue

        data['hour'] = data['Timestamp'].dt.hour
        data['day'] = data['Timestamp'].dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday',
                      'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = data.groupby(
            ['day', 'hour']).size().unstack(fill_value=0).reindex(days_order)

        fig = px.imshow(
            pivot,
            title=f"{apartment_name} - {appliance} Operation Pattern",
            labels=dict(x="Hour of Day", y="Day of Week", color="Count"),
            aspect="auto"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True,
                        key=f"{apartment_name}_heatmap_{i}_{tab_name}")


# ================= ON/OFF OCCURRENCES =================
def plot_on_off_occurrences(df, apartment_name, timeframe='daily', tab_name=""):
    """Plot ON event counts per appliance (daily or weekly)."""
    appliance_cols = get_energy_columns(df)
    if not appliance_cols:
        return

    occurrences = []
    for appliance in appliance_cols:
        events = detect_on_off_events(df, appliance)
        if events.empty:
            continue

        if timeframe == 'daily':
            events['date'] = events['Timestamp'].dt.date
            daily = events.groupby('date').size().reset_index(name='ON_Events')
            for _, row in daily.iterrows():
                occurrences.append(
                    {'Date': row['date'], 'Appliance': appliance,
                     'ON_Events': row['ON_Events']}
                )
        else:  # weekly
            events['week'] = events['Timestamp'].dt.isocalendar().week
            events['year'] = events['Timestamp'].dt.year
            weekly = events.groupby(
                ['year', 'week']).size().reset_index(name='ON_Events')
            for _, row in weekly.iterrows():
                occurrences.append(
                    {'Week': f"{row['year']}-W{row['week']}",
                     'Appliance': appliance,
                     'ON_Events': row['ON_Events']}
                )

    if not occurrences:
        st.info("No ON/OFF events detected.")
        return

    occ_df = pd.DataFrame(occurrences)
    if timeframe == 'daily':
        fig = px.line(
            occ_df, x='Date', y='ON_Events', color='Appliance',
            title=f'{apartment_name} - Daily ON Events (Count)',
            labels={'ON_Events': 'ON Events'}
        )
    else:
        fig = px.bar(
            occ_df, x='Week', y='ON_Events', color='Appliance',
            title=f'{apartment_name} - Weekly ON Events (Count)',
            barmode='group'
        )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"{apartment_name}_on_off_{timeframe}_{tab_name}")


# ================= DATA DIAGNOSIS =================
def diagnose_data_issues(all_dfs):
    """Quick diagnosis for missing/invalid power data."""
    st.header("🔍 Data Diagnosis")
    for apt_name, df in all_dfs.items():
        with st.expander(f"Diagnosis for {apt_name}"):
            st.write(f"Records: {len(df)}")
            if 'Timestamp' in df.columns:
                st.write(
                    f"Date Range: {df['Timestamp'].min()} → {df['Timestamp'].max()}")

            appliance_cols = get_energy_columns(df)
            st.write(f"Potential appliance columns: {appliance_cols[:10]}")

            for col in appliance_cols[:5]:
                nonzero = (df[col] > 0.01).sum()
                st.write(
                    f"**{col}** → Non-zero: {nonzero}/{len(df)}  "
                    f"Mean: {df[col].mean():.3f} kW  Max: {df[col].max():.3f} kW"
                )

                test = calculate_operation_duration_improved(df, col)
                st.write(f"ON events detected: {len(test)}")


# ================= EPI MODULE (LOADERS) =================
@st.cache_data
def load_apartment_areas(path="processed_apartments2/metadata/apartment_area.csv"):
    """Load apartment areas (m²) for EPI calculation."""
    if not os.path.exists(path):
        st.warning("❌ apartment_area.csv not found.")
        return {}
    df = pd.read_csv(path)
    if not {"Apartment", "Area_m2"}.issubset(df.columns):
        st.error("apartment_area.csv must have columns: Apartment, Area_m2")
        return {}
    df["key"] = df["Apartment"].apply(normalize_name)
    return dict(zip(df["key"], df["Area_m2"]))


@st.cache_data
def load_historical_consumption(path="processed_apartments2/metadata/historical_epi.csv"):
    """
    Load historical monthly consumption (kWh).
    Expected columns: Apartment, Year, Month, Total_kWh
    """
    if not os.path.exists(path):
        st.warning("⚠️ historical_epi.csv not found.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required_cols = {"Apartment", "Year", "Month", "Total_kWh"}
    if not required_cols.issubset(df.columns):
        st.error("historical_epi.csv must have columns: Apartment, Year, Month, Total_kWh")
        return pd.DataFrame()
    df["key"] = df["Apartment"].apply(normalize_name)
    return df


def plot_monthly_epi(df, apartment_name, area_m2, historical_df=None):
    """Compute and plot monthly EPI (kWh/m²) for a given apartment."""
    if "Timestamp" not in df.columns:
        st.warning("No timestamp column found — cannot calculate EPI.")
        return

    energy_cols = get_energy_columns(df)
    if not energy_cols:
        st.warning("No appliance columns found.")
        return

    df["Month"] = df["Timestamp"].dt.to_period("M")
    # Sum kW over month and convert to kWh
    monthly_kwh = (
        df.groupby("Month")[energy_cols]
        .sum(numeric_only=True)
        .sum(axis=1) * INTERVAL_HOURS
    )

    epi_current = pd.DataFrame({
        "Apartment": apartment_name,
        "Year": [m.year for m in monthly_kwh.index],
        "Month": [m.month for m in monthly_kwh.index],
        "EPI (kWh/m²)": monthly_kwh.values / area_m2
    })

    if historical_df is not None and not historical_df.empty:
        key = normalize_name(apartment_name)
        hist = historical_df[historical_df["key"] == key].copy()
        hist["EPI (kWh/m²)"] = hist["Total_kWh"] / area_m2
        epi_all = pd.concat([epi_current, hist], ignore_index=True)
    else:
        epi_all = epi_current

    epi_all["Month_str"] = epi_all["Month"].apply(lambda m: f"{int(m):02d}")
    epi_all["YearMonth"] = epi_all["Year"].astype(str) + "-" + epi_all["Month_str"]

    fig = px.line(
        epi_all,
        x="YearMonth",
        y="EPI (kWh/m²)",
        color="Year",
        markers=True,
        title=f"{apartment_name} - Monthly Energy Performance Index (EPI)",
        labels={"EPI (kWh/m²)": "EPI (kWh/m²)", "YearMonth": "Month"}
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def plot_may_epi_overview(df_full, apartment_name, area_m2, historical_df):
    """
    Show May EPI for the current dataset and compare with historical May 2023 & 2024.

    - df_full: FULL apartment dataframe (NOT date-filtered)
    - apartment_name: display name
    - area_m2: apartment area in m²
    - historical_df: dataframe from load_historical_consumption()
    """
    if not area_m2:
        st.warning("Area not available – cannot compute EPI for this apartment.")
        return

    if "Timestamp" not in df_full.columns:
        st.warning("No timestamp column found – cannot compute EPI.")
        return

    energy_cols = get_energy_columns(df_full)
    if not energy_cols:
        st.warning("No appliance columns found – cannot compute EPI.")
        return

    epi_rows = []

    # -------- Current dataset: May EPI --------
    df_may = df_full[df_full["Timestamp"].dt.month == 5].copy()
    if not df_may.empty:
        df_may["Year"] = df_may["Timestamp"].dt.year

        may_totals = (
            df_may.groupby("Year")[energy_cols]
            .sum(numeric_only=True)
            .sum(axis=1) * INTERVAL_HOURS
        )

        current_year = int(may_totals.index.max())
        epi_current = may_totals.loc[current_year] / area_m2

        # Metric for May EPI (current year)
        st.metric(f"EPI (May {current_year})", f"{epi_current:.1f} kWh/m²")

        epi_rows.append({
            "Year": current_year,
            "EPI (kWh/m²)": epi_current,
            "Source": "Current"
        })
    else:
        st.info("No May data available in the current dataset for this apartment.")

    # -------- Historical May EPI: 2023 & 2024 --------
    if historical_df is not None and not historical_df.empty:
        key = normalize_name(apartment_name)
        hist_may = historical_df[
            (historical_df["key"] == key) &
            (historical_df["Month"] == 5) &
            (historical_df["Year"].isin([2023, 2024]))
        ].copy()

        if not hist_may.empty:
            hist_may["EPI (kWh/m²)"] = hist_may["Total_kWh"] / area_m2
            for _, row in hist_may.iterrows():
                epi_rows.append({
                    "Year": int(row["Year"]),
                    "EPI (kWh/m²)": row["EPI (kWh/m²)"],
                    "Source": "Historical"
                })

    if not epi_rows:
        st.info("No May EPI data available in historical or current dataset.")
        return

    epi_df = pd.DataFrame(epi_rows).sort_values("Year")

    fig = px.bar(
        epi_df,
        x="Year",
        y="EPI (kWh/m²)",
        color="Source",
        barmode="group",
        title=f"{apartment_name} – May EPI Comparison (Current vs 2023 & 2024)",
        labels={"Year": "Year", "EPI (kWh/m²)": "EPI (kWh/m²)"}
    )
    fig.update_layout(xaxis=dict(type="category"))
    st.plotly_chart(fig, use_container_width=True)


# ================= MAIN APP =================
def main():
    st.set_page_config(page_title="Apartment Energy Analytics", layout="wide")
    st.title("🏢 Apartment Energy Consumption Dashboard")

    with st.spinner("Loading apartment data..."):
        all_dfs = load_all_data()

    if not all_dfs:
        st.error("No data files found. Please check the DATA_FOLDER path.")
        st.stop()

    st.success(f"✅ Loaded data for {len(all_dfs)} apartments")
    areas = load_apartment_areas()
    historical_df = load_historical_consumption()

    # --- Sidebar Controls ---
    st.sidebar.header("Apartment Selection")
    apartment_names = sorted(list(all_dfs.keys()))
    selected_option = st.sidebar.selectbox(
        "Select Option", ["All Apartments"] + apartment_names)

    # --- Date Range Slicer ---
    st.sidebar.subheader("📆 Date Range Filter")
    all_timestamps = pd.concat(
        [df["Timestamp"] for df in all_dfs.values()
         if "Timestamp" in df.columns],
        ignore_index=True
    )
    all_timestamps = all_timestamps.dropna()
    if all_timestamps.empty:
        st.error("No valid timestamps found in the loaded data.")
        st.stop()

    min_date, max_date = all_timestamps.min().date(), all_timestamps.max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    else:
        start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

    # Optionally diagnose data
    if st.sidebar.checkbox("Show Data Diagnosis", False):
        diagnose_data_issues(all_dfs)

    # ================= ALL APARTMENTS VIEW =================
    if selected_option == "All Apartments":
        st.header("🏘️ All Apartments - Comparative Analysis")

        # Apply date filter to all apartments
        for apt, df_apt in all_dfs.items():
            if "Timestamp" in df_apt.columns:
                mask = (
                    (df_apt["Timestamp"] >= start_date) &
                    (df_apt["Timestamp"] <= end_date)
                )
                all_dfs[apt] = df_apt[mask]

        plot_hourly_consumption_all_apartments(all_dfs)
        plot_weekday_weekend_comparison_all_apartments(all_dfs)
        plot_epi_all_apartments(all_dfs, areas, start_date, end_date)

        return

    # ================= INDIVIDUAL APARTMENT VIEW =================
    selected_apartment = selected_option
    if selected_apartment not in all_dfs:
        st.error("Selected apartment data not found.")
        return

    # Full dataset for EPI calculations
    df_full = all_dfs[selected_apartment].copy()

    # Date-filtered dataset for plots/metrics
    df = df_full.copy()
    if "Timestamp" in df.columns:
        df = df[
            (df["Timestamp"] >= start_date) &
            (df["Timestamp"] <= end_date)
        ]

    st.sidebar.subheader("Apartment Info")
    st.sidebar.write(f"**Apartment:** {selected_apartment}")
    st.sidebar.write(f"**Filtered Records:** {len(df)}")
    if not df.empty and "Timestamp" in df.columns:
        st.sidebar.write(
            f"**Date Range:** {df['Timestamp'].min().strftime('%Y-%m-%d')} → "
            f"{df['Timestamp'].max().strftime('%Y-%m-%d')}"
        )

    st.header(f"Energy Analytics for {selected_apartment}")

    # ================= DASHBOARD TABS =================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview",
        "🔌 Appliance Usage",
        "📅 Time Analysis",
        "❄️ AC & Weather Analysis",
        "🔄 ON/OFF Events",
        "⏱️ Operation Duration",
        "🚨 Anomalies",
        # "📊 EPI Analysis"
    ])

    # --- TAB 1: Overview ---
    with tab1:
        st.subheader("Energy Overview")
        plot_energy_consumption_over_time(df, selected_apartment, "overview")

        appliance_cols = get_energy_columns(df)
        total_energy = (
            sum(df[c].sum() for c in appliance_cols) * INTERVAL_HOURS
            if appliance_cols else 0
        )
        avg_power = (
            np.mean([df[c].mean() for c in appliance_cols])
            if appliance_cols else 0
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Energy (kWh)", f"{total_energy:.2f}")
        with col2:
            st.metric("Average Power (kW)", f"{avg_power:.2f}")

        # ---- May EPI + comparison with historical May 2023 & 2024 ----
        apt_key = normalize_name(selected_apartment)
        area_m2 = areas.get(apt_key)
        plot_may_epi_overview(df_full, selected_apartment, area_m2, historical_df)

        col3, col4 = st.columns(2)
        with col3:
            plot_appliance_wise_energy(df, selected_apartment, "overview")
        with col4:
            plot_weekday_weekend_comparison(df, selected_apartment, "overview")

    # --- TAB 2: Appliance Usage ---
    with tab2:
        st.subheader("Detailed Appliance Analysis")
        plot_appliance_operation_heatmap(df, selected_apartment, "appliance_usage")

    # --- TAB 3: Time Analysis ---
    with tab3:
        st.subheader("Time-based Patterns")
        plot_weekday_weekend_comparison(df, selected_apartment, "time_analysis")
        #plot_hourly_profile(df, selected_apartment, "time_analysis")

    # --- TAB 4: AC & Weather Analysis ---
    with tab4:
        st.subheader("AC Power Patterns")
        plot_ac_energy_time_bins(df, selected_apartment, "ac_analysis")

        if selected_apartment.lower() == "lakeside k 502":
            st.subheader("🌦️ Weather-Linked Analysis")
            weather_dict = load_weather_data()
            if weather_dict:
                for zone, weather_df in weather_dict.items():
                    st.markdown(f"### 🏠 {zone} Bedroom")
                    plot_weather_correlation(df, weather_df)
            else:
                st.warning("No weather files found.")

    # --- TAB 5: ON/OFF Events ---
    with tab5:
        st.subheader("ON/OFF Events Detection")
        timeframe = st.radio("Select Timeframe", ["daily", "weekly"], horizontal=True)
        plot_on_off_occurrences(df, selected_apartment, timeframe, "on_off_events")

    # --- TAB 6: Operation Duration ---
    with tab6:
        st.subheader("Operation Duration Analysis")
        plot_operation_duration(df, selected_apartment, "operation_duration")

        stats = []
        for c in get_energy_columns(df):
            op_df = calculate_operation_duration_improved(df, c)
            if not op_df.empty:
                stats.append({
                    "Appliance": c,
                    "Avg Duration (h)": op_df["duration_hours"].mean(),
                    "Max Duration (h)": op_df["duration_hours"].max(),
                    "Events": len(op_df)
                })
        if stats:
            st.dataframe(pd.DataFrame(stats).round(2), use_container_width=True)

    # --- TAB 7: Anomalies ---
    with tab7:
        st.subheader("Operation Anomalies")
        anomalies = []
        for c in get_energy_columns(df):
            anomalies.extend(detect_anomalies_operation(df, c))

        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            st.metric("Total Anomalies", len(anomalies_df))
            for severity in ["High", "Medium", "Low"]:
                subset = anomalies_df[anomalies_df["severity"] == severity]
                if not subset.empty:
                    with st.expander(f"{severity} Severity ({len(subset)})"):
                        st.dataframe(
                            subset[['appliance', 'type', 'description', 'start_time']],
                            use_container_width=True
                        )
        else:
            st.success("✅ No anomalies detected!")

    # with tab8:
    #     st.subheader("Monthly Energy Performance Index (EPI)")
    #     apt_key = normalize_name(selected_apartment)
    #     area_m2 = areas.get(apt_key)
    #     if area_m2:
    #         plot_monthly_epi(df_full, selected_apartment, area_m2, historical_df)
    #     else:
    #         st.warning("Area not available — cannot compute EPI.")


if __name__ == "__main__":
    main()
