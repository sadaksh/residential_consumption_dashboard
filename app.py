import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
import numpy as np
from datetime import datetime
import warnings
import re

warnings.filterwarnings("ignore")

# ================= CONFIG =================
DATA_FOLDER = "processed_apartments2"

# 5-min interval power data (kW) → energy (kWh)
INTERVAL_MINUTES = 5
INTERVAL_HOURS = INTERVAL_MINUTES / 60.0  # 0.083333...

# ================= NAME HELPERS =================
def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

def strip_apartment_prefix(label: str, apartment_name: str = "") -> str:
    """
    Robustly remove apartment prefixes from device labels.

    Fixes the prefix error even when:
    - apartment_name is an anonymised RMI code (doesn't match raw prefix),
    - label uses real apartment name prefix,
    - separators vary: " - ", "-", "–", "—", ":", "|".

    Examples:
    "Aurora C 102 - Washing Machine" -> "Washing Machine"
    "Cielo F106- Living AC" -> "Living AC"
    "LAKNW05 - Master AC" -> "Master AC"
    """
    if not isinstance(label, str):
        return label

    s = label.strip()

    # If any common separator exists, keep everything after the first separator
    parts = re.split(r"\s*[-–—:|]\s*", s, maxsplit=1)
    if len(parts) == 2:
        s = parts[1].strip()

    # cleanup any leading separators
    s = re.sub(r"^\s*[-–—:|]+\s*", "", s).strip()
    return s

# ================= AC DETECTION (edge case safe) =================
FORBIDDEN_AC_SUBSTRINGS = [
    "washing", "washer", "wm", "machine",
    "fridge", "refrigerator",
    "geyser",
    "microwave", "oven",
    "light", "lighting",
    "fan", "fans",
    "socket", "plug"
]

def is_ac_column(name: str) -> bool:
    n = str(name).lower()
    if any(f in n for f in FORBIDDEN_AC_SUBSTRINGS):
        return False
    if "air conditioner" in n:
        return True
    clean = re.sub(r"[^a-z0-9]+", " ", n)
    tokens = clean.split()
    return ("ac" in tokens) or ("a/c" in tokens)

# ================= COLUMN HELPERS =================
def get_energy_columns(df: pd.DataFrame) -> list:
    """
    Returns likely appliance power columns (kW).
    Excludes meta + phase columns.
    """
    exclude_keys = [
        "timestamp", "apartment", "phase", "date", "time",
        "b phase", "r phase", "y phase",
        "device_id", "ts"
    ]
    cols = []
    for c in df.columns:
        cname = c.lower().strip()
        if not any(k in cname for k in exclude_keys):
            cols.append(c)
    return cols

def get_nonzero_appliance_columns(df: pd.DataFrame, eps_kwh: float = 0.01) -> list:
    """
    Hide zero-usage appliances:
    keep columns whose total energy over filtered range is > eps_kwh.
    """
    cols = get_energy_columns(df)
    keep = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        total_kwh = s.fillna(0).sum() * INTERVAL_HOURS
        if total_kwh > eps_kwh:
            keep.append(c)
    return keep

# ================= RESIDENT INSIGHTS =================
def fmt_period(hour: int) -> str:
    if 0 <= hour < 6:
        return "Early morning (12 AM–6 AM)"
    if 6 <= hour < 10:
        return "Breakfast (6 AM–10 AM)"
    if 10 <= hour < 15:
        return "Lunch (10 AM–3 PM)"
    if 15 <= hour < 19:
        return "Evening (3 PM–7 PM)"
    return "Dinner & late night (7 PM–12 AM)"

def insight_top_appliance_share(df: pd.DataFrame, apartment_name: str) -> str:
    cols = get_nonzero_appliance_columns(df)
    if not cols:
        return "No measurable appliance usage in the selected period."
    energy = {c: df[c].sum() * INTERVAL_HOURS for c in cols}
    total = sum(energy.values())
    top = max(energy, key=energy.get)
    share = (energy[top] / total) * 100 if total > 0 else 0
    return f"Most of your electricity in this period comes from **{strip_apartment_prefix(top, apartment_name)}** (~{share:.0f}% of total)."

def insight_weekday_weekend(df: pd.DataFrame) -> str:
    cols = get_nonzero_appliance_columns(df)
    if "Timestamp" not in df.columns or not cols:
        return "Not enough data to compare weekdays vs weekends."
    d = df.dropna(subset=["Timestamp"]).copy()
    d["day_type"] = d["Timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    d["Total Power (kW)"] = d[cols].sum(axis=1)
    wk = d[d["day_type"] == "Weekday"]["Total Power (kW)"].mean()
    we = d[d["day_type"] == "Weekend"]["Total Power (kW)"].mean()
    if np.isnan(wk) or np.isnan(we) or wk == 0:
        return "Not enough data to compare weekdays vs weekends."
    delta = (we - wk) / wk * 100
    if delta >= 0:
        return f"Your home is **more active on weekends** (~{delta:.0f}% higher average usage than weekdays)."
    return f"Your home is **less active on weekends** (~{abs(delta):.0f}% lower average usage than weekdays)."

def insight_peak_hour(df: pd.DataFrame) -> str:
    cols = get_nonzero_appliance_columns(df)
    if "Timestamp" not in df.columns or not cols:
        return "Not enough data to identify peak usage hours."
    d = df.dropna(subset=["Timestamp"]).copy()
    d["hour"] = d["Timestamp"].dt.hour
    d["Total Power (kW)"] = d[cols].sum(axis=1)
    hourly = d.groupby("hour")["Total Power (kW)"].mean()
    if hourly.empty:
        return "Not enough data to identify peak usage hours."
    h = int(hourly.idxmax())
    return f"Your **highest average electricity use** happens in **{fmt_period(h)}**."

# ================= APARTMENT ANONYMISATION =================
ANON_MAP = {
    "Cielo F106": "CIFNW01",
    "Urabno Q606": "URQSE06",
    "Aurelia F204": "AUFS02",
    "Elite A401": "ELANW04",
    "Lagoona F402": "LAFNW04",
    "Lakeside B 1203": "LABNW12",
    "Lakeside I 1001": "LAINW10",
    "Lakeside K 502": "LAKNW05",
    "Elite I 901": "ELINW09",
    "Lagoona B104": "LABNE01",
    "Lakeside G 1203(2)": "LABNW12",
    "Lakeside O 1106": "LAONW11",
    "Uno H 505": "UNHNW05",
    "Elite E 1006": "ELENW10",
    "Viento A 1102": "VIANW11",
    "Clara G 1404": "CLGSE14",
    "Urbano M501": "URMSE05",
    "Estella E201": "ESESE02",
    "Marvella G1403": "MAGSE14",
    "Urbano E8022": "URESE08",
    "Urbano J005": "URJSE05",
    "Eviva L1702": "EVLSE17",
    "Marvella A1801": "MAASE18",
    "Uno H1602": "UNHNW16",
    "Uno K 1505": "UNKNW15",
    "Urbano J1601": "URJSE16",
    "Adriana N1204": "ADNSE12",
    "Aurora D604": "AUDSE06",
    "Lakeside A 701": "LAANW07",
    "Aurora C 102": "AUCSE01",
    "Estella E 1106": "ESESE11",
    "Fresca C909": "FRCNE09",
    "Lakeside B 101": "LABNW01",
    "Uno M404": "UNMNW04"
}

def anonymise(apt_name: str) -> str:
    """Return the anonymised RMI code for an apartment name."""
    return ANON_MAP.get(apt_name, apt_name)

# ================= LOAD DATA =================
@st.cache_data
def load_all_data():
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    dfs = {}

    for f in all_files:
        try:
            df = pd.read_csv(f)
            apt_name = os.path.splitext(os.path.basename(f))[0]
            df["Apartment"] = anonymise(apt_name)

            # remove phase columns
            df = df[[c for c in df.columns if not any(k in c.lower() for k in ["b phase", "r phase", "y phase"])]]

            # timestamp
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

            # numeric conversion + abs
            energy_cols = get_energy_columns(df)
            if energy_cols:
                df[energy_cols] = (
                    df[energy_cols]
                    .apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce"))
                    .abs()
                )

            dfs[anonymise(apt_name)] = df

        except Exception as e:
            st.warning(f"Could not load {f}: {e}")

    return dfs

# ================= WEATHER (unchanged) =================
@st.cache_data
def load_weather_data(data_folder="processed_apartments2"):
    weather_files = {
        "Living": [
            os.path.join(data_folder, "Ambient Temperature & Humidity LakesideK502_Living Bedroom.xlsx"),
            "Ambient Temperature & Humidity LakesideK502_Living Bedroom.xlsx",
        ],
        "Master": [
            os.path.join(data_folder, "Ambient Temperature & Humidity LakesideK502_Master Bedroom.xlsx"),
            "Ambient Temperature & Humidity LakesideK502_Master Bedroom.xlsx",
        ],
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
                time_col = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
                if not time_col:
                    continue
                df.rename(columns={time_col[0]: "Timestamp"}, inplace=True)
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                df = df[["Timestamp"] + numeric_cols]
                df["Source"] = os.path.basename(f)
                dfs.append(df)
            except Exception as e:
                st.warning(f"Could not load {zone} file {f}: {e}")

        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged = merged.groupby("Timestamp").mean(numeric_only=True).reset_index()
            rename_map = {c: c.strip().title() for c in merged.columns}
            merged.rename(columns=rename_map, inplace=True)
            results[zone] = merged
    return results

def plot_weather_correlation(df_energy, df_weather):
    if df_weather.empty:
        st.warning("⚠️ No weather data available.")
        return

    merged = pd.merge_asof(
        df_energy.sort_values("Timestamp"),
        df_weather.sort_values("Timestamp"),
        on="Timestamp",
        tolerance=pd.Timedelta("15min"),
        direction="nearest",
    )

    ac_cols = [c for c in merged.columns if is_ac_column(c)]
    if not ac_cols:
        st.warning("No AC column found.")
        return
    ac_col = ac_cols[0]

    temp_candidates = [c for c in merged.columns if "temp" in c.lower()]
    hum_candidates = [c for c in merged.columns if "humid" in c.lower() or "rh" in c.lower()]
    if not temp_candidates:
        st.warning("No temperature column found.")
        return

    temp_col = temp_candidates[0]
    color_col = hum_candidates[0] if hum_candidates else None

    st.caption("Insight: Hotter outdoor conditions usually increase AC power use—this helps explain high-cooling days.")

    fig = px.scatter(
        merged,
        x=temp_col,
        y=ac_col,
        color=color_col,
        #trendline="ols",
        title="AC Power (kW) vs Temperature (°C)",
        labels={temp_col: "Temperature (°C)", ac_col: "AC Power (kW)"},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    merged["Date"] = merged["Timestamp"].dt.date
    daily = merged.groupby("Date").agg({ac_col: "sum", temp_col: "mean"}).reset_index()
    daily["AC Energy (kWh)"] = daily[ac_col] * INTERVAL_HOURS

    BASE_TEMP = 24
    daily["CDD"] = (daily[temp_col] - BASE_TEMP).clip(lower=0)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=daily["Date"], y=daily["CDD"], name="Cooling Degree Days", opacity=0.6), secondary_y=False)
    fig2.add_trace(
        go.Scatter(x=daily["Date"], y=daily["AC Energy (kWh)"], name="Daily AC Energy (kWh)", mode="lines+markers"),
        secondary_y=True,
    )
    fig2.update_layout(title="Daily AC Energy (kWh) vs Cooling Degree Days", xaxis_title="Date")
    fig2.update_yaxes(rangemode="tozero", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

# ================= ON/OFF + OP DURATION HELPERS (kept, with y-axis fixes) =================
def calculate_operation_duration_improved(df, appliance_col, threshold=0.001):
    if appliance_col not in df.columns:
        return pd.DataFrame()

    data = df[["Timestamp", appliance_col]].dropna().sort_values("Timestamp")
    if data.empty:
        return pd.DataFrame()

    data["is_on"] = data[appliance_col] > threshold
    data["state_change"] = data["is_on"].ne(data["is_on"].shift())
    data["group_id"] = data["state_change"].cumsum()

    events = []
    for _, group in data.groupby("group_id"):
        if group["is_on"].iloc[0]:
            start_time = group["Timestamp"].iloc[0]
            end_time = group["Timestamp"].iloc[-1]
            duration_hours = (end_time - start_time).total_seconds() / 3600
            if duration_hours >= INTERVAL_HOURS:
                events.append(
                    {
                        "appliance": appliance_col,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": duration_hours,
                        "avg_power": group[appliance_col].mean(),
                        "max_power": group[appliance_col].max(),
                        "day_of_week": start_time.strftime("%A"),
                    }
                )
    return pd.DataFrame(events)

def detect_on_off_events(df, appliance_col, threshold=0.01):
    if appliance_col not in df.columns:
        return pd.DataFrame()
    data = df[["Timestamp", appliance_col]].dropna().sort_values("Timestamp")
    if data.empty:
        return pd.DataFrame()
    data["state"] = data[appliance_col] > threshold
    data["state_change"] = data["state"].ne(data["state"].shift())
    data["event_id"] = data["state_change"].cumsum()
    return data[data["state_change"] & data["state"]]

# ================= COMPARATIVE PLOTS (ALL APARTMENTS) =================
def plot_hourly_consumption_all_apartments(all_dfs):
    hourly_data = []
    for apt_name, df in all_dfs.items():
        if "Timestamp" not in df.columns:
            continue
        appliance_cols = get_nonzero_appliance_columns(df)
        if not appliance_cols:
            continue
        df = df.dropna(subset=["Timestamp"]).copy()
        df["hour"] = df["Timestamp"].dt.hour
        hourly_avg = df.groupby("hour")[appliance_cols].mean().mean(axis=1)
        for hour, consumption in hourly_avg.items():
            hourly_data.append({"Apartment": apt_name, "Hour": hour, "Average Power (kW)": consumption})

    if not hourly_data:
        st.warning("No hourly data available")
        return

    st.caption("Insight: This compares when homes are most active during the day (higher dots = higher typical usage).")

    hourly_df = pd.DataFrame(hourly_data)
    fig = px.scatter(
        hourly_df,
        x="Hour",
        y="Average Power (kW)",
        color="Apartment",
        title="Average Power by Hour of Day (kW)",
        labels={"Hour": "Hour of Day", "Average Power (kW)": "Average Power (kW)"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

def plot_weekday_weekend_comparison_all_apartments(all_dfs):
    comparison_data = []
    for apt_name, df in all_dfs.items():
        if "Timestamp" not in df.columns:
            continue
        appliance_cols = get_nonzero_appliance_columns(df)
        if not appliance_cols:
            continue
        d = df.dropna(subset=["Timestamp"]).copy()
        d["day_type"] = d["Timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
        daily_mean = d.groupby("day_type")[appliance_cols].mean().mean(axis=1)
        for day_type, val in daily_mean.items():
            comparison_data.append({"Apartment": apt_name, "Day Type": day_type, "Average Power (kW)": val})

    if not comparison_data:
        st.warning("No weekday/weekend data available")
        return

    st.caption("Insight: Weekend vs weekday usage differences often reflect time spent at home and AC/plug load behaviour.")

    comp_df = pd.DataFrame(comparison_data)
    fig = px.scatter(
        comp_df,
        x="Day Type",
        y="Average Power (kW)",
        color="Apartment",
        title="Weekday vs Weekend - Average Power (kW)",
        labels={"Average Power (kW)": "Average Power (kW)"},
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

# ================= INDIVIDUAL APARTMENT PLOTS =================
def plot_energy_consumption_over_time(df, apartment_name, tab_name):
    """
    Instantaneous power time-series (kW) — NOT kWh.
    Prefix removal is applied to ALL appliance labels.
    """
    appliance_cols = get_nonzero_appliance_columns(df)
    if not appliance_cols:
        st.warning("No measurable appliance usage found for this apartment in the selected period.")
        return

    st.caption(insight_peak_hour(df))

    melt_df = df.melt(id_vars=["Timestamp"], value_vars=appliance_cols, var_name="Appliance", value_name="Power (kW)")
    melt_df = melt_df.dropna(subset=["Power (kW)"])
    melt_df["Appliance"] = melt_df["Appliance"].apply(lambda x: strip_apartment_prefix(x, apartment_name))

    fig = px.line(
        melt_df,
        x="Timestamp",
        y="Power (kW)",
        color="Appliance",
        title=f"{apartment_name} - Power Over Time (kW)",
        labels={"Power (kW)": "Power (kW)", "Timestamp": "Time"},
    )
    fig.update_layout(height=500, showlegend=True, legend_title_text="Appliance")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_energy_time_{tab_name}")

def plot_appliance_wise_energy(df, apartment_name, tab_name):
    """
    Total consumption by appliance (kWh) — hides zero-usage appliances and cleans labels.
    """
    appliance_cols = get_nonzero_appliance_columns(df)
    if not appliance_cols:
        st.info("No measurable appliance energy to display for this period.")
        return

    st.caption(insight_top_appliance_share(df, apartment_name))

    totals = []
    for col in appliance_cols:
        totals.append(
            {
                "Appliance": strip_apartment_prefix(col, apartment_name),
                "Total Consumption (kWh)": df[col].sum() * INTERVAL_HOURS,
            }
        )
    energy_df = pd.DataFrame(totals).sort_values("Total Consumption (kWh)", ascending=False)

    fig = px.bar(
        energy_df,
        x="Appliance",
        y="Total Consumption (kWh)",
        title=f"{apartment_name} - Appliance-wise Total Consumption (kWh)",
        labels={"Total Consumption (kWh)": "Total Consumption (kWh)"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_appliance_wise_{tab_name}")

def plot_weekday_weekend_comparison(df, apartment_name, tab_name):
    """
    Average power by appliance (kW), weekdays vs weekends.
    Cleans appliance labels via strip_apartment_prefix().
    """
    appliance_cols = get_nonzero_appliance_columns(df)
    if not appliance_cols or "Timestamp" not in df.columns:
        st.info("Not enough data for weekday/weekend comparison.")
        return

    st.caption(insight_weekday_weekend(df))

    data = df.dropna(subset=["Timestamp"]).copy()
    data["day_type"] = data["Timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")

    comparison_data = []
    for col in appliance_cols:
        weekday_avg = data[data["day_type"] == "Weekday"][col].mean()
        weekend_avg = data[data["day_type"] == "Weekend"][col].mean()
        if not np.isnan(weekday_avg) and not np.isnan(weekend_avg):
            short = strip_apartment_prefix(col, apartment_name)
            comparison_data.extend(
                [
                    {"Appliance": short, "Day Type": "Weekday", "Average Power (kW)": weekday_avg},
                    {"Appliance": short, "Day Type": "Weekend", "Average Power (kW)": weekend_avg},
                ]
            )

    comp_df = pd.DataFrame(comparison_data)
    if comp_df.empty:
        st.info("No weekday/weekend comparison data after filtering zero-usage appliances.")
        return

    fig = px.bar(
        comp_df,
        x="Appliance",
        y="Average Power (kW)",
        color="Day Type",
        barmode="group",
        title=f"{apartment_name} - Weekday vs Weekend Power (kW)",
        labels={"Average Power (kW)": "Average Power (kW)"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500, legend_title_text="Day Type")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_weekday_weekend_{tab_name}")

def plot_hourly_profile(df, apartment_name, tab_name):
    """
    Average total power (kW) by hour (single apartment).
    """
    if "Timestamp" not in df.columns or df.empty:
        st.info("No timestamped data available for hourly profile.")
        return

    cols = get_nonzero_appliance_columns(df)
    if not cols:
        st.info("No measurable appliance usage for hourly profile.")
        return

    st.caption(insight_peak_hour(df))

    data = df.dropna(subset=["Timestamp"]).copy()
    data["hour"] = data["Timestamp"].dt.hour
    data["Total Power (kW)"] = data[cols].sum(axis=1)
    hourly = data.groupby("hour")["Total Power (kW)"].mean().reset_index().sort_values("hour")

    fig = px.line(
        hourly,
        x="hour",
        y="Total Power (kW)",
        markers=True,
        title=f"{apartment_name} – Average Power by Hour of Day (kW)",
        labels={"hour": "Hour of Day", "Total Power (kW)": "Average Power (kW)"},
    )
    fig.update_layout(xaxis=dict(dtick=1))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_hourly_profile_{tab_name}")

def plot_appliance_operation_heatmap(df, apartment_name, tab_name):
    """
    Heatmap: appliance "working harder" occurrences by hour/day (resident-friendly)
    - hides zero-usage appliances
    - removes prefixes in appliance names for titles/labels
    """
    cols = get_nonzero_appliance_columns(df)
    if not cols or "Timestamp" not in df.columns:
        st.info("No appliance operation pattern available.")
        return

    # Top 5 by energy (kWh)
    totals = {c: df[c].sum() * INTERVAL_HOURS for c in cols}
    top_appliances = sorted(totals, key=totals.get, reverse=True)[:5]

    st.caption("Insight: Darker cells show periods when an appliance is active more often (working harder).")

    for i, appliance in enumerate(top_appliances):
        data = df[["Timestamp", appliance]].copy()
        data = data.dropna(subset=["Timestamp"])
        data = data[data[appliance] > 0.01]
        if data.empty:
            continue

        data["hour"] = data["Timestamp"].dt.hour
        data["day"] = data["Timestamp"].dt.day_name()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = data.groupby(["day", "hour"]).size().unstack(fill_value=0).reindex(days_order)

        short_name = strip_apartment_prefix(appliance, apartment_name)
        fig = px.imshow(
            pivot,
            title=f"When {short_name} works the hardest during the week",
            labels=dict(x="Hour of Day", y="Day of Week", color="Working harder"),
            aspect="auto",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_heatmap_{i}_{tab_name}")

def plot_on_off_occurrences(df, apartment_name, timeframe="daily", tab_name=""):
    """
    ON events:
    - removes prefixes in legend labels
    - hides zero-usage appliances before event detection
    - y-axis forced to zero
    """
    cols = get_nonzero_appliance_columns(df)
    if not cols:
        st.info("No measurable appliance usage for event detection.")
        return

    st.caption("Insight: This shows how often appliances turn ON (useful to spot frequent cycling).")

    occurrences = []
    for appliance in cols:
        events = detect_on_off_events(df, appliance)
        if events.empty:
            continue

        clean_name = strip_apartment_prefix(appliance, apartment_name)

        if timeframe == "daily":
            events = events.copy()
            events["date"] = events["Timestamp"].dt.date
            daily = events.groupby("date").size().reset_index(name="ON_Events")
            for _, row in daily.iterrows():
                occurrences.append({"Date": row["date"], "Appliance": clean_name, "ON_Events": row["ON_Events"]})
        else:
            events = events.copy()
            events["week"] = events["Timestamp"].dt.isocalendar().week
            events["year"] = events["Timestamp"].dt.year
            weekly = events.groupby(["year", "week"]).size().reset_index(name="ON_Events")
            for _, row in weekly.iterrows():
                occurrences.append(
                    {"Week": f"{row['year']}-W{row['week']}", "Appliance": clean_name, "ON_Events": row["ON_Events"]}
                )

    if not occurrences:
        st.info("No ON events detected after filtering.")
        return

    occ_df = pd.DataFrame(occurrences)

    if timeframe == "daily":
        fig = px.line(
            occ_df,
            x="Date",
            y="ON_Events",
            color="Appliance",
            title=f"{apartment_name} - Daily ON Events",
            labels={"ON_Events": "Times turned ON (count)"},
        )
    else:
        fig = px.bar(
            occ_df,
            x="Week",
            y="ON_Events",
            color="Appliance",
            title=f"{apartment_name} - Weekly ON Events",
            barmode="group",
            labels={"ON_Events": "Times turned ON (count)"},
        )

    fig.update_layout(height=500, legend_title_text="Appliance")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_on_off_{timeframe}_{tab_name}")

# ================= EPI (kept) =================
@st.cache_data
def load_apartment_areas(path="processed_apartments2/metadata/apartment_area.csv"):
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

def plot_epi(selected_apartment, areas, df_full):
    """
    Computes and plots EPI (kWh/m²) using ONLY the loaded apartment dataset.
    """

    if df_full.empty or "Timestamp" not in df_full.columns:
        st.warning("No timestamped data available for EPI calculation.")
        return

    REAL_NAME_MAP = {v: k for k, v in ANON_MAP.items()}
    if selected_apartment not in REAL_NAME_MAP:
        st.error(f"No mapping found for anonymised apartment '{selected_apartment}'.")
        return

    real_name = REAL_NAME_MAP[selected_apartment]
    real_key = normalize_name(real_name)

    apt_area = areas.get(real_key)
    if apt_area is None:
        st.warning(f"Area data not available for apartment {selected_apartment}. EPI normalisation may be approximate.")
        return

    appliance_cols = get_nonzero_appliance_columns(df_full)
    if not appliance_cols:
        st.warning("No measurable appliance energy available for EPI calculation.")
        return

    df = df_full.dropna(subset=["Timestamp"]).copy()
    df["MonthStart"] = df["Timestamp"].dt.to_period("M").dt.to_timestamp()
    df["TotalPower_kW"] = df[appliance_cols].sum(axis=1)

    monthly = (
        df.groupby("MonthStart", as_index=False)["TotalPower_kW"]
        .sum()
        .rename(columns={"TotalPower_kW": "TotalPower_kW_Sum"})
    )
    monthly["Total_kWh"] = monthly["TotalPower_kW_Sum"] * INTERVAL_HOURS
    monthly["EPI_kWh_per_m2"] = monthly["Total_kWh"] / apt_area

    if monthly.empty:
        st.warning("No monthly EPI values could be computed.")
        return

    monthly = monthly.sort_values("MonthStart")
    monthly["MonthLabel"] = monthly["MonthStart"].dt.strftime("%b %Y")

    first = monthly["MonthStart"].min().strftime("%b %Y")
    last = monthly["MonthStart"].max().strftime("%b %Y")
    st.info(f"📅 **Data available from {first} to {last}**")

    month_labels = monthly["MonthLabel"].tolist()
    selected_month = st.selectbox("Select Month for EPI", ["All Months"] + month_labels)

    if selected_month == "All Months":
        df_plot = monthly
        title = f"EPI Across All Months ({selected_apartment})"
    else:
        df_plot = monthly[monthly["MonthLabel"] == selected_month]
        title = f"EPI for {selected_month} ({selected_apartment})"

    fig = px.bar(
        df_plot,
        x="MonthLabel",
        y="EPI_kWh_per_m2",
        title=title,
        labels={"MonthLabel": "Month", "EPI_kWh_per_m2": "EPI (kWh/m²)"},
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(xaxis=dict(type="category"))
    st.plotly_chart(fig, use_container_width=True)

    if selected_month == "All Months":
        peak = df_plot.loc[df_plot["EPI_kWh_per_m2"].idxmax()]
        st.caption(
            f"💡 **Insight:** Your highest EPI was in **{peak['MonthLabel']}**, indicating higher energy use per square metre."
        )
    else:
        val = float(df_plot["EPI_kWh_per_m2"].iloc[0])
        st.caption(f"💡 **Insight:** Your EPI for **{selected_month}** is **{val:.2f} kWh/m²**.")

# ================= MAIN APP =================
def main():
    st.set_page_config(page_title="Apartment Energy Analytics", layout="wide")
    st.title("🏢 Residential Energy Consumption Dashboard")

    with st.spinner("Loading apartment data..."):
        all_dfs = load_all_data()

    if not all_dfs:
        st.error("No data files found. Please check the DATA_FOLDER path.")
        st.stop()

    st.success(f"✅ Loaded data for {len(all_dfs)} apartments")

    areas = load_apartment_areas()
    historical_df = load_historical_consumption()

    st.sidebar.header("Apartment Selection")
    apartment_names = sorted(list(all_dfs.keys()))
    selected_option = st.sidebar.selectbox("Select Option", ["All Apartments"] + apartment_names)

    st.sidebar.subheader("📆 Date Range Filter")
    all_timestamps = pd.concat([df["Timestamp"] for df in all_dfs.values() if "Timestamp" in df.columns], ignore_index=True)
    all_timestamps = all_timestamps.dropna()
    if all_timestamps.empty:
        st.error("No valid timestamps found in the loaded data.")
        st.stop()

    min_date, max_date = all_timestamps.min().date(), all_timestamps.max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    else:
        start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

    # ================= ALL APARTMENTS VIEW =================
    if selected_option == "All Apartments":
        st.header("🏘️ All Apartments - Comparative Analysis")

        for apt, df_apt in all_dfs.items():
            if "Timestamp" in df_apt.columns:
                mask = (df_apt["Timestamp"] >= start_date) & (df_apt["Timestamp"] <= end_date)
                all_dfs[apt] = df_apt[mask]

        st.caption("Insight: Use this view to compare typical daily patterns and weekend effects across homes.")
        plot_hourly_consumption_all_apartments(all_dfs)
        plot_weekday_weekend_comparison_all_apartments(all_dfs)
        return

    # ================= INDIVIDUAL APARTMENT VIEW =================
    selected_apartment = selected_option
    if selected_apartment not in all_dfs:
        st.error("Selected apartment data not found.")
        return

    df_full = all_dfs[selected_apartment].copy()
    df = df_full.copy()
    if "Timestamp" in df.columns:
        df = df[(df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)]

    st.sidebar.subheader("Apartment Info")
    st.sidebar.write(f"**Apartment:** {selected_apartment}")
    st.sidebar.write(f"**Filtered Records:** {len(df)}")
    if not df.empty and "Timestamp" in df.columns:
        st.sidebar.write(f"**Date Range:** {df['Timestamp'].min().strftime('%Y-%m-%d')} → {df['Timestamp'].max().strftime('%Y-%m-%d')}")

    st.header(f"Energy Analytics for {selected_apartment}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📈 Overview", "🔌 Appliance Usage", "📅 Time Analysis", "❄️ AC & Weather", "🔄 ON Events", "📊 EPI Analysis"]
    )

    with tab1:
        st.subheader("Energy Overview")
        plot_energy_consumption_over_time(df, selected_apartment, "overview")

        cols = get_nonzero_appliance_columns(df)
        total_kwh = df[cols].sum(numeric_only=True).sum() * INTERVAL_HOURS if cols else 0
        avg_power_kw = df[cols].sum(axis=1).mean() if cols else 0

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Consumption (kWh)", f"{total_kwh:.2f}")
        with c2:
            st.metric("Average Power (kW)", f"{avg_power_kw:.2f}")

        colA, colB = st.columns(2)
        with colA:
            plot_appliance_wise_energy(df, selected_apartment, "overview")
        with colB:
            plot_weekday_weekend_comparison(df, selected_apartment, "overview")

    with tab2:
        st.subheader("Appliance Operation Patterns")
        plot_appliance_operation_heatmap(df, selected_apartment, "appliance_usage")

    with tab3:
        st.subheader("Time-based Patterns")
        plot_hourly_profile(df, selected_apartment, "time_analysis")
        plot_weekday_weekend_comparison(df, selected_apartment, "time_analysis")

    with tab4:
        st.subheader("AC & Weather")
        st.caption("Insight: This helps explain why AC energy rises on hotter or more humid days.")
        RMI_LAKESIDE_CODE = ANON_MAP.get("Lakeside K 502", "LAKNW05")
        if selected_apartment.strip().lower() == RMI_LAKESIDE_CODE.strip().lower():
            weather_dict = load_weather_data()
            if weather_dict:
                for zone, weather_df in weather_dict.items():
                    safe_zone = str(zone).replace("Lakeside K 502", RMI_LAKESIDE_CODE)
                    st.markdown(f"### 🏠 {safe_zone} Bedroom")
                    plot_weather_correlation(df, weather_df)
            else:
                st.warning("No weather files found for this apartment.")
        else:
            st.info(f"Weather files are configured only for apartment {RMI_LAKESIDE_CODE} in the current build.")

    with tab5:
        st.subheader("Appliance ON Events (Count)")
        timeframe = st.radio("Select Timeframe", ["daily", "weekly"], horizontal=True)
        plot_on_off_occurrences(df, selected_apartment, timeframe, "on_events")

    with tab6:
        st.subheader("Energy Performance Index (EPI)")
        plot_epi(selected_apartment, areas, df)

if __name__ == "__main__":
    main()
