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

# Default thresholds
ACTIVE_POWER_KW = 0.01  # 10 W treated as "active" (standby below this is ignored)

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

    parts = re.split(r"\s*[-–—:|]\s*", s, maxsplit=1)
    if len(parts) == 2:
        s = parts[1].strip()

    s = re.sub(r"^\s*[-–—:|]+\s*", "", s).strip()
    return s

# ================= RESIDENT OUTPUT HELPERS =================
def resident_insight(text: str):
    """One-line resident-friendly insight. Keep it behavioural."""
    st.caption(f"💡 {text}")

def time_block_from_hour(hour: int) -> str:
    if 0 <= hour < 6:
        return "Early morning (12 AM–6 AM)"
    if 6 <= hour < 10:
        return "Breakfast (6 AM–10 AM)"
    if 10 <= hour < 15:
        return "Lunch (10 AM–3 PM)"
    if 15 <= hour < 19:
        return "Evening (3 PM–7 PM)"
    return "Dinner & late night (7 PM–12 AM)"

TIME_BLOCK_ORDER = [
    "Early morning (12 AM–6 AM)",
    "Breakfast (6 AM–10 AM)",
    "Lunch (10 AM–3 PM)",
    "Evening (3 PM–7 PM)",
    "Dinner & late night (7 PM–12 AM)",
]

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
    "Uno M404": "UNMNW04",
}

def anonymise(apt_name: str) -> str:
    """Return the anonymised RMI code for an apartment name."""
    return ANON_MAP.get(apt_name, apt_name)

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

# ================= APPLIANCE CATEGORISATION (client-required) =================
CATEGORIES = ["ACs", "Lights + Fans + Plug Loads", "Geysers", "Kitchen Loads"]

KITCHEN_KEYS = [
    "fridge", "refrigerator", "microwave", "oven", "dishwasher", "mixer",
    "kitchen", "chimney", "hob", "induction", "toaster", "kettle"
]
LIGHT_FAN_PLUG_KEYS = [
    "light", "lighting", "lamp", "fan", "socket", "plug", "tv", "router",
    "computer", "laptop", "charger", "set top", "stb"
]
GEYSER_KEYS = ["geyser", "water heater", "heater"]

def categorise_appliance(raw_col: str, apartment_name: str = "") -> str:
    """
    Map a raw metering channel to one of the 4 resident-facing categories.
    Any unclassified device is treated as Plug Loads (resident-relevant default).
    """
    device = strip_apartment_prefix(raw_col, apartment_name)
    n = str(device).lower()

    if is_ac_column(device):
        return "ACs"
    if any(k in n for k in GEYSER_KEYS):
        return "Geysers"
    if any(k in n for k in KITCHEN_KEYS):
        return "Kitchen Loads"
    if any(k in n for k in LIGHT_FAN_PLUG_KEYS):
        return "Lights + Fans + Plug Loads"

    # Default bucket to avoid "Other" clutter in resident view
    return "Lights + Fans + Plug Loads"

def aggregate_to_categories(df: pd.DataFrame, apartment_name: str, eps_kwh: float = 0.01) -> pd.DataFrame:
    """
    Return a dataframe with Timestamp + 4 category power columns (kW).
    Drops categories with zero energy in the selected period.
    """
    if df.empty or "Timestamp" not in df.columns:
        return pd.DataFrame()

    cols = get_nonzero_appliance_columns(df, eps_kwh=eps_kwh)
    if not cols:
        return pd.DataFrame()

    out = pd.DataFrame({"Timestamp": df["Timestamp"]})
    for cat in CATEGORIES:
        out[cat] = 0.0

    for c in cols:
        cat = categorise_appliance(c, apartment_name)
        out[cat] = out[cat] + pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # drop zero-energy categories
    keep = ["Timestamp"]
    for cat in CATEGORIES:
        if (out[cat].fillna(0.0).sum() * INTERVAL_HOURS) > eps_kwh:
            keep.append(cat)
    out = out[keep]

    return out

def category_totals_kwh(df_cat: pd.DataFrame) -> pd.DataFrame:
    """Return totals (kWh) per category."""
    if df_cat.empty:
        return pd.DataFrame(columns=["Category", "Total Consumption (kWh)"])
    cats = [c for c in df_cat.columns if c != "Timestamp"]
    rows = [{"Category": c, "Total Consumption (kWh)": float(df_cat[c].fillna(0).sum() * INTERVAL_HOURS)} for c in cats]
    return pd.DataFrame(rows).sort_values("Total Consumption (kWh)", ascending=False)

# ================= LOAD DATA =================
@st.cache_data
def load_all_data():
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    dfs = {}

    for f in all_files:
        try:
            df = pd.read_csv(f)
            apt_name = os.path.splitext(os.path.basename(f))[0]

            # Anonymise immediately (display and internal)
            apt_code = anonymise(apt_name)
            df["Apartment"] = apt_code

            # remove phase columns
            df = df[[c for c in df.columns if not any(k in c.lower() for k in ["b phase", "r phase", "y phase"]) ]]

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

            dfs[apt_code] = df

        except Exception as e:
            st.warning(f"Could not load {f}: {e}")

    return dfs

# ================= WEATHER (kept, UI anonymised) =================
@st.cache_data
def load_weather_data(data_folder="processed_apartments2"):
    """
    Weather files currently available for one flat only (file names are fixed).
    Returns a dict: zone -> dataframe
    """
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

def plot_weather_correlation(df_energy, df_weather, apartment_code: str):
    if df_weather.empty:
        st.warning("No weather data available.")
        return

    if df_energy.empty or "Timestamp" not in df_energy.columns:
        st.warning("No apartment data available for correlation.")
        return

    merged = pd.merge_asof(
        df_energy.sort_values("Timestamp"),
        df_weather.sort_values("Timestamp"),
        on="Timestamp",
        tolerance=pd.Timedelta("15min"),
        direction="nearest",
    )

    # Use category aggregation so we correlate Temperature with Cooling (ACs)
    df_cat = aggregate_to_categories(merged, apartment_code)
    if df_cat.empty or "ACs" not in df_cat.columns:
        st.warning("No AC data available to correlate with weather.")
        return

    temp_candidates = [c for c in merged.columns if "temp" in c.lower()]
    hum_candidates = [c for c in merged.columns if "humid" in c.lower() or "rh" in c.lower()]
    if not temp_candidates:
        st.warning("No temperature column found.")
        return

    temp_col = temp_candidates[0]
    hum_col = hum_candidates[0] if hum_candidates else None

    resident_insight("On hotter days, cooling demand usually rises; this explains high-AC days.")

    fig = px.scatter(
        merged,
        x=temp_col,
        y=df_cat["ACs"],
        color=hum_col,
        title=f"{apartment_code} – Cooling Power (kW) vs Temperature (°C)",
        labels={temp_col: "Temperature (°C)", "y": "Cooling Power (kW)"},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    merged["Date"] = merged["Timestamp"].dt.date
    daily = merged.groupby("Date").agg({temp_col: "mean"}).reset_index()
    daily["Cooling Energy (kWh)"] = df_cat.groupby(merged["Date"])["ACs"].sum().values * INTERVAL_HOURS

    BASE_TEMP = 24
    daily["CDD"] = (daily[temp_col] - BASE_TEMP).clip(lower=0)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=daily["Date"], y=daily["CDD"], name="Cooling Degree Days", opacity=0.6), secondary_y=False)
    fig2.add_trace(
        go.Scatter(x=daily["Date"], y=daily["Cooling Energy (kWh)"], name="Cooling Energy (kWh)", mode="lines+markers"),
        secondary_y=True,
    )
    fig2.update_layout(title=f"{apartment_code} – Cooling Energy (kWh) vs Cooling Degree Days", xaxis_title="Date")
    fig2.update_yaxes(rangemode="tozero", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

# ================= INSIGHTS (category-aware) =================
def insight_top_category_share(df_cat: pd.DataFrame) -> str:
    if df_cat.empty:
        return "No measurable usage in the selected period."
    cats = [c for c in df_cat.columns if c != "Timestamp"]
    if not cats:
        return "No measurable usage in the selected period."
    energy = {c: float(df_cat[c].fillna(0).sum() * INTERVAL_HOURS) for c in cats}
    total = sum(energy.values())
    top = max(energy, key=energy.get)
    share = (energy[top] / total) * 100 if total > 0 else 0
    return f"Most electricity in this period comes from **{top}** (~{share:.0f}% of total)."

def insight_peak_time_block(df_cat: pd.DataFrame) -> str:
    if df_cat.empty or "Timestamp" not in df_cat.columns:
        return "Not enough data to identify peak time blocks."
    cats = [c for c in df_cat.columns if c != "Timestamp"]
    if not cats:
        return "Not enough data to identify peak time blocks."
    d = df_cat.dropna(subset=["Timestamp"]).copy()
    d["hour"] = d["Timestamp"].dt.hour
    d["Time Block"] = d["hour"].apply(time_block_from_hour)
    d["Total Power (kW)"] = d[cats].sum(axis=1)
    block = d.groupby("Time Block")["Total Power (kW)"].mean().reindex(TIME_BLOCK_ORDER)
    if block.dropna().empty:
        return "Not enough data to identify peak time blocks."
    b = block.idxmax()
    return f"Your highest typical usage happens in **{b}**."

def insight_weekday_weekend_delta(df_cat: pd.DataFrame) -> str:
    if df_cat.empty or "Timestamp" not in df_cat.columns:
        return "Not enough data to compare weekdays vs weekends."
    cats = [c for c in df_cat.columns if c != "Timestamp"]
    if not cats:
        return "Not enough data to compare weekdays vs weekends."
    d = df_cat.dropna(subset=["Timestamp"]).copy()
    d["day_type"] = d["Timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    d["Total Power (kW)"] = d[cats].sum(axis=1)
    wk = d[d["day_type"] == "Weekday"]["Total Power (kW)"].mean()
    we = d[d["day_type"] == "Weekend"]["Total Power (kW)"].mean()
    if np.isnan(wk) or np.isnan(we) or wk == 0:
        return "Not enough data to compare weekdays vs weekends."
    delta = (we - wk) / wk * 100
    if delta >= 0:
        return f"Your home is typically more active on weekends (~{delta:.0f}% higher)."
    return f"Your home is typically less active on weekends (~{abs(delta):.0f}% lower)."

# ================= CATEGORY VISUALS =================
def plot_category_power_over_time(df_cat: pd.DataFrame, apartment_code: str, key_suffix: str):
    """Time series of category power (kW)."""
    if df_cat.empty:
        st.info("No measurable usage for the selected period.")
        return

    cats = [c for c in df_cat.columns if c != "Timestamp"]
    resident_insight(insight_peak_time_block(df_cat))

    melt_df = df_cat.melt(id_vars=["Timestamp"], value_vars=cats, var_name="Category", value_name="Power (kW)")
    melt_df = melt_df.dropna(subset=["Power (kW)"])

    fig = px.line(
        melt_df,
        x="Timestamp",
        y="Power (kW)",
        color="Category",
        title=f"{apartment_code} – Power by Major Category (kW)",
        labels={"Timestamp": "Time"},
    )
    fig.update_layout(height=500, legend_title_text="Category")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_code}_cat_power_time_{key_suffix}")

def plot_category_energy_bar(df_cat: pd.DataFrame, apartment_code: str, key_suffix: str):
    """Total energy (kWh) by category."""
    if df_cat.empty:
        st.info("No measurable usage for the selected period.")
        return
    df_tot = category_totals_kwh(df_cat)
    if df_tot.empty:
        st.info("No measurable category energy for the selected period.")
        return

    resident_insight(insight_top_category_share(df_cat))

    fig = px.bar(
        df_tot,
        x="Category",
        y="Total Consumption (kWh)",
        title=f"{apartment_code} – Total Consumption by Category (kWh)",
        labels={"Total Consumption (kWh)": "Energy (kWh)"},
    )
    fig.update_layout(height=450)
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_code}_cat_energy_{key_suffix}")

def plot_weekday_weekend_energy_by_category(df_cat: pd.DataFrame, apartment_code: str, key_suffix: str):
    """
    Weekday vs Weekend energy per day (kWh/day) by category.
    This keeps units comparable with other energy charts.
    """
    if df_cat.empty or "Timestamp" not in df_cat.columns:
        st.info("Not enough data for weekday/weekend comparison.")
        return

    cats = [c for c in df_cat.columns if c != "Timestamp"]
    if not cats:
        st.info("Not enough data for weekday/weekend comparison.")
        return

    d = df_cat.dropna(subset=["Timestamp"]).copy()
    d["Date"] = d["Timestamp"].dt.date
    d["DayType"] = d["Timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")

    # daily kWh by category
    for c in cats:
        d[c] = d[c].fillna(0.0) * INTERVAL_HOURS

    daily = d.groupby(["Date", "DayType"], as_index=False)[cats].sum()

    # average kWh/day by day type
    avg = daily.groupby("DayType", as_index=False)[cats].mean()

    long = avg.melt(id_vars=["DayType"], value_vars=cats, var_name="Category", value_name="kWh_per_day")
    long = long[long["kWh_per_day"] > 0]  # drop zero categories

    resident_insight(insight_weekday_weekend_delta(df_cat))

    fig = px.bar(
        long,
        x="Category",
        y="kWh_per_day",
        color="DayType",
        barmode="group",
        title=f"{apartment_code} – Weekday vs Weekend Energy by Category (kWh/day)",
        labels={"kWh_per_day": "Energy per day (kWh/day)"},
    )
    fig.update_layout(height=450, legend_title_text="Day type")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_code}_wkwe_cat_kwh_{key_suffix}")

def plot_time_block_profile(df_cat: pd.DataFrame, apartment_code: str, key_suffix: str):
    """Average total power (kW) by resident time block."""
    if df_cat.empty or "Timestamp" not in df_cat.columns:
        st.info("No timestamped data available.")
        return
    cats = [c for c in df_cat.columns if c != "Timestamp"]
    if not cats:
        st.info("No measurable usage for the selected period.")
        return

    d = df_cat.dropna(subset=["Timestamp"]).copy()
    d["Time Block"] = d["Timestamp"].dt.hour.apply(time_block_from_hour)
    d["Total Power (kW)"] = d[cats].sum(axis=1)

    block = d.groupby("Time Block", as_index=False)["Total Power (kW)"].mean()
    block["Time Block"] = pd.Categorical(block["Time Block"], categories=TIME_BLOCK_ORDER, ordered=True)
    block = block.sort_values("Time Block")

    resident_insight(insight_peak_time_block(df_cat))

    fig = px.bar(
        block,
        x="Time Block",
        y="Total Power (kW)",
        title=f"{apartment_code} – Typical Home Activity by Time of Day (kW)",
        labels={"Total Power (kW)": "Average power (kW)"},
    )
    fig.update_layout(height=450, xaxis_title="")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_code}_timeblock_{key_suffix}")

def plot_category_activity_heatmaps(df_cat: pd.DataFrame, apartment_code: str, key_suffix: str):
    """
    For each category, show when it is active across the week.
    Metric = share of intervals where category power > ACTIVE_POWER_KW.
    Uses resident time blocks (not 0-23 hour).
    """
    if df_cat.empty or "Timestamp" not in df_cat.columns:
        st.info("No operation pattern available.")
        return

    cats = [c for c in df_cat.columns if c != "Timestamp"]
    if not cats:
        st.info("No operation pattern available.")
        return

    d = df_cat.dropna(subset=["Timestamp"]).copy()
    d["Day"] = d["Timestamp"].dt.day_name()
    d["Time Block"] = d["Timestamp"].dt.hour.apply(time_block_from_hour)

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for cat in cats:
        tmp = d[["Day", "Time Block", cat]].copy()
        tmp["Active"] = (tmp[cat].fillna(0.0) > ACTIVE_POWER_KW).astype(int)

        # share of active intervals per day/block
        pivot = tmp.groupby(["Day", "Time Block"])["Active"].mean().unstack(fill_value=0.0)
        pivot = pivot.reindex(days_order)
        pivot = pivot.reindex(columns=TIME_BLOCK_ORDER, fill_value=0.0)

        resident_insight(f"Darker cells show when **{cat}** is active more often during the week.")

        fig = px.imshow(
            pivot,
            title=f"{apartment_code} – When {cat} is active during the week",
            labels=dict(x="Time of day", y="Day of week", color="Active share"),
            aspect="auto",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True, key=f"{apartment_code}_cat_heat_{cat}_{key_suffix}")

# ================= RESIDENT-FACING ACTIVITY INSIGHTS (replaces ON-events view) =================
def activity_insights_panel(df_cat: pd.DataFrame, apartment_code: str, key_suffix: str):
    """
    Replace raw ON-event plots with resident-relevant summaries:
    - Top categories by active hours
    - Day of highest home energy
    - Weekend vs weekday difference
    """
    if df_cat.empty or "Timestamp" not in df_cat.columns:
        st.info("No activity insights available for the selected period.")
        return

    cats = [c for c in df_cat.columns if c != "Timestamp"]
    if not cats:
        st.info("No activity insights available for the selected period.")
        return

    # Active hours per category
    d = df_cat.dropna(subset=["Timestamp"]).copy()
    for c in cats:
        d[f"{c}__active"] = (d[c].fillna(0.0) > ACTIVE_POWER_KW).astype(int)

    active_hours = []
    for c in cats:
        hours = float(d[f"{c}__active"].sum() * INTERVAL_HOURS)
        if hours > 0:
            active_hours.append({"Category": c, "Active Hours": hours})

    active_df = pd.DataFrame(active_hours).sort_values("Active Hours", ascending=False).head(5)

    # Day of highest energy (kWh)
    d["Date"] = d["Timestamp"].dt.date
    d["Total_kWh_interval"] = d[cats].fillna(0.0).sum(axis=1) * INTERVAL_HOURS
    daily_kwh = d.groupby("Date", as_index=False)["Total_kWh_interval"].sum()
    peak_day = None
    if not daily_kwh.empty:
        peak = daily_kwh.loc[daily_kwh["Total_kWh_interval"].idxmax()]
        peak_day = (str(peak["Date"]), float(peak["Total_kWh_interval"]))

    # Weekend vs weekday (kWh/day)
    d["DayType"] = d["Timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    daily_by_type = d.groupby(["Date", "DayType"], as_index=False)["Total_kWh_interval"].sum()
    avg_by_type = daily_by_type.groupby("DayType", as_index=False)["Total_kWh_interval"].mean()
    wk = float(avg_by_type[avg_by_type["DayType"] == "Weekday"]["Total_kWh_interval"].iloc[0]) if "Weekday" in avg_by_type["DayType"].values else np.nan
    we = float(avg_by_type[avg_by_type["DayType"] == "Weekend"]["Total_kWh_interval"].iloc[0]) if "Weekend" in avg_by_type["DayType"].values else np.nan

    # Panel insights
    resident_insight("Focus on reducing long ‘idle ON’ time for lights and cooling; small daily reductions add up over a month.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if peak_day:
            st.metric("Highest activity day", peak_day[0], f"{peak_day[1]:.1f} kWh")
        else:
            st.metric("Highest activity day", "NA")
    with c2:
        st.metric("Weekday average", f"{wk:.1f} kWh/day" if not np.isnan(wk) else "NA")
    with c3:
        st.metric("Weekend average", f"{we:.1f} kWh/day" if not np.isnan(we) else "NA")

    st.markdown("### Top 5 categories by active time")
    if active_df.empty:
        st.info("No active usage detected above the activity threshold for the selected period.")
    else:
        fig = px.bar(
            active_df,
            x="Category",
            y="Active Hours",
            title=f"{apartment_code} – Categories active for the longest time (hours)",
            labels={"Active Hours": "Active time (hours)"},
        )
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True, key=f"{apartment_code}_active_hours_{key_suffix}")

    with st.expander("How this is calculated"):
        st.write(
            f"""
            - Data resolution: {INTERVAL_MINUTES}-minute intervals.
            - A category is treated as **active** when power is above **{ACTIVE_POWER_KW:.2f} kW**.
            - Standby/phantom loads below the threshold are not counted as active time.
            - Active hours = (number of active intervals) × ({INTERVAL_MINUTES}/60).
            """
        )

# ================= EPI (month-clean) =================
@st.cache_data
def load_apartment_areas(path="processed_apartments2/metadata/apartment_area.csv"):
    if not os.path.exists(path):
        st.warning("apartment_area.csv not found.")
        return {}
    df = pd.read_csv(path)
    if not {"Apartment", "Area_m2"}.issubset(df.columns):
        st.error("apartment_area.csv must have columns: Apartment, Area_m2")
        return {}
    df["key"] = df["Apartment"].apply(normalize_name)
    return dict(zip(df["key"], df["Area_m2"]))

def plot_epi_all_months(apartment_code: str, areas: dict, df_full: pd.DataFrame):
    """
    Monthly EPI (kWh/m²) computed from full available apartment data (not sidebar range),
    and displayed as separate months (no merging).
    """
    if df_full.empty or "Timestamp" not in df_full.columns:
        st.warning("No timestamped data available for EPI calculation.")
        return

    # Area lookup expects the real apartment name in the metadata file.
    # Keep reverse mapping internal only; never show real names in UI.
    REAL_NAME_MAP = {v: k for k, v in ANON_MAP.items()}
    if apartment_code not in REAL_NAME_MAP:
        st.warning(f"No area mapping available for {apartment_code}.")
        return

    real_name = REAL_NAME_MAP[apartment_code]
    real_key = normalize_name(real_name)
    apt_area = areas.get(real_key)

    if apt_area is None:
        st.warning(f"Area data not available for {apartment_code}.")
        return

    df = df_full.dropna(subset=["Timestamp"]).copy()
    df_cat = aggregate_to_categories(df, apartment_code)

    if df_cat.empty:
        st.warning("No measurable usage available for EPI calculation.")
        return

    cats = [c for c in df_cat.columns if c != "Timestamp"]
    df_cat["MonthStart"] = df_cat["Timestamp"].dt.to_period("M").dt.to_timestamp()
    df_cat["TotalPower_kW"] = df_cat[cats].sum(axis=1)

    monthly = df_cat.groupby("MonthStart", as_index=False)["TotalPower_kW"].sum()
    monthly["Total_kWh"] = monthly["TotalPower_kW"] * INTERVAL_HOURS
    monthly["EPI_kWh_per_m2"] = monthly["Total_kWh"] / apt_area
    monthly = monthly.sort_values("MonthStart")
    monthly["MonthLabel"] = monthly["MonthStart"].dt.strftime("%b %Y")

    if monthly.empty:
        st.warning("No monthly EPI values could be computed.")
        return

    first = monthly["MonthLabel"].iloc[0]
    last = monthly["MonthLabel"].iloc[-1]
    st.info(f"Data available for EPI: **{first} → {last}**")

    resident_insight("EPI helps compare your monthly energy use per square metre; lower is better for the same comfort level.")

    fig = px.bar(
        monthly,
        x="MonthLabel",
        y="EPI_kWh_per_m2",
        title=f"{apartment_code} – Monthly Energy Performance Index (kWh/m²)",
        labels={"MonthLabel": "Month", "EPI_kWh_per_m2": "EPI (kWh/m²)"},
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(xaxis=dict(type="category"))
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_code}_epi_all")

# ================= MAIN APP =================
def main():
    st.set_page_config(page_title="Apartment Energy Analytics", layout="wide")
    st.title("Residential Energy Consumption Dashboard")

    with st.spinner("Loading apartment data..."):
        all_dfs = load_all_data()

    if not all_dfs:
        st.error("No data files found. Please check the DATA_FOLDER path.")
        st.stop()

    st.success(f"Loaded data for {len(all_dfs)} apartments")

    areas = load_apartment_areas()

    st.sidebar.header("Apartment Selection")
    apartment_names = sorted(list(all_dfs.keys()))
    selected_option = st.sidebar.selectbox("Select Option", ["All Apartments"] + apartment_names)

    st.sidebar.subheader("Date Range Filter")
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
        st.header("All Apartments – Comparative View")

        # Apply date filter
        filtered = {}
        for apt, df_apt in all_dfs.items():
            if "Timestamp" in df_apt.columns:
                mask = (df_apt["Timestamp"] >= start_date) & (df_apt["Timestamp"] <= end_date)
                filtered[apt] = df_apt[mask]
            else:
                filtered[apt] = df_apt

        resident_insight("Use this view to compare typical daily activity patterns across homes.")

        # Simple comparison: total average power by time block
        rows = []
        for apt, df_apt in filtered.items():
            if "Timestamp" not in df_apt.columns or df_apt.empty:
                continue
            df_cat = aggregate_to_categories(df_apt, apt)
            if df_cat.empty:
                continue
            cats = [c for c in df_cat.columns if c != "Timestamp"]
            d = df_cat.dropna(subset=["Timestamp"]).copy()
            d["Time Block"] = d["Timestamp"].dt.hour.apply(time_block_from_hour)
            d["Total Power (kW)"] = d[cats].sum(axis=1)
            agg = d.groupby("Time Block")["Total Power (kW)"].mean().reindex(TIME_BLOCK_ORDER)
            for tb, val in agg.items():
                rows.append({"Apartment": apt, "Time Block": tb, "Average Power (kW)": val})

        if not rows:
            st.warning("No comparable data available for the selected range.")
            return

        comp_df = pd.DataFrame(rows)
        fig = px.scatter(
            comp_df,
            x="Time Block",
            y="Average Power (kW)",
            color="Apartment",
            title="Typical Home Activity by Time Block (kW)",
            labels={"Average Power (kW)": "Average power (kW)"},
        )
        fig.update_traces(marker=dict(size=8, opacity=0.75))
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)

        return

    # ================= INDIVIDUAL APARTMENT VIEW =================
    apartment_code = selected_option  # already anonymised
    if apartment_code not in all_dfs:
        st.error("Selected apartment data not found.")
        return

    df_full = all_dfs[apartment_code].copy()
    df = df_full.copy()
    if "Timestamp" in df.columns:
        df = df[(df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)]

    st.sidebar.subheader("Apartment Info")
    st.sidebar.write(f"Apartment: {apartment_code}")
    st.sidebar.write(f"Filtered records: {len(df)}")
    if not df.empty and "Timestamp" in df.columns:
        st.sidebar.write(f"Date range: {df['Timestamp'].min().strftime('%Y-%m-%d')} → {df['Timestamp'].max().strftime('%Y-%m-%d')}")

    st.header(f"Energy Analytics – {apartment_code}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Overview", "Category Patterns", "Time Analysis", "AC & Weather", "Activity Insights", "EPI (Monthly)"]
    )

    # Pre-compute category aggregation for this filtered window
    df_cat = aggregate_to_categories(df, apartment_code)

    with tab1:
        st.subheader("Energy Overview (Major Categories)")

        plot_category_power_over_time(df_cat, apartment_code, "overview")

        # KPI row
        cats = [c for c in df_cat.columns if c != "Timestamp"] if not df_cat.empty else []
        total_kwh = float(df_cat[cats].fillna(0.0).sum().sum() * INTERVAL_HOURS) if cats else 0.0
        avg_power_kw = float(df_cat[cats].fillna(0.0).sum(axis=1).mean()) if cats else 0.0

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total consumption (kWh)", f"{total_kwh:.1f}")
        with c2:
            st.metric("Average power (kW)", f"{avg_power_kw:.2f}")

        colA, colB = st.columns(2)
        with colA:
            plot_category_energy_bar(df_cat, apartment_code, "overview")
        with colB:
            plot_weekday_weekend_energy_by_category(df_cat, apartment_code, "overview")

    with tab2:
        st.subheader("When Categories Are Active During the Week")
        plot_category_activity_heatmaps(df_cat, apartment_code, "patterns")

    with tab3:
        st.subheader("Time-based Patterns")
        plot_time_block_profile(df_cat, apartment_code, "time")

    with tab4:
        st.subheader("AC & Weather")
        resident_insight("This shows whether hotter or more humid periods align with higher cooling use.")

        # Current build: weather configured only for LAKNW05 (display only anonymised code)
        WEATHER_APT_CODE = ANON_MAP.get("Lakeside K 502", "LAKNW05")
        if apartment_code.strip().lower() == WEATHER_APT_CODE.strip().lower():
            weather_dict = load_weather_data()
            if weather_dict:
                for zone, weather_df in weather_dict.items():
                    st.markdown(f"### Zone: {zone}")
                    plot_weather_correlation(df, weather_df, apartment_code)
            else:
                st.warning("No weather files found for this apartment.")
        else:
            st.info(f"Weather correlation is available only for {WEATHER_APT_CODE} in the current build.")

    with tab5:
        st.subheader("Activity Insights")
        activity_insights_panel(df_cat, apartment_code, "activity")

    with tab6:
        st.subheader("Energy Performance Index (Monthly EPI)")
        # IMPORTANT: use full data to avoid partial-month ambiguity
        plot_epi_all_months(apartment_code, areas, df_full)

if __name__ == "__main__":
    main()
