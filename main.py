import pandas as pd
import glob
import os
import re

# ========== CONFIG ==========
INPUT_FOLDER = "data"
MAPPING_FILE = "Lodha-Apartments-Sensors-mapping_modified.xlsx"
OUTPUT_FOLDER = "processed_apartments2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load mapping file
mapping_df = pd.read_excel(MAPPING_FILE)

# --- helper to parse Phase + MacID from filename ---
def parse_filename(filename):
    # Example: "B Phase(7C2C678DEEAC)_91.csv"
    match = re.match(r"(.+?)\((.+?)\)", filename)
    if match:
        phase = match.group(1).strip()
        macid = match.group(2).strip()
        return phase, macid
    return None, None

# --- preprocess single phase file ---
def preprocess_file(file_path, map_row):
    df = pd.read_csv(file_path)

    # 1. Keep only required columns
    keep_cols = ["ts", "P[0]", "P[1]", "P[2]", "P[3]", "P[4]", "P[5]"]
    df = df[keep_cols].copy()

    # 2. Standardize timestamp
    df = df.rename(columns={"ts": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")

    # 3. Resample 5-min
    df = df.set_index("Timestamp").resample("5min").sum().reset_index()

    # 4. Rename channels using mapping row
    channel_map = {
        f"P[{i}]": map_row[f"Channel {i+1}"]
        for i in range(6)
        if pd.notna(map_row[f"Channel {i+1}"])
    }
    df = df.rename(columns=channel_map)

    # 🔹 5. Collapse duplicate appliance names inside same phase
    duplicate_cols = df.columns[df.columns.duplicated()].unique()
    for col in duplicate_cols:
        same_cols = [c for c in df.columns if c == col]
        df[col] = df[same_cols].sum(axis=1)   # sum duplicates
        df = df.drop(columns=same_cols[1:])   # drop extras

    # 6. Convert Power (W) → Consumption (kWh)
    for col in df.columns:
        if col != "Timestamp":
            df[col] = df[col] * (5 / 60) / 1000  # W → kWh

    return df

# --- merge all phases for one apartment ---
def merge_apartment(apartment_name, files_info):
    dfs = [preprocess_file(file_path, map_row) for file_path, map_row in files_info]

    # Outer join on Timestamp across all phases
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(
            merged,
            df,
            on="Timestamp",
            how="outer",
            suffixes=("", "_dup")  # prevent merge errors
        )

        # Handle duplicate columns from merge
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        for col in dup_cols:
            base_col = col.replace("_dup", "")
            if base_col in merged.columns:
                # Sum duplicates into base_col
                merged[base_col] = merged[[base_col, col]].sum(axis=1, skipna=True)
                merged.drop(columns=[col], inplace=True)

    # 🔹 Collapse duplicate appliance names across phases
    merged = (
        merged.set_index("Timestamp")
        .stack()
        .groupby(level=[0, 1])
        .sum()
        .unstack()
        .reset_index()
    )

    # Add Apartment Name column
    merged.insert(1, "Apartment", apartment_name)

    # Save apartment file
    output_file = os.path.join(OUTPUT_FOLDER, f"{apartment_name}.csv")
    merged.to_csv(output_file, index=False)
    print(f"✅ Saved {output_file}")

# ========== MAIN ==========
files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

# Collect files per apartment
apartments = {}
for f in files:
    fname = os.path.basename(f)
    phase, macid = parse_filename(fname)

    if not phase or not macid:
        print(f"⚠️ Skipping {fname} (couldn’t parse)")
        continue

    # Find mapping row for this phase+macid
    row = mapping_df[(mapping_df["Phases"] == phase) & (mapping_df["MacID"] == macid)]
    if row.empty:
        print(f"⚠️ No mapping found for {fname}")
        continue

    apt_name = row.iloc[0]["Apartment Name"]

    # Group by Apartment Name
    apartments.setdefault(apt_name, []).append((f, row.iloc[0]))

# Process each apartment (merging its phases)
for apt, files_info in apartments.items():
    merge_apartment(apt, files_info)
