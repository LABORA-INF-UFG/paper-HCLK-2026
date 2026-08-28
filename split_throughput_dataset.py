import pandas as pd
import re
import os

# ============================================================
# Settings
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_CSV = os.path.join(DATA_DIR, "throughput_users.csv")
OUTPUT_DIR = DATA_DIR


# Subsets of base stations (TX IDs)
STATION_SETS = [
    [1, 4, 5],
    [1, 3, 4, 5, 8],
    [1, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
]

# User counts
USER_COUNTS = [10, 20, 30, 40, 50]

# ============================================================
# Load full dataset
# ============================================================
df = pd.read_csv(INPUT_CSV)

BASE_COLS = ["row_idx", "id_user"]

# All throughput columns (all stations, bands and approaches)
THR_COLS = [c for c in df.columns if c not in BASE_COLS]

# Regex to extract TX id
TX_REGEX = re.compile(r"_tx(\d+)_")

# ============================================================
# 1) Subset by BASE STATIONS
#    → Keeps ALL users
# ============================================================
for stations in STATION_SETS:
    stations = sorted(stations)
    stations_str = "_".join(str(s) for s in stations)

    selected_cols = BASE_COLS.copy()

    for col in THR_COLS:
        match = TX_REGEX.search(col)
        if match:
            tx_id = int(match.group(1))
            if tx_id in stations:
                selected_cols.append(col)

    # Keep ALL rows (50 users)
    out_df = df[selected_cols].copy()

    out_file = os.path.join(OUTPUT_DIR, f"throughput_stations_{stations_str}.csv")
    out_df.to_csv(out_file, index=False)

    print(f"[OK] Stations {stations_str} | Users kept: {out_df['id_user'].nunique()}")

# ============================================================
# 2) Subset by USERS
#    → Keeps ALL 9 base stations
# ============================================================
users_sorted = sorted(df["id_user"].unique())

for n_users in USER_COUNTS:
    selected_users = users_sorted[:n_users]

    # Filter users
    out_df = df[df["id_user"].isin(selected_users)].copy()

    # Explicitly keep ALL TX columns
    out_df = out_df[BASE_COLS + THR_COLS]

    out_file = os.path.join(OUTPUT_DIR, f"throughput_users_{n_users}.csv")
    out_df.to_csv(out_file, index=False)

    print(
        f"[OK] Users: {n_users} | "
        f"Stations kept: {len(set(TX_REGEX.findall(' '.join(THR_COLS))))}"
    )
