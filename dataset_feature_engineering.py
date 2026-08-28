"""
Generates feature dataset for ESN (location prediction)
---------------------------------------------------------
Input:  dataset.csv  with columns: id, x, y, o, height
          - id      : user identifier
          - x, y    : position in meters (samples every 1s for 1h -> 3600 rows/user)
          - o       : orientation in DEGREES [0..359], direction the user is facing
          - height  : user height (m) - not used as a direct predictor for location

Output: derived_features_full.csv
          - Contains only attributes useful for the ESN to learn displacement patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Basic settings
# -----------------------------
DATA_DIR = Path("data")
IN_PATH  = DATA_DIR / "dataset.csv"
OUT_PATH = DATA_DIR / "derived_features_full.csv"

# Small tolerance to avoid division by zero / numerical instability
EPS = 1e-9

# -----------------------------
# Load original dataset
# -----------------------------
df = pd.read_csv(IN_PATH)

# Minimal column sanity check
expected_cols = {"id", "x", "y", "o", "height"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# -----------------------------
# Utility functions
# -----------------------------
def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Wraps angle in radians to the interval (-pi, pi]."""
    return (angle_rad + np.pi) % (2*np.pi) - np.pi

def zscore_per_user(s: pd.Series) -> pd.Series:
    """Standardizes (z-score) per user (mean 0, std 1). Prevents reservoir saturation.
    Returns zeros if std=0."""
    mu = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd < EPS:
        return s*0.0
    return (s - mu) / (sd + EPS)

def derive_user_block(g: pd.DataFrame) -> pd.DataFrame:
    """
    Derives features for ONE user (one group 'g').
    Assumes dt = 1s and rows are already in temporal order.
    """
    g = g.copy().reset_index(drop=True)

    # -----------------------------
    # 1) First-order kinematics (first differences)
    #    - dx, dy: displacements per second (m/s), best predictors for next x,y.
    # -----------------------------
    g["dx"] = g["x"].diff().fillna(0.0)
    g["dy"] = g["y"].diff().fillna(0.0)

    # Velocity vector (m/s)
    #  - vx, vy equal dx, dy since dt=1s
    g["vx"] = g["dx"]
    g["vy"] = g["dy"]

    # Speed magnitude (m/s) — useful to distinguish "stationary", "walking", "running"
    g["speed"] = np.sqrt(g["vx"]**2 + g["vy"]**2)

    # Movement heading in radians [-pi, pi]
    #  - atan2(dy, dx) is more stable than arctan(dy/dx)
    g["heading_rad"] = np.arctan2(g["dy"], g["dx"]).fillna(0.0)

    # Trigonometric representation of heading (avoids angular ambiguity)
    g["cos_heading"] = np.cos(g["heading_rad"])
    g["sin_heading"] = np.sin(g["heading_rad"])

    # -----------------------------
    # 2) Second-order kinematics (acceleration)
    #    - strong signals for predicting changes in direction/speed
    # -----------------------------
    g["ax"] = g["vx"].diff().fillna(0.0)         # m/s^2
    g["ay"] = g["vy"].diff().fillna(0.0)         # m/s^2
    g["accel"] = np.sqrt(g["ax"]**2 + g["ay"]**2)

    # Heading turn rate: delta heading per second (rad/s)
    #  - uses unwrapped heading to avoid numerical jumps (e.g., -pi -> +pi)
    heading_unwrapped = np.unwrap(g["heading_rad"].values)
    d_heading = np.diff(np.concatenate([[heading_unwrapped[0]], heading_unwrapped]))
    g["turn_rate"] = d_heading  # rad/s

    # -----------------------------
    # 3) User orientation as a movement intention cue
    #    - we do not predict 'o', but it helps predict "where" the user tends to move
    # -----------------------------
    o_rad = np.radians(g["o"].astype(float).values)  # column 'o' is in DEGREES [0..359]
    g["cos_o"] = np.cos(o_rad)                       # stable representation in [-1, 1]
    g["sin_o"] = np.sin(o_rad)

    # Difference between gaze and movement direction (gaze - heading), wrapped to [-pi, pi]
    gaze_move_diff = wrap_to_pi(o_rad - g["heading_rad"].values)
    g["cos_gaze_move"] = np.cos(gaze_move_diff)      # gaze <-> movement correlation
    g["sin_gaze_move"] = np.sin(gaze_move_diff)

    # Binary mobility state — useful for the ESN to distinguish dynamic regimes
    g["is_stationary"] = (g["speed"] < 1e-3).astype(np.int8)

    # -----------------------------
    # 4) Per-user z-score standardization — reservoir stability
    #    Standardize only unbounded variables to help the ESN.
    #    Trigonometric features are already in [-1,1], so cos/sin are not standardized.
    # -----------------------------
    for col in ["x", "y", "vx", "vy", "speed", "ax", "ay", "accel", "turn_rate"]:
        g[f"z_{col}"] = zscore_per_user(g[col])

    # Height (height) is NOT used as input: the goal is to predict location, and height adds
    # no dynamic signal useful here. Kept only in the original dataset.

    # Final guarantees: no NaN/Inf in features
    g = g.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return g

# -----------------------------
# Derive features for all users
# -----------------------------
feat = df.groupby("id", group_keys=False).apply(derive_user_block)

# -----------------------------
# Select output columns
#   - Raw x,y kept as TARGETS (what the ESN will predict in the future)
#   - z-features and stable trigonometric/boolean terms used as INPUTS
# -----------------------------
out_cols = [
    # User identifier
    "id",

    # TARGETS - current raw positions
    # The ESN will predict x_{t+H}, y_{t+H}.
    # This file provides x_t, y_t (current). The time shift (H) is handled
    # during dataset preparation for training/validation (sliding window).
    "x", "y",

    # STANDARDIZED DYNAMIC INPUTS (z-score per user)
    #  - standardized x, y help the reservoir "sense" current position without saturation
    "z_x", "z_y",
    "z_vx", "z_vy",     # velocity (m/s) standardized
    "z_speed",          # speed magnitude
    "z_ax", "z_ay",     # acceleration (m/s²) standardized
    "z_accel",          # acceleration magnitude
    "z_turn_rate",      # turn rate (rad/s) standardized

    # ANGULAR INPUTS (bounded, informative, and stable)
    "cos_heading", "sin_heading",  # movement heading in plane
    "cos_o", "sin_o",              # user gaze/orientation (intention cue)
    "cos_gaze_move", "sin_gaze_move",  # gaze vs. heading alignment

    # MOBILITY STATE (dynamic regime)
    "is_stationary",
]

derived = feat[out_cols].copy()

derived.to_csv(OUT_PATH, index=False)
print(f"[OK] Output file generated: {OUT_PATH.resolve()}")
