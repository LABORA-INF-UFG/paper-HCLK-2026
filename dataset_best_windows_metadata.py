import pandas as pd
import numpy as np
import argparse

# ==============================
# A3-like association function
# ==============================
def a3_like_association(user_xy: np.ndarray, ru_xy: np.ndarray, d_hyst: float=5.0, ttt: int=2) -> np.ndarray:
    """Associates each user position to a Radio Unit using an A3-like policy."""
    T = user_xy.shape[0]
    dists = np.linalg.norm(user_xy[:, None, :] - ru_xy[None, :, :], axis=2)
    best  = np.argmin(dists, axis=1)
    assoc = np.zeros(T, dtype=int)
    cur = best[0]; hold = 0
    for t in range(T):
        cand = best[t]
        if dists[t, cand] + d_hyst < dists[t, cur]:
            hold += 1
            if hold >= ttt:
                cur = cand
                hold = 0
        else:
            hold = 0
        assoc[t] = cur
    return assoc

# ==============================
# Best window selection
# ==============================
def best_window(user_df: pd.DataFrame, ru_xy: np.ndarray, W: int=60, d_hyst=5.0, ttt=2):
    """Selects the best contiguous window of W samples."""
    user_df = user_df.reset_index(drop=True)
    xy = user_df[["x_real", "y_real"]].to_numpy(dtype=float)
    assoc = a3_like_association(xy, ru_xy, d_hyst=d_hyst, ttt=ttt)

    # Search all windows of size W
    T = len(user_df)
    rows = []
    for s in range(0, T - W + 1):
        e = s + W
        aw = assoc[s:e]
        n_ho = int(np.sum(np.diff(aw) != 0))  # number of handovers
        rows.append((s, e, n_ho))

    if not rows:
        return None

    win = pd.DataFrame(rows, columns=["s","e","n_ho"])
    # Choose the window with the highest number of handovers
    win = win.sort_values(["n_ho"], ascending=[False]).reset_index(drop=True)
    s, e = int(win.loc[0,"s"]), int(win.loc[0,"e"])
    return user_df.iloc[s:e].copy()

# ==============================
# Main script
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select best contiguous windows per user")
    parser.add_argument("--window", type=int, default=60,
                        help="Contiguous window size in samples/seconds (default=60)")
    args = parser.parse_args()
    W = args.window

    # Load input data (unchanged)
    df = pd.read_csv("data/predictions/users_predicted_location.csv")
    ru = pd.read_csv("data/ru_positions.csv").sort_values("ru_id")
    ru_xy = ru[["x","y"]].to_numpy(dtype=float)

    users = sorted(df["id_user"].unique().tolist())
    selected_list = []

    for uid in users:
        u_df = df[df["id_user"] == uid]
        subset = best_window(u_df, ru_xy, W=W, d_hyst=5.0, ttt=2)
        if subset is not None:
            # Keep only original columns
            subset = subset[["id_user","x_real","y_real",
                             "x_central_offline","y_central_offline",
                             "x_central_online","y_central_online",
                             "x_federated","y_federated"]].copy()
            selected_list.append(subset)

    # Concatenate all selected windows
    if selected_list:
        best_out = pd.concat(selected_list, ignore_index=True)
    else:
        best_out = pd.DataFrame(columns=["id_user","x_real","y_real",
                                         "x_central_offline","y_central_offline",
                                         "x_central_online","y_central_online",
                                         "x_federated","y_federated"])

    # Save final output file
    out_path = f"data/predictions/users_predicted_location_best{W}.csv"
    best_out.to_csv(out_path, index=False)

    print(f"[OK] Output file saved to: {out_path}")
