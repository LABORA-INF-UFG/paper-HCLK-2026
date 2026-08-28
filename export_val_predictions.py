# export_val_predictions.py
# -*- coding: utf-8 -*-
"""
Generates, for each user, a CSV with predictions from the three training scenarios
on the SAME holdout (final 20%), time-aligned:

  id_user, x_real, y_real,
  x_central_offline, y_central_offline,
  x_central_online,  y_central_online,
  x_federated,       y_federated

- Central OFFLINE: uses validation washout (no context bridge).
- Central/Federated ONLINE: uses context bridge (K) + washout_eval.
- All three series are aligned at the largest common t0 and truncated to minimum length.

Requirements:
  - The supplied directories must contain:
      Central OFFLINE:
        models/central_offline_user_<id>.pkl
        scalers/central_offline_x_scaler_user_<id>.pkl
        scalers/central_offline_y_scaler_user_<id>.pkl
      Central ONLINE:
        models/central_online_user_<id>.pkl          (snapshot dict)
        scalers/central_online_x_scaler_user_<id>.pkl  (pickle bytes)
        scalers/central_online_y_scaler_user_<id>.pkl  (pickle bytes)
      Federated ONLINE:
        models/federated_online_user_<id>.pkl        (snapshot dict)
        scalers/federated_online_x_scaler_user_<id>.pkl (pickle bytes)
        scalers/federated_online_y_scaler_user_<id>.pkl (pickle bytes)
"""

from __future__ import annotations
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# same class/utils as training scripts
from model.ESNStreaming import ESNStreaming

FEATURES = [
    "z_x", "z_y",
    "z_vx", "z_vy", "z_speed",
    "z_ax", "z_ay", "z_accel",
    "z_turn_rate",
    "cos_heading", "sin_heading",
    "cos_o", "sin_o",
    "cos_gaze_move", "sin_gaze_move"
]
TARGETS = ["x", "y"]

def load_offline_artifacts(base_dir: str, uid: int):
    mpath = os.path.join(base_dir, "models",  f"central_offline_user_{uid}.pkl")
    xsp   = os.path.join(base_dir, "scalers", f"central_offline_x_scaler_user_{uid}.pkl")
    ysp   = os.path.join(base_dir, "scalers", f"central_offline_y_scaler_user_{uid}.pkl")
    if not (os.path.isfile(mpath) and os.path.isfile(xsp) and os.path.isfile(ysp)):
        return None, None, None
    with open(mpath, "rb") as f:
        esn = pickle.load(f)  # full ESNStreaming object
    with open(xsp, "rb") as f:
        xsc: StandardScaler = pickle.load(f)
    with open(ysp, "rb") as f:
        ysc: StandardScaler = pickle.load(f)
    return esn, xsc, ysc

def load_online_snapshot_and_scalers(base_dir: str, uid: int, prefix: str):
    """
    prefix: 'central_online' or 'federated_online'
    """
    mpath = os.path.join(base_dir, "models",  f"{prefix}_user_{uid}.pkl")
    xsp   = os.path.join(base_dir, "scalers", f"{prefix}_x_scaler_user_{uid}.pkl")
    ysp   = os.path.join(base_dir, "scalers", f"{prefix}_y_scaler_user_{uid}.pkl")
    if not (os.path.isfile(mpath) and os.path.isfile(xsp) and os.path.isfile(ysp)):
        return None, None, None
    with open(mpath, "rb") as f:
        snap: dict = pickle.load(f)  # {"Win","W","Wout"}
    with open(xsp, "rb") as f:
        xsc_bytes = f.read()
    with open(ysp, "rb") as f:
        ysc_bytes = f.read()
    xsc: StandardScaler = pickle.loads(xsc_bytes)
    ysc: StandardScaler = pickle.loads(ysc_bytes)
    return snap, xsc, ysc

def predict_offline_holdout(esn: ESNStreaming,
                            xsc: StandardScaler, ysc: StandardScaler,
                            U_all: np.ndarray, Y_all: np.ndarray,
                            H: int, washout_offline: int):
    """
    Offline validation:
      U_va = U[ntr .. T-H), Y_va = Y[ntr+H .. T), washout=washout_offline within the val window.
    Returns: (t0, y_real[N], y_pred[N])
      where t0 = ntr + washout_offline (input reference; real target is Y[t0+H:]).
    """
    T_all = U_all.shape[0]
    T_pairs = T_all - H
    ntr = int(0.8 * T_pairs)
    if ntr <= 0 or T_pairs <= 0:
        return None

    U_va = U_all[ntr : T_all - H]
    Y_va = Y_all[ntr + H : T_all]
    if U_va.shape[0] <= washout_offline:
        return None

    U_va_s = xsc.transform(U_va)
    preds_s = esn.predict_segment_scaled(U_va_s, washout=washout_offline)
    preds   = ysc.inverse_transform(preds_s)
    Y_eff   = Y_va[washout_offline:]  # align with washout

    t0 = ntr + washout_offline
    return t0, Y_eff, preds  # Y_eff ~ Y[t0+H :], preds same length

def predict_online_holdout(snap: dict,
                           xsc: StandardScaler, ysc: StandardScaler,
                           U_all: np.ndarray, Y_all: np.ndarray,
                           H: int, ctx_k: int, washout_eval: int):
    """
    Validation for online scripts (central/federated):
      Uses context bridge K (last K training steps) + washout_eval over concat([U_ctx, U_val]).
    Returns: (t0, y_real[N], y_pred[N])
      where t0 = ntr + washout_eval (input reference; target in Y[t0+H:]).
    """
    T_all = U_all.shape[0]
    T_pairs = T_all - H
    ntr = int(0.8 * T_pairs)
    if ntr <= 0 or T_pairs <= 0:
        return None

    K = int(max(0, min(ctx_k, ntr)))
    U_ctx = U_all[ntr - K : ntr] if K > 0 else np.zeros((0, U_all.shape[1]), dtype=np.float64)
    U_va  = U_all[ntr : T_all - H]
    Y_va  = Y_all[ntr + H : T_all]
    if U_va.shape[0] <= washout_eval:
        return None

    esn = ESNStreaming(
        n_inputs=U_all.shape[1],
        n_reservoir=snap["W"].shape[0],
        n_outputs=2,
        spectral_radius=0.9, sparsity=0.1, leak_rate=0.3,
        include_inputs_in_state=True, seed=42, rls_lambda=0.99, rls_alpha=1e-3
    )
    esn.load_snapshot(snap)

    U_concat = np.vstack([U_ctx, U_va]) if K > 0 else U_va
    U_concat_s = xsc.transform(U_concat)

    preds_concat_s = esn.predict_segment_scaled(U_concat_s, washout=K + washout_eval)
    preds   = ysc.inverse_transform(preds_concat_s)

    Y_eff   = Y_va[washout_eval:]  # discard washout_eval steps within the holdout
    t0 = ntr + washout_eval
    return t0, Y_eff, preds

def align_and_make_df(uid: int,
                      Y_all: np.ndarray, H: int,
                      off_pack, cen_pack, fed_pack):
    """
    off_pack/cen_pack/fed_pack: tuples (t0, Y_eff, Yhat) or None.
    Aligns at the largest common t0 and truncates to minimum common length.
    Returns DataFrame with the requested columns.
    """
    packs = {"offline": off_pack, "central": cen_pack, "federated": fed_pack}
    valid = {k:v for k,v in packs.items() if v is not None}
    if not valid:
        return None

    t0s = {k:v[0] for k,v in valid.items()}
    t0_common = max(t0s.values())

    # apply shifts and compute common length
    eff = {}
    lengths = []
    for k,(t0,Y_eff,Yhat) in valid.items():
        shift = t0_common - t0
        if shift < 0:
            return None
        if shift >= len(Y_eff):
            return None
        eff[k] = (Y_eff[shift:], Yhat[shift:])
        lengths.append(len(eff[k][0]))

    N = min(lengths)
    if N <= 0:
        return None

    # Real: use Y_all at absolute index t0_common + H
    y_real = Y_all[t0_common + H : t0_common + H + N]
    if y_real.shape[0] != N:
        N = min(N, y_real.shape[0])
        for k in eff:
            eff[k] = (eff[k][0][:N], eff[k][1][:N])
        y_real = y_real[:N]

    df = pd.DataFrame({
        "id_user": [uid]*N,
        "x_real":  y_real[:,0],
        "y_real":  y_real[:,1],
        "x_central_offline": np.nan,
        "y_central_offline": np.nan,
        "x_central_online":  np.nan,
        "y_central_online":  np.nan,
        "x_federated":       np.nan,
        "y_federated":       np.nan,
    })

    if "offline" in eff:
        df.loc[:, "x_central_offline"] = eff["offline"][1][:,0]
        df.loc[:, "y_central_offline"] = eff["offline"][1][:,1]
    if "central" in eff:
        df.loc[:, "x_central_online"]  = eff["central"][1][:,0]
        df.loc[:, "y_central_online"]  = eff["central"][1][:,1]
    if "federated" in eff:
        df.loc[:, "x_federated"]       = eff["federated"][1][:,0]
        df.loc[:, "y_federated"]       = eff["federated"][1][:,1]

    return df

# aggregate per-user CSVs into a single users_predicted_location.csv
def aggregate_users_location(out_dir: str, user_ids: list[int], out_name: str = "users_predicted_location.csv") -> None:
    """
    Reads predictions_user_<id>.csv files (in the order given by user_ids)
    and concatenates them into a single CSV: users_predicted_location.csv (with header).
    Only users with existing files are included; warns about any missing files.
    """
    dfs = []
    for uid in user_ids:
        p = os.path.join(out_dir, f"predictions_user_{uid}.csv")
        if os.path.isfile(p):
            df_u = pd.read_csv(p)
            dfs.append(df_u)
        else:
            print(f"[WARN] File not found for user {uid}: {p} (skipping in aggregate).")
    if len(dfs) == 0:
        print("[WARN] No user files found for aggregation. users_predicted_location.csv will not be generated.")
        return
    agg = pd.concat(dfs, axis=0, ignore_index=True)
    out_path = os.path.join(out_dir, out_name)
    agg.to_csv(out_path, index=False)
    print(f"[OK] Aggregated file saved to: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/derived_features_full.csv", help="full dataset (features + targets)")
    ap.add_argument("--central_offline_dir", default="model/centralized", help="central OFFLINE training directory (contains models/ and scalers/)")
    ap.add_argument("--central_online_dir",  default="model/centralized", help="central ONLINE training directory (contains models/ and scalers/)")
    ap.add_argument("--federated_dir",       default="model/federated", help="FEDERATED ONLINE training directory (contains models/ and scalers/)")
    ap.add_argument("--out_dir", default="data/predictions", help="output directory for per-user CSVs")
    ap.add_argument("--steps_ahead", type=int, default=3, help="Horizon H (seconds)")
    # evaluation parameters (must match those used in training validation)
    ap.add_argument("--offline_washout", type=int, default=50, help="washout used in central OFFLINE validation")
    ap.add_argument("--ctx_k", type=int, default=300, help="context bridge K for ONLINE (central/federated)")
    ap.add_argument("--washout_eval", type=int, default=120, help="washout in ONLINE validation (central/federated)")

    ap.add_argument("--user_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49",
                    help="comma-separated user IDs in the desired order for aggregation in users_location.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    need = set(["id"] + FEATURES + TARGETS)
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in CSV: {sorted(miss)}")

    users = sorted(df["id"].unique().tolist())
    H = max(1, int(args.steps_ahead))

    for uid in users:
        du = df[df["id"] == uid].copy()
        U_all = du[FEATURES].to_numpy(dtype=np.float64)
        Y_all = du[TARGETS].to_numpy(dtype=np.float64)

        # --- central OFFLINE ---
        esn_off, xsc_off, ysc_off = load_offline_artifacts(args.central_offline_dir, uid)
        off_pack = None
        if esn_off is not None:
            off_pack = predict_offline_holdout(esn_off, xsc_off, ysc_off, U_all, Y_all, H, args.offline_washout)

        # --- central ONLINE ---
        cen_snap, xsc_cen, ysc_cen = load_online_snapshot_and_scalers(args.central_online_dir, uid, "central_online")
        cen_pack = None
        if cen_snap is not None:
            cen_pack = predict_online_holdout(cen_snap, xsc_cen, ysc_cen, U_all, Y_all, H, args.ctx_k, args.washout_eval)

        # --- federated ONLINE ---
        fed_snap, xsc_fed, ysc_fed = load_online_snapshot_and_scalers(args.federated_dir, uid, "federated_online")
        fed_pack = None
        if fed_snap is not None:
            fed_pack = predict_online_holdout(fed_snap, xsc_fed, ysc_fed, U_all, Y_all, H, args.ctx_k, args.washout_eval)

        df_user = align_and_make_df(uid, Y_all, H, off_pack, cen_pack, fed_pack)
        if df_user is None or df_user.empty:
            print(f"[WARN] user {uid}: unable to align predictions (missing artifacts or incompatible windows).")
            continue

        out_path = os.path.join(args.out_dir, f"predictions_user_{uid}.csv")
        df_user.to_csv(out_path, index=False)
        print(f"[OK] {out_path} ({len(df_user)} rows)")

    print("[DONE] Prediction files generated in:", args.out_dir)

    # aggregate per-user CSVs into users_location.csv in the specified order
    try:
        user_ids = [int(s.strip()) for s in args.user_ids.split(",") if s.strip() != ""]
    except Exception:
        raise RuntimeError("Invalid --user_ids parameter. Provide comma-separated integers, e.g.: 0,1,2,...,49")
    aggregate_users_location(args.out_dir, user_ids)

if __name__ == "__main__":
    main()
