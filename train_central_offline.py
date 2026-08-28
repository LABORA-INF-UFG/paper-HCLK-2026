# train_central_offline.py
# -*- coding: utf-8 -*-
"""
Trains one ESN per user to predict (x,y) at H steps ahead.
Temporal split: 80% training, 20% validation (no shuffling).

Outputs per user:
  - models/esn_user_<id>.pkl           (ESN model)
  - scalers/x_scaler_user_<id>.pkl     (StandardScaler for inputs)
  - scalers/y_scaler_user_<id>.pkl     (StandardScaler for targets)
  - plots/user_<id>_val_xy.png         (2D trajectory: Real vs. Predicted, validation)
And a summary CSV with metrics (RMSE in meters).

Usage:
  python train_central_offline.py --csv data/derived_features_full.csv --out model/centralized \
      --reservoir 600 --washout 50 --alpha 1e-4 --leak 0.3 --radius 0.9 --sparsity 0.1 --seed 42 --progress 5
"""

from __future__ import annotations
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import ESN class and rmse utility
from model.ESNStreaming import ESNStreaming, rmse

# Model inputs (raw x,y are excluded from features to avoid leaking the target at the same timestep)
FEATURES = [
    "z_x", "z_y",
    "z_vx", "z_vy", "z_speed",
    "z_ax", "z_ay", "z_accel",
    "z_turn_rate",
    "cos_heading", "sin_heading",
    "cos_o", "sin_o",
    "cos_gaze_move", "sin_gaze_move"
]
TARGETS = ["x", "y"]  # targets in meters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/derived_features_full.csv")
    ap.add_argument("--out", default="model/centralized")
    ap.add_argument("--reservoir", type=int, default=600)
    ap.add_argument("--washout",   type=int, default=120)
    ap.add_argument("--alpha",     type=float, default=1e-4)
    ap.add_argument("--leak",      type=float, default=0.3)
    ap.add_argument("--radius",    type=float, default=0.9)
    ap.add_argument("--sparsity",  type=float, default=0.1)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--progress",  type=int, default=5, help="Progress checkpoints (training loss log)")
    ap.add_argument("--steps_ahead", type=int, default=3,
                    help="Horizon H in steps/seconds for prediction Y(t+H) from U(t)")
    args = ap.parse_args()

    # Output directories
    os.makedirs(args.out, exist_ok=True)
    plots_dir   = os.path.join(args.out, "plots");   os.makedirs(plots_dir, exist_ok=True)
    models_dir = os.path.join(args.out, "models");  os.makedirs(models_dir, exist_ok=True)
    scalers_dir= os.path.join(args.out, "scalers"); os.makedirs(scalers_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv)
    need = set(["id"] + FEATURES + TARGETS)
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in CSV: {sorted(miss)}")

    user_ids = sorted(df["id"].unique().tolist())
    print(f"[INFO] Users: {user_ids}")

    rows = []

    # accumulators for aggregate plot
    agg_real = []  # each entry: 1D array of Real values (x or y) from post-washout validation
    agg_pred = []  # each entry: 1D array of Predicted values (x or y) from post-washout validation

    # horizon H (maintains causality U(t)->Y(t+H))
    H = max(1, int(args.steps_ahead))

    for uid in user_ids:
        d = df[df["id"] == uid].copy()

        # Inputs (U) and Targets (Y)
        U_all = d[FEATURES].to_numpy(dtype=np.float64)
        Y_all = d[TARGETS].to_numpy(dtype=np.float64)

        # H-step prediction: Y_{t+H} <- U_t
        U_seq = U_all[:-H]
        Y_seq = Y_all[ H:]

        # Temporal 80/20 split
        T = U_seq.shape[0]
        if T <= 0:
            print(f"[WARN] User {uid} | T-H <= 0 (H={H}). Skipping user.")
            continue
        ntr = int(0.8 * T)
        U_tr, Y_tr = U_seq[:ntr], Y_seq[:ntr]
        U_va, Y_va = U_seq[ntr:], Y_seq[ntr:]

        # Per-user normalization
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        U_tr_s = x_scaler.fit_transform(U_tr)
        Y_tr_s = y_scaler.fit_transform(Y_tr)
        U_va_s = x_scaler.transform(U_va)

        # ESN (streaming with RLS)
        esn = ESNStreaming(
            n_inputs=U_tr_s.shape[1],
            n_reservoir=args.reservoir,
            n_outputs=2,
            spectral_radius=args.radius,
            sparsity=args.sparsity,
            leak_rate=args.leak,
            include_inputs_in_state=True,
            seed=args.seed + int(uid),
            # maps alpha (offline ridge) to RLS alpha (P0 = 1/alpha I)
            rls_alpha=args.alpha,
            rls_lambda=1  # no forgetting: the offline paradigm weighs the entire history equally
        )

        # Streaming training with washout
        # warm-up: advance state without updating Wout
        for t in range(min(args.washout, U_tr_s.shape[0])):
            esn.predict_scaled_and_features(U_tr_s[t])

        # RLS: from washout onwards, update sample-by-sample
        for t in range(args.washout, U_tr_s.shape[0]):
            yhat_s, z = esn.predict_scaled_and_features(U_tr_s[t])
            esn.rls_update(z, Y_tr_s[t])

        # Save artifacts
        model_path = os.path.join(models_dir, f"central_offline_user_{uid}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(esn, f)
        xsc_path = os.path.join(scalers_dir, f"central_offline_x_scaler_user_{uid}.pkl")
        ysc_path = os.path.join(scalers_dir, f"central_offline_y_scaler_user_{uid}.pkl")
        with open(xsc_path, "wb") as f:
            pickle.dump(x_scaler, f)
        with open(ysc_path, "wb") as f:
            pickle.dump(y_scaler, f)

        # Validation (washout discards the transient)
        preds_s = esn.predict_segment_scaled(U_va_s, washout=args.washout)
        preds   = y_scaler.inverse_transform(preds_s)
        Y_eff   = Y_va[args.washout:]   # align with washout

        rmse_x_val = rmse(preds[:, 0], Y_eff[:, 0])
        rmse_y_val = rmse(preds[:, 1], Y_eff[:, 1])
        rmse_e_val = rmse(preds, Y_eff)
        print(f"[RESULT] User {uid} | H={H} | Val RMSE_x={rmse_x_val:.4f} m | RMSE_y={rmse_y_val:.4f} m | RMSE_e={rmse_e_val:.4f} m")

        rows.append({
            "user_id": uid,
            "horizon": H,
            "n_train": int(U_tr.shape[0]),
            "n_val": int(U_va.shape[0]),
            "washout": args.washout,
            "rmse_x_m": rmse_x_val,
            "rmse_y_m": rmse_y_val,
            "rmse_euclid_m": rmse_e_val,
        })

        # accumulate for aggregate plot
        agg_real.append(Y_eff[:, 0])   # x real
        agg_pred.append(preds[:, 0])   # x predicted
        agg_real.append(Y_eff[:, 1])   # y real
        agg_pred.append(preds[:, 1])   # y predicted

        # Per-user 2D validation trajectory plot
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(Y_eff[:, 0], Y_eff[:, 1], label="Real")
        ax.plot(preds[:, 0], preds[:, 1], label="Predicted")
        ax.set_title(f"User {uid} - Validation Trajectory (Real vs Predicted)")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend()
        plt.savefig(os.path.join(plots_dir, f"user_{uid}_val_xy_central_offline.png"), bbox_inches="tight", dpi=120)
        plt.close(fig)

    # Summary
    summary = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
    summary_path = os.path.join(args.out, "summary_central_offline.csv")
    summary.to_csv(summary_path, index=False)
    print("[DONE] Validation metrics saved to:", summary_path)

    # Aggregate Predicted vs Real scatter plot
    if len(agg_real) > 0 and len(agg_pred) > 0:
        real_all = np.concatenate(agg_real)  # all real values (x and y) from validation, post-washout
        pred_all = np.concatenate(agg_pred)  # all predicted values (x and y) from validation, post-washout

        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(real_all, pred_all, s=8, alpha=0.25)

        mn = float(min(real_all.min(), pred_all.min()))
        mx = float(max(real_all.max(), pred_all.max()))
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2)

        ax.set_title("Real vs Predicted - All users (validation)")
        ax.set_xlabel("Real [m]")
        ax.set_ylabel("Predicted [m]")

        out_path = os.path.join(plots_dir, "all_users_pred_vs_real_central_offline.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print("[PLOT] Aggregate plot saved to:", out_path)

if __name__ == "__main__":
    main()
