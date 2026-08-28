# train_central_online.py
# -*- coding: utf-8 -*-
"""
Online (streaming) training per user with ESN + RLS.

Training uses 80% of samples (temporal split, no shuffling). A3-like association
partitions training timesteps by RU across rounds. Global scalers are frozen per
round using only training statistics (n, Σ, Σ²) — no holdout data leaks into
normalization. Evaluation uses a context bridge (K) + washout over the 20% holdout.

Output per user: user_id, horizon, rmse_x_m, rmse_y_m, rmse_euclid_m.
"""

from __future__ import annotations
import os
import argparse
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from model.ESNStreaming import ESNStreaming, rmse

# Input features and targets
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

# ---------- util: numeric floor on scaler scales ----------
def apply_scale_floor(scaler: StandardScaler, eps: float, who: str, uid: int, t: int) -> None:
    if not hasattr(scaler, "scale_") or scaler.scale_ is None:
        return
    scale = scaler.scale_
    bad = (scale < eps) | ~np.isfinite(scale)
    if np.any(bad):
        print(f"[WARN] scaler '{who}' user {uid} t={t}: scale_ min={np.nanmin(scale):.3e} -> floor={eps:.3e}.")
        scaler.scale_[bad] = eps
        if hasattr(scaler, "var_") and scaler.var_ is not None:
            scaler.var_[bad] = eps ** 2

# ---------- holdout evaluation with context bridge + washout ----------
def eval_on_validation_with_context(
    esn_snap: dict,
    xsc_bytes: bytes,
    ysc_bytes: bytes,
    U_all: np.ndarray,
    Y_all: np.ndarray,
    ntr_pairs: int,
    washout_eval: int,
    ctx_k: int,
    horizon: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray, int]:
    esn_eval = ESNStreaming(
        n_inputs=U_all.shape[1],
        n_reservoir=esn_snap["W"].shape[0],
        n_outputs=2,
        spectral_radius=0.9, sparsity=0.1, leak_rate=0.3,
        include_inputs_in_state=True, seed=42, rls_lambda=0.99, rls_alpha=1e-3
    )
    esn_eval.load_snapshot(esn_snap)

    x_scaler_eval: StandardScaler = pickle.loads(xsc_bytes)
    y_scaler_eval: StandardScaler = pickle.loads(ysc_bytes)

    T_all = U_all.shape[0]
    H = horizon

    U_va = U_all[ntr_pairs : T_all - H]
    Y_va = Y_all[ntr_pairs + H : T_all]

    K = int(max(0, min(ctx_k, ntr_pairs)))
    if K > 0:
        U_ctx = U_all[ntr_pairs - K : ntr_pairs]
        U_ctx_s = x_scaler_eval.transform(U_ctx)
    else:
        U_ctx_s = np.zeros((0, U_all.shape[1]), dtype=np.float64)

    U_va_s = x_scaler_eval.transform(U_va)
    U_concat = np.vstack([U_ctx_s, U_va_s])

    preds_concat_s = esn_eval.predict_segment_scaled(U_concat, washout=K + washout_eval)
    yhat_va = y_scaler_eval.inverse_transform(preds_concat_s)

    Y_eff = Y_va[washout_eval:]
    n_eff = min(len(Y_eff), len(yhat_va))
    Y_eff = Y_eff[:n_eff]
    yhat_eff = yhat_va[:n_eff]

    t0_eff = ntr_pairs + washout_eval
    return (
        rmse(yhat_eff[:, 0], Y_eff[:, 0]),
        rmse(yhat_eff[:, 1], Y_eff[:, 1]),
        rmse(yhat_eff, Y_eff),
        Y_eff,
        yhat_eff,
        t0_eff,
    )

# ---------- running statistics aggregator (no raw data) ----------
@dataclass
class RunningStats:
    n: int
    sum: np.ndarray
    sumsq: np.ndarray
    @classmethod
    def zeros(cls, d: int): return cls(0, np.zeros(d, dtype=np.float64), np.zeros(d, dtype=np.float64))
    def add_batch(self, X: np.ndarray):
        if X.size == 0: return
        self.n    += int(X.shape[0])
        self.sum  += X.sum(axis=0)
        self.sumsq+= (X*X).sum(axis=0)
    def merge(self, other: "RunningStats"):
        self.n += other.n; self.sum += other.sum; self.sumsq += other.sumsq
    def mean_var(self):
        if self.n <= 1:
            mean = self.sum / max(self.n,1)
            var  = np.maximum(self.sumsq / max(self.n,1) - mean*mean, 1e-6)
            return mean, var
        mean = self.sum/self.n
        var  = self.sumsq/self.n - mean*mean
        return mean, np.maximum(var, 1e-6)

def scaler_from_mean_var(mean: np.ndarray, var: np.ndarray) -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = mean.astype(np.float64)
    sc.var_  = var.astype(np.float64)
    sc.scale_= np.sqrt(sc.var_)
    sc.n_features_in_ = mean.shape[0]
    return sc

# ---------- A3-like association (no normalization) ----------
def a3_like_association(user_xy: np.ndarray, ru_xy: np.ndarray, d_hyst: float=5.0, ttt: int=2) -> np.ndarray:
    T = user_xy.shape[0]
    dists = np.linalg.norm(user_xy[:,None,:] - ru_xy[None,:,:], axis=2)  # (T,Nru)
    best  = np.argmin(dists, axis=1)
    assoc = np.zeros(T, dtype=int)
    cur = best[0]; hold = 0
    for t in range(T):
        cand = best[t]
        if dists[t, cand] + d_hyst < dists[t, cur]:
            hold += 1
            if hold >= ttt: cur = cand; hold = 0
        else:
            hold = 0
        assoc[t] = cur
    return assoc

# ================================ MAIN =====================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/derived_features_full.csv")
    ap.add_argument("--out", default="model/centralized")

    # ESN
    ap.add_argument("--reservoir", type=int, default=600)
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--radius", type=float, default=0.9)
    ap.add_argument("--sparsity", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # Streaming / RLS
    ap.add_argument("--lambda_rls", type=float, default=0.99)
    ap.add_argument("--alpha_rls", type=float, default=1e-3)
    ap.add_argument("--scale_eps", type=float, default=1e-3)

    # Evaluation
    ap.add_argument("--washout_eval", type=int, default=120)
    ap.add_argument("--ctx_k", type=int, default=300, help="context bridge (K last steps of the 80% training)")

    # Horizon
    ap.add_argument("--steps_ahead", type=int, default=3,
                    help="Horizon H in steps/seconds for prediction Y(t+H) from U(t)")

    ap.add_argument("--rounds", type=int, default=100,
                    help="number of rounds (contiguous temporal slices in the 80% training)")
    ap.add_argument("--ru_csv", default="data/ru_positions.csv")
    ap.add_argument("--a3_hysteresis_m", type=float, default=5.0)
    ap.add_argument("--a3_ttt", type=int, default=2)

    args = ap.parse_args()

    # Output directories
    os.makedirs(args.out, exist_ok=True)
    models_dir  = os.path.join(args.out, "models");  os.makedirs(models_dir, exist_ok=True)
    scalers_dir = os.path.join(args.out, "scalers"); os.makedirs(scalers_dir, exist_ok=True)
    plots_dir   = os.path.join(args.out, "plots");   os.makedirs(plots_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv)
    need = set(["id"] + FEATURES + TARGETS)
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in CSV: {sorted(miss)}")
    user_ids = sorted(df["id"].unique().tolist())
    print(f"[INFO] Users: {user_ids}")

    # RUs (for A3-like association and validation RU owner)
    ru_tab = pd.read_csv(args.ru_csv).sort_values("ru_id")
    if set(["ru_id","x","y"]) - set(ru_tab.columns):
        raise RuntimeError("ru_csv must contain columns: ru_id,x,y")
    ru_xy = ru_tab[["x","y"]].to_numpy(np.float64)
    Nru = ru_xy.shape[0]

    H = max(1, int(args.steps_ahead))

    # Buffers for aggregate scatter plot
    agg_val_real_x, agg_val_pred_x = [], []
    agg_val_real_y, agg_val_pred_y = [], []

    summary_rows = []

    # --- Per-user preparation: U/Y, A3-like association, 80/20 split, RU owner ---
    U_all_by_u: Dict[int,np.ndarray] = {}
    Y_all_by_u: Dict[int,np.ndarray] = {}
    assoc_by_u: Dict[int,np.ndarray] = {}
    n_inputs_by_u: Dict[int,int]     = {}
    owner_ru_by_u: Dict[int,int]     = {}

    for uid in user_ids:
        du = df[df["id"] == uid].copy()
        U_all = du[FEATURES].to_numpy(dtype=np.float64)
        Y_all = du[TARGETS].to_numpy(dtype=np.float64)
        U_all_by_u[uid] = U_all
        Y_all_by_u[uid] = Y_all
        n_inputs_by_u[uid] = U_all.shape[1]

        T_all = U_all.shape[0]
        if T_all <= H:
            print(f"[WARN] user {uid}: short sequence (T={T_all} <= H={H}). Skipping.")
            continue

        assoc = a3_like_association(Y_all, ru_xy, args.a3_hysteresis_m, args.a3_ttt)
        assoc_by_u[uid] = assoc

        T_pairs = T_all - H
        ntr_pairs = int(0.8 * T_pairs)
        owner_ru_by_u[uid] = int(assoc[ntr_pairs - 1])  # RU that receives the last training window

    # --- Slice the 80% training by RU and round (contiguous in time) ---
    ts_train_by_ru: List[Dict[int,np.ndarray]] = [ {} for _ in range(Nru) ]
    ts_val_by_ru:   List[Dict[int,np.ndarray]] = [ {} for _ in range(Nru) ]
    for uid in user_ids:
        U_all = U_all_by_u[uid]; Y_all = Y_all_by_u[uid]
        if U_all is None: continue
        T_pairs = U_all.shape[0] - H
        if T_pairs <= 0: continue
        ntr_pairs = int(0.8 * T_pairs)

        # training: distribute t by RU in the 80%
        assoc = assoc_by_u[uid]
        for ruid in range(Nru):
            ts = np.where(assoc[:ntr_pairs] == ruid)[0]
            ts_train_by_ru[ruid][uid] = ts

        # holdout (20%): fully assigned to the owner RU
        ru_owner = owner_ru_by_u[uid]
        ts_val_by_ru[ru_owner][uid] = np.arange(ntr_pairs, T_pairs, dtype=int)

    rounds = max(1, int(args.rounds))
    # slices by RU and round: {ru}[round]{uid:(s,e)} contiguous in the ts_train vector
    slices_by_ru: List[List[Dict[int,Tuple[int,int]]]] = []
    for ruid in range(Nru):
        per_round = [ {} for _ in range(rounds) ]
        for uid, ts_all in ts_train_by_ru[ruid].items():
            L = len(ts_all)
            sizes = [L//rounds + (1 if i < (L%rounds) else 0) for i in range(rounds)]
            st = 0
            for rd in range(rounds):
                en = st + sizes[rd]
                per_round[rd][uid] = (st, en)
                st = en
        slices_by_ru.append(per_round)

    # --- Shared ESN base (Win, W) for all users; per-user initial Wout ---
    # dummy ESN to freeze identical Win/W across all rounds
    dummy = ESNStreaming(
        n_inputs=n_inputs_by_u[user_ids[0]], n_reservoir=args.reservoir, n_outputs=2,
        spectral_radius=args.radius, sparsity=args.sparsity, leak_rate=args.leak,
        include_inputs_in_state=True, seed=args.seed, rls_lambda=args.lambda_rls, rls_alpha=args.alpha_rls
    )
    Win, W = dummy.Win.copy(), dummy.W.copy()

    Wout_by_u: Dict[int,np.ndarray] = {}
    for uid in user_ids:
        esn_tmp = ESNStreaming(
            n_inputs=n_inputs_by_u[uid], n_reservoir=args.reservoir, n_outputs=2,
            spectral_radius=args.radius, sparsity=args.sparsity, leak_rate=args.leak,
            include_inputs_in_state=True, seed=args.seed + int(uid),
            rls_lambda=args.lambda_rls, rls_alpha=args.alpha_rls
        )
        esn_tmp.Win, esn_tmp.W = Win.copy(), W.copy()
        Wout_by_u[uid] = esn_tmp.Wout.copy()

    # --- cumulative statistics per user (training only) ---
    stats_in_by_u  = {uid: RunningStats.zeros(n_inputs_by_u[uid]) for uid in user_ids}
    stats_out_by_u = {uid: RunningStats.zeros(2)                  for uid in user_ids}

    # ========================= ROUND LOOP =========================
    for rd in range(rounds):
        print(f"\n===== ROUND {rd+1}/{rounds} (central) =====")

        # (A) aggregate statistics from training timesteps in this round (t < ntr_pairs)
        delta_in_by_u  = {uid: RunningStats.zeros(n_inputs_by_u[uid]) for uid in user_ids}
        delta_out_by_u = {uid: RunningStats.zeros(2)                  for uid in user_ids}

        for ruid in range(Nru):
            for uid, (s, e) in slices_by_ru[ruid][rd].items():
                ts_all = ts_train_by_ru[ruid][uid]
                if e <= s or len(ts_all) == 0:
                    continue
                ts = ts_all[s:e]
                T_user = U_all_by_u[uid].shape[0]
                # valid training pairs (no leakage)
                ts = ts[ ts + H < T_user ]
                if ts.size == 0: continue
                Uc = U_all_by_u[uid][ts]
                Yc = Y_all_by_u[uid][ts + H]
                delta_in_by_u[uid].add_batch(Uc)
                delta_out_by_u[uid].add_batch(Yc)

        # update cumulatives + build round-global scalers (frozen within round)
        xsc_by_u: Dict[int, StandardScaler] = {}
        ysc_by_u: Dict[int, StandardScaler] = {}
        for uid in user_ids:
            stats_in_by_u[uid].merge(delta_in_by_u[uid])
            stats_out_by_u[uid].merge(delta_out_by_u[uid])
            mean_x, var_x = stats_in_by_u[uid].mean_var()
            mean_y, var_y = stats_out_by_u[uid].mean_var()
            xsc = scaler_from_mean_var(mean_x, var_x);  apply_scale_floor(xsc, args.scale_eps, "X(round)", uid, rd)
            ysc = scaler_from_mean_var(mean_y, var_y);  apply_scale_floor(ysc, args.scale_eps, "Y(round)", uid, rd)
            xsc_by_u[uid] = xsc; ysc_by_u[uid] = ysc

        # (B) train in this round (only these timesteps), per user, in global temporal order
        for uid in user_ids:
            # collect timesteps for this round from ALL RUs and sort temporally
            ts_round: List[int] = []
            for ruid in range(Nru):
                s, e = slices_by_ru[ruid][rd].get(uid, (0,0))
                ts_all = ts_train_by_ru[ruid].get(uid, np.zeros((0,), dtype=int))
                if e <= s or len(ts_all) == 0:
                    continue
                ts = ts_all[s:e]
                T_user = U_all_by_u[uid].shape[0]
                ts = ts[ ts + H < T_user ]  # valid training pairs
                ts_round.extend(ts.tolist())
            if not ts_round:
                continue
            ts_round = np.array(sorted(ts_round), dtype=int)

            esn = ESNStreaming(
                n_inputs=n_inputs_by_u[uid], n_reservoir=args.reservoir, n_outputs=2,
                spectral_radius=args.radius, sparsity=args.sparsity, leak_rate=args.leak,
                include_inputs_in_state=True, seed=args.seed + int(uid),
                rls_lambda=args.lambda_rls, rls_alpha=args.alpha_rls
            )
            esn.Win, esn.W, esn.Wout = Win.copy(), W.copy(), Wout_by_u[uid].copy()

            xsc = xsc_by_u[uid]; ysc = ysc_by_u[uid]
            for t in ts_round:
                u_s = xsc.transform(U_all_by_u[uid][t:t+1])[0]
                yhat_s, z = esn.predict_scaled_and_features(u_s)
                y_true_s  = ysc.transform(Y_all_by_u[uid][t + H].reshape(1, -1))[0]
                esn.rls_update(z, y_true_s)

            Wout_by_u[uid] = esn.Wout.copy()

    # ========================= FINAL EVALUATION =========================
    for uid in user_ids:
        U_all = U_all_by_u[uid]; Y_all = Y_all_by_u[uid]
        if U_all is None: continue
        T_pairs = U_all.shape[0] - H
        if T_pairs <= 0: continue
        ntr_pairs = int(0.8 * T_pairs)

        # final global snapshot and scalers (training only)
        best_snap = {"Win": Win.copy(), "W": W.copy(), "Wout": Wout_by_u[uid].copy()}
        mean_x, var_x = stats_in_by_u[uid].mean_var()
        mean_y, var_y = stats_out_by_u[uid].mean_var()
        xsc_final = scaler_from_mean_var(mean_x, var_x); apply_scale_floor(xsc_final, args.scale_eps, "X(final)", uid, -1)
        ysc_final = scaler_from_mean_var(mean_y, var_y); apply_scale_floor(ysc_final, args.scale_eps, "Y(final)", uid, -1)
        best_xsc  = pickle.dumps(xsc_final)
        best_ysc  = pickle.dumps(ysc_final)

        rmse_x_val, rmse_y_val, rmse_e_val, Y_eff, yhat_eff, t0_eff = eval_on_validation_with_context(
            best_snap, best_xsc, best_ysc, U_all, Y_all,
            ntr_pairs=ntr_pairs, washout_eval=args.washout_eval, ctx_k=args.ctx_k, horizon=H,
        )
        print(f"[RESULT central-rounds] user {uid} | H={H} | Val RMSE_x={rmse_x_val:.3f} m | RMSE_y={rmse_y_val:.3f} m | RMSE_e={rmse_e_val:.3f} m")

        # per-user plot
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(Y_eff[:, 0], Y_eff[:, 1], label="Real")
        ax.plot(yhat_eff[:, 0], yhat_eff[:, 1], label="Predicted")
        ax.set_title(f"User {uid} - Validation (20%) with bridge (K={args.ctx_k}) - H={H}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend()
        plt.savefig(os.path.join(plots_dir, f"user_{uid}_val_xy_central_online.png"), bbox_inches="tight", dpi=120)
        plt.close(fig)

        # save artifacts
        with open(os.path.join(models_dir,  f"central_online_user_{uid}.pkl"), "wb") as f:
            pickle.dump(best_snap, f)
        with open(os.path.join(scalers_dir, f"central_online_x_scaler_user_{uid}.pkl"), "wb") as f:
            f.write(best_xsc)
        with open(os.path.join(scalers_dir, f"central_online_y_scaler_user_{uid}.pkl"), "wb") as f:
            f.write(best_ysc)

        summary_rows.append({
            "user_id": uid,
            "horizon": H,
            "rmse_x_m": rmse_x_val,
            "rmse_y_m": rmse_y_val,
            "rmse_euclid_m": rmse_e_val,
        })

        # accumulate for scatter plot
        agg_val_real_x.extend(Y_eff[:, 0].tolist()); agg_val_pred_x.extend(yhat_eff[:, 0].tolist())
        agg_val_real_y.extend(Y_eff[:, 1].tolist()); agg_val_pred_y.extend(yhat_eff[:, 1].tolist())

    # Global summary
    summary_df = pd.DataFrame(summary_rows).sort_values("user_id").reset_index(drop=True)
    summary_df.to_csv(os.path.join(args.out, "summary_central_online.csv"), index=False)
    print("[DONE] summary_central_online.csv saved.")

    # Aggregate scatter plot
    if len(agg_val_real_x) > 0:
        real_all = np.concatenate([np.asarray(agg_val_real_x, dtype=float),
                                   np.asarray(agg_val_real_y, dtype=float)])
        pred_all = np.concatenate([np.asarray(agg_val_pred_x, dtype=float),
                                   np.asarray(agg_val_pred_y, dtype=float)])
        fig = plt.figure(); ax = plt.gca()
        ax.scatter(real_all, pred_all, s=8, alpha=0.25)
        mn = float(min(real_all.min(), pred_all.min()))
        mx = float(max(real_all.max(), pred_all.max()))
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
        ax.set_title("Real vs Predicted - All users (20% validation with bridge)")
        ax.set_xlabel("Real [m]"); ax.set_ylabel("Predicted [m]")
        plt.savefig(os.path.join(plots_dir, "all_users_pred_vs_real_central_online.png"), bbox_inches="tight", dpi=120)
        plt.close(fig)
        print("[PLOT] Aggregate plot saved to: plots/all_users_pred_vs_real_central_online.png")

if __name__ == "__main__":
    main()
