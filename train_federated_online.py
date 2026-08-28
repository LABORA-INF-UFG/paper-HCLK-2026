# train_federated_online.py
# -*- coding: utf-8 -*-
"""
Federated online training with ESNStreaming (fixed reservoir + RLS readout).

- CU (server): holds global Win/W, orchestrates rounds and performs FedAvg of Wout per user.
  Maintains global scalers per user by aggregating only statistics (no raw data):
  (n, sum, sum-of-squares) for inputs and targets.

- RUs (clients): receive Win/W, Wout_global per user and global scalers per user each round.
  In each round, for each served user, the RU:
   (Phase A) computes statistics deltas for the round chunk and sends to CU (no training).
   (Phase B) after receiving round scalers, trains the readout (RLS) on the temporal chunk,
             and sends Wout_local + sample count (m) to CU (weighted FedAvg by m).

- A3-like association (simplified, no data normalization):
  selects RU by nearest distance with hysteresis (d_hyst) and time-to-trigger (TTT) to avoid
  ping-pong. Temporal index 't' is the row index in the CSV for (id == user), 0-based.

- 80/20 temporal split per user with horizon H:
    T_pairs = T_user - H
    training:   pairs U(t) -> Y(t+H) for t in [0, ntr_pairs-1], ntr_pairs = floor(0.8 * T_pairs)
    validation: t in [ntr_pairs, T_pairs-1]; full holdout is owned by the RU at t = ntr_pairs-1.

- Evaluation uses a context bridge (K) + washout_eval over the holdout.
  Per-user metrics (rmse_x, rmse_y, rmse_euclid) are written to summary_federated_online.csv.

Requirements:
- Python 3.9+
- numpy, pandas, scikit-learn, matplotlib
- ESNStreaming class (API: predict_scaled_and_features, rls_update, predict_segment_scaled,
  snapshot, load_snapshot)

Usage:
  python train_federated_online.py --csv data/derived_features_full.csv --ru_csv data/ru_positions.csv \
    --out model/federated --num_rus 9 --rounds 100 --steps_ahead 3 --ctx_k 300 --washout_eval 120

Notes:
- Federated scalers are per USER (not per RU), updated each round and frozen within the round
  to ensure FedAvg compatibility.
- No raw data is exchanged: only Wout (readout) and aggregated statistics (n, Σ, Σ²).
"""

from __future__ import annotations
import os, json, math, argparse, pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from model.ESNStreaming import ESNStreaming, rmse

# ==== Feature/target configuration (same as centralized online) ====
FEATURES = [
    "z_x", "z_y",
    "z_vx", "z_vy", "z_speed",
    "z_ax", "z_ay", "z_accel",
    "z_turn_rate",
    "cos_heading", "sin_heading",
    "cos_o", "sin_o",
    "cos_gaze_move", "sin_gaze_move"
]
TARGETS = ["x", "y"]   # in meters

# ========= Util: statistics aggregation (no raw data) =========
@dataclass
class RunningStats:
    n: int
    sum: np.ndarray     # sum per dimension
    sumsq: np.ndarray   # sum of squares per dimension

    @classmethod
    def zeros(cls, d: int) -> "RunningStats":
        return cls(n=0, sum=np.zeros(d, dtype=np.float64), sumsq=np.zeros(d, dtype=np.float64))

    def add_batch(self, X: np.ndarray):
        # X: shape (N, d)
        if X.size == 0:
            return
        self.n    += int(X.shape[0])
        self.sum  += np.sum(X, axis=0)
        self.sumsq+= np.sum(X * X, axis=0)

    def merge(self, other: "RunningStats"):
        self.n     += other.n
        self.sum   += other.sum
        self.sumsq += other.sumsq

    def mean_var(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n <= 1:
            # avoid degenerate scales
            mean = self.sum / max(self.n, 1)
            var  = np.maximum(self.sumsq / max(self.n, 1) - mean*mean, 1e-6)
            return mean, var
        mean = self.sum / self.n
        var  = self.sumsq / self.n - mean * mean
        # numerical clamp
        var  = np.maximum(var, 1e-6)
        return mean, var

def scaler_from_mean_var(mean: np.ndarray, var: np.ndarray) -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = mean.astype(np.float64)
    sc.var_  = var.astype(np.float64)
    sc.scale_= np.sqrt(sc.var_)
    sc.n_features_in_ = mean.shape[0]
    return sc

# ========= A3-like association (simplified, no normalization) =========
def a3_like_association(user_xy: np.ndarray,
                        ru_xy: np.ndarray,
                        d_hyst: float = 5.0,
                        ttt: int = 2) -> np.ndarray:
    """
    user_xy: (T, 2) user positions over time (x,y)
    ru_xy:   (N_ru, 2) RU positions
    Returns: array (T,) with ru_id (0..N_ru-1) associated to each t, applying:
      - selection by nearest distance,
      - hysteresis d_hyst (switches only if candidate RU is at least d_hyst meters better),
      - time-to-trigger (advantage must hold for 'ttt' consecutive steps).
    """
    T = user_xy.shape[0]
    N = ru_xy.shape[0]
    dists = np.linalg.norm(user_xy[:, None, :] - ru_xy[None, :, :], axis=2)  # (T,N)
    best  = np.argmin(dists, axis=1)  # nearest RU per t

    assoc = np.zeros(T, dtype=int)
    current = best[0]
    hold_count = 0
    for t in range(T):
        cand = best[t]
        # check hysteresis: switch only if candidate is significantly better
        if dists[t, cand] + d_hyst < dists[t, current]:
            hold_count += 1
            if hold_count >= ttt:
                current = cand
                hold_count = 0
        else:
            hold_count = 0
        assoc[t] = current
    return assoc

# ========= Context bridge + holdout evaluation =========
def eval_on_validation_with_context(esn_snap: dict,
                                    xsc_bytes: bytes,
                                    ysc_bytes: bytes,
                                    U_all: np.ndarray,
                                    Y_all: np.ndarray,
                                    ntr_pairs: int,
                                    washout_eval: int,
                                    ctx_k: int,
                                    horizon: int) -> Tuple[float,float,float,np.ndarray,np.ndarray,int]:
    """
    Evaluates the holdout [ntr_pairs .. T_pairs-1] with context bridge K.
    H-step alignment: U_va[t] -> Y_va[t+H].
    """
    # Evaluation ESN uses the provided snapshot (Wout)
    esn_eval = ESNStreaming(
        n_inputs=U_all.shape[1], n_reservoir=esn_snap["W"].shape[0], n_outputs=2,
        spectral_radius=0.9, sparsity=0.1, leak_rate=0.3,
        include_inputs_in_state=True, seed=42, rls_lambda=0.99, rls_alpha=1e-3
    )
    esn_eval.load_snapshot(esn_snap)

    xsc_eval: StandardScaler = pickle.loads(xsc_bytes)
    ysc_eval: StandardScaler = pickle.loads(ysc_bytes)

    T_all = U_all.shape[0]
    H = int(horizon)
    # Ranges: validation uses U in [ntr_pairs .. T_all-H), Y in [ntr_pairs+H .. T_all)
    U_va = U_all[ntr_pairs : T_all - H]
    Y_va = Y_all[ntr_pairs + H : T_all]

    # Bridge K: last K training steps
    K = max(0, min(ctx_k, ntr_pairs))
    if K > 0:
        U_ctx = U_all[ntr_pairs - K : ntr_pairs]
        U_ctx_s = xsc_eval.transform(U_ctx)
    else:
        U_ctx_s = np.zeros((0, U_all.shape[1]), dtype=np.float64)

    U_va_s = xsc_eval.transform(U_va)
    U_concat = np.vstack([U_ctx_s, U_va_s])

    preds_concat_s = esn_eval.predict_segment_scaled(U_concat, washout=K + washout_eval)
    yhat_va = ysc_eval.inverse_transform(preds_concat_s)

    Y_eff = Y_va[washout_eval:]
    n_eff = min(len(Y_eff), len(yhat_va))
    Y_eff = Y_eff[:n_eff]
    yhat_eff = yhat_va[:n_eff]

    t0_eff = ntr_pairs + washout_eval
    return rmse(yhat_eff[:,0], Y_eff[:,0]), rmse(yhat_eff[:,1], Y_eff[:,1]), rmse(yhat_eff, Y_eff), Y_eff, yhat_eff, t0_eff

# ========= RU (client) =========
@dataclass
class RUClient:
    ru_id: int
    users_times_train: Dict[int, np.ndarray]  # per user: array of t (0-based) in TRAINING assigned to this RU
    users_times_val:   Dict[int, np.ndarray]  # per user: array of t (0-based) in HOLDOUT (only for the owner RU)
    U_by_user: Dict[int, np.ndarray]          # full U_all per user (not sent to CU)
    Y_by_user: Dict[int, np.ndarray]          # full Y_all per user (not sent to CU)
    H: int
    # ESN/RLS hyperparameters (constant throughout federation)
    reservoir: int
    leak: float
    radius: float
    sparsity: float
    seed: int
    lambda_rls: float
    alpha_rls: float
    # Model skeleton (Win/W fixed)
    Win: np.ndarray
    W:   np.ndarray

    # Per-user state (current RU Wout — updated each round)
    Wout_by_user: Dict[int, np.ndarray] = field(default_factory=dict)

    def _init_esn(self, user_id: int, Wout_init: np.ndarray) -> ESNStreaming:
        esn = ESNStreaming(
            n_inputs=self.U_by_user[user_id].shape[1],
            n_reservoir=self.reservoir,
            n_outputs=2,
            spectral_radius=self.radius,
            sparsity=self.sparsity,
            leak_rate=self.leak,
            include_inputs_in_state=True,
            seed=self.seed + int(user_id),
            rls_lambda=self.lambda_rls,
            rls_alpha=self.alpha_rls,
        )
        esn.Win = self.Win.copy()
        esn.W   = self.W.copy()

        # --- Wout shape/orientation adjustment ---
        if Wout_init.shape == esn.Wout.shape:
            esn.Wout = Wout_init.copy()
        elif Wout_init.T.shape == esn.Wout.shape:
            esn.Wout = Wout_init.T.copy()
        else:
            # safe fallback: zero with expected shape
            esn.Wout = np.zeros_like(esn.Wout)
        # ------------------------------------------

        return esn

    # --------- Phase A: compute statistics deltas for the round chunk (no training) ----------
    def collect_stats_for_round(self,
                                global_scalers_in: Dict[int, StandardScaler],
                                global_scalers_out: Dict[int, StandardScaler],
                                round_slices: Dict[int, Tuple[int,int]]) -> Tuple[Dict[int, RunningStats], Dict[int, RunningStats], Dict[int,int]]:
        """
        round_slices: per user, (start_idx, end_idx) in this RU's training t vector (round slice in temporal order)
        Returns: stats_in[u], stats_out[u], m_by_user[u] (samples in this round)
        """
        stats_in:  Dict[int, RunningStats] = {}
        stats_out: Dict[int, RunningStats] = {}
        m_by_user: Dict[int, int]          = {}
        H = self.H

        for uid, (s, e) in round_slices.items():
            ts_all = self.users_times_train.get(uid, np.zeros((0,), dtype=int))
            if ts_all.size == 0 or s >= e:
                stats_in[uid]  = RunningStats.zeros(self.U_by_user[uid].shape[1])
                stats_out[uid] = RunningStats.zeros(2)
                m_by_user[uid] = 0
                continue

            # temporal slice for this round (as indices into ts_all)
            ts = ts_all[s:e]
            # keep only t with t+H < T (valid pairs)
            T_user = self.U_by_user[uid].shape[0]
            ts = ts[ ts + H < T_user ]
            if ts.size == 0:
                stats_in[uid]  = RunningStats.zeros(self.U_by_user[uid].shape[1])
                stats_out[uid] = RunningStats.zeros(2)
                m_by_user[uid] = 0
                continue

            U_chunk = self.U_by_user[uid][ts]           # (N, d)
            Y_chunk = self.Y_by_user[uid][ts + H]       # (N, 2)

            rs_in  = RunningStats.zeros(U_chunk.shape[1]); rs_in.add_batch(U_chunk)
            rs_out = RunningStats.zeros(2);              rs_out.add_batch(Y_chunk)
            stats_in[uid]  = rs_in
            stats_out[uid] = rs_out
            m_by_user[uid] = int(U_chunk.shape[0])

        return stats_in, stats_out, m_by_user

    # --------- Phase B: local training with global round scalers ----------
    def train_local_for_round(self,
                              Wout_global_by_user: Dict[int, np.ndarray],
                              global_scalers_in: Dict[int, StandardScaler],
                              global_scalers_out: Dict[int, StandardScaler],
                              round_slices: Dict[int, Tuple[int,int]]) -> Dict[int, np.ndarray]:
        """
        Returns Wout_local_by_user trained in this round (for FedAvg).
        """
        H = self.H
        Wout_local: Dict[int, np.ndarray] = {}

        for uid, (s, e) in round_slices.items():
            ts_all = self.users_times_train.get(uid, np.zeros((0,), dtype=int))
            if ts_all.size == 0 or s >= e:
                # nothing to train
                Wout_local[uid] = Wout_global_by_user[uid].copy()
                continue

            ts = ts_all[s:e]
            T_user = self.U_by_user[uid].shape[0]
            ts = ts[ ts + H < T_user ]
            if ts.size == 0:
                Wout_local[uid] = Wout_global_by_user[uid].copy()
                continue

            # initialize ESN with Wout_global at the start of this round
            esn = self._init_esn(uid, Wout_global_by_user[uid])

            xsc = global_scalers_in[uid]
            ysc = global_scalers_out[uid]

            for t in ts:
                u_s = xsc.transform(self.U_by_user[uid][t:t+1])[0]
                yhat_s, z = esn.predict_scaled_and_features(u_s)
                y_true = self.Y_by_user[uid][t + H]
                y_true_s = ysc.transform(y_true.reshape(1, -1))[0]
                esn.rls_update(z, y_true_s)

            Wout_local[uid] = esn.Wout.copy()

        return Wout_local

# ========= CU (server) =========
@dataclass
class CUServer:
    users: List[int]
    H: int
    n_inputs_by_user: Dict[int, int]
    n_reservoir: int
    leak: float
    radius: float
    sparsity: float
    seed: int
    lambda_rls: float
    alpha_rls: float

    # Global Win/W (fixed)
    Win: np.ndarray
    W:   np.ndarray

    # Global per-user state
    Wout_by_user: Dict[int, np.ndarray] = field(default_factory=dict)
    stats_in_by_user:  Dict[int, RunningStats] = field(default_factory=dict)
    stats_out_by_user: Dict[int, RunningStats] = field(default_factory=dict)

    def init_user_state(self):
        for uid in self.users:
            d = self.n_inputs_by_user[uid]
            # temporary ESN to obtain the correct Wout shape (z_dim x n_outputs)
            esn_tmp = ESNStreaming(
                n_inputs=d,
                n_reservoir=self.n_reservoir,
                n_outputs=2,
                spectral_radius=self.radius,
                sparsity=self.sparsity,
                leak_rate=self.leak,
                include_inputs_in_state=True,
                seed=self.seed + int(uid),
                rls_lambda=self.lambda_rls,
                rls_alpha=self.alpha_rls,
            )
            esn_tmp.Win = self.Win.copy()
            esn_tmp.W   = self.W.copy()

            # ZEROS with the exact shape expected by the ESN
            self.Wout_by_user[uid] = np.zeros_like(esn_tmp.Wout)

            # global stats (federated scaler)
            self.stats_in_by_user[uid]  = RunningStats.zeros(d)
            self.stats_out_by_user[uid] = RunningStats.zeros(2)

    def aggregate_scaler_stats(self, delta_in: Dict[int, RunningStats], delta_out: Dict[int, RunningStats]):
        for uid in self.users:
            if uid in delta_in:
                self.stats_in_by_user[uid].merge(delta_in[uid])
            if uid in delta_out:
                self.stats_out_by_user[uid].merge(delta_out[uid])

    def make_global_scalers(self) -> Tuple[Dict[int, StandardScaler], Dict[int, StandardScaler]]:
        xsc: Dict[int, StandardScaler] = {}
        ysc: Dict[int, StandardScaler] = {}
        for uid in self.users:
            mean_x, var_x = self.stats_in_by_user[uid].mean_var()
            mean_y, var_y = self.stats_out_by_user[uid].mean_var()
            xsc[uid] = scaler_from_mean_var(mean_x, var_x)
            ysc[uid] = scaler_from_mean_var(mean_y, var_y)
        return xsc, ysc

    def fedavg_wout(self, Wout_locals: List[Tuple[int, Dict[int, np.ndarray], Dict[int,int]]]):
        """
        Wout_locals: list per RU of (ru_id, {uid: Wout_local}, {uid: m_samples})
        """
        # accumulate per user
        num_by_uid: Dict[int, np.ndarray] = {}
        den_by_uid: Dict[int, float]      = {}
        for (ru_id, wouts, m_counts) in Wout_locals:
            for uid, Wl in wouts.items():
                m = float(m_counts.get(uid, 0))
                if m <= 0:
                    continue
                if uid not in num_by_uid:
                    num_by_uid[uid] = m * Wl
                    den_by_uid[uid] = m
                else:
                    num_by_uid[uid] += m * Wl
                    den_by_uid[uid] += m
        for uid in self.users:
            if uid in num_by_uid and den_by_uid[uid] > 0:
                self.Wout_by_user[uid] = num_by_uid[uid] / den_by_uid[uid]
            # if no RU trained uid in this round, keep the previous Wout

# ========= Federation partitioning and main loop =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/derived_features_full.csv")
    ap.add_argument("--ru_csv", default="data/ru_positions.csv")
    ap.add_argument("--out", default="model/federated")

    # ESN
    ap.add_argument("--reservoir", type=int, default=600)
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--radius", type=float, default=0.9)
    ap.add_argument("--sparsity", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # RLS
    ap.add_argument("--lambda_rls", type=float, default=0.99)
    ap.add_argument("--alpha_rls", type=float, default=1e-3)

    # Evaluation
    ap.add_argument("--washout_eval", type=int, default=120)
    ap.add_argument("--ctx_k", type=int, default=300)

    # Federation
    ap.add_argument("--num_rus", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=100)

    # Horizon
    ap.add_argument("--steps_ahead", type=int, default=3)

    # A3-like association
    ap.add_argument("--a3_hysteresis_m", type=float, default=5.0)
    ap.add_argument("--a3_ttt", type=int, default=2)

    args = ap.parse_args()
    H = max(1, int(args.steps_ahead))

    # Output directories
    os.makedirs(args.out, exist_ok=True)
    models_dir  = os.path.join(args.out, "models");  os.makedirs(models_dir, exist_ok=True)
    scalers_dir = os.path.join(args.out, "scalers"); os.makedirs(scalers_dir, exist_ok=True)
    plots_dir   = os.path.join(args.out, "plots");   os.makedirs(plots_dir, exist_ok=True)

    # Load CSVs
    df = pd.read_csv(args.csv)
    need = set(["id"] + FEATURES + TARGETS)
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in CSV: {sorted(miss)}")

    ru_tab = pd.read_csv(args.ru_csv)
    if set(["ru_id","x","y"]) - set(ru_tab.columns):
        raise RuntimeError("ru_csv must contain columns: ru_id,x,y")
    ru_tab = ru_tab.sort_values("ru_id")
    if args.num_rus != len(ru_tab):
        print(f"[WARN] --num_rus={args.num_rus} but ru_csv has {len(ru_tab)} rows. Using {len(ru_tab)}.")
    R = len(ru_tab)
    ru_xy = ru_tab[["x","y"]].to_numpy(dtype=np.float64)

    # Users
    users = sorted(df["id"].unique().tolist())
    print(f"[INFO] Users: {users}")

    # ---------- Per user: build U_all, Y_all and per-t RU association ----------
    U_by_user: Dict[int, np.ndarray] = {}
    Y_by_user: Dict[int, np.ndarray] = {}
    assoc_ru_by_user: Dict[int, np.ndarray] = {}  # ru per t (0..T-1)
    n_inputs_by_user: Dict[int, int] = {}

    for uid in users:
        du = df[df["id"] == uid].copy()
        U = du[FEATURES].to_numpy(dtype=np.float64)  # (T, d)
        Y = du[TARGETS].to_numpy(dtype=np.float64)   # (T, 2)
        U_by_user[uid] = U
        Y_by_user[uid] = Y
        n_inputs_by_user[uid] = U.shape[1]

        # A3-like association based on true user (x,y) positions (TARGETS)
        user_xy = Y  # (T,2)
        assoc = a3_like_association(user_xy, ru_xy, d_hyst=args.a3_hysteresis_m, ttt=args.a3_ttt)
        assoc_ru_by_user[uid] = assoc

    # ---------- 80/20 split per user and holdout RU owner ----------
    # training: t in [0 .. ntr_pairs-1]; val: t in [ntr_pairs .. T_pairs-1], T_pairs = T-H
    users_times_train_by_ru: List[Dict[int,np.ndarray]] = [ {} for _ in range(R) ]
    users_times_val_by_ru:   List[Dict[int,np.ndarray]] = [ {} for _ in range(R) ]
    owner_ru_by_user: Dict[int, int] = {}

    for uid in users:
        T_all = U_by_user[uid].shape[0]
        T_pairs = T_all - H
        if T_pairs <= 0:
            print(f"[WARN] user {uid}: T_pairs={T_pairs} (too short). Skipping.")
            continue
        ntr_pairs = int(0.8 * T_pairs)
        # owner RU: ru at the last training timestep (ntr_pairs-1)
        ru_owner = int(assoc_ru_by_user[uid][ntr_pairs - 1])
        owner_ru_by_user[uid] = ru_owner

        # distribute training timesteps by RU
        for ruid in range(R):
            ts = np.where((assoc_ru_by_user[uid][:ntr_pairs]) == ruid)[0]
            # keep only trainable pairs without raw data exchange: same RU at t and t+H
            ts = ts[(ts + H < T_all) & (assoc_ru_by_user[uid][ts + H] == ruid)]
            users_times_train_by_ru[ruid][uid] = ts

        # full holdout assigned only to the owner RU
        ts_val = np.arange(ntr_pairs, T_pairs, dtype=int)
        users_times_val_by_ru[ru_owner][uid] = ts_val

    # ---------- Instantiate CU and RUs ----------
    # Generate global Win/W once (from a reference ESN)
    dummy = ESNStreaming(n_inputs=U_by_user[users[0]].shape[1],
                         n_reservoir=args.reservoir, n_outputs=2,
                         spectral_radius=args.radius, sparsity=args.sparsity,
                         leak_rate=args.leak, include_inputs_in_state=True,
                         seed=args.seed, rls_lambda=args.lambda_rls, rls_alpha=args.alpha_rls)
    Win = dummy.Win.copy(); W = dummy.W.copy()

    cu = CUServer(users=users, H=H,
                  n_inputs_by_user=n_inputs_by_user,
                  n_reservoir=args.reservoir, leak=args.leak,
                  radius=args.radius, sparsity=args.sparsity, seed=args.seed,
                  lambda_rls=args.lambda_rls, alpha_rls=args.alpha_rls,
                  Win=Win, W=W)
    cu.init_user_state()

    rus: List[RUClient] = []
    for ruid in range(R):
        ru = RUClient(
            ru_id=ruid,
            users_times_train=users_times_train_by_ru[ruid],
            users_times_val=users_times_val_by_ru[ruid],
            U_by_user=U_by_user, Y_by_user=Y_by_user, H=H,
            reservoir=args.reservoir, leak=args.leak, radius=args.radius, sparsity=args.sparsity,
            seed=args.seed + 1000 * ruid, lambda_rls=args.lambda_rls, alpha_rls=args.alpha_rls,
            Win=Win, W=W
        )
        rus.append(ru)

    # ---------- Round slices per RU (contiguous temporal segments) ----------
    # slices training times by RU into 'rounds' approximately equal segments (in temporal order)
    rounds = max(1, int(args.rounds))
    round_slices_by_ru: List[List[Dict[int, Tuple[int,int]]]] = []
    for ruid in range(R):
        per_round: List[Dict[int, Tuple[int,int]]] = []
        for rd in range(rounds):
            per_round.append({})
        for uid, ts_all in users_times_train_by_ru[ruid].items():
            L = len(ts_all)
            if L == 0:
                for rd in range(rounds):
                    per_round[rd][uid] = (0, 0)
                continue
            # divide L indices into rounds contiguous slices
            sizes = [L // rounds + (1 if i < (L % rounds) else 0) for i in range(rounds)]
            start = 0
            for rd in range(rounds):
                end = start + sizes[rd]
                per_round[rd][uid] = (start, end)
                start = end
        round_slices_by_ru.append(per_round)

    # ---------- Federated round loop ----------
    for rd in range(rounds):
        print(f"\n===== ROUND {rd+1}/{rounds} =====")

        # ---- Phase A: RUs compute statistics deltas (no training) ----
        delta_in_all: Dict[int, RunningStats]  = {uid: RunningStats.zeros(n_inputs_by_user[uid]) for uid in users}
        delta_out_all: Dict[int, RunningStats] = {uid: RunningStats.zeros(2) for uid in users}
        m_counts_by_ru: List[Dict[int,int]] = []

        for ru in rus:
            slices = round_slices_by_ru[ru.ru_id][rd]  # {uid: (s,e)}
            stats_in, stats_out, m_by_user = ru.collect_stats_for_round(
                global_scalers_in={}, global_scalers_out={}, round_slices=slices
            )
            # aggregate on server
            for uid in users:
                if uid in stats_in:
                    delta_in_all[uid].merge(stats_in[uid])
                if uid in stats_out:
                    delta_out_all[uid].merge(stats_out[uid])
            m_counts_by_ru.append(m_by_user)

        # CU updates global scalers (incremental)
        cu.aggregate_scaler_stats(delta_in_all, delta_out_all)
        xsc_global_by_uid, ysc_global_by_uid = cu.make_global_scalers()

        # ---- Phase B: RUs train locally with round global scalers ----
        local_wouts_pack: List[Tuple[int, Dict[int,np.ndarray], Dict[int,int]]] = []
        for idx, ru in enumerate(rus):
            slices = round_slices_by_ru[ru.ru_id][rd]
            # snapshot of current global Wout per user
            Wout_g = {uid: cu.Wout_by_user[uid] for uid in users}
            # local training
            Wout_loc = ru.train_local_for_round(Wout_g, xsc_global_by_uid, ysc_global_by_uid, slices)
            local_wouts_pack.append((ru.ru_id, Wout_loc, m_counts_by_ru[idx]))

        # CU performs FedAvg per user
        cu.fedavg_wout(local_wouts_pack)

    # ---------- Final evaluation: "owner" RU of each user's holdout ----------
    rows = []
    for uid in users:
        ru_owner = owner_ru_by_user[uid]
        ru = rus[ru_owner]

        U_all = ru.U_by_user[uid]
        Y_all = ru.Y_by_user[uid]
        T_all = U_all.shape[0]
        T_pairs = T_all - H
        if T_pairs <= 0:
            continue
        ntr_pairs = int(0.8 * T_pairs)

        # final model snapshot (global Wout) + current global scalers
        esn_snap = {
            "Win": cu.Win.copy(),
            "W":   cu.W.copy(),
            "Wout": cu.Wout_by_user[uid].copy()
        }
        # freeze final global scalers
        xsc_global_by_uid, ysc_global_by_uid = cu.make_global_scalers()
        xsc = xsc_global_by_uid[uid]; ysc = ysc_global_by_uid[uid]
        xsc_bytes = pickle.dumps(xsc); ysc_bytes = pickle.dumps(ysc)

        rmse_x, rmse_y, rmse_e, Y_eff, yhat_eff, t0_eff = eval_on_validation_with_context(
            esn_snap, xsc_bytes, ysc_bytes, U_all, Y_all, ntr_pairs,
            washout_eval=args.washout_eval, ctx_k=args.ctx_k, horizon=H
        )
        print(f"[RESULT] user {uid} (owner RU={ru_owner}) | H={H} | RMSE_x={rmse_x:.3f} m | RMSE_y={rmse_y:.3f} m | RMSE_e={rmse_e:.3f} m")

        # save user artifacts at the owner RU
        with open(os.path.join(models_dir, f"federated_online_user_{uid}.pkl"), "wb") as f:
            pickle.dump(esn_snap, f)
        with open(os.path.join(scalers_dir, f"federated_online_x_scaler_user_{uid}.pkl"), "wb") as f:
            f.write(xsc_bytes)
        with open(os.path.join(scalers_dir, f"federated_online_y_scaler_user_{uid}.pkl"), "wb") as f:
            f.write(ysc_bytes)

        rows.append({
            "user_id": uid,
            "horizon": H,
            "rmse_x_m": rmse_x,
            "rmse_y_m": rmse_y,
            "rmse_euclid_m": rmse_e,
        })

        # per-user plot (holdout trajectories)
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(Y_eff[:,0], Y_eff[:,1], label="Real")
        ax.plot(yhat_eff[:,0], yhat_eff[:,1], label="Predicted")
        ax.set_title(f"User {uid} - Holdout (20%) - RU owner={ru_owner} - H={H}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend()
        plt.savefig(os.path.join(plots_dir, f"user_{uid}_val_xy_federated_online.png"), bbox_inches="tight", dpi=120)
        plt.close(fig)

    # Federated summary
    summary = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
    summary_path = os.path.join(args.out, "summary_federated_online.csv")
    summary.to_csv(summary_path, index=False)
    print("[DONE] summary_federated_online.csv saved to:", summary_path)

    # Aggregate Real vs Predicted plot (holdout for all users)
    if len(rows) > 0:
        all_real = []; all_pred = []
        for uid in users:
            ru_owner = owner_ru_by_user[uid]
            ru = rus[ru_owner]
            U_all = ru.U_by_user[uid]; Y_all = ru.Y_by_user[uid]
            T_pairs = U_all.shape[0] - H
            if T_pairs <= 0: continue
            ntr_pairs = int(0.8 * T_pairs)

            # reuse evaluation (to collect arrays)
            esn_snap = {"Win": cu.Win.copy(), "W": cu.W.copy(), "Wout": cu.Wout_by_user[uid].copy()}
            xsc_global_by_uid, ysc_global_by_uid = cu.make_global_scalers()
            xsc = xsc_global_by_uid[uid]; ysc = ysc_global_by_uid[uid]
            xsc_bytes = pickle.dumps(xsc); ysc_bytes = pickle.dumps(ysc)
            _, _, _, Y_eff, yhat_eff, _ = eval_on_validation_with_context(
                esn_snap, xsc_bytes, ysc_bytes, U_all, Y_all, ntr_pairs,
                washout_eval=args.washout_eval, ctx_k=args.ctx_k, horizon=H
            )
            all_real.append(Y_eff[:,0]); all_real.append(Y_eff[:,1])
            all_pred.append(yhat_eff[:,0]); all_pred.append(yhat_eff[:,1])

        if len(all_real) > 0:
            real_all = np.concatenate(all_real); pred_all = np.concatenate(all_pred)
            fig = plt.figure()
            ax = plt.gca()
            ax.scatter(real_all, pred_all, s=8, alpha=0.25)
            mn = float(min(real_all.min(), pred_all.min()))
            mx = float(max(real_all.max(), pred_all.max()))
            ax.plot([mn,mx],[mn,mx],'r--',linewidth=2)
            ax.set_title("Real vs Predicted - Holdout (20%) - All users (Federated)")
            ax.set_xlabel("Real [m]"); ax.set_ylabel("Predicted [m]")
            plt.savefig(os.path.join(plots_dir, "all_users_pred_vs_real_federated_online.png"), bbox_inches="tight", dpi=120)
            plt.close(fig)

if __name__ == "__main__":
    main()
