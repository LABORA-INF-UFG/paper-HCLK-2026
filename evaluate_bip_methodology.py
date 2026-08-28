# evaluate_bip_methodology.py
# -*- coding: utf-8 -*-

import os
import math
import re
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp

# ============================================================
# QoE / QoS PARAMETERS
# ============================================================
GAMMA_D_ms = 20.0
GAMMA_D_s  = GAMMA_D_ms / 1000.0
GAMMA_Q    = 0.8

FPS = 30.0
A = 0.05                 # Mbit (payload sub-6 UL)
R_DL_req_Mbps  = 30.0    # requis. downlink mmWave
R_CAM_req_Mbps = 20.0    # requis. camera UL
D = R_DL_req_Mbps / FPS  # Mbit/frame
S = R_CAM_req_Mbps / FPS # Mbit/frame

D_FULL_PER_T_Mbit = R_DL_req_Mbps * 1.0  # 1s horizon used in tile-quality model

N_TILES_SIDE = 8
SIGMA_NORM   = 0.35
ETA_T        = 0.9  # fraction of available bandwidth allocated to transmission

MMW_MAX_USERS_PER_BS = 17

ALL_POLICIES   = ["real", "central_offline", "central_online", "federated"]
PRED_POLICIES  = ["central_offline", "central_online", "federated"]

# ============================================================
# Baseline A3-like parameters (DATA-RATE DOMAIN)
# ============================================================
# Decisions:
# - MUST NOT use any extra window W.
# - Only A3 resources: offset, hysteresis, TTT (short-term state).
# - Decisions use INSTANTANEOUS REAL throughput at time t.
#
# IMPORTANT:
# - Offset/Hysteresis are in Mbps (since the metric is data rate).
# - TTT is in frames (steps). With FPS=30, 3 frames ~ 100 ms.

# Dual-band (independent A3-like on each band)
A3_OFFSET_SUB6_Mbps     = 10.0   # conservative for sub-6
A3_HYSTERESIS_SUB6_Mbps = 5.0
A3_TTT_SUB6_STEPS       = 6      # 6 frames @30fps = 200 ms

A3_OFFSET_MMW_Mbps      = 100.0  # more aggressive in mmWave
A3_HYSTERESIS_MMW_Mbps  = 50.0
A3_TTT_MMW_STEPS        = 3      # 3 frames @30fps = 100 ms

# mmWave-only mode:
# - mmWave: same as above (can be different if desired)
# - sub-6: anchor more stable (larger TTT and/or offset/hys)
A3_OFFSET_SUB6_ANCHOR_Mbps     = 15.0
A3_HYSTERESIS_SUB6_ANCHOR_Mbps = 8.0
A3_TTT_SUB6_ANCHOR_STEPS       = 9   # 300 ms

# ============================================================
# Eq.(8) / evaluation params
# ============================================================
GA_VALUE = 11.0
T_WINDOW = 3  # used ONLY in evaluation and ILP objective; NOT used by A3-like decisions

FULL     = "_full_"
MMW_ONLY = "_mmwave_only_"

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "opt")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Helpers
# ============================================================
def stable_seed_from(*args) -> int:
    h = hashlib.sha256()
    for a in args:
        h.update(str(a).encode())
    return int.from_bytes(h.digest()[:8], "little") & 0x7FFFFFFF


ARM_RE = re.compile(r"^(?P<arm>real|central_offline|central_online|federated)_tx(?P<tx>\d+)_(?P<band>sub6|mmWave)$")


def discover_topology_from_columns(columns):
    arms, bands, tx_ids = set(), set(), set()
    for c in columns:
        m = ARM_RE.match(c)
        if not m:
            continue
        arms.add(m.group("arm"))
        bands.add(m.group("band"))
        tx_ids.add(int(m.group("tx")))
    return sorted(arms), sorted(bands), sorted(tx_ids)


def gaussian_center_weights(n_side, sigma_norm):
    xs = np.linspace(-1, 1, n_side)
    ys = np.linspace(-1, 1, n_side)
    X, Y = np.meshgrid(xs, ys)
    R2   = X**2 + Y**2
    W    = np.exp(-R2 / (2 * (sigma_norm**2)))
    W   /= W.sum()
    W   /= W.max()
    return W


W_TILES = gaussian_center_weights(N_TILES_SIDE, SIGMA_NORM)


def quality_from_tiles(c_dl_mbps: float, eta_t: float, d_full_mbit: float) -> float:
    if (c_dl_mbps is None) or (c_dl_mbps <= 0) or (eta_t <= 0):
        return 0.0
    n_tiles = W_TILES.size
    budget  = eta_t * float(c_dl_mbps) * 1.0  # Mbit in 1 second
    d_tile  = d_full_mbit / n_tiles

    flat = W_TILES.flatten()
    order = np.argsort(-flat)

    q = 0.0
    remaining = budget
    for idx in order:
        if remaining >= d_tile:
            q += float(flat[idx])
            remaining -= d_tile
        else:
            break
    return float(min(1.0, q))


def safe_rate(row, col):
    v = row.get(col, None)
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def compute_bip_from_rates(c_sub6_ul, c_mmw_ul, c_mmw_dl, eta_t=ETA_T):
    term_sub6 = math.inf if (c_sub6_ul is None or c_sub6_ul <= 0) else (A / c_sub6_ul)
    term_mmwU = math.inf if (c_mmw_ul  is None or c_mmw_ul  <= 0) else (S / c_mmw_ul)
    term_mmwD = math.inf if (c_mmw_dl  is None or c_mmw_dl  <= 0) else (D / c_mmw_dl)
    delay_s   = term_sub6 + term_mmwU + term_mmwD
    q = quality_from_tiles(c_mmw_dl, eta_t, D_FULL_PER_T_Mbit)
    delay_exceeded  = (delay_s > GAMMA_D_s)
    quality_not_met = (q < GAMMA_Q)
    omega = int(delay_exceeded or quality_not_met)
    return {
        "omega": omega,
        "delay_s": delay_s,
        "q": q,
        "cause_delay_only": int(delay_exceeded and not quality_not_met),
        "cause_quality_only": int(quality_not_met and not delay_exceeded),
    }


@dataclass
class NoiseSampler:
    sigma_i: float = 0.193
    sigma_ga: float = 0.151
    sigma_B: float = 0.05
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.eps_i = {}
        self.eps_ga = {}

    def prepare_users(self, users: Iterable[int]):
        for u in users:
            self.eps_i[u]  = float(self.rng.normal(loc=0.0, scale=self.sigma_i))
            self.eps_ga[u] = float(self.rng.normal(loc=0.0, scale=self.sigma_ga))

    def eps_B_for_t(self, t: int, users: Iterable[int]):
        rng_t = np.random.default_rng(stable_seed_from("epsB", self.seed, t))
        return {
            (i, t): float(rng_t.normal(loc=0.0, scale=self.sigma_B))
            for i in users
        }


def c_eq8(GA, omega, eps_i, eps_ga, eps_B):
    return GA + omega + GA * omega + eps_i + eps_ga + eps_B


# ============================================================
# Global data caches (populated inside run_pipeline)
# ============================================================
df_idx: Dict[Tuple[int, int], Any] = {}
tx_sub6: List[int] = []
tx_mmw:  List[int] = []
TIMES_LIST: List[int] = []
MAX_T: int = 0

NOISE = NoiseSampler()
EPS_I: Dict[int, float] = {}
EPS_GA: Dict[int, float] = {}
EPSB: Dict[Tuple[int, int], float] = {}


def pi_eq8_instant(i: int, t: int, s: int, m: int,
                   policy: str, use_real_rates: bool):
    """
    Instantaneous Eq.(8) at time t:
        GA + ω + GA*ω + ε_i + ε_{GA|i} + ε_B(i,t)
    where ω is obtained from Eq.(7) using:
        sub-6 UL:  <policy>_tx{s}_sub6
        mmWave UL: <policy>_tx{m}_mmWave
        mmWave DL: <policy>_tx{m}_mmWave  (same trace in this dataset)
    """
    row = df_idx.get((i, t))
    if row is None:
        return None, None

    prefix = "real" if use_real_rates else policy

    ul_s = safe_rate(row, f"{prefix}_tx{s}_sub6")
    ul_m = safe_rate(row, f"{prefix}_tx{m}_mmWave")
    dl_m = safe_rate(row, f"{prefix}_tx{m}_mmWave")

    r = compute_bip_from_rates(ul_s, ul_m, dl_m, ETA_T)
    omega   = r["omega"]
    delay_s = r["delay_s"]
    q       = r["q"]

    e_i  = EPS_I.get(i, 0.0)
    e_ga = EPS_GA.get(i, 0.0)

    if (i, t) not in EPSB:
        EPSB.update(NOISE.eps_B_for_t(t, [i]))
    eB = EPSB.get((i, t), 0.0)

    cost_inst = c_eq8(GA_VALUE, omega, e_i, e_ga, eB)
    return cost_inst, {"omega": omega, "delay_s": delay_s, "q": q}


def pi_eq8_mean_window(i: int, t0: int, s: int, m: int,
                       policy: str, use_real_rates: bool):
    """
    Windowed mean of Eq.(8) over T_WINDOW steps.
    NOTE: used for ILP objective and evaluation only. NOT used by A3-like decisions.
    """
    t_start = t0
    t_end   = min(t0 + T_WINDOW - 1, MAX_T)

    total = 0.0
    count = 0

    for tau in range(t_start, t_end + 1):
        cost_inst, _ = pi_eq8_instant(i, tau, s, m, policy, use_real_rates)
        if cost_inst is None:
            continue
        total += cost_inst
        count += 1

    if count == 0:
        return GA_VALUE * 2.0

    return total / count


# ============================================================
# ILP (unchanged)
# ============================================================
def flow_assign_for_t_full(users, t, policy_for_decision,
                           tx_mmw_local, mmw_capacity):
    EPSB.update(NOISE.eps_B_for_t(t, users))

    S_list = tx_sub6
    M_list = tx_mmw_local

    n_users = len(users)
    n_s = len(S_list)
    n_m = len(M_list)

    cost: Dict[Tuple[int, int, int], float] = {}

    for ui, i in enumerate(users):
        row = df_idx.get((i, t))
        if row is None:
            for si in range(n_s):
                for mj in range(n_m):
                    cost[(ui, si, mj)] = GA_VALUE * 2.0
        else:
            for mj, m in enumerate(M_list):
                for si, s in enumerate(S_list):
                    base = pi_eq8_mean_window(
                        i, t, s, m,
                        policy_for_decision,
                        use_real_rates=False
                    )
                    cost[(ui, si, mj)] = base

    prob = pulp.LpProblem("JointSub6MmWaveAssignment", pulp.LpMinimize)

    x: Dict[Tuple[int, int, int], Any] = {}
    for ui in range(n_users):
        for si in range(n_s):
            for mj in range(n_m):
                x[(ui, si, mj)] = pulp.LpVariable(
                    f"x_{ui}_{si}_{mj}", lowBound=0, upBound=1, cat="Binary"
                )

    prob += pulp.lpSum(
        cost[(ui, si, mj)] * x[(ui, si, mj)]
        for ui in range(n_users)
        for si in range(n_s)
        for mj in range(n_m)
    )

    for ui in range(n_users):
        prob += pulp.lpSum(
            x[(ui, si, mj)] for si in range(n_s) for mj in range(n_m)
        ) == 1

    for mj, _m in enumerate(M_list):
        prob += pulp.lpSum(
            x[(ui, si, mj)] for ui in range(n_users) for si in range(n_s)
        ) <= mmw_capacity

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    final_s: Dict[int, int] = {}
    final_m: Dict[int, int] = {}

    for ui, i in enumerate(users):
        chosen = False
        for si in range(n_s):
            for mj in range(n_m):
                val = pulp.value(x[(ui, si, mj)])
                if val is not None and val > 0.5:
                    final_s[i] = S_list[si]
                    final_m[i] = M_list[mj]
                    chosen = True
                    break
            if chosen:
                break

        if not chosen:
            best_val = None
            best_s = S_list[0] if S_list else None
            best_m = M_list[0] if M_list else None
            for si in range(n_s):
                for mj in range(n_m):
                    v = cost.get((ui, si, mj), GA_VALUE * 2.0)
                    if best_val is None or v < best_val:
                        best_val = v
                        best_s = S_list[si]
                        best_m = M_list[mj]
            if best_s is not None and best_m is not None:
                final_s[i] = best_s
                final_m[i] = best_m

    return final_s, final_m


def flow_assign_for_t_mmwave_only(users, t, policy_for_decision,
                                 tx_mmw_local, mmw_capacity):
    EPSB.update(NOISE.eps_B_for_t(t, users))

    M_list = tx_mmw_local
    n_users = len(users)
    n_m = len(M_list)

    cost = {}
    for ui, i in enumerate(users):
        row = df_idx.get((i, t))
        for mj, m in enumerate(M_list):
            if row is None:
                base = GA_VALUE * 2.0
            else:
                s = m
                base = pi_eq8_mean_window(
                    i, t, s, m,
                    policy_for_decision,
                    use_real_rates=False
                )
            cost[(ui, mj)] = base

    prob = pulp.LpProblem("MmWaveOnlyAssignment", pulp.LpMinimize)

    x = {}
    for ui in range(n_users):
        for mj in range(n_m):
            x[(ui, mj)] = pulp.LpVariable(
                f"x_{ui}_{mj}", lowBound=0, upBound=1, cat="Binary"
            )

    prob += pulp.lpSum(
        cost[(ui, mj)] * x[(ui, mj)]
        for ui in range(n_users)
        for mj in range(n_m)
    )

    for ui in range(n_users):
        prob += pulp.lpSum(
            x[(ui, mj)] for mj in range(n_m)
        ) == 1

    for mj, _m in enumerate(M_list):
        prob += pulp.lpSum(
            x[(ui, mj)] for ui in range(n_users)
        ) <= mmw_capacity

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    final_m = {}
    final_s = {}
    for ui, i in enumerate(users):
        chosen = False
        for mj in range(n_m):
            val = pulp.value(x[(ui, mj)])
            if val is not None and val > 0.5:
                m = M_list[mj]
                final_m[i] = m
                final_s[i] = m
                chosen = True
                break
        if not chosen:
            best_mj = min(range(n_m), key=lambda mj: cost[(ui, mj)])
            m = M_list[best_mj]
            final_m[i] = m
            final_s[i] = m

    return final_s, final_m


# ============================================================
# A3-like helpers (NO WINDOW, REAL INSTANT, CAPACITY-AWARE)
# ============================================================
def _a3_enter_condition_rate(r_best: float, r_curr: float, offset: float, hyst: float) -> bool:
    # A3-style entering: r_best - hyst > r_curr + offset
    return (r_best - hyst) > (r_curr + offset)


def _rank_candidates_by_rate_instant(
    row: Any,
    tx_candidates: List[int],
    prefix: str,
    band: str
) -> List[Tuple[int, float]]:
    """
    band in {"sub6","mmWave"}.
    Returns list of (tx_id, rate_mbps) sorted descending.
    """
    ranked: List[Tuple[int, float]] = []
    for tx in tx_candidates:
        r = safe_rate(row, f"{prefix}_tx{tx}_{band}")
        ranked.append((tx, float(r)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _capacity_aware_mmwave_assignment_rounds(
    users: List[int],
    stayers_m: Dict[int, int],
    switchers_pref: Dict[int, List[int]],
    cap: int
) -> Dict[int, int]:
    """
    Capacity-aware assignment for mmWave at time t (reactive admission control).
    - stayers_m: users who stay => fixed occupancy
    - switchers_pref: preference list of mmWave candidates (best first)
    Rule:
      If chosen mmWave is full, search next preference until vacancy exists.
    Implementation:
      Round-based proposals, deterministic tie-break by user_id.
    """
    assigned: Dict[int, int] = dict(stayers_m)

    occ: Dict[int, int] = {}
    for _, m in stayers_m.items():
        occ[m] = occ.get(m, 0) + 1

    unassigned = sorted(list(switchers_pref.keys()))
    if not unassigned:
        return assigned

    max_len = 0
    for u in unassigned:
        max_len = max(max_len, len(switchers_pref.get(u, [])))

    for r in range(max_len):
        proposals: Dict[int, List[int]] = {}
        still_unassigned: List[int] = []
        for u in unassigned:
            pref = switchers_pref.get(u, [])
            if r >= len(pref):
                still_unassigned.append(u)
                continue
            m_try = pref[r]
            proposals.setdefault(m_try, []).append(u)

        for m_try, us in proposals.items():
            current = occ.get(m_try, 0)
            remaining = cap - current
            if remaining <= 0:
                still_unassigned.extend(us)
                continue

            us_sorted = sorted(us)
            accepted = us_sorted[:remaining]
            rejected = us_sorted[remaining:]

            for u in accepted:
                assigned[u] = m_try
            occ[m_try] = current + len(accepted)

            still_unassigned.extend(rejected)

        unassigned = sorted(set(still_unassigned))
        if not unassigned:
            break

    return assigned


# ============================================================
# A3-like Baseline (DUAL-BAND):
# - sub-6 updated by A3-like on SUB-6 data rate
# - mmWave updated by A3-like on mmWave data rate + capacity constraint
# - NO window W in decisions
# - REAL instantaneous data rate only
# ============================================================
def baseline_a3_like_dual_band(
    users: List[int],
    times: List[int],
    tx_sub6_local: List[int],
    tx_mmw_local: List[int],
    mmw_capacity: int = MMW_MAX_USERS_PER_BS,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    if not tx_sub6_local or not tx_mmw_local:
        return {}

    serving_s = {i: tx_sub6_local[0] for i in users}
    serving_m = {i: tx_mmw_local[0] for i in users}

    ttt_s = {i: 0 for i in users}
    ttt_m = {i: 0 for i in users}

    decisions: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for t in times:
        # ---------- (1) SUB-6 A3-like ----------
        for i in users:
            row = df_idx.get((i, t))
            if row is None:
                ttt_s[i] = 0
                continue

            prefix = "real"
            s_curr = serving_s[i]
            r_curr = safe_rate(row, f"{prefix}_tx{s_curr}_sub6")

            ranked_s = _rank_candidates_by_rate_instant(row, tx_sub6_local, prefix, "sub6")
            if not ranked_s:
                ttt_s[i] = 0
                continue

            s_best, r_best = ranked_s[0]
            if s_best != s_curr and _a3_enter_condition_rate(r_best, r_curr, A3_OFFSET_SUB6_Mbps, A3_HYSTERESIS_SUB6_Mbps):
                ttt_s[i] += 1
                if ttt_s[i] >= A3_TTT_SUB6_STEPS:
                    serving_s[i] = s_best
                    ttt_s[i] = 0
            else:
                ttt_s[i] = 0

        # ---------- (2) mmWave A3-like + capacity ----------
        stayers_m: Dict[int, int] = {}
        switchers_pref: Dict[int, List[int]] = {}
        fallback_m: Dict[int, int] = {}

        for i in users:
            row = df_idx.get((i, t))
            m_curr = serving_m[i]
            s_curr = serving_s[i]  # note: used only in evaluation, not in mmWave decision itself

            if row is None:
                ttt_m[i] = 0
                stayers_m[i] = m_curr
                continue

            prefix = "real"
            r_curr = safe_rate(row, f"{prefix}_tx{m_curr}_mmWave")

            ranked_m = _rank_candidates_by_rate_instant(row, tx_mmw_local, prefix, "mmWave")
            if not ranked_m:
                ttt_m[i] = 0
                stayers_m[i] = m_curr
                continue

            m_best, r_best = ranked_m[0]

            if m_best != m_curr and _a3_enter_condition_rate(r_best, r_curr, A3_OFFSET_MMW_Mbps, A3_HYSTERESIS_MMW_Mbps):
                ttt_m[i] += 1
                if ttt_m[i] >= A3_TTT_MMW_STEPS:
                    # eligible to switch; capacity-aware admission will decide final RU
                    pref_list = [m for (m, _r) in ranked_m]
                    switchers_pref[i] = pref_list
                    fallback_m[i] = m_curr
                    ttt_m[i] = 0
                else:
                    stayers_m[i] = m_curr
            else:
                ttt_m[i] = 0
                stayers_m[i] = m_curr

        assigned_m = _capacity_aware_mmwave_assignment_rounds(
            users=users,
            stayers_m=stayers_m,
            switchers_pref=switchers_pref,
            cap=mmw_capacity
        )

        for i in users:
            if i in assigned_m:
                serving_m[i] = assigned_m[i]
            else:
                serving_m[i] = fallback_m.get(i, serving_m[i])

            decisions[(i, t)] = (serving_s[i], serving_m[i])

    return decisions


# ============================================================
# A3-like Baseline (mmWave-only):
# - mmWave updated by A3-like on mmWave data rate + capacity constraint
# - sub-6 uses an ANCHOR association rule (A3-like conservative on sub-6)
# ============================================================
def baseline_a3_like_mmwave_only_with_sub6_anchor(
    users: List[int],
    times: List[int],
    tx_sub6_local: List[int],
    tx_mmw_local: List[int],
    mmw_capacity: int = MMW_MAX_USERS_PER_BS,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    if not tx_mmw_local or not tx_sub6_local:
        return {}

    # sub-6 anchor serving
    serving_s = {i: tx_sub6_local[0] for i in users}
    ttt_s = {i: 0 for i in users}

    # mmWave serving
    serving_m = {i: tx_mmw_local[0] for i in users}
    ttt_m = {i: 0 for i in users}

    decisions: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for t in times:
        # ---------- (0) sub-6 anchor update (conservative A3-like) ----------
        for i in users:
            row = df_idx.get((i, t))
            if row is None:
                ttt_s[i] = 0
                continue

            prefix = "real"
            s_curr = serving_s[i]
            r_curr = safe_rate(row, f"{prefix}_tx{s_curr}_sub6")

            ranked_s = _rank_candidates_by_rate_instant(row, tx_sub6_local, prefix, "sub6")
            if not ranked_s:
                ttt_s[i] = 0
                continue

            s_best, r_best = ranked_s[0]
            if s_best != s_curr and _a3_enter_condition_rate(r_best, r_curr, A3_OFFSET_SUB6_ANCHOR_Mbps, A3_HYSTERESIS_SUB6_ANCHOR_Mbps):
                ttt_s[i] += 1
                if ttt_s[i] >= A3_TTT_SUB6_ANCHOR_STEPS:
                    serving_s[i] = s_best
                    ttt_s[i] = 0
            else:
                ttt_s[i] = 0

        # ---------- (1) mmWave-only A3-like + capacity ----------
        stayers_m: Dict[int, int] = {}
        switchers_pref: Dict[int, List[int]] = {}
        fallback_m: Dict[int, int] = {}

        for i in users:
            row = df_idx.get((i, t))
            m_curr = serving_m[i]

            if row is None:
                ttt_m[i] = 0
                stayers_m[i] = m_curr
                continue

            prefix = "real"
            r_curr = safe_rate(row, f"{prefix}_tx{m_curr}_mmWave")

            ranked_m = _rank_candidates_by_rate_instant(row, tx_mmw_local, prefix, "mmWave")
            if not ranked_m:
                ttt_m[i] = 0
                stayers_m[i] = m_curr
                continue

            m_best, r_best = ranked_m[0]
            if m_best != m_curr and _a3_enter_condition_rate(r_best, r_curr, A3_OFFSET_MMW_Mbps, A3_HYSTERESIS_MMW_Mbps):
                ttt_m[i] += 1
                if ttt_m[i] >= A3_TTT_MMW_STEPS:
                    pref_list = [m for (m, _r) in ranked_m]
                    switchers_pref[i] = pref_list
                    fallback_m[i] = m_curr
                    ttt_m[i] = 0
                else:
                    stayers_m[i] = m_curr
            else:
                ttt_m[i] = 0
                stayers_m[i] = m_curr

        assigned_m = _capacity_aware_mmwave_assignment_rounds(
            users=users,
            stayers_m=stayers_m,
            switchers_pref=switchers_pref,
            cap=mmw_capacity
        )

        for i in users:
            if i in assigned_m:
                serving_m[i] = assigned_m[i]
            else:
                serving_m[i] = fallback_m.get(i, serving_m[i])

            # mmWave-only mode: m chosen by A3-like; sub-6 anchor chosen separately
            decisions[(i, t)] = (serving_s[i], serving_m[i])

    return decisions


# ============================================================
# Evaluation (Eq.(8) cost over window) - uses REAL
# ============================================================
def evaluate_eq8_block(df, arm_label, policy_for_decision,
                       t0, T, users, decisions) -> pd.DataFrame:
    print(f"[EQ8] Evaluating arm={arm_label} policy={policy_for_decision} "
          f"t∈[{t0},{t0+T-1}] users={len(users)} (T_WINDOW={T_WINDOW})", flush=True)

    start = time.time()
    rows = []

    for u_idx, i in enumerate(users, 1):
        for t in range(t0, t0 + T):
            if (i, t) not in decisions:
                continue

            s, m = decisions[(i, t)]
            Pi_t = pi_eq8_mean_window(i, t, s, m, policy="real", use_real_rates=True)

            _, r_inst = pi_eq8_instant(i, t, s, m, policy="real", use_real_rates=True)
            if r_inst is None:
                continue

            rows.append({
                "arm": arm_label,
                "policy": policy_for_decision,
                "t": t,
                "id_user": i,
                "j_sub6": s,
                "j_mmw": m,
                "omega": r_inst["omega"],
                "delay_s": r_inst["delay_s"],
                "q": r_inst["q"],
                "c_eq8": Pi_t
            })

        if (u_idx % 10 == 0) or (u_idx == len(users)):
            elapsed = time.time() - start
            print(f"   -> users {u_idx}/{len(users)} processed ({elapsed:.1f}s)", flush=True)

    return pd.DataFrame(rows)


# ============================================================
# Main pipeline
# ============================================================
def run_pipeline(BSs: str, approach: str, user_count: int = None):
    # (1) Load CSV
    if user_count is None:
        inp = os.path.join(DATA_DIR, "throughput_stations_" + BSs + ".csv")
        out_suffix = BSs
    else:
        inp = os.path.join(DATA_DIR, f"throughput_users_{user_count}.csv")
        out_suffix = f"users_{user_count}"
        BSs = out_suffix

    df  = pd.read_csv(inp)

    if "id_user" not in df.columns:
        if "user_id" in df.columns:
            df = df.rename(columns={"user_id": "id_user"})
        else:
            raise ValueError("CSV must contain 'id_user' (or 'user_id').")

    if "t" not in df.columns:
        df["t"] = df.groupby("id_user").cumcount()

    arms, bands, tx_ids = discover_topology_from_columns(df.columns)
    if not set(["sub6", "mmWave"]).issubset(set(bands)):
        raise ValueError("CSV must have *_sub6 and *_mmWave columns for each tx.")

    global tx_sub6, tx_mmw, df_idx, NOISE, EPS_I, EPS_GA, EPSB, TIMES_LIST, MAX_T

    tx_sub6 = tx_ids
    tx_mmw  = tx_ids

    users   = sorted(df["id_user"].unique().tolist())
    times   = sorted(df["t"].unique().tolist())
    TIMES_LIST = times
    MAX_T = int(max(times))

    df_idx = {(int(r.id_user), int(r.t)): r for _, r in df.iterrows()}

    # Noise (fixed seed for reproducibility)
    NOISE = NoiseSampler(seed=stable_seed_from("eq8-noise", "v6"))
    NOISE.prepare_users(users)
    EPS_I  = dict(NOISE.eps_i)
    EPS_GA = dict(NOISE.eps_ga)
    EPSB   = {}

    results_rows = []
    dec_all      = {}

    # ============================================================
    # (A) BASELINE A3-like (DATA-RATE based)
    # - No window in decisions
    # - Real instantaneous throughput only
    # - mmWave capacity enforced by searching other available RU
    # ============================================================
    print("[BASELINE] Running A3-like (DATA-RATE, REAL, capacity-aware, no decision window) ...", flush=True)
    if approach == FULL:
        dec_a3 = baseline_a3_like_dual_band(users, times, tx_sub6, tx_mmw, mmw_capacity=MMW_MAX_USERS_PER_BS)
    else:
        dec_a3 = baseline_a3_like_mmwave_only_with_sub6_anchor(users, times, tx_sub6, tx_mmw, mmw_capacity=MMW_MAX_USERS_PER_BS)
    dec_all[("baseline", "a3_like")] = dec_a3

    # ============================================================
    # (B) PROPOSED – ILP with predictions (unchanged)
    # ============================================================
    for pp in PRED_POLICIES:
        print(f"[FLOW] Running ILP with prediction policy '{pp}'...", flush=True)
        dec_pred = {}
        for t in times:
            EPSB.update(NOISE.eps_B_for_t(t, users))
            if approach == FULL:
                chosen_s, chosen_m = flow_assign_for_t_full(users, t, pp, tx_mmw, MMW_MAX_USERS_PER_BS)
            else:
                chosen_s, chosen_m = flow_assign_for_t_mmwave_only(users, t, pp, tx_mmw, MMW_MAX_USERS_PER_BS)
            for i in users:
                dec_pred[(i, t)] = (chosen_s[i], chosen_m[i])
        dec_all[("proposed", pp)] = dec_pred

    # ============================================================
    # (C) IDEAL (upper bound) – ILP with REAL (unchanged)
    # ============================================================
    dec_oracle = {}
    print("[REAL] Running ILP with REAL (oracle) ...", flush=True)
    for t in times:
        EPSB.update(NOISE.eps_B_for_t(t, users))
        if approach == FULL:
            chosen_s, chosen_m = flow_assign_for_t_full(users, t, "real", tx_mmw, MMW_MAX_USERS_PER_BS)
        else:
            chosen_s, chosen_m = flow_assign_for_t_mmwave_only(users, t, "real", tx_mmw, MMW_MAX_USERS_PER_BS)
        for i in users:
            dec_oracle[(i, t)] = (chosen_s[i], chosen_m[i])
    dec_all[("upper_bound", "perfect_knowledge")] = dec_oracle

    # ============================================================
    # (D) Evaluation of all policies under REAL Eq.(8)
    # ============================================================
    for (arm_label, policy), decisions in dec_all.items():
        block = evaluate_eq8_block(df, arm_label, policy,
                                   t0=int(min(times)), T=len(times),
                                   users=users, decisions=decisions)
        results_rows.append(block)

    df_eq8 = pd.concat(results_rows, ignore_index=True)
    df_eq8.to_csv(os.path.join(OUT_DIR, "eq8_pi_by_user_ILP" + approach + BSs + ".csv"), index=False)

    summary = (df_eq8.groupby(["arm", "policy"])["c_eq8"]
               .sum().reset_index().rename(columns={"c_eq8": "sum_P"}))
    summary.to_csv(os.path.join(OUT_DIR, "summary_by_arm_ILP" + approach + BSs + ".csv"), index=False)

    # ============================================================
    # Paper-friendly plots (basic set)
    # ============================================================
    labels = []
    vals_flow = []
    for pp in PRED_POLICIES:
        fl = summary[(summary["arm"] == "proposed") & (summary["policy"] == pp)]
        if not fl.empty:
            labels.append(pp)
            vals_flow.append(float(fl["sum_P"].iloc[0]))

    base = summary[(summary["arm"] == "baseline") & (summary["policy"] == "a3_like")]
    base_v = float(base["sum_P"].iloc[0]) if not base.empty else np.nan

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 4.8))
    plt.bar(x, vals_flow, label="Optimized (ILP + ML predictions)")
    plt.axhline(base_v, linestyle="--", label="A3-like (reactive baseline)")
    plt.xticks(x, labels)
    plt.ylabel("Σ P_i (lower is better)")
    plt.title("A3-like baseline vs ILP+ML policies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_pred_vs_baseline_ILP" + approach + BSs + ".png"), dpi=160)
    plt.close()

    mean_flow = float(np.mean(vals_flow)) if vals_flow else np.nan
    upper = summary[(summary["arm"] == "upper_bound") & (summary["policy"] == "perfect_knowledge")]
    upv   = float(upper["sum_P"].iloc[0]) if not upper.empty else np.nan

    plt.figure(figsize=(10, 4.8))
    labs = ["A3-like", "mean(ILP+ML)", "IDEAL"]
    vals = [base_v, mean_flow, upv]
    plt.bar(np.arange(len(labs)), vals)
    plt.xticks(np.arange(len(labs)), labs, rotation=15, ha="right")
    plt.ylabel("Σ P_i (lower is better)")
    plt.title("Baseline vs optimized vs upper bound")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_summary_vs_upper_ILP" + approach + BSs + ".png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    box_data = []
    labels_bp = []
    for (arm_label, policy) in df_eq8.groupby(["arm", "policy"]).groups.keys():
        sub = df_eq8[(df_eq8["arm"] == arm_label) & (df_eq8["policy"] == policy)]
        box_data.append(sub["c_eq8"].values)
        labels_bp.append(f"{arm_label}|{policy}")
    plt.boxplot(box_data, labels=labels_bp, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("P_i(t) (windowed mean)")
    plt.title("Distribution of P_i(t) by policy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "boxplot_policies_ILP" + approach + BSs + ".png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    for (arm_label, policy), g in df_eq8.groupby(["arm", "policy"]):
        mean_t = g.groupby("t")["c_eq8"].mean()
        plt.plot(mean_t.index, mean_t.values, label=f"{arm_label}|{policy}")
    plt.xlabel("t")
    plt.ylabel("Mean P_i(t)")
    plt.title("Temporal evolution of mean P_i(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "timeseries_mean_P_ILP" + approach + BSs + ".png"), dpi=160)
    plt.close()

    base_row = summary[(summary["arm"] == "baseline") & (summary["policy"] == "a3_like")]
    base_p = float(base_row["sum_P"].iloc[0]) if not base_row.empty else None

    gain_rows = []
    if base_p is not None and base_p > 0:
        for pp in PRED_POLICIES:
            flow_series = summary[(summary["arm"] == "proposed") & (summary["policy"] == pp)]["sum_P"]
            if flow_series.empty:
                continue
            flow_p = float(flow_series.iloc[0])
            gain = 100.0 * (base_p - flow_p) / base_p
            gain_rows.append([pp, gain])

    df_gain = pd.DataFrame(gain_rows, columns=["policy", "%gain_vs_A3_like"])
    df_gain.to_csv(os.path.join(OUT_DIR, "percentual_gain_vs_baseline_ILP" + approach + BSs + ".csv"), index=False)

    if not df_gain.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(df_gain["policy"], df_gain["%gain_vs_A3_like"])
        plt.ylabel("% gain (optimized vs A3-like)")
        plt.title("Percent gain over A3-like baseline")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "gain_vs_baseline_ILP" + approach + BSs + ".png"), dpi=160)
        plt.close()

    print("[OK] A3-like updated: DATA-RATE, reactive, REAL instantaneous decisions, no decision window, mmWave capacity-aware.")
    print("[OK] Outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    BASE_STATIONS_GP = ["1_4_5", "1_3_4_5_8", "1_3_4_5_6_7_8", "0_1_2_3_4_5_6_7_8"]
    APPROACHES = [FULL, MMW_ONLY]

    for BSs in BASE_STATIONS_GP:
        for approach in APPROACHES:
            run_pipeline(BSs, approach)

    USER_COUNTS = [10, 20, 30, 40, 50]
    for n_users in USER_COUNTS:
        for approach in APPROACHES:
            run_pipeline("", approach, user_count=n_users)