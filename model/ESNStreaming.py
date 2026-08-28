
# esn_streaming.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

@dataclass
class ESNStreaming:
    """
    ESN with persistent state (streaming) and RLS readout.
    - Reservoir state x_t is maintained between calls (1-sample latency).
    - Readout (Wout) updated sample-by-sample (RLS with forgetting factor).
    """
    n_inputs: int
    n_reservoir: int
    n_outputs: int = 2

    spectral_radius: float = 0.9
    sparsity: float = 0.1
    leak_rate: float = 0.3
    input_scale: float = 0.5
    bias_scale: float = 0.2
    include_inputs_in_state: bool = True
    seed: int = 42

    # RLS
    rls_lambda: float = 0.99     # forgetting factor
    rls_alpha: float = 1e-3      # P0 = (1/alpha) I

    Win: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None
    Wout: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)

        # Win: [bias; inputs]
        self.Win = (rng.random((self.n_reservoir, 1 + self.n_inputs)) - 0.5) * 2.0
        self.Win[:, 0] *= self.bias_scale
        self.Win[:, 1:] *= self.input_scale

        # Sparse reservoir matrix
        W = rng.random((self.n_reservoir, self.n_reservoir)) - 0.5
        mask = rng.random((self.n_reservoir, self.n_reservoir)) < self.sparsity
        W *= mask

        # Scale spectral radius
        v = rng.random(self.n_reservoir)
        for _ in range(30):
            v = W @ v
            n = np.linalg.norm(v) + 1e-12
            v /= n
        radius = float(np.linalg.norm(W @ v))
        if radius > 0:
            W *= (self.spectral_radius / radius)
        self.W = W

        self.x = np.zeros((self.n_reservoir,), dtype=np.float64)

        D = 1 + self.n_reservoir + (self.n_inputs if self.include_inputs_in_state else 0)
        self.Wout = np.zeros((D, self.n_outputs), dtype=np.float64)
        self.P = (1.0 / self.rls_alpha) * np.eye(D, dtype=np.float64)

    # ----- dynamic core -----
    def _update_state(self, u: np.ndarray) -> None:
        pre = self.Win @ np.concatenate(([1.0], u), axis=0) + self.W @ self.x
        x_new = np.tanh(pre)
        self.x = (1.0 - self.leak_rate) * self.x + self.leak_rate * x_new

    def _phi(self, u: np.ndarray) -> np.ndarray:
        if self.include_inputs_in_state:
            return np.concatenate(([1.0], self.x, u), axis=0)
        else:
            return np.concatenate(([1.0], self.x), axis=0)

    # ----- inference (scaled space) -----
    def predict_scaled_and_features(self, u_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._update_state(u_scaled)
        z = self._phi(u_scaled)     # (D,)
        yhat_s = z @ self.Wout      # (n_outputs,)
        return yhat_s, z

    # ----- RLS update -----
    def rls_update(self, z: np.ndarray, y_target_scaled: np.ndarray) -> Dict[str, Any]:
        lam = self.rls_lambda
        z = z.reshape(-1, 1)                        # (D,1)
        denom = lam + (z.T @ self.P @ z)[0, 0]
        if denom < 1e-9:  # numerical safeguard
            denom = 1e-9
        K = (self.P @ z) / denom                    # (D,1)
        err = y_target_scaled - (self.Wout.T @ z).ravel()
        self.Wout += K @ err.reshape(1, -1)         # (D,2) += (D,1) @ (1,2)
        self.P = (self.P - K @ z.T @ self.P) / lam  # P update
        # symmetrize P slightly (numerical drift)
        self.P = 0.5 * (self.P + self.P.T)

        return {
            "denom": float(denom),
            "gain_norm": float(np.linalg.norm(K)),
            "err_norm": float(np.linalg.norm(err)),
        }

    # ----- snapshot -----
    def snapshot(self) -> Dict[str, np.ndarray]:
        return {"Win": self.Win.copy(), "W": self.W.copy(), "Wout": self.Wout.copy()}

    def load_snapshot(self, snap: Dict[str, np.ndarray]) -> None:
        self.Win = snap["Win"].copy()
        self.W = snap["W"].copy()
        self.Wout = snap["Wout"].copy()
        self.x = np.zeros((self.n_reservoir,), dtype=np.float64)

    # ----- batch prediction (evaluation) -----
    def predict_segment_scaled(self, U_scaled: np.ndarray, washout: int = 0) -> np.ndarray:
        self.x = np.zeros((self.n_reservoir,), dtype=np.float64)
        yhat_list = []
        for t in range(U_scaled.shape[0]):
            yhat_s, _ = self.predict_scaled_and_features(U_scaled[t])
            yhat_list.append(yhat_s)
        yhat_arr = np.vstack(yhat_list)
        if washout > 0:
            yhat_arr = yhat_arr[washout:]
        return yhat_arr
