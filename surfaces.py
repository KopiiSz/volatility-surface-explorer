"""Implied-vol surfaces, smiles, term structures and summary numbers."""
from __future__ import annotations

import numpy as np

from pricing import Model, bs_otm, density, implied_vol_otm, otm_prices

S0 = 100.0
PRICE_FLOOR = 1e-12  # OTM prices below this (x S) are numerical noise: no meaningful IV

TENORS = {"1W": 1 / 52, "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0}


def maturity_grid(t_max: float, n: int = 24) -> np.ndarray:
    return np.geomspace(1 / 52, t_max, n)


def smile(m: Model, T: float, k: np.ndarray, r: float, q: float) -> np.ndarray:
    """Implied vol across log-moneyness k = ln(K/F) for one maturity."""
    if m.kind == "bs":
        return np.full(k.shape, m.sigma)
    F = S0 * np.exp((r - q) * T)
    K = F * np.exp(k)
    px, _ = otm_prices(S0, K, T, r, q, m)
    iv = implied_vol_otm(px, S0, K, T, r, q)
    return np.where(px > PRICE_FLOOR * S0, iv, np.nan)


def surface(m: Model, T_grid: np.ndarray, k: np.ndarray, r: float, q: float) -> np.ndarray:
    return np.vstack([smile(m, T, k, r, q) for T in T_grid])


def atm_vol(m: Model, T: float, r: float, q: float) -> float:
    return float(smile(m, T, np.array([0.0]), r, q)[0])


def summary(m: Model, r: float, q: float) -> dict:
    k = np.array([-0.1, 0.0, 0.1])
    s3 = smile(m, 0.25, k, r, q)
    return {
        "atm_1m": atm_vol(m, 1 / 12, r, q),
        "atm_1y": atm_vol(m, 1.0, r, q),
        "skew_3m": float(s3[0] - s3[2]),
        "fly_3m": float(0.5 * (s3[0] + s3[2]) - s3[1]),
    }


def log_return_density(m: Model, T: float, x: np.ndarray) -> np.ndarray:
    return density(x, T, m)


def jump_variance_share(m: Model) -> float | None:
    """Share of annual return variance that comes from jumps."""
    jt = m.jump_type
    if jt is None:
        return None
    if jt == "normal":
        jv = m.lam * (m.mu_j ** 2 + m.delta_j ** 2)
    else:  # double exponential: E[J^2] = 2p/eta1^2 + 2(1-p)/eta2^2
        jv = m.lam * (2 * m.p_up / m.eta1 ** 2 + 2 * (1 - m.p_up) / m.eta2 ** 2)
    dv = m.theta if m.has_sv else m.sigma ** 2
    return float(jv / (jv + dv))


__all__ = ["S0", "TENORS", "maturity_grid", "smile", "surface", "atm_vol", "summary",
           "log_return_density", "jump_variance_share", "bs_otm"]
