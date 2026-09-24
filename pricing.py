"""Characteristic-function option pricing for five models.

    Black-Scholes   dS/S = (r - q) dt + sigma dW
    Heston (1993)   dv = kappa (theta - v) dt + xi sqrt(v) dZ,  corr(dW, dZ) = rho
    Merton (1976)   GBM + Poisson jumps, log-jump ~ N(mu_J, delta^2)
    Kou (2002)      GBM + Poisson jumps, double-exponential log-jump
                    (up with prob p, mean 1/eta1; down with prob 1-p, mean 1/eta2)
    Bates (1996)    Heston + Merton jumps

All are priced with the Lewis (2001) single-integral formula

    C = e^{-rT} [ F - sqrt(F K) / pi * int_0^inf Re( e^{i u x} phi(u - i/2) ) / (u^2 + 1/4) du ],

x = ln(F / K), phi the characteristic function of X_T = ln(S_T / F). The integral is
done with composite Gauss-Legendre whose range adapts to maturity and volatility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

MODEL_NAMES = {
    "bs": "Black-Scholes",
    "heston": "Heston",
    "merton": "Merton",
    "kou": "Kou",
    "bates": "Bates",
}


@dataclass(frozen=True)
class Model:
    kind: str = "bs"
    sigma: float = 0.20      # diffusion vol (bs / merton / kou)
    kappa: float = 2.0       # Heston/Bates: mean-reversion speed
    theta: float = 0.04      # long-run variance
    v0: float = 0.04         # initial variance
    rho: float = -0.7        # spot/vol correlation
    xi: float = 0.5          # vol of vol
    lam: float = 0.5         # jump intensity per year
    mu_j: float = -0.10      # Merton/Bates: mean log-jump
    delta_j: float = 0.10    # Merton/Bates: std of log-jump
    p_up: float = 0.3        # Kou: prob of an up-jump
    eta1: float = 25.0       # Kou: up-jump rate (mean up-jump = 1/eta1)
    eta2: float = 10.0       # Kou: down-jump rate (mean down-jump = 1/eta2)

    @property
    def has_sv(self) -> bool:
        return self.kind in ("heston", "bates")

    @property
    def jump_type(self) -> str | None:
        if self.lam <= 0:
            return None
        return {"merton": "normal", "bates": "normal", "kou": "dexp"}.get(self.kind)

    def min_variance(self) -> float:
        if self.has_sv:
            return min(self.theta, self.v0)
        return self.sigma ** 2

    def jump_comp(self) -> float:
        """E[e^J] - 1."""
        jt = self.jump_type
        if jt == "normal":
            return float(np.exp(self.mu_j + 0.5 * self.delta_j ** 2) - 1)
        if jt == "dexp":
            return float(self.p_up * self.eta1 / (self.eta1 - 1)
                         + (1 - self.p_up) * self.eta2 / (self.eta2 + 1) - 1)
        return 0.0

    def feller(self) -> float:
        """2 kappa theta - xi^2 (>= 0 means the variance never touches zero)."""
        return 2 * self.kappa * self.theta - self.xi ** 2


def log_cf(u, T: float, m: Model):
    """log E[exp(i u X_T)], X_T = ln(S_T / F_T). Works for complex u."""
    u = np.asarray(u, complex)
    iu = 1j * u
    if m.has_sv:
        k, th, xi, rho, v0 = m.kappa, m.theta, m.xi, m.rho, m.v0
        b = k - rho * xi * iu
        d = np.sqrt(b ** 2 + xi ** 2 * (iu + u ** 2))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        out = (k * th / xi ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
               + v0 * (b - d) / xi ** 2 * (1 - e) / (1 - g * e))
    else:
        out = -0.5 * m.sigma ** 2 * T * (iu + u ** 2)
    jt = m.jump_type
    if jt == "normal":
        phi_j = np.exp(iu * m.mu_j - 0.5 * m.delta_j ** 2 * u ** 2)
    elif jt == "dexp":
        phi_j = m.p_up * m.eta1 / (m.eta1 - iu) + (1 - m.p_up) * m.eta2 / (m.eta2 + iu)
    else:
        return out
    return out + m.lam * T * (phi_j - 1 - iu * m.jump_comp())


# composite Gauss-Legendre nodes on [0, U]
_GX, _GW = np.polynomial.legendre.leggauss(32)


def _nodes(U: float, seg: float = 20.0):
    # fine near 0 (the 1/(u^2 + 1/4) peak), then uniform segments out to U
    head = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0, seg])
    n = max(1, int(np.ceil((U - seg) / seg)))
    edges = np.concatenate([head, np.linspace(seg, max(U, 2 * seg), n + 1)[1:]])
    a, b = edges[:-1, None], edges[1:, None]
    u = (0.5 * (b - a) * (_GX + 1) + a).ravel()
    w = (0.5 * (b - a) * _GW[None, :]).ravel()
    return u, w


def _upper_limit(m: Model, T: float) -> float:
    v = max(m.min_variance(), 2.5e-3)
    return float(min(max(60.0, 18.0 / np.sqrt(v * T)), 6000.0))


def call_prices(S: float, K, T: float, r: float, q: float, m: Model):
    """European calls for one maturity and a vector of strikes."""
    K = np.atleast_1d(np.asarray(K, float))
    F = S * np.exp((r - q) * T)
    if m.kind == "bs":
        return bs_call(S, K, T, r, q, m.sigma)
    u, w = _nodes(_upper_limit(m, T))
    phi = np.exp(log_cf(u - 0.5j, T, m))
    x = np.log(F / K)[:, None]
    integrand = np.real(np.exp(1j * u[None, :] * x) * phi[None, :]) / (u ** 2 + 0.25)
    call = np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * (integrand @ w))
    lower = np.exp(-r * T) * np.maximum(F - K, 0.0)
    return np.clip(call, lower, S * np.exp(-q * T))


def otm_prices(S: float, K, T: float, r: float, q: float, m: Model):
    """Out-of-the-money option prices (puts below the forward, calls above)."""
    K = np.atleast_1d(np.asarray(K, float))
    F = S * np.exp((r - q) * T)
    c = call_prices(S, K, T, r, q, m)
    p = c - np.exp(-r * T) * (F - K)
    return np.where(K >= F, c, p), K >= F


def density(x, T: float, m: Model):
    """Risk-neutral density of X_T = ln(S_T / F) by Fourier inversion."""
    x = np.atleast_1d(np.asarray(x, float))
    if m.kind == "bs":
        s = m.sigma * np.sqrt(T)
        return np.exp(-0.5 * ((x + 0.5 * s * s) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    u, w = _nodes(_upper_limit(m, T))
    phi = np.exp(log_cf(u, T, m))
    f = np.real(np.exp(-1j * u[None, :] * x[:, None]) * phi[None, :]) @ w / np.pi
    return np.maximum(f, 0.0)


# ---------------------------------------------------------------------------
# Black-Scholes and implied vol
# ---------------------------------------------------------------------------

def bs_call(S, K, T, r, q, sigma):
    K = np.asarray(K, float)
    sigma = np.asarray(sigma, float)
    vs = np.maximum(sigma * np.sqrt(T), 1e-12)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * vs ** 2) / vs
    return np.exp(-r * T) * (F * ndtr(d1) - K * ndtr(d1 - vs))


def bs_otm(S, K, T, r, q, sigma):
    K = np.asarray(K, float)
    F = S * np.exp((r - q) * T)
    c = bs_call(S, K, T, r, q, sigma)
    return np.where(K >= F, c, c - np.exp(-r * T) * (F - K))


def implied_vol_otm(price, S, K, T, r, q, lo=1e-3, hi=5.0, iters=70):
    """Vectorised bisection on OTM prices (monotone in vol, no cancellation)."""
    price = np.asarray(price, float)
    K = np.asarray(K, float)
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        high = bs_otm(S, K, T, r, q, mid) > price
        b = np.where(high, mid, b)
        a = np.where(high, a, mid)
    iv = 0.5 * (a + b)
    ok = (price > bs_otm(S, K, T, r, q, lo) * (1 + 1e-9)) & (price < bs_otm(S, K, T, r, q, hi))
    return np.where(ok, iv, np.nan)
