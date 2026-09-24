import math

import numpy as np
import pytest

from pricing import Model, bs_call, bs_otm, call_prices, density, implied_vol_otm, otm_prices
from surfaces import smile

S, R, Q = 100.0, 0.02, 0.01
K = np.array([70.0, 90.0, 100.0, 110.0, 140.0])


def mc_calls(m: Model, T: float, n=200_000, steps=200, seed=3):
    """Euler Monte Carlo under Q, forward moment-matched."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    x = np.zeros(n)
    v = np.full(n, m.v0 if m.has_sv else m.sigma ** 2)
    kj = m.jump_comp()
    for _ in range(steps):
        vp = np.maximum(v, 0)
        z1 = rng.standard_normal(n)
        x += -(0.5 * vp + m.lam * kj * (m.jump_type is not None)) * dt + np.sqrt(vp * dt) * z1
        if m.has_sv:
            z2 = m.rho * z1 + np.sqrt(1 - m.rho ** 2) * rng.standard_normal(n)
            v += m.kappa * (m.theta - vp) * dt + m.xi * np.sqrt(vp * dt) * z2
        if m.jump_type is not None:
            nj = rng.poisson(m.lam * dt, n)
            if m.jump_type == "normal":
                x += nj * m.mu_j + np.sqrt(nj) * m.delta_j * rng.standard_normal(n)
            else:
                hit = nj > 0
                up = rng.random(n) < m.p_up
                size = np.where(up, rng.exponential(1 / m.eta1, n), -rng.exponential(1 / m.eta2, n))
                x += np.where(hit, size, 0.0)  # at most one jump per small step
    F = S * np.exp((R - Q) * T)
    ST = F * np.exp(x)
    ST *= F / ST.mean()
    return np.exp(-R * T) * np.maximum(ST[:, None] - K, 0).mean(0)


def test_cf_pricer_matches_black_scholes_everywhere():
    m = Model("merton", lam=0.0, sigma=0.2)  # BS dynamics through the Fourier pricer
    for T in (1 / 52, 1 / 12, 0.25, 1.0, 2.0):
        F = S * np.exp((R - Q) * T)
        Kg = F * np.exp(np.linspace(-0.2, 0.2, 41))
        px, _ = otm_prices(S, Kg, T, R, Q, m)
        np.testing.assert_allclose(px, bs_otm(S, Kg, T, R, Q, 0.2), atol=1e-12)


def test_merton_matches_series():
    m = Model("merton", sigma=0.15, lam=0.5, mu_j=-0.1, delta_j=0.1)
    T = 0.5
    k = m.jump_comp()
    lp = m.lam * (1 + k)
    ref = 0.0
    for n in range(60):
        sn = math.sqrt(m.sigma ** 2 + n * m.delta_j ** 2 / T)
        rn = R - m.lam * k + n * math.log(1 + k) / T
        ref += math.exp(-lp * T) * (lp * T) ** n / math.factorial(n) * bs_call(S, K, T, rn, Q, sn)
    np.testing.assert_allclose(call_prices(S, K, T, R, Q, m), ref, atol=1e-8)


@pytest.mark.parametrize("m", [
    Model("heston", kappa=1.5, theta=0.04, v0=0.04, rho=-0.7, xi=0.5),
    Model("kou", sigma=0.15, lam=1.0, p_up=0.3, eta1=25, eta2=10),
    Model("bates", kappa=1.5, theta=0.04, v0=0.05, rho=-0.6, xi=0.4, lam=0.5, mu_j=-0.1, delta_j=0.1),
], ids=["heston", "kou", "bates"])
def test_models_match_monte_carlo(m):
    T = 0.5
    np.testing.assert_allclose(call_prices(S, K, T, R, Q, m), mc_calls(m, T), atol=0.06)


@pytest.mark.parametrize("kind", ["heston", "merton", "kou", "bates"])
def test_density_is_a_martingale_density(kind):
    m = Model(kind)
    T = 0.25
    x = np.linspace(-1.5, 1.0, 5001)
    f = density(x, T, m)
    dx = x[1] - x[0]
    assert abs(f.sum() * dx - 1) < 2e-3
    assert abs((np.exp(x) * f).sum() * dx - 1) < 2e-3


def test_iv_roundtrip():
    T = 0.3
    F = S * np.exp((R - Q) * T)
    Kg = F * np.exp(np.linspace(-0.3, 0.3, 13))
    for sig in (0.08, 0.25, 0.9):
        px = bs_otm(S, Kg, T, R, Q, sig)
        ok = px > 1e-10  # below this an implied vol is not meaningful
        np.testing.assert_allclose(implied_vol_otm(px, S, Kg, T, R, Q)[ok], sig, atol=1e-7)


def test_smile_shapes():
    k = np.array([-0.1, 0.0, 0.1])
    # negative rho -> downward skew
    iv = smile(Model("heston", rho=-0.8), 0.25, k, R, Q)
    assert iv[0] > iv[1] > iv[2]
    # rho = 0 -> symmetric-ish smile (curvature, little skew)
    iv0 = smile(Model("heston", rho=0.0, xi=0.8), 0.25, np.array([-0.1, 0.0, 0.1]), R, Q)
    assert iv0[0] > iv0[1] and iv0[2] > iv0[1]
    # Black-Scholes is flat
    np.testing.assert_allclose(smile(Model("bs", sigma=0.3), 0.25, k, R, Q), 0.3)
