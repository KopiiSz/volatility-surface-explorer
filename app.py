"""Volatility Surface Explorer: implied-vol surfaces of Black-Scholes, Heston, Merton, Kou and Bates.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts  # noqa: E402
import content as C  # noqa: E402
from pricing import Model  # noqa: E402
from surfaces import TENORS, atm_vol, jump_variance_share, maturity_grid, smile, surface, summary  # noqa: E402
from pricing import density  # noqa: E402

st.set_page_config(page_title="Volatility Surface Explorer", page_icon=":material/landscape:",
                   layout="wide", initial_sidebar_state="auto")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e6e5df; --card:#fcfcfb; --accent:#4a3aa7; }
html, body, [class*="css"], .stMarkdown, button, input, label { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1480px; }
.vs-top { font-size: .78rem; font-weight: 700; letter-spacing: .16em; color: var(--ink2); text-transform: uppercase;
          padding-bottom: .9rem; border-bottom: 1px solid var(--line); margin-bottom: .3rem; }
.vs-top span { color: var(--muted); font-weight: 500; letter-spacing: .08em; }
.vs-eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .14em; color: var(--accent); margin-top: .6rem; }
.vs-title { font-family: "Source Serif 4", Georgia, serif; font-size: 2rem; font-weight: 600; color: var(--ink); margin: .15rem 0 .4rem; line-height: 1.15; }
.vs-intro { font-family: "Source Serif 4", Georgia, serif; font-size: 1.08rem; line-height: 1.6; color: var(--ink2); max-width: 820px; margin-bottom: .6rem; }
.vs-card-title { font-size: .7rem; font-weight: 700; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; }
.vs-fixed { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: .8rem; color: var(--ink2);
            padding: .5rem 0 .6rem; border-bottom: 1px solid var(--line); margin-bottom: .4rem; }
.vs-status { font-size: .8rem; line-height: 1.45; border-radius: 8px; padding: 8px 10px; margin-top: .4rem; }
.vs-ok { background:#ecf8f1; border:1px solid #9fd8b8; color:#0d5b2f; }
.vs-warn { background:#fff6e8; border:1px solid #f3c77a; color:#6b4300; }
.vs-info { background:#f3f2ee; border:1px solid var(--line); color: var(--ink2); }
.vs-try { background: #f4f2fd; border-left: 3px solid var(--accent); border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 1.2rem 0 .8rem; }
.vs-try h4 { font-size: .74rem; font-weight: 700; letter-spacing: .14em; color: var(--accent); margin: 0 0 .5rem; }
.vs-try ol { margin: 0; padding-left: 1.2rem; font-family: "Source Serif 4", Georgia, serif; font-size: 1.02rem; line-height: 1.6; color: #2b2a28; }
.vs-try li { margin-bottom: .35rem; }
.vs-method { font-family: "Source Serif 4", Georgia, serif; color: var(--muted); font-size: .95rem; line-height: 1.6;
             max-width: 860px; border-top: 1px solid var(--line); padding-top: 1rem; margin-top: .6rem; }
div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
div[data-testid="stMetricValue"] { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 1.3rem; }
div[data-testid="stPlotlyChart"] { background:#ffffff; border:1px solid var(--line); border-radius: 10px; padding: 4px; }
div[data-testid="stTabs"] button p { font-size: .95rem; font-weight: 500; }
section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

ss = st.session_state


# ---------------------------------------------------------------------------
# Parameter state
# ---------------------------------------------------------------------------
def pkey(kind: str, key: str) -> str:
    return f"{kind}__{key}"


def reset(kind: str):
    for p in C.MODELS[kind]["params"]:
        ss[pkey(kind, p[0])] = p[4]


for _kind in C.MODEL_ORDER:
    for _p in C.MODELS[_kind]["params"]:
        ss.setdefault(pkey(_kind, _p[0]), _p[4])


def model_from_state(kind: str) -> Model:
    kw = {}
    for key, _label, _lo, _hi, _d, _s, _f, how, _h in C.MODELS[kind]["params"]:
        v = float(ss[pkey(kind, key)])
        kw[key] = v / 100 if how == "pct" else (v / 100) ** 2 if how == "vpct" else v
    return Model(kind=kind, **kw)


# ---------------------------------------------------------------------------
# Sidebar: market & grid
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.markdown("### Market & grid")
r = sb.slider("Rate r", 0.0, 10.0, 2.0, 0.25, format="%.2f%%",
              help="Continuously compounded risk-free rate. It moves the forward F = S·e^{(r−q)T}; "
                   "because strikes are measured relative to F, the surface shape barely changes.") / 100
q = sb.slider("Dividend yield q", 0.0, 10.0, 0.0, 0.25, format="%.2f%%",
              help="Continuous dividend yield. Also enters only through the forward.") / 100
t_label = sb.select_slider("Longest maturity", options=["6M", "1Y", "2Y"], value="1Y",
                           help="Surfaces run from 1 week to this maturity (log-spaced).")
t_max = TENORS[t_label]
width = sb.slider("Strike range ± (log-moneyness)", 10, 40, 20, 5, format="%d%%",
                  help="Strikes from ln(K/F) = −x to +x. ±20% means strikes from about 82% to 122% of the forward.") / 100
show_bs = sb.toggle("Show Black-Scholes reference", value=True,
                    help="Overlay the flat Black-Scholes surface (grey plane) and dashed reference lines, "
                         "at each model's 3-month ATM vol.")
sb.caption("Spot S = 100. Implied vols don't depend on the spot level.")


# ---------------------------------------------------------------------------
# Computation (cached per parameter set)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=64)
def compute(m: Model, t_max: float, width: float, r: float, q: float) -> dict:
    T = maturity_grid(t_max, 24)
    k = np.linspace(-width, width, 33)
    iv = surface(m, T, k, r, q)
    wings = np.vstack([smile(m, t, np.array([-0.1, 0.0, 0.1]), r, q) for t in T])
    k_fine = np.linspace(-width, width, 81)
    smiles = {lab: smile(m, t, k_fine, r, q) for lab, t in TENORS.items() if t <= t_max * 1.001}
    return dict(T=T, k=k, iv=iv, atm=wings[:, 1], skew=wings[:, 0] - wings[:, 2],
                fly=0.5 * (wings[:, 0] + wings[:, 2]) - wings[:, 1], k_fine=k_fine, smiles=smiles,
                summary=summary(m, r, q), atm3m=atm_vol(m, 0.25, r, q))


@st.cache_data(show_spinner=False, max_entries=64)
def density_data(m: Model, T: float, r: float, q: float):
    atm = atm_vol(m, T, r, q)
    s = atm * np.sqrt(T)
    x = np.linspace(-(6 * s + 0.12), 5 * s + 0.06, 500)
    f = density(x, T, m)
    f_bs = np.exp(-0.5 * ((x + 0.5 * s * s) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return x, f, f_bs, atm


def fmt_vol(v):
    return "–" if v is None or not np.isfinite(v) else f"{v * 100:.1f}%"


def fmt_pts(v):
    return "–" if v is None or not np.isfinite(v) else f"{v * 100:+.2f} pts"


CFG = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown('<div class="vs-top">Volatility Surface Explorer <span>· Black-Scholes, Heston, Merton, Kou, Bates</span></div>',
            unsafe_allow_html=True)
tabs = st.tabs([C.MODELS[k]["tab"] for k in C.MODEL_ORDER] + ["Compare all"])


def render_model(kind: str):
    spec = C.MODELS[kind]
    st.markdown(f'<div class="vs-eyebrow">{spec["eyebrow"]}</div><div class="vs-title">{spec["title"]}</div>'
                f'<div class="vs-intro">{spec["intro"]}</div>', unsafe_allow_html=True)
    with st.expander("Model dynamics"):
        for line in spec["latex"]:
            st.latex(line)

    left, right = st.columns([1, 2.7], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="vs-card-title">Parameters</div>'
                        f'<div class="vs-fixed">S = 100<br>r = {r * 100:.2f}%, q = {q * 100:.2f}%</div>',
                        unsafe_allow_html=True)
            for key, label, lo, hi, _d, step, fmt, _how, help_ in spec["params"]:
                st.slider(label, lo, hi, step=step, format=fmt, key=pkey(kind, key), help=help_)
            m = model_from_state(kind)
            if m.has_sv:
                fel = m.feller()
                cls, word = ("vs-ok", "satisfied") if fel >= 0 else ("vs-warn", "violated")
                st.markdown(f'<div class="vs-status {cls}">Feller condition 2κθ ≥ σ² {word} '
                            f'(2κθ − σ² = {fel:+.3f}). Half-life of vol shocks ≈ {np.log(2) / m.kappa * 12:.1f} months.</div>',
                            unsafe_allow_html=True)
            share = jump_variance_share(m)
            if share is not None:
                st.markdown(f'<div class="vs-status vs-info">Jumps account for <b>{share * 100:.0f}%</b> of '
                            f'long-run return variance.</div>', unsafe_allow_html=True)
            st.button("Reset to defaults", key=f"reset_{kind}", on_click=reset, args=(kind,), width="stretch")

    d = compute(m, t_max, width, r, q)
    bs_level = d["atm3m"] if (show_bs and kind != "bs") else None
    with right:
        with st.container(border=True):
            st.markdown('<div class="vs-card-title">Implied volatility surface</div>', unsafe_allow_html=True,
                        help=C.HELP["surface"])
            st.plotly_chart(charts.surface_fig(d["T"], d["k"], d["iv"], bs_level), width="stretch",
                            theme=None, config={"displayModeBar": False, "scrollZoom": True}, key=f"surf_{kind}")
            cap = "Drag to rotate, scroll to zoom, double-click to reset."
            if bs_level is not None:
                cap += f" Grey plane: Black-Scholes at the 3M ATM vol ({bs_level * 100:.1f}%)."
            st.caption(cap)

    s = d["summary"]
    cols = st.columns(5)
    cols[0].metric("ATM vol · 1M", fmt_vol(s["atm_1m"]), help=C.HELP["atm_1m"])
    cols[1].metric("ATM vol · 1Y", fmt_vol(s["atm_1y"]), help=C.HELP["atm_1y"])
    cols[2].metric("Skew · 3M", fmt_pts(s["skew_3m"]), help=C.HELP["skew"])
    cols[3].metric("Butterfly · 3M", fmt_pts(s["fly_3m"]), help=C.HELP["fly"])
    if m.has_sv:
        cols[4].metric("Feller 2κθ − σ²", f"{m.feller():+.3f}", help=C.HELP["feller"])
    elif share is not None:
        cols[4].metric("Jump var. share", f"{share * 100:.0f}%", help=C.HELP["jump_share"])
    else:
        cols[4].metric("Excess kurtosis", "0.00", help="Black-Scholes log-returns are exactly normal: "
                                                        "no skewness, no excess kurtosis.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.smiles_fig(d["k_fine"], d["smiles"], bs_level if kind != "bs" else None),
                        width="stretch", theme=None, config=CFG, key=f"smiles_{kind}")
        st.caption(C.HELP["smiles"])
    with c2:
        st.plotly_chart(charts.term_fig(d["T"], d["atm"], bs_level, t_max), width="stretch", theme=None,
                        config=CFG, key=f"term_{kind}")
        st.caption(C.HELP["term"])

    c1, c2 = st.columns(2)
    with c1:
        opts = [lab for lab, t in TENORS.items() if t <= t_max * 1.001]
        cc1, cc2 = st.columns([2, 1])
        dlab = cc1.select_slider("Density maturity", options=opts, value="3M" if "3M" in opts else opts[-1],
                                 key=f"dens_T_{kind}", help=C.HELP["density"])
        log_y = cc2.toggle("Log scale", value=True, key=f"dens_log_{kind}",
                           help="Log scale makes the tails visible. A normal density is a parabola here.")
        x, f, f_bs, _ = density_data(m, TENORS[dlab], r, q)
        st.plotly_chart(charts.density_fig(x, f, None if kind == "bs" else f_bs, C.MODELS[kind]["tab"], log_y,
                                           title=f"Risk-neutral density · {dlab}"),
                        width="stretch", theme=None, config=CFG, key=f"dens_{kind}")
    with c2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.plotly_chart(charts.skew_term_fig(d["T"], d["skew"], d["fly"], t_max), width="stretch", theme=None,
                        config=CFG, key=f"skewterm_{kind}")
        st.caption(C.HELP["skewterm"])

    items = "".join(f"<li>{t}</li>" for t in spec["tryit"])
    st.markdown(f'<div class="vs-try"><h4>TRY IT YOURSELF</h4><ol>{items}</ol></div>', unsafe_allow_html=True)


for tab, kind in zip(tabs, C.MODEL_ORDER):
    with tab:
        render_model(kind)

with tabs[-1]:
    st.markdown('<div class="vs-eyebrow">SIDE BY SIDE</div><div class="vs-title">Five models, one maturity</div>'
                '<div class="vs-intro">Each model uses the parameters currently set in its own tab. '
                'Black-Scholes (dashed) is flat. Every departure from it is a statement about how the '
                'return distribution differs from the normal.</div>', unsafe_allow_html=True)
    opts = [lab for lab, t in TENORS.items() if t <= t_max * 1.001]
    cc1, cc2, _ = st.columns([2, 1, 2])
    clab = cc1.select_slider("Maturity", options=opts, value="3M" if "3M" in opts else opts[-1], key="cmp_T")
    clog = cc2.toggle("Log-scale densities", value=True, key="cmp_log")
    Tc = TENORS[clab]
    models = {kind: model_from_state(kind) for kind in C.MODEL_ORDER}
    k_fine = np.linspace(-width, width, 81)
    smile_curves = {kind: (C.MODELS[kind]["tab"], smile(m, Tc, k_fine, r, q)) for kind, m in models.items()}
    dens_curves = {}
    x_ref = None
    for kind, m in models.items():
        x, f, _, _ = density_data(m, Tc, r, q)
        if x_ref is None:
            atms = [atm_vol(mm, Tc, r, q) for mm in models.values()]
            sm = max(a for a in atms if np.isfinite(a)) * np.sqrt(Tc)
            x_ref = np.linspace(-(6 * sm + 0.12), 5 * sm + 0.06, 500)
        dens_curves[kind] = (C.MODELS[kind]["tab"], density(x_ref, Tc, m))
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.compare_smiles_fig(k_fine, smile_curves), width="stretch", theme=None, config=CFG,
                    key="cmp_smiles")
    c2.plotly_chart(charts.compare_density_fig(x_ref, dens_curves, clog), width="stretch", theme=None,
                    config=CFG, key="cmp_dens")
    term_curves = {kind: (C.MODELS[kind]["tab"], compute(m, t_max, width, r, q)["atm"]) for kind, m in models.items()}
    T_grid = compute(models["bs"], t_max, width, r, q)["T"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.compare_term_fig(T_grid, term_curves, t_max), width="stretch", theme=None,
                    config=CFG, key="cmp_term")
    with c2:
        rows = []
        for kind, m in models.items():
            sm = compute(m, t_max, width, r, q)["summary"]
            iv3 = smile(m, Tc, np.array([-0.1, 0.0, 0.1]), r, q)
            rows.append({"Model": C.MODELS[kind]["tab"], f"ATM {clab}": fmt_vol(iv3[1]),
                         f"Skew {clab}": fmt_pts(iv3[0] - iv3[2]),
                         f"Butterfly {clab}": fmt_pts(0.5 * (iv3[0] + iv3[2]) - iv3[1]),
                         "ATM 1M": fmt_vol(sm["atm_1m"]), "ATM 1Y": fmt_vol(sm["atm_1y"])})
        st.markdown('<div class="vs-card-title" style="margin:.6rem 0 .4rem">Summary</div>', unsafe_allow_html=True,
                    help="Skew = IV(ln K/F = −10%) − IV(+10%). Butterfly = average of those wings minus ATM. "
                         "Both are zero under Black-Scholes.")
        st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")
    st.markdown('<div class="vs-try"><h4>TRY IT YOURSELF</h4><ol>'
                '<li>Tune Merton and Heston so that their 3M ATM vols and skews roughly match. Now compare '
                'their 1W and 1Y smiles. Which model keeps its skew at long maturities, and why?</li>'
                '<li>On the log-scale densities, rank the models by how fat their left tail is. Does the ranking '
                'match the ranking of 3M skew?</li>'
                '<li>Two models can agree on every ATM vol and still price a −20% put very differently. '
                'Which risk is a trader exposed to if they calibrate only to ATM options?</li></ol></div>',
                unsafe_allow_html=True)

st.markdown(f'<div class="vs-method">{C.METHOD_NOTE}</div>', unsafe_allow_html=True)
