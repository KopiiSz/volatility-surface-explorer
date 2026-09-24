"""Plotly figures, light theme."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#ecebe6"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
REF = "#898781"  # Black-Scholes reference (dashed)
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# sequential blue ramp (light -> dark)
SURFACE_SCALE = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
# ordinal ramp for maturities (short = light, long = dark); starts at step 250 for contrast
TENOR_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
MODEL_COLORS = {"bs": REF, "heston": "#2a78d6", "merton": "#eb6834", "kou": "#1baf7a", "bates": "#4a3aa7"}

TENOR_TICKS = ([1 / 52, 1 / 12, 0.25, 0.5, 1.0, 2.0], ["1W", "1M", "3M", "6M", "1Y", "2Y"])


def _style(fig: go.Figure, height: int, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=44 if title else 16, b=112 if legend else 56),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        font=dict(family=FONT, size=12, color=INK2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=MONO, size=11, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0,
                    font=dict(size=11, color=INK2), bgcolor="rgba(0,0,0,0)"),
        uirevision="keep",
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.01, y=0.985, yanchor="top",
                                     font=dict(size=13, color=INK)))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS, ticks="outside",
                     tickcolor=AXIS, ticklen=4, tickfont=dict(family=MONO, size=10, color=MUTED),
                     title_font=dict(size=11, color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS,
                     tickfont=dict(family=MONO, size=10, color=MUTED), title_font=dict(size=11, color=MUTED))
    return fig


def _tenor_axis(fig, t_max, **kw):
    vals, text = TENOR_TICKS
    keep = [i for i, v in enumerate(vals) if v <= t_max * 1.001]
    fig.update_xaxes(type="log", tickvals=[vals[i] for i in keep], ticktext=[text[i] for i in keep], **kw)


def surface_fig(T: np.ndarray, k: np.ndarray, iv: np.ndarray, bs_level: float | None,
                height: int = 560) -> go.Figure:
    z = iv * 100
    finite = z[np.isfinite(z)]
    zmin, zmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if bs_level is not None:
        zmin, zmax = min(zmin, bs_level * 100), max(zmax, bs_level * 100)
    if zmax - zmin < 6:  # nearly flat (e.g. Black-Scholes): keep a sensible vertical scale
        mid = 0.5 * (zmin + zmax)
        zmin, zmax = mid - 3, mid + 3
    pad = max(0.6, 0.12 * (zmax - zmin))
    x = k * 100
    y = np.log10(T)  # log-maturity on a linear axis (robust in 3D scenes)
    Tcd = np.repeat(T[:, None], len(k), axis=1)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, customdata=Tcd, colorscale=SURFACE_SCALE, cmin=zmin, cmax=zmax, name="Model",
        colorbar=dict(title=dict(text="IV %", font=dict(size=11, color=MUTED)), thickness=12, len=0.6,
                      tickfont=dict(family=MONO, size=10, color=MUTED), outlinewidth=0),
        lighting=dict(ambient=0.8, diffuse=0.6, specular=0.05, roughness=0.9),
        hovertemplate="ln(K/F) %{x:+.1f}%<br>T %{customdata:.2f}y<br>IV %{z:.2f}%<extra></extra>",
    ))
    if bs_level is not None:
        fig.add_trace(go.Surface(
            x=x, y=y, z=np.full_like(z, bs_level * 100), showscale=False, opacity=0.35,
            colorscale=[[0, "#9f9e97"], [1, "#9f9e97"]], name="Black-Scholes",
            hovertemplate="Black-Scholes (flat)<br>IV %{z:.2f}%<extra></extra>"))
    vals, text = TENOR_TICKS
    keep = [i for i, v in enumerate(vals) if T[0] * 0.999 <= v <= T[-1] * 1.001]
    axis = dict(backgroundcolor="#ffffff", gridcolor="#e6e5df", showbackground=True, zerolinecolor=AXIS,
                tickfont=dict(family=MONO, size=10, color=MUTED), title_font=dict(size=11, color=INK2))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK2), uirevision="surface", showlegend=False,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=MONO, size=11, color=INK)),
        scene=dict(
            xaxis=dict(title="ln(K/F) %", **axis),
            yaxis=dict(title="maturity", tickvals=[float(np.log10(vals[i])) for i in keep],
                       ticktext=[text[i] for i in keep], **axis),
            zaxis=dict(title="implied vol %", range=[zmin - 3 * pad, zmax + pad], **axis),
            camera=dict(eye=dict(x=-1.45, y=-1.55, z=1.25), center=dict(x=0, y=0, z=-0.12)),
            aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=0.62),
        ),
    )
    return fig


def smiles_fig(k: np.ndarray, smiles: dict, bs_level: float | None, height: int = 430) -> go.Figure:
    fig = go.Figure()
    for (label, iv), col in zip(smiles.items(), TENOR_RAMP[-len(smiles):] if len(smiles) <= 6 else TENOR_RAMP):
        fig.add_trace(go.Scatter(x=k * 100, y=iv * 100, mode="lines", name=label, line=dict(color=col, width=2),
                                 hovertemplate=f"{label}: " + "%{y:.2f}%<extra></extra>"))
    if bs_level is not None:
        fig.add_trace(go.Scatter(x=[k[0] * 100, k[-1] * 100], y=[bs_level * 100] * 2, mode="lines",
                                 name="Black-Scholes (flat)", line=dict(color=REF, width=1.5, dash="dash"),
                                 hoverinfo="skip"))
    fig.update_xaxes(title_text="log-moneyness ln(K/F) %", zeroline=True, zerolinecolor=AXIS)
    fig.update_yaxes(title_text="implied vol %")
    _style(fig, height, "Smile by maturity")
    fig.update_layout(hovermode="x unified")
    return fig


def term_fig(T: np.ndarray, atm: np.ndarray, bs_level: float | None, t_max: float,
             height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=atm * 100, mode="lines", name="ATM implied vol",
                             line=dict(color=BLUE, width=2.4), hovertemplate="%{y:.2f}%<extra></extra>"))
    if bs_level is not None:
        fig.add_trace(go.Scatter(x=[T[0], T[-1]], y=[bs_level * 100] * 2, mode="lines",
                                 name="Black-Scholes (flat)", line=dict(color=REF, width=1.5, dash="dash"),
                                 hoverinfo="skip"))
    _tenor_axis(fig, t_max, title_text="maturity")
    fig.update_yaxes(title_text="implied vol %")
    lo, hi = np.nanmin(atm) * 100, np.nanmax(atm) * 100
    if bs_level is not None:
        lo, hi = min(lo, bs_level * 100), max(hi, bs_level * 100)
    mid, half = (lo + hi) / 2, max((hi - lo) / 2 * 1.3, 1.5)
    fig.update_yaxes(range=[mid - half, mid + half])
    _style(fig, height, "ATM term structure")
    fig.update_layout(hovermode="x unified")
    return fig


def skew_term_fig(T: np.ndarray, skew: np.ndarray, fly: np.ndarray, t_max: float,
                  height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=skew * 100, mode="lines", name="Skew: IV(−10%) − IV(+10%)",
                             line=dict(color=BLUE, width=2.2), hovertemplate="%{y:+.2f} vol pts<extra></extra>"))
    fig.add_trace(go.Scatter(x=T, y=fly * 100, mode="lines", name="Butterfly: wings − ATM",
                             line=dict(color=ORANGE, width=2.2), hovertemplate="%{y:+.2f} vol pts<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=REF, width=1.5, dash="dash"),
                  annotation_text="Black-Scholes = 0", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=MUTED))
    _tenor_axis(fig, t_max, title_text="maturity")
    fig.update_yaxes(title_text="vol points")
    _style(fig, height, "Skew & curvature by maturity")
    fig.update_layout(hovermode="x unified")
    return fig


def density_fig(x: np.ndarray, f_model: np.ndarray, f_bs: np.ndarray | None, label: str, log_y: bool,
                height: int = 430, title: str = "Risk-neutral density") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x * 100, y=f_model / 100, mode="lines", name=label, line=dict(color=BLUE, width=2.4),
                             fill="tozeroy" if not log_y else None, fillcolor="rgba(42,120,214,0.10)",
                             hovertemplate="%{y:.4f}<extra></extra>"))
    if f_bs is not None:
        fig.add_trace(go.Scatter(x=x * 100, y=f_bs / 100, mode="lines", name="Black-Scholes (normal, same ATM vol)",
                                 line=dict(color=REF, width=1.6, dash="dash"), hovertemplate="%{y:.4f}<extra></extra>"))
    _density_axes(fig, [f_model] + ([f_bs] if f_bs is not None else []), log_y)
    _style(fig, height, title)
    fig.update_layout(hovermode="x unified")
    return fig


def _density_axes(fig, arrays, log_y):
    fig.update_xaxes(title_text="log-return ln(S_T / F) %")
    if log_y:
        top = max(float(np.nanmax(a)) for a in arrays) / 100
        fig.update_yaxes(type="log", title_text="density (log scale)", range=[np.log10(top) - 5, np.log10(top) + 0.3])
    else:
        fig.update_yaxes(title_text="density", rangemode="tozero")


def compare_smiles_fig(k, curves: dict, height=460) -> go.Figure:
    fig = go.Figure()
    for kind, (label, iv) in curves.items():
        fig.add_trace(go.Scatter(x=k * 100, y=iv * 100, mode="lines", name=label,
                                 line=dict(color=MODEL_COLORS[kind], width=1.6 if kind == "bs" else 2.4,
                                           dash="dash" if kind == "bs" else "solid"),
                                 hovertemplate=f"{label}: " + "%{y:.2f}%<extra></extra>"))
    fig.update_xaxes(title_text="log-moneyness ln(K/F) %", zeroline=True, zerolinecolor=AXIS)
    fig.update_yaxes(title_text="implied vol %")
    _style(fig, height, "Smiles")
    fig.update_layout(hovermode="x unified")
    return fig


def compare_density_fig(x, curves: dict, log_y: bool, height=460) -> go.Figure:
    fig = go.Figure()
    for kind, (label, f) in curves.items():
        fig.add_trace(go.Scatter(x=x * 100, y=f / 100, mode="lines", name=label,
                                 line=dict(color=MODEL_COLORS[kind], width=1.6 if kind == "bs" else 2.2,
                                           dash="dash" if kind == "bs" else "solid"),
                                 hovertemplate=f"{label}: " + "%{y:.4f}<extra></extra>"))
    _density_axes(fig, [f for _, f in curves.values()], log_y)
    _style(fig, height, "Risk-neutral densities")
    fig.update_layout(hovermode="x unified")
    return fig


def compare_term_fig(T, curves: dict, t_max, height=460) -> go.Figure:
    fig = go.Figure()
    for kind, (label, atm) in curves.items():
        fig.add_trace(go.Scatter(x=T, y=atm * 100, mode="lines", name=label,
                                 line=dict(color=MODEL_COLORS[kind], width=1.6 if kind == "bs" else 2.4,
                                           dash="dash" if kind == "bs" else "solid"),
                                 hovertemplate=f"{label}: " + "%{y:.2f}%<extra></extra>"))
    _tenor_axis(fig, t_max, title_text="maturity")
    fig.update_yaxes(title_text="implied vol %")
    _style(fig, height, "ATM term structures")
    fig.update_layout(hovermode="x unified")
    return fig
