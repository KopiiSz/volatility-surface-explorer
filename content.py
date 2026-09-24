"""Text content: model descriptions, parameter definitions with "?" help, exercises."""

# Each parameter: (key, label, lo, hi, default, step, fmt, kind, help)
#   kind: "raw"  -> value used as is
#         "pct"  -> slider in %, model gets value / 100
#         "vpct" -> slider is a vol in %, model gets the variance (value / 100)^2

P_SIGMA = ("sigma", "σ — diffusion vol", 5.0, 80.0, 20.0, 1.0, "%.0f%%", "pct",
           "Volatility of the continuous (Brownian) part of returns, annualised. "
           "On its own it produces a flat implied-vol surface at exactly this level.")

HESTON_PARAMS = [
    ("kappa", "κ — mean-reversion speed", 0.05, 10.0, 2.5, 0.05, "%.2f", "raw",
     "How fast variance is pulled back towards θ. The half-life of a variance shock is "
     "ln 2 / κ years (κ = 2.5 → about 3.3 months). Low κ: today's vol persists and the "
     "term structure is long and sloped. High κ: vol snaps back quickly, so all but the "
     "shortest maturities look alike."),
    ("theta", "√θ — long-run vol", 5.0, 80.0, 20.0, 1.0, "%.0f%%", "vpct",
     "The level volatility mean-reverts to (θ is the long-run variance; the slider shows √θ as a vol). "
     "Long-dated ATM implied vol converges towards it."),
    ("v0", "√v₀ — initial vol", 5.0, 80.0, 20.0, 1.0, "%.0f%%", "vpct",
     "Today's instantaneous volatility (v₀ is today's variance). Very short-dated ATM vol ≈ √v₀; "
     "the ATM term structure bends from √v₀ towards √θ at speed κ."),
    ("rho", "ρ — spot/vol correlation", -0.99, 0.99, -0.7, 0.01, "%.2f", "raw",
     "Correlation between stock and variance shocks. ρ < 0 (typical for equities): volatility "
     "rises when the stock falls, which fattens the left tail and makes OTM puts dearer, "
     "giving a downward-sloping skew. ρ > 0 tilts the smile the other way."),
    ("xi", "σ — vol of vol", 0.05, 1.5, 0.4, 0.01, "%.2f", "raw",
     "Volatility of the variance process. Random volatility fattens both tails of the return "
     "distribution, so both wings lift: curvature, the 'smile'. It also scales how strongly ρ "
     "turns into skew. (Lecture notation: σ here is vol of vol, not the stock's vol.)"),
]

JUMP_NORMAL_PARAMS = [
    ("lam", "λ — jumps per year", 0.0, 5.0, 0.5, 0.05, "%.2f", "raw",
     "Poisson intensity: the expected number of jumps per year. λ = 0 switches jumps off."),
    ("mu_j", "μ_J — mean jump (log)", -40.0, 20.0, -10.0, 1.0, "%.0f%%", "pct",
     "Average size of a jump in log-return. Negative = crash-type jumps, which make OTM puts "
     "expensive (downward skew). Positive tilts the smile towards calls."),
    ("delta_j", "δ — jump size std", 1.0, 40.0, 10.0, 1.0, "%.0f%%", "pct",
     "Standard deviation of the log-jump size. Uncertainty about how big a jump is fattens "
     "both tails: curvature, even when μ_J = 0."),
]

KOU_PARAMS = [
    ("lam", "λ — jumps per year", 0.0, 5.0, 1.0, 0.05, "%.2f", "raw",
     "Poisson intensity: the expected number of jumps per year. λ = 0 switches jumps off."),
    ("p_up", "p — probability a jump is up", 0.0, 1.0, 0.3, 0.01, "%.2f", "raw",
     "Each jump is up with probability p and down with probability 1 − p. p < 0.5 means more "
     "down-jumps: downward skew."),
    ("eta1", "η₁ — up-jump decay", 3.0, 100.0, 25.0, 0.5, "%.1f", "raw",
     "Up-jump sizes are exponentially distributed with mean 1/η₁ (η₁ = 25 → +4% on average). "
     "Smaller η₁ = bigger up-jumps = a higher right wing. Must exceed 1 so that E[S] is finite."),
    ("eta2", "η₂ — down-jump decay", 2.0, 100.0, 10.0, 0.5, "%.1f", "raw",
     "Down-jump sizes are exponential with mean 1/η₂ (η₂ = 10 → −10% on average). "
     "Smaller η₂ = bigger crashes = a higher left wing."),
]

BATES_JUMPS = [
    ("lam", "λ — jumps per year", 0.0, 3.0, 0.3, 0.05, "%.2f", "raw",
     "Poisson intensity of the Merton-style jumps added on top of Heston. λ = 0 gives back Heston."),
    JUMP_NORMAL_PARAMS[1],
    JUMP_NORMAL_PARAMS[2],
]

MODELS = {
    "bs": dict(
        tab="Black-Scholes", eyebrow="BENCHMARK", title="Black-Scholes (1973): the flat world",
        intro="Log-returns are normal with one constant volatility. Every option, whatever its strike or "
              "maturity, therefore has the same implied volatility, and the surface is a flat plane. "
              "Real option markets never look like this. Use this tab as the reference for "
              "what the other models add.",
        latex=[r"dS_t = (r - q)\,S_t\,dt + \sigma\,S_t\,dW_t",
               r"\ln\frac{S_T}{F_T} \sim N\!\left(-\tfrac12\sigma^2 T,\ \sigma^2 T\right)"],
        params=[P_SIGMA],
        tryit=[
            "Move σ. The surface shifts up or down but stays perfectly flat. Which assumption about "
            "returns makes one number enough for every strike and maturity?",
            "Look at the density on log scale: a parabola, i.e. a normal distribution. Now open Merton or "
            "Kou and compare the tails. Which options does the difference make more expensive?",
            "Price a 3M option struck 10% below the forward in Black-Scholes using the 3M ATM vol of "
            "the Heston tab (ρ = −0.7). Is Black-Scholes too cheap or too expensive for that put?",
            "Why does a market maker who quotes every strike at the ATM vol lose money to anyone who knows "
            "the real distribution? (See the Market Making Simulator.)",
        ]),
    "heston": dict(
        tab="Heston", eyebrow="STOCHASTIC VOLATILITY", title="Heston (1993) stochastic volatility",
        intro="The variance vₜ follows a mean-reverting square-root process with its own volatility "
              "(σ, the vol of vol) and a correlation ρ with the stock. Random volatility gives fat tails "
              "(the smile), correlation gives asymmetry (the skew), and mean reversion shapes the term "
              "structure.",
        latex=[r"dS_t = (r-q)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW^{1}_t",
               r"dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW^{2}_t,\qquad d\langle W^1, W^2\rangle_t = \rho\,dt"],
        params=HESTON_PARAMS,
        tryit=[
            "Set ρ = 0: the skew almost vanishes and a symmetric smile is left. Now sweep ρ from −0.9 to "
            "+0.9. Which wing moves, and why does correlation create asymmetry while σ creates curvature?",
            "Make √v₀ much higher than √θ (a vol spike) with κ small. How is the ATM term structure "
            "sloped, and how long until it approaches √θ? Compare with the half-life ln 2 / κ.",
            "Raise κ. Short and long maturities converge. Why does fast mean reversion make long-dated "
            "options look more Black-Scholes-like?",
            "Increase σ (vol of vol) and watch the Feller indicator. What does it mean for the variance "
            "to hit zero, and does the surface change abruptly when the condition breaks?",
        ]),
    "merton": dict(
        tab="Merton", eyebrow="JUMP DIFFUSION", title="Merton (1976) jump diffusion",
        intro="Black-Scholes plus sudden jumps: a Poisson process with intensity λ triggers log-normal "
              "jumps of mean μ_J and standard deviation δ. Jumps create fat tails that diffusion alone "
              "can't, which is why the smile is steepest at short maturities and flattens as the "
              "central limit theorem takes over.",
        latex=[r"\frac{dS_t}{S_{t^-}} = (r - q - \lambda\bar k)\,dt + \sigma\,dW_t + (e^{J}-1)\,dN_t",
               r"J \sim N(\mu_J, \delta^2),\qquad N_t \sim \text{Poisson}(\lambda t),\qquad \bar k = e^{\mu_J + \delta^2/2} - 1"],
        params=[("sigma", "σ — diffusion vol", 5.0, 80.0, 15.0, 1.0, "%.0f%%", "pct", P_SIGMA[8])]
               + JUMP_NORMAL_PARAMS,
        tryit=[
            "Set λ = 0: the surface is flat at σ, because Merton nests Black-Scholes. Raise λ slowly. "
            "Which maturities react most, and why do jumps matter more for short-dated options?",
            "Flip μ_J from negative to positive. Explain the direction of the skew by what the OTM "
            "options are insuring against.",
            "Set μ_J = 0 and raise δ: a symmetric smile. Why does uncertainty about jump size alone "
            "create curvature?",
            "On the log-scale density, how much more likely is a −20% move than the lognormal says?",
        ]),
    "kou": dict(
        tab="Kou", eyebrow="JUMP DIFFUSION", title="Kou (2002) double-exponential jumps",
        intro="Like Merton, but jump sizes follow an asymmetric double-exponential law: up-jumps with "
              "probability p and mean 1/η₁, down-jumps with probability 1 − p and mean 1/η₂. "
              "Exponential tails are heavier than normal ones, and the two sides are set "
              "separately, so crash risk and rally risk each get their own parameters.",
        latex=[r"\frac{dS_t}{S_{t^-}} = (r - q - \lambda\bar k)\,dt + \sigma\,dW_t + (e^{J}-1)\,dN_t",
               r"f_J(y) = p\,\eta_1 e^{-\eta_1 y}\,\mathbf 1_{y\ge 0} + (1-p)\,\eta_2 e^{\eta_2 y}\,\mathbf 1_{y<0},\qquad \eta_1 > 1"],
        params=[("sigma", "σ — diffusion vol", 5.0, 80.0, 15.0, 1.0, "%.0f%%", "pct", P_SIGMA[8])] + KOU_PARAMS,
        tryit=[
            "Set p = 0.5 and η₁ = η₂: symmetric jumps, symmetric smile. Now lower p. What does p "
            "control that Merton's μ_J also controls?",
            "Keep p fixed and make η₂ small (big crashes) but η₁ large (tiny rallies). Merton can't "
            "set the two tails separately. Which part of the surface shows the difference?",
            "Push η₁ towards 3: average up-jump ≈ +33%. What happens to OTM calls? Where might a "
            "market price that (biotech, meme stocks, some commodities)?",
            "Why must η₁ exceed 1? Hint: compute E[e^J] for an exponential jump.",
        ]),
    "bates": dict(
        tab="Bates", eyebrow="STOCHASTIC VOL + JUMPS", title="Bates (1996): Heston with jumps",
        intro="Heston's stochastic volatility plus Merton's log-normal jumps. Jumps produce the steep "
              "short-dated skew that Heston struggles with, while stochastic volatility still controls the "
              "long end, so the two together fit real equity surfaces much better than either alone.",
        latex=[r"\frac{dS_t}{S_{t^-}} = (r - q - \lambda\bar k)\,dt + \sqrt{v_t}\,dW^1_t + (e^{J}-1)\,dN_t",
               r"dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW^2_t,\quad d\langle W^1,W^2\rangle_t=\rho\,dt,\quad J\sim N(\mu_J,\delta^2)"],
        params=HESTON_PARAMS + BATES_JUMPS,
        tryit=[
            "Set λ = 0: Bates is Heston. Add jumps back. Which part of the surface do jumps change "
            "most (short end), and which part does stochastic vol still control (long end)?",
            "Try to make a steep 1-week skew with Heston alone (λ = 0). How extreme do ρ and σ have to "
            "be? Now do it with jumps and moderate ρ, σ.",
            "Use μ_J for the short-dated skew and ρ for the long-dated skew. Does the skew term "
            "structure chart look more like a real equity index?",
            "Push the jump share of variance above 50%. How does the density change, and why are "
            "jump risks hard to delta-hedge?",
        ]),
}

MODEL_ORDER = ["bs", "heston", "merton", "kou", "bates"]

HELP = {
    "surface": "Each point is the Black-Scholes implied vol of a European option priced in the "
               "selected model. x: log-moneyness ln(K/F), where F is the forward. y: maturity (log scale). "
               "Gaps mean the option is so far out of the money that its price is below 1e-12 × spot, "
               "where no implied vol can be read reliably.",
    "bs_plane": "A flat plane at the model's 3-month ATM vol: what the surface would look like if "
                "Black-Scholes were true and calibrated to that single option.",
    "atm_1m": "Implied vol of the at-the-money-forward option (K = F) expiring in 1 month.",
    "atm_1y": "Implied vol of the at-the-money-forward option expiring in 1 year.",
    "skew": "3-month implied vol at ln(K/F) = −10% minus at +10%, in vol points. Positive = OTM puts "
            "cost more than OTM calls (equity-style skew). Zero under Black-Scholes.",
    "fly": "3-month 'butterfly': average of the ±10% wing vols minus the ATM vol, in vol points. "
           "Measures curvature, i.e. how fat both tails are. Zero under Black-Scholes.",
    "feller": "2κθ − σ². If ≥ 0 the variance process can never reach zero (Feller condition). "
              "Violations are common in calibrations to equity data; pricing still works.",
    "jump_share": "Share of the long-run annual return variance that comes from jumps rather than "
                  "diffusion. Jump risk can't be delta-hedged.",
    "smiles": "Implied vol across strikes for fixed maturities. Short maturities are light, long ones dark. "
              "The dashed line is Black-Scholes: flat.",
    "term": "At-the-money-forward implied vol by maturity. Under Black-Scholes it is a flat line at σ.",
    "skewterm": "How the skew (−10% vs +10% wing) and the curvature (butterfly) change with maturity. "
                "Jumps: large at short maturities, fading with T. Stochastic vol: builds up with T.",
    "density": "Risk-neutral density of the log-return ln(S_T / F) at the chosen maturity, from Fourier "
               "inversion of the characteristic function, against the normal density Black-Scholes "
               "assumes with the same ATM vol. Use log scale to see the tails: that's where the smile comes from.",
}

METHOD_NOTE = (
    "Every point is priced from the model's characteristic function with the Lewis (2001) single-integral "
    "formula, integrated by composite Gauss-Legendre quadrature. The integration range adapts to "
    "maturity and volatility, so prices are accurate to about 1e-12 across 1 week to 2 years. Implied vols are "
    "backed out of out-of-the-money prices by bisection. The test suite checks the pricer against "
    "closed-form Black-Scholes, the Merton series and Monte Carlo for Heston, Kou and Bates."
)
