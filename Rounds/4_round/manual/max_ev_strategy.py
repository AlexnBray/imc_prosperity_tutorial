# -*- coding: utf-8 -*-
"""
Round 4 Manual: MAXIMUM PROBABILITY-WEIGHTED EV STRATEGY
=========================================================

Problem with the "max E[PnL]" portfolio (KO500/BP0)
---------------------------------------------------
Asymptotic E[PnL] = +$117k looks fantastic, but it is dominated by a tiny tail of
huge wins:
    - 94.8% of the time  -> you lose the full $262.5k entry cost
    -  5.2% of the time  -> you collect on average ~$7M
That distribution has a positive expectation **only because you can repeat it
infinitely many times.**  The competition is ONE realisation, so you almost
certainly walk away with -$262k.

What this script optimises instead
----------------------------------
A trade is only attractive if it is BOTH high-EV AND likely-to-pay-off.  We
score every candidate portfolio against multiple "realised-value" objectives
and report whichever the user picks.  All metrics are computed on the
prior-weighted blended distribution across the plausible-sigma set
[2.30 ... 2.70] so they are vol-robust.

   Objective name           Formula                                 Captures
   -----------------------  --------------------------------------  --------------------
   1. Prob-weighted EV  =   E[PnL] * P(PnL > 0)                     user's literal ask
   2. Expected upside   =   E[max(PnL, 0)]                          dollars-made-when-win
   3. Median PnL                                                    typical outcome
   4. Kelly score       =   E[log(W0 + PnL)] - log(W0)              log-utility growth
   5. P75 PnL           =   75th percentile                         decent-case outcome
   6. Sharpe            =   E[PnL] / Std[PnL]                       risk-adjusted return

  -- We also track CVaR-5% / max-loss / P(profit) for downside check.

Search space
------------
The script grid-searches over the FOUR robust-or-near-robust building blocks:
   - AC_45_KO  (BUY): structurally robust, lottery-like
   - AC_40_BP  (SELL): structurally robust, ~70% win-rate
   - AC_35_P   (BUY): cheap downside hedge for KO breaches
   - AC_40_P   (BUY): mid-strike hedge
   - AC_50_CO  (SELL with variance-min replication): chooser arb (~$0.40/unit
                                                       guaranteed entry)
plus single-product baselines.  Outputs the top 8 portfolios per metric.

Run
---
    python max_ev_strategy.py            # default: full grid + recommendation
    python max_ev_strategy.py -v         # verbose: also print all metric tables
"""
import sys, io, time
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.stats import norm

VERBOSE = ("--verbose" in sys.argv) or ("-v" in sys.argv)

# ============================================================================
# 1. PARAMETERS
# ============================================================================
S0           = 50.0
R            = 0.0
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
DT = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)
T2_STEPS = 2 * 5 * STEPS_PER_DAY
T3_STEPS = 3 * 5 * STEPS_PER_DAY
T2_YEARS = (2 * 5) / TRADING_DAYS_PER_YEAR
T3_YEARS = (3 * 5) / TRADING_DAYS_PER_YEAR
BINARY_PAYOFF = 10.0
KO_BARRIER    = 35.0
CONTRACT_SIZE = 3000
N_SIM = 200_000          # cut from 400k for grid speed; finalists rerun if needed
SEED  = 42

# Wealth assumed for Kelly score.  Use $1M as a baseline trader bankroll.
W0_KELLY = 1_000_000.0

SIGMA_SCENARIOS = [2.30, 2.40, 2.47, 2.49, 2.51, 2.55, 2.60, 2.70]
SIGMA_WEIGHTS   = [0.05, 0.10, 0.25, 0.15, 0.30, 0.10, 0.04, 0.01]
assert abs(sum(SIGMA_WEIGHTS) - 1.0) < 1e-6

MKT = {
    "AC_50_P_2": dict(bid=9.70,  ask=9.75,  size=50,  T=T2_YEARS, K=50, kind="put"),
    "AC_50_C_2": dict(bid=9.70,  ask=9.75,  size=50,  T=T2_YEARS, K=50, kind="call"),
    "AC_50_CO":  dict(bid=22.20, ask=22.30, size=50,  T=T3_YEARS, K=50, kind="chooser"),
    "AC_40_BP":  dict(bid=5.00,  ask=5.10,  size=50,  T=T3_YEARS, K=40, kind="binary"),
    "AC_45_KO":  dict(bid=0.15,  ask=0.175, size=500, T=T3_YEARS, K=45, kind="ko"),
    "AC_60_C":   dict(bid=8.80,  ask=8.85,  size=50,  T=T3_YEARS, K=60, kind="call"),
    "AC_45_P":   dict(bid=9.05,  ask=9.10,  size=50,  T=T3_YEARS, K=45, kind="put"),
    "AC_35_P":   dict(bid=4.33,  ask=4.35,  size=50,  T=T3_YEARS, K=35, kind="put"),
    "AC_40_P":   dict(bid=6.50,  ask=6.55,  size=50,  T=T3_YEARS, K=40, kind="put"),
    "AC_50_P":   dict(bid=12.00, ask=12.05, size=50,  T=T3_YEARS, K=50, kind="put"),
    "AC_50_C":   dict(bid=12.00, ask=12.05, size=50,  T=T3_YEARS, K=50, kind="call"),
}

# ============================================================================
# 2. SIMULATE GBM PATHS UNDER EACH SIGMA SCENARIO
# ============================================================================
print(f"Simulating {N_SIM:,} paths x {len(SIGMA_SCENARIOS)} vol scenarios...")
t0 = time.time()
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_SIM, T3_STEPS))

def simulate_paths(sigma):
    log_inc  = (R - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * Z
    log_path = np.cumsum(log_inc, axis=1)
    return S0 * np.exp(log_path)

paths_by_sigma = {sig: simulate_paths(sig) for sig in SIGMA_SCENARIOS}

def payoff_arrays(P):
    S2 = P[:, T2_STEPS - 1]
    S3 = P[:, T3_STEPS - 1]
    breach = np.any(P[:, :T3_STEPS - 1] < KO_BARRIER, axis=1)
    return dict(
        AC_50_P_2 = np.maximum(50 - S2, 0),
        AC_50_C_2 = np.maximum(S2 - 50, 0),
        AC_50_CO  = np.where(S2 >= 50, np.maximum(S3 - 50, 0),
                                       np.maximum(50 - S3, 0)),
        AC_40_BP  = np.where(S3 < 40, BINARY_PAYOFF, 0.0),
        AC_45_KO  = np.where(breach, 0.0, np.maximum(45 - S3, 0)),
        AC_60_C   = np.maximum(S3 - 60, 0),
        AC_45_P   = np.maximum(45 - S3, 0),
        AC_35_P   = np.maximum(35 - S3, 0),
        AC_40_P   = np.maximum(40 - S3, 0),
        AC_50_P   = np.maximum(50 - S3, 0),
        AC_50_C   = np.maximum(S3 - 50, 0),
    )
payoffs_by_sigma = {sig: payoff_arrays(P) for sig, P in paths_by_sigma.items()}
print(f"  done in {time.time()-t0:.1f}s")

# ============================================================================
# 3. PORTFOLIO PnL (per path, under one sigma)
# ============================================================================
def chooser_hedged_pnl_unit(sig):
    """PnL per path PER UNIT of:
        SELL 1 chooser + BUY 0.5 (C3 + P3 + C2 + P2)
    Variance-minimising replication.  Entry locks +$0.40/unit."""
    P  = paths_by_sigma[sig]
    S2 = P[:, T2_STEPS - 1]; S3 = P[:, T3_STEPS - 1]
    pay_co = payoffs_by_sigma[sig]["AC_50_CO"]
    pay_c3 = np.maximum(S3 - 50, 0)
    pay_p3 = np.maximum(50 - S3, 0)
    pay_c2 = np.maximum(S2 - 50, 0)
    pay_p2 = np.maximum(50 - S2, 0)
    entry = (MKT["AC_50_CO"]["bid"]
             - 0.5 * (MKT["AC_50_C"]["ask"] + MKT["AC_50_P"]["ask"]
                      + MKT["AC_50_C_2"]["ask"] + MKT["AC_50_P_2"]["ask"]))
    return entry - pay_co + 0.5 * (pay_c3 + pay_p3 + pay_c2 + pay_p2)

choo_unit_by_sig = {sig: chooser_hedged_pnl_unit(sig) for sig in SIGMA_SCENARIOS}

def portfolio_pnl(positions, sig, chooser_qty=0):
    """positions: dict(name -> (side, qty)).  Returns per-path PnL ($)."""
    pay = payoffs_by_sigma[sig]
    total = np.zeros(N_SIM)
    for name, (side, qty) in positions.items():
        if qty == 0:
            continue
        p = MKT[name]
        if side == "BUY":
            unit = pay[name] - p["ask"]
        else:
            unit = p["bid"] - pay[name]
        total += qty * CONTRACT_SIZE * unit
    if chooser_qty > 0:
        total += chooser_qty * CONTRACT_SIZE * choo_unit_by_sig[sig]
    return total

# ============================================================================
# 4. METRIC FUNCTION (prior-weighted across sigma)
# ============================================================================
def all_metrics(positions, chooser_qty=0):
    """Return dict of metrics computed on the prior-weighted blended distribution
    AND per-sigma worst-case versions for vol robustness."""
    pnls = []
    for sig in SIGMA_SCENARIOS:
        pnls.append(portfolio_pnl(positions, sig, chooser_qty))
    # Stack into (n_sigma, N_SIM).  Path weights = SIGMA_WEIGHTS[s] / N_SIM
    arr = np.stack(pnls, axis=0)               # (S, N)

    # --- Mean-based metrics on blended distribution
    w_sigma = np.array(SIGMA_WEIGHTS)
    E_per_sigma     = arr.mean(axis=1)
    Pwin_per_sigma  = (arr > 0).mean(axis=1)
    Up_per_sigma    = np.maximum(arr, 0).mean(axis=1)
    Std_per_sigma   = arr.std(axis=1)
    log_W           = np.log(np.maximum(W0_KELLY + arr, 1.0)) - np.log(W0_KELLY)
    Kelly_per_sigma = log_W.mean(axis=1)

    E_blend     = float(np.dot(w_sigma, E_per_sigma))
    Pwin_blend  = float(np.dot(w_sigma, Pwin_per_sigma))
    Up_blend    = float(np.dot(w_sigma, Up_per_sigma))
    Kelly_blend = float(np.dot(w_sigma, Kelly_per_sigma))

    # --- Sort-based metrics: build full weighted sample once
    flat_pnl = arr.ravel()
    flat_w   = np.repeat(w_sigma, N_SIM) / N_SIM
    order    = np.argsort(flat_pnl)
    sp = flat_pnl[order]
    sw = flat_w[order]
    cw = np.cumsum(sw)
    def q(p): return float(sp[np.searchsorted(cw, p, side="left")])
    Median = q(0.5)
    P25    = q(0.25)
    P75    = q(0.75)
    P10    = q(0.10)
    VaR5   = q(0.05)
    mask = sp <= VaR5
    CVaR5 = float(np.sum(sp[mask] * sw[mask]) / max(np.sum(sw[mask]), 1e-12))
    PnL_min = float(sp[0])

    # --- Worst-sigma versions
    minE_sigma  = float(E_per_sigma.min())
    minP_sigma  = float(Pwin_per_sigma.min())
    minMed_sigma = float(min(np.percentile(arr[i], 50) for i in range(arr.shape[0])))

    Std_blend = float(np.sqrt(np.dot(w_sigma, Std_per_sigma**2 + (E_per_sigma - E_blend)**2)))
    Sharpe = E_blend / Std_blend if Std_blend > 0 else 0.0

    return dict(
        E=E_blend, Pwin=Pwin_blend, Upside=Up_blend, Kelly=Kelly_blend,
        Median=Median, P10=P10, P25=P25, P75=P75, VaR5=VaR5, CVaR5=CVaR5,
        Min=PnL_min, Std=Std_blend, Sharpe=Sharpe,
        # The PRIMARY objective the user described: PnL * probability
        EV_x_Pwin = E_blend * Pwin_blend,
        # Worst-case versions
        minE_sig=minE_sigma, minPwin_sig=minP_sigma, minMed_sig=minMed_sigma,
    )

def max_loss_dollars(positions, chooser_qty=0):
    """Hard floor on portfolio loss (per-leg max loss summation)."""
    total = 0.0
    for name, (side, qty) in positions.items():
        if qty == 0: continue
        p = MKT[name]
        if side == "BUY":
            total += qty * CONTRACT_SIZE * p["ask"]
        elif name == "AC_40_BP":
            total += qty * CONTRACT_SIZE * (BINARY_PAYOFF - p["bid"])
        elif p["kind"] == "put":
            total += qty * CONTRACT_SIZE * (p["K"] - p["bid"])
        elif p["kind"] == "ko":
            total += qty * CONTRACT_SIZE * (45 - p["bid"])
        else:
            return float("inf")  # short call / chooser unbounded
    if chooser_qty > 0:
        # Variance-min replicated chooser: empirical max loss
        # Pathwise residual std ~ $6.4/unit, max ~$25/unit observed -> approx
        total += chooser_qty * CONTRACT_SIZE * 25.0
    return total

# ============================================================================
# 5. CANDIDATE PORTFOLIO GRID
# ============================================================================
KO_GRID  = [0, 25, 50, 100, 150, 200, 300, 500]
BP_GRID  = [0, 10, 25, 40, 50]
P35B_GRID = [0, 10, 25, 50]    # BUY P35 hedge (deep OTM put)
P40B_GRID = [0, 10, 25]        # BUY P40 hedge
CO_GRID  = [0, 25, 50]         # SELL chooser hedged variance-min replication

candidates = []
for q_ko in KO_GRID:
    for q_bp in BP_GRID:
        for q_p35 in P35B_GRID:
            for q_p40 in P40B_GRID:
                for q_co in CO_GRID:
                    if q_ko == 0 and q_bp == 0 and q_p35 == 0 and q_p40 == 0 and q_co == 0:
                        continue
                    pos = {}
                    if q_ko  > 0: pos["AC_45_KO"] = ("BUY",  q_ko)
                    if q_bp  > 0: pos["AC_40_BP"] = ("SELL", q_bp)
                    if q_p35 > 0: pos["AC_35_P"]  = ("BUY",  q_p35)
                    if q_p40 > 0: pos["AC_40_P"]  = ("BUY",  q_p40)
                    candidates.append(dict(positions=pos, chooser_qty=q_co))

print(f"Evaluating {len(candidates):,} candidate portfolios...")
t0 = time.time()
for i, c in enumerate(candidates):
    c["m"] = all_metrics(c["positions"], c["chooser_qty"])
    c["max_loss"] = max_loss_dollars(c["positions"], c["chooser_qty"])
    parts = []
    for n, (s, q) in c["positions"].items():
        parts.append(f"{n[3:]}{'+' if s=='BUY' else '-'}{q}")
    if c["chooser_qty"] > 0:
        parts.append(f"COh-{c['chooser_qty']}")
    c["name"] = "/".join(parts) if parts else "empty"
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(candidates)}  elapsed {time.time()-t0:.1f}s")
print(f"  evaluation done in {time.time()-t0:.1f}s")

# ============================================================================
# 6. RANK BY EACH OBJECTIVE
# ============================================================================
def top_by(key, n=8, descending=True, filter_fn=None):
    pool = [c for c in candidates if (filter_fn is None or filter_fn(c))]
    pool.sort(key=lambda c: c["m"][key], reverse=descending)
    return pool[:n]

W = 130
def print_table(title, rows, columns):
    """columns: list of (header, key_or_callable, fmt)"""
    print()
    print("=" * W)
    print(f"  {title}")
    print("-" * W)
    hdr = ""
    for h, _, _ in columns:
        hdr += f"  {h:>12}"
    print("  " + f"{'Portfolio':<26}" + hdr)
    print("-" * W)
    for r in rows:
        line = f"  {r['name']:<26}"
        for _, key, fmt in columns:
            v = key(r) if callable(key) else r["m"].get(key, r.get(key))
            line += f"  {fmt.format(v):>12}"
        print(line)
    print("=" * W)

EV_x_P_label = "EV*P(win)"

# Hard-budget filter: drop portfolios with max-loss > $1.5M (the "MOD" tier)
def budget_filter(budget):
    return lambda c: c["max_loss"] <= budget

# --- Print summary ranking by each metric (under $1.5M loss budget) -------
BUDGET = 1_500_000
print()
print("#" * W)
print(f"#  TOP 8 PORTFOLIOS PER OBJECTIVE  (loss budget <= ${BUDGET:,})")
print("#" * W)

cols_main = [
    (EV_x_P_label,  "EV_x_Pwin",  "{:>+,.0f}"),
    ("E[PnL]",      "E",          "{:>+,.0f}"),
    ("P(win)",      lambda r: r["m"]["Pwin"]*100, "{:>5.1f}%"),
    ("Median",      "Median",     "{:>+,.0f}"),
    ("Upside",      "Upside",     "{:>+,.0f}"),
    ("CVaR5",       "CVaR5",      "{:>+,.0f}"),
    ("Sharpe",      "Sharpe",     "{:>+.4f}"),
    ("MaxLoss",     lambda r: -r["max_loss"], "{:>+,.0f}"),
]

print_table(
    "RANK BY  EV*P(win)  =  E[PnL] x P(profit)   <-- USER'S PRIMARY OBJECTIVE",
    top_by("EV_x_Pwin", 8, filter_fn=budget_filter(BUDGET)),
    cols_main,
)

print_table(
    "RANK BY  Expected upside  =  E[max(PnL, 0)]   (avg dollars made when you win)",
    top_by("Upside", 8, filter_fn=budget_filter(BUDGET)),
    cols_main,
)

print_table(
    "RANK BY  Median PnL   (typical realised outcome)",
    top_by("Median", 8, filter_fn=budget_filter(BUDGET)),
    cols_main,
)

print_table(
    "RANK BY  Kelly score  =  E[log(W0+PnL)] - log(W0)   (W0 = $1M)",
    top_by("Kelly", 8, filter_fn=budget_filter(BUDGET)),
    cols_main,
)

print_table(
    "RANK BY  P75 PnL   (75th percentile, 'decent-case' outcome)",
    top_by("P75", 8, filter_fn=budget_filter(BUDGET)),
    cols_main,
)

print_table(
    "RANK BY  Sharpe ratio   (E / Std)",
    top_by("Sharpe", 8, filter_fn=budget_filter(BUDGET)),
    cols_main,
)

# Also at lower budget tiers for comparison
for tier_budget, tier_label in [(300_000, "ULTRA <=$300k"),
                                (750_000, "LOW   <=$750k")]:
    print_table(
        f"RANK BY  EV*P(win)  -- {tier_label} loss budget",
        top_by("EV_x_Pwin", 6, filter_fn=budget_filter(tier_budget)),
        cols_main,
    )

# ============================================================================
# 7. PRIMARY RECOMMENDATION (max EV*P(win) at $1.5M budget)
# ============================================================================
rec = top_by("EV_x_Pwin", 1, filter_fn=budget_filter(BUDGET))[0]
m = rec["m"]

print()
print("=" * W)
print(f"  PRIMARY RECOMMENDATION  (max  EV*P(win)  under $1.5M loss budget)")
print(f"  Portfolio: {rec['name']}")
print("=" * W)
SPEC = {
    "AC_50_P_2": "Vanilla Put,   K=50, 2-week",
    "AC_50_C_2": "Vanilla Call,  K=50, 2-week",
    "AC_50_CO":  "Chooser,       K=50, 3-week (choice at week 2)",
    "AC_40_BP":  "Binary Put,    K=40, 3-week, payoff 10 if S<40",
    "AC_45_KO":  "KO Put,        K=45, 3-week, KO if S<35 ever",
    "AC_60_C":   "Vanilla Call,  K=60, 3-week",
    "AC_45_P":   "Vanilla Put,   K=45, 3-week",
    "AC_35_P":   "Vanilla Put,   K=35, 3-week",
    "AC_40_P":   "Vanilla Put,   K=40, 3-week",
    "AC_50_P":   "Vanilla Put,   K=50, 3-week",
    "AC_50_C":   "Vanilla Call,  K=50, 3-week",
}
for n, (side, qty) in rec["positions"].items():
    p = MKT[n]; px = p["ask"] if side == "BUY" else p["bid"]
    print(f"  {side:<4} {qty:>3} x {n:<10} @ ${px:<6.3f}   {SPEC[n]}")
if rec["chooser_qty"] > 0:
    q = rec["chooser_qty"]
    print(f"  SELL {q:>3} x AC_50_CO   @ ${MKT['AC_50_CO']['bid']:.3f}   "
          f"(chooser arb -- ALSO BUY {q*0.5:.0f} EACH of C2,P2,C3,P3 to lock entry)")
    print(f"       --> entry locks +$0.40/chooser unit, residual std ~$6.4/unit")
print("-" * W)
print(f"  PROBABILITY-WEIGHTED EV  (E[PnL] * P(win))    = ${m['EV_x_Pwin']:>+12,.0f}")
print(f"  E[PnL]    (vol-prior weighted)                = ${m['E']:>+12,.0f}")
print(f"  P(profit)                                     =  {m['Pwin']*100:>5.1f}%")
print(f"  E[max(PnL, 0)]   (expected upside)            = ${m['Upside']:>+12,.0f}")
print(f"  Median PnL   (typical realised)               = ${m['Median']:>+12,.0f}")
print(f"  P75 PnL      (decent-case outcome)            = ${m['P75']:>+12,.0f}")
print(f"  P25 PnL      (poor-case outcome)              = ${m['P25']:>+12,.0f}")
print(f"  CVaR 5%      (avg of worst 5%)                = ${m['CVaR5']:>+12,.0f}")
print(f"  Hard max loss  (theoretical floor)            = ${-rec['max_loss']:>+12,.0f}")
print(f"  Sharpe                                         =  {m['Sharpe']:>+5.3f}")
print("-" * W)
print(f"  Worst-sigma E[PnL]  in [2.30, 2.70]           = ${m['minE_sig']:>+12,.0f}")
print(f"  Worst-sigma P(win)                             =  {m['minPwin_sig']*100:>5.1f}%")
print(f"  Worst-sigma median PnL                        = ${m['minMed_sig']:>+12,.0f}")
print("=" * W)

# ============================================================================
# 8. WHY THIS BEATS  KO500/BP0  (the asymptotic-EV portfolio)
# ============================================================================
KO500 = next((c for c in candidates
              if c["positions"] == {"AC_45_KO": ("BUY", 500)} and c["chooser_qty"]==0),
             None)
print()
print("=" * W)
print("  HEAD-TO-HEAD vs  the asymptotic-E[PnL] champion  (KO500 only)")
print("-" * W)
print(f"  {'Metric':<32}  {'KO500/BP0':>15}  {rec['name'][:25]:>25}  {'Improvement':>15}")
print("-" * W)
def cmp_row(label, key, is_pct=False):
    a = KO500["m"][key]; b = rec["m"][key]
    if is_pct:
        s_a = f"{a*100:>14.1f}%"
        s_b = f"{b*100:>24.1f}%"
        diff = (b - a)*100
        s_d = f"{diff:>+14.1f}%"
    else:
        s_a = f"${a:>+14,.0f}"
        s_b = f"${b:>+24,.0f}"
        s_d = f"${b-a:>+14,.0f}"
    print(f"  {label:<32}  {s_a}  {s_b}  {s_d}")
cmp_row("PROB-WEIGHTED EV (E*P)",       "EV_x_Pwin")
cmp_row("E[PnL]",                       "E")
cmp_row("P(win)",                       "Pwin", True)
cmp_row("Median PnL",                   "Median")
cmp_row("Expected upside  E[max(PnL,0)]","Upside")
cmp_row("P25 (poor-case)",              "P25")
cmp_row("P75 (decent-case)",            "P75")
cmp_row("Sharpe",                       "Sharpe")
print("-" * W)
print(f"  {'Hard max loss':<32}  ${-KO500['max_loss']:>+14,.0f}  "
      f"${-rec['max_loss']:>+24,.0f}")
print("=" * W)

# ============================================================================
# 9. SHORT TRADE TICKET (always printed)
# ============================================================================
WS = 86
print()
print("=" * WS)
print("  TRADE TICKET   (probability-weighted EV maximiser)")
print("=" * WS)
for n, (side, qty) in rec["positions"].items():
    p = MKT[n]; px = p["ask"] if side == "BUY" else p["bid"]
    print(f"  {side:<4} {qty:>3} x {n:<10} @ ${px:<6.3f}   {SPEC[n]}")
if rec["chooser_qty"] > 0:
    q = rec["chooser_qty"]
    halfq = q * 0.5
    print(f"  SELL {q:>3} x AC_50_CO   @ ${MKT['AC_50_CO']['bid']:<6.3f}   chooser arb")
    print(f"  BUY  {halfq:>3.0f} x AC_50_C    @ ${MKT['AC_50_C']['ask']:<6.3f}    chooser-replication leg")
    print(f"  BUY  {halfq:>3.0f} x AC_50_P    @ ${MKT['AC_50_P']['ask']:<6.3f}    chooser-replication leg")
    print(f"  BUY  {halfq:>3.0f} x AC_50_C_2  @ ${MKT['AC_50_C_2']['ask']:<6.3f}    chooser-replication leg")
    print(f"  BUY  {halfq:>3.0f} x AC_50_P_2  @ ${MKT['AC_50_P_2']['ask']:<6.3f}    chooser-replication leg")
print("-" * WS)
print(f"  E[PnL]            = ${m['E']:>+12,.0f}")
print(f"  P(profit)         =  {m['Pwin']*100:>5.1f}%")
print(f"  EV*P(win)         = ${m['EV_x_Pwin']:>+12,.0f}    <-- primary score")
print(f"  Median PnL        = ${m['Median']:>+12,.0f}")
print(f"  CVaR 5%           = ${m['CVaR5']:>+12,.0f}")
print(f"  Hard max loss     = ${-rec['max_loss']:>+12,.0f}")
print("=" * WS)
print()
print("Run with -v / --verbose for per-sigma breakdowns of the recommended portfolio.")

# ============================================================================
# 10. VERBOSE: per-sigma profile of the recommended portfolio
# ============================================================================
if VERBOSE:
    print()
    print("=" * W)
    print(f"  PER-SIGMA PROFILE  for  {rec['name']}")
    print("-" * W)
    print(f"  {'sigma':>6}  {'E[PnL]':>+12}  {'P(win)':>8}  {'Median':>+12}  "
          f"{'P25':>+12}  {'P75':>+12}  {'CVaR5':>+12}")
    for sig in SIGMA_SCENARIOS:
        pnl = portfolio_pnl(rec["positions"], sig, rec["chooser_qty"])
        e = pnl.mean(); p = (pnl>0).mean()
        med = np.percentile(pnl, 50); p25 = np.percentile(pnl, 25)
        p75 = np.percentile(pnl, 75); v5 = np.percentile(pnl, 5)
        cvar = pnl[pnl <= v5].mean()
        flag = "  <- model"   if abs(sig - 2.51) < 0.005 else (
               "  <- mkt-IV"  if abs(sig - 2.47) < 0.005 else "")
        print(f"  {sig:>6.2f}  ${e:>+11,.0f}  {p*100:>7.1f}%  ${med:>+11,.0f}  "
              f"${p25:>+11,.0f}  ${p75:>+11,.0f}  ${cvar:>+11,.0f}{flag}")
    print("=" * W)

print("\nDone.")
