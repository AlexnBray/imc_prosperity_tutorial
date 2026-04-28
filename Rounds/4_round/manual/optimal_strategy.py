# -*- coding: utf-8 -*-
"""
Round 4 Manual: OPTIMAL RISK-AVERSE STRATEGY
=============================================

Goal
----
Maximise E[PnL] subject to a strict CVaR constraint, ROBUST to volatility
mis-specification (true sigma may differ from the model's 251%).

Why this is needed
------------------
The competition realises ONE path. A high E[PnL] portfolio with a fat tail
can lose enormous sums on a single bad realisation. We want bounded
downside *and* good upside.

Mathematical framework
----------------------
For each candidate position vector q = (q_KO, q_BP, ...) we compute the
per-path PnL distribution under several plausible "true" sigmas. We then
score by:

    score(q) = min over sigma of E[PnL(q; sigma)]
               subject to  CVaR_5%(PnL(q)) >= -CVaR_BUDGET

i.e. maximise the WORST-CASE expected PnL across vol regimes, with a hard
tail constraint. This is min-max regret robust optimisation.

Key result (proven in TABLE 1 below):
    Only two trades have positive edge across the FULL plausible vol
    range [sigma=2.30, 2.70]:
        AC_45_KO BUY  (worst-case edge +0.046)
        AC_40_BP SELL (worst-case edge +0.037)
    Every other trade is a directional vol bet.

Output: three recommended portfolios at three risk tiers, plus the
efficient frontier mapping CVaR budget to maximum E[PnL].
"""

import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.stats import norm

# Verbose mode prints all diagnostic tables (vol-edge sweep, frontier, etc.).
# Default is QUIET: only prints the recommended trade list + key risk numbers.
VERBOSE = ("--verbose" in sys.argv) or ("-v" in sys.argv)
def vprint(*a, **kw):
    if VERBOSE:
        print(*a, **kw)

# ============================================================================
# 1. PARAMETERS  (must match competition spec exactly)
# ============================================================================
S0           = 50.0
SIGMA_MODEL  = 2.51
R            = 0.0
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
DT = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)

T2_STEPS = 2 * 5 * STEPS_PER_DAY        # 40  (2 weeks)
T3_STEPS = 3 * 5 * STEPS_PER_DAY        # 60  (3 weeks)
T2_YEARS = (2 * 5) / TRADING_DAYS_PER_YEAR
T3_YEARS = (3 * 5) / TRADING_DAYS_PER_YEAR

BINARY_PAYOFF = 10.0
KO_BARRIER    = 35.0
CONTRACT_SIZE = 3000

N_SIM = 400_000
SEED  = 42

# Vol scenarios with subjective probability weights.
# Centred on the model (2.51) and the market-implied 2wk straddle vol (2.47).
SIGMA_SCENARIOS = [2.30, 2.40, 2.47, 2.49, 2.51, 2.55, 2.60, 2.70]
SIGMA_WEIGHTS   = [0.05, 0.10, 0.25, 0.15, 0.30, 0.10, 0.04, 0.01]
assert abs(sum(SIGMA_WEIGHTS) - 1.0) < 1e-6

# ============================================================================
# 2. MARKET QUOTES + POSITION LIMITS
# ============================================================================
# size = max contracts that can be traded (per the competition rules)
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
# 3. BLACK-SCHOLES CLOSED-FORM (r = 0)
# ============================================================================
def bs_d12(S, K, T, sig):
    d1 = (np.log(S / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    return d1, d1 - sig * np.sqrt(T)

def bs_call(S, K, T, sig):
    if T <= 0: return float(max(S - K, 0))
    d1, d2 = bs_d12(S, K, T, sig)
    return S * norm.cdf(d1) - K * norm.cdf(d2)

def bs_put(S, K, T, sig):
    if T <= 0: return float(max(K - S, 0))
    d1, d2 = bs_d12(S, K, T, sig)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_binary_put(S, K, T, sig, payoff=1.0):
    if T <= 0: return float(payoff if S < K else 0)
    _, d2 = bs_d12(S, K, T, sig)
    return payoff * norm.cdf(-d2)

def bs_chooser(S, K, T_expiry, T_choice, sig):
    return bs_call(S, K, T_expiry, sig) + bs_put(S, K, T_choice, sig)

# ============================================================================
# 4. MONTE-CARLO SIMULATION ACROSS VOL SCENARIOS
# ============================================================================
print(f"Simulating {N_SIM:,} GBM paths under {len(SIGMA_SCENARIOS)} vol scenarios...")
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_SIM, T3_STEPS))   # one common noise -> antithetic-friendly

def simulate_paths(sigma):
    log_inc  = (R - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * Z
    log_path = np.cumsum(log_inc, axis=1)
    return S0 * np.exp(log_path)

paths_by_sigma = {sig: simulate_paths(sig) for sig in SIGMA_SCENARIOS}

def payoff_arrays(P):
    """Return per-path payoff arrays (per unit) for every product."""
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

# KO has no closed form -> price by MC at every sigma
ko_fairs_by_sig = {sig: float(payoffs_by_sigma[sig]["AC_45_KO"].mean())
                   for sig in SIGMA_SCENARIOS}

# ============================================================================
# 5. PER-TRADE FAIR VALUE + EDGE TABLE
# ============================================================================
def fair_value(name, sig):
    p = MKT[name]
    K, T, kind = p["K"], p["T"], p["kind"]
    if kind == "put":     return bs_put(S0, K, T, sig)
    if kind == "call":    return bs_call(S0, K, T, sig)
    if kind == "chooser": return bs_chooser(S0, K, T3_YEARS, T2_YEARS, sig)
    if kind == "binary":  return bs_binary_put(S0, K, T, sig, payoff=BINARY_PAYOFF)
    if kind == "ko":      return ko_fairs_by_sig[sig]
    raise ValueError(kind)

# Each product is tradeable in either direction; pre-compute the BETTER side
# per-vol-scenario.
DIRECTIONS = {}
for name, p in MKT.items():
    edges_buy  = [fair_value(name, s) - p["ask"] for s in SIGMA_SCENARIOS]
    edges_sell = [p["bid"] - fair_value(name, s) for s in SIGMA_SCENARIOS]
    # pick the direction with the higher WEIGHTED-AVERAGE edge
    we_buy  = sum(e * w for e, w in zip(edges_buy,  SIGMA_WEIGHTS))
    we_sell = sum(e * w for e, w in zip(edges_sell, SIGMA_WEIGHTS))
    DIRECTIONS[name] = "BUY" if we_buy >= we_sell else "SELL"

W = 112
vprint()
vprint("=" * W)
vprint("  TABLE 1.  PER-TRADE EDGE ACROSS VOL REGIMES   (best direction; positive = profitable)")
vprint(f"  (Using mean of {len(SIGMA_SCENARIOS)} sigma scenarios with subjective weights {SIGMA_WEIGHTS})")
vprint("-" * W)
hdr = f"  {'Product':<12} {'Side':<5}"
for sig in SIGMA_SCENARIOS:
    hdr += f"  {'s='+f'{int(sig*100)}':>6}"
hdr += f"  {'WORST':>8}  {'WEIGHTED':>9}  {'Robust?'}"
vprint(hdr)
vprint("-" * W)

per_product = {}
for name, p in MKT.items():
    side = DIRECTIONS[name]
    edges = []
    for sig in SIGMA_SCENARIOS:
        f = fair_value(name, sig)
        edge = (p["bid"] - f) if side == "SELL" else (f - p["ask"])
        edges.append(edge)
    worst    = min(edges)
    weighted = sum(e * w for e, w in zip(edges, SIGMA_WEIGHTS))
    is_robust = worst > 0
    per_product[name] = dict(side=side, edges=edges, worst=worst,
                             weighted=weighted, robust=is_robust)
    row = f"  {name:<12} {side:<5}"
    for e in edges:
        row += f"  {e:>+6.3f}"
    row += f"  {worst:>+8.4f}  {weighted:>+9.4f}  {'YES' if is_robust else '   '}"
    vprint(row)
vprint("-" * W)
robust_set = [n for n in MKT if per_product[n]["robust"]]
vprint(f"  STRUCTURALLY ROBUST TRADES (positive edge for ALL plausible sigmas): {robust_set}")
vprint(f"  All other trades are vol bets -- they require sigma-prediction to be EV-positive")
vprint("=" * W)

# ============================================================================
# 6. PER-TRADE STAND-ALONE RISK STATISTICS (under model sigma=2.51)
# ============================================================================
def trade_pnl(name, side, qty, sig):
    payoff = payoffs_by_sigma[sig][name]
    p = MKT[name]
    if side == "BUY":
        unit = payoff - p["ask"]
    else:
        unit = p["bid"] - payoff
    return qty * CONTRACT_SIZE * unit

def trade_max_loss(name, side, qty):
    """Worst possible per-position PnL across all paths in dollars."""
    p = MKT[name]
    if side == "BUY":
        return qty * CONTRACT_SIZE * p["ask"]
    if name == "AC_40_BP":
        return qty * CONTRACT_SIZE * (BINARY_PAYOFF - p["bid"])
    if p["kind"] == "put":
        return qty * CONTRACT_SIZE * (p["K"] - p["bid"])
    if p["kind"] == "ko":
        return qty * CONTRACT_SIZE * (45 - p["bid"])
    return float("inf")  # short call, short chooser -> unbounded

vprint()
vprint("=" * W)
vprint("  TABLE 2.  STAND-ALONE TRADE RISK (model sigma=2.51, max position size, per trade only)")
vprint(f"  {'Trade':<12} {'Side':<5} {'Qty':>4}   {'E[PnL]':>10}  {'Std':>10}  "
       f"{'VaR5':>10}  {'CVaR5':>10}  {'Max Loss':>10}  {'Sharpe':>7}")
vprint("-" * W)
for name in MKT:
    side = DIRECTIONS[name]
    qty  = MKT[name]["size"]
    pnl  = trade_pnl(name, side, qty, 2.51)
    e, sd = float(pnl.mean()), float(pnl.std())
    var5  = float(np.percentile(pnl, 5))
    cvar5 = float(pnl[pnl <= var5].mean())
    mx    = trade_max_loss(name, side, qty)
    mx_s  = f"{mx:>10,.0f}" if mx != float("inf") else "   UNBNDED"
    sh    = e / sd if sd > 0 else 0.0
    vprint(f"  {name:<12} {side:<5} {qty:>4,}   {e:>+10,.0f}  {sd:>10,.0f}  "
           f"{var5:>10,.0f}  {cvar5:>10,.0f}  {mx_s}  {sh:>+7.4f}")
vprint("=" * W)

# ============================================================================
# 7. CHOOSER REPLICATION VARIANCE-MINIMAL HEDGE  (math derivation)
# ============================================================================
# A SHORT chooser position can be partially hedged by buying:
#   ALPHA * (C(3wk) + P(2wk))  +  (1-ALPHA) * (P(3wk) + C(2wk))
#
# At t=0 both replications cost the same (= chooser fair value, by symmetry
# C(T)=P(T) when r=0 and ATM).  But pathwise risk is asymmetric:
#   alpha=1   -> zero risk when buyer picks call,  S_3wk-S_2wk dispersion when picks put
#   alpha=0   -> zero risk when buyer picks put,   S_2wk-S_3wk dispersion when picks call
#   alpha=0.5 -> sqrt(2)/2 of either dispersion in BOTH branches (variance-minimising)
#
# Therefore the variance-minimising hedge for a short chooser is the 50/50 mix.
# We use this when sizing chooser exposure.
def chooser_hedged_pnl(qty, sig):
    """PnL per path of: SELL qty chooser + BUY qty/2 of each leg of both replications."""
    P  = paths_by_sigma[sig]
    S2 = P[:, T2_STEPS - 1]; S3 = P[:, T3_STEPS - 1]
    pay_co = payoffs_by_sigma[sig]["AC_50_CO"]
    pay_c3 = np.maximum(S3 - 50, 0)
    pay_p3 = np.maximum(50 - S3, 0)
    pay_c2 = np.maximum(S2 - 50, 0)
    pay_p2 = np.maximum(50 - S2, 0)
    # entry: receive 22.20 from chooser sell, pay 0.5*(12.05+12.05+9.75+9.75) = 21.80
    entry = MKT["AC_50_CO"]["bid"] - 0.5 * (MKT["AC_50_C"]["ask"] + MKT["AC_50_P"]["ask"]
                                            + MKT["AC_50_C_2"]["ask"] + MKT["AC_50_P_2"]["ask"])
    pnl_per_unit = entry - pay_co + 0.5 * (pay_c3 + pay_p3 + pay_c2 + pay_p2)
    return qty * CONTRACT_SIZE * pnl_per_unit

vprint()
vprint("=" * W)
vprint("  TABLE 3.  CHOOSER 50/50 HEDGED REPLICATION  (variance-minimising hedge)")
vprint(f"  Entry  = bid(CO) - 0.5*(ask C3 + ask P3 + ask C2 + ask P2)")
vprint(f"         = {22.20:.2f} - 0.5*({12.05}+{12.05}+{9.75}+{9.75}) = +0.40 / unit  (locked in)")
vprint("-" * W)
vprint(f"  Note: 'qty' here applies to ALL FIVE legs simultaneously.  Each leg")
vprint(f"        consumes its own position-size budget (50 max each), so qty<=50.")
vprint("-" * W)
vprint(f"  {'Qty':>4}   {'E[PnL]':>10}  {'Std':>10}  {'VaR5':>10}  {'CVaR5':>10}  {'Sharpe':>7}")
for q in [10, 25, 50]:
    pnl = chooser_hedged_pnl(q, 2.51)
    e, sd = float(pnl.mean()), float(pnl.std())
    v5 = float(np.percentile(pnl, 5))
    c5 = float(pnl[pnl <= v5].mean())
    vprint(f"  {q:>4,}   {e:>+10,.0f}  {sd:>10,.0f}  {v5:>10,.0f}  {c5:>10,.0f}  {e/sd:>+7.4f}")
vprint("=" * W)

# ============================================================================
# 8. PORTFOLIO HELPER + CORE STATS
# ============================================================================
def portfolio_pnl(positions, sig, include_chooser_hedge=False, chooser_qty=0):
    """positions: dict(name -> (side, qty)).
    If include_chooser_hedge, also add SELL chooser_qty x AC_50_CO + variance-min hedge.
    """
    P = paths_by_sigma[sig]
    S2 = P[:, T2_STEPS - 1]; S3 = P[:, T3_STEPS - 1]
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
    if include_chooser_hedge and chooser_qty > 0:
        total += chooser_hedged_pnl(chooser_qty, sig)
    return total

def stats(pnl):
    e   = float(pnl.mean())
    sd  = float(pnl.std())
    v5  = float(np.percentile(pnl, 5))
    c5  = float(pnl[pnl <= v5].mean())
    pl  = float((pnl < 0).mean())
    pmin = float(pnl.min())
    median = float(np.median(pnl))
    p25  = float(np.percentile(pnl, 25))
    p75  = float(np.percentile(pnl, 75))
    return dict(E=e, Std=sd, VaR=v5, CVaR=c5, Ploss=pl, Min=pmin,
                Sharpe=e/sd if sd > 0 else 0.0,
                Median=median, P25=p25, P75=p75)

def evaluate(positions, chooser_qty=0):
    """Compute statistics under each sigma scenario."""
    by = {}
    for sig in SIGMA_SCENARIOS:
        pnl = portfolio_pnl(positions, sig,
                            include_chooser_hedge=(chooser_qty > 0),
                            chooser_qty=chooser_qty)
        by[sig] = stats(pnl)
    return by

def aggregate_metrics(by_sig):
    """Combine per-sigma stats into a robust score."""
    weighted_E    = sum(by_sig[s]["E"]    * w for s, w in zip(SIGMA_SCENARIOS, SIGMA_WEIGHTS))
    weighted_CVaR = sum(by_sig[s]["CVaR"] * w for s, w in zip(SIGMA_SCENARIOS, SIGMA_WEIGHTS))
    min_E         = min(by_sig[s]["E"]    for s in SIGMA_SCENARIOS)
    worst_CVaR    = min(by_sig[s]["CVaR"] for s in SIGMA_SCENARIOS)
    worst_min     = min(by_sig[s]["Min"]  for s in SIGMA_SCENARIOS)
    return dict(wE=weighted_E, wCVaR=weighted_CVaR,
                minE=min_E, worstCVaR=worst_CVaR, worstMin=worst_min)

# ============================================================================
# 9. EFFICIENT FRONTIER GRID  (KO + BP only -- the robust core)
# ============================================================================
# Optionally augmented with chooser-hedge or other near-robust trades.
vprint()
vprint("Computing the mean / worst-CVaR efficient frontier across position sizes...")

KO_GRID  = [0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
BP_GRID  = [0, 5, 10, 15, 20, 25, 30, 40, 50]
P35_GRID = [0, 5, 10, 15, 25, 50]

frontier = []  # list of (positions, agg, max_loss)

# 2D grid: KO x BP (no put hedge)
for q_ko in KO_GRID:
    for q_bp in BP_GRID:
        if q_ko == 0 and q_bp == 0:
            continue
        pos = {}
        if q_ko > 0: pos["AC_45_KO"] = ("BUY",  q_ko)
        if q_bp > 0: pos["AC_40_BP"] = ("SELL", q_bp)
        by = evaluate(pos)
        agg = aggregate_metrics(by)
        ml = (q_ko * CONTRACT_SIZE * MKT["AC_45_KO"]["ask"]
              + q_bp * CONTRACT_SIZE * (BINARY_PAYOFF - MKT["AC_40_BP"]["bid"]))
        frontier.append(dict(name=f"KO{q_ko}/BP{q_bp}",
                             positions=pos, by=by, agg=agg, max_loss=ml,
                             chooser_qty=0))

# 3D grid: KO x BP x P35 (downside-tail hedge)
# Skipping KO==0 because BP-only without KO is unlikely to win frontier
for q_ko in [50, 100, 200, 250, 300, 400, 500]:
    for q_bp in [0, 10, 25, 50]:
        for q_p35 in [5, 10, 15, 25, 50]:
            pos = {"AC_45_KO": ("BUY", q_ko),
                   "AC_35_P":  ("BUY", q_p35)}
            if q_bp > 0: pos["AC_40_BP"] = ("SELL", q_bp)
            by = evaluate(pos)
            agg = aggregate_metrics(by)
            ml = (q_ko * CONTRACT_SIZE * MKT["AC_45_KO"]["ask"]
                  + q_bp * CONTRACT_SIZE * (BINARY_PAYOFF - MKT["AC_40_BP"]["bid"])
                  + q_p35 * CONTRACT_SIZE * MKT["AC_35_P"]["ask"])
            frontier.append(dict(name=f"KO{q_ko}/BP{q_bp}/P35_{q_p35}",
                                 positions=pos, by=by, agg=agg, max_loss=ml,
                                 chooser_qty=0))

# Extension 2: KO + BP + 40_P BUY (closer-to-strike hedge)
for q_ko in [250, 500]:
    for q_bp in [25, 50]:
        for q_p40 in [5, 10, 15, 25, 50]:
            pos = {"AC_45_KO": ("BUY", q_ko),
                   "AC_40_BP": ("SELL", q_bp),
                   "AC_40_P":  ("BUY", q_p40)}
            by = evaluate(pos)
            agg = aggregate_metrics(by)
            ml = (q_ko * CONTRACT_SIZE * MKT["AC_45_KO"]["ask"]
                  + q_bp * CONTRACT_SIZE * (BINARY_PAYOFF - MKT["AC_40_BP"]["bid"])
                  + q_p40 * CONTRACT_SIZE * MKT["AC_40_P"]["ask"])
            frontier.append(dict(name=f"KO{q_ko}/BP{q_bp}/P40_{q_p40}",
                                 positions=pos, by=by, agg=agg, max_loss=ml,
                                 chooser_qty=0))

# Extension 3: BP-only portfolios (in case the user wants to avoid KO entirely)
for q_bp in [5, 10, 15, 20, 25, 30, 40, 50]:
    pos = {"AC_40_BP": ("SELL", q_bp)}
    by = evaluate(pos)
    agg = aggregate_metrics(by)
    ml = q_bp * CONTRACT_SIZE * (BINARY_PAYOFF - MKT["AC_40_BP"]["bid"])
    frontier.append(dict(name=f"BP{q_bp}_only",
                         positions=pos, by=by, agg=agg, max_loss=ml,
                         chooser_qty=0))

# Extension 4: A few chooser-hedged points (variance-min replication)
# Included for completeness - chooser is NOT robust (negative edge at sigma>=2.55)
# so these portfolios have lower min-E, but they may improve E[PnL] under model.
for q_co in [10, 25]:
    for q_ko in [250, 500]:
        for q_bp in [0, 25, 50]:
            pos = {}
            if q_ko > 0: pos["AC_45_KO"] = ("BUY",  q_ko)
            if q_bp > 0: pos["AC_40_BP"] = ("SELL", q_bp)
            by = evaluate(pos, chooser_qty=q_co)
            agg = aggregate_metrics(by)
            ml_chooser_emp = -agg["worstMin"]
            ml = (q_ko * CONTRACT_SIZE * MKT["AC_45_KO"]["ask"]
                  + q_bp * CONTRACT_SIZE * (BINARY_PAYOFF - MKT["AC_40_BP"]["bid"])
                  + ml_chooser_emp)
            frontier.append(dict(name=f"KO{q_ko}/BP{q_bp}/CO{q_co}h",
                                 positions=pos, by=by, agg=agg, max_loss=ml,
                                 chooser_qty=q_co))

# ============================================================================
# 10. PARETO FRONTIER  (max-E[PnL] vs CVaR budget)
# ============================================================================
# Sort by aggregated worst-CVaR descending (less negative = better),
# then keep only those that improve E across the curve.
frontier.sort(key=lambda f: f["agg"]["worstCVaR"], reverse=True)

pareto = []
best_E = -float("inf")
for f in frontier:
    if f["agg"]["minE"] > best_E:
        pareto.append(f)
        best_E = f["agg"]["minE"]

vprint()
vprint("=" * W)
vprint("  TABLE 4.  PARETO FRONTIER  (worst-sigma E[PnL] vs worst-sigma CVaR)")
vprint("-" * W)
vprint(f"  {'Portfolio':<22} {'KO':>4} {'BP':>4} {'CO':>4}  "
       f"{'wE':>10} {'min-E':>10}  {'wCVaR':>11}  {'worstCVaR':>11}  "
       f"{'maxLoss':>10}")
for f in pareto:
    q_ko = f["positions"].get("AC_45_KO", ("",0))[1]
    q_bp = f["positions"].get("AC_40_BP", ("",0))[1]
    q_co = f["chooser_qty"]
    a    = f["agg"]
    vprint(f"  {f['name']:<22} {q_ko:>4} {q_bp:>4} {q_co:>4}  "
           f"{a['wE']:>10,.0f} {a['minE']:>10,.0f}  "
           f"{a['wCVaR']:>11,.0f}  {a['worstCVaR']:>11,.0f}  "
           f"{f['max_loss']:>10,.0f}")
vprint("=" * W)

# ============================================================================
# 11. THREE-TIER RECOMMENDATIONS
# ============================================================================
def recommend(max_loss_budget, label, prefer="minE"):
    """Pick the portfolio with the highest score subject to max_loss <= budget.
    prefer = 'minE' (default, robust to vol) or 'sharpe' or 'lowPloss'.
    """
    feasible = [f for f in frontier if f["max_loss"] <= max_loss_budget
                                       and f["agg"]["minE"] > 0]
    if not feasible:
        return None
    if prefer == "minE":
        feasible.sort(key=lambda f: -f["agg"]["minE"])
    elif prefer == "sharpe":
        feasible.sort(key=lambda f: -f["by"][2.51]["Sharpe"])
    elif prefer == "lowPloss":
        # Higher win rate (lower loss prob) under model sigma
        feasible.sort(key=lambda f: f["by"][2.51]["Ploss"])
    return feasible[0]

TIERS = [
    ("ULTRA (loss<300k)",     300_000),
    ("LOW   (loss<750k)",     750_000),
    ("MOD   (loss<1.5M)",   1_500_000),
    ("AGGR  (loss<3M)",     3_000_000),
]

vprint()
vprint("=" * W)
vprint("  TIER RECOMMENDATIONS  (max-EV portfolio under each loss-budget; vol-robust score)")
vprint("-" * W)
vprint(f"  {'Tier':<22}  {'Portfolio':<22}  {'min-E':>10}  {'E@2.51':>10}  "
       f"{'CVaR@2.51':>11}  {'P(loss)':>8}  {'Sharpe':>7}  {'maxLoss':>10}")
for label, budget in TIERS:
    rec = recommend(budget, label, prefer="minE")
    if rec is None:
        vprint(f"  {label:<22}  {'(infeasible)':<22}")
        continue
    bs = rec["by"][2.51]; a = rec["agg"]
    vprint(f"  {label:<22}  {rec['name']:<22}  "
           f"{a['minE']:>10,.0f}  {bs['E']:>10,.0f}  {bs['CVaR']:>11,.0f}  "
           f"{bs['Ploss']*100:>7.1f}%  {bs['Sharpe']:>+7.4f}  "
           f"{rec['max_loss']:>10,.0f}")
vprint("=" * W)

# Alternate ranking: prefer HIGH win-rate (low P(loss))
vprint()
vprint("=" * W)
vprint("  ALT RANKING:  highest WIN-RATE portfolio under each loss-budget")
vprint(f"  (Important: 'KO BUY only' has 95%% loss rate -- positive-EV but LOTTERY-like.)")
vprint("-" * W)
vprint(f"  {'Tier':<22}  {'Portfolio':<22}  {'P(profit)':>9}  "
       f"{'Median':>10}  {'E@2.51':>10}  {'Sharpe':>7}  {'CVaR@2.51':>11}  {'maxLoss':>10}")
for label, budget in TIERS:
    rec = recommend(budget, label, prefer="lowPloss")
    if rec is None:
        vprint(f"  {label:<22}  {'(infeasible)':<22}")
        continue
    bs = rec["by"][2.51]
    vprint(f"  {label:<22}  {rec['name']:<22}  "
           f"{(1-bs['Ploss'])*100:>8.1f}%  {bs['Median']:>10,.0f}  "
           f"{bs['E']:>10,.0f}  {bs['Sharpe']:>+7.4f}  "
           f"{bs['CVaR']:>11,.0f}  {rec['max_loss']:>10,.0f}")
vprint("=" * W)

# Balanced criterion: highest Sharpe ratio under the loss budget
vprint()
vprint("=" * W)
vprint("  BALANCED RANKING:  highest SHARPE RATIO portfolio under each loss-budget")
vprint(f"  (Sharpe = E[PnL] / Std[PnL] -- the textbook risk-adjusted return measure.)")
vprint("-" * W)
vprint(f"  {'Tier':<22}  {'Portfolio':<22}  {'Sharpe':>7}  "
       f"{'E@2.51':>10}  {'Std@2.51':>10}  {'P(loss)':>8}  {'maxLoss':>10}")
for label, budget in TIERS:
    rec = recommend(budget, label, prefer="sharpe")
    if rec is None:
        vprint(f"  {label:<22}  {'(infeasible)':<22}")
        continue
    bs = rec["by"][2.51]
    vprint(f"  {label:<22}  {rec['name']:<22}  "
           f"{bs['Sharpe']:>+7.4f}  {bs['E']:>10,.0f}  {bs['Std']:>10,.0f}  "
           f"{bs['Ploss']*100:>7.1f}%  {rec['max_loss']:>10,.0f}")
vprint("=" * W)

# ============================================================================
# 11b.  DECISION MATRIX:  "best of" portfolios under user budget = $300k
# ============================================================================
USER_BUDGET = 300_000
candidates = [f for f in frontier if f["max_loss"] <= USER_BUDGET
                                     and f["agg"]["minE"] > 0]
# Identify each "best by criterion"
def best_by(cands, key):
    return max(cands, key=key)
best_minE   = best_by(candidates, lambda f: f["agg"]["minE"])
best_E      = best_by(candidates, lambda f: f["by"][2.51]["E"])
best_sharpe = best_by(candidates, lambda f: f["by"][2.51]["Sharpe"])
best_winrate= best_by(candidates, lambda f: -f["by"][2.51]["Ploss"])
best_median = best_by(candidates, lambda f: f["by"][2.51]["Median"])

shortlist = []
for tag, rec in [("max E[PnL]@2.51",   best_E),
                 ("max worst-vol E",   best_minE),
                 ("max Sharpe",        best_sharpe),
                 ("max win-rate",      best_winrate),
                 ("max median PnL",    best_median)]:
    bs = rec["by"][2.51]
    shortlist.append((tag, rec, bs))

vprint()
vprint("=" * W)
vprint(f"  DECISION MATRIX  (all portfolios under user's max-loss budget = "
       f"${USER_BUDGET:,})")
vprint(f"  Pick whichever best matches your preference profile.")
vprint("-" * W)
vprint(f"  {'Best by':<22}  {'Portfolio':<22}  {'E@2.51':>10}  "
       f"{'Median':>10}  {'Sharpe':>7}  {'P(profit)':>9}  {'maxLoss':>10}")
for tag, rec, bs in shortlist:
    vprint(f"  {tag:<22}  {rec['name']:<22}  {bs['E']:>10,.0f}  "
           f"{bs['Median']:>10,.0f}  {bs['Sharpe']:>+7.4f}  "
           f"{(1-bs['Ploss'])*100:>8.1f}%  {rec['max_loss']:>10,.0f}")
vprint("-" * W)
vprint("  Trade-off explanation:")
vprint("    'max E[PnL]' usually concentrates in KO BUY (long-shot lottery): high")
vprint("    expected value but 95% chance of taking the bounded loss this round.")
vprint("    'max Sharpe' or 'max win-rate' uses smaller KO + small BP/P35 hedges --")
vprint("    less expected value per round but more reliable path-by-path.")
vprint("=" * W)

# ============================================================================
# 12. DETAILED DIVE INTO THE USER-PREFERRED TIER (ultra-conservative)
# ============================================================================
top = recommend(300_000, "ULTRA-CONSERVATIVE") or recommend(400_000, "fallback")
if top is None:
    print("No feasible portfolio at any tier."); sys.exit(0)

# ============================================================================
# 12. VERBOSE-ONLY:  detailed dive + status-quo head-to-head
# ============================================================================
SPEC_MAP = {
    "AC_50_P_2": "Vanilla Put,   strike 50, 2-week expiry",
    "AC_50_C_2": "Vanilla Call,  strike 50, 2-week expiry",
    "AC_50_CO":  "Chooser,       strike 50, 3-week expiry, choice at week 2",
    "AC_40_BP":  "Binary Put,    strike 40, 3-week expiry, payoff 10 if S<40",
    "AC_45_KO":  "KO Put,        strike 45, 3-week expiry, knocked out if S<35 ever",
    "AC_60_C":   "Vanilla Call,  strike 60, 3-week expiry",
    "AC_45_P":   "Vanilla Put,   strike 45, 3-week expiry",
    "AC_35_P":   "Vanilla Put,   strike 35, 3-week expiry",
    "AC_40_P":   "Vanilla Put,   strike 40, 3-week expiry",
    "AC_50_P":   "Vanilla Put,   strike 50, 3-week expiry",
    "AC_50_C":   "Vanilla Call,  strike 50, 3-week expiry",
}

if VERBOSE:
    print()
    print("=" * W)
    print(f"  RECOMMENDED PORTFOLIO  (ultra-conservative, max loss <= $300k):")
    print(f"  Name = {top['name']}")
    print("-" * W)
    for n, (side, qty) in top["positions"].items():
        p = MKT[n]; px = p["ask"] if side == "BUY" else p["bid"]
        fair_251 = fair_value(n, 2.51); fair_247 = fair_value(n, 2.47)
        edge_251 = (p["bid"] - fair_251) if side == "SELL" else (fair_251 - p["ask"])
        edge_247 = (p["bid"] - fair_247) if side == "SELL" else (fair_247 - p["ask"])
        print(f"  {side:<4} {qty:>3} x  {n:<10} @ {px:>6.3f}   "
              f"edge(s=2.51)={edge_251:+.4f}   edge(s=2.47)={edge_247:+.4f}")
    print()
    print("  Risk profile under each plausible TRUE sigma:")
    print(f"  {'sigma':>6}   {'E[PnL]':>11}  {'Median':>10}  "
          f"{'P25':>10}  {'P75':>10}  {'CVaR 5%':>11}  {'P(loss)':>8}  {'Sharpe':>7}")
    for sig in SIGMA_SCENARIOS:
        s = top["by"][sig]
        flag = "  <- model"   if abs(sig - 2.51) < 0.005 else (
               "  <- mkt-IV"  if abs(sig - 2.47) < 0.005 else "")
        print(f"  {sig:>5.2f}   {s['E']:>11,.0f}  {s['Median']:>10,.0f}  "
              f"{s['P25']:>10,.0f}  {s['P75']:>10,.0f}  {s['CVaR']:>11,.0f}  "
              f"{s['Ploss']*100:>7.1f}%  {s['Sharpe']:>+7.4f}{flag}")
    print("=" * W)

# Status quo data is needed for the figure later; compute regardless of verbose.
status_quo = {
    "AC_50_P_2": ("BUY",  50),
    "AC_50_C_2": ("BUY",  50),
    "AC_50_CO":  ("SELL", 50),
    "AC_40_BP":  ("SELL", 50),
    "AC_45_KO":  ("BUY",  500),
}
sq_by  = evaluate(status_quo)
sq_agg = aggregate_metrics(sq_by)

if VERBOSE:
    print()
    print("=" * W)
    print("  HEAD-TO-HEAD: status quo vs three risk tiers")
    print("-" * W)
    print(f"  {'Portfolio':<22}  {'min-E':>10}  {'E@2.51':>10}  "
          f"{'CVaR@2.51':>11}  {'worstCVaR':>11}  {'P(loss)':>9}  {'maxLoss':>11}")
    print(f"  {'A_status_quo':<22}  {sq_agg['minE']:>10,.0f}  "
          f"{sq_by[2.51]['E']:>10,.0f}  {sq_by[2.51]['CVaR']:>11,.0f}  "
          f"{sq_agg['worstCVaR']:>11,.0f}  {sq_by[2.51]['Ploss']*100:>8.1f}%  "
          f"  UNBOUNDED")
    for label, budget in TIERS:
        rec = recommend(budget, label)
        if rec is None: continue
        a = rec["agg"]; b = rec["by"][2.51]
        print(f"  {label:<22}  {a['minE']:>10,.0f}  {b['E']:>10,.0f}  "
              f"{b['CVaR']:>11,.0f}  {a['worstCVaR']:>11,.0f}  "
              f"{b['Ploss']*100:>8.1f}%  {rec['max_loss']:>11,.0f}")
    print("=" * W)

# ============================================================================
# 13. CLEAN SIMPLIFIED SUMMARY  (always printed)
# ============================================================================
bs   = top["by"][2.51]
best_E_sigma = max(top["by"][s]["E"] for s in SIGMA_SCENARIOS)

# Build alternate-criterion shortlist for the summary footer
alt_lines = []
for tag, rec, b in shortlist:
    name = rec["name"]
    e    = b["E"]; med = b["Median"]; pp = (1 - b["Ploss"]) * 100
    alt_lines.append((tag, name, e, med, pp, rec["max_loss"]))

WS = 78
print()
print("=" * WS)
print(f"  ULTRA-LOW-RISK TRADE LIST   (max loss <= ${USER_BUDGET:,})")
print("=" * WS)
for n, (side, qty) in top["positions"].items():
    p = MKT[n]; px = p["ask"] if side == "BUY" else p["bid"]
    spec = SPEC_MAP.get(n, "")
    print(f"  {side:<4} {qty:>3}  x  {n:<10} @ ${px:<6.3f}   {spec}")
print("-" * WS)
print(f"  Expected PnL  (model sigma=2.51) ........... ${bs['E']:>+12,.0f}")
print(f"  Expected PnL  (worst sigma in [2.30, 2.70])  ${top['agg']['minE']:>+12,.0f}")
print(f"  Expected PnL  (best sigma) ................. ${best_E_sigma:>+12,.0f}")
print(f"  MAX DRAWDOWN  (hard floor on loss) ......... ${-top['max_loss']:>+12,.0f}")
print(f"  CVaR 5%       (avg PnL of worst 5%) ........ ${bs['CVaR']:>+12,.0f}")
print(f"  Probability of profit ......................  {(1-bs['Ploss'])*100:>5.1f}%")
print(f"  Probability of max-drawdown loss ...........  {bs['Ploss']*100:>5.1f}%")
print(f"  Sharpe ratio (E / Std) .....................  {bs['Sharpe']:>+5.3f}")
print("=" * WS)
print("  ALTERNATIVES under the same $300k budget (different optimisation goals):")
print("-" * WS)
print(f"  {'Best by':<22} {'Portfolio':<22} {'E[PnL]':>10}  {'Median':>10}  {'P(win)':>7}")
for tag, name, e, med, pp, _ml in alt_lines:
    print(f"  {tag:<22} {name:<22} {e:>+10,.0f}  {med:>+10,.0f}  {pp:>6.1f}%")
print("=" * WS)
print("  Run with -v / --verbose for the full edge-table + Pareto frontier.")
print("=" * WS)

# ============================================================================
# 14. VISUALISATION  (verbose only -- skip when running quietly)
# ============================================================================
if not VERBOSE:
    sys.exit(0)

print()
print("Generating optimal_strategy.png...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#0d1b2a", "axes.facecolor": "#1b263b",
    "axes.edgecolor":   "#5a6e8a", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "text.color":  "#f0f0f0", "grid.color":  "#3a4a6a", "grid.alpha": 0.4,
    "axes.titlecolor": "#ffffff", "axes.titlesize": 11,
    "font.family": "monospace",
})
CYAN, ORANGE, GREEN, RED, YELLOW, PURPLE, GREY = (
    "#00d4ff", "#ff8c42", "#2ecc71", "#e74c3c", "#f1c40f", "#9b59b6", "#7f8c8d")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Round 4  |  OPTIMAL RISK-AVERSE STRATEGY  "
             f"(efficient frontier across {N_SIM:,} MC paths x {len(SIGMA_SCENARIOS)} vol regimes)",
             fontsize=13)

# --- TL: Efficient frontier scatter
ax = axes[0, 0]
ax.set_title("Mean-CVaR efficient frontier (each dot = candidate portfolio)", fontsize=11)
xs = np.array([f["agg"]["worstCVaR"] for f in frontier])
ys = np.array([f["agg"]["minE"]      for f in frontier])
mls = np.array([f["max_loss"] for f in frontier])
sc = ax.scatter(xs, ys, c=np.log10(np.maximum(mls, 1e3)), cmap="plasma", s=22,
                alpha=0.7, edgecolor="white", lw=0.2)
cb = plt.colorbar(sc, ax=ax); cb.set_label("log10(max loss budget) [$]")
# Pareto front
xs_p = [f["agg"]["worstCVaR"] for f in pareto]
ys_p = [f["agg"]["minE"]      for f in pareto]
ax.plot(xs_p, ys_p, color=CYAN, lw=1.5, alpha=0.7, label=f"Pareto frontier ({len(pareto)} pts)")
# Status quo and tier recommendations
ax.scatter([sq_agg["worstCVaR"]], [sq_agg["minE"]], c=ORANGE, s=160,
           edgecolor="white", lw=1.2, marker="X", zorder=8, label="Status Quo")
for label, budget in TIERS:
    rec = recommend(budget, label)
    if rec is None: continue
    ax.scatter([rec["agg"]["worstCVaR"]], [rec["agg"]["minE"]],
               c=GREEN, s=160, edgecolor="white", lw=1.2, marker="*", zorder=9)
    ax.annotate(label[:5], (rec["agg"]["worstCVaR"], rec["agg"]["minE"]),
                fontsize=8, xytext=(8, 6), textcoords="offset points",
                color="white")
ax.axhline(0, color="white", lw=0.5, alpha=0.4)
ax.axvline(0, color="white", lw=0.5, alpha=0.4)
ax.set_xlabel("worst-sigma CVaR 5% (more positive = less downside risk)")
ax.set_ylabel("worst-sigma E[PnL]   (more positive = better)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, lw=0.4)

# --- TR: PnL distribution comparison at sigma=2.51
ax = axes[0, 1]
ax.set_title("PnL distribution @ sigma=2.51 (model)", fontsize=11)
sq_pnl  = portfolio_pnl(status_quo, 2.51)
top_pnl = portfolio_pnl(top["positions"], 2.51,
                         include_chooser_hedge=(top["chooser_qty"]>0),
                         chooser_qty=top["chooser_qty"])
lo, hi = np.percentile(sq_pnl, [0.5, 99.5])
ax.hist(np.clip(sq_pnl,  lo, hi), bins=120, color=ORANGE, alpha=0.55,
        density=True, label=f"status quo  E={sq_pnl.mean():,.0f}")
ax.hist(np.clip(top_pnl, lo, hi), bins=120, color=CYAN,   alpha=0.55,
        density=True, label=f"{top['name']}  E={top_pnl.mean():,.0f}")
ax.axvline(0, color="white", lw=0.6, alpha=0.5)
ax.axvline(np.percentile(sq_pnl, 5), color=ORANGE, lw=1.4, ls="--",
           label=f"SQ VaR5={np.percentile(sq_pnl,5):,.0f}")
ax.axvline(np.percentile(top_pnl,5), color=CYAN, lw=1.4, ls="--",
           label=f"top VaR5={np.percentile(top_pnl,5):,.0f}")
ax.set_xlabel("Portfolio PnL ($)"); ax.set_ylabel("Density")
ax.legend(fontsize=8); ax.grid(True, lw=0.4)

# --- BL: E[PnL] across sigmas for top tiers
ax = axes[1, 0]
ax.set_title("E[PnL] across vol regimes  (vol-robustness check)", fontsize=11)
plot_set = [("status quo", status_quo, ORANGE)]
for label, budget in TIERS:
    rec = recommend(budget, label)
    if rec is None: continue
    plot_set.append((label, rec["positions"], None))
palette = [ORANGE, GREEN, CYAN, YELLOW, PURPLE]
for (label, pos, _), col in zip(plot_set, palette):
    es = []
    for s in SIGMA_SCENARIOS:
        # for tier rows we need chooser-hedge support
        if label != "status quo":
            rec = next((r for r in pareto if r["positions"] == pos), None)
            cq = rec["chooser_qty"] if rec else 0
        else:
            cq = 0
        pnl = portfolio_pnl(pos, s, include_chooser_hedge=cq>0, chooser_qty=cq)
        es.append(float(pnl.mean()))
    ax.plot(SIGMA_SCENARIOS, es, color=col, lw=2.0, marker="o", ms=6, label=label)
ax.axhline(0, color="white", lw=0.6, alpha=0.4)
ax.axvline(2.51, color="white", lw=0.6, ls=":", alpha=0.6)
ax.axvline(2.47, color=YELLOW,  lw=0.6, ls=":", alpha=0.6, label="market-implied (2.47)")
ax.set_xlabel("True sigma"); ax.set_ylabel("E[PnL] ($)")
ax.legend(fontsize=8); ax.grid(True, lw=0.4)

# --- BR: CVaR across sigmas for top tiers
ax = axes[1, 1]
ax.set_title("CVaR 5% across vol regimes  (tail-risk profile)", fontsize=11)
for (label, pos, _), col in zip(plot_set, palette):
    cs = []
    for s in SIGMA_SCENARIOS:
        if label != "status quo":
            rec = next((r for r in pareto if r["positions"] == pos), None)
            cq = rec["chooser_qty"] if rec else 0
        else:
            cq = 0
        pnl = portfolio_pnl(pos, s, include_chooser_hedge=cq>0, chooser_qty=cq)
        v = float(np.percentile(pnl, 5))
        c = float(pnl[pnl <= v].mean())
        cs.append(c)
    ax.plot(SIGMA_SCENARIOS, cs, color=col, lw=2.0, marker="o", ms=6, label=label)
ax.axhline(0, color="white", lw=0.6, alpha=0.4)
ax.axvline(2.51, color="white", lw=0.6, ls=":", alpha=0.6)
ax.axvline(2.47, color=YELLOW,  lw=0.6, ls=":", alpha=0.6)
ax.set_xlabel("True sigma"); ax.set_ylabel("CVaR 5% ($)")
ax.legend(fontsize=8); ax.grid(True, lw=0.4)

plt.tight_layout()
out = "fig8_optimal_strategy.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"  Saved {out}")
print()
print("Done.")
