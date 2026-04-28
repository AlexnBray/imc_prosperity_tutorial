# -*- coding: utf-8 -*-
import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

"""
Round 4 Manual: RISK-AVERSE PORTFOLIO OPTIMIZER
================================================
Goal: Find the trade portfolio that maximises *risk-adjusted* PnL while remaining
robust to the single biggest hidden risk -- VOLATILITY MIS-SPECIFICATION.

Problem statement
-----------------
Our pricing model assumes sigma = 251%.
The market-implied vol on the 2wk straddle is 247% (bid-ask 9.70-9.75).
At sigma = 247%, the 2wk straddle BUY is a LOSING trade.
The competition realises ONE path -- bad luck can wipe out the entire round.

This script:
  1. Stress-tests every trade's edge across sigma in [2.20, 2.70]
  2. Simulates portfolio PnL under multiple TRUE-vol assumptions
  3. Compares 7 candidate portfolios on E[PnL], VaR 5%, CVaR 5%, Sharpe
  4. Ranks them by *worst-case CVaR across vol regimes* and recommends the winner

Key principle: prefer trades whose edge survives at the LOWER end of the plausible
vol range (since lower vol is the market's implied estimate -- our 251% may be too high).
"""

import numpy as np
from scipy.stats import norm

# ============================================================================
# Parameters (must match competition spec exactly)
# ============================================================================
S0           = 50.0
SIGMA_MODEL  = 2.51
R            = 0.0
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY
DT = 1.0 / STEPS_PER_YEAR

T2_STEPS = 2 * 5 * STEPS_PER_DAY        # 40  (2 weeks)
T3_STEPS = 3 * 5 * STEPS_PER_DAY        # 60  (3 weeks)
T2_YEARS = (2 * 5) / TRADING_DAYS_PER_YEAR
T3_YEARS = (3 * 5) / TRADING_DAYS_PER_YEAR

BINARY_PAYOFF = 10.0    # AC_40_BP: pays 10 if S_3wk < 40
KO_BARRIER    = 35.0    # AC_45_KO: knocked out if price ever falls below 35
CONTRACT_SIZE = 3000

N_SIM = 400_000
SEED  = 42

# Plausible sigma scenarios:
#   2.30 = stress-low (market mis-estimates BOTH durations downward)
#   2.47 = market-implied 2wk vol (bid-ask 9.70/9.75 -> IV ~ 247%)
#   2.51 = competition model (251%)
#   2.60 = stress-high
# We weight 2.47 and 2.51 most heavily (highest plausibility).
SIGMA_SCENARIOS = [2.30, 2.40, 2.47, 2.49, 2.51, 2.55, 2.60, 2.70]
SIGMA_WEIGHTS   = [0.05, 0.10, 0.25, 0.15, 0.30, 0.10, 0.04, 0.01]   # sums to 1.0

W = 110

# ============================================================================
# Black-Scholes helpers (r=0 throughout)
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
# Market quotes (max contract sizes from simulate.py)
# ============================================================================
MKT = {
    "AC_50_P2": dict(bid=9.70,  ask=9.75,  size=50,  T=T2_YEARS, K=50, kind="put"),
    "AC_50_C2": dict(bid=9.70,  ask=9.75,  size=50,  T=T2_YEARS, K=50, kind="call"),
    "AC_50_CO": dict(bid=22.20, ask=22.30, size=50,  T=T3_YEARS, K=50, kind="chooser"),
    "AC_40_BP": dict(bid=5.00,  ask=5.10,  size=50,  T=T3_YEARS, K=40, kind="binary"),
    "AC_45_KO": dict(bid=0.15,  ask=0.175, size=500, T=T3_YEARS, K=45, kind="ko"),
    # Near-miss vanilla candidates
    "AC_60_C":  dict(bid=8.80,  ask=8.85,  size=50,  T=T3_YEARS, K=60, kind="call"),
    "AC_45_P":  dict(bid=9.05,  ask=9.10,  size=50,  T=T3_YEARS, K=45, kind="put"),
    "AC_35_P":  dict(bid=4.33,  ask=4.35,  size=50,  T=T3_YEARS, K=35, kind="put"),
    "AC_40_P":  dict(bid=6.50,  ask=6.55,  size=50,  T=T3_YEARS, K=40, kind="put"),
    "AC_50_P":  dict(bid=12.00, ask=12.05, size=50,  T=T3_YEARS, K=50, kind="put"),
    "AC_50_C":  dict(bid=12.00, ask=12.05, size=50,  T=T3_YEARS, K=50, kind="call"),
}

# ============================================================================
# Fair-value pricer for a single product at a given sigma
# (KO requires MC and is filled in later from a path-cube)
# ============================================================================
def fair_value(name, sig, ko_fairs_by_sig=None):
    p = MKT[name]
    K = p["K"]; T = p["T"]; kind = p["kind"]
    if kind == "put":     return bs_put(S0, K, T, sig)
    if kind == "call":    return bs_call(S0, K, T, sig)
    if kind == "chooser": return bs_chooser(S0, K, T_expiry=T3_YEARS, T_choice=T2_YEARS, sig=sig)
    if kind == "binary":  return bs_binary_put(S0, K, T, sig, payoff=BINARY_PAYOFF)
    if kind == "ko":      return ko_fairs_by_sig[sig]
    raise ValueError(kind)

# ============================================================================
# 1.  Simulate paths under SIGMA_MODEL (for KO fair) and under each TRUE-vol
# ============================================================================
print("Pre-computing GBM cubes for each sigma scenario...")
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_SIM, T3_STEPS))

def simulate_paths(sigma):
    log_inc  = (R - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * Z
    log_path = np.cumsum(log_inc, axis=1)
    return S0 * np.exp(log_path)

paths_by_sigma = {sig: simulate_paths(sig) for sig in SIGMA_SCENARIOS}

# KO fair value via MC at every sigma (KO has no closed form)
ko_fairs_by_sig = {}
for sig, P in paths_by_sigma.items():
    breach   = np.any(P[:, :T3_STEPS - 1] < KO_BARRIER, axis=1)
    S3       = P[:, T3_STEPS - 1]
    ko_pay   = np.where(breach, 0.0, np.maximum(45 - S3, 0))
    ko_fairs_by_sig[sig] = float(np.mean(ko_pay))

# ============================================================================
# 2.  Vol-robust edge table   (edge = bid - fair if SELL, fair - ask if BUY)
# ============================================================================
print()
print("=" * W)
print("  TABLE 1.  PER-TRADE EDGE ACROSS VOL REGIMES   (positive = profitable)")
print("-" * W)
prods = ["AC_50_P2", "AC_50_C2", "AC_50_CO", "AC_40_BP", "AC_45_KO",
         "AC_60_C",  "AC_45_P",  "AC_35_P",  "AC_40_P",  "AC_50_P", "AC_50_C"]
sides = {"AC_50_P2": "BUY", "AC_50_C2": "BUY", "AC_50_CO": "SELL",
         "AC_40_BP": "SELL", "AC_45_KO": "BUY",
         "AC_60_C":  "SELL", "AC_45_P":  "SELL", "AC_35_P":  "BUY",
         "AC_40_P":  "BUY",  "AC_50_P":  "SELL", "AC_50_C":  "SELL"}

hdr = f"  {'Product':<11} {'Side':<5}"
for sig in SIGMA_SCENARIOS:
    hdr += f"  {'σ='+f'{int(sig*100)}%':>7}"
hdr += f"  {'WORST':>8}  {'WEIGHTED':>9}"
print(hdr)
print("-" * W)

per_product_edges = {}
for name in prods:
    p = MKT[name]
    side = sides[name]
    row = f"  {name:<11} {side:<5}"
    edges = []
    for sig in SIGMA_SCENARIOS:
        f = fair_value(name, sig, ko_fairs_by_sig)
        edge = (p["bid"] - f) if side == "SELL" else (f - p["ask"])
        edges.append(edge)
        row += f"  {edge:>+7.4f}"
    worst = min(edges)
    weighted = sum(e * w for e, w in zip(edges, SIGMA_WEIGHTS))
    per_product_edges[name] = dict(edges=edges, worst=worst, weighted=weighted, side=side)
    row += f"  {worst:>+8.4f}  {weighted:>+9.4f}"
    print(row)
print("-" * W)
print("  WORST = edge under the worst sigma scenario tested")
print("  WEIGHTED = vol-prior-weighted average edge")
print("  Note: trades positive across ALL sigmas are 'structurally robust' (no vol bet)")
print("=" * W)

# ============================================================================
# 3.  Per-path PnL machinery -- helper to build payoff arrays at any sigma
# ============================================================================
def payoff_arrays(P):
    """Return dict of per-path payoff arrays for each product, given a path cube."""
    S2 = P[:, T2_STEPS - 1]
    S3 = P[:, T3_STEPS - 1]
    breach = np.any(P[:, :T3_STEPS - 1] < KO_BARRIER, axis=1)
    return dict(
        AC_50_P2 = np.maximum(50 - S2, 0),
        AC_50_C2 = np.maximum(S2 - 50, 0),
        AC_50_CO = np.where(S2 >= 50, np.maximum(S3 - 50, 0), np.maximum(50 - S3, 0)),
        AC_40_BP = np.where(S3 < 40, BINARY_PAYOFF, 0.0),
        AC_45_KO = np.where(breach, 0.0, np.maximum(45 - S3, 0)),
        AC_60_C  = np.maximum(S3 - 60, 0),
        AC_45_P  = np.maximum(45 - S3, 0),
        AC_35_P  = np.maximum(35 - S3, 0),
        AC_40_P  = np.maximum(40 - S3, 0),
        AC_50_P  = np.maximum(50 - S3, 0),
        AC_50_C  = np.maximum(S3 - 50, 0),
    )

payoffs_by_sigma = {sig: payoff_arrays(P) for sig, P in paths_by_sigma.items()}

def trade_pnl_per_unit(name, side, payoffs):
    """Per-path PnL per unit (price units) for one trade."""
    p = MKT[name]
    if side == "BUY":
        return payoffs[name] - p["ask"]
    else:
        return p["bid"] - payoffs[name]

def portfolio_pnl(positions, payoffs):
    """positions = dict(name -> (side, qty)); returns per-path total PnL in $."""
    total = np.zeros(payoffs["AC_50_P2"].shape[0])
    for name, (side, qty) in positions.items():
        unit_pnl = trade_pnl_per_unit(name, side, payoffs)
        total += qty * CONTRACT_SIZE * unit_pnl
    return total

def stats(pnl):
    e   = float(pnl.mean())
    sd  = float(pnl.std())
    var = float(np.percentile(pnl, 5))
    cvar = float(pnl[pnl <= var].mean())
    pl  = float((pnl < 0).mean())
    return dict(E=e, Std=sd, Sharpe=e/sd if sd > 0 else 0.0,
                VaR=var, CVaR=cvar, Ploss=pl)

# ============================================================================
# 4.  Candidate portfolios
# ============================================================================
PORTFOLIOS = {
    "A_status_quo":
        # Current 5-trade plan: long-straddle + short-chooser is a *self-hedge*
        # (the long calls/puts cap the chooser's tail loss on extreme moves)
        {"AC_50_P2": ("BUY", 50), "AC_50_C2": ("BUY", 50),
         "AC_50_CO": ("SELL", 50), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},

    "B_no_2wk_just_short_vol":
        # Drop hedges -- exposes naked short chooser. Tail risk explodes.
        {"AC_50_CO": ("SELL", 50), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},

    "C_half_size_everywhere":
        # Linear 50% scale of A: same Sharpe, half the variance
        {"AC_50_P2": ("BUY", 25), "AC_50_C2": ("BUY", 25),
         "AC_50_CO": ("SELL", 25), "AC_40_BP": ("SELL", 25),
         "AC_45_KO": ("BUY", 250)},

    "D_75pct_everywhere":
        # 75% scale -- gentle de-risking
        {"AC_50_P2": ("BUY", 38), "AC_50_C2": ("BUY", 38),
         "AC_50_CO": ("SELL", 38), "AC_40_BP": ("SELL", 38),
         "AC_45_KO": ("BUY", 375)},

    "E_status_plus_35P_hedge":
        # Status quo + BUY 200 x AC_35_P (cheap tail-down protection)
        # Pays off in the deep-down scenarios where chooser/binary blow up
        {"AC_50_P2": ("BUY", 50), "AC_50_C2": ("BUY", 50),
         "AC_50_CO": ("SELL", 50), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500), "AC_35_P": ("BUY", 50)},

    "F_status_plus_45P_sell":
        # Status quo + SELL 50 x AC_45_P (offsets KO's vanilla-put exposure;
        # crystalises the BGK edge and hedges the long-puts variance)
        {"AC_50_P2": ("BUY", 50), "AC_50_C2": ("BUY", 50),
         "AC_50_CO": ("SELL", 50), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500), "AC_45_P": ("SELL", 50)},

    "G_chooser_half_only":
        # Half the chooser position (the short with the deepest tail risk).
        # Keep KO and binary at full -- they have the best Sharpes.
        {"AC_50_P2": ("BUY", 25), "AC_50_C2": ("BUY", 25),
         "AC_50_CO": ("SELL", 25), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},

    "H_robust_core":
        # Risk-averse build: only the trades that are positive across the
        # full sigma range (KO + binary), plus a controlled chooser+straddle
        # hedge (chooser is robust to lower vol, straddle hedges its tail).
        {"AC_50_P2": ("BUY", 30), "AC_50_C2": ("BUY", 30),
         "AC_50_CO": ("SELL", 30), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},

    "I_KO_binary_only":
        # Most conservative: only the two highest-Sharpe vol-robust trades.
        # Both trades have BOUNDED max loss: KO=262.5k, BP=750k, total=1.0125M cap
        {"AC_40_BP": ("SELL", 50), "AC_45_KO": ("BUY", 500)},

    "J_KO_BP_chooser25":
        # I + half chooser (vol-robust below 2.55, but unbounded tail)
        {"AC_50_CO": ("SELL", 25), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},

    "K_KO_BP_chooser10":
        # I + tiny chooser exposure -- small extra E[PnL], small extra tail
        {"AC_50_CO": ("SELL", 10), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},

    "L_KO_BP_chooser_hedged":
        # I + chooser-25 + tiny long straddle (10) to cap chooser's tail
        # Long straddle pays off on extreme moves which is exactly when
        # the chooser SELL bleeds.
        {"AC_50_P2": ("BUY", 10), "AC_50_C2": ("BUY", 10),
         "AC_50_CO": ("SELL", 25), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},

    # --- USER-FOCUSED: MAX EV WITH BOUNDED RISK ---
    "M_user_KO250_only":
        # User's image: KO=250 (half max), binary SELL 50. Pure bounded risk.
        # Max loss hard-capped: 250x3000x0.175 + 50x3000x5 = 131.25k + 750k = 881.25k
        {"AC_40_BP": ("SELL", 50), "AC_45_KO": ("BUY", 250)},

    "N_KO500_BP_chooser15_hedged":
        # Sweet spot: max KO (all 500 available), full binary, tiny chooser=15
        # Hedge the chooser tail with 10 units of 2wk straddle
        # Straddle pays on big moves = exactly when short chooser bleeds most
        {"AC_50_P2": ("BUY", 10), "AC_50_C2": ("BUY", 10),
         "AC_50_CO": ("SELL", 15), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},

    "O_KO500_BP_chooser20_hedged":
        # Slightly more aggressive: chooser=20 with straddle hedge (15 each)
        {"AC_50_P2": ("BUY", 15), "AC_50_C2": ("BUY", 15),
         "AC_50_CO": ("SELL", 20), "AC_40_BP": ("SELL", 50),
         "AC_45_KO": ("BUY", 500)},
}

# ============================================================================
# 5.  Evaluate each portfolio under each sigma scenario
# ============================================================================
print()
print("=" * W)
print("  TABLE 2.  PORTFOLIO PnL UNDER EACH TRUE-VOL SCENARIO   (PnL in $)")
print("-" * W)
hdr = f"  {'Portfolio':<24} {'Metric':<8}"
for sig in SIGMA_SCENARIOS:
    hdr += f"  {'σ='+f'{int(sig*100)}':>9}"
hdr += f"  {'worst-σ':>10}"
print(hdr)
print("-" * W)

results = {}
for pname, pos in PORTFOLIOS.items():
    by_sig = {}
    for sig, payoffs in payoffs_by_sigma.items():
        pnl = portfolio_pnl(pos, payoffs)
        by_sig[sig] = stats(pnl)
    results[pname] = by_sig

    # Print rows: E[PnL], VaR 5%, CVaR 5%
    for metric in ["E", "VaR", "CVaR"]:
        row = f"  {pname:<24} {metric:<8}"
        worst = float("inf")
        for sig in SIGMA_SCENARIOS:
            v = by_sig[sig][metric]
            row += f"  {v:>9,.0f}"
            worst = min(worst, v)
        row += f"  {worst:>10,.0f}"
        print(row)
    print()
print("=" * W)

# ============================================================================
# 6.  Composite ranking score
#       0.5  *  weighted-E[PnL]      (across vol prior)
#     + 0.5  *  worst-CVaR           (downside protection)
# ============================================================================
print()
print("=" * W)
print("  TABLE 3.  COMPOSITE RANKING   (balance of expected and worst-case)")
print("-" * W)
print(f"  {'Portfolio':<24} {'wE[PnL]':>12} {'min-E':>10} {'wCVaR5%':>12} "
      f"{'minCVaR':>10} {'P(loss)51%':>11} {'SCORE':>10}")
print("-" * W)

ranked = []
for pname, by_sig in results.items():
    weighted_E    = sum(by_sig[s]["E"]    * w for s, w in zip(SIGMA_SCENARIOS, SIGMA_WEIGHTS))
    min_E         = min(by_sig[s]["E"]    for s in SIGMA_SCENARIOS)
    weighted_CVaR = sum(by_sig[s]["CVaR"] * w for s, w in zip(SIGMA_SCENARIOS, SIGMA_WEIGHTS))
    min_CVaR      = min(by_sig[s]["CVaR"] for s in SIGMA_SCENARIOS)
    p_loss_251    = by_sig[2.51]["Ploss"]
    # Score: half expected + half tail. Both signed positive -> larger is better.
    score = 0.5 * weighted_E + 0.5 * min_CVaR
    ranked.append((pname, weighted_E, min_E, weighted_CVaR, min_CVaR, p_loss_251, score))

ranked.sort(key=lambda r: -r[6])
for r in ranked:
    pname, wE, mE, wC, mC, pl, sc = r
    print(f"  {pname:<24} {wE:>12,.0f} {mE:>10,.0f} {wC:>12,.0f} "
          f"{mC:>10,.0f} {pl*100:>10.1f}% {sc:>10,.0f}")
print("=" * W)

# ============================================================================
# 7.  Detailed look at the recommended (top-ranked) portfolio
# ============================================================================
top_name = ranked[0][0]
top_pos  = PORTFOLIOS[top_name]
print()
print("=" * W)
print(f"  RECOMMENDED PORTFOLIO:  {top_name}")
print("-" * W)
for name, (side, qty) in top_pos.items():
    p = MKT[name]
    fair_251 = fair_value(name, 2.51, ko_fairs_by_sig)
    fair_247 = fair_value(name, 2.47, ko_fairs_by_sig)
    edge_251 = (p["bid"] - fair_251) if side == "SELL" else (fair_251 - p["ask"])
    edge_247 = (p["bid"] - fair_247) if side == "SELL" else (fair_247 - p["ask"])
    px       = p["bid"] if side == "SELL" else p["ask"]
    print(f"  {side:<4} {qty:>3} x  {name:<10} @ {px:>6.3f}   "
          f"edge(σ=2.51)={edge_251:+.4f}   edge(σ=2.47)={edge_247:+.4f}")
print("-" * W)
print(f"  Detailed PnL stats under each TRUE-vol scenario:")
print(f"  {'σ':>6}   {'E[PnL]':>11}  {'Std':>11}  {'VaR 5%':>11}  {'CVaR 5%':>11}  "
      f"{'P(loss)':>9}  {'Sharpe':>7}")
for sig in SIGMA_SCENARIOS:
    s = results[top_name][sig]
    flag = " <-- model" if abs(sig - 2.51) < 0.005 else (" <-- mkt-IV" if abs(sig - 2.47) < 0.005 else "")
    print(f"  {sig:>5.2f}   {s['E']:>11,.0f}  {s['Std']:>11,.0f}  "
          f"{s['VaR']:>11,.0f}  {s['CVaR']:>11,.0f}  "
          f"{s['Ploss']*100:>8.1f}%  {s['Sharpe']:>7.4f}{flag}")
print("=" * W)

# ============================================================================
# 8.  SAVINGS / DOWNGRADES vs. baseline (status quo)
# ============================================================================
print()
print("=" * W)
print(f"  TRADE-OFF vs. status_quo at sigma in [2.47, 2.51]")
print("-" * W)
sq = results["A_status_quo"]
for r in ranked:
    pname = r[0]
    p = results[pname]
    dE_251 = p[2.51]["E"]    - sq[2.51]["E"]
    dE_247 = p[2.47]["E"]    - sq[2.47]["E"]
    dC_251 = p[2.51]["CVaR"] - sq[2.51]["CVaR"]
    dC_247 = p[2.47]["CVaR"] - sq[2.47]["CVaR"]
    print(f"  {pname:<24}  ΔE@2.51={dE_251:>+10,.0f}  ΔE@2.47={dE_247:>+10,.0f}  "
          f"ΔCVaR@2.51={dC_251:>+10,.0f}  ΔCVaR@2.47={dC_247:>+10,.0f}")
print("=" * W)

# ============================================================================
# 9.  Side-by-side: status quo vs recommended -- what *changed*
# ============================================================================
print()
print("=" * W)
print(f"  SIDE-BY-SIDE  status_quo  vs  {top_name}")
print("-" * W)
all_names = sorted(set(PORTFOLIOS["A_status_quo"]) | set(top_pos))
for n in all_names:
    a = PORTFOLIOS["A_status_quo"].get(n, ("--", 0))
    b = top_pos.get(n, ("--", 0))
    delta = ""
    if a == b:
        delta = "(unchanged)"
    elif a[1] == 0:
        delta = "(NEW)"
    elif b[1] == 0:
        delta = "(REMOVED)"
    else:
        delta = f"(qty {a[1]} -> {b[1]})"
    print(f"  {n:<11}   status_quo: {a[0]:<5} {a[1]:>3}u    "
          f"{top_name}: {b[0]:<5} {b[1]:>3}u    {delta}")
print("=" * W)

# ============================================================================
# 10. Final summary line
# ============================================================================
top_E_251    = results[top_name][2.51]["E"]
top_E_247    = results[top_name][2.47]["E"]
top_CVaR_251 = results[top_name][2.51]["CVaR"]
sq_E_251     = sq[2.51]["E"]
print()
print(f"SUMMARY")
print(f"  Recommended:  {top_name}")
print(f"  E[PnL] @ sigma=2.51:  {top_E_251:>10,.0f}   (vs status_quo {sq_E_251:>10,.0f})")
print(f"  E[PnL] @ sigma=2.47:  {top_E_247:>10,.0f}   <- market-implied scenario")
print(f"  CVaR  @ sigma=2.51:  {top_CVaR_251:>10,.0f}   (worst-5% mean)")
print()

# ============================================================================
# 11. FINAL TRADE LIST -- exact wording for submission
# ============================================================================
print("=" * W)
print("  FINAL TRADE LIST FOR SUBMISSION  (recommended portfolio)")
print("-" * W)
print("  Match competition product names from README.md (AC_50_P_2, AC_50_C_2, etc.)")
print()
NAME_MAP = {"AC_50_P2": "AC_50_P_2", "AC_50_C2": "AC_50_C_2",
            "AC_50_CO": "AC_50_CO",  "AC_40_BP": "AC_40_BP",
            "AC_45_KO": "AC_45_KO",  "AC_60_C":  "AC_60_C",
            "AC_45_P":  "AC_45_P",   "AC_35_P":  "AC_35_P",
            "AC_40_P":  "AC_40_P",   "AC_50_P":  "AC_50_P",  "AC_50_C": "AC_50_C"}
SPEC_MAP = {
    "AC_50_P2": "Vanilla Put,  strike 50, 2-week expiry",
    "AC_50_C2": "Vanilla Call, strike 50, 2-week expiry",
    "AC_50_CO": "Chooser,      strike 50, 3-week expiry, choice at week 2",
    "AC_40_BP": "Binary Put,   strike 40, 3-week expiry, payoff 10 if S<40",
    "AC_45_KO": "KO Put,       strike 45, 3-week expiry, knocked out if S<35 ever",
    "AC_60_C":  "Vanilla Call, strike 60, 3-week expiry",
}
for name, (side, qty) in top_pos.items():
    p = MKT[name]
    px = p["bid"] if side == "SELL" else p["ask"]
    spec = SPEC_MAP.get(name, "")
    full = NAME_MAP.get(name, name)
    print(f"  {side:<4} {qty:>3} contracts of {full:<11}  @ {px:>6.3f}   ({spec})")
print("-" * W)
print()
print("  Bounded max loss check (sums per-trade max-loss across portfolio):")
max_loss = 0
for name, (side, qty) in top_pos.items():
    p = MKT[name]
    if side == "BUY":
        ml = qty * CONTRACT_SIZE * p["ask"]
        print(f"    BUY  {name}: max loss if expires worthless = "
              f"{qty} x 3000 x {p['ask']:.3f} = {ml:>10,.0f}")
        max_loss += ml
    else:
        # binary worst-case payoff is BINARY_PAYOFF, vanilla is unbounded (skip)
        if name == "AC_40_BP":
            ml = qty * CONTRACT_SIZE * (BINARY_PAYOFF - p["bid"])
            print(f"    SELL {name}: max loss if S<40 = "
                  f"{qty} x 3000 x ({BINARY_PAYOFF}-{p['bid']:.2f}) = {ml:>10,.0f}")
            max_loss += ml
        else:
            print(f"    SELL {name}: UNBOUNDED max loss (no hard cap)")
            max_loss = float("inf")
    print(f"    {'TOTAL portfolio max loss':<40} = {max_loss:>10,.0f}" if max_loss != float("inf")
      else "    TOTAL: UNBOUNDED (some short has no cap)")
print("=" * W)

# ============================================================================
# 12. Visualization: head-to-head plots
# ============================================================================
print()
print("Generating comparison figure...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor":   "#4a4a6a", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "text.color":  "#e0e0e0", "grid.color":  "#2a2a4a", "grid.alpha": 0.4,
    "axes.titlecolor": "#ffffff", "axes.titlesize": 11, "font.family": "monospace",
})
CYAN, ORANGE, GREEN, RED, YELLOW, PURPLE = ("#00d4ff", "#ff6b35",
                                              "#2ecc71", "#e74c3c",
                                              "#f1c40f", "#9b59b6")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Figure 7  |  Risk-Averse Portfolio vs Status Quo  "
             "(8 sigma scenarios, 400k MC paths)", fontsize=13)

# ---- TL: PnL distribution at sigma=2.51 ----
ax = axes[0, 0]
ax.set_title("PnL distribution at sigma=2.51 (model)", fontsize=11)
sq_pnl_251  = portfolio_pnl(PORTFOLIOS["A_status_quo"],     payoffs_by_sigma[2.51])
top_pnl_251 = portfolio_pnl(top_pos,                         payoffs_by_sigma[2.51])
lo, hi = np.percentile(sq_pnl_251, [0.5, 99.5])
ax.hist(np.clip(sq_pnl_251,  lo, hi), bins=120, color=ORANGE, alpha=0.55,
        density=True, label=f"A_status_quo  E={sq_pnl_251.mean():,.0f}")
ax.hist(np.clip(top_pnl_251, lo, hi), bins=120, color=CYAN,   alpha=0.55,
        density=True, label=f"{top_name}  E={top_pnl_251.mean():,.0f}")
ax.axvline(0, color="white", lw=0.8, alpha=0.4)
ax.axvline(np.percentile(sq_pnl_251, 5),  color=ORANGE, lw=1.4, ls="--",
           label=f"A VaR5%={np.percentile(sq_pnl_251,5):,.0f}")
ax.axvline(np.percentile(top_pnl_251, 5), color=CYAN,   lw=1.4, ls="--",
           label=f"{top_name} VaR5%={np.percentile(top_pnl_251,5):,.0f}")
ax.set_xlabel("Portfolio PnL ($)"); ax.set_ylabel("Density")
ax.legend(fontsize=8); ax.grid(True, lw=0.4)

# ---- TR: E[PnL] vs CVaR across portfolios ----
ax = axes[0, 1]
ax.set_title("Risk-reward map (each dot = 1 portfolio @ sigma=2.51)", fontsize=11)
labels, ex, cvarx = [], [], []
for n in PORTFOLIOS:
    s = results[n][2.51]
    labels.append(n); ex.append(s["E"]); cvarx.append(s["CVaR"])
ex = np.array(ex); cvarx = np.array(cvarx)
colors = [CYAN if l == top_name else ORANGE if l == "A_status_quo" else PURPLE for l in labels]
for (x, y, l, c) in zip(cvarx, ex, labels, colors):
    ax.scatter([x], [y], color=c, s=120, edgecolor="white", lw=0.8, zorder=5)
    ax.annotate(l, (x, y), fontsize=7, xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("CVaR 5%  (worst-case mean PnL)")
ax.set_ylabel("E[PnL] @ sigma=2.51")
ax.axvline(0, color="white", lw=0.8, alpha=0.3)
ax.axhline(0, color="white", lw=0.8, alpha=0.3)
ax.grid(True, lw=0.4)
ax.text(0.02, 0.98, "Up-and-right is better\n(more E, less negative CVaR)",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="#16213e", ec=YELLOW, alpha=0.8))

# ---- BL: E[PnL] vs sigma for top portfolios ----
ax = axes[1, 0]
ax.set_title("E[PnL] across vol regimes (vol-robustness check)", fontsize=11)
top_4 = [r[0] for r in ranked[:4]] + ["A_status_quo"]
sigs  = list(SIGMA_SCENARIOS)
palette = [CYAN, GREEN, YELLOW, PURPLE, ORANGE]
for c, n in zip(palette, top_4):
    es = [results[n][s]["E"] for s in sigs]
    ax.plot(sigs, es, color=c, lw=2, marker="o", ms=5, label=n)
ax.axhline(0, color="white", lw=0.8, alpha=0.4)
ax.axvline(2.51, color="white", lw=0.8, ls=":", alpha=0.6)
ax.axvline(2.47, color=YELLOW,  lw=0.8, ls=":", alpha=0.6, label="market-implied (2.47)")
ax.set_xlabel("True sigma"); ax.set_ylabel("E[PnL]")
ax.legend(fontsize=7); ax.grid(True, lw=0.4)

# ---- BR: CVaR across vol regimes ----
ax = axes[1, 1]
ax.set_title("CVaR 5% across vol regimes (tail-risk profile)", fontsize=11)
for c, n in zip(palette, top_4):
    cs = [results[n][s]["CVaR"] for s in sigs]
    ax.plot(sigs, cs, color=c, lw=2, marker="o", ms=5, label=n)
ax.axvline(2.51, color="white", lw=0.8, ls=":", alpha=0.6)
ax.axvline(2.47, color=YELLOW,  lw=0.8, ls=":", alpha=0.6)
ax.set_xlabel("True sigma"); ax.set_ylabel("CVaR 5%  (worst-case mean PnL)")
ax.legend(fontsize=7); ax.grid(True, lw=0.4)

plt.tight_layout()
fig.savefig("fig7_risk_averse_comparison.png", dpi=140, bbox_inches="tight")
print("  Saved fig7_risk_averse_comparison.png")
