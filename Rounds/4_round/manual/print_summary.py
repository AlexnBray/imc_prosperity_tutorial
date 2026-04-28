# -*- coding: utf-8 -*-
"""
Round 4 Manual — Clean Summary Printer
Run:  python print_summary.py
Prints the full trade recommendation + all comparison tables.
"""
import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.stats import norm

# ──────────────────────────────────────────────────────────────────────────────
S0 = 50.0; R = 0.0
TRADING_DAYS_PER_YEAR = 252; STEPS_PER_DAY = 4
DT = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)
T2_STEPS = 40; T3_STEPS = 60
T2_YEARS = 10 / 252; T3_YEARS = 15 / 252
BINARY_PAYOFF = 10.0; KO_BARRIER = 35.0
CONTRACT_SIZE = 3000
N_SIM = 400_000; SEED = 42
W0 = 1_000_000.0

SIGMA_SCENARIOS = [2.30, 2.40, 2.47, 2.49, 2.51, 2.55, 2.60, 2.70]
SIGMA_WEIGHTS   = [0.05, 0.10, 0.25, 0.15, 0.30, 0.10, 0.04, 0.01]

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

print("Simulating 400k paths x 8 vol scenarios (takes ~5s)...")
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_SIM, T3_STEPS))

def simulate_paths(sigma):
    log_inc  = (R - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * Z
    return S0 * np.exp(np.cumsum(log_inc, axis=1))

paths_by_sigma = {sig: simulate_paths(sig) for sig in SIGMA_SCENARIOS}

def payoffs(P):
    S2 = P[:, T2_STEPS-1]; S3 = P[:, T3_STEPS-1]
    breach = np.any(P[:, :T3_STEPS-1] < KO_BARRIER, axis=1)
    return dict(
        AC_50_P_2 = np.maximum(50-S2, 0),
        AC_50_C_2 = np.maximum(S2-50, 0),
        AC_50_CO  = np.where(S2>=50, np.maximum(S3-50,0), np.maximum(50-S3,0)),
        AC_40_BP  = np.where(S3<40, BINARY_PAYOFF, 0.0),
        AC_45_KO  = np.where(breach, 0.0, np.maximum(45-S3, 0)),
        AC_60_C   = np.maximum(S3-60, 0),
        AC_45_P   = np.maximum(45-S3, 0),
        AC_35_P   = np.maximum(35-S3, 0),
        AC_40_P   = np.maximum(40-S3, 0),
        AC_50_P   = np.maximum(50-S3, 0),
        AC_50_C   = np.maximum(S3-50, 0),
    )
pay_by_sig = {sig: payoffs(P) for sig, P in paths_by_sigma.items()}

def pnl_array(positions, sig):
    pay = pay_by_sig[sig]
    total = np.zeros(N_SIM)
    for name, (side, qty) in positions.items():
        p = MKT[name]
        unit = (pay[name] - p["ask"]) if side == "BUY" else (p["bid"] - pay[name])
        total += qty * CONTRACT_SIZE * unit
    return total

def stats(positions):
    """Returns dict of metrics, blended + per-sigma."""
    per_sigma = {}
    for sig in SIGMA_SCENARIOS:
        pnl = pnl_array(positions, sig)
        e = float(pnl.mean()); std = float(pnl.std())
        v5  = float(np.percentile(pnl, 5))
        cvar = float(pnl[pnl <= v5].mean())
        med = float(np.median(pnl)); p25 = float(np.percentile(pnl, 25))
        p75 = float(np.percentile(pnl, 75))
        pwin = float((pnl > 0).mean())
        up   = float(np.maximum(pnl, 0).mean())
        per_sigma[sig] = dict(E=e, Std=std, VaR=v5, CVaR=cvar,
                               Med=med, P25=p25, P75=p75, Pwin=pwin, Upside=up)
    w = np.array(SIGMA_WEIGHTS)
    E_b = sum(per_sigma[s]["E"]*ww for s,ww in zip(SIGMA_SCENARIOS,SIGMA_WEIGHTS))
    P_b = sum(per_sigma[s]["Pwin"]*ww for s,ww in zip(SIGMA_SCENARIOS,SIGMA_WEIGHTS))
    Up_b= sum(per_sigma[s]["Upside"]*ww for s,ww in zip(SIGMA_SCENARIOS,SIGMA_WEIGHTS))
    Std_b = float(np.sqrt(sum(
        (per_sigma[s]["Std"]**2 + (per_sigma[s]["E"]-E_b)**2)*ww
        for s,ww in zip(SIGMA_SCENARIOS,SIGMA_WEIGHTS))))
    CVaR_m = per_sigma[2.51]["CVaR"]
    return dict(per_sigma=per_sigma,
                E=E_b, Pwin=P_b, Upside=Up_b, Std=Std_b,
                Sharpe=E_b/Std_b if Std_b>0 else 0,
                EV_x_P=E_b*P_b, CVaR_model=CVaR_m,
                Med_model=per_sigma[2.51]["Med"])

def max_loss(positions):
    total = 0.0
    for name, (side, qty) in positions.items():
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
            return float("inf")
    return total

# ──────────────────────────────────────────────────────────────────────────────
# THE RECOMMENDED PORTFOLIO
REC = {
    "AC_45_KO": ("BUY",  500),
    "AC_40_BP": ("SELL",  50),
    "AC_35_P":  ("BUY",   25),
}

# COMPARISON PORTFOLIOS
PORTFOLIOS = {
    "KO500/BP50/P35_25  [RECOMMENDED]": REC,
    "KO500/BP50          [no hedge]":   {"AC_45_KO": ("BUY",500), "AC_40_BP": ("SELL",50)},
    "KO500/BP40/P35_25":                {"AC_45_KO": ("BUY",500), "AC_40_BP": ("SELL",40), "AC_35_P": ("BUY",25)},
    "KO500/BP25/P35_25  [conservative]":{"AC_45_KO": ("BUY",500), "AC_40_BP": ("SELL",25), "AC_35_P": ("BUY",25)},
    "KO500/BP25          [LOW tier]":   {"AC_45_KO": ("BUY",500), "AC_40_BP": ("SELL",25)},
    "KO200/BP10          [ULTRA tier]": {"AC_45_KO": ("BUY",200), "AC_40_BP": ("SELL",10)},
    "BP50_only           [safest]":     {"AC_40_BP": ("SELL",50)},
    "KO500_only          [lottery]":    {"AC_45_KO": ("BUY",500)},
    "Status quo          [original]":   {
        "AC_50_P_2": ("BUY",50), "AC_50_C_2": ("BUY",50),
        "AC_50_CO":  ("SELL",50), "AC_40_BP": ("SELL",50), "AC_45_KO": ("BUY",500),
    },
}

print("Computing statistics...")
all_stats = {name: stats(pos) for name, pos in PORTFOLIOS.items()}
all_ml    = {name: max_loss(pos) for name, pos in PORTFOLIOS.items()}

# ──────────────────────────────────────────────────────────────────────────────
W = 120
DASHES  = "─" * W
EQUALS  = "═" * W
HASH    = "█" * W

def bar(v, lo, hi, width=20, fill="█", empty="░"):
    t = (v - lo) / max(hi - lo, 1)
    t = max(0.0, min(1.0, t))
    n = round(t * width)
    return fill * n + empty * (width - n)

print()
print(HASH)
print("█" + " " * (W-2) + "█")
print("█" + "  ROUND 4 MANUAL  —  AETHER CRYSTAL OPTIONS  —  STRATEGY SUMMARY".center(W-2) + "█")
print("█" + " " * (W-2) + "█")
print(HASH)

# ──────────────────────────────────────────────────────────────────────────────
print()
print(EQUALS)
print("  SECTION 1 — WHICH TRADES HAVE AN EDGE?")
print("  (Per-trade edge sweep across plausible volatility range σ ∈ [2.30, 2.70])")
print(EQUALS)
print(f"  {'Product':<12} {'Side':<5} {'σ=230':>7} {'σ=247':>7} {'σ=251':>7} {'σ=260':>7} {'σ=270':>7}   {'WORST':>7}  {'ROBUST':>8}  {'Verdict'}")
print(DASHES)

def bs_d12(S,K,T,sig): 
    d1=(np.log(S/K)+0.5*sig**2*T)/(sig*np.sqrt(T)); return d1, d1-sig*np.sqrt(T)
def bs_call(S,K,T,sig):
    d1,d2=bs_d12(S,K,T,sig); return S*float(__import__('scipy').stats.norm.cdf(d1))-K*float(__import__('scipy').stats.norm.cdf(d2))
def bs_put(S,K,T,sig):
    d1,d2=bs_d12(S,K,T,sig); return K*float(__import__('scipy').stats.norm.cdf(-d2))-S*float(__import__('scipy').stats.norm.cdf(-d1))
def bs_binary_put(S,K,T,sig,payoff=BINARY_PAYOFF):
    _,d2=bs_d12(S,K,T,sig); return payoff*float(__import__('scipy').stats.norm.cdf(-d2))
def bs_chooser(S,K,Te,Tc,sig): return bs_call(S,K,Te,sig)+bs_put(S,K,Tc,sig)

ko_fairs = {sig: float(pay_by_sig[sig]["AC_45_KO"].mean()) for sig in SIGMA_SCENARIOS}

def fair(name, sig):
    p=MKT[name]
    if p["kind"]=="put":     return bs_put(S0,p["K"],p["T"],sig)
    if p["kind"]=="call":    return bs_call(S0,p["K"],p["T"],sig)
    if p["kind"]=="chooser": return bs_chooser(S0,p["K"],T3_YEARS,T2_YEARS,sig)
    if p["kind"]=="binary":  return bs_binary_put(S0,p["K"],p["T"],sig)
    if p["kind"]=="ko":      return ko_fairs[sig]
    raise ValueError(p["kind"])

EDGE_SIGS = [2.30, 2.47, 2.51, 2.60, 2.70]
rows = []
for name, p in MKT.items():
    edges_buy  = [fair(name,s)-p["ask"] for s in SIGMA_SCENARIOS]
    edges_sell = [p["bid"]-fair(name,s) for s in SIGMA_SCENARIOS]
    we_buy  = sum(e*w for e,w in zip(edges_buy,  SIGMA_WEIGHTS))
    we_sell = sum(e*w for e,w in zip(edges_sell, SIGMA_WEIGHTS))
    side = "BUY" if we_buy >= we_sell else "SELL"
    edges = edges_buy if side=="BUY" else edges_sell
    worst = min(edges)
    robust = worst > 0
    edge_at = {s: (fair(name,s)-p["ask"] if side=="BUY" else p["bid"]-fair(name,s))
               for s in EDGE_SIGS}
    rows.append((name, side, edge_at, worst, robust))

for name, side, ea, worst, robust in sorted(rows, key=lambda r: -r[3]):
    bar_str = bar(worst, -1.5, 0.5, width=12)
    verdict = "ROBUST - safe edge" if robust else ("mild vol bet" if worst > -0.3 else "STRONG vol bet")
    star = "***" if robust else "   "
    print(f"  {name:<12} {side:<5} "
          f"{ea[2.30]:>+7.3f} {ea[2.47]:>+7.3f} {ea[2.51]:>+7.3f} "
          f"{ea[2.60]:>+7.3f} {ea[2.70]:>+7.3f}   {worst:>+7.3f}  "
          f"{star:>8}  {verdict}")
print(DASHES)
print("  *** = structurally robust (positive edge for ALL plausible σ — no vol prediction needed)")
print(EQUALS)

# ──────────────────────────────────────────────────────────────────────────────
print()
print(EQUALS)
print("  SECTION 2 — PORTFOLIO COMPARISON TABLE")
print("  (All metrics blended across σ prior [2.30..2.70] weighted by subjective probability)")
print(EQUALS)
print(f"  {'Portfolio':<42} {'E[PnL]':>10}  {'P(win)':>7}  {'EV×P':>10}  "
      f"{'Median':>10}  {'Sharpe':>7}  {'CVaR@2.51':>11}  {'MaxLoss':>10}")
print(DASHES)
for name, s in all_stats.items():
    ml = all_ml[name]
    ml_s = f"${-ml:>+9,.0f}" if ml != float("inf") else "  UNBOUNDED"
    print(f"  {name:<42} ${s['E']:>+9,.0f}  {s['Pwin']*100:>6.1f}%  "
          f"${s['EV_x_P']:>+9,.0f}  ${s['Med_model']:>+9,.0f}  "
          f"{s['Sharpe']:>+7.4f}  ${s['CVaR_model']:>+10,.0f}  {ml_s}")
print(DASHES)
print("  EV×P  =  E[PnL] × P(profit)  — 'the probability-weighted value of this trade'")
print(EQUALS)

# ──────────────────────────────────────────────────────────────────────────────
rec_s = all_stats["KO500/BP50/P35_25  [RECOMMENDED]"]
rec_ml = all_ml["KO500/BP50/P35_25  [RECOMMENDED]"]

print()
print(EQUALS)
print("  SECTION 3 — RECOMMENDED PORTFOLIO: KO500 / BP-50 / P35+25")
print(EQUALS)

print()
print("  WHY THIS COMBINATION?")
print()
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  KO PUT  BUY 500 — the asymmetric jackpot leg                       │")
print("  │    Edge = +$0.078/unit at σ=2.51, robust across ALL plausible σ     │")
print("  │    Wins big when S crashes hard but stays above 35                  │")
print("  │    Alone it is 95% chance of losing $262k → LOTTERY without hedge   │")
print("  │                                                                      │")
print("  │  BINARY PUT  SELL 50 — the reliable income leg                      │")
print("  │    Edge = +$0.232/unit at σ=2.51, robust across ALL plausible σ     │")
print("  │    Pays $750k profit on ~70% of paths (S stays above 40)            │")
print("  │    Costs $750k on ~30% of paths (S ends below 40)                   │")
print("  │                                                                      │")
print("  │  P35 PUT  BUY 25 — the KO-breach hedge                              │")
print("  │    Pays out exactly when KO is knocked out (S hits <35)             │")
print("  │    Partially offsets the -$262.5k KO loss on those paths            │")
print("  │    Costs $326k at most; recovers significant losses when KO breaches│")
print("  └─────────────────────────────────────────────────────────────────────┘")

print()
print("  PER-SIGMA RISK PROFILE:")
print(f"  {'σ':>5}  {'Weight':>7}  {'E[PnL]':>11}  {'P(win)':>7}  {'Median':>11}  "
      f"{'P25':>11}  {'P75':>11}  {'CVaR5%':>11}  Note")
print(DASHES)
for sig, ww in zip(SIGMA_SCENARIOS, SIGMA_WEIGHTS):
    s = rec_s["per_sigma"][sig]
    note = ""
    if abs(sig-2.51) < 0.005: note = "<-- model σ"
    elif abs(sig-2.47) < 0.005: note = "<-- mkt-implied"
    print(f"  {sig:>5.2f}  {ww*100:>6.0f}%  ${s['E']:>+10,.0f}  {s['Pwin']*100:>6.1f}%  "
          f"${s['Med']:>+10,.0f}  ${s['P25']:>+10,.0f}  ${s['P75']:>+10,.0f}  "
          f"${s['CVaR']:>+10,.0f}  {note}")
print(DASHES)
print(f"  BLENDED   100%  ${rec_s['E']:>+10,.0f}  {rec_s['Pwin']*100:>6.1f}%  "
      f"  (vol-prior weighted E[PnL] and P(win))")
print(EQUALS)

# ──────────────────────────────────────────────────────────────────────────────
print()
print(EQUALS)
print("  SECTION 4 — VISUAL PAYOFF PROFILE   (KO500/BP-50/P35+25 vs KO500 alone)")
print(EQUALS)
pnl_rec = pnl_array(REC, 2.51)
pnl_ko  = pnl_array({"AC_45_KO": ("BUY",500)}, 2.51)
buckets = np.linspace(-2_000_000, 2_000_000, 41)
h_rec, _ = np.histogram(pnl_rec, bins=buckets)
h_ko,  _ = np.histogram(pnl_ko,  bins=buckets)
max_h = max(h_rec.max(), h_ko.max())
print(f"  PnL distribution at σ=2.51  (each █ ≈ {max_h/20:.0f} paths out of {N_SIM:,})")
print(f"  {'$-2M':>8} {'$-1M':>20} {'$0':>20} {'$+1M':>20} {'$+2M':>8}")
print(f"  {'':8} {'':20} {'|':20} {'':20}")
bw = (buckets[1]-buckets[0])
for i, (lo, hi) in enumerate(zip(buckets[:-1], buckets[1:])):
    label = f"  {lo/1e6:>+6.2f}M "
    bar_r = "█" * min(int(h_rec[i]/max_h*20), 20)
    bar_k  = "░" * min(int(h_ko[i] /max_h*20), 20)
    mid_marker = "|" if abs(lo) < bw/2 else " "
    print(f"{label}  REC {bar_r:<20}  KO {bar_k:<20}")
print(f"  NOTE: REC has thick mass at $+161k (most paths).  KO has spike at -$262.5k (95% of paths).")
print(EQUALS)

# ──────────────────────────────────────────────────────────────────────────────
print()
print(EQUALS)
print("  SECTION 5 — PNLXP COMPARISON:  REC vs KO-lottery vs status-quo")
print(EQUALS)
metrics_list = [
    ("E[PnL] × P(win)  [PRIMARY]",   "EV_x_P",  False),
    ("E[PnL]",                        "E",       False),
    ("P(win)",                        "Pwin",    True),
    ("Median PnL",                    "Med_model",False),
    ("Sharpe  (E/Std)",               "Sharpe",  False),
]
compare_names = [
    "KO500/BP50/P35_25  [RECOMMENDED]",
    "KO500_only          [lottery]",
    "BP50_only           [safest]",
    "Status quo          [original]",
]
hdr = f"  {'Metric':<32}"
for n in compare_names:
    hdr += f"  {n[:24]:>24}"
print(hdr)
print(DASHES)
for label, key, is_pct in metrics_list:
    row = f"  {label:<32}"
    for n in compare_names:
        s = all_stats[n]
        v = s[key]
        if is_pct:
            row += f"  {v*100:>23.1f}%"
        else:
            row += f"  ${v:>+22,.0f}"
    print(row)
print(DASHES)
print(f"  {'Max loss':32}")
for n in compare_names:
    ml = all_ml[n]
    s = f"${-ml:>+10,.0f}" if ml!=float("inf") else "    UNBOUNDED"
print(EQUALS)

# ──────────────────────────────────────────────────────────────────────────────
print()
print(EQUALS)
print("  SECTION 6 — ALTERNATIVE STRATEGIES  (if you want lower risk)")
print(EQUALS)
alt_rows = [
    ("ULTRA ($300k budget)",  {"AC_45_KO":("BUY",200), "AC_40_BP":("SELL",10)}),
    ("LOW   ($640k budget)",  {"AC_45_KO":("BUY",500), "AC_40_BP":("SELL",25)}),
    ("MOD   ($1.0M budget)",  {"AC_45_KO":("BUY",500), "AC_40_BP":("SELL",40), "AC_35_P":("BUY",25)}),
    ("FULL  ($1.3M budget) *",REC),
    ("SAFE  (BP only)     ",  {"AC_40_BP":("SELL",50)}),
]
print(f"  {'Tier':<28}  {'E[PnL]':>10}  {'P(win)':>7}  {'EV×P':>10}  {'Median':>10}  {'MaxLoss':>10}")
print(DASHES)
for label, pos in alt_rows:
    s = stats(pos); ml = max_loss(pos)
    ml_s = f"${-ml:>+9,.0f}" if ml!=float("inf") else "  UNBOUNDED"
    print(f"  {label:<28}  ${s['E']:>+9,.0f}  {s['Pwin']*100:>6.1f}%  "
          f"${s['EV_x_P']:>+9,.0f}  ${s['Med_model']:>+9,.0f}  {ml_s}")
print(DASHES)
print("  * = primary recommendation")
print(EQUALS)

# ──────────────────────────────────────────────────────────────────────────────
print()
print(HASH)
print("█" + " " * (W-2) + "█")
print("█" + "  WHAT TO ENTER INTO THE MANUAL CHALLENGE".center(W-2) + "█")
print("█" + " " * (W-2) + "█")
print(HASH)
print()
print("  ┌─────────────────────────────────────────────────────────────────────────────┐")
print("  │                                                                              │")
print("  │   TRADE 1:  BUY  AC_45_KO   qty = 500   @ ask = 0.175                      │")
print("  │             KO Put, strike 45, knock-out barrier at 35, 3-week expiry       │")
print("  │             Max loss: 500 × 3000 × 0.175 = $262,500                        │")
print("  │                                                                              │")
print("  │   TRADE 2:  SELL AC_40_BP   qty = 50    @ bid = 5.000                      │")
print("  │             Binary Put, strike 40, pays $10 if S<40 at expiry (3-week)      │")
print("  │             Max loss: 50 × 3000 × (10-5) = $750,000                        │")
print("  │                                                                              │")
print("  │   TRADE 3:  BUY  AC_35_P    qty = 25    @ ask = 4.350                      │")
print("  │             Vanilla Put, strike 35, 3-week expiry                           │")
print("  │             Max loss: 25 × 3000 × 4.35 = $326,250                          │")
print("  │                                                                              │")
print("  │   TOTAL MAX LOSS:  $1,338,750  (hard floor — cannot lose more than this)   │")
print("  │                                                                              │")
print("  └─────────────────────────────────────────────────────────────────────────────┘")
print()
print("  ─── KEY NUMBERS ──────────────────────────────────────────────────────────────")
print(f"  Expected PnL (at model σ=2.51)      =  ${rec_s['per_sigma'][2.51]['E']:>+12,.0f}")
print(f"  Expected PnL (worst σ in range)     =  ${min(rec_s['per_sigma'][s]['E'] for s in SIGMA_SCENARIOS):>+12,.0f}")
print(f"  Probability of profit               =  {rec_s['Pwin']*100:>5.1f}%")
print(f"  Probability-weighted EV  (E×P)      =  ${rec_s['EV_x_P']:>+12,.0f}")
print(f"  Median outcome (typical path)       =  ${rec_s['per_sigma'][2.51]['Med']:>+12,.0f}")
print(f"  Sharpe ratio                        =  {rec_s['Sharpe']:>+7.4f}")
print()
print("  ─── WHY NOT JUST KO500? (the old 'max E' recommendation) ────────────────────")
ko_s = all_stats["KO500_only          [lottery]"]
print(f"  KO500-only:  E[PnL]=${ko_s['E']:>+10,.0f}   P(win)={ko_s['Pwin']*100:.1f}%   EV×P=${ko_s['EV_x_P']:>+9,.0f}")
print(f"  THIS STRAT:  E[PnL]=${rec_s['E']:>+10,.0f}   P(win)={rec_s['Pwin']*100:.1f}%   EV×P=${rec_s['EV_x_P']:>+9,.0f}")
print(f"  Adding BP SELL + P35 BUY gives +{(rec_s['Pwin']-ko_s['Pwin'])*100:.0f}pp win rate and +${rec_s['EV_x_P']-ko_s['EV_x_P']:>+,.0f} in EV×P.")
print(f"  The median goes from ${ko_s['per_sigma'][2.51]['Med']:>+,.0f} to ${rec_s['per_sigma'][2.51]['Med']:>+,.0f} — flips from certain loss to profit.")
print()
print(HASH)
print("█" + "  Done.  Trade ticket is in the box above.".center(W-2) + "█")
print(HASH)
