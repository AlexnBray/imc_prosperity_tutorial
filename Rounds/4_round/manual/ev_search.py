# -*- coding: utf-8 -*-
"""
Round 4 Manual: EXHAUSTIVE EV×P(win) MAXIMISER
================================================
Searches ALL products in BOTH directions and ALL size combinations.
Ranks purely by E[PnL] × P(profit) — the user's stated metric.

Key question: Is KO BUY helping or hurting EV×P?
We answer this by including KO SELL, vanilla put/call SELLs, chooser,
and letting the grid tell us the truth.
"""
import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.stats import norm
import time

S0 = 50.0; R = 0.0
TRADING_DAYS_PER_YEAR = 252; STEPS_PER_DAY = 4
DT = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)
T2_STEPS = 40; T3_STEPS = 60
T2_YEARS = 10 / 252; T3_YEARS = 15 / 252
BINARY_PAYOFF = 10.0; KO_BARRIER = 35.0
CONTRACT_SIZE = 3000
N_SIM = 300_000; SEED = 42

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

# ── Simulate ──────────────────────────────────────────────────────────────────
print(f"Simulating {N_SIM:,} paths x {len(SIGMA_SCENARIOS)} vol scenarios...")
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
print("  done.")

# ── BS fair values for edge table ─────────────────────────────────────────────
def bs_d12(S,K,T,sig):
    d1=(np.log(S/K)+0.5*sig**2*T)/(sig*np.sqrt(T)); return d1,d1-sig*np.sqrt(T)
def bs_call(S,K,T,sig):
    d1,d2=bs_d12(S,K,T,sig)
    return S*float(norm.cdf(d1))-K*float(norm.cdf(d2))
def bs_put(S,K,T,sig):
    d1,d2=bs_d12(S,K,T,sig)
    return K*float(norm.cdf(-d2))-S*float(norm.cdf(-d1))
def bs_binary_put(S,K,T,sig):
    _,d2=bs_d12(S,K,T,sig); return BINARY_PAYOFF*float(norm.cdf(-d2))
def bs_chooser(S,K,Te,Tc,sig): return bs_call(S,K,Te,sig)+bs_put(S,K,Tc,sig)
ko_fairs = {sig: float(pay_by_sig[sig]["AC_45_KO"].mean()) for sig in SIGMA_SCENARIOS}

def fair(name, sig):
    p=MKT[name]
    if p["kind"]=="put":     return bs_put(S0,p["K"],p["T"],sig)
    if p["kind"]=="call":    return bs_call(S0,p["K"],p["T"],sig)
    if p["kind"]=="chooser": return bs_chooser(S0,p["K"],T3_YEARS,T2_YEARS,sig)
    if p["kind"]=="binary":  return bs_binary_put(S0,p["K"],p["T"],sig)
    if p["kind"]=="ko":      return ko_fairs[sig]

# ── Portfolio PnL ─────────────────────────────────────────────────────────────
def pnl_sigma(positions, sig):
    """positions: dict(name -> (side, qty))"""
    pay = pay_by_sig[sig]
    total = np.zeros(N_SIM)
    for name, (side, qty) in positions.items():
        p = MKT[name]
        unit = (pay[name] - p["ask"]) if side == "BUY" else (p["bid"] - pay[name])
        total += qty * CONTRACT_SIZE * unit
    return total

def ev_metrics(positions):
    w = np.array(SIGMA_WEIGHTS)
    E_s = []; Pw_s = []; Up_s = []; Std_s = []
    for sig in SIGMA_SCENARIOS:
        pnl = pnl_sigma(positions, sig)
        E_s.append(float(pnl.mean()))
        Pw_s.append(float((pnl > 0).mean()))
        Up_s.append(float(np.maximum(pnl, 0).mean()))
        Std_s.append(float(pnl.std()))
    E_b  = float(np.dot(w, E_s))
    Pw_b = float(np.dot(w, Pw_s))
    Up_b = float(np.dot(w, Up_s))
    Std_b = float(np.sqrt(np.dot(w, np.array(Std_s)**2 + (np.array(E_s)-E_b)**2)))
    # Model sigma stats for CVaR / median
    pnl_m = pnl_sigma(positions, 2.51)
    v5    = float(np.percentile(pnl_m, 5))
    cvar  = float(pnl_m[pnl_m <= v5].mean())
    med   = float(np.median(pnl_m))
    p25   = float(np.percentile(pnl_m, 25))
    return dict(
        E=E_b, Pwin=Pw_b, Upside=Up_b, Std=Std_b,
        EVxP = E_b * Pw_b,
        Sharpe = E_b/Std_b if Std_b > 0 else 0.0,
        CVaR=cvar, Median=med, P25=p25,
        minE=min(E_s),
    )

def max_loss(positions):
    total = 0.0
    for name, (side, qty) in positions.items():
        p = MKT[name]
        if side == "BUY":
            total += qty * CONTRACT_SIZE * p["ask"]
        elif p["kind"] == "binary" and side == "SELL":
            total += qty * CONTRACT_SIZE * (BINARY_PAYOFF - p["bid"])
        elif p["kind"] in ("put","ko") and side == "SELL":
            total += qty * CONTRACT_SIZE * (p["K"] - p["bid"])
        elif p["kind"] == "call" and side == "SELL":
            return float("inf")  # short call unbounded
        elif p["kind"] == "chooser" and side == "SELL":
            return float("inf")
    return total

# ── SINGLE-PRODUCT STANDALONE TABLE ──────────────────────────────────────────
W = 126
print()
print("=" * W)
print("  TABLE A — STANDALONE SINGLE-PRODUCT STATISTICS  (at max position size)")
print(f"  Each product shown in its POSITIVE-EDGE direction, then the OPPOSITE direction.")
print("-" * W)
print(f"  {'Product':<12} {'Side':<5} {'Qty':>4}  "
      f"{'E[PnL]':>11}  {'P(win)':>7}  {'EV x P':>11}  "
      f"{'Median':>11}  {'CVaR5':>11}  {'MaxLoss':>11}  {'Edge?'}")
print("-" * W)

for name, p in MKT.items():
    for side in ("BUY","SELL"):
        qty = p["size"]
        pos = {name: (side, qty)}
        ml = max_loss(pos)
        if ml == float("inf"):
            ml_s = "  UNBOUNDED"
        else:
            ml_s = f"${-ml:>+10,.0f}"
        m = ev_metrics(pos)
        # Edge check
        edges = [(fair(name,s)-p["ask"] if side=="BUY" else p["bid"]-fair(name,s))
                 for s in SIGMA_SCENARIOS]
        worst_e = min(edges)
        edge_tag = "ROBUST ***" if worst_e > 0 else (f"vol-bet  (worst={worst_e:+.3f})")
        print(f"  {name:<12} {side:<5} {qty:>4}  "
              f"${m['E']:>+10,.0f}  {m['Pwin']*100:>6.1f}%  "
              f"${m['EVxP']:>+10,.0f}  ${m['Median']:>+10,.0f}  "
              f"${m['CVaR']:>+10,.0f}  {ml_s}  {edge_tag}")
    print("-" * W)
print("=" * W)
print("  *** = structurally robust edge (positive across ALL plausible vol scenarios)")
print("  EVxP = E[PnL] x P(profit)  -- the primary optimisation target")

# ── GRID SEARCH — focused, manageable combos ─────────────────────────────────
# Build a grid of interesting 2-3 leg portfolios.
# KO both BUY and SELL; BP SELL; vanilla put SELLs; P35/P40 BUY hedge
print()
print(f"Building candidate grid...")

candidates = []

def add(pos):
    ml = max_loss(pos)
    if ml == float("inf"): return   # skip unbounded shorts
    if ml > 3_000_000:     return   # skip unacceptably large risk
    label = "  ".join(f"{'+'if s=='BUY' else '-'}{q}{n[3:]}"
                      for n,(s,q) in sorted(pos.items()))
    candidates.append(dict(pos=pos, ml=ml, label=label))

# Single leg
for name in ["AC_40_BP","AC_45_KO","AC_35_P","AC_40_P","AC_45_P","AC_50_P","AC_60_C"]:
    for side in ("BUY","SELL"):
        for qty_frac in [0.25, 0.5, 1.0]:
            qty = max(1, int(MKT[name]["size"] * qty_frac))
            add({name: (side, qty)})

# Two-leg: BP SELL × (something)
for q_bp in [10, 25, 40, 50]:
    for name2 in ["AC_45_KO","AC_35_P","AC_40_P","AC_45_P","AC_50_P"]:
        for side2 in ("BUY","SELL"):
            for qty_frac in [0.25, 0.5, 1.0]:
                qty2 = max(1, int(MKT[name2]["size"] * qty_frac))
                add({"AC_40_BP": ("SELL", q_bp), name2: (side2, qty2)})

# Two-leg: KO both sides × BP SELL
for q_ko in [50, 100, 200, 300, 500]:
    for ko_side in ("BUY","SELL"):
        for q_bp in [10, 25, 50]:
            add({"AC_45_KO": (ko_side, q_ko), "AC_40_BP": ("SELL", q_bp)})

# Three-leg: BP SELL + KO + hedge
for q_bp in [25, 50]:
    for q_ko in [50, 100, 200, 300, 500]:
        for ko_side in ("BUY","SELL"):
            for hedge_name in ["AC_35_P","AC_40_P"]:
                for hqty in [10, 25, 50]:
                    for hside in ("BUY","SELL"):
                        add({
                            "AC_40_BP": ("SELL", q_bp),
                            "AC_45_KO": (ko_side, q_ko),
                            hedge_name: (hside, hqty),
                        })

# Three-leg: BP SELL + two vanilla put SELLs
for q_bp in [25, 50]:
    for name_a in ["AC_45_P","AC_40_P","AC_35_P"]:
        for q_a in [10, 25, 50]:
            for name_b in ["AC_45_P","AC_40_P","AC_35_P"]:
                if name_b <= name_a: continue
                for q_b in [10, 25, 50]:
                    add({
                        "AC_40_BP": ("SELL", q_bp),
                        name_a:     ("SELL", q_a),
                        name_b:     ("SELL", q_b),
                    })

# BP SELL + vanilla put SELL (two-leg)
for q_bp in [10, 25, 50]:
    for name2 in ["AC_45_P","AC_40_P","AC_35_P","AC_50_P"]:
        for q2 in [10, 25, 50]:
            add({"AC_40_BP": ("SELL", q_bp), name2: ("SELL", q2)})

# KO SELL + vanilla put BUY (hedge)
for q_ko in [50, 100, 200, 300, 500]:
    for hedge in ["AC_35_P","AC_40_P","AC_45_P"]:
        for qh in [10, 25, 50]:
            add({"AC_45_KO": ("SELL", q_ko), hedge: ("BUY", qh)})

# Pure vanilla combos (no KO)
for n_a in ["AC_45_P","AC_40_P","AC_35_P","AC_60_C"]:
    for n_b in ["AC_45_P","AC_40_P","AC_35_P","AC_60_C","AC_40_BP"]:
        if n_b <= n_a: continue
        for qa in [10, 25, 50]:
            for qb in [10, 25, 50]:
                for sa in ("BUY","SELL"):
                    for sb in ("BUY","SELL"):
                        add({n_a: (sa, qa), n_b: (sb, qb)})

# Dedup
seen = set()
unique = []
for c in candidates:
    key = tuple(sorted((n,s,q) for n,(s,q) in c["pos"].items()))
    if key not in seen:
        seen.add(key); unique.append(c)
candidates = unique

print(f"  {len(candidates):,} unique portfolios to evaluate.")
print(f"Computing EV×P for all candidates...")
t0 = time.time()
for i, c in enumerate(candidates):
    c["m"] = ev_metrics(c["pos"])
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(candidates)}  {time.time()-t0:.1f}s")
print(f"  done in {time.time()-t0:.1f}s")

# ── DISPLAY RESULTS ───────────────────────────────────────────────────────────
def top(key, n=10, budget=None, filt=None):
    pool = [c for c in candidates
            if (budget is None or c["ml"] <= budget)
            and (filt is None or filt(c))]
    pool.sort(key=lambda c: c["m"][key], reverse=True)
    return pool[:n]

def print_table(title, rows):
    print()
    print("=" * W)
    print(f"  {title}")
    print("-" * W)
    print(f"  {'Portfolio':<42}  {'EVxP':>10}  {'E[PnL]':>10}  {'P(win)':>7}  "
          f"{'Median':>10}  {'CVaR5':>10}  {'Sharpe':>7}  {'MaxLoss':>11}")
    print("-" * W)
    for c in rows:
        m = c["m"]; ml = c["ml"]
        print(f"  {c['label']:<42}  ${m['EVxP']:>+9,.0f}  ${m['E']:>+9,.0f}  "
              f"{m['Pwin']*100:>6.1f}%  ${m['Median']:>+9,.0f}  ${m['CVaR']:>+9,.0f}  "
              f"{m['Sharpe']:>+7.4f}  ${-ml:>+10,.0f}")
    print("=" * W)

print_table("TOP 10 BY EV×P  —  all budgets  (no limit)", top("EVxP", 10))
print_table("TOP 10 BY EV×P  —  max loss <= $300k", top("EVxP", 10, budget=300_000))
print_table("TOP 10 BY EV×P  —  max loss <= $750k", top("EVxP", 10, budget=750_000))
print_table("TOP 10 BY EV×P  —  max loss <= $1.5M", top("EVxP", 10, budget=1_500_000))
print_table("TOP 10 BY P(win)  —  max loss <= $1.5M", top("Pwin", 10, budget=1_500_000))
print_table("TOP 10 BY Median PnL  —  max loss <= $1.5M", top("Median", 10, budget=1_500_000))
print_table("TOP 10 BY Sharpe  —  max loss <= $1.5M", top("Sharpe", 10, budget=1_500_000))

# ── NO-KO-BUY section ──────────────────────────────────────────────────────────
no_ko_buy = lambda c: "AC_45_KO" not in c["pos"] or c["pos"]["AC_45_KO"][0] != "BUY"
print_table("TOP 10 BY EV×P  —  NO KO BUY  (by user request)  <=  $1.5M",
            top("EVxP", 10, budget=1_500_000, filt=no_ko_buy))
print_table("TOP 10 BY EV×P  —  NO KO BUY  <=  $750k",
            top("EVxP", 10, budget=750_000, filt=no_ko_buy))

# ── FINAL VERDICT ─────────────────────────────────────────────────────────────
best_overall = top("EVxP", 1)[0]
best_no_ko   = top("EVxP", 1, filt=no_ko_buy)[0]
best_300k    = top("EVxP", 1, budget=300_000)[0]
best_750k    = top("EVxP", 1, budget=750_000)[0]

print()
print("=" * W)
print("  VERDICT: WHICH IS BETTER — KO BUY or NO KO BUY?")
print("-" * W)
SPEC = {
    "AC_40_BP":  "Binary Put SELL,  K=40, payoff 10 if S<40",
    "AC_45_KO":  "KO Put BUY,       K=45, KO if S<35 ever",
    "AC_35_P":   "Vanilla Put BUY,  K=35",
    "AC_40_P":   "Vanilla Put BUY/SELL K=40",
    "AC_45_P":   "Vanilla Put SELL, K=45",
    "AC_50_P":   "Vanilla Put SELL, K=50",
    "AC_60_C":   "Vanilla Call SELL K=60",
}
for label, port in [
    ("BEST WITH KO BUY (all budgets)", best_overall),
    ("BEST WITHOUT KO BUY (all budgets)", best_no_ko),
    ("BEST <= $300k", best_300k),
    ("BEST <= $750k", best_750k),
]:
    m = port["m"]
    print(f"\n  [{label}]")
    for name, (side, qty) in sorted(port["pos"].items()):
        p = MKT[name]; px = p["ask"] if side=="BUY" else p["bid"]
        e_side = [(fair(name,s)-p["ask"] if side=="BUY" else p["bid"]-fair(name,s))
                  for s in SIGMA_SCENARIOS]
        worst_e = min(e_side)
        rob = "ROBUST" if worst_e > 0 else f"vol-bet(worst={worst_e:+.3f})"
        print(f"    {side:<4} {qty:>3} x {name:<10} @ ${px:<6.3f}  [{rob}]")
    print(f"    EV×P = ${m['EVxP']:>+10,.0f}   E[PnL] = ${m['E']:>+10,.0f}   "
          f"P(win) = {m['Pwin']*100:.1f}%   Median = ${m['Median']:>+10,.0f}"
          f"   MaxLoss = ${-port['ml']:>+10,.0f}")
print()
print("=" * W)
print("  INSIGHT: Does KO BUY increase or decrease EV×P?")
print(f"  Best WITH    KO BUY:  EV×P = ${best_overall['m']['EVxP']:>+,.0f}")
print(f"  Best WITHOUT KO BUY:  EV×P = ${best_no_ko['m']['EVxP']:>+,.0f}")
diff = best_overall["m"]["EVxP"] - best_no_ko["m"]["EVxP"]
if diff > 0:
    print(f"  -> KO BUY adds ${diff:>+,.0f} to EV×P.  Keep KO BUY.")
else:
    print(f"  -> Without KO BUY is BETTER by ${-diff:>+,.0f} in EV×P.  Drop or sell KO.")
print("=" * W)
print("\nDone.")
