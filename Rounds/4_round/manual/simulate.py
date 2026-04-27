"""
Round 4 Manual Challenge: Aether Crystal Options Simulator
GBM: zero drift, sigma=251%, 4 steps/day, 252 trading days/year
All options written on AC with S0 = 50
"""

import numpy as np
from scipy.stats import norm

# --- Simulation Parameters ---------------------------------------------------
S0 = 50.0
SIGMA = 2.51          # 251% annualised vol
R = 0.0               # zero risk-neutral drift
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY
DT = 1.0 / STEPS_PER_YEAR

T2_STEPS = 2 * 5 * STEPS_PER_DAY   # 40  (2 weeks)
T3_STEPS = 3 * 5 * STEPS_PER_DAY   # 60  (3 weeks)
T2_YEARS = (2 * 5) / TRADING_DAYS_PER_YEAR   # 10/252
T3_YEARS = (3 * 5) / TRADING_DAYS_PER_YEAR   # 15/252
T1_YEARS = (1 * 5) / TRADING_DAYS_PER_YEAR   # 5/252  (1 week remaining after choice)

N_SIM = 500_000
SEED = 42

# Confirmed from competition rules
BINARY_PAYOFF = 10.0   # AC_40_BP: fixed cash payoff if S_T < 40 at expiry
KO_BARRIER = 35.0      # AC_45_KO: option knocked out if price ever falls below 35

# --- Black-Scholes Analytical ------------------------------------------------
def bs_d1d2(S, K, T, sigma, r=0.0):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_call(S, K, T, sigma, r=0.0):
    if T <= 0:
        return float(max(S - K, 0))
    d1, d2 = bs_d1d2(S, K, T, sigma, r)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, sigma, r=0.0):
    if T <= 0:
        return float(max(K - S, 0))
    d1, d2 = bs_d1d2(S, K, T, sigma, r)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_binary_put(S, K, T, sigma, r=0.0, payoff=1.0):
    """Cash-or-nothing put: pays `payoff` if S_T < K."""
    if T <= 0:
        return float(payoff if S < K else 0)
    _, d2 = bs_d1d2(S, K, T, sigma, r)
    return payoff * np.exp(-r * T) * norm.cdf(-d2)

def bs_chooser(S, K, T_expiry, T_choice, sigma, r=0.0):
    """
    Simple chooser: at T_choice buyer picks call or put, expiring at T_expiry.
    Formula (r=0 friendly):  V = C(T_expiry) + P(T_choice)
    Derivation: at T_choice, value = max(C_tc, P_tc) = C_tc + max(0, K - S_tc)
    which discounts back to C(T_expiry) + P(T_choice).
    """
    return bs_call(S, K, T_expiry, sigma, r) + bs_put(S, K, T_choice, sigma, r)

# --- Monte Carlo Simulation ---------------------------------------------------
print(f"Simulating {N_SIM:,} paths x {T3_STEPS} steps...")
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_SIM, T3_STEPS))
log_increments = (R - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * Z
log_paths = np.cumsum(log_increments, axis=1)
prices = S0 * np.exp(log_paths)          # shape (N_SIM, T3_STEPS)

S_2wk = prices[:, T2_STEPS - 1]          # price at step 40
S_3wk = prices[:, T3_STEPS - 1]          # price at step 60

# --- Payoffs ------------------------------------------------------------------

# Vanilla 3-week
p_ac_50_P = np.maximum(50 - S_3wk, 0)
p_ac_50_C = np.maximum(S_3wk - 50, 0)
p_ac_35_P = np.maximum(35 - S_3wk, 0)
p_ac_40_P = np.maximum(40 - S_3wk, 0)
p_ac_45_P = np.maximum(45 - S_3wk, 0)
p_ac_60_C = np.maximum(S_3wk - 60, 0)

# Vanilla 2-week
p_ac_50_P2 = np.maximum(50 - S_2wk, 0)
p_ac_50_C2 = np.maximum(S_2wk - 50, 0)

# Chooser (T+14/21, K=50):
#   at week 2 choose call if S_2wk >= 50 else put, then collect T3 payoff
choose_call = S_2wk >= 50
p_chooser = np.where(choose_call, np.maximum(S_3wk - 50, 0), np.maximum(50 - S_3wk, 0))

# Binary put (K=40, T+21): pays BINARY_PAYOFF if S_3wk < 40
p_binary_put = np.where(S_3wk < 40, BINARY_PAYOFF, 0.0)

# Knock-out put (K=45, T+21, barrier=KO_BARRIER):
# "before expiry" --> check steps 1..59 (indices 0..58), NOT the expiry step itself
# The option is worthless if any of those steps breach the barrier;
# the expiry step only determines the payoff.
barrier_breach = np.any(prices[:, :T3_STEPS - 1] < KO_BARRIER, axis=1)
p_ko_put = np.where(barrier_breach, 0.0, np.maximum(45 - S_3wk, 0))

# --- Results Table ------------------------------------------------------------

def sim_fair(payoffs):
    return float(np.mean(payoffs))

# Analytical fair values
ana = {
    "AC_50_P  (3wk, K=50)": bs_put(S0, 50, T3_YEARS, SIGMA),
    "AC_50_C  (3wk, K=50)": bs_call(S0, 50, T3_YEARS, SIGMA),
    "AC_35_P  (3wk, K=35)": bs_put(S0, 35, T3_YEARS, SIGMA),
    "AC_40_P  (3wk, K=40)": bs_put(S0, 40, T3_YEARS, SIGMA),
    "AC_45_P  (3wk, K=45)": bs_put(S0, 45, T3_YEARS, SIGMA),
    "AC_60_C  (3wk, K=60)": bs_call(S0, 60, T3_YEARS, SIGMA),
    "AC_50_P2 (2wk, K=50)": bs_put(S0, 50, T2_YEARS, SIGMA),
    "AC_50_C2 (2wk, K=50)": bs_call(S0, 50, T2_YEARS, SIGMA),
    "AC_50_CO chooser     ": bs_chooser(S0, 50, T3_YEARS, T2_YEARS, SIGMA),
    f"AC_40_BP binary(pay={BINARY_PAYOFF})": bs_binary_put(S0, 40, T3_YEARS, SIGMA, payoff=BINARY_PAYOFF),
    f"AC_45_KO (barrier={KO_BARRIER}) ": None,  # no closed-form; use sim
}

sim_vals = {
    "AC_50_P  (3wk, K=50)": sim_fair(p_ac_50_P),
    "AC_50_C  (3wk, K=50)": sim_fair(p_ac_50_C),
    "AC_35_P  (3wk, K=35)": sim_fair(p_ac_35_P),
    "AC_40_P  (3wk, K=40)": sim_fair(p_ac_40_P),
    "AC_45_P  (3wk, K=45)": sim_fair(p_ac_45_P),
    "AC_60_C  (3wk, K=60)": sim_fair(p_ac_60_C),
    "AC_50_P2 (2wk, K=50)": sim_fair(p_ac_50_P2),
    "AC_50_C2 (2wk, K=50)": sim_fair(p_ac_50_C2),
    "AC_50_CO chooser     ": sim_fair(p_chooser),
    f"AC_40_BP binary(pay={BINARY_PAYOFF})": sim_fair(p_binary_put),
    f"AC_45_KO (barrier={KO_BARRIER}) ": sim_fair(p_ko_put),
}

market = {
    "AC_50_P  (3wk, K=50)": (12.00, 12.05, 50),
    "AC_50_C  (3wk, K=50)": (12.00, 12.05, 50),
    "AC_35_P  (3wk, K=35)": ( 4.33,  4.35, 50),
    "AC_40_P  (3wk, K=40)": ( 6.50,  6.55, 50),
    "AC_45_P  (3wk, K=45)": ( 9.05,  9.10, 50),
    "AC_60_C  (3wk, K=60)": ( 8.80,  8.85, 50),
    "AC_50_P2 (2wk, K=50)": ( 9.70,  9.75, 50),
    "AC_50_C2 (2wk, K=50)": ( 9.70,  9.75, 50),
    "AC_50_CO chooser     ": (22.20, 22.30, 50),
    f"AC_40_BP binary(pay={BINARY_PAYOFF})": (5.00, 5.10, 50),
    f"AC_45_KO (barrier={KO_BARRIER}) ": (0.15, 0.175, 500),
}

CONTRACT_SIZE = 3000
EDGE_THRESHOLD = 0.01

# --- Build per-product rows ---------------------------------------------------
products = []
for name in sim_vals:
    bid, ask, size = market[name]
    fair_bs = ana.get(name)
    fair_mc = sim_vals[name]
    fair = fair_bs if fair_bs is not None else fair_mc
    fair_src = "BS" if fair_bs is not None else "MC"
    edge_buy  = fair - ask
    edge_sell = bid  - fair
    if edge_buy > EDGE_THRESHOLD:
        action, trade_price, edge = "BUY ", ask, edge_buy
    elif edge_sell > EDGE_THRESHOLD:
        action, trade_price, edge = "SELL", bid, edge_sell
    else:
        action, trade_price, edge = "n/a ", None, max(edge_buy, edge_sell)
    exp_pnl = edge * size * CONTRACT_SIZE if action != "n/a " else 0.0
    products.append({
        "name": name.strip(),
        "fair": fair,
        "fair_src": fair_src,
        "bid": bid,
        "ask": ask,
        "action": action,
        "trade_price": trade_price,
        "edge": edge,
        "size": size,
        "exp_pnl": exp_pnl,
    })

# --- All Products Table -------------------------------------------------------
W = 100
print()
print("=" * W)
print("  AETHER CRYSTAL - ALL PRODUCTS")
print(f"  {'Product':<30} {'Fair':>7} {'src':>4}   {'Bid':>7}  {'Ask':>7}   {'Edge/unit':>10}   {'Verdict'}")
print("-" * W)
for p in products:
    src = f"({p['fair_src']})"
    edge_str = f"{p['edge']:+.4f}" if p['action'] != "n/a " else f"{p['edge']:+.4f}"
    verdict = f"{p['action']} @ {p['trade_price']}" if p['action'] != "n/a " else "fairly priced"
    print(f"  {p['name']:<30} {p['fair']:>7.4f} {src:>4}   {p['bid']:>7.4f}  {p['ask']:>7.4f}   {edge_str:>10}   {verdict}")
print("=" * W)

# --- Recommended Trades Summary ----------------------------------------------
trades = [p for p in products if p['action'] != "n/a "]
total_pnl = sum(p['exp_pnl'] for p in trades)

print()
print("=" * W)
print("  RECOMMENDED TRADES")
print(f"  {'Product':<30} {'Action':<5}  {'@ Price':>8}  {'Edge/unit':>10}  {'Max vol':>8}  {'Expected PnL':>14}")
print(f"  {'':30}  {'':5}  {'':8}  {'':10}  {'':8}  {'(edge x vol x 3000)':>14}")
print("-" * W)
for p in trades:
    print(
        f"  {p['name']:<30} {p['action']:<5}  {p['trade_price']:>8.4f}"
        f"  {p['edge']:>+10.4f}  {p['size']:>8,}  {p['exp_pnl']:>14,.0f}"
    )
print("-" * W)
print(f"  {'TOTAL EXPECTED PnL (if all trades filled at full volume)':>67}  {total_pnl:>14,.0f}")
print("=" * W)

# --- Chooser breakdown -------------------------------------------------------
c3 = bs_call(S0, 50, T3_YEARS, SIGMA)
p2 = bs_put(S0,  50, T2_YEARS, SIGMA)
chooser_mkt_bid = 22.20
print()
print("=" * W)
print("  CHOOSER OPTION - why it's overpriced")
print("-" * W)
print(f"  Fair value formula:  V = Call(3wk, K=50) + Put(2wk, K=50)")
print(f"    Call(3wk, K=50)  = {c3:.4f}")
print(f"    Put (2wk, K=50)  = {p2:.4f}")
print(f"    Chooser fair     = {c3 + p2:.4f}")
print(f"    Market bid       = {chooser_mkt_bid:.4f}   <-- sell here")
print(f"    Edge per unit    = {chooser_mkt_bid - (c3 + p2):+.4f}")
print("=" * W)

# --- KO Barrier Sensitivity --------------------------------------------------
ko_mkt_bid = 0.15
ko_mkt_ask = 0.175
print()
print("=" * W)
print(f"  AC 45 KO - barrier sensitivity  (market bid={ko_mkt_bid}, ask={ko_mkt_ask})")
print(f"  Vanilla AC_45_P fair = {bs_put(S0, 45, T3_YEARS, SIGMA):.4f}")
print("-" * W)
print(f"  {'Barrier':>8}  {'KO fair':>8}  {'Breach rate':>12}  {'vs market bid':>14}  {'Trade?'}")
print("-" * W)
for barrier in [35, 40, 42, 44, 45, 46, 48, 50]:
    breach = np.any(prices[:, :T3_STEPS - 1] < barrier, axis=1)
    ko_val = float(np.mean(np.where(breach, 0.0, np.maximum(45 - S_3wk, 0))))
    edge_b = ko_val - ko_mkt_ask   # edge if buy
    edge_s = ko_mkt_bid - ko_val   # edge if sell
    if edge_b > EDGE_THRESHOLD:
        verdict = f"BUY  (edge {edge_b:+.4f})"
    elif edge_s > EDGE_THRESHOLD:
        verdict = f"SELL (edge {edge_s:+.4f})"
    else:
        verdict = "fairly priced"
    marker = " <-- CONFIRMED" if barrier == KO_BARRIER else ""
    print(f"  {barrier:>8.0f}  {ko_val:>8.4f}  {breach.mean()*100:>11.1f}%  {ko_val - (ko_mkt_bid+ko_mkt_ask)/2:>+14.4f}  {verdict}{marker}")
print("=" * W)

# --- Binary Put Payoff Sensitivity -------------------------------------------
bp_mkt_bid = 5.00
bp_mkt_ask = 5.10
prob_itm = float((S_3wk < 40).mean())
print()
print("=" * W)
print(f"  AC 40 BP - binary put payoff sensitivity  (market bid={bp_mkt_bid}, ask={bp_mkt_ask})")
print(f"  P(finish below K=40 at 3wk) = {prob_itm:.4f}  -->  fair = payoff x {prob_itm:.4f}")
print("-" * W)
print(f"  {'Payoff':>8}  {'BS fair':>8}  {'vs bid':>8}  {'vs ask':>8}  {'Trade?'}")
print("-" * W)
for pay in [8, 9, 10, 10.5, 11, 12]:
    fv = bs_binary_put(S0, 40, T3_YEARS, SIGMA, payoff=pay)
    edge_b = fv - bp_mkt_ask
    edge_s = bp_mkt_bid - fv
    if edge_b > EDGE_THRESHOLD:
        verdict = f"BUY  (edge {edge_b:+.4f})"
    elif edge_s > EDGE_THRESHOLD:
        verdict = f"SELL (edge {edge_s:+.4f})"
    else:
        verdict = "fairly priced"
    marker = " <-- CONFIRMED" if pay == BINARY_PAYOFF else ""
    print(f"  {pay:>8.1f}  {fv:>8.4f}  {fv-bp_mkt_bid:>+8.4f}  {fv-bp_mkt_ask:>+8.4f}  {verdict}{marker}")
print("=" * W)
