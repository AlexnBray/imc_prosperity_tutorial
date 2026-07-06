"""
Round 4 Manual Challenge: Aether Crystal Options Simulator
GBM: zero drift, sigma=251%, 4 steps/day, 252 trading days/year
All options written on AC with S0 = 50
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

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

# Sigma prior for blended portfolio risk (matches max_ev_strategy.py)
SIGMA_SCENARIOS = [2.30, 2.40, 2.47, 2.49, 2.51, 2.55, 2.60, 2.70]
SIGMA_WEIGHTS   = [0.05, 0.10, 0.25, 0.15, 0.30, 0.10, 0.04, 0.01]

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

# --- Greeks ------------------------------------------------------------------
def bs_greeks(S, K, T, sigma, r=0.0, option_type="put"):
    """Return delta, gamma, vega, theta for a vanilla European option."""
    if T <= 0:
        intrinsic = max(K - S, 0) if option_type == "put" else max(S - K, 0)
        return {"delta": (-1 if option_type == "put" and S < K else 0),
                "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1, d2 = bs_d1d2(S, K, T, sigma, r)
    phi_d1 = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    gamma = phi_d1 / (S * sigma * np.sqrt(T))
    vega  = S * phi_d1 * np.sqrt(T)          # $ change per +1 absolute sigma (e.g. 0→1 annualised)
    theta = -(S * phi_d1 * sigma) / (2 * np.sqrt(T))  # per year (r=0 assumed)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

def bs_binary_put_greeks(S, K, T, sigma, payoff=1.0, r=0.0):
    """Delta and vega for a cash-or-nothing put."""
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    d1, d2 = bs_d1d2(S, K, T, sigma, r)
    phi_d2 = norm.pdf(d2)
    delta  = -payoff * phi_d2 / (S * sigma * np.sqrt(T))
    vega   =  payoff * phi_d2 * (d1 / sigma)    # approximation (r=0)
    return {"delta": delta, "gamma": 0.0, "vega": vega}

# --- Implied volatility -------------------------------------------------------
def implied_vol(market_price, S, K, T, r=0.0, option_type="put",
                lo=0.01, hi=50.0):
    """Invert BS to recover implied vol. Returns None if no solution."""
    try:
        pricer = bs_put if option_type == "put" else bs_call
        f = lambda v: pricer(S, K, T, v, r) - market_price
        if f(lo) * f(hi) > 0:
            return None
        return brentq(f, lo, hi, xtol=1e-8)
    except Exception:
        return None

# --- Kelly criterion ---------------------------------------------------------
def kelly_fraction(pnl_array):
    """
    Continuous-payoff Kelly approximation: f* = E[PnL] / E[PnL^2] (log-utility).
    Returns fraction of wealth to risk on this trade.
    """
    e  = float(np.mean(pnl_array))
    e2 = float(np.mean(pnl_array ** 2))
    if e2 <= 0 or e <= 0:
        return 0.0
    return e / e2

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
    exp_pnl = edge * size if action != "n/a " else 0.0
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
print(f"  {'':30}  {'':5}  {'':8}  {'':10}  {'':8}  {'(edge x vol)':>14}")
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

# --- Discrete vs continuous barrier comparison --------------------------------
# First-passage probability for GBM: P(min_t S_t < H) with drift mu = r - sigma^2/2
# Analytical (continuous monitoring):
#   P = N((log(H/S) - mu*T)/(sigma*sqrt(T))) + exp(2*mu*log(H/S)/sigma^2) * N((log(H/S) + mu*T)/(sigma*sqrt(T)))
mu = R - 0.5 * SIGMA**2          # = -3.15  (strong downward drift in log-space)
lhs = (np.log(KO_BARRIER / S0) - mu * T3_YEARS) / (SIGMA * np.sqrt(T3_YEARS))
rhs = (np.log(KO_BARRIER / S0) + mu * T3_YEARS) / (SIGMA * np.sqrt(T3_YEARS))
exp_factor = np.exp(2 * mu * np.log(KO_BARRIER / S0) / SIGMA**2)
p_breach_continuous = norm.cdf(lhs) + exp_factor * norm.cdf(rhs)
p_breach_discrete   = float(np.mean(barrier_breach))

# Broadie-Glasserman-Kou correction: shift barrier by beta*sigma*sqrt(dt) inward
BGK_BETA = 0.5826
H_eff    = KO_BARRIER * np.exp(-BGK_BETA * SIGMA * np.sqrt(DT))
lhs2 = (np.log(H_eff / S0) - mu * T3_YEARS) / (SIGMA * np.sqrt(T3_YEARS))
rhs2 = (np.log(H_eff / S0) + mu * T3_YEARS) / (SIGMA * np.sqrt(T3_YEARS))
exp_factor2 = np.exp(2 * mu * np.log(H_eff / S0) / SIGMA**2)
p_breach_bgk = norm.cdf(lhs2) + exp_factor2 * norm.cdf(rhs2)

print()
print("=" * W)
print(f"  AC 45 KO - Discrete vs continuous barrier  (barrier={KO_BARRIER}, {STEPS_PER_DAY} checks/day)")
print(f"  Continuous first-passage  P(breach)  = {p_breach_continuous:.4f}  ({p_breach_continuous*100:.1f}%)")
print(f"  BGK-corrected  H_eff={H_eff:.3f}  P(breach)  = {p_breach_bgk:.4f}  ({p_breach_bgk*100:.1f}%)")
print(f"  MC discrete monitoring     P(breach)  = {p_breach_discrete:.4f}  ({p_breach_discrete*100:.1f}%)")
print(f"  --> Discrete monitoring ~MATCHES BGK correction ({abs(p_breach_discrete-p_breach_bgk)*100:.1f}% gap)")
print(f"  --> Continuous would breach MORE often; discrete barrier PROTECTS the KO buyer (+value)")
ko_mc_fair = sim_vals[f"AC_45_KO (barrier={KO_BARRIER}) "]
print(f"  MC KO fair = {ko_mc_fair:.4f}  (the discrete price, correct for this competition)")
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

# --- Greeks Table ------------------------------------------------------------
print()
print("=" * W)
print("  GREEKS  (at S0=50, sigma=251%, r=0  |  per unit)  -- trades are DIRECTIONAL, size matters!")
print(f"  {'Position':<32} {'Qty':>5}  {'Delta':>8}  {'Gamma':>8}  {'Vega':>8}  {'Port. Delta':>12}")
print("-" * W)

g_p2   = bs_greeks(S0, 50, T2_YEARS, SIGMA, option_type="put")
g_c2   = bs_greeks(S0, 50, T2_YEARS, SIGMA, option_type="call")
g_c3   = bs_greeks(S0, 50, T3_YEARS, SIGMA, option_type="call")
g_p2_3 = bs_greeks(S0, 50, T2_YEARS, SIGMA, option_type="put")    # chooser's put leg
g_bp   = bs_binary_put_greeks(S0, 40, T3_YEARS, SIGMA, payoff=BINARY_PAYOFF)

# Chooser = call(3wk) + put(2wk); same quantities
g_co_delta = g_c3["delta"] + g_p2_3["delta"]
g_co_gamma = g_c3["gamma"] + g_p2_3["gamma"]
g_co_vega  = g_c3["vega"]  + g_p2_3["vega"]

# KO: finite-difference delta via MC (S±2%)
S_up, S_dn = S0 * 1.02, S0 * 0.98
prices_up = S_up * np.exp(np.cumsum(log_increments, axis=1))
prices_dn = S_dn * np.exp(np.cumsum(log_increments, axis=1))
breach_up = np.any(prices_up[:, :T3_STEPS - 1] < KO_BARRIER, axis=1)
breach_dn = np.any(prices_dn[:, :T3_STEPS - 1] < KO_BARRIER, axis=1)
ko_up     = float(np.mean(np.where(breach_up, 0.0, np.maximum(45 - prices_up[:, T3_STEPS-1], 0))))
ko_dn     = float(np.mean(np.where(breach_dn, 0.0, np.maximum(45 - prices_dn[:, T3_STEPS-1], 0))))
ko_delta  = (ko_up - ko_dn) / (S_up - S_dn)

pos_rows = [
    ("BUY  50 x AC_50_P2",   50,  g_p2["delta"],  g_p2["gamma"],  g_p2["vega"],   50  *  g_p2["delta"]),
    ("BUY  50 x AC_50_C2",   50,  g_c2["delta"],  g_c2["gamma"],  g_c2["vega"],   50  *  g_c2["delta"]),
    ("SELL 50 x AC_50_CO",  -50,  g_co_delta,     g_co_gamma,     g_co_vega,     -50  *  g_co_delta),
    ("SELL 50 x AC_40_BP",  -50,  g_bp["delta"],  g_bp["gamma"],  g_bp["vega"],  -50  *  g_bp["delta"]),
    ("BUY 500 x AC_45_KO",  500,  ko_delta,       float("nan"),   float("nan"),  500  *  ko_delta),
]

for label, qty, delta, gamma, vega, port_delta in pos_rows:
    g_str = f"{gamma:.5f}" if not np.isnan(gamma) else "  MC/n.a"
    v_str = f"{vega:.4f}"  if not np.isnan(vega)  else "  MC/n.a"
    print(f"  {label:<32} {qty:>5}  {delta:>+8.4f}  {g_str:>8}  {v_str:>8}  {port_delta:>+12.3f}")

total_delta = sum(r[-1] for r in pos_rows)
print("-" * W)
print(f"  {'NET PORTFOLIO DELTA':>71}  {total_delta:>+12.3f}")
print(f"  (negative = profits when AC falls; portfolio is biased to downside)")
print("=" * W)

# --- Implied Volatility Calibration ------------------------------------------
print()
print("=" * W)
print("  IMPLIED VOLATILITY  (back-solved from market mid-price, model sigma=251%)")
print(f"  {'Product':<28}  {'Mid':>7}  {'IV (%)':>8}  {'vs model':>10}  {'Smile?'}")
print("-" * W)

iv_products = [
    ("AC_50_P3 (3wk K=50)",  "put",  50, T3_YEARS, 12.025),
    ("AC_45_P3 (3wk K=45)",  "put",  45, T3_YEARS,  9.075),
    ("AC_40_P3 (3wk K=40)",  "put",  40, T3_YEARS,  6.525),
    ("AC_35_P3 (3wk K=35)",  "put",  35, T3_YEARS,  4.340),
    ("AC_60_C3 (3wk K=60)",  "call", 60, T3_YEARS,  8.825),
    ("AC_50_P2 (2wk K=50)",  "put",  50, T2_YEARS,  9.725),
    ("AC_50_C2 (2wk K=50)",  "call", 50, T2_YEARS,  9.725),
]

ivs = []
for name, otype, K, T, mid in iv_products:
    iv = implied_vol(mid, S0, K, T, option_type=otype)
    ivs.append(iv)
    if iv is not None:
        diff = (iv - SIGMA) * 100
        smile = "FLAT" if abs(diff) < 2 else (f"HIGHER +{diff:.0f}%" if diff > 0 else f"LOWER {diff:.0f}%")
        print(f"  {name:<28}  {mid:>7.4f}  {iv*100:>7.1f}%  {diff:>+9.1f}%  {smile}")
    else:
        print(f"  {name:<28}  {mid:>7.4f}  {'no sol':>7}  {'':>10}")
print("  --> Any deviation from 251% across strikes reveals a vol smile/skew")
print("=" * W)

# --- Kelly Position Sizing ---------------------------------------------------
print()
print("=" * W)
print("  KELLY POSITION SIZING  (log-utility approx, per unit of wealth)")
print(f"  {'Trade':<30}  {'E[PnL/unit]':>12}  {'Std[PnL/unit]':>14}  {'Kelly f*':>10}  {'Half-Kelly':>10}")
print("-" * W)

def _unit_pnl(payoff, entry, qty, is_buy):
    if is_buy:
        return payoff - entry
    return entry - payoff

kelly_items = [
    ("BUY 2wk put K=50",   p_ac_50_P2, 9.75,  True),
    ("BUY 2wk call K=50",  p_ac_50_C2, 9.75,  True),
    ("SELL chooser K=50",  p_chooser,  22.20, False),
    ("SELL binary put K=40", p_binary_put, 5.00, False),
    ("BUY KO put (barrier=35)", p_ko_put, 0.175, True),
]

for label, payoffs, entry, is_buy in kelly_items:
    u_pnl = _unit_pnl(payoffs, entry, 1, is_buy)
    e_u   = float(np.mean(u_pnl))
    sd_u  = float(np.std(u_pnl))
    # Kelly: maximize E[log(1 + f*X/W)]; small-bet approx: f* = E[X]/(W*E[X^2])
    # Express as fraction of max-loss (= entry price for buys)
    max_loss = entry if is_buy else max(payoffs)  # rough worst-case
    e2   = float(np.mean(u_pnl**2))
    kf   = e_u / e2 if e2 > 0 else 0.0
    print(f"  {label:<30}  {e_u:>+12.4f}  {sd_u:>14.4f}  {kf:>10.4f}  {kf/2:>10.4f}")

print("  Note: Kelly f* here is fraction of (unit payoff)^2 normalisation.")
print("  Use half-Kelly in practice for robustness against model error.")
print("=" * W)

# --- YOUR PORTFOLIO (mutate_small strategy) ----------------------------------
# Signed quantities: positive = BUY, negative = SELL
PORT_POSITIONS = {
    "AC_50_P":   ("BUY",   5,  12.05),
    "AC_50_C":   ("BUY",  42,  12.05),
    "AC_35_P":   ("SELL",  7,   4.33),
    "AC_45_P":   ("BUY",  50,   9.10),
    "AC_50_P_2": ("BUY",  50,   9.75),
    "AC_50_C_2": ("BUY",  19,   9.75),
    "AC_50_CO":  ("SELL", 50,  22.20),
    "AC_40_BP":  ("SELL", 50,   5.00),
    "AC_45_KO":  ("BUY", 370,   0.175),
}

PORT_LABELS = {
    "AC_50_P":   "AC_50_P  (3wk put  K=50)",
    "AC_50_C":   "AC_50_C  (3wk call K=50)",
    "AC_35_P":   "AC_35_P  (3wk put  K=35)",
    "AC_45_P":   "AC_45_P  (3wk put  K=45)",
    "AC_50_P_2": "AC_50_P_2 (2wk put  K=50)",
    "AC_50_C_2": "AC_50_C_2 (2wk call K=50)",
    "AC_50_CO":  "AC_50_CO  (chooser  K=50)",
    "AC_40_BP":  "AC_40_BP  (binary put K=40)",
    "AC_45_KO":  "AC_45_KO  (KO put   K=45)",
}

def payoffs_for_sigma(sigma):
    """Reuse the shared Z matrix; compute all option payoffs for a given sigma."""
    log_inc  = (R - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * Z
    px = S0 * np.exp(np.cumsum(log_inc, axis=1))
    s2 = px[:, T2_STEPS - 1]
    s3 = px[:, T3_STEPS - 1]
    breach = np.any(px[:, :T3_STEPS - 1] < KO_BARRIER, axis=1)
    return {
        "AC_50_P":   np.maximum(50 - s3, 0),
        "AC_50_C":   np.maximum(s3 - 50, 0),
        "AC_35_P":   np.maximum(35 - s3, 0),
        "AC_45_P":   np.maximum(45 - s3, 0),
        "AC_50_P_2": np.maximum(50 - s2, 0),
        "AC_50_C_2": np.maximum(s2 - 50, 0),
        "AC_50_CO":  np.where(s2 >= 50, np.maximum(s3 - 50, 0), np.maximum(50 - s3, 0)),
        "AC_40_BP":  np.where(s3 < 40, BINARY_PAYOFF, 0.0),
        "AC_45_KO":  np.where(breach, 0.0, np.maximum(45 - s3, 0)),
    }

def portfolio_pnl_for_sigma(sigma):
    """Per-path portfolio PnL (vol x unit_pnl, no CONTRACT_SIZE) for a given sigma."""
    pay = payoffs_for_sigma(sigma)
    total = np.zeros(N_SIM)
    for name, (side, qty, entry) in PORT_POSITIONS.items():
        unit = (pay[name] - entry) if side == "BUY" else (entry - pay[name])
        total += qty * unit
    return total

# --- Per-position table at model sigma (sigma=2.51) --------------------------
pay_model = payoffs_for_sigma(SIGMA)

print()
print("=" * W)
print(f"  YOUR PORTFOLIO  (per-position breakdown at model sigma={SIGMA})")
print(f"  {'Position':<32} {'Dir':>4}  {'Vol':>5}  {'Entry':>7}  {'E[pay/u]':>9}  {'E[PnL/u]':>10}  {'Tot E[PnL]':>11}")
print("-" * W)

for name, (side, qty, entry) in PORT_POSITIONS.items():
    pay = pay_model[name]
    unit_pnl = (pay - entry) if side == "BUY" else (entry - pay)
    e_pay  = float(np.mean(pay))
    e_unit = float(np.mean(unit_pnl))
    tot    = e_unit * qty
    print(f"  {PORT_LABELS[name]:<32} {side:>4}  {qty:>5}  {entry:>7.4f}  {e_pay:>9.4f}  {e_unit:>+10.4f}  {tot:>+11.2f}")

print("=" * W)

# --- Sigma-blended portfolio risk summary ------------------------------------
print()
print(f"Blending portfolio PnL across {len(SIGMA_SCENARIOS)} sigma scenarios...")
pnl_per_sigma = [portfolio_pnl_for_sigma(sig) for sig in SIGMA_SCENARIOS]

w = np.array(SIGMA_WEIGHTS)
E_per_sigma   = np.array([float(p.mean())       for p in pnl_per_sigma])
P5_per_sigma  = np.array([float(np.percentile(p, 5))  for p in pnl_per_sigma])
P1_per_sigma  = np.array([float(np.percentile(p, 1))  for p in pnl_per_sigma])
Pl_per_sigma  = np.array([float((p < 0).mean()) for p in pnl_per_sigma])

# Blended moments (weighted average across sigmas)
E_blend  = float(np.dot(w, E_per_sigma))
Pl_blend = float(np.dot(w, Pl_per_sigma))

# Blended percentiles: build one giant weighted sample
flat_pnl = np.concatenate(pnl_per_sigma)
flat_w   = np.repeat(w, N_SIM) / N_SIM
order    = np.argsort(flat_pnl)
sp, sw   = flat_pnl[order], flat_w[order]
cw       = np.cumsum(sw)
def q_blend(p): return float(sp[np.searchsorted(cw, p, side="left")])
P5_blend  = q_blend(0.05)
P1_blend  = q_blend(0.01)
P25_blend = q_blend(0.25)
P75_blend = q_blend(0.75)
P95_blend = q_blend(0.95)
Std_blend = float(np.sqrt(np.dot(w, np.array([p.std()**2 for p in pnl_per_sigma])
                                 + (E_per_sigma - E_blend)**2)))

# Analytic mean: BS fair value for vanilla positions, MC for KO
def analytic_unit_pnl(name, side, entry, sigma):
    p = MKT_ANALYTIC[name]
    if p["kind"] == "put":
        fv = bs_put(S0, p["K"], p["T"], sigma)
    elif p["kind"] == "call":
        fv = bs_call(S0, p["K"], p["T"], sigma)
    elif p["kind"] == "binary":
        fv = bs_binary_put(S0, p["K"], p["T"], sigma, payoff=BINARY_PAYOFF)
    elif p["kind"] == "chooser":
        fv = bs_chooser(S0, p["K"], p["T"], T2_YEARS, sigma)
    else:
        return None  # KO: use MC
    return fv - entry if side == "BUY" else entry - fv

MKT_ANALYTIC = {
    "AC_50_P":   dict(K=50, T=T3_YEARS, kind="put"),
    "AC_50_C":   dict(K=50, T=T3_YEARS, kind="call"),
    "AC_35_P":   dict(K=35, T=T3_YEARS, kind="put"),
    "AC_45_P":   dict(K=45, T=T3_YEARS, kind="put"),
    "AC_50_P_2": dict(K=50, T=T2_YEARS, kind="put"),
    "AC_50_C_2": dict(K=50, T=T2_YEARS, kind="call"),
    "AC_50_CO":  dict(K=50, T=T3_YEARS, kind="chooser"),
    "AC_40_BP":  dict(K=40, T=T3_YEARS, kind="binary"),
    "AC_45_KO":  dict(K=45, T=T3_YEARS, kind="ko"),
}

ana_E_per_sigma = []
for sig, pnl_mc in zip(SIGMA_SCENARIOS, pnl_per_sigma):
    e_ana = 0.0
    for name, (side, qty, entry) in PORT_POSITIONS.items():
        u = analytic_unit_pnl(name, side, entry, sig)
        if u is not None:
            e_ana += qty * u
        else:
            # KO: fall back to MC mean for this sigma
            pay = payoffs_for_sigma(sig)["AC_45_KO"]
            ko_unit = (pay - entry) if side == "BUY" else (entry - pay)
            e_ana += qty * float(ko_unit.mean())
    ana_E_per_sigma.append(e_ana)

Ana_E_blend = float(np.dot(w, np.array(ana_E_per_sigma)))

# Build blended analytic percentile via normal approximation (sigma of per-path PnL)
# For a proper analytic p5 we use the blended MC p5 of vanilla legs + analytic means
# Simple approach: report analytic mean, use MC for tails
Ana_P5_blend = P5_blend + (Ana_E_blend - E_blend)   # shift MC tails by analytic vs MC mean gap

print()
print("=" * W)
print("  PORTFOLIO RISK SUMMARY  (sigma-blended prior)")
print(f"  Sigma scenarios : {SIGMA_SCENARIOS}")
print(f"  Weights         : {SIGMA_WEIGHTS}")
print("-" * W)
print(f"  {'Sigma':>5}  {'Weight':>7}  {'MC mean':>9}  {'MC p5':>9}  {'MC p1':>9}  {'P(loss)':>9}")
print("-" * W)
for sig, wt, e, p5, p1, pl in zip(SIGMA_SCENARIOS, SIGMA_WEIGHTS,
                                   E_per_sigma, P5_per_sigma, P1_per_sigma, Pl_per_sigma):
    flag = "  <- model" if abs(sig - SIGMA) < 0.005 else ""
    print(f"  {sig:>5.2f}  {wt:>6.0%}   {e:>+9.2f}  {p5:>+9.2f}  {p1:>+9.2f}  {pl*100:>8.1f}%{flag}")
print("-" * W)
print(f"  {'BLENDED':>5}  {'100%':>7}   {E_blend:>+9.2f}  {P5_blend:>+9.2f}  {P1_blend:>+9.2f}  {Pl_blend*100:>8.1f}%")
print("=" * W)
print()
print("=" * W)
print("  PORTFOLIO RISK SUMMARY  (blended distribution)")
print("-" * W)
print(f"  MC   mean        : {E_blend:>+12.2f}")
print(f"  MC   p5          : {P5_blend:>+12.2f}")
print(f"  MC   p1          : {P1_blend:>+12.2f}")
print(f"  MC   loss        : {Pl_blend*100:>12.2f}%")
print(f"  Analytic mean    : {Ana_E_blend:>+12.2f}")
print(f"  Analytic p5 (est): {Ana_P5_blend:>+12.2f}")
print("-" * W)
print(f"  Std Dev          : {Std_blend:>+12.2f}")
print(f"  25th percentile  : {P25_blend:>+12.2f}")
print(f"  75th percentile  : {P75_blend:>+12.2f}")
print(f"  95th percentile  : {P95_blend:>+12.2f}")
print("=" * W)
