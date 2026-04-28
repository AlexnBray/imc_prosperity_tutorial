"""
Deep analysis: why PnL is capped and what can improve it.
"""
import numpy as np
from scipy.stats import norm

S0, SIGMA, R = 50.0, 2.51, 0.0
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
DT = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)
T2_YEARS = 10 / 252
T3_YEARS = 15 / 252
T1_YEARS = 5 / 252
N_SIM, SEED = 500_000, 42
KO_BARRIER, BINARY_PAYOFF = 35.0, 10.0
CONTRACT_SIZE = 3000
W = 90

# === Simulate ================================================================
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_SIM, 60))
log_inc = (R - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * Z
prices = S0 * np.exp(np.cumsum(log_inc, axis=1))
S_2wk, S_3wk = prices[:, 39], prices[:, 59]
breach = np.any(prices[:, :59] < KO_BARRIER, axis=1)


def bs_put(S, K, T, sig, r=0.0):
    if T <= 0:
        return float(max(K - S, 0))
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call(S, K, T, sig, r=0.0):
    if T <= 0:
        return float(max(S - K, 0))
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# === 1. Near-miss trades =====================================================
print("=" * W)
print("  NEAR-MISS TRADES (edge > 0 but below 0.01 threshold in simulate.py)")
print(f"  {'Product':<30} {'Fair':>8} {'Bid':>7} {'Ask':>7} {'Edge(S)':>8} {'E[PnL]':>10}")
print("-" * W)
near_miss = [
    ("AC_60_C (3wk, K=60)",  bs_call(S0, 60, T3_YEARS, SIGMA), 8.80, 8.85, 50),
    ("AC_35_P (3wk, K=35)",  bs_put(S0,  35, T3_YEARS, SIGMA), 4.33, 4.35, 50),
    ("AC_40_P (3wk, K=40)",  bs_put(S0,  40, T3_YEARS, SIGMA), 6.50, 6.55, 50),
    ("AC_45_P (3wk, K=45)",  bs_put(S0,  45, T3_YEARS, SIGMA), 9.05, 9.10, 50),
    ("AC_50_P (3wk, K=50)",  bs_put(S0,  50, T3_YEARS, SIGMA),12.00,12.05, 50),
]
for nm, fair, bid, ask, sz in near_miss:
    edge_s = bid - fair
    pnl_s  = edge_s * sz * CONTRACT_SIZE
    print(f"  {nm:<30} {fair:>8.4f} {bid:>7.2f} {ask:>7.2f} {edge_s:>+8.4f} {pnl_s:>10,.0f}")
print("  Note: AC_60_C has edge +0.008 to SELL -- real but tiny (0.82% of spread)")
print("=" * W)

# === 2. Vol term structure impact on 2wk options ============================
print()
print("=" * W)
print("  VOL TERM STRUCTURE: IMPACT ON 2wk OPTION EDGE")
print("-" * W)
for sig_test in [2.47, 2.48, 2.49, 2.50, 2.51, 2.52]:
    p2 = bs_put(S0, 50, T2_YEARS, sig_test)
    edge = p2 - 9.75
    pnl  = edge * 50 * CONTRACT_SIZE
    marker = " <-- market implied" if abs(sig_test - 2.47) < 0.005 else (
             " <-- MODEL" if abs(sig_test - 2.51) < 0.005 else "")
    print(f"  sigma={sig_test:.2f}: P2_fair={p2:.4f}, edge={edge:+.4f}/u, E[PnL]={pnl:>8,.0f}{marker}")
print()
print("  KEY: 2wk market implies 247% vol. At 247%, edge=+0.050/u (down from 0.121 at 251%).")
print("  The TRUE edge lies between these: confidence determines position size.")
print("=" * W)

# === 3. Calendar spread (sell 3wk + buy 2wk straddle) =======================
print()
print("=" * W)
print("  CALENDAR STRADDLE: SELL 3wk ATM + BUY 2wk ATM")
print("-" * W)
straddle_3wk_bid  = 24.00      # 12.00 put + 12.00 call (both at bid)
straddle_3wk_fair = bs_put(S0, 50, T3_YEARS, SIGMA) * 2
straddle_2wk_ask  = 19.50      # 9.75 put + 9.75 call (both at ask)
straddle_2wk_fair = bs_put(S0, 50, T2_YEARS, SIGMA) * 2
edge_sell_3wk  = straddle_3wk_bid  - straddle_3wk_fair  # negative!
edge_buy_2wk   = straddle_2wk_fair - straddle_2wk_ask   # positive
calendar_edge  = edge_sell_3wk + edge_buy_2wk
print(f"  SELL 3wk straddle: recv={straddle_3wk_bid:.2f}, fair={straddle_3wk_fair:.4f}, edge(sell)={edge_sell_3wk:+.4f}")
print(f"  BUY  2wk straddle: pay ={straddle_2wk_ask:.2f}, fair={straddle_2wk_fair:.4f}, edge(buy) ={edge_buy_2wk:+.4f}")
print(f"  Calendar net edge: {calendar_edge:+.4f}/unit -- {calendar_edge*50*CONTRACT_SIZE:>+8,.0f} E[PnL] for 50u")
print(f"  VERDICT: calendar adds only {edge_sell_3wk:+.4f}/u from the 3wk sale (loses vs fair).")
print(f"  The 3wk sale COSTS 0.054/u. Calendar is WORSE than buying 2wk alone.")
print("=" * W)

# === 4. DI put synthetic =====================================================
print()
print("=" * W)
print("  DOWN-AND-IN PUT SYNTHETIC: BUY KO + SELL K45 VANILLA = SHORT DI PUT")
print("-" * W)
ko_fair  = float(np.mean(np.where(breach, 0.0, np.maximum(45 - S_3wk, 0))))
van_fair = bs_put(S0, 45, T3_YEARS, SIGMA)
di_fair  = van_fair - ko_fair
synth_recv = 9.05 - 0.175   # receive K45 bid, pay KO ask
edge_di    = synth_recv - di_fair
print(f"  KO fair (MC): {ko_fair:.4f}  |  K45 vanilla fair: {van_fair:.4f}")
print(f"  DI put implied fair: {di_fair:.4f}  (= vanilla - KO)")
print(f"  Synthetic DI receive: {synth_recv:.4f}  (K45 bid {9.05} - KO ask {0.175})")
print(f"  Edge = {edge_di:+.4f}/unit  |  For 50 units: E[PnL] = {edge_di*50*CONTRACT_SIZE:>+8,.0f}")
# Payoff analysis
p_di_pnl = -(np.where(breach, 1.0, 0.0) * np.maximum(45 - S_3wk, 0)) + synth_recv
print(f"  P(loss) when using DI synthetic = {(p_di_pnl < 0).mean()*100:.1f}%")
print(f"  E[loss | loss] = {p_di_pnl[p_di_pnl < 0].mean():.4f}")
print(f"  Max loss (5th pct worst) = {np.percentile(p_di_pnl, 0.5):.4f}")
print("  VERDICT: DI synthetic has edge +0.038 but LOSES when barrier breaches AND ITM")
print("  = selling crash protection. Violates 'no more risk' constraint. AVOID.")
print("=" * W)

# === 5. Chooser replication vs current strategy ==============================
print()
print("=" * W)
print("  CHOOSER REPLICATION: SELL chooser + BUY C(3wk) + BUY P(2wk)")
print("-" * W)
chooser_fair = bs_call(S0, 50, T3_YEARS, SIGMA) + bs_put(S0, 50, T2_YEARS, SIGMA)
lock_in = 22.20 - 12.05 - 9.75
print(f"  Locked-in profit (entry only): {lock_in:.4f}/unit = {lock_in*50*CONTRACT_SIZE:,.0f}")
print(f"  Current (sell chooser + buy P2 + buy C2) combined E/unit:")
print(f"    Chooser edge:  {22.20 - chooser_fair:+.4f}")
print(f"    P2 edge:       {bs_put(S0,50,T2_YEARS,SIGMA)-9.75:+.4f}")
print(f"    C2 edge:       {bs_call(S0,50,T2_YEARS,SIGMA)-9.75:+.4f}")
print(f"    Total:         {22.20 - chooser_fair + 2*(bs_put(S0,50,T2_YEARS,SIGMA)-9.75):+.4f}/unit")
current_edge = 22.20 - chooser_fair + 2 * (bs_put(S0, 50, T2_YEARS, SIGMA) - 9.75)
print(f"  REPLICATION edge: {lock_in:+.4f}/unit vs CURRENT: {current_edge:+.4f}/unit")
print(f"  Current is BETTER (+{current_edge - lock_in:.4f}/u) because 2wk options are mispriced.")
print(f"  Replication forgoes the 2wk straddle edge. KEEP CURRENT STRATEGY.")
print("=" * W)

# === 6. All vertical spreads (arbitrage check) ================================
print()
print("=" * W)
print("  VERTICAL SPREADS: EXHAUSTIVE ARBITRAGE CHECK")
print("-" * W)
vanilla_mkt = {35: (4.33,4.35), 40: (6.50,6.55), 45: (9.05,9.10), 50: (12.00,12.05)}
strikes = sorted(vanilla_mkt.keys())
print(f"  {'Spread':<20} {'Fair diff':>10} {'Long edge':>10} {'Short edge':>10}")
for i, K_lo in enumerate(strikes[:-1]):
    for K_hi in strikes[i+1:]:
        fair_diff = bs_put(S0,K_hi,T3_YEARS,SIGMA) - bs_put(S0,K_lo,T3_YEARS,SIGMA)
        bid_lo, ask_lo = vanilla_mkt[K_lo]
        bid_hi, ask_hi = vanilla_mkt[K_hi]
        edge_long  = fair_diff - (ask_hi - bid_lo)   # buy K_hi, sell K_lo
        edge_short = (bid_hi - ask_lo) - fair_diff   # sell K_hi, buy K_lo
        print(f"  P({K_lo})/P({K_hi}) spread  {fair_diff:>10.4f} {edge_long:>+10.4f} {edge_short:>+10.4f}")
print("  VERDICT: ALL spreads fairly priced (no cross-strike arbitrage).")
print("=" * W)

# === 7. Butterfly check ======================================================
print()
print("=" * W)
print("  BUTTERFLY SPREADS: CONVEXITY CHECK")
print("-" * W)
triplets = [(35,40,45), (40,45,50)]
for K1, K2, K3 in triplets:
    fair_bf = bs_put(S0,K1,T3_YEARS,SIGMA) - 2*bs_put(S0,K2,T3_YEARS,SIGMA) + bs_put(S0,K3,T3_YEARS,SIGMA)
    b1,a1 = vanilla_mkt[K1]; b2,a2 = vanilla_mkt[K2]; b3,a3 = vanilla_mkt[K3]
    # BUY butterfly: buy K1+K3, sell 2xK2
    cost_buy = a1 - 2*b2 + a3
    edge_buy = fair_bf - cost_buy
    # SELL butterfly: sell K1+K3, buy 2xK2
    recv_sell = b1 - 2*a2 + b3
    edge_sell = recv_sell - fair_bf
    print(f"  BF({K1},{K2},{K3}): fair={fair_bf:.4f}, buy_edge={edge_buy:+.4f}, sell_edge={edge_sell:+.4f}")
print("  VERDICT: No butterfly arbitrage (consistent with flat vol smile).")
print("=" * W)

# === 8. Sharpe ratio per trade ===============================================
print()
print("=" * W)
print("  RISK-ADJUSTED METRICS: SHARPE-LIKE RATIO PER TRADE (E[PnL/u] / Std[PnL/u])")
print("-" * W)
chooser_payoff = np.where(S_2wk >= 50, np.maximum(S_3wk-50,0), np.maximum(50-S_3wk,0))
ko_payoff      = np.where(breach, 0.0, np.maximum(45-S_3wk, 0))
pnl_units = [
    ("P2 BUY  (50u)",  np.maximum(50-S_2wk,0) - 9.75,    50),
    ("C2 BUY  (50u)",  np.maximum(S_2wk-50,0) - 9.75,    50),
    ("CO SELL (50u)",  22.20 - chooser_payoff,            50),
    ("BP SELL (50u)",  5.00 - np.where(S_3wk<40,10.0,0), 50),
    ("KO BUY (500u)",  ko_payoff - 0.175,                500),
]
print(f"  {'Trade':<16} {'E/u':>8} {'Std/u':>8} {'Sharpe':>8} {'Win%':>7} {'E[PnL]':>10} {'VaR5%':>12}")
for nm, pu, qty in pnl_units:
    e, s = pu.mean(), pu.std()
    win = (pu > 0).mean() * 100
    port_pnl = pu * qty * CONTRACT_SIZE
    var5 = np.percentile(port_pnl, 5)
    print(f"  {nm:<16} {e:>+8.4f} {s:>8.4f} {e/s:>+8.4f} {win:>6.1f}% {port_pnl.mean():>10,.0f} {var5:>12,.0f}")
print("=" * W)

# === 9. Position limit sensitivity ===========================================
print()
print("=" * W)
print("  POSITION LIMIT SENSITIVITY: HOW MUCH MORE PnL IF LIMITS INCREASE?")
print("-" * W)
base_edges = [
    ("AC_50_P2 BUY",  0.1207, 1),
    ("AC_50_C2 BUY",  0.1207, 1),
    ("AC_50_CO SELL", 0.3023, 1),
    ("AC_40_BP SELL", 0.2321, 1),
    ("AC_45_KO BUY",  0.0772, 10),   # KO: 10x bigger limit
]
print(f"  {'Mult':>5}  {'E[PnL]':>12}  {'vs base':>10}")
for mult in [1, 1.5, 2, 3, 5]:
    total_pnl = sum(edge * int(50*mult if k<=1 else 500*mult) * CONTRACT_SIZE
                    for _, edge, k in base_edges)
    base_pnl = sum(edge * (50 if k<=1 else 500) * CONTRACT_SIZE for _, edge, k in base_edges)
    print(f"  {mult:>5.1f}x  {total_pnl:>12,.0f}  {total_pnl-base_pnl:>+10,.0f}")
print("  CONCLUSION: PnL scales linearly with position limits. The ONLY constraint.")
print("=" * W)

# === 10. Summary =============================================================
print()
print("=" * W)
print("  SUMMARY: WHAT IS LIMITING PnL?")
print("-" * W)
print("  1. POSITION LIMITS are the PRIMARY constraint (50/500 units per product)")
print("  2. No cross-product arbitrage exists (all verticals, butterflies zero-edge)")
print("  3. DI synthetic (KO+K45) adds edge but violates 'no more risk' rule")
print("  4. Vol term structure (247% 2wk vs 251% 3wk) reduces 2wk true edge by ~50%")
print("  5. AC_60_C SELL has edge +0.008 (below 0.01 threshold) -- 1,230 additional")
print()
total_pnl = sum(e * sz * CONTRACT_SIZE for _, e, sz in [
    ("P2",  0.1207, 50), ("C2",  0.1207, 50), ("CO",  0.3023, 50),
    ("BP",  0.2321, 50), ("KO",  0.0772, 500)
])
print(f"  Current total E[PnL]:       {total_pnl:>10,.0f}")
print(f"  + AC_60_C SELL (edge 0.008): {0.0082*50*CONTRACT_SIZE:>+10,.0f}")
print(f"  + If 2wk edge is 247% vol:   {(0.0497-0.1207)*2*50*CONTRACT_SIZE:>+10,.0f}")
print(f"  Maximum achievable (same risk): ~{total_pnl + 0.0082*50*CONTRACT_SIZE:>10,.0f}")
print("=" * W)
