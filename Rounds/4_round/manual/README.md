# Round 4 Manual: Aether Crystal Options

## Products (confirmed from competition rules)

| Product | Type | Strike | Expiry | Notes |
|---|---|---|---|---|
| AC_50_P_2 | Put | 50 | 14 days (2 weeks) | Vanilla |
| AC_50_C_2 | Call | 50 | 14 days (2 weeks) | Vanilla |
| AC_50_CO | Chooser | 50 | 21 days (3 weeks) | Buyer picks side at day 14 |
| AC_40_BP | Binary Put | 40 | 21 days (3 weeks) | Pays **10 flat** if S_T < 40, else 0 |
| AC_45_KO | KO Put | 45 | 21 days (3 weeks) | Knocked out if price **ever** falls below **35** |
| AC_50_P | Put | 50 | 21 days (3 weeks) | Vanilla |
| AC_50_C | Call | 50 | 21 days (3 weeks) | Vanilla |
| AC_45_P | Put | 45 | 21 days (3 weeks) | Vanilla |
| AC_40_P | Put | 40 | 21 days (3 weeks) | Vanilla |
| AC_35_P | Put | 35 | 21 days (3 weeks) | Vanilla |
| AC_60_C | Call | 60 | 21 days (3 weeks) | Vanilla |

---

## Files

| File | Purpose |
|---|---|
| `simulate.py` | Prices all products, prints fair value vs market, flags mispricings |
| `analysis.py` | Generates six PNG figures — paths, KO deep dive, trade PnL, portfolio risk, time-vs-price, and option-price trade signals |

---

## How to run

```bash
cd Rounds/4_round/manual

# Pricing and trade recommendations (text output)
python simulate.py

# Visual analysis + hedging study (saves 6 PNG files)
python analysis.py
```

Requires: `numpy`, `scipy`, `matplotlib` (`pip install matplotlib` if missing).

---

## Simulation model

Aether Crystal follows Geometric Brownian Motion on a discrete grid:
- **S0 = 50**, **σ = 251%**, **r = 0** (zero drift)
- **4 steps per trading day**, 252 trading days/year → dt = 1/1008
- **Step 40** = end of day 14 (chooser choice point). **Step 60** = expiry (day 21).

Pricing methods:
- **Black-Scholes (BS)** — closed-form, used for all vanilla options and the chooser
- **Monte Carlo (MC)** — 300k–500k simulated paths, used for path-dependent products (KO put, binary) and cross-validation

---

## Adjustable parameters

Set these at the top of either script before running:

| Variable | Value | Status |
|---|---|---|
| `N_SIM` | 500_000 / 300_000 | Tunable — more = more accurate but slower |
| `BINARY_PAYOFF` | 10.0 | **Confirmed** — pays 10 flat if S_T < 40 |
| `KO_BARRIER` | 35.0 | **Confirmed** — knocked out if price ever < 35 |
| `SEED` | 42 | RNG seed for reproducibility |

**`N_SIM`** — Only affects MC-priced products (KO, binary). Black-Scholes prices are unaffected. Raise if prices look unstable across runs.

**`BINARY_PAYOFF = 10` (confirmed).** Pays a fixed 10 if AC finishes below K=40 at expiry, regardless of how far below. Fair value ≈ payoff × P(S_T < 40).

**`KO_BARRIER = 35` (confirmed).** The barrier is 10 below the strike of 45. The option only knocks out if AC falls to 35 or below at any point during the 3 weeks. With a lower barrier, fewer paths get knocked out compared to the original assumption of 45 — the KO has real value and the market bid of 0.15 is likely **below** fair → **BUY** the KO.

**`SEED`** — Fixes the RNG so MC output is identical across runs. Change it to verify prices are stable; if they jump a lot across seeds, increase `N_SIM`.

---

## Trade recommendations

> **Note:** KO trade direction has flipped from initial analysis. With barrier=35 (confirmed), the KO has real value — **buy** it, don't sell. Re-run `simulate.py` for updated fair values and edges.

| Product | Action | Reason |
|---|---|---|
| AC_50_P_2 (2wk put K=50) | **BUY** | Fair > ask |
| AC_50_C_2 (2wk call K=50) | **BUY** | Fair > ask |
| AC_50_CO (chooser) | **SELL** | Market bid > fair (see chooser pricing below) |
| AC_40_BP (binary, payoff=10) | **SELL** | Fair ≈ 4.77, market bids 5.00 |
| AC_45_KO (barrier=35) | **BUY** | Fair > market ask of 0.175 |
| All 3-week vanilla options | fairly priced | ~0 edge |

---

## Chooser pricing

```
V_chooser = Call(3wk, K=50) + Put(2wk, K=50)
          = 12.027 + 9.871
          = 21.898
```

Market bids 22.20 — sell it. At the choice point (day 14) the buyer picks whichever side is ITM, so the value equals a 3-week call plus a 2-week put at the same strike.

---

## KO option risk (barrier=35, confirmed)

With the confirmed barrier of 35 (not 45 as originally assumed), significantly fewer paths knock out — the option survives more often and has material value. Run `simulate.py` to see the updated breach rate and fair value. The market ask of 0.175 is expected to be **below** fair value → **BUY**.

---

## Figures produced by analysis.py

| File | Contents |
|---|---|
| `fig1_paths_distribution.png` | 100 sample GBM paths + AC price distribution at expiry |
| `fig2_ko_analysis.png` | KO paths coloured by outcome, scenario pie chart, payoff distribution, fair value vs barrier |
| `fig3_trade_pnl.png` | Per-trade PnL distribution for all 5 recommended trades with VaR/CVaR marked |
| `fig4_portfolio_risk.png` | Portfolio PnL for 4 strategies: no KO, full, hedge A, hedge B |
| `fig5_time_option_prices_trades.png` | Fair option prices vs time with trade entry markers at day 0 |
| `fig6_option_prices_with_trades.png` | Each option's bid/ask/fair value with BUY/SELL signal markers |
