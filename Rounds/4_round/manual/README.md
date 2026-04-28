# Round 4 Manual: Aether Crystal Options

## Products (confirmed from competition rules)

| Product | Type | Strike | Expiry | Position limit | Notes |
|---|---|---|---|---|---|
| AC_50_P_2 | Put | 50 | 14 days (2 weeks) | 50 | Vanilla |
| AC_50_C_2 | Call | 50 | 14 days (2 weeks) | 50 | Vanilla |
| AC_50_CO | Chooser | 50 | 21 days (3 weeks) | 50 | Buyer picks side at day 14 |
| AC_40_BP | Binary Put | 40 | 21 days (3 weeks) | 50 | Pays **10 flat** if S_T < 40, else 0 |
| AC_45_KO | KO Put | 45 | 21 days (3 weeks) | 500 | Knocked out if price **ever** falls below **35** |
| AC_50_P | Put | 50 | 21 days (3 weeks) | 50 | Vanilla |
| AC_50_C | Call | 50 | 21 days (3 weeks) | 50 | Vanilla |
| AC_45_P | Put | 45 | 21 days (3 weeks) | 50 | Vanilla |
| AC_40_P | Put | 40 | 21 days (3 weeks) | 50 | Vanilla |
| AC_35_P | Put | 35 | 21 days (3 weeks) | 50 | Vanilla |
| AC_60_C | Call | 60 | 21 days (3 weeks) | 50 | Vanilla |

---

## Files

| File | Purpose |
|---|---|
| `simulate.py` | Prices all products, prints fair value vs market, flags mispricings |
| `analysis.py` | Generates six PNG figures (paths, KO deep-dive, trade PnL, portfolio risk, time, signals) |
| `deep_analysis.py` | Cross-product arbitrage checks (verticals, butterflies, DI synthetic) |
| `risk_averse_optimizer.py` | Older candidate-portfolio comparison (kept for reference) |
| **`optimal_strategy.py`** | **Authoritative**: efficient-frontier + tier recommendations |

---

## How to run

```bash
cd Rounds/4_round/manual
python optimal_strategy.py        # quiet: trade list + key risk numbers (~50s)
python optimal_strategy.py -v     # verbose: full diagnostic tables + fig8
python simulate.py                # per-product fair value + edges
python analysis.py                # 6 PNG figures
python deep_analysis.py           # arbitrage checks
```

Requires: `numpy`, `scipy`, `matplotlib`.

---

## Simulation model

Aether Crystal follows discrete-monitored Geometric Brownian Motion:
- **S0 = 50**, **σ_model = 251%**, **r = 0** (zero drift, martingale)
- **4 steps per trading day**, 252 trading days/year → dt = 1/1008
- **Step 40** = end of day 14 (chooser choice point). **Step 60** = expiry (day 21).
- All vanillas + chooser use closed-form Black-Scholes; KO and binary by Monte Carlo (400k paths).

---

## Strategy: mathematical foundations

### Step 1 — Identify *structurally robust* trades

Volatility is uncertain. The model assumes σ = 2.51, but the market-implied 2-week
straddle vol is σ ≈ 2.47. We treat σ ∈ [2.30, 2.70] as the plausible range.

A trade is **structurally robust** iff its edge is positive across *every* sigma
in this range. From the per-vol edge sweep in `optimal_strategy.py` (Table 1):

| Trade | Worst-σ edge | Status |
|---|---|---|
| AC_45_KO **BUY** | **+0.0464** | **ROBUST** |
| AC_40_BP **SELL** | **+0.0374** | **ROBUST** |
| AC_50_CO SELL | -1.31 | vol bet (loses if σ ≥ 2.55) |
| AC_60_C SELL | -0.92 | vol bet (loses if σ ≥ 2.54) |
| 2-week straddle BUY | -0.69 | vol bet (loses if σ ≤ 2.47) |
| All vanilla SELLs | -0.6 to -0.9 | vol bets |

**Only KO BUY and BP SELL are mathematical edge plays. Everything else is a
directional bet on σ being above or below some threshold.**

### Step 2 — Bound the maximum loss

For each product the worst-case-per-unit loss is:

| Trade direction | Max loss per unit |
|---|---|
| BUY anything | ask price |
| SELL binary put (BP) | 10 − bid = $5 |
| SELL vanilla put | strike − bid |
| SELL KO put | 45 − bid |
| **SELL vanilla call / chooser** | **UNBOUNDED** ⚠ |

The current ("status quo") portfolio includes SELL chooser → unbounded
loss potential. Empirically its CVaR-5% under model = **−$7.3M** for an E[PnL]
of just **$240k**. Awful risk-adjusted profile.

### Step 3 — Mean-CVaR efficient frontier

`optimal_strategy.py` grid-searches over `(KO, BP, P35, P40)` position sizes,
computes the per-path PnL distribution under each σ scenario, and reports the
Pareto-optimal frontier in the (worst-σ E[PnL], worst-σ CVaR) plane.

### Step 4 — Three-tier recommendations

| Tier | Max-loss budget | Best portfolio | E[PnL]@2.51 | min-σ E | P(loss) | Sharpe |
|---|---:|---|---:|---:|---:|---:|
| ULTRA | $300k | KO500/BP0 | $117k | $69k | **94.8%** | 0.060 |
| LOW | $750k | KO500/BP10/P35_25 | $123k | $116k | 74.2% | 0.063 |
| MOD | $1.5M | KO500/BP25/P40_25 | $131k | $124k | 74.5% | 0.068 |
| AGGR | $3M | KO500/BP50/P35_50 | $150k | $149k | 76.3% | 0.077 |

> **Critical caveat for the ULTRA tier:** KO BUY alone has a **94.8% probability
> of losing the full $262.5k**. It is mathematically positive-EV (the rare wins
> are huge — average win ≈ $7M) but path-by-path it behaves like a lottery
> ticket. The competition realises ONE path; if you want a high *probability*
> of profit this round, pick the LOW or MOD tier instead.

### Decision matrix at $300k budget — pick by your priority

| Priority | Best portfolio | E[PnL] | Median | Sharpe | P(profit) |
|---|---|---:|---:|---:|---:|
| max E[PnL] (lottery) | KO500/BP0 | $117k | -$262k | 0.060 | **5%** |
| max Sharpe / win rate | KO50/BP10/P35_5 | $19k | $58k | 0.091 | 65% |
| max **median** PnL | KO0/BP20 | $14k | **$300k** | 0.048 | 52% |

These are different mathematically optimal portfolios for different objectives.
There is no universal "best" — the user picks by objective.

---

## Chooser pricing (reference)

```
V_chooser_t=0  =  C(3wk, K=50)  +  P(2wk, K=50)  =  12.027 + 9.871  =  21.898
Market bid     =  22.20    (chooser is overpriced by ~$0.30)
```

**Variance-minimising replication** (proof in `optimal_strategy.py` §7):
the symmetric 50/50 mix
```
SELL 1 chooser + BUY 0.5 [C(3wk) + P(3wk) + C(2wk) + P(2wk)]
```
locks in $0.40/unit at entry. Pathwise residual std ≈ $6.4/unit per chooser
because GBM is a martingale (E[S_3wk − S_2wk]=0 but Var > 0). Sharpe ≈ 0.04 →
not worth the position-limit budget it consumes.

---

## KO option pricing (barrier = 35, confirmed)

The KO put is monitored 4×/day discretely. Continuous-monitoring formulae
*overestimate* breach probability. Broadie-Glasserman-Kou correction:

```
H_eff  =  H × exp(-0.5826 × σ × √dt)  =  35 × exp(-0.0915)  ≈  31.94
```

Discrete-monitoring breach probability matches `H_eff = 31.94` continuous
formula within 0.2 percentage points in MC. Therefore the discrete barrier
**protects** the KO buyer compared to a continuously-monitored barrier — this
is real, structural value.

```
KO fair (MC, 400k paths, σ=2.51)  ≈  0.253
Market ask                         =  0.175
Edge per unit                      =  +0.078  →  BUY
```

---

## Adjustable parameters

Set at the top of `optimal_strategy.py`:

| Variable | Value | Purpose |
|---|---|---|
| `N_SIM` | 400_000 | MC paths; raise for tighter KO fair |
| `BINARY_PAYOFF` | 10.0 | Confirmed: pays 10 if S<40 |
| `KO_BARRIER` | 35.0 | Confirmed: KO if S<35 ever |
| `SIGMA_SCENARIOS` | [2.30, 2.40, 2.47, 2.49, 2.51, 2.55, 2.60, 2.70] | Plausible TRUE-vol range |
| `SIGMA_WEIGHTS` | subjective probability prior | Edit if your view differs |
| `SEED` | 42 | RNG reproducibility |

---

## Figures produced

| File | Contents |
|---|---|
| `fig1_paths_distribution.png` | 100 sample GBM paths + AC price distribution at expiry |
| `fig2_ko_analysis.png` | KO paths coloured by outcome, scenario pie chart, payoff distribution, fair value vs barrier |
| `fig3_trade_pnl.png` | Per-trade PnL distribution for the original 5 trades with VaR/CVaR |
| `fig4_portfolio_risk.png` | Portfolio PnL for 4 strategies: no KO, full, hedge A, hedge B |
| `fig5_time_option_prices_trades.png` | Fair option prices vs time with trade entry markers at day 0 |
| `fig6_option_prices_with_trades.png` | Each option's bid/ask/fair value with BUY/SELL signal markers |
| `fig7_risk_averse_comparison.png` | Older risk-averse comparison (status quo vs candidate portfolios) |
| **`fig8_optimal_strategy.png`** | **Pareto frontier, tier recommendations, vol-robustness, CVaR profiles** |
