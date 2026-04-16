---
name: Hampel Strategy Peer Review
overview: A comprehensive peer review of the Hampel-Storm Fade strategy for ASH_COATED_OSMIUM, auditing filter selection, regime topology, local reversion edge, and mechanical risks against the raw data and notebook code.
todos:
  - id: fix-causal
    content: Switch Hampel filter to center=False (causal) and re-run all tests -- this is the highest priority fix
    status: completed
  - id: revalidate-ic
    content: Re-run IC, Edge, Ljung-Box, and alpha durability with causal Hampel residuals
    status: completed
  - id: spread-tax
    content: Add spread cost (2 ticks round-trip) to capacity analysis in Cell 8
    status: completed
  - id: entry-persist
    content: Add 2-tick contraction persistence requirement to Wait-and-Fade entry logic
    status: completed
  - id: grid-w
    content: Grid-search W in {9,11,13,15,17,21} on causal filter, using day -2 as holdout
    status: completed
  - id: adaptive-threshold
    content: Make n_sigma threshold adaptive using online MAD rather than fixed tick count
    status: completed
isProject: false
---

# Technical Verdict: Hampel-Storm Fade Strategy Audit

## Executive Summary

The pipeline demonstrates strong quantitative intuition -- correctly identifying the discrete-tick structure, fat-tailed residuals, and conditional mean-reversion opportunity. However, several methodological issues range from **minor refinements** to **critical red flags** that must be addressed before production deployment.

---

## Pillar 1: The Statistical Anchor (Filter Selection)

### Confirmations

- The Hampel Filter is a **reasonable choice** for an asset with excess kurtosis ~8.5. Its MAD-based scale estimator has a breakdown point of 50%, meaning up to half the data can be outliers without corrupting the estimate. This is strictly better than any mean/variance-based approach for fat-tailed data.
- The `n_sigma=3` threshold correctly avoids triggering on bid-ask bounce (typically 1-2 ticks) while catching genuine storm excursions.

### Red Flags

**1. The Ljung-Box "pass" is misleading and window-dependent.**

The notebook ([osmium.ipynb](Rounds/1_round/data/osmium.ipynb), Cell 4) tests Ljung-Box at lag 10 on `mid_price - hampel_mid`. But the Hampel filter with `W=17` and `center=True` uses **8 future ticks** in its computation. This creates mechanical look-ahead: the residual at tick `t` is computed using data from `t-8` to `t+8`. The Ljung-Box test on centered residuals will always show *less* autocorrelation than exists in real-time because the filter has already absorbed it.

**Verdict**: The white-noise result is an artifact of centering. In a live implementation you must use a **causal (right-aligned)** Hampel filter: `rolling(window=17, center=False)`. Re-run Ljung-Box on the causal version -- expect p to drop, revealing the true signal leakage.

**2. W=17 is not validated against alternatives.**

The window was chosen but never compared to W=9, W=13, W=21, etc. Given that:

- Per-day Markov analysis shows day -2 is structurally different (sigma2 ~0.009 vs ~0.4 for days -1/0)
- Storm durations are short (3-6 ticks from the micro-regime analysis in [alex_explore.ipynb](Rounds/1_round/data/alex_explore.ipynb))

A W=17 window is **slow relative to storm duration**. By the time 17 ticks of data accumulate around a storm, the reversion may already be underway.

### Refinement

- Switch to causal Hampel: `rolling(window=17, center=False)`
- Grid-search W in {9, 11, 13, 15, 17, 21} and evaluate:
  - Out-of-sample IC (not in-sample)
  - Ljung-Box on causal residuals
  - Mean reversion edge at each W
- Consider the **recursive Hampel** (online update) which updates median and MAD incrementally -- this is what a live trader must use anyway.

---

## Pillar 2: Regime Topology (Storm vs. Quiet)

### Confirmations

- Student's t fit with nu=2.12 is consistent with the excess kurtosis of ~8.5 (for t-distribution, kurtosis = 6/(nu-4) is undefined for nu<4, and empirical kurtosis grows as nu approaches 2). This correctly identifies the "infinite variance" property.
- Runs Test p=0.00 correctly identifies temporal clustering of storms. This is a real structural feature.

### Red Flags

**3. "Wait-and-Fade" entry (|eps_t| < |eps_{t-1}|) is fragile under infinite variance.**

With nu ~= 2, the distribution has **no finite variance**. This means:

- The probability of a storm *extending* at tick t+1 given it's active at tick t is **much higher** than for Gaussian noise
- A single-tick contraction (|eps_t| < |eps_{t-1}|) can easily be a dead-cat bounce within a continuing storm
- The Runs Test confirms this: storms cluster, so one tick of contraction within a cluster is noise, not signal

**The entry trigger needs a persistence filter**: require N consecutive contracting residuals (e.g., N=2 or 3) before entering. This is directly implied by the calibration philosophy in [analysis.md](Rounds/1_round/data/analysis.md) -- "never trust eyeballed distributions... run the stat test."

**4. The critical question: can a "Storm" become a permanent "Regime Shift"?**

From the per-day Markov analysis (Cell 17 of [alex_explore.ipynb](Rounds/1_round/data/alex_explore.ipynb)):

- Day -2: sigma2_stormy = 0.009 (no real storms)
- Day -1: sigma2_stormy = 0.423
- Day 0: sigma2_stormy = 0.381

The regime structure **changes between days**. All three CV values exceed 0.68, marked DIFFERS. This means the fair value anchor (10k) could shift on a new day. However, within each day, the raw data shows mid always stays within [-23, +23] of 10k across all 3 days. So within a single day, "permanent regime shift" is unlikely -- but the **operating environment** (storm magnitude, frequency) shifts between days.

**Practical impact**: Your fixed `n_sigma=3` threshold is calibrated to one noise level. On day -2 (quiet), nearly everything beyond 1-2 ticks is a "storm" by its standards. On day 0, 3 ticks is barely outside normal range. The threshold should be **adaptive**, not fixed.

### Refinement

- Entry persistence: require `|eps_t| < |eps_{t-1}| AND |eps_{t-1}| < |eps_{t-2}|` (2-tick contraction confirmation)
- Adaptive threshold: compute MAD online and use `n_sigma * current_MAD` rather than a fixed tick count
- Add a **storm duration guard**: if the current storm has lasted > median(historical storm duration), do not enter (the storm may be structural)

---

## Pillar 3: The Local Reversion Paradox

### Confirmations

- The finding that VR ~= 1.006 globally but conditional edge = 7.4 ticks at threshold 4 is **consistent and not contradictory**. This is the classic "conditional efficiency" result: the unconditional process is efficient, but conditional on extreme deviations, there is exploitable reversion. This is theoretically sound.

### Red Flags

**5. The Edge Test has critical look-ahead bias.**

In [osmium.ipynb](Rounds/1_round/data/osmium.ipynb), Cell 6:

```python
subset['future_return'] = subset['mid_price'].shift(-5) - subset['mid_price']
```

But `subset['hampel_residuals']` is computed using the **centered** Hampel filter which uses future data up to t+8. So the signal at tick t already knows what happens at t+1...t+8, and the "future return" at t+5 is within that look-ahead window. This is a **severe bias** that inflates the edge.

**The 7.4-tick edge and -0.49 IC are almost certainly overstated.** A real IC of -0.49 on a single financial asset would be extraordinary -- most production alpha signals have IC in the range of 0.02-0.10.

**6. The Alpha Durability test (Cell 7) suffers the same bias.**

`test_alpha_durability` computes Spearman correlation between `hampel_residuals` (centered, using future data) and `fwd_return` at various horizons. The IC "peaking at Horizon 3" is exactly where the centered filter's look-ahead window most overlaps with the forward return window. This is mechanical correlation, not alpha.

### Refinement (CRITICAL)

- **Re-run ALL edge and IC tests using a causal Hampel filter** (center=False). This is the single most important fix.
- Expected outcome: IC will drop substantially (likely to -0.05 to -0.15 range), and the edge will shrink
- The "sweet spot horizon" will likely shift from 3 to a longer horizon
- If the edge disappears entirely with causal filtering, the strategy has no alpha -- it's pure look-ahead

---

## Pillar 4: IMC Prosperity Mechanical Risks

### Red Flags

**7. Trade data is extremely sparse.**

From `trades_round_1_day_0.csv`: only 743 rows total across ALL products. ASH trades are a small subset. The `buyer` and `seller` fields are NaN (anonymized). This means:

- You cannot reverse-engineer counterparty behavior from trade data
- The "~900/day" trade frequency claimed in verification.md cannot be verified from this data alone
- Capacity analysis (Cell 8) computes `total_profit = num_trades * avg_profit`, but this assumes fills at mid price with zero spread cost, which is unrealistic

**8. The spread tax is not modeled.**

The edge calculation assumes you can enter at `mid_price` and exit at `mid_price + edge`. In reality:

- To enter long, you pay the spread (buy at ask, which is mid + half-spread)
- To exit, you sell at bid (mid - half-spread)
- Each round trip costs approximately 1 full spread (typically 2 ticks for ASH)

With a 2-tick round-trip cost, the 7.4-tick edge becomes 5.4 ticks at best (and that's before the look-ahead correction, which will reduce it further).

**9. Position limits and order sizing are not addressed.**

The notebook does not check:

- ASH position limit (50 lots in Prosperity Round 1)
- Impact of linear scaling on fill probability (larger orders are harder to fill passively)
- Queue priority (no guarantee your passive order is first in queue)

### Refinement

- Subtract 1 full spread from all edge calculations as a baseline "spread tax"
- Model fill probability: passive fills depend on order book depth, not guaranteed
- The capacity test should use actual L1 volume from the price data to estimate realistic fill rates

---

## Overall Verdict

### Ready for Production (with causal fix)

- The Hampel Filter as a fair value estimator -- robust to fat tails, appropriate for kurtosis ~8.5
- The conditional mean-reversion framework -- VR ~1 globally but exploitable at extremes is sound theory
- The storm-clustering observation (Runs Test) -- genuine structural feature of the data
- The calibration philosophy in [analysis.md](Rounds/1_round/data/analysis.md) -- excellent methodological framework

### Must Fix Before Production

- **CRITICAL**: Switch ALL computations to causal Hampel (center=False) and re-validate every metric. The current IC of -0.49 and edge of 7.4 are contaminated by look-ahead
- **CRITICAL**: The entry trigger (single-tick contraction) is too aggressive given infinite-variance storms. Add persistence filter
- **HIGH**: Subtract spread tax from edge calculations. With 2-tick spread and causal residuals, the real edge may be marginal

### Overfitting Risk Assessment

- **3-day sample**: All findings are based on 3 days of data (~30k ticks). The per-day Markov analysis shows these days have **different regime structures**. Any parameter (W=17, n_sigma=3, threshold=4, horizon=10) tuned to this sample may not generalize
- **Recommendation**: Treat day -2 as a **holdout** (it has different characteristics). Calibrate on days -1 and 0 only. If the strategy works on day -2 out-of-sample despite its different regime structure, that is strong evidence of robustness
- **The high IC is a red flag, not a confirmation**: In quant finance, an IC of -0.49 on a 3-day sample almost always indicates overfitting or look-ahead bias. A realistic target IC after fixing the causal issue is -0.05 to -0.15

### Priority Action Items

1. Fix Hampel to causal (center=False) -- invalidates or confirms the entire thesis
2. Re-run IC, Edge, and Ljung-Box on causal residuals
3. Add spread-cost adjustment to capacity analysis
4. Add entry persistence filter (2-tick contraction requirement)
5. Grid-search W on causal filter using day -2 as holdout

---

## Conclusion: Causal Revalidation Results

All six action items have been implemented in [osmium.ipynb](Rounds/1_round/data/osmium.ipynb) (cells 4-9) and executed. The results are definitive.

### 1. What the Causal Fix Revealed

The centered Hampel filter (`center=True`, W=17) used 8 future ticks in every fair value estimate. Switching to causal (`center=False`) eliminated this look-ahead. The impact:

| Metric | Centered (old) | Causal (new) | Change |
|--------|---------------|-------------|--------|
| Raw edge @ threshold 4 | 7.43 ticks | 0.03 - 0.97 ticks | **-93% to -99%** |
| IC (Spearman, all ticks) | -0.49 | -0.44 to -0.46 | Modest drop |
| Ljung-Box p(10) | 0.96 / 0.48 / 0.59 | 0.57 / 0.45 / 0.71 | Still white noise |
| Student-t nu | 2.12 (all days) | 1.72 - 1.99 | Heavier tails |
| Storm rate | ~16% | ~18% | Slightly more storms |

The centered filter's look-ahead was responsible for **over 90% of the reported edge**. The 7.4-tick conditional edge was almost entirely mechanical correlation, not exploitable alpha. The IC only dropped from -0.49 to -0.44 because directional prediction (sign of reversion) is genuinely correct -- but the magnitude of the predicted move collapsed.

### 2. What Remains Statistically Valid

Several structural findings survived the causal fix and are confirmed across all 3 days:

**Storm clustering is real.** Runs test p=0.00 on all days. Storms (outliers) are temporally clustered, not random. This is a structural feature of how the bots operate -- not an artifact.

**Mean reversion in residual increments is strong.** The variance ratio on `dResidual` (residual changes, not levels) is ~0.20 across all days. VR=0.20 indicates powerful mean reversion in how the residual evolves tick-to-tick. The old VR=1.006 on levels was misleading -- it tested the wrong thing.

**The Hampel filter produces genuine white noise residuals.** Ljung-Box p > 0.45 on all days even with causal filtering. The filter correctly separates fair value from noise without look-ahead assistance.

**The adaptive threshold is day-invariant.** Median adaptive threshold = 4.45 ticks on all 3 days, despite the per-day Markov analysis showing different regime structures. The MAD-based threshold self-calibrates.

**Tail behavior is extreme.** Student-t nu = 1.72 - 1.99 (causal), even worse than the centered nu = 2.12. With nu < 2, the distribution has no finite variance. "Storms" are not just large moves -- they are drawn from a distribution where arbitrarily large moves have non-negligible probability.

### 3. Why the Strategy Fails Economically

The Hampel-Storm Fade, as originally designed, is a **taker strategy**: it crosses the spread to enter when a storm is detected. This is unviable because:

**Edge < Spread on every combination tested.**

| Day | Horizon | Raw Edge (ticks) | Spread Cost | Net Edge |
|-----|---------|-----------------|-------------|----------|
| -1 | 3 | 0.33 | 2.0 | **-1.68** |
| -1 | 10 | 0.14 | 2.0 | **-1.86** |
| -2 | 3 | 0.53 | 2.0 | **-1.47** |
| -2 | 10 | 0.48 | 2.0 | **-1.52** |
| 0 | 5 | 0.72 | 2.0 | **-1.28** |
| 0 | 15 | 0.97 | 2.0 | **-1.03** |

The best single observation (Day 0, H=15, edge_raw=0.97) still loses ~1 tick per trade after spread. Over 46 trades, this compounds to -38 ticks.

**Grid search confirmed: no window saves the strategy.** Testing W in {9, 11, 13, 15, 17, 21} on training days (-1, 0):

| W | Mean Net Edge (H=10) | Mean IC | Mean Trades |
|---|---------------------|---------|-------------|
| 21 | -1.75 | -0.435 | 40.5 |
| 9 | -1.78 | -0.430 | 60.0 |
| 13 | -1.84 | -0.437 | 63.5 |
| 17 | -1.89 | -0.435 | 53.0 |
| 15 | -1.93 | -0.435 | 60.0 |
| 11 | -1.95 | -0.434 | 69.0 |

Best W=21 on training, holdout (day -2) confirms: edge_raw=0.31, edge_net=-1.69. The IC is remarkably stable across all windows (~-0.43), confirming the directional signal is real but the magnitude is insufficient.

**The persistence filter (2-tick contraction) works as designed** -- it reduces trade count from hundreds (1-tick trigger) to ~50-60 per day, correctly filtering out dead-cat bounces within continuing storms. But it cannot create edge where the underlying magnitude is too small.

### 4. The Core Insight

There is no contradiction between a strong IC (-0.44) and zero exploitable edge. The IC measures **directional accuracy** -- the Hampel residual correctly predicts which way the price will move ~72% of the time (at short horizons). But the **magnitude** of that correct prediction is ~0.3-0.5 ticks, while the cost of acting on it is 2 ticks.

This is the classic microstructure result: **the market is efficient enough that the edge exists below the spread**. The bots that set the prices already incorporate mean-reversion at this timescale. What remains for a taker is the scraps below transaction costs.

### 5. Three Paths Forward

The Hampel filter and storm detection are not useless -- they need to be deployed differently:

**Path A: Passive Market Making (do not cross the spread).**
Use the Hampel fair value as the center for an Avellaneda-Stoikov style market maker. Post limit orders at `FV +/- half_spread`. The residual signal tilts the reservation price: when residual is positive (price above FV), shade the ask down to attract sells. When negative, shade the bid up. This way the signal improves fill quality without paying the spread.

**Path B: Inventory Management Signal.**
Do not use the signal for entry/exit decisions. Instead, use it as a risk overlay for an existing market-making strategy. When the adaptive threshold flags a storm (`|residual| > 4.45`), widen spreads or reduce quote size. When the persistence filter confirms contraction, resume normal quoting. This monetizes the storm-clustering finding without requiring a directional edge above the spread.

**Path C: Reduce Effective Spread.**
The 2-tick spread assumption is for full taker execution (market orders). If the strategy can post limit orders 1 tick inside the spread (improve the quote), the effective round-trip cost drops to ~1 tick. At W=21, the best raw edge is ~0.97 ticks (Day 0, H=15). With a 1-tick cost, this is marginally positive. However, fill probability on improved quotes is uncertain and would need separate validation against L1 volume data.

### 6. Final Assessment

| Component | Status | Evidence |
|-----------|--------|----------|
| Hampel filter as FV estimator | **Valid** | White noise residuals on causal filter, all days |
| Storm detection via MAD threshold | **Valid** | 18% storm rate, adaptive threshold = 4.45, day-invariant |
| Storm clustering | **Valid** | Runs test p=0.00, all days |
| Directional prediction (IC) | **Valid** | IC = -0.44, consistent across days and windows |
| Mean reversion in residual | **Valid** | VR(dResidual) = 0.20, all days |
| Taker strategy ("Storm Fade") | **Rejected** | Net edge negative on all day/horizon/W combinations |
| Entry persistence filter | **Functional** | Correctly reduces false entries, but cannot overcome spread deficit |
| Adaptive threshold | **Functional** | Self-calibrates to 4.45 ticks; day-invariant despite different regime structures |

**Bottom line**: The statistical foundation is sound. The Hampel filter, storm detection, and mean-reversion signal are all real and cross-day robust. The failure is purely economic -- the signal's magnitude is below the cost of execution as a taker. Redeploying the signal as a passive market-making tilt (Path A) or risk overlay (Path B) is the recommended next step.

