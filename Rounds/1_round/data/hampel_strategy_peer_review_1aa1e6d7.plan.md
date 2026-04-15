---
name: Hampel Strategy Peer Review
overview: A comprehensive peer review of the Hampel-Storm Fade strategy for ASH_COATED_OSMIUM, auditing filter selection, regime topology, local reversion edge, and mechanical risks against the raw data and notebook code.
todos:
  - id: fix-causal
    content: Switch Hampel filter to center=False (causal) and re-run all tests -- this is the highest priority fix
    status: pending
  - id: revalidate-ic
    content: Re-run IC, Edge, Ljung-Box, and alpha durability with causal Hampel residuals
    status: pending
  - id: spread-tax
    content: Add spread cost (2 ticks round-trip) to capacity analysis in Cell 8
    status: pending
  - id: entry-persist
    content: Add 2-tick contraction persistence requirement to Wait-and-Fade entry logic
    status: pending
  - id: grid-w
    content: Grid-search W in {9,11,13,15,17,21} on causal filter, using day -2 as holdout
    status: pending
  - id: adaptive-threshold
    content: Make n_sigma threshold adaptive using online MAD rather than fixed tick count
    status: pending
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

