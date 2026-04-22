# Robust Market-Making Strategy Plan (OSMIUM)

## Objective

Deploy a robust market-making strategy for `OSMIUM` that uses:

- Causal robust fair value estimation (`Hampel` + optional `Huber/Kalman` smoothing)
- Inventory-aware AS-style quote control
- Regime-aware volatility and adverse-selection overlays

The goal is to convert a statistically valid but economically weak taker signal into a maker strategy with better net expectancy.

---

## Current Evidence Summary

- Heavy tails and storm clustering are real in OSMIUM.
- Directional mean-reversion signal exists, but taker edge is below spread cost.
- Causal filtering is mandatory; centered filters create look-ahead bias.
- Robust dispersions (MAD-based) are more stable than variance-only assumptions.
- Student-t volatility modeling is useful, but residual diagnostics indicate remaining misspecification.

---

## Target Strategy Stack

1. **Fair Value Anchor**
   - Causal Hampel baseline (`center=False` behavior).
   - Optional smoother on top (Huber or local-trend Kalman) for cleaner quote center.
   - Keep light pull to structural anchor near `10000`.

2. **Quote Engine (AS-style, Robustified)**
   - Reservation price = robust FV - inventory lean.
   - Spread/distance driven by MAD-regime and volatility regime.
   - Multi-layer quoting (inner/mid/outer) with size scaling.

3. **Risk/Regime Overlay**
   - Storm vs calm gating.
   - Adverse-selection guard (reduce/remove inner quotes under high markout risk).
   - Inventory and quote-distance hard caps.

---

## To-Do List

## Phase 1 - Core robustness hardening

- [ ] **P1.1** Make inventory lean regime-normalized and capped (avoid over-steering in thin books).
- [ ] **P1.2** Replace fixed layer multipliers with calibrated quantile-based distances from residual distribution.
- [ ] **P1.3** Add explicit storm/calm regime gates that adjust spread, size, and quote depth.
- [ ] **P1.4** Add quote sanity protections: no crossing, min distance, and safe behavior on sparse book states.

## Phase 2 - Empirical calibration loop

- [ ] **P2.1** Fit per-regime fill probability curves from historical replay outputs.
- [ ] **P2.2** Fit per-regime adverse markout proxy and use it as an execution gate.
- [ ] **P2.3** Refit layer distances and size splits using holdout day validation.
- [ ] **P2.4** Re-estimate robust volatility inputs (Student-t / GARCH features) and test whether they improve net maker PnL.

## Phase 3 - Strategy validation and acceptance

- [ ] **P3.1** Run walk-forward evaluation (train: days -1/0, holdout: day -2).
- [ ] **P3.2** Track maker KPIs: spread capture, fill ratio, adverse selection, inventory variance, net PnL.
- [ ] **P3.3** Compare three FV anchors: Hampel-only vs Hampel+Kalman vs Hampel+Huber.
- [ ] **P3.4** Promote only if holdout performance is stable and diagnostics do not degrade in storm regimes.

---

## Acceptance Criteria

- Positive holdout net expectancy after estimated spread/adverse selection costs.
- Reduced inventory volatility versus current baseline.
- No fragile dependence on a single day or parameter set.
- Stable performance in both calm and storm regimes.

---

## Implementation Notes

- Treat Student-t `nu` and fill-law shape as empirical, regime-dependent quantities (not fixed constants).
- Use robust metrics first (MAD, quantiles, markout) and avoid purely Gaussian assumptions.
- Keep all filtering and feature computation causal.

