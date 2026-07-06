# Round 5 — Multi-Agent Stat-Arb Plan

## 0. Core Thesis

Round 5 introduces ~50 tradable goods at once. Spreads on most products are
narrow relative to mid-price volatility, so the marginal income from passive
market making is dominated by adverse-selection cost. We **pivot away from
market making** and bet the book on **statistical arbitrage**: only trade where
there is a structural relationship — within a single product (mean reversion),
across legs of a basket (cointegration), or across a leader/follower pair
(lead-lag). Where there is no structural edge, we sit out.

Hint integration:
- "Filter by clusters" → we group products by structural family using the
  `--- AI DATA PACKAGE ---` block at the bottom of this file (returns
  correlation, scaled covariance, lag-1 cross-autocorrelation, per-ticker
  vol/skew/kurt).
- "Same but slower" → we map lead/lag gaps to size every cross-leg signal so
  the faster leg drives the timing on the slower leg. The data here shows the
  cross-lag matrix is very weak (|ρ| ≲ 0.02 everywhere) but the autocorr
  diagonal is rich on the spike-MR family, so Agent C is built but kept dormant
  by default while Agent A monetises the diagonal directly.

## 1. Agent Archetypes

Three sub-agents trade non-overlapping product sets. There is no shared symbol
between agents, which removes the most common stat-arb failure mode (hidden
leverage on a shared leg).

### Agent A — Mean Reversion (`SpikeMRTrader`)

- **Universe.** Heavy-tailed names with strong negative lag-1 autocorrelation:
  `ROBOT_DISHES` (autocorr ≈ −0.22, kurt ≈ 20.1), `ROBOT_IRONING` (≈ −0.12,
  ≈ 8.8), `OXYGEN_SHAKE_CHOCOLATE` (≈ −0.08, ≈ 10.8),
  `OXYGEN_SHAKE_EVENING_BREATH` (≈ −0.12, ≈ 10.5).
- **Signal.** Rolling z-score of mid price vs window mean,
  `z = (mid − μ_w) / σ_w`. The previous version of this codebase used `mean =
  0` and effectively divided the price level by the level-std, which is *not* a
  z-score; it always fired in the same direction. The new implementation uses
  the proper rolling mean.
- **Entry.** Cross the spread on `|z| > z_in` (default 2.5–2.7). Sell at the
  best bid when `z > +z_in`, buy at the best ask when `z < −z_in`.
- **Exit.**
  - Soft exit: full unwind to flat when `|z| < z_exit` (default 0.4).
  - Time stop: force flat after `time_stop` ticks held (default 30).
  - Hard stop: force flat once `|z| > z_stop` *while still adverse* (default
    4.5) to cap regime-break tail loss.
- **Sizing.** `target_size` (default 8) capped by `pos_limit` (10) and by
  available `max_allowed_*_volume`.

### Agent B — Cointegration / Basket (`BasketTrader`, `ComplexPairTrader`)

#### B1. PEBBLES basket (`BasketTrader`)

- **Universe.** All 5 PEBBLES legs.
- **Structure.** `PEBBLES_XL` is anti-correlated with each non-XL leg
  (ρ ≈ −0.49 to −0.51); the non-XL legs are weakly correlated with each other
  (ρ ≈ 0.01). This is a classic "anchor leg vs basket of hedges" identity.
- **Residual.** With non-XL legs nearly diagonal in the covariance matrix,
  multivariate OLS reduces to per-leg univariate regressions:
  `α_i ≈ −Cov(XL, i) / Var(i)`. From the data block:
  | leg     | α (default) |
  |---------|-------------|
  | XS      | 0.524       |
  | S       | 0.668       |
  | M       | 0.804       |
  | L       | 0.783       |
  Residual: `ε = mid(XL) + Σ α_i · mid(i)`.
- **Trading.** Compute rolling mean and std of `ε` over a long window
  (default 200). On `|z| > z_in` (default 2.0), put on the residual (long
  residual = long XL, short non-XL by `α_i`). Soft exit at `|z| < z_exit`
  (default 0.4).
- **Risk.** Compute desired position per leg, scale globally by the most
  binding `max_allowed_*_volume`/`pos_limit` ratio so the basket stays neutral.
  Liquidation triggered if any leg reaches 90% of `pos_limit` (per the
  document-level rule: "close when any leg hits 90% cap").

#### B2. SNACKPACK complex (`ComplexPairTrader`)

- **Universe.** All 5 SNACKPACK legs.
- **Structure.** Strong pairwise cointegration:
  | pair          | ρ      | initial β |
  |---------------|--------|-----------|
  | VAN ↔ CHOC    | −0.915 | −0.883    |
  | STR ↔ PIST    | +0.913 | +1.260    |
  | STR ↔ RASP    | −0.923 | −0.875    |
  | RASP ↔ PIST   | −0.831 | −1.211    |
- **Sizing rule.** `STRAWBERRY`, `RASPBERRY`, `PISTACHIO` each appear in
  multiple pairs. We must net the desired contributions across pairs *before*
  issuing orders, otherwise we can flag-stop one pair while another still tries
  to push the same leg further. The trader computes per-pair desired position
  contribution to each leg, sums them, caps the total target to `pos_limit`,
  and then issues market orders for the delta from current position.
- **Trading.** For pair `(i, j, β)` with spread `s = mid_i − β · mid_j` and
  rolling z-score `z_s`:
  - `z_s > z_in` → short spread → contrib `−size` on `i`, `+β · size` on `j`.
  - `z_s < −z_in` → long spread → contrib `+size` on `i`, `−β · size` on `j`.
  - `|z_s| < z_exit` → contrib `0`.

### Agent C — Momentum / Lead-Lag (`LeadLagTrader`, dormant by default)

- **Theory.** Identify a leader-follower pair `(L → F)` where the lag-1
  cross-autocorrelation `Corr(r_F[t], r_L[t−1])` is large.
- **Reality check.** In the supplied data block, the lag-1 cross-autocorr off
  the diagonal is at most |ρ| ≈ 0.02 across every cluster. The cluster *is*
  rich in lag-1 autocorr on the diagonal (the spike-MR products), but that
  signal is already harvested by Agent A. We therefore **scaffold** the
  Agent C class so a future regression-derived β/threshold can be plugged in,
  but ship with `enabled=False` to avoid trading a near-zero edge.
- **When enabled.** For pair `(L, F, β, gate)`:
  - `predicted_return_F = β · last_return_L`
  - If `|predicted_return_F| > gate` and the same direction does not already
    have an open position, take liquidity in the predicted direction up to
    `target_size`, and unwind on the next tick if the prediction reverses.

## 2. Filtered-Out Universe

Per the "filter by clusters" hint, the rest of the products show
ρ ≲ 0.02 in every off-diagonal of the within-cluster correlation matrix and
near-zero lag-1 autocorr / kurt close to 0. They look like independent random
walks with no cluster-level structure: Galaxy Sounds, Sleep Pods, UV Visors,
Translators, Construction Panels, plus `MICROCHIP_CIRCLE`. The four high-vol
microchips (`OVAL`, `SQUARE`, `RECTANGLE`, `TRIANGLE`) also show
near-zero cross-correlation.

We **do not trade** these. The user constraint is to avoid limit-order market
making, so monetising the spread on a near-random-walk is off-table. Sitting
out is correct: the expected information ratio is negative once
adverse-selection cost is subtracted from the captured spread.

## 3. Risk Framework

- **Per-leg cap.** Every order respects `pos_limit` (default 10) via
  `max_allowed_buy_volume` / `max_allowed_sell_volume`.
- **Soft exit.** Every agent exits to flat on `|z| < z_exit`, never relying on
  the next entry to flip the inventory.
- **Hard stop.** Every agent has a `z_stop` band (default 4.5) and a
  `time_stop` (default 30 ticks for Spike MR, none for the basket because it
  has natural mean-reversion bound by the residual).
- **Multi-leg de-risk rule.** If any leg reaches 90% of `pos_limit`, the
  multi-leg agent (`BasketTrader` / `ComplexPairTrader`) snaps the entire
  group toward flat instead of doubling down.
- **Cross-agent isolation.** The per-cluster product universes are disjoint by
  design, so risk does not leak across agents through shared symbols.

## 4. Implementation Layers

- `trader.py` — production trader. Single source of truth for agent classes
  and per-product configs. The `Trader.run()` loop dispatches single-leg
  traders by `type(cfg)` and then runs the multi-leg coordinators
  (`BasketTrader`, `ComplexPairTrader`) once each.
- `test_bed.py` — same agent classes, but every config field is overridable
  through environment variables. Used by Bayesian optimisation as a
  per-trial parametric trader.
- `bayesian_optimisation.py` — Optuna driver. Reads `TARGET_AGENT` and selects
  an agent-specific search space (`SPIKE_MR`, `BASKET`, `COMPLEX_PAIR`).
  Launches `rust_backtester` once per day per trial, parses the
  `D+<day> ... <pnl>` summary line, and scores
  `final_pnl − stability_penalty · pnl_spread`.
- For non-coefficient sweeps (e.g., enable/disable, single A/B compare),
  drive `rust_backtester` directly from the terminal against `trader.py`.

## 5. Validation Order

1. Smoke test: `rust_backtester --trader trader.py --day 2 --products summary`
   completes without parse errors.
2. Per-agent sweep (`bayesian_optimisation.py` with `TARGET_AGENT=SPIKE_MR`)
   on Robots / Oxygen-Shakes to lock in `var_window`, `z_in`, `z_exit`,
   `time_stop`.
3. Basket alpha calibration: `TARGET_AGENT=BASKET` to refine `α_XS / α_S /
   α_M / α_L` and `var_window` together.
4. SNACKPACK complex calibration: `TARGET_AGENT=COMPLEX_PAIR` to refine the
   four pair betas and `z_in / z_exit`.
5. Backtest the union on all 3 days; check no day drives more than ~50% of
   total PnL (regime drift sanity check).

---

## 6. Per-Product Reference (raw data)

The next section is the original AI data package emitted by `dp_explore.ipynb`
(Cluster correlations, scaled covariances, lag-1 cross-autocorrelations, and
per-ticker vol/autocorr/skew/kurt). The `BasketTrader` defaults above were
derived directly from these covariances; the SNACKPACK pair betas were derived
directly from the SNACKPACK covariance/correlation block.

--- START AI DATA PACKAGE ---
{
  "Galaxy Sounds": {
    "correlations": {
      "GALAXY_SOUNDS_BLACK_HOLES": {
        "GALAXY_SOUNDS_BLACK_HOLES": 1.0,
        "GALAXY_SOUNDS_DARK_MATTER": 0.00011605300153369568,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 0.001215211301241774,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 0.0019372093477228662,
        "GALAXY_SOUNDS_SOLAR_WINDS": 0.00894170037673907
      },
      "GALAXY_SOUNDS_DARK_MATTER": {
        "GALAXY_SOUNDS_BLACK_HOLES": 0.00011605300153369568,
        "GALAXY_SOUNDS_DARK_MATTER": 1.0,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 0.007440499427896093,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 0.004495662770864101,
        "GALAXY_SOUNDS_SOLAR_WINDS": -0.001317938771841247
      },
      "GALAXY_SOUNDS_PLANETARY_RINGS": {
        "GALAXY_SOUNDS_BLACK_HOLES": 0.001215211301241774,
        "GALAXY_SOUNDS_DARK_MATTER": 0.007440499427896093,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 1.0,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 0.01297412560992671,
        "GALAXY_SOUNDS_SOLAR_WINDS": 0.00015762693112884887
      },
      "GALAXY_SOUNDS_SOLAR_FLAMES": {
        "GALAXY_SOUNDS_BLACK_HOLES": 0.0019372093477228662,
        "GALAXY_SOUNDS_DARK_MATTER": 0.004495662770864101,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 0.01297412560992671,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 1.0,
        "GALAXY_SOUNDS_SOLAR_WINDS": 0.004018003304095929
      },
      "GALAXY_SOUNDS_SOLAR_WINDS": {
        "GALAXY_SOUNDS_BLACK_HOLES": 0.00894170037673907,
        "GALAXY_SOUNDS_DARK_MATTER": -0.001317938771841247,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 0.00015762693112884887,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 0.004018003304095929,
        "GALAXY_SOUNDS_SOLAR_WINDS": 1.0
      }
    },
    "ticker_metrics": {
      "GALAXY_SOUNDS_DARK_MATTER":     {"vol_annal": 1.0016, "autocorr": -0.0116, "skew": -0.0155, "kurt": -0.0108},
      "GALAXY_SOUNDS_BLACK_HOLES":     {"vol_annal": 0.9974, "autocorr": -0.0167, "skew":  0.0064, "kurt":  0.0384},
      "GALAXY_SOUNDS_PLANETARY_RINGS": {"vol_annal": 1.0088, "autocorr": -0.0031, "skew":  0.0203, "kurt":  0.0087},
      "GALAXY_SOUNDS_SOLAR_WINDS":     {"vol_annal": 1.0087, "autocorr": -0.0073, "skew":  0.0062, "kurt":  0.0192},
      "GALAXY_SOUNDS_SOLAR_FLAMES":    {"vol_annal": 0.9996, "autocorr": -0.0120, "skew":  0.0093, "kurt": -0.0070}
    }
  },
  "Microchips": {
    "ticker_metrics": {
      "MICROCHIP_CIRCLE":    {"vol_annal": 0.9994, "autocorr": -0.0050, "skew": -0.0185, "kurt":  0.0157},
      "MICROCHIP_OVAL":      {"vol_annal": 1.4989, "autocorr": -0.0074, "skew": -0.0087, "kurt":  0.0652},
      "MICROCHIP_SQUARE":    {"vol_annal": 1.5078, "autocorr": -0.0220, "skew":  0.0066, "kurt":  0.0389},
      "MICROCHIP_RECTANGLE": {"vol_annal": 1.4991, "autocorr": -0.0025, "skew":  0.0122, "kurt": -0.0017},
      "MICROCHIP_TRIANGLE":  {"vol_annal": 1.4915, "autocorr": -0.0077, "skew":  0.0147, "kurt":  0.0136}
    }
  },
  "Pebbles": {
    "correlations": {
      "PEBBLES_L":  {"PEBBLES_L":  1.000, "PEBBLES_M":  0.011, "PEBBLES_S":  0.006, "PEBBLES_XL": -0.493, "PEBBLES_XS":  0.005},
      "PEBBLES_M":  {"PEBBLES_L":  0.011, "PEBBLES_M":  1.000, "PEBBLES_S":  0.012, "PEBBLES_XL": -0.506, "PEBBLES_XS":  0.016},
      "PEBBLES_S":  {"PEBBLES_L":  0.006, "PEBBLES_M":  0.012, "PEBBLES_S":  1.000, "PEBBLES_XL": -0.483, "PEBBLES_XS": -0.005},
      "PEBBLES_XL": {"PEBBLES_L": -0.493, "PEBBLES_M": -0.506, "PEBBLES_S": -0.483, "PEBBLES_XL":  1.000, "PEBBLES_XS": -0.475},
      "PEBBLES_XS": {"PEBBLES_L":  0.005, "PEBBLES_M":  0.016, "PEBBLES_S": -0.005, "PEBBLES_XL": -0.475, "PEBBLES_XS":  1.000}
    },
    "covariances_scaled": {
      "PEBBLES_L":  {"PEBBLES_L":  2.208, "PEBBLES_M":  0.024, "PEBBLES_S":  0.016, "PEBBLES_XL": -1.729, "PEBBLES_XS":  0.016},
      "PEBBLES_M":  {"PEBBLES_L":  0.024, "PEBBLES_M":  2.203, "PEBBLES_S":  0.029, "PEBBLES_XL": -1.772, "PEBBLES_XS":  0.050},
      "PEBBLES_S":  {"PEBBLES_L":  0.016, "PEBBLES_M":  0.029, "PEBBLES_S":  2.908, "PEBBLES_XL": -1.943, "PEBBLES_XS": -0.020},
      "PEBBLES_XL": {"PEBBLES_L": -1.729, "PEBBLES_M": -1.772, "PEBBLES_S": -1.943, "PEBBLES_XL":  5.566, "PEBBLES_XS": -2.400},
      "PEBBLES_XS": {"PEBBLES_L":  0.016, "PEBBLES_M":  0.050, "PEBBLES_S": -0.020, "PEBBLES_XL": -2.400, "PEBBLES_XS":  4.582}
    },
    "ticker_metrics": {
      "PEBBLES_XS": {"vol_annal": 2.1406, "autocorr": -0.0188, "skew": -0.0033, "kurt": 0.3201},
      "PEBBLES_S":  {"vol_annal": 1.7052, "autocorr":  0.0087, "skew":  0.0101, "kurt": 0.1041},
      "PEBBLES_M":  {"vol_annal": 1.4844, "autocorr": -0.0041, "skew":  0.0242, "kurt": 0.0482},
      "PEBBLES_L":  {"vol_annal": 1.4858, "autocorr":  0.0073, "skew": -0.0025, "kurt": 0.0456},
      "PEBBLES_XL": {"vol_annal": 2.3592, "autocorr":  0.0080, "skew":  0.0122, "kurt": 0.2389}
    }
  },
  "Robots": {
    "ticker_metrics": {
      "ROBOT_VACUUMING": {"vol_annal": 1.0057, "autocorr": -0.0079, "skew": -0.0045, "kurt":  0.0194},
      "ROBOT_MOPPING":   {"vol_annal": 1.0021, "autocorr": -0.0121, "skew":  0.0116, "kurt": -0.0076},
      "ROBOT_DISHES":    {"vol_annal": 1.7036, "autocorr": -0.2216, "skew":  0.1019, "kurt": 20.0769},
      "ROBOT_LAUNDRY":   {"vol_annal": 0.9983, "autocorr":  0.0060, "skew":  0.0165, "kurt": -0.0686},
      "ROBOT_IRONING":   {"vol_annal": 1.1766, "autocorr": -0.1209, "skew":  0.0082, "kurt":  8.8362}
    }
  },
  "Oxygen Shakes": {
    "ticker_metrics": {
      "OXYGEN_SHAKE_MORNING_BREATH": {"vol_annal": 1.0071, "autocorr": -0.0052, "skew":  0.0175, "kurt":  0.0063},
      "OXYGEN_SHAKE_EVENING_BREATH": {"vol_annal": 1.1704, "autocorr": -0.1180, "skew":  0.0353, "kurt": 10.4977},
      "OXYGEN_SHAKE_MINT":           {"vol_annal": 1.0038, "autocorr": -0.0032, "skew":  0.0128, "kurt": -0.0467},
      "OXYGEN_SHAKE_CHOCOLATE":      {"vol_annal": 1.1170, "autocorr": -0.0822, "skew": -0.0690, "kurt": 10.7643},
      "OXYGEN_SHAKE_GARLIC":         {"vol_annal": 1.0054, "autocorr": -0.0032, "skew":  0.0146, "kurt": -0.0133}
    }
  },
  "Snack Packs": {
    "correlations": {
      "SNACKPACK_CHOCOLATE":  {"CHOC": 1.000, "PIST":  0.025, "RASP":  0.031, "STR":  0.017, "VAN": -0.915},
      "SNACKPACK_VANILLA":    {"CHOC":-0.915, "PIST":  0.040, "RASP":  0.014, "STR":  0.031, "VAN":  1.000},
      "SNACKPACK_PISTACHIO":  {"CHOC": 0.025, "PIST":  1.000, "RASP": -0.831, "STR":  0.913, "VAN":  0.040},
      "SNACKPACK_RASPBERRY":  {"CHOC": 0.031, "PIST": -0.831, "RASP":  1.000, "STR": -0.923, "VAN":  0.014},
      "SNACKPACK_STRAWBERRY": {"CHOC": 0.017, "PIST":  0.913, "RASP": -0.923, "STR":  1.000, "VAN":  0.031}
    },
    "covariances_scaled": {
      "SNACKPACK_CHOCOLATE":  {"VAN": -0.395, "CHOC":  0.447, "PIST":  0.009, "RASP":  0.017, "STR":  0.009},
      "SNACKPACK_VANILLA":    {"VAN":  0.416, "CHOC": -0.395, "PIST":  0.014, "RASP":  0.007, "STR":  0.015},
      "SNACKPACK_PISTACHIO":  {"VAN":  0.014, "CHOC":  0.009, "PIST":  0.304, "RASP": -0.368, "STR":  0.383},
      "SNACKPACK_RASPBERRY":  {"VAN":  0.007, "CHOC":  0.017, "PIST": -0.368, "RASP":  0.645, "STR": -0.564},
      "SNACKPACK_STRAWBERRY": {"VAN":  0.015, "CHOC":  0.009, "PIST":  0.383, "RASP": -0.564, "STR":  0.579}
    },
    "ticker_metrics": {
      "SNACKPACK_CHOCOLATE":  {"vol_annal": 0.6685, "autocorr": -0.0310, "skew":  0.0134, "kurt":  0.0710},
      "SNACKPACK_VANILLA":    {"vol_annal": 0.6452, "autocorr": -0.0270, "skew":  0.0004, "kurt":  0.0564},
      "SNACKPACK_PISTACHIO":  {"vol_annal": 0.5518, "autocorr": -0.0250, "skew": -0.0004, "kurt": -0.0282},
      "SNACKPACK_STRAWBERRY": {"vol_annal": 0.7611, "autocorr": -0.0137, "skew": -0.0047, "kurt": -0.0185},
      "SNACKPACK_RASPBERRY":  {"vol_annal": 0.8031, "autocorr": -0.0170, "skew":  0.0020, "kurt": -0.0086}
    }
  }
}
--- END AI DATA PACKAGE ---
