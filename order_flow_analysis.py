"""
order_flow_analysis.py
======================
Prosperity-specific order book analysis functions.

Responsibilities:
- Price estimation  (mid, micro, multi-level micro)
- Order book depth  (level detection, coverage reporting)
- Order flow pattern analysis (L3 regime, spread, volume, Granger)

Imports FROM stat_utils  →  check_granger_causality
stat_utils knows nothing about order book column structure.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene, binomtest, spearmanr

from stat_utils import check_granger_causality


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Order Book Depth Detection
# ═════════════════════════════════════════════════════════════════════════════

def detect_levels(df: pd.DataFrame) -> int:
    """
    Automatically detect the maximum order book depth present in the dataframe.
    Looks for consecutive bid_price_N / ask_price_N column pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Order book dataframe with columns bid_price_1, ask_price_1, etc.

    Returns
    -------
    int
        Maximum level N for which both bid_price_N and ask_price_N exist.

    Example
    -------
    >>> detect_levels(df)
    3
    """
    level = 1
    while (f'bid_price_{level}' in df.columns and
           f'ask_price_{level}' in df.columns):
        level += 1
    return level - 1


def level_coverage(df: pd.DataFrame, threshold: float = 10.0) -> tuple[dict, int]:
    """
    Report population rate for each detected order book level and recommend
    the number of levels to use in micro price calculations.

    A level is considered usable if both bid and ask are populated above
    `threshold` percent of rows.

    Parameters
    ----------
    df        : Order book dataframe.
    threshold : Minimum population % for a level to be considered usable.
                Default 10.0%.

    Returns
    -------
    coverage    : dict  {level: (bid_pct, ask_pct, use: bool)}
    recommended : int   highest level that meets the threshold on both sides.

    Example
    -------
    >>> coverage, n = level_coverage(df)
    Detected 3 order book levels
      Level    Bid %   Ask %    Use?
      L1      100.0%  100.0%      ✅
      L2      100.0%  100.0%      ✅
      L3        2.6%    2.6%      ❌
    Recommended n_levels: 2
    """
    max_levels  = detect_levels(df)
    coverage    = {}
    recommended = 0

    print(f"Detected {max_levels} order book levels\n")
    print(f"  {'Level':<8} {'Bid %':>8} {'Ask %':>8} {'Use?':>8}")
    print(f"  {'-' * 36}")

    for i in range(1, max_levels + 1):
        bid_pct = df[f'bid_price_{i}'].notna().mean() * 100
        ask_pct = df[f'ask_price_{i}'].notna().mean() * 100
        use     = bid_pct > threshold and ask_pct > threshold
        if use:
            recommended = i
        coverage[i] = (bid_pct, ask_pct, use)
        print(f"  L{i:<7} {bid_pct:>7.1f}% {ask_pct:>7.1f}% {'✅' if use else '❌':>8}")

    print(f"\n  Recommended n_levels: {recommended}")
    return coverage, recommended


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Price Estimation
# ═════════════════════════════════════════════════════════════════════════════

def mid_price(df: pd.DataFrame) -> pd.Series:
    """
    Mid price using best bid and ask (L1 only).

        mid = (bid_price_1 + ask_price_1) / 2

    Parameters
    ----------
    df : Order book dataframe.

    Returns
    -------
    pd.Series of mid prices.
    """
    return (df['bid_price_1'] + df['ask_price_1']) / 2


def micro_price(df: pd.DataFrame) -> pd.Series:
    """
    Micro price using best bid/ask prices and volumes (L1 only).

    Volume-weighted mid that adjusts for order book imbalance:

        micro = ask_1 * (bid_v1 / total_v) + bid_1 * (ask_v1 / total_v)

    When bid volume > ask volume, micro > mid (upward pressure), and vice versa.
    Returns NaN where volumes are missing or total volume is zero.

    Parameters
    ----------
    df : Order book dataframe.

    Returns
    -------
    pd.Series of micro prices.
    """
    bid_p, ask_p = df['bid_price_1'], df['ask_price_1']
    bid_v, ask_v = df['bid_volume_1'], df['ask_volume_1']

    total_v = bid_v + ask_v
    valid   = total_v > 0

    micro         = pd.Series(np.nan, index=df.index)
    micro[valid]  = (
        ask_p[valid] * (bid_v[valid] / total_v[valid])
        + bid_p[valid] * (ask_v[valid] / total_v[valid])
    )
    return micro


def multi_level_micro_price(df: pd.DataFrame, n_levels: int = None) -> pd.Series:
    """
    Deep micro price aggregated across multiple order book levels.

    Formula (Weighted across N levels):
        numerator   = Σ (bid_vol_i * ask_price_i + ask_vol_i * bid_price_i)
        denominator = Σ (bid_vol_i + ask_vol_i)
        micro       = numerator / denominator

    Only levels with population > threshold are used (via level_coverage).
    Prices at missing levels are filled with 0; since their volumes are also 0,
    their contribution to numerator and denominator is exactly 0 — safe.

    Parameters
    ----------
    df       : Order book dataframe.
    n_levels : Number of levels to include. If None, auto-detected via
               level_coverage with threshold=10%.

    Returns
    -------
    pd.Series of deep micro prices.

    Example
    -------
    >>> df['micro'] = multi_level_micro_price(df)          # auto-detect
    >>> df['micro'] = multi_level_micro_price(df, n_levels=2)  # explicit
    """
    if n_levels is None:
        _, n_levels = level_coverage(df)

    numerator   = 0
    denominator = 0

    for i in range(1, n_levels + 1):
        s_bi = df[f'bid_volume_{i}'].fillna(0)
        s_ai = df[f'ask_volume_{i}'].fillna(0)
        p_bi = df[f'bid_price_{i}'].fillna(0)
        p_ai = df[f'ask_price_{i}'].fillna(0)

        numerator   += (s_bi * p_ai + s_ai * p_bi)
        denominator += (s_bi + s_ai)

    return numerator / denominator.replace(0, float('nan'))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Order Flow Pattern Analysis
# ═════════════════════════════════════════════════════════════════════════════

def order_levels_pattern(df: pd.DataFrame, n_levels: int = None) -> None:
    """
    Full statistical analysis of the deep order book pattern.

    Identifies and tests a structural Prosperity bot behaviour:
      - When an L3 order appears, L1 spread compresses toward micro price
      - L2 prices remain stable (true fair value anchor)
      - Mid price moves toward micro price during L3 events
      - L3 is always one-sided (bid only OR ask only, never both)

    Tests run:
      1.  L3 presence flags and coverage
      2.  Spread compression at L1 and L2
      3.  Individual bid/ask price movement at L1 and L2
      4.  Mid price movement and direction relative to micro
      5.  Micro price stability (variance test)
      6.  L2 volume stability (variance test)
      7.  Directional asymmetry (bid only vs ask only)
      8.  Volume correlation with spread and forward returns (Spearman)
      9.  Granger causality: does L3 flag temporally predict returns?

    Parameters
    ----------
    df       : Order book dataframe. Must contain mid_price column.
               Micro price is computed internally.
    n_levels : Levels to use for micro price. If None, auto-detected.

    Output
    ------
    Prints a formatted report with significance stars (*, **, ***).
    Summary section shows ✅/❌ per component and overall verdict.
    """
    df = df.copy()

    # ── Auto-detect levels ────────────────────────────────────────────────────
    if n_levels is None:
        _, n_levels = level_coverage(df)

    # Deepest level present in data (for pattern detection column)
    deep_level = detect_levels(df)

    # ── Prices ────────────────────────────────────────────────────────────────
    df['micro'] = multi_level_micro_price(df, n_levels=n_levels)
    df['delta'] = df['micro'] - df['mid_price']

    # ── L3 presence flags (generalised to deep_level) ─────────────────────────
    df['l_deep_bid']      = df[f'bid_price_{deep_level}'].notna().astype(int)
    df['l_deep_ask']      = df[f'ask_price_{deep_level}'].notna().astype(int)
    df['l_deep_both']     = ((df['l_deep_bid'] == 1) & (df['l_deep_ask'] == 1)).astype(int)
    df['l_deep_either']   = ((df['l_deep_bid'] == 1) | (df['l_deep_ask'] == 1)).astype(int)
    df['l_deep_bid_only'] = ((df['l_deep_bid'] == 1) & (df['l_deep_ask'] == 0)).astype(int)
    df['l_deep_ask_only'] = ((df['l_deep_bid'] == 0) & (df['l_deep_ask'] == 1)).astype(int)

    # ── Derived metrics ───────────────────────────────────────────────────────
    df['l1_spread']      = df['ask_price_1'] - df['bid_price_1']
    df['l2_spread']      = df['ask_price_2'] - df['bid_price_2']
    df['forward_return'] = df['mid_price'].diff().shift(-1)
    df['abs_fwd_return'] = df['forward_return'].abs()
    df['micro_change']   = df['micro'].diff().abs()
    df['l2_vol_change']  = (df['bid_volume_2'].diff().abs() +
                             df['ask_volume_2'].diff().abs())
    df['l1_total_vol']   = df['bid_volume_1'] + df['ask_volume_1']
    df['l2_total_vol']   = df['bid_volume_2'] + df['ask_volume_2']
    df['vol_ratio']      = df['l1_total_vol'] / df['l2_total_vol']

    # ── Deep level volume signal ──────────────────────────────────────────────
    df['l_deep_bid_vol'] = df[f'bid_volume_{deep_level}'].fillna(0)
    df['l_deep_ask_vol'] = df[f'ask_volume_{deep_level}'].fillna(0)
    df['l_deep_vol']     = df['l_deep_bid_vol'] + df['l_deep_ask_vol']
    df['l_deep_signal']  = df['l_deep_bid_vol'] - df['l_deep_ask_vol']  # +ve=buy, -ve=sell

    # ── Masks ─────────────────────────────────────────────────────────────────
    l_on  = df['l_deep_either'] == 1
    l_off = df['l_deep_either'] == 0

    # ── Internal helpers ──────────────────────────────────────────────────────
    def compare(label, col, mask_on, mask_off, test='ttest'):
        on  = df[mask_on][col].dropna()
        off = df[mask_off][col].dropna()
        if len(on) < 2 or len(off) < 2:
            print(f"  {label:<40} insufficient data")
            return None
        _, p = ttest_ind(on, off) if test == 'ttest' else levene(on, off)
        sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {label:<40} present={on.mean():>10.4f}  absent={off.mean():>10.4f}  "
              f"diff={on.mean()-off.mean():>10.4f}  p={p:.4f} {sig}")
        return p

    def spear(label, x, y):
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            print(f"  {label:<50} insufficient data")
            return None
        r, p = spearmanr(x[mask], y[mask])
        sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'
        print(f"  {label:<50} r={r:>7.4f}  p={p:.4f} {sig}")
        return p

    DL = deep_level  # shorthand for print labels

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1: Coverage
    # ═════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print(f"SECTION 1: L{DL} Coverage (deepest detected level)")
    print("=" * 80)
    print(f"  L{DL} bid only:  {df['l_deep_bid_only'].mean()*100:.2f}%  "
          f"(n={df['l_deep_bid_only'].sum()})")
    print(f"  L{DL} ask only:  {df['l_deep_ask_only'].mean()*100:.2f}%  "
          f"(n={df['l_deep_ask_only'].sum()})")
    print(f"  L{DL} both:      {df['l_deep_both'].mean()*100:.2f}%  "
          f"(n={df['l_deep_both'].sum()})")
    print(f"  L{DL} either:    {df['l_deep_either'].mean()*100:.2f}%  "
          f"(n={df['l_deep_either'].sum()})")
    print(f"  L{DL} absent:    {l_off.mean()*100:.2f}%  "
          f"(n={l_off.sum()})")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2: Spread Compression
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"SECTION 2: Spread Compression at L1 and L2 During L{DL} Events")
    print("=" * 80)
    p_spread = compare(f"L1 spread (L{DL} either)",   'l1_spread', l_on,                      l_off)
    compare(f"L2 spread (L{DL} either)",   'l2_spread', l_on,                      l_off)
    compare(f"L1 spread (L{DL} bid only)", 'l1_spread', df['l_deep_bid_only'] == 1, l_off)
    compare(f"L1 spread (L{DL} ask only)", 'l1_spread', df['l_deep_ask_only'] == 1, l_off)
    compare(f"L1 spread (L{DL} both)",     'l1_spread', df['l_deep_both'] == 1,     l_off)
    compare( "L1/L2 vol ratio",             'vol_ratio', l_on,                      l_off)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3: Individual Price Levels
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 3: Individual Bid/Ask Prices at L1 and L2")
    print("=" * 80)
    for col, label in [
        ('bid_price_1', 'L1 bid'),
        ('ask_price_1', 'L1 ask'),
        ('bid_price_2', 'L2 bid'),
        ('ask_price_2', 'L2 ask'),
    ]:
        compare(label, col, l_on, l_off)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4: Mid Price Movement
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"SECTION 4: Mid Price Movement During L{DL} Events")
    print("=" * 80)
    p_mid = compare("Abs forward return", 'abs_fwd_return', l_on, l_off)

    # Binomial test: does mid move toward micro during deep level events?
    l_rows               = df[l_on].copy()
    l_rows['delta_sign'] = np.sign(l_rows['delta'])
    l_rows['ret_sign']   = np.sign(l_rows['forward_return'])
    l_rows['correct']    = (l_rows['delta_sign'] == l_rows['ret_sign'])

    valid  = l_rows['correct'].dropna()
    n, k   = len(valid), int(valid.sum())
    binom  = binomtest(k, n, p=0.5, alternative='greater')
    b_sig  = '***' if binom.pvalue < 0.001 else '**' if binom.pvalue < 0.01 else '*' if binom.pvalue < 0.05 else ''
    print(f"  {'Mid moves toward micro':<40} rate={k/n*100:.1f}%  "
          f"(k={k}, n={n})  p={binom.pvalue:.4f} {b_sig}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5: Micro Price Stability
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"SECTION 5: Micro Price Stability During L{DL} Events")
    print("=" * 80)
    p_micro_var  = compare("Micro change (variance)", 'micro_change', l_on, l_off, test='levene')
    p_micro_mean = compare("Micro change (mean)",     'micro_change', l_on, l_off)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 6: L2 Volume Stability
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"SECTION 6: L2 Volume Stability During L{DL} Events")
    print("=" * 80)
    p_l2vol_var  = compare("L2 vol change (variance)", 'l2_vol_change', l_on, l_off, test='levene')
    p_l2vol_mean = compare("L2 vol change (mean)",     'l2_vol_change', l_on, l_off)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 7: Directional Asymmetry
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"SECTION 7: Directional Asymmetry (L{DL} Bid Only vs Ask Only)")
    print("=" * 80)
    bid_only = df[df['l_deep_bid_only'] == 1]['forward_return'].dropna()
    ask_only = df[df['l_deep_ask_only'] == 1]['forward_return'].dropna()
    absent   = df[l_off]['forward_return'].dropna()

    print(f"  {'Group':<20} {'Mean return':>14} {'n':>8}")
    print(f"  {'-' * 44}")
    print(f"  {f'L{DL} bid only':<20} {bid_only.mean():>14.6f} {len(bid_only):>8}")
    print(f"  {f'L{DL} ask only':<20} {ask_only.mean():>14.6f} {len(ask_only):>8}")
    print(f"  {'Absent':<20} {absent.mean():>14.6f} {len(absent):>8}")

    p_dir = None
    if len(bid_only) > 1 and len(ask_only) > 1:
        _, p_dir = ttest_ind(bid_only, ask_only)
        d_sig    = '***' if p_dir < 0.001 else '**' if p_dir < 0.01 else '*' if p_dir < 0.05 else ''
        print(f"\n  Bid only vs ask only p-value: {p_dir:.4f} {d_sig}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 8: Volume Correlation
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"SECTION 8: L{DL} Volume Correlation (Spearman)")
    print("=" * 80)

    l_df = df[l_on].copy()
    print(f"  [All L{DL} rows]")
    p_vol_spread  = spear(f"L{DL} total vol  vs  L1 spread",
                          l_df['l_deep_vol'], l_df['l1_spread'])
    p_vol_ret     = spear(f"L{DL} total vol  vs  abs fwd return",
                          l_df['l_deep_vol'], l_df['abs_fwd_return'])
    spear(f"L{DL} total vol  vs  L1/L2 vol ratio",
          l_df['l_deep_vol'], l_df['vol_ratio'])

    print(f"\n  [L{DL} bid-only rows]")
    bid_df = df[df['l_deep_bid_only'] == 1].copy()
    spear(f"L{DL} bid vol  vs  forward return",
          bid_df['l_deep_bid_vol'], bid_df['forward_return'])
    spear(f"L{DL} bid vol  vs  L1 spread",
          bid_df['l_deep_bid_vol'], bid_df['l1_spread'])

    print(f"\n  [L{DL} ask-only rows]")
    ask_df = df[df['l_deep_ask_only'] == 1].copy()
    spear(f"L{DL} ask vol  vs  forward return",
          ask_df['l_deep_ask_vol'], ask_df['forward_return'])
    spear(f"L{DL} ask vol  vs  L1 spread",
          ask_df['l_deep_ask_vol'], ask_df['l1_spread'])

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 9: Granger Causality
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"SECTION 9: Granger Causality (does L{DL} temporally predict returns?)")
    print("=" * 80)
    print("  Note: binary flag used — raw volume series is ~93% zeros which")
    print("  makes Granger unreliable on volume directly.\n")

    granger_flag = pd.DataFrame({
        "forward_return": df['forward_return'],
        "l_deep_flag":    df['l_deep_either'].astype(float)
    }).dropna()
    p_granger_flag = check_granger_causality(granger_flag, max_lag=4)
    g1_sig = '***' if p_granger_flag < 0.001 else '**' if p_granger_flag < 0.01 else '*' if p_granger_flag < 0.05 else '(ns)'
    print(f"  {f'L{DL} binary flag':<50} p={p_granger_flag:.4f} {g1_sig}")

    granger_dir = pd.DataFrame({
        "forward_return": df['forward_return'],
        "l_deep_signal":  df['l_deep_signal'].astype(float)
    }).dropna()
    p_granger_sig = check_granger_causality(granger_dir, max_lag=4)
    g2_sig = '***' if p_granger_sig < 0.001 else '**' if p_granger_sig < 0.01 else '*' if p_granger_sig < 0.05 else '(ns)'
    print(f"  {f'L{DL} signal (bid vol - ask vol)':<50} p={p_granger_sig:.4f} {g2_sig}")

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)

    checks = {
        f"L1 spread compresses during L{DL}":        (p_spread      is not None and p_spread      < 0.05),
        f"Mid price moves more during L{DL}":         (p_mid         is not None and p_mid         < 0.05),
        f"Micro price stays stable during L{DL}":     (p_micro_var   is not None and p_micro_var   < 0.05),
        f"L2 volume stays stable during L{DL}":       (p_l2vol_var   is not None and p_l2vol_var   < 0.05),
        f"Mid moves toward micro during L{DL}":        binom.pvalue < 0.05,
        f"L{DL} volume correlates with compression":  (p_vol_spread  is not None and p_vol_spread  < 0.05),
        f"L{DL} flag Granger-causes returns":         p_granger_flag < 0.05,
    }

    for desc, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {desc}")

    all_pass = all(checks.values())

    # Actionable verdict
    print()
    if p_granger_flag < 0.05 and p_vol_spread is not None and p_vol_spread < 0.05:
        print(f"  ✅ VOLUME SIGNAL CONFIRMED — use continuous L{DL} signal for sizing")
    elif p_granger_flag < 0.05:
        print(f"  ⚠️  USE BINARY FLAG ONLY — volume adds no additional signal")
    else:
        print(f"  ⚠️  L{DL} IS REGIME FLAG ONLY — no temporal predictive power confirmed")

    print(f"\n  {'✅ PATTERN FULLY CONFIRMED' if all_pass else '⚠️  PATTERN PARTIALLY CONFIRMED'}")
    print("=" * 80)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Trade & Execution Analysis (Spoofing vs. Real)
# ═════════════════════════════════════════════════════════════════════════════

def analyze_deep_level_spoofing(ob_df: pd.DataFrame, trade_df: pd.DataFrame, deep_level: int = None) -> dict:
    """
    Cross-references order book volume withdrawals against the trade logs 
    to calculate the Fill Ratio of deep-level (L3) orders.

    Identifies if the bot placing deep-level orders in Prosperity is:
      A) Spoofing (placing and cancelling orders to manipulate the mid-price)
      B) Providing real liquidity (orders are actually getting filled by market participants)

    Parameters
    ----------
    ob_df      : pd.DataFrame
                 Order book dataframe with timestamps, prices, and volumes.
    trade_df   : pd.DataFrame
                 Trade execution logs containing 'timestamp', 'price', and 'quantity' (or 'volume').
    deep_level : int
                 The order book level to analyze. If None, auto-detected.

    Returns
    -------
    dict
        Dictionary containing Fill Ratio, Total Volume Withdrawn, and Total Executed Volume.
    """
    if deep_level is None:
        deep_level = detect_levels(ob_df)
    
    DL = deep_level
    print("=" * 80)
    print(f"SECTION 4: Spoofing & Trade Execution Analysis (L{DL})")
    print("=" * 80)

    # Standardize trade dataframe volume column (Prosperity logs usually use 'quantity')
    trade_vol_col = 'quantity' if 'quantity' in trade_df.columns else 'volume'
    if trade_vol_col not in trade_df.columns:
        print("  ❌ ERROR: Trade dataframe must contain 'quantity' or 'volume' column.")
        return {}

    # Isolate relevant columns to prevent altering the main dataframe
    df = ob_df[['timestamp', f'bid_price_{DL}', f'ask_price_{DL}', f'bid_volume_{DL}', f'ask_volume_{DL}']].copy()
    
    # Calculate step-to-step volume differences
    df['bid_vol_delta'] = df[f'bid_volume_{DL}'].fillna(0).diff()
    df['ask_vol_delta'] = df[f'ask_volume_{DL}'].fillna(0).diff()

    # Identify rows where volume was WITHDRAWN (negative delta)
    bid_withdrawals = df[df['bid_vol_delta'] < 0].copy()
    ask_withdrawals = df[df['ask_vol_delta'] < 0].copy()

    # Format dataframes to merge with trade logs
    bid_withdrawals['price'] = bid_withdrawals[f'bid_price_{DL}']
    bid_withdrawals['withdrawn_vol'] = bid_withdrawals['bid_vol_delta'].abs()
    
    ask_withdrawals['price'] = ask_withdrawals[f'ask_price_{DL}']
    ask_withdrawals['withdrawn_vol'] = ask_withdrawals['ask_vol_delta'].abs()

    all_withdrawals = pd.concat([
        bid_withdrawals[['timestamp', 'price', 'withdrawn_vol']], 
        ask_withdrawals[['timestamp', 'price', 'withdrawn_vol']]
    ])

    # Calculate total theoretical volume that disappeared from the deep level
    total_withdrawn = all_withdrawals['withdrawn_vol'].sum()

    # Merge withdrawals with actual trades that occurred at the same timestamp and price
    matched_trades = pd.merge(
        all_withdrawals, 
        trade_df[['timestamp', 'price', trade_vol_col]], 
        on=['timestamp', 'price'], 
        how='inner'
    )

    # Calculate total volume that was actually filled at that specific timestamp and price
    total_executed = matched_trades[trade_vol_col].sum() if not matched_trades.empty else 0

    # Calculate Fill Ratio
    fill_ratio = (total_executed / total_withdrawn) if total_withdrawn > 0 else 0

    print(f"  Total L{DL} Volume Withdrawn : {total_withdrawn:,.0f}")
    print(f"  Total L{DL} Volume Executed  : {total_executed:,.0f}")
    print(f"  Fill Ratio                 : {fill_ratio:.2%}")
    print("-" * 80)

    # Actionable Verdict based on mathematical threshold
    if total_withdrawn == 0:
        print(f"  ⚠️  INSUFFICIENT DATA: No L{DL} order withdrawals detected.")
    elif fill_ratio < 0.05:
        print(f"  🚨 VERDICT: HIGH PROBABILITY SPOOFING")
        print(f"  L{DL} orders vanish without being hit. FADE THE SPIKE.")
    elif fill_ratio > 0.40:
        print(f"  ✅ VERDICT: REAL LIQUIDITY PROVIDER")
        print(f"  L{DL} orders are being executed. DO NOT FADE, follow the flow.")
    else:
        print(f"  ⚠️  VERDICT: MIXED REGIME")
        print(f"  Partial fills detected. Treat L{DL} as a soft support/resistance wall.")
        
    print("=" * 80)

    return {
        "fill_ratio": fill_ratio,
        "total_withdrawn": total_withdrawn,
        "total_executed": total_executed
    }