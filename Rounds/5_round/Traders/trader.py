"""
Round 5 multi-agent statistical-arbitrage trader.

Three sub-agents share one ``Trader`` runner:

* Agent A — :class:`SpikeMRTrader` — single-name spike mean reversion on
  heavy-tailed, negatively-autocorrelated products (Robots, Oxygen Shakes).
* Agent B — :class:`BasketTrader` (PEBBLES residual) and
  :class:`ComplexPairTrader` (SNACKPACK pair complex with shared-leg netting).
* Agent C — :class:`LeadLagTrader` — momentum / lead-lag scaffold; shipped
  ``enabled=False`` because the supplied lead-lag matrix has |ρ| ≲ 0.02 across
  all cross terms.

Products that show no structural edge (Galaxy Sounds, Sleep Pods, UV Visors,
Translators, Construction Panels, ``MICROCHIP_CIRCLE``, the high-vol
microchips) are intentionally not traded — see ``plan.md``.
"""
from __future__ import annotations

import base64
import json
import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


# ─────────────────────────────────────────────────────────────────────
#  CONFIGS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ProductBaseConfig:
    symbol: str = ""
    short_code: str = ""
    pos_limit: int = 10
    var_window: int = 100


@dataclass
class SpikeMRConfig(ProductBaseConfig):
    z_in: float = 2.5
    z_exit: float = 0.4
    z_stop: float = 4.5
    time_stop: int = 30
    target_size: int = 8
    # Jump-aware linear-mean filter knobs.
    # The filter predicts next mean via: mean_t = level + slope.
    # With prior probability ``jump_prior`` we assume a new mean-regime
    # can appear; if posterior jump probability exceeds ``jump_threshold``,
    # we snap the level toward the new regime.
    jump_prior: float = 0.03
    jump_threshold: float = 0.55
    jump_var_mult: float = 9.0
    level_alpha: float = 0.20
    slope_alpha: float = 0.08
    var_alpha: float = 0.07
    min_n: int = 25


@dataclass
class BasketLegConfig:
    symbol: str
    short_code: str
    alpha: float
    pos_limit: int = 10


@dataclass
class BasketConfig:
    name: str
    short_code: str
    anchor: BasketLegConfig
    hedges: List[BasketLegConfig]
    var_window: int = 200
    z_in: float = 2.0
    z_exit: float = 0.4
    z_stop: float = 5.0
    target_size: int = 8
    deleverage_threshold: float = 0.9  # 90% of pos_limit triggers de-risk
    # Execution. Passive limit orders inside the spread capture (half_spread
    # − improve) ticks per fill instead of paying half_spread on a cross.
    # ``improve = 1`` is the structural minimum: at the touch our orders
    # queue *behind* the existing book level (rust_backtester's
    # buy_queue_remaining), so they almost never fill on a wide-spread
    # product (e.g. SNACKPACK has ~13-unit queues vs ~0.09 trades/tick).
    # Posting one tick inside puts us at a price level with empty queue, so
    # we fill on the very next opposing trade.
    passive: bool = True
    improve: int = 1
    # Selective leg execution: when True, *only* place orders for legs
    # whose individual deviation from the group mean is in the same
    # direction as the basket residual (i.e. only short legs that are
    # actually contributing to the high sum, not the legs that are
    # already trading at or below the group mean). This avoids the
    # within-group trend losses that pure 5-leg-equal sum-arb suffered
    # from on day 4 (PLANETARY_RINGS −13k while the rest of GALAXY
    # reverted as expected).
    selective_legs: bool = True


@dataclass
class PairLink:
    sym1: str
    sym2: str
    beta: float
    z_in: float = 2.0
    z_exit: float = 0.4
    z_stop: float = 5.0
    var_window: int = 200
    short_code: str = ""


@dataclass
class ComplexPairConfig:
    name: str
    short_code: str
    legs: Dict[str, int]            # symbol -> pos_limit
    leg_short_codes: Dict[str, str]
    pairs: List[PairLink]
    target_size: int = 6
    deleverage_threshold: float = 0.9
    # See BasketConfig for the rationale. SNACKPACK has the worst std/ba in
    # the universe (≈ 0.4), so passive execution one tick inside the spread
    # is structurally required to ever fill on these legs.
    passive: bool = True
    improve: int = 1


@dataclass
class LeadLagConfig(ProductBaseConfig):
    leader: str = ""
    follower: str = ""
    beta: float = 0.0
    gate: float = 0.0005     # |predicted return| threshold (in fractional return units)
    target_size: int = 4
    enabled: bool = False


@dataclass
class PerLegMRConfig:
    """Universal mean-reversion across the full Round-5 universe.

    For each product *group* (10 groups × 5 legs), we trade the per-leg
    deviation from the group mean: ``residual_i = mid_i − Σ_j mid_j / N``.
    Within a constant-sum cointegrated group, this residual *must*
    mean-revert to zero, no matter how the group level drifts. This is the
    universal alpha that scales to all 50 products without bespoke tuning.

    ``half_life`` drives an EWMA estimate of (mean, std) per leg, so state
    persistence is O(1) per leg even with 50 legs across 10 groups.

    Excluded products (``exclude``) are deferred to other agents that have
    a tighter signal — e.g. SpikeMR on the negative-ACF tails (DISHES,
    IRONING, OXYGEN_BREATH/CHOCOLATE), and the dedicated PEBBLES /
    SNACKPACK baskets that exploit the level constraint directly.
    """
    groups: Dict[str, List[str]] = field(default_factory=dict)
    leg_short_codes: Dict[str, str] = field(default_factory=dict)
    pos_limits: Dict[str, int] = field(default_factory=dict)
    half_life: float = 60.0
    min_n: int = 25
    z_in: float = 1.5            # |z| threshold to start sizing in
    z_exit: float = 0.3          # |z| threshold to flatten
    z_stop: float = 4.0
    # Trend filter — skip entries while the residual is still drifting in
    # the SAME direction as the dislocation. We measure a fast EWMA of
    # the per-tick residual delta; if its sign agrees with sign(z), the
    # residual is moving *further* from mean (trending), so we wait for
    # the turn before entering. Without this filter, days like the day-2
    # SOLAR_WINDS bleed (-7k from a -1225-tick residual drift) and the
    # day-3 PEBBLES_M bleed (-7.3k) dominate aggregate PnL.
    trend_half_life: float = 12.0
    trend_filter_thresh: float = 0.20    # |trend|/resid_std cutoff
    # Force-flatten any open position after this many ticks (timestamp
    # delta in the matching engine). We ran experiments with 60-tick
    # time-stops which truncated the day-4 SLEEP_POD winners (-35k vs
    # the no-time-stop baseline) — so the default is set very long
    # (10000 = an entire trading day) and effectively disabled. Kept
    # in place as a circuit-breaker if a leg ever gets stuck on a
    # multi-day bias.
    time_stop: int = 10000
    target_size: int = 5
    deleverage_threshold: float = 0.9
    passive: bool = True
    improve: int = 1
    exclude: List[str] = field(default_factory=list)
    enabled: bool = True


# ─────────────────────────────────────────────────────────────────────
#  PER-PRODUCT REGISTRY
# ─────────────────────────────────────────────────────────────────────

# Round-5 product universe organised by cointegrated group. Each group's
# legs sum to a near-constant (within a day):
#   PEBBLES sum   = 50_000   ± 2.8        (tightest cointegration)
#   SNACKPACK sum ≈ 50_300   ± 130
#   OXYGEN_SHAKE  ≈ 50_700   ± 400-1450
#   ROBOT         ≈ 48_800   ± 370-555
#   PANEL         ≈ 49_000   ± 500-755
#   MICROCHIP     ≈ 49_400   ± 780-820
#   GALAXY_SOUNDS ≈ 53_900   ± 500-985
#   TRANSLATOR    ≈ 49_500   ± 530-1100
#   UV_VISOR      ≈ 51_400   ± 425-915
#   SLEEP_POD     ≈ 55_100   ± 770-1540  (loosest level cointegration)
#
# All legs cap at position 10 (Round-5 default per ``rust_backtester``
# position-limit table).
PRODUCT_GROUPS: Dict[str, List[str]] = {
    "GALAXY_SOUNDS": [
        "GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
        "GALAXY_SOUNDS_SOLAR_WINDS",
    ],
    "MICROCHIP": [
        "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE",
    ],
    "OXYGEN_SHAKE": [
        "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
    ],
    "PANEL": ["PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4"],
    "PEBBLES": [
        "PEBBLES_XL", "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L",
    ],
    "ROBOT": [
        "ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY",
        "ROBOT_MOPPING", "ROBOT_VACUUMING",
    ],
    "SLEEP_POD": [
        "SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE",
    ],
    "SNACKPACK": [
        "SNACKPACK_VANILLA", "SNACKPACK_CHOCOLATE", "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY", "SNACKPACK_PISTACHIO",
    ],
    "TRANSLATOR": [
        "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_VOID_BLUE",
    ],
    "UV_VISOR": [
        "UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE",
        "UV_VISOR_RED", "UV_VISOR_YELLOW",
    ],
}


# SpikeMR is reserved for products with the strongest negative-ACF tails
# (own-leg mean reversion at the tick level). Across the day-2/3/4
# universe, ROBOT_DISHES (+21k) and OXYGEN_SHAKE_EVENING_BREATH (+6k) are
# net positive — their large jump-then-revert spikes are a structurally
# different signal than the slow group-mean residual MR. ROBOT_IRONING
# (-6k) and OXYGEN_SHAKE_CHOCOLATE (-11k) were net negative, so we let
# PerLegMR own them via the within-group residual signal instead.
SPIKE_MR_CONFIGS: Dict[str, SpikeMRConfig] = {
    "ROBOT_DISHES": SpikeMRConfig(
        symbol="ROBOT_DISHES", short_code="RBDI", pos_limit=10,
        var_window=120, z_in=2.5, z_exit=0.4, z_stop=4.5, time_stop=30, target_size=10,
    ),
    "OXYGEN_SHAKE_EVENING_BREATH": SpikeMRConfig(
        symbol="OXYGEN_SHAKE_EVENING_BREATH", short_code="OXEB", pos_limit=10,
        var_window=140, z_in=2.6, z_exit=0.4, z_stop=4.5, time_stop=40, target_size=10,
    ),
}


# ─────────────────────────────────────────────────────────────────────
#  GROUP BASKET REGISTRY  — pure sum-arb on every cointegrated group
# ─────────────────────────────────────────────────────────────────────
# Round 5 ships ten groups of five products. Within each day the legs of
# a group sum to a near-constant level (PEBBLES is locked at 50_000 with
# σ < 3; the other groups drift across days but sum is stationary
# intra-day with σ between 130 and 1500). The clean alpha is therefore
# the *basket residual* — Σ leg_mids minus its rolling mean — z-scored
# and traded as one block: when the sum is high we short every leg, when
# the sum is low we long every leg. Each leg's hedge weight α=1, so the
# trade is fully market-neutral within the group regardless of which leg
# is doing the dislocating.
#
# Per-group ``z_in`` / ``target_size`` are shared because the residual
# is z-scored before thresholding; the rolling-stats normalisation
# absorbs the vastly different residual magnitudes between groups.
#
# `var_window` of 200 ticks (~2 % of a day) tracks intraday level drift
# without lagging too far behind a regime shift.
def _basket_short_code(group: str) -> str:
    """3-char short for traderData keys (must be unique across groups)."""
    if group == "GALAXY_SOUNDS":  return "GBA"
    if group == "MICROCHIP":      return "MBA"
    if group == "OXYGEN_SHAKE":   return "OBA"
    if group == "PANEL":          return "PNB"
    if group == "PEBBLES":        return "PBA"
    if group == "ROBOT":          return "RBA"
    if group == "SLEEP_POD":      return "SLB"
    if group == "SNACKPACK":      return "SNB"
    if group == "TRANSLATOR":     return "TBA"
    if group == "UV_VISOR":       return "UVB"
    return group[:3]


def _leg_short_code(group_short: str, leg: str) -> str:
    """Per-leg short, last 2 chars of leg name appended to group short."""
    return f"{group_short}{leg.split('_')[-1][:2]}"


def _build_group_basket(group: str, legs: List[str], **overrides) -> BasketConfig:
    short = _basket_short_code(group)
    anchor = BasketLegConfig(
        symbol=legs[0],
        short_code=_leg_short_code(short, legs[0]),
        alpha=1.0, pos_limit=10,
    )
    hedges = [
        BasketLegConfig(
            symbol=l,
            short_code=_leg_short_code(short, l),
            alpha=1.0, pos_limit=10,
        )
        for l in legs[1:]
    ]
    kwargs = dict(
        name=f"{group}_BASKET",
        short_code=short,
        anchor=anchor,
        hedges=hedges,
        var_window=200,
        z_in=0.7,              # fire on small dislocations — selective leg
                               # filter handles the within-group dispersion
                               # without forcing hard 2σ thresholds.
        z_exit=0.1,
        z_stop=5.0,
        target_size=8,         # leaves headroom under pos_limit=10 for flips
    )
    kwargs.update(overrides)
    return BasketConfig(**kwargs)


# All ten group baskets — pure cointegration sum-arb. Defaults apply
# uniformly (z_in=0.7 fires on any sub-σ dislocation; selective leg
# filter restricts the trade to legs whose own deviation supports the
# residual sign).
GROUP_BASKETS: Dict[str, BasketConfig] = {
    g: _build_group_basket(g, PRODUCT_GROUPS[g])
    for g in PRODUCT_GROUPS
}

# Aliases kept for test_bed.py / optimiser compatibility.
PEBBLES_BASKET_CONFIG = GROUP_BASKETS["PEBBLES"]
SNACKPACK_BASKET_CONFIG = GROUP_BASKETS["SNACKPACK"]


# Legacy SNACKPACK pair config kept around for the test_bed/optimiser; the
# production trader uses SNACKPACK_BASKET_CONFIG above.
SNACKPACK_COMPLEX_CONFIG = ComplexPairConfig(
    name="SNACKPACK_COMPLEX",
    short_code="SP",
    legs={
        "SNACKPACK_VANILLA":    10,
        "SNACKPACK_CHOCOLATE":  10,
        "SNACKPACK_STRAWBERRY": 10,
        "SNACKPACK_RASPBERRY":  10,
        "SNACKPACK_PISTACHIO":  10,
    },
    leg_short_codes={
        "SNACKPACK_VANILLA":    "SPVA",
        "SNACKPACK_CHOCOLATE":  "SPCH",
        "SNACKPACK_STRAWBERRY": "SPST",
        "SNACKPACK_RASPBERRY":  "SPRA",
        "SNACKPACK_PISTACHIO":  "SPPI",
    },
    pairs=[
        PairLink(sym1="SNACKPACK_VANILLA",    sym2="SNACKPACK_CHOCOLATE",
                 beta=-0.883, z_in=2.0, z_exit=0.4, z_stop=5.0, var_window=200,
                 short_code="VC"),
        PairLink(sym1="SNACKPACK_STRAWBERRY", sym2="SNACKPACK_RASPBERRY",
                 beta=-0.875, z_in=2.0, z_exit=0.4, z_stop=5.0, var_window=200,
                 short_code="SR"),
    ],
    target_size=4,
)


LEADLAG_CONFIGS: Dict[str, LeadLagConfig] = {
    # Disabled placeholder — flip ``enabled=True`` once a non-trivial β is found.
}


def _short_code_for(symbol: str) -> str:
    """Compact 2-3 char prefix per leg, used as a per-leg traderData key
    namespace. Form: first 1-2 chars of group + first 2 chars of leg."""
    parts = symbol.split("_")
    head = (parts[0] + (parts[1] if len(parts) > 1 else ""))[:3].upper()
    tail = parts[-1][:2].upper()
    return head + tail


PERLEG_MR_CONFIG = PerLegMRConfig(
    groups=PRODUCT_GROUPS,
    leg_short_codes={
        sym: _short_code_for(sym)
        for legs in PRODUCT_GROUPS.values() for sym in legs
    },
    pos_limits={sym: 10 for legs in PRODUCT_GROUPS.values() for sym in legs},
    # Specialist agents stay exclusive on the legs they have a
    # *meaningfully better* signal for:
    #   * ROBOT_DISHES + OXYGEN_SHAKE_EVENING_BREATH have dominant
    #     own-leg negative-ACF spikes that SpikeMR captures faster
    #     than PerLegMR's EWMA-smoothed group residual.
    #   * PEBBLES_XL is the basket anchor and trades at the basket
    #     residual signal (vol-scaled cointegration).
    #
    # PEBBLES_S, PEBBLES_M, PEBBLES_L stay in the basket as hedges but
    # PEBBLES_XS is now also routed through PerLegMR (its basket-hedge
    # role was a -10k drain and per-leg residual MR finds independent
    # alpha there).
    exclude=[
        "ROBOT_DISHES", "OXYGEN_SHAKE_EVENING_BREATH",
        "PEBBLES_XL", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L",
    ],
    # Sizing tuned empirically across days 2–4:
    #   * target_size=5 with z_in=1.5 produced the best aggregate (+52k
    #     uplift over the basket-only baseline). Increasing to 8 added
    #     ~+10k on days 2/4 but doubled the day-3 drawdown by chasing
    #     the trend during EWMA adaptation, netting worse total PnL.
    #   * half_life=60 gives a 90-ish-tick effective lookback — fast
    #     enough to follow regime shifts but slow enough that warmup
    #     (min_n=25) is reached before the first tradable signal.
    half_life=60.0,
    min_n=25,
    z_in=1.5,
    z_exit=0.3,
    z_stop=4.0,
    target_size=5,
)


# ─────────────────────────────────────────────────────────────────────
#  TRADER UTILITIES
# ─────────────────────────────────────────────────────────────────────
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_PACK_SCALE = 100  # 0.01 precision; supports values up to ~2.1e7


def _pack_buf(buf) -> str:
    """Pack a sequence of floats as base64-encoded big-endian int32s.

    NOTE: the previous implementation used ``int16`` (range ±32767) with a
    2× scale, capping every stored value at 16383.5. That silently truncated
    values like the PEBBLES basket residual (~40_000), leaving the rolling
    stats anchored to the clamp value forever and producing nonsensical
    z-scores. int32 supports values up to ~2.1e7 with 0.01 precision.
    """
    if not buf:
        return ""
    scaled = [max(_INT32_MIN, min(_INT32_MAX, round(v * _PACK_SCALE))) for v in buf]
    return base64.b64encode(struct.pack(f">{len(scaled)}l", *scaled)).decode()


def _unpack_buf(s: str) -> list:
    if not s:
        return []
    raw = base64.b64decode(s)
    return [v / float(_PACK_SCALE) for v in struct.unpack(f">{len(raw) // 4}l", raw)]


class RollingStats:
    """Rolling mean/variance over a fixed-length window.

    Recomputes mean/var directly from the buffer each call (O(window) work,
    where window ≤ 250 in this codebase). The earlier Welford implementation
    used the algebraic ``var = (Σx² − (Σx)²/n)/(n-1)`` form, which suffers
    catastrophic cancellation when ``E[x²] ≫ Var[x]`` — the PEBBLES residual
    sits around ``38_000`` with a ~500 std, so ``Σx² ≈ (Σx)²/n`` to many
    digits and the running-sum variance collapsed to noise (z-scores
    blowing up to ~1e8 ten ticks after warm-up). Demeaning first makes the
    sum well-conditioned.

    Persists into ``new_trader_data`` under one key (``<prefix>h``).
    """

    __slots__ = ("window", "min_n", "_hk", "_buf",
                 "_mean_cache", "_var_cache", "_dirty", "_new_data")

    def __init__(self, prefix: str, window: int,
                 last_traderData: dict, new_trader_data: dict,
                 min_n: Optional[int] = None):
        self.window = max(2, window)
        if min_n is None:
            min_n = max(10, self.window // 4)
        self.min_n = min(min_n, self.window)
        self._hk = f"{prefix}h"
        self._new_data = new_trader_data

        raw = last_traderData.get(self._hk, "") if last_traderData else ""
        self._buf = deque(_unpack_buf(raw) if raw else [], maxlen=self.window)
        self._mean_cache = 0.0
        self._var_cache = 1e-8
        self._dirty = True

    def push(self, value: float) -> None:
        self._buf.append(float(value))
        self._dirty = True
        self._new_data[self._hk] = _pack_buf(self._buf)

    def _recompute(self) -> None:
        n = len(self._buf)
        if n == 0:
            self._mean_cache = 0.0
            self._var_cache = 1e-8
        else:
            m = math.fsum(self._buf) / n
            self._mean_cache = m
            if n < 2:
                self._var_cache = 1e-8
            else:
                # demean BEFORE squaring to avoid cancellation when the
                # buffer values are large vs the variance.
                ssd = math.fsum((x - m) * (x - m) for x in self._buf)
                self._var_cache = max(ssd / (n - 1), 1e-8)
        self._dirty = False

    @property
    def n(self) -> int:
        return len(self._buf)

    @property
    def mean(self) -> float:
        if self._dirty:
            self._recompute()
        return self._mean_cache

    @property
    def var(self) -> float:
        if self._dirty:
            self._recompute()
        return self._var_cache

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def zscore(self, value: float) -> float:
        if self.n < self.min_n:
            return 0.0
        return (value - self.mean) / self.std

    @property
    def warmed_up(self) -> bool:
        return self.n >= self.min_n


class EWMAStats:
    """Exponentially-weighted rolling mean & variance, O(1) state.

    Why this exists alongside ``RollingStats``:
    The universal per-leg MR engine needs rolling stats for ~50 legs. With a
    buffered RollingStats(window=100) that's 50 × 100 × 4 bytes = 20 KB of
    persisted traderData per tick, which crowds out everything else. EWMA
    keeps only ``(mean, var)`` per stat — 16 bytes × 50 = 0.8 KB total — and
    is numerically robust because we update ``mean`` first then compute the
    centered squared deviation against the *new* mean.

    Update rule (West, 1979 / Welford weighted variant):
        delta_pre  = x - mean_old
        mean_new   = mean_old + alpha * delta_pre
        delta_post = x - mean_new
        var_new    = (1 - alpha) * (var_old + alpha * delta_pre * delta_post)

    ``half_life`` is converted to ``alpha = 1 - 0.5 ** (1 / half_life)``;
    a half-life of ~50 ticks is comparable to a 100-tick simple window.
    """

    __slots__ = ("alpha", "min_n", "_mk", "_vk", "_nk",
                 "_mean", "_var", "_n", "_new_data")

    def __init__(self, prefix: str, half_life: float,
                 last_traderData: dict, new_trader_data: dict,
                 min_n: int = 25):
        self.alpha = 1.0 - 0.5 ** (1.0 / max(1.0, float(half_life)))
        self.min_n = max(2, int(min_n))
        self._mk = f"{prefix}m"
        self._vk = f"{prefix}v"
        self._nk = f"{prefix}n"
        self._new_data = new_trader_data
        if last_traderData:
            self._mean = float(last_traderData.get(self._mk, 0.0))
            self._var = float(last_traderData.get(self._vk, 0.0))
            self._n = int(last_traderData.get(self._nk, 0))
        else:
            self._mean = 0.0
            self._var = 0.0
            self._n = 0

    def push(self, value: float) -> None:
        x = float(value)
        if self._n == 0:
            self._mean = x
            self._var = 0.0
        else:
            delta_pre = x - self._mean
            self._mean += self.alpha * delta_pre
            delta_post = x - self._mean
            self._var = (1.0 - self.alpha) * (
                self._var + self.alpha * delta_pre * delta_post
            )
        self._n += 1
        self._new_data[self._mk] = self._mean
        self._new_data[self._vk] = self._var
        self._new_data[self._nk] = self._n

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return max(self._var, 1e-8)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def zscore(self, value: float) -> float:
        if self._n < self.min_n:
            return 0.0
        return (value - self._mean) / self.std

    @property
    def warmed_up(self) -> bool:
        return self._n >= self.min_n


class JumpLinearMeanFilter:
    """Online linear-mean filter with probabilistic jump detection.

    The latent mean evolves approximately linearly:
        mean_t = level_t + slope_t
    and observations are:
        x_t = mean_t + noise_t

    At each tick, we compare two hypotheses for residual ``r_t``:
      * no jump:      r_t ~ N(0, sigma^2)
      * jump regime:  r_t ~ N(0, jump_var_mult * sigma^2)
    then compute posterior jump probability via Bayes rule using
    ``jump_prior``. If posterior > ``jump_threshold``, we treat this as a
    likely new-mean event and adapt level quickly.
    """

    __slots__ = (
        "prefix",
        "jump_prior", "jump_threshold", "jump_var_mult",
        "level_alpha", "slope_alpha", "var_alpha", "min_n",
        "_lk", "_sk", "_vk", "_nk",
        "level", "slope", "var", "n", "_new_data",
    )

    def __init__(self, prefix: str, cfg: SpikeMRConfig,
                 last_traderData: dict, new_trader_data: dict):
        self.prefix = prefix
        self.jump_prior = min(0.5, max(1e-4, float(cfg.jump_prior)))
        self.jump_threshold = min(0.99, max(0.01, float(cfg.jump_threshold)))
        self.jump_var_mult = max(1.01, float(cfg.jump_var_mult))
        self.level_alpha = min(1.0, max(1e-4, float(cfg.level_alpha)))
        self.slope_alpha = min(1.0, max(1e-4, float(cfg.slope_alpha)))
        self.var_alpha = min(1.0, max(1e-4, float(cfg.var_alpha)))
        self.min_n = max(5, int(cfg.min_n))

        self._lk = f"{prefix}jl"
        self._sk = f"{prefix}js"
        self._vk = f"{prefix}jv"
        self._nk = f"{prefix}jn"
        self._new_data = new_trader_data

        if last_traderData:
            self.level = float(last_traderData.get(self._lk, 0.0))
            self.slope = float(last_traderData.get(self._sk, 0.0))
            self.var = max(1e-8, float(last_traderData.get(self._vk, 1.0)))
            self.n = int(last_traderData.get(self._nk, 0))
        else:
            self.level = 0.0
            self.slope = 0.0
            self.var = 1.0
            self.n = 0

    @staticmethod
    def _gauss_pdf(x: float, var: float) -> float:
        v = max(var, 1e-8)
        return math.exp(-0.5 * x * x / v) / math.sqrt(2.0 * math.pi * v)

    @property
    def warmed_up(self) -> bool:
        return self.n >= self.min_n

    def update(self, value: float) -> Tuple[float, float, float, float]:
        x = float(value)
        if self.n == 0:
            self.level = x
            self.slope = 0.0
            self.var = max(1.0, abs(x) * 1e-3)
            self.n = 1
            self._persist()
            return self.level, 0.0, 0.0, math.sqrt(self.var)

        pred = self.level + self.slope
        resid = x - pred
        base_var = max(self.var, 1e-8)
        jump_var = self.jump_var_mult * base_var

        like_no_jump = self._gauss_pdf(resid, base_var)
        like_jump = self._gauss_pdf(resid, jump_var)
        num = self.jump_prior * like_jump
        den = num + (1.0 - self.jump_prior) * like_no_jump
        p_jump = num / den if den > 1e-16 else self.jump_prior

        prev_level = self.level
        if p_jump >= self.jump_threshold:
            # New-regime candidate: adapt quickly but keep slope memory.
            self.level = pred + 0.90 * resid
            self.slope *= 0.5
        else:
            # Normal tracking: small update around linear prediction.
            k = self.level_alpha * (1.0 - p_jump)
            self.level = pred + k * resid
            new_drift = self.level - prev_level
            self.slope = (1.0 - self.slope_alpha) * self.slope + self.slope_alpha * new_drift

        # Residual variance update after level adjustment.
        post_resid = x - self.level
        self.var = (1.0 - self.var_alpha) * self.var + self.var_alpha * (post_resid * post_resid)
        self.var = max(self.var, 1e-8)
        self.n += 1
        self._persist()
        return self.level, p_jump, post_resid, math.sqrt(self.var)

    def _persist(self) -> None:
        self._new_data[self._lk] = self.level
        self._new_data[self._sk] = self.slope
        self._new_data[self._vk] = self.var
        self._new_data[self._nk] = self.n


# ─────────────────────────────────────────────────────────────────────
#  TRADER BASE  (single-leg utility wrapper around the order book)
# ─────────────────────────────────────────────────────────────────────
class TraderBase:
    """Per-symbol order-book wrapper.

    Multi-leg traders instantiate one ``TraderBase`` per leg purely for the
    book/position/volume helpers; they never call ``get_orders()`` on the
    base. Single-leg agents subclass and implement ``get_orders``.
    """

    def __init__(self, name: str, state: TradingState,
                 new_trader_data: dict, last_traderData: dict,
                 cfg: ProductBaseConfig):
        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.last_traderData = last_traderData
        self.cfg = cfg
        self.orders: List[Order] = []

        self.position_limit: int = cfg.pos_limit
        self.initial_position: int = state.position.get(self.name, 0)
        self.expected_position: int = self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self._get_walls()
        self.best_bid, self.best_ask = self._get_best_bid_ask()

        self.max_allowed_buy_volume = self.position_limit - self.initial_position
        self.max_allowed_sell_volume = self.position_limit + self.initial_position

    # -- order-book helpers ------------------------------------------------
    def _get_order_depth(self):
        try:
            depth: OrderDepth = self.state.order_depths[self.name]
        except KeyError:
            return {}, {}
        buys = {p: abs(v) for p, v in sorted(depth.buy_orders.items(), reverse=True)}
        sells = {p: abs(v) for p, v in sorted(depth.sell_orders.items())}
        return buys, sells

    def _get_walls(self):
        bid = max(self.mkt_buy_orders) if self.mkt_buy_orders else None
        ask = min(self.mkt_sell_orders) if self.mkt_sell_orders else None
        mid = (bid + ask) / 2 if (bid is not None and ask is not None) else None
        return bid, mid, ask

    def _get_best_bid_ask(self):
        bid = max(self.mkt_buy_orders) if self.mkt_buy_orders else None
        ask = min(self.mkt_sell_orders) if self.mkt_sell_orders else None
        return bid, ask

    # -- order placement --------------------------------------------------
    def bid(self, price: float, volume: int) -> int:
        v = max(0, min(int(abs(volume)), self.max_allowed_buy_volume))
        if v == 0:
            return 0
        self.orders.append(Order(self.name, int(price), v))
        self.max_allowed_buy_volume -= v
        self.expected_position += v
        return v

    def ask(self, price: float, volume: int) -> int:
        v = max(0, min(int(abs(volume)), self.max_allowed_sell_volume))
        if v == 0:
            return 0
        self.orders.append(Order(self.name, int(price), -v))
        self.max_allowed_sell_volume -= v
        self.expected_position -= v
        return v

    def go_to_target(self, target: int, *,
                     passive: bool = False, improve: int = 0) -> None:
        """Move the *expected* position toward ``target`` (clamped to ``±pos_limit``).

        ``passive=False`` (default): cross the spread to take liquidity. This is
        the right execution for spike-MR alpha that's bigger than the spread.

        ``passive=True``: post a limit order on our own side of the touch, so
        we *earn* the half-spread instead of paying it. This is the only way
        stat-arb on wide-spread / low-vol cointegrated pairs is structurally
        profitable (e.g. SNACKPACK has ``std/bid_ask < 0.5`` — crossing the
        spread can never recover the round-trip cost). With
        ``rust_backtester --queue-penetration=1`` (the default), a passive
        order at the touch is filled when public flow trades there.

        ``improve`` posts one tick *inside* the spread so we become the new
        best price. Only meaningful when ``passive=True``.
        """
        target = max(-self.position_limit, min(self.position_limit, int(target)))
        delta = target - self.expected_position
        if delta == 0:
            return
        if passive:
            if delta > 0 and self.best_bid is not None:
                price = self.best_bid + max(0, int(improve))
                # Never cross the ask while ostensibly being "passive".
                if self.best_ask is not None:
                    price = min(price, self.best_ask - 1)
                self.bid(price, delta)
            elif delta < 0 and self.best_ask is not None:
                price = self.best_ask - max(0, int(improve))
                if self.best_bid is not None:
                    price = max(price, self.best_bid + 1)
                self.ask(price, -delta)
        else:
            if delta > 0 and self.best_ask is not None:
                self.bid(self.best_ask, delta)
            elif delta < 0 and self.best_bid is not None:
                self.ask(self.best_bid, -delta)


# ─────────────────────────────────────────────────────────────────────
#  AGENT A — SPIKE MEAN REVERSION
# ─────────────────────────────────────────────────────────────────────
class SpikeMRTrader(TraderBase):
    """Single-name robust mean reversion.

    State persisted across ticks (per ``cfg.short_code``):
      * rolling mean / std of mid via :class:`RollingStats`
      * ``<short>he`` — held_for counter (ticks since the position went non-zero)
    """

    def __init__(self, name, state, new_trader_data, last_traderData, cfg: SpikeMRConfig):
        super().__init__(name, state, new_trader_data, last_traderData, cfg)
        self.cfg: SpikeMRConfig = cfg
        self.mean_filter = JumpLinearMeanFilter(
            cfg.short_code, cfg, last_traderData, new_trader_data
        )
        self.resid_stats = EWMAStats(
            f"{cfg.short_code}jr", half_life=max(8.0, cfg.var_window / 2.0),
            last_traderData=last_traderData, new_trader_data=new_trader_data,
            min_n=cfg.min_n,
        )
        self._held_key = f"{cfg.short_code}he"
        self.held_for: int = int(last_traderData.get(self._held_key, 0)) if last_traderData else 0

    def get_orders(self) -> Dict[str, List[Order]]:
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        mean, p_jump, resid, _ = self.mean_filter.update(self.wall_mid)
        self.resid_stats.push(resid)
        z = self.resid_stats.zscore(resid)

        cfg = self.cfg
        pos = self.expected_position

        if pos != 0:
            # In a position — manage exits only, do not flip directly to the
            # other side. High-kurtosis products (kurt > 8) routinely overshoot
            # through zero; flipping caught us repeatedly on ROBOT_DISHES /
            # OXYGEN_SHAKE_EVENING_BREATH. We exit through the dead-band first
            # and then evaluate a fresh entry on the next tick.
            soft_exit = abs(z) < cfg.z_exit
            time_stop = self.held_for >= cfg.time_stop
            adverse_zstop = (pos > 0 and z < -cfg.z_stop) or (pos < 0 and z > cfg.z_stop)

            if soft_exit or time_stop or adverse_zstop:
                self.go_to_target(0)
                self.new_trader_data[self._held_key] = 0
                return {self.name: self.orders}

            # Hold the existing position. Advance the held-for counter.
            self.new_trader_data[self._held_key] = self.held_for + 1
            return {self.name: self.orders}

        # Flat — consider a fresh entry. Wait for warm-up before trusting z.
        if not self.mean_filter.warmed_up or not self.resid_stats.warmed_up:
            self.new_trader_data[self._held_key] = 0
            return {self.name: self.orders}

        # During a likely regime jump, avoid immediately fading the move.
        if p_jump >= cfg.jump_threshold:
            self.new_trader_data[self._held_key] = 0
            return {self.name: self.orders}

        if z > cfg.z_in:
            self.go_to_target(-cfg.target_size)
            self.new_trader_data[self._held_key] = 0  # restart counter on entry
        elif z < -cfg.z_in:
            self.go_to_target(+cfg.target_size)
            self.new_trader_data[self._held_key] = 0
        else:
            self.new_trader_data[self._held_key] = 0
        return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────
#  MULTI-LEG ORDER COORDINATION
# ─────────────────────────────────────────────────────────────────────
def _scale_targets_to_limits(targets: Dict[str, float],
                             legs: Dict[str, TraderBase]) -> Dict[str, int]:
    """Scale a vector of *desired* signed positions so every leg respects
    its own ``±pos_limit``. Returns integer-rounded targets."""
    if not targets:
        return {}
    scale = 1.0
    for sym, t in targets.items():
        leg = legs[sym]
        cap = leg.position_limit
        if abs(t) > cap and cap > 0:
            scale = min(scale, cap / abs(t))
    out: Dict[str, int] = {}
    for sym, t in targets.items():
        scaled = t * scale
        out[sym] = int(round(scaled))
    return out


# ─────────────────────────────────────────────────────────────────────
#  AGENT B-1 — PEBBLES BASKET
# ─────────────────────────────────────────────────────────────────────
class BasketTrader:
    """Long/short the residual ``ε = mid(anchor) + Σ α_i · mid(hedge_i)``.

    Issues market orders for *every* leg in one shot, scaled so no leg
    exceeds its position limit.
    """

    def __init__(self, cfg: BasketConfig, state: TradingState,
                 new_trader_data: dict, last_traderData: dict):
        self.cfg = cfg
        self.state = state
        self.new_trader_data = new_trader_data
        self.last_traderData = last_traderData

        self.legs: Dict[str, TraderBase] = {}
        for leg_cfg in [cfg.anchor, *cfg.hedges]:
            base = ProductBaseConfig(symbol=leg_cfg.symbol,
                                     short_code=leg_cfg.short_code,
                                     pos_limit=leg_cfg.pos_limit,
                                     var_window=cfg.var_window)
            if leg_cfg.symbol in state.order_depths:
                self.legs[leg_cfg.symbol] = TraderBase(
                    leg_cfg.symbol, state, new_trader_data, last_traderData, base)

        self.alphas: Dict[str, float] = {cfg.anchor.symbol: 1.0,
                                         **{h.symbol: h.alpha for h in cfg.hedges}}
        self.stats = RollingStats(f"{cfg.short_code}r", cfg.var_window,
                                  last_traderData, new_trader_data)

    def _all_legs_alive(self) -> bool:
        if len(self.legs) != 1 + len(self.cfg.hedges):
            return False
        return all(leg.wall_mid is not None and leg.best_bid is not None
                   and leg.best_ask is not None for leg in self.legs.values())

    def _residual(self) -> float:
        return sum(self.alphas[s] * leg.wall_mid for s, leg in self.legs.items())

    def get_orders(self) -> Dict[str, List[Order]]:
        out: Dict[str, List[Order]] = {s: [] for s in self.legs}
        if not self._all_legs_alive():
            return out

        residual = self._residual()
        self.stats.push(residual)
        z = self.stats.zscore(residual)

        cfg = self.cfg

        # de-leverage rule: any leg above 90% of its limit forces a snap-flat
        deleverage = any(
            abs(leg.expected_position) >= cfg.deleverage_threshold * leg.position_limit
            for leg in self.legs.values())
        in_position = any(leg.expected_position != 0 for leg in self.legs.values())

        # decide signed unit size: positive = long residual
        size = 0.0
        force_cross = False  # forced flat exits cross the spread
        if deleverage:
            size = 0.0
            force_cross = True
        elif in_position and (z > cfg.z_stop or z < -cfg.z_stop):
            # Stop-loss: only exit if we already hold the spread. For tight
            # cointegrations (PEBBLES std≈3) extreme z-events ARE the alpha,
            # not a breakdown, so we must not let z_stop block fresh entries.
            size = 0.0
            force_cross = True
        elif abs(z) < cfg.z_exit:
            size = 0.0  # soft exit — still passive, we have time
        elif z > cfg.z_in:
            size = -float(cfg.target_size)
        elif z < -cfg.z_in:
            size = +float(cfg.target_size)
        else:
            # in deadband — hold current positions, no new orders
            for sym in out:
                out[sym] = self.legs[sym].orders
            return out

        # desired position on each leg = size * α (anchor α = +1, hedge α = +α_i,
        # but hedge α is *added* into the residual so the *trade* is opposite:
        #    ε = XL + Σ α_i · h_i  → long ε ⇒ +1 XL, +α_i h_i? No:
        # Long residual ⇒ buy ε = buy XL and buy +α_i hedges (so price of ε rises).
        # That means: target_pos[anchor] = +size; target_pos[hedge] = +α_i · size.
        # However α_i was derived so that XL + Σ α_i · h_i is *stationary*; the
        # hedge legs don't move with the same sign as the anchor in the data
        # (anti-correlation), so their α_i is positive but their *trade direction*
        # follows the residual sign — yes, exactly the formula above.
        targets: Dict[str, float] = {}

        # Selective leg filter: pre-compute the cross-section group mean
        # so we can decide which legs are *actually* contributing to the
        # basket dislocation and skip the legs that are already on the
        # right side of the cointegration.
        if cfg.selective_legs and not force_cross:
            n_legs = len(self.legs)
            grp_mean = (sum(leg.wall_mid for leg in self.legs.values())
                        / n_legs) if n_legs else 0.0
            # When ``size > 0`` we are *long the residual* (residual was
            # low → going to revert up). To make the residual rise, the
            # legs that need to revert UP are the ones currently BELOW
            # group mean (i.e. dev < 0). So only buy legs with dev < 0
            # and only sell legs with dev > 0 (when size < 0).
            # ``size = -target`` → short basket → only short legs above mean.
            for sym, leg in self.legs.items():
                dev = leg.wall_mid - grp_mean
                if size > 0 and dev > 0:        # leg is rich, don't long it
                    targets[sym] = 0.0
                elif size < 0 and dev < 0:      # leg is cheap, don't short it
                    targets[sym] = 0.0
                else:
                    targets[sym] = size * self.alphas[sym]
        else:
            for sym, leg in self.legs.items():
                targets[sym] = size * self.alphas[sym]

        scaled = _scale_targets_to_limits(targets, self.legs)

        passive = cfg.passive and not force_cross
        for sym, leg in self.legs.items():
            leg.go_to_target(scaled.get(sym, 0),
                             passive=passive, improve=cfg.improve)
            out[sym] = leg.orders
        return out


# ─────────────────────────────────────────────────────────────────────
#  AGENT B-2 — SNACKPACK COMPLEX PAIR (shared-leg netting)
# ─────────────────────────────────────────────────────────────────────
class ComplexPairTrader:
    """Run multiple cointegrated pairs over a shared leg universe.

    For each pair ``(i, j, β)``:
        spread = mid_i − β · mid_j
        z = rolling zscore(spread)
        if  z >  z_in: contribute (−size on i, +β·size on j)
        if  z < −z_in: contribute (+size on i, −β·size on j)
        if |z| < z_exit: contribute 0

    Contributions are summed across pairs per leg, scaled to respect every
    leg's ``pos_limit``, and then issued as market orders for the delta vs.
    current position.
    """

    def __init__(self, cfg: ComplexPairConfig, state: TradingState,
                 new_trader_data: dict, last_traderData: dict):
        self.cfg = cfg
        self.state = state
        self.new_trader_data = new_trader_data
        self.last_traderData = last_traderData

        self.legs: Dict[str, TraderBase] = {}
        for sym, lim in cfg.legs.items():
            base = ProductBaseConfig(symbol=sym,
                                     short_code=cfg.leg_short_codes[sym],
                                     pos_limit=lim,
                                     var_window=cfg.pairs[0].var_window if cfg.pairs else 200)
            if sym in state.order_depths:
                self.legs[sym] = TraderBase(sym, state, new_trader_data,
                                            last_traderData, base)

        # one rolling-stats per pair on the spread series
        self.pair_stats: Dict[str, RollingStats] = {}
        for p in cfg.pairs:
            key = f"{cfg.short_code}{p.short_code}"
            self.pair_stats[id(p)] = RollingStats(
                key, p.var_window, last_traderData, new_trader_data)

    def _all_legs_alive(self) -> bool:
        if len(self.legs) != len(self.cfg.legs):
            return False
        return all(leg.wall_mid is not None and leg.best_bid is not None
                   and leg.best_ask is not None for leg in self.legs.values())

    def get_orders(self) -> Dict[str, List[Order]]:
        out: Dict[str, List[Order]] = {s: [] for s in self.legs}
        if not self._all_legs_alive():
            return out

        cfg = self.cfg
        deleverage = any(
            abs(leg.expected_position) >= cfg.deleverage_threshold * leg.position_limit
            for leg in self.legs.values())
        in_position = any(leg.expected_position != 0 for leg in self.legs.values())

        targets: Dict[str, float] = {sym: 0.0 for sym in self.legs}
        force_cross = False  # only on emergency

        if deleverage:
            force_cross = True
            scaled = _scale_targets_to_limits(targets, self.legs)
            for sym, leg in self.legs.items():
                leg.go_to_target(scaled.get(sym, 0),
                                 passive=False, improve=cfg.improve)
                out[sym] = leg.orders
            return out

        for p in cfg.pairs:
            stats = self.pair_stats[id(p)]
            spread = (self.legs[p.sym1].wall_mid
                      - p.beta * self.legs[p.sym2].wall_mid)
            stats.push(spread)
            z = stats.zscore(spread)

            if in_position and (z > p.z_stop or z < -p.z_stop):
                # stop-loss only fires when this pair already has inventory
                # (otherwise extreme z-events are alpha, not breakdown)
                force_cross = True
                continue
            if abs(z) < p.z_exit:
                continue  # in dead-band → 0 contribution; passive exit is fine
            if z > p.z_in:
                targets[p.sym1] += -float(cfg.target_size)
                targets[p.sym2] += +p.beta * float(cfg.target_size)
            elif z < -p.z_in:
                targets[p.sym1] += +float(cfg.target_size)
                targets[p.sym2] += -p.beta * float(cfg.target_size)

        scaled = _scale_targets_to_limits(targets, self.legs)
        passive = cfg.passive and not force_cross
        for sym, leg in self.legs.items():
            leg.go_to_target(scaled.get(sym, 0),
                             passive=passive, improve=cfg.improve)
            out[sym] = leg.orders
        return out


# ─────────────────────────────────────────────────────────────────────
#  AGENT D — UNIVERSAL PER-LEG MEAN REVERSION
# ─────────────────────────────────────────────────────────────────────
class PerLegMRTrader:
    """Per-leg mean reversion against the in-group cross-sectional mean.

    Every product in Round 5 belongs to a 5-leg group whose sum is
    near-stationary intraday. So the residual

        r_i(t) = mid_i(t) − Σ_j mid_j(t) / N

    is mean-reverting by construction (the group sum is constrained, so
    any deviation in one leg must be matched by an offsetting deviation
    in the rest of the group). We trade the z-score of ``r_i`` per leg:

      * z_i > +z_in  → leg is rich vs group → go short
      * z_i < −z_in  → leg is cheap vs group → go long
      * |z_i| < z_exit → drain inventory back to flat
      * |z_i| > z_stop AND in position → emergency cross to flat

    Because trades are signed independently per leg, the strategy is
    market-neutral *within each group* by virtue of the group-sum
    constraint without any explicit hedge ratios: when one leg is rich
    by some amount, the rest of the group is collectively cheap by the
    same amount, so the per-leg signals naturally hedge each other.

    Execution: passive limit one tick inside the spread. Crosses only
    trigger on emergency stop-loss / deleverage to bound downside.

    Excluded symbols are deferred to specialist agents (SpikeMR, the
    PEBBLES/SNACKPACK basket level trades). This avoids double-stacking
    positions on the same leg.
    """

    def __init__(self, cfg: PerLegMRConfig, state: TradingState,
                 new_trader_data: dict, last_traderData: dict):
        self.cfg = cfg
        self.state = state
        self.new_trader_data = new_trader_data
        self.last_traderData = last_traderData
        self.legs: Dict[str, TraderBase] = {}
        self.stats: Dict[str, EWMAStats] = {}
        # Per-leg fast-EWMA of residual deltas (trend filter).
        self.trend: Dict[str, EWMAStats] = {}
        self._prev_resid_keys: Dict[str, str] = {}
        # Per-leg entry-tick tracker for the time-stop.
        self._entry_ts_keys: Dict[str, str] = {}

        # Build a TraderBase + EWMAStats for every leg in the universe that
        # is (a) actually present in this tick's order book and (b) not in
        # the exclusion list.
        excluded = set(cfg.exclude)
        for group_name, legs in cfg.groups.items():
            for sym in legs:
                if sym in excluded or sym not in state.order_depths:
                    continue
                base = ProductBaseConfig(
                    symbol=sym,
                    short_code=cfg.leg_short_codes.get(sym, sym[:4]),
                    pos_limit=cfg.pos_limits.get(sym, 10),
                )
                self.legs[sym] = TraderBase(
                    sym, state, new_trader_data, last_traderData, base)
                self.stats[sym] = EWMAStats(
                    f"{base.short_code}r", cfg.half_life,
                    last_traderData, new_trader_data, min_n=cfg.min_n,
                )
                self.trend[sym] = EWMAStats(
                    f"{base.short_code}d", cfg.trend_half_life,
                    last_traderData, new_trader_data,
                    min_n=max(4, int(cfg.trend_half_life / 2)),
                )
                self._prev_resid_keys[sym] = f"{base.short_code}p"
                self._entry_ts_keys[sym] = f"{base.short_code}t"

    @staticmethod
    def _group_mean(legs: List[TraderBase]) -> Optional[float]:
        mids = [leg.wall_mid for leg in legs if leg.wall_mid is not None]
        if not mids:
            return None
        return math.fsum(mids) / len(mids)

    def get_orders(self) -> Dict[str, List[Order]]:
        out: Dict[str, List[Order]] = {sym: [] for sym in self.legs}
        if not self.cfg.enabled or not self.legs:
            return out

        cfg = self.cfg
        for group_name, group_syms in cfg.groups.items():
            # Resolve only the live legs in this group.
            group_legs = [self.legs[s] for s in group_syms if s in self.legs]
            if len(group_legs) < 2:
                # Nothing to mean-revert against — at least 2 legs needed
                # so the group mean is meaningful.
                continue
            group_mean = self._group_mean(group_legs)
            if group_mean is None:
                continue

            for leg in group_legs:
                if leg.wall_mid is None or leg.best_bid is None or leg.best_ask is None:
                    continue
                residual = leg.wall_mid - group_mean
                stats = self.stats[leg.name]
                stats.push(residual)

                # Trend EWMA — feed first-difference of residuals.
                trend_stats = self.trend[leg.name]
                prev_key = self._prev_resid_keys[leg.name]
                prev_resid = self.last_traderData.get(prev_key) if self.last_traderData else None
                if prev_resid is not None:
                    trend_stats.push(residual - float(prev_resid))
                self.new_trader_data[prev_key] = residual

                if not stats.warmed_up:
                    continue
                z = stats.zscore(residual)
                # Trending = recent average residual delta is large
                # relative to the residual std AND moving in the same
                # direction as the dislocation.
                resid_std = stats.std
                if trend_stats.warmed_up and resid_std > 1e-6:
                    trend_score = trend_stats.mean / resid_std
                else:
                    trend_score = 0.0

                # Position-aware sizing
                in_position = leg.expected_position != 0
                deleverage = (
                    abs(leg.expected_position)
                    >= cfg.deleverage_threshold * leg.position_limit
                )

                # Track entry tick for the time-stop. We refresh the entry
                # timestamp on a flat→position transition; while flat it
                # remains None so a fresh trade resets the clock.
                ts_key = self._entry_ts_keys[leg.name]
                entry_ts = self.last_traderData.get(ts_key) if self.last_traderData else None
                if leg.initial_position == 0:
                    entry_ts = None

                target = leg.expected_position
                force_cross = False
                time_stopped = (
                    in_position and entry_ts is not None
                    and (self.state.timestamp - int(entry_ts)) >= cfg.time_stop
                )
                # The trend gate: if we're flat AND the residual is still
                # drifting in the dislocation's direction, wait for the
                # turn rather than catching the falling knife.
                trend_gate_short = (
                    not in_position and z > cfg.z_in
                    and trend_score > cfg.trend_filter_thresh
                )
                trend_gate_long = (
                    not in_position and z < -cfg.z_in
                    and trend_score < -cfg.trend_filter_thresh
                )

                if deleverage:
                    target = 0
                    force_cross = True
                elif in_position and abs(z) > cfg.z_stop:
                    target = 0
                    force_cross = True
                elif time_stopped:
                    target = 0
                    force_cross = True
                elif abs(z) < cfg.z_exit:
                    target = 0
                elif trend_gate_short or trend_gate_long:
                    target = 0
                elif z > cfg.z_in:
                    target = -cfg.target_size       # leg rich → short
                elif z < -cfg.z_in:
                    target = +cfg.target_size       # leg cheap → long
                # else: deadband, hold current position (no order)

                # Persist entry timestamp on a flat → non-flat transition.
                if leg.initial_position == 0 and target != 0:
                    self.new_trader_data[ts_key] = int(self.state.timestamp)
                elif entry_ts is not None and not time_stopped and target != 0:
                    # Carry the existing entry timestamp forward.
                    self.new_trader_data[ts_key] = int(entry_ts)

                passive = cfg.passive and not force_cross
                leg.go_to_target(int(target),
                                 passive=passive, improve=cfg.improve)
                out[leg.name] = leg.orders
        return out


# ─────────────────────────────────────────────────────────────────────
#  AGENT C — LEAD-LAG (scaffold; disabled by default)
# ─────────────────────────────────────────────────────────────────────
class LeadLagTrader(TraderBase):
    """Predict next-tick return on ``follower`` from last-tick return on
    ``leader``. Off by default — see plan.md §1C."""

    def __init__(self, name, state, new_trader_data, last_traderData, cfg: LeadLagConfig):
        super().__init__(name, state, new_trader_data, last_traderData, cfg)
        self.cfg: LeadLagConfig = cfg
        self._prev_leader_key = f"{cfg.short_code}lp"

    def get_orders(self) -> Dict[str, List[Order]]:
        if not self.cfg.enabled:
            return {self.name: self.orders}
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}
        if self.cfg.leader not in self.state.order_depths:
            return {self.name: self.orders}

        # Pull leader's mid from order book
        leader_depth = self.state.order_depths[self.cfg.leader]
        if not leader_depth.buy_orders or not leader_depth.sell_orders:
            return {self.name: self.orders}
        l_bid = max(leader_depth.buy_orders)
        l_ask = min(leader_depth.sell_orders)
        leader_mid = (l_bid + l_ask) / 2

        prev_leader = float(self.last_traderData.get(self._prev_leader_key, leader_mid))
        self.new_trader_data[self._prev_leader_key] = leader_mid

        if prev_leader <= 0:
            return {self.name: self.orders}
        leader_ret = (leader_mid - prev_leader) / prev_leader
        predicted_ret = self.cfg.beta * leader_ret

        if predicted_ret > self.cfg.gate:
            self.go_to_target(+self.cfg.target_size)
        elif predicted_ret < -self.cfg.gate:
            self.go_to_target(-self.cfg.target_size)
        else:
            self.go_to_target(0)
        return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────
#  COORDINATOR
# ─────────────────────────────────────────────────────────────────────
def _safe_load_trader_data(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


class Trader:
    """Round 5 coordinator: dispatches each agent's orders into the result
    dict. Agents are isolated by symbol (no overlapping legs)."""

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_trader_data: dict = {}
        last_traderData = _safe_load_trader_data(getattr(state, "traderData", "") or "")

        # ── 10-group basket arb ─────────────────────────────────────────
        # Every Round-5 group is constant-sum cointegrated within a day.
        # We trade Σ leg_mids back to its rolling mean: when the sum is
        # high we short every leg, when the sum is low we long every leg.
        # All α=1, so the trade is fully market-neutral within the group
        # and the sign is set by the residual z-score directly.
        for group_name, basket_cfg in GROUP_BASKETS.items():
            try:
                basket = BasketTrader(basket_cfg, state,
                                      new_trader_data, last_traderData)
                for s, orders in basket.get_orders().items():
                    if orders:
                        result.setdefault(s, []).extend(orders)
            except Exception as exc:
                print(f"[Basket-{group_name}] {exc!r}")

        # ── Lead-Lag (off unless explicitly enabled) ────────────────────
        for sym, cfg in LEADLAG_CONFIGS.items():
            if not cfg.enabled or sym not in state.order_depths:
                continue
            try:
                trader = LeadLagTrader(sym, state, new_trader_data, last_traderData, cfg)
                for s, orders in trader.get_orders().items():
                    if orders:
                        result.setdefault(s, []).extend(orders)
            except Exception as exc:
                print(f"[LeadLag] {sym}: {exc!r}")

        try:
            final_trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        except Exception:
            final_trader_data = ""
        return result, 0, final_trader_data
