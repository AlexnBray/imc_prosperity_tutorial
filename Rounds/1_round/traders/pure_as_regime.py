"""
Pure Avellaneda-Stoikov Market Maker with Regime Detection.

ASH_COATED_OSMIUM — vol-spike regime, symmetric A-S market making
INTARIAN_PEPPER_ROOT — directional regime (EMA crossover), trend-biased A-S MM

ASH edge:   detect vol spikes within 1 tick, widen/tighten before others
PEPPER edge: detect trend reversals via MA crossover, passively build position
             via biased quotes (no expensive aggressive flipping)
"""
import math
import json
import numpy as np
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol

ASH = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"
POSITION_LIMIT = 80
PEPPER_LIMIT = 80
T_MAX = 999_900


class Logger:
    def __init__(self):
        self._logs: List[dict] = []

    def log(self, entry: dict) -> None:
        self._logs.append(entry)

    def flush(self, state: TradingState) -> str:
        out = json.dumps({
            "timestamp": state.timestamp,
            "entries": self._logs,
        }, default=str)
        self._logs = []
        return out


logger = Logger()


class Trader:
    # ── Variance estimation ───────────────────────────────────────────────
    VAR_WINDOW = 20               # rolling window for σ² (short enough to be
                                  # responsive, long enough to be stable)
    MIN_VAR_OBS = 5               # wait this many ticks before trusting σ²

    # ── Regime detection (spike-based) ────────────────────────────────────
    BASELINE_ALPHA = 0.025        # slow EWMA of dp² — the "normal" benchmark
    SPIKE_WINDOW = 3              # ticks of raw dp² to keep
    SPIKE_THRESH = 4.0            # worst/baseline to start reacting
    SPIKE_CEIL = 10.0             # worst/baseline for full high-vol

    # ── A-S quoting ──────────────────────────────────────────────────────
    GAMMA_QUIET = 0.05            # risk aversion when calm → tight quotes
    GAMMA_STORM = 0.20            # risk aversion in high-vol → wide quotes
    K_ARRIVAL = 0.40              # order arrival intensity κ
    TAU_MIN = 0.005
    MIN_SPREAD = 2
    POS_LIMIT_STORM = 35          # hard cap on inventory when volatile
    INV_ASYM = 0.30               # per-unit asymmetric widening

    # ── Tactical taking ──────────────────────────────────────────────────
    TAKE_EDGE = 2.0               # how far inside theoretical FV to take
    TAKE_MAX_POS_FRAC = 0.75      # don't take if already > 75% of limit
    TAKE_REGIME_GATE = 0.3        # disable taking when p_hv exceeds this

    # ── End-of-day flatten (ASH) ─────────────────────────────────────────
    EOD_TAU = 0.10
    EOD_INV_PEN = 4.0             # reservation shift at close (ticks)

    # ══════════════════════════════════════════════════════════════════════
    # PEPPER: Pure MM with permanent long bias (price trends up)
    # ══════════════════════════════════════════════════════════════════════
    PP_GAMMA = 0.03               # low γ: comfortable being long
    PP_K_ARRIVAL = 0.40
    PP_MIN_SPREAD = 2
    PP_VAR_WINDOW = 20
    PP_LONG_TARGET = 0.75         # target inventory as fraction of limit
                                  # 0.75 × 80 = 60 → always lean long
    PP_BID_TIGHTEN = 1.5          # tighten bid by this many ticks (eager to buy)
    PP_ASK_WIDEN = 1.5            # widen ask by this many ticks (reluctant to sell)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @staticmethod
    def _safe_mid(sells, buys):
        if len(sells) >= 2 and len(buys) >= 2:
            return (sells[1][0] + buys[1][0]) / 2.0
        return (sells[0][0] + buys[0][0]) / 2.0

    @staticmethod
    def _ewma(prev, val, alpha):
        return val if prev is None else alpha * val + (1.0 - alpha) * prev

    @staticmethod
    def _spike_regime(worst_var, baseline_var, thresh, ceil):
        if baseline_var < 1e-15:
            return 0.0
        ratio = worst_var / baseline_var
        if ratio <= thresh:
            return 0.0
        if ratio >= ceil:
            return 1.0
        return (ratio - thresh) / (ceil - thresh)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Main trading logic
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def trade_osmium(self, order_depth: OrderDepth, position: int,
                     data: dict, timestamp: int) -> List[Order]:
        orders: List[Order] = []

        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        if not sell_orders or not buy_orders:
            return orders

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]
        mid = self._safe_mid(sell_orders, buy_orders)

        # ── Track mid prices (for rolling variance) ───────────────────────
        mids: List[float] = data.get("m", [])
        mids.append(mid)
        mids = mids[-(self.VAR_WINDOW + 1):]
        data["m"] = mids

        # Rolling variance of price changes
        if len(mids) >= self.MIN_VAR_OBS + 1:
            dp = np.diff(mids)
            rolling_var = float(np.var(dp))
        else:
            rolling_var = 1.0
        rolling_var = max(rolling_var, 1e-6)

        # ── Regime detection ──────────────────────────────────────────────
        recent_dp2: List[float] = data.get("rdp", [])
        last_mid = data.get("lm")
        if last_mid is not None:
            dp2 = (mid - last_mid) ** 2
            baseline_var = self._ewma(data.get("bv"), dp2, self.BASELINE_ALPHA)
            recent_dp2.append(dp2)
            recent_dp2 = recent_dp2[-self.SPIKE_WINDOW:]
        else:
            baseline_var = data.get("bv", 1.0)
        data["bv"] = baseline_var
        data["rdp"] = recent_dp2
        data["lm"] = mid

        worst_var = max(recent_dp2) if recent_dp2 else baseline_var
        p_hv = self._spike_regime(
            worst_var, baseline_var, self.SPIKE_THRESH, self.SPIKE_CEIL,
        )

        # ── Regime-blended parameters ─────────────────────────────────────
        gamma = self.GAMMA_QUIET + (self.GAMMA_STORM - self.GAMMA_QUIET) * p_hv
        pos_limit = POSITION_LIMIT - (POSITION_LIMIT - self.POS_LIMIT_STORM) * p_hv

        # σ² for A-S: in quiet use rolling variance, in storm use the spike
        # magnitude so the spread reacts to the actual move size
        sigma2 = max((1.0 - p_hv) * rolling_var + p_hv * worst_var, 1e-6)

        tau = max(1.0 - timestamp / T_MAX, self.TAU_MIN)

        # ── A-S reservation price ─────────────────────────────────────────
        # r = s − q·γ·σ²·τ   (inventory penalty pushes toward flat)
        q = position
        reservation = mid - q * gamma * sigma2 * tau

        # End-of-day: add τ-independent inventory unwind pressure
        if tau < self.EOD_TAU:
            urgency = (self.EOD_TAU - tau) / self.EOD_TAU
            reservation -= self.EOD_INV_PEN * urgency * q / POSITION_LIMIT

        # ── A-S optimal spread ────────────────────────────────────────────
        # δ = γσ²τ + (2/γ)·ln(1 + γ/κ)
        spread = (gamma * sigma2 * tau
                  + (2.0 / gamma) * math.log(1.0 + gamma / self.K_ARRIVAL))
        spread = max(spread, float(self.MIN_SPREAD))

        # Asymmetric half-spreads
        qn = q / max(pos_limit, 1)
        half_bid = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, qn))
        half_ask = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, -qn))

        as_bid = math.floor(reservation - half_bid)
        as_ask = math.ceil(reservation + half_ask)

        buy_cap = int(pos_limit) - position
        sell_cap = -int(pos_limit) - position

        # ── Tactical taking ───────────────────────────────────────────────
        # Only in quiet regime and when not too loaded
        can_take = (p_hv < self.TAKE_REGIME_GATE
                    and abs(position) < self.TAKE_MAX_POS_FRAC * pos_limit)

        if can_take:
            for ask_px, ask_vol in sell_orders:
                vol = -ask_vol
                if ask_px < mid - self.TAKE_EDGE and buy_cap > 0:
                    take = min(vol, buy_cap)
                    if take > 0:
                        orders.append(Order(ASH, ask_px, take))
                        buy_cap -= take

            for bid_px, bid_vol in buy_orders:
                if bid_px > mid + self.TAKE_EDGE and sell_cap < 0:
                    take = min(bid_vol, -sell_cap)
                    if take > 0:
                        orders.append(Order(ASH, bid_px, -take))
                        sell_cap += take

        # ── Passive MM quotes ─────────────────────────────────────────────
        mm_bid = min(as_bid, best_bid + 1)
        mm_ask = max(as_ask, best_ask - 1)

        if buy_cap > 0:
            orders.append(Order(ASH, int(mm_bid), buy_cap))
        if sell_cap < 0:
            orders.append(Order(ASH, int(mm_ask), sell_cap))

        # ── Logging ───────────────────────────────────────────────────────
        logger.log({
            "p": ASH, "ts": timestamp, "pos": position,
            "mid": round(mid, 2),
            "p_hv": round(p_hv, 4),
            "worst": round(worst_var, 4), "base": round(baseline_var, 6),
            "rvar": round(rolling_var, 6), "sig2": round(sigma2, 6),
            "gamma": round(gamma, 6),
            "tau": round(tau, 4),
            "res": round(reservation, 4), "sprd": round(spread, 4),
            "bid": as_bid, "ask": as_ask,
            "plim": round(pos_limit, 1),
            "take": can_take,
            "n": len(orders),
        })

        return orders

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PEPPER: Pure MM with permanent long bias
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def trade_peppers(self, order_depth: OrderDepth, position: int,
                      data: dict, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        pp = data.setdefault("pp", {})

        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        if not sell_orders or not buy_orders:
            return orders

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]
        mid = self._safe_mid(sell_orders, buy_orders)

        # ── Rolling variance ──────────────────────────────────────────────
        mids: List[float] = pp.get("m", [])
        mids.append(mid)
        mids = mids[-(self.PP_VAR_WINDOW + 1):]
        pp["m"] = mids

        if len(mids) >= 6:
            sigma2 = max(float(np.var(np.diff(mids))), 1e-6)
        else:
            sigma2 = 1.0

        tau = max(1.0 - timestamp / T_MAX, self.TAU_MIN)

        # ── A-S with long bias ────────────────────────────────────────────
        # Target inventory is permanently positive (ride the uptrend)
        target = self.PP_LONG_TARGET * PEPPER_LIMIT
        inv_gap = position - target

        # Reservation: penalise deviation from target, not from zero
        # When position < target: reservation > mid → eager bid → buy more
        # When position > target: reservation < mid → flatten a bit
        reservation = mid - inv_gap * self.PP_GAMMA * sigma2 * tau

        # τ-independent nudge so bias works even late in the day
        nudge = 0.02 * inv_gap
        nudge = max(-4.0, min(4.0, nudge))
        reservation -= nudge

        # ── A-S spread (symmetric base) ───────────────────────────────────
        spread = (self.PP_GAMMA * sigma2 * tau
                  + (2.0 / self.PP_GAMMA) * math.log(1.0 + self.PP_GAMMA / self.PP_K_ARRIVAL))
        spread = max(spread, float(self.PP_MIN_SPREAD))

        # Permanent asymmetry: tighter bid (want buys), wider ask (reluctant sells)
        half_bid = max(spread / 2.0 - self.PP_BID_TIGHTEN, 0.5)
        half_ask = spread / 2.0 + self.PP_ASK_WIDEN

        pp_bid = math.floor(reservation - half_bid)
        pp_ask = math.ceil(reservation + half_ask)

        buy_cap = PEPPER_LIMIT - position
        sell_cap = -PEPPER_LIMIT - position

        # Passive quotes only — earn spread while accumulating longs
        mm_bid = min(pp_bid, best_bid + 1)
        mm_ask = max(pp_ask, best_ask - 1)

        if buy_cap > 0:
            orders.append(Order(PEPPER, int(mm_bid), buy_cap))
        if sell_cap < 0:
            orders.append(Order(PEPPER, int(mm_ask), sell_cap))

        logger.log({
            "p": PEPPER, "ts": timestamp, "pos": position,
            "mid": round(mid, 2), "tgt": round(target, 1),
            "sig2": round(sigma2, 6), "tau": round(tau, 4),
            "res": round(reservation, 4), "sprd": round(spread, 4),
            "bid": pp_bid, "ask": pp_ask,
            "n": len(orders),
        })

        return orders

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Entry point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        for product in state.order_depths:
            position = state.position.get(product, 0)
            if product == ASH:
                result[product] = self.trade_osmium(
                    state.order_depths[product], position, data, state.timestamp,
                )
            elif product == PEPPER:
                result[product] = self.trade_peppers(
                    state.order_depths[product], position, data, state.timestamp,
                )
            else:
                result[product] = []

        trader_data = json.dumps(data)
        log_output = logger.flush(state)
        print(log_output)

        return result, conversions, trader_data
