import json
import math
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
        signals: Dict[str, Any],
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                    signals,
                ]
            )
        )
        max_item_length = max(0, (self.max_log_length - base_length) // 3)
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                    signals,
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        out = []
        for arr in trades.values():
            for trade in arr:
                out.append(
                    [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                )
        return out

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, obs in observations.conversionObservations.items():
            conversion_observations[product] = [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sunlight,
                obs.humidity,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        out: List[List[Any]] = []
        for arr in orders.values():
            for order in arr:
                out.append([order.symbol, order.price, order.quantity])
        return out

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        if len(json.dumps(value)) <= max_length:
            return value
        return value[: max(0, max_length - 3)] + "..."


logger = Logger()


class Trader:
    OSMIUM_PRODUCTS = ("OSMIUM", "ASH_COATED_OSMIUM")
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM_FV_ANCHOR = 10000.0

    OSMIUM_LIMIT = 80
    PEPPER_LIMIT = 80

    HUBER_CLIP_MULT = 1.5
    LAYERS: Tuple[Tuple[float, float], ...] = ((0.40, 1.2), (0.40, 3.5), (0.20, 8.5))
    INNER_MIN_DIST = 2
    LEAN_COEF = 0.12
    LEAN_CAP = 8.0
    PENNY_QTY_PCT = 0.18

    AGGRESSIVE_BUILD_TICKS = 5000
    TAKE_CHUNK = 80

    # OSMIUM Kalman + AS (ported from high-PnL reference), plus layered quoting.
    KF_R_OBS = 80.0
    KF_Q_LEVEL = 0.5
    KF_Q_DRIFT = 0.02
    KF_DRIFT_EPS = 0.04
    OSMIUM_GAMMA = 0.001
    OSMIUM_K = 0.07
    OSMIUM_T = 1.0
    OSMIUM_TAKE_EDGE = 0.25
    OSMIUM_LAYER_TICKS: Tuple[int, ...] = (0, 2, 5)
    OSMIUM_LAYER_WEIGHTS: Tuple[float, ...] = (0.55, 0.30, 0.15)

    # ---- Taker strategy params (evidence-backed) ----
    HAMPEL_WINDOW = 31
    MR_EXTREME_THRESH = 5.0
    MR_EXTREME_QTY = 10
    # Maker bias: shift reservation price toward expected reversion
    MR_RESERVATION_LEAN = 0.4
    OBI_RESERVATION_LEAN = 0.8
    LVL_IMB_RESERVATION_LEAN = 0.6

    # Conservative PEPPER regime detector; default remains long-hold.
    PEPPER_EWMA_FAST_ALPHA = 0.28
    PEPPER_EWMA_SLOW_ALPHA = 0.06
    PEPPER_EWMA_WARMUP = 30
    PEPPER_DOWN_DIFF_THRESH = -0.9
    PEPPER_UP_DIFF_THRESH = 0.8
    PEPPER_DOWN_CONFIRM = 6
    PEPPER_UP_CONFIRM = 8

    def _update_pepper_regime(
        self,
        mid: float,
        state_data: Dict[str, Any],
    ) -> Tuple[str, float, Dict[str, Any]]:
        fast = float(state_data.get("ewma_fast", mid))
        slow = float(state_data.get("ewma_slow", mid))
        ticks = int(state_data.get("ewma_ticks", 0)) + 1
        down_streak = int(state_data.get("down_streak", 0))
        up_streak = int(state_data.get("up_streak", 0))
        regime = str(state_data.get("regime", "long_hold"))

        fast = self.PEPPER_EWMA_FAST_ALPHA * mid + (1.0 - self.PEPPER_EWMA_FAST_ALPHA) * fast
        slow = self.PEPPER_EWMA_SLOW_ALPHA * mid + (1.0 - self.PEPPER_EWMA_SLOW_ALPHA) * slow
        ewma_diff = fast - slow

        if ticks >= self.PEPPER_EWMA_WARMUP:
            if ewma_diff <= self.PEPPER_DOWN_DIFF_THRESH:
                down_streak += 1
                up_streak = 0
                if down_streak >= self.PEPPER_DOWN_CONFIRM:
                    regime = "short"
            elif ewma_diff >= self.PEPPER_UP_DIFF_THRESH:
                up_streak += 1
                down_streak = 0
                if up_streak >= self.PEPPER_UP_CONFIRM:
                    regime = "long_hold"
            else:
                down_streak = 0
                up_streak = 0

        state_data["ewma_fast"] = float(fast)
        state_data["ewma_slow"] = float(slow)
        state_data["ewma_ticks"] = int(ticks)
        state_data["down_streak"] = int(down_streak)
        state_data["up_streak"] = int(up_streak)
        state_data["regime"] = regime
        return regime, ewma_diff, state_data

    @staticmethod
    def _kalman_local_linear_tick(
        z: float,
        mu: float,
        beta: float,
        p00: float,
        p01: float,
        p10: float,
        p11: float,
        r_obs: float,
        q_level: float,
        q_drift: float,
    ) -> Tuple[float, float, float, float, float, float]:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.array([[q_level, 0.0], [0.0, q_drift]])
        H = np.array([[1.0, 0.0]])
        x = np.array([mu, beta])
        P = np.array([[p00, p01], [p10, p11]])

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        y = z - float(np.squeeze(H @ x_pred))
        s_cov = float(np.squeeze(H @ P_pred @ H.T)) + r_obs
        if s_cov <= 1e-12:
            s_cov = 1e-12
        k = (P_pred @ H.T).flatten() / s_cov

        x_new = x_pred + k * y
        P_new = (np.eye(2) - np.outer(k, H)) @ P_pred

        return (
            float(x_new[0]),
            float(x_new[1]),
            float(P_new[0, 0]),
            float(P_new[0, 1]),
            float(P_new[1, 0]),
            float(P_new[1, 1]),
        )

    @staticmethod
    def _sorted_books(order_depth: OrderDepth) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        return sell_orders, buy_orders

    @staticmethod
    def _micro_price(buy_orders: List[Tuple[int, int]], sell_orders: List[Tuple[int, int]], fallback: float) -> float:
        if not buy_orders or not sell_orders:
            return fallback
        bp, bv = buy_orders[0]
        sp, sv = sell_orders[0]
        bv = abs(bv)
        sv = abs(sv)
        denom = bv + sv
        if denom <= 0:
            return fallback
        return (bp * sv + sp * bv) / denom

    @staticmethod
    def _mad(values: List[float]) -> Tuple[float, float]:
        arr = np.array(values, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return med, mad

    @staticmethod
    def _causal_median(values: List[float], window: int) -> float:
        if len(values) < 5:
            return values[-1] if values else 0.0
        tail = values[-min(window, len(values)):]
        return float(np.median(tail))

    @staticmethod
    def _count_book_levels(order_depth: OrderDepth) -> Tuple[int, int]:
        n_bids = len(order_depth.buy_orders)
        n_asks = len(order_depth.sell_orders)
        return n_bids, n_asks

    @staticmethod
    def _l1_obi(buy_orders: List[Tuple[int, int]], sell_orders: List[Tuple[int, int]]) -> float:
        if not buy_orders or not sell_orders:
            return 0.0
        bv = abs(buy_orders[0][1])
        sv = abs(sell_orders[0][1])
        denom = bv + sv
        if denom <= 0:
            return 0.0
        return (bv - sv) / denom

    @staticmethod
    def _obi_bin_from_l1(obi_l1: float) -> str:
        # Mirrors the bins used in the CSV analysis:
        # [-1, -0.4] -> '--', (-0.4, -0.15] -> '-', (-0.15, 0.15] -> '0', (0.15, 0.4] -> '+', (0.4, 1] -> '++'
        if obi_l1 <= -0.4:
            return "--"
        if obi_l1 <= -0.15:
            return "-"
        if obi_l1 <= 0.15:
            return "0"
        if obi_l1 <= 0.4:
            return "+"
        return "++"

    def _trade_pepper(
        self,
        order_depth: OrderDepth,
        position: int,
        timestamp: int,
        state_data: Dict[str, Any],
    ) -> Tuple[List[Order], Dict[str, Any], Dict[str, Any]]:
        orders: List[Order] = []
        sells, buys = self._sorted_books(order_depth)
        if not sells and not buys:
            return orders, state_data, {"active": False}

        if sells and buys:
            mid = 0.5 * (sells[0][0] + buys[0][0])
        elif sells:
            mid = float(sells[0][0])
        else:
            mid = float(buys[0][0])

        regime, ewma_diff, state_data = self._update_pepper_regime(mid, state_data)

        # Regime break: sweep bids across all levels to rotate inventory to full short.
        if regime == "short":
            sell_need = self.PEPPER_LIMIT + position
            sold = 0
            if sell_need > 0:
                for bid_price, bid_vol in buys:
                    if sell_need <= 0:
                        break
                    take = min(abs(bid_vol), sell_need)
                    if take > 0:
                        orders.append(Order(self.PEPPER, int(bid_price), -int(take)))
                        sell_need -= take
                        sold += take
            return orders, state_data, {
                "active": True,
                "regime": regime,
                "ewma_diff": round(ewma_diff, 4),
                "n_orders": len(orders),
                "sell_swept": sold,
                "ts": timestamp,
            }

        buy_cap = self.PEPPER_LIMIT - position

        # Aggressively take L1 ask only until full.
        if buy_cap > 0 and sells:
            best_ask, best_ask_vol = sells[0]
            take = min(abs(best_ask_vol), buy_cap)
            if take > 0:
                orders.append(Order(self.PEPPER, best_ask, take))
                buy_cap -= take

        # Passive fill for remaining capacity at top-of-book
        if buy_cap > 0 and buys:
            orders.append(Order(self.PEPPER, buys[0][0] + 1, buy_cap))

        return orders, state_data, {
            "active": True,
            "regime": regime,
            "ewma_diff": round(ewma_diff, 4),
            "buy_cap": buy_cap,
            "n_orders": len(orders),
            "ts": timestamp,
        }

    def _trade_osmium(
        self,
        order_depth: OrderDepth,
        position: int,
        state_data: Dict[str, Any],
    ) -> Tuple[List[Order], Dict[str, Any], Dict[str, Any]]:
        orders: List[Order] = []
        sells, buys = self._sorted_books(order_depth)
        if not sells or not buys:
            return orders, state_data, {"active": False}

        sym = state_data["symbol"]
        best_ask, best_ask_vol = sells[0]
        best_bid, best_bid_vol = buys[0]
        spread = float(best_ask - best_bid)
        micro = self._micro_price(buys, sells, 0.5 * (best_ask + best_bid))

        # ---- Kalman state ----
        kf_mu = float(state_data.get("kf_mu", 0.0))
        kf_beta = float(state_data.get("kf_beta", 0.0))
        kf_p = state_data.get("kf_p", [25.0, 0.0, 0.0, 4.0])
        if not isinstance(kf_p, list) or len(kf_p) != 4:
            kf_p = [25.0, 0.0, 0.0, 4.0]
        p00, p01, p10, p11 = [float(x) for x in kf_p]

        if kf_mu == 0.0:
            kf_mu = micro
            kf_beta = 0.0
        else:
            kf_mu, kf_beta, p00, p01, p10, p11 = self._kalman_local_linear_tick(
                micro, kf_mu, kf_beta, p00, p01, p10, p11,
                self.KF_R_OBS, self.KF_Q_LEVEL, self.KF_Q_DRIFT,
            )

        mids = deque(state_data.get("mid_prices", []), maxlen=101)
        mids.append(float(micro))
        if len(mids) > 2:
            returns = np.diff(np.array(mids, dtype=float))
            var = max(float(np.var(returns)), 1e-4)
        else:
            var = 0.01

        s = kf_mu
        reservation = s - (position * self.OSMIUM_GAMMA * var * self.OSMIUM_T)
        delta = (
            self.OSMIUM_GAMMA * var * self.OSMIUM_T
            + (2.0 / self.OSMIUM_GAMMA) * math.log(1.0 + (self.OSMIUM_GAMMA / self.OSMIUM_K))
        )

        buy_cap = self.OSMIUM_LIMIT - position
        sell_cap = -self.OSMIUM_LIMIT - position

        # ================================================================
        # SIGNAL COMPUTATION (no orders yet -- just measure the signals)
        # ================================================================
        mid_val = 0.5 * (best_ask + best_bid)
        hampel_fv = self._causal_median(list(mids), self.HAMPEL_WINDOW)
        deviation = mid_val - hampel_fv

        n_bids, n_asks = self._count_book_levels(order_depth)
        level_imbalance = n_bids - n_asks

        obi_l1 = self._l1_obi(buys, sells)
        obi_bin = self._obi_bin_from_l1(obi_l1)

        sell_signal_kf = kf_beta < -self.KF_DRIFT_EPS
        buy_signal_kf = kf_beta > self.KF_DRIFT_EPS

        taker_took = 0

        # ================================================================
        # TAKER 1: Original Kalman drift taker (PRESERVED from baseline)
        # ================================================================
        for ask_price, ask_vol in sells:
            if ask_price < s + self.OSMIUM_TAKE_EDGE and buy_signal_kf and buy_cap > 0:
                take_vol = min(abs(ask_vol), buy_cap)
                if take_vol > 0:
                    orders.append(Order(sym, int(ask_price), int(take_vol)))
                    buy_cap -= take_vol
        for bid_price, bid_vol in buys:
            if bid_price > s - self.OSMIUM_TAKE_EDGE and sell_signal_kf and sell_cap < 0:
                take_vol = min(abs(bid_vol), abs(sell_cap))
                if take_vol > 0:
                    orders.append(Order(sym, int(bid_price), -int(take_vol)))
                    sell_cap += take_vol

        # ================================================================
        # TAKER 2: Extreme MR fade (ONLY at 5+ tick deviation, 98.6% HR)
        # Small size (10 lots) to avoid starving the maker.
        # ================================================================
        if len(mids) >= 15 and abs(deviation) >= self.MR_EXTREME_THRESH:
            # Rule A (CSV best): deviation sign + L1 OBI confirmation
            # If mid is far BELOW fair (deviation negative), expect reversion UP:
            # buy only when OBI indicates bid pressure (obi_bin == '++').
            if deviation <= -self.MR_EXTREME_THRESH and buy_cap > 0:
                take_qty = min(self.MR_EXTREME_QTY, buy_cap, abs(best_ask_vol))
                if take_qty > 0:
                    orders.append(Order(sym, best_ask, int(take_qty)))
                    buy_cap -= take_qty
                    taker_took += take_qty
            # If mid is far ABOVE fair (deviation positive), expect reversion DOWN:
            # sell only when OBI indicates ask pressure (obi_bin == '--').
            elif deviation >= self.MR_EXTREME_THRESH and sell_cap < 0:
                take_qty = min(self.MR_EXTREME_QTY, abs(sell_cap), abs(best_bid_vol))
                if take_qty > 0:
                    orders.append(Order(sym, best_bid, -int(take_qty)))
                    sell_cap += take_qty
                    taker_took += take_qty

        # ================================================================
        # MAKER: Bias reservation price using MR + OBI + level imbalance
        # This is the key insight: use taker signals to IMPROVE maker quotes
        # rather than taking aggressively and starving the maker.
        # ================================================================
        reservation_bias = 0.0

        # MR bias: lean quotes toward expected reversion direction
        if len(mids) >= 10:
            reservation_bias -= deviation * self.MR_RESERVATION_LEAN

        # OBI bias: IC=0.62 for fwd1, lean toward OBI direction
        reservation_bias += obi_l1 * self.OBI_RESERVATION_LEAN

        # Level imbalance bias: more bid levels -> price drops -> lean ask tighter
        reservation_bias -= level_imbalance * self.LVL_IMB_RESERVATION_LEAN

        biased_reservation = reservation + reservation_bias

        # ================================================================
        # MAKER LAYER: Pennying + Layered quoting with biased reservation
        # ================================================================
        base_bid = max(int(math.floor(biased_reservation - delta / 2.0)), best_bid + 1)
        base_ask = min(int(math.ceil(biased_reservation + delta / 2.0)), best_ask - 1)
        if base_bid >= base_ask:
            base_bid = min(base_bid, best_ask - 1)
            base_ask = max(base_ask, best_bid + 1)

        for i, offset in enumerate(self.OSMIUM_LAYER_TICKS):
            w = self.OSMIUM_LAYER_WEIGHTS[i] if i < len(self.OSMIUM_LAYER_WEIGHTS) else 0.0
            if w <= 0:
                continue
            layer_qty = int(max(1, round(self.OSMIUM_LIMIT * w * 0.5)))

            bid_px = base_bid - offset
            ask_px = base_ask + offset
            if bid_px >= best_ask:
                bid_px = best_ask - 1
            if ask_px <= best_bid:
                ask_px = best_bid + 1
            if bid_px >= ask_px:
                continue

            if buy_cap > 0:
                bq = min(layer_qty, buy_cap)
                if bq > 0:
                    orders.append(Order(sym, int(bid_px), int(bq)))
                    buy_cap -= bq
            if sell_cap < 0:
                sq = min(layer_qty, -sell_cap)
                if sq > 0:
                    orders.append(Order(sym, int(ask_px), -int(sq)))
                    sell_cap += sq

            if buy_cap <= 0 and sell_cap >= 0:
                break

        state_data["mid_prices"] = list(mids)
        state_data["kf_mu"] = float(kf_mu)
        state_data["kf_beta"] = float(kf_beta)
        state_data["kf_p"] = [float(p00), float(p01), float(p10), float(p11)]
        signals = {
            "fv": round(s, 2),
            "r": round(biased_reservation, 2),
            "beta": round(kf_beta, 4),
            "var": round(var, 5),
            "delta": round(delta, 3),
            "spread": spread,
            "n_orders": len(orders),
            "dev": round(deviation, 2),
            "bias": round(reservation_bias, 2),
            "taker": taker_took,
            "obi_l1": round(obi_l1, 3),
            "obi_bin": obi_bin,
        }
        return orders, state_data, signals

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        data.setdefault("osmium", {})
        data.setdefault("pepper", {})
        result: Dict[Symbol, List[Order]] = {}
        signals: Dict[str, Any] = {}

        for osmium_symbol in self.OSMIUM_PRODUCTS:
            if osmium_symbol in state.order_depths:
                data["osmium"]["symbol"] = osmium_symbol
                pos = state.position.get(osmium_symbol, 0)
                orders, new_state, sig = self._trade_osmium(
                    state.order_depths[osmium_symbol], pos, data["osmium"]
                )
                result[osmium_symbol] = orders
                data["osmium"] = new_state
                signals[osmium_symbol] = sig
                break

        if self.PEPPER in state.order_depths:
            pos = state.position.get(self.PEPPER, 0)
            orders, new_state, sig = self._trade_pepper(
                state.order_depths[self.PEPPER], pos, state.timestamp, data["pepper"]
            )
            result[self.PEPPER] = orders
            data["pepper"] = new_state
            signals["PEPPER"] = sig

        conversions = 0
        trader_data = json.dumps(data)
        logger.flush(state, result, conversions, trader_data, signals)
        return result, conversions, trader_data