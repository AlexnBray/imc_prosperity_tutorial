"""
Basic AS market maker for ASH_COATED_OSMIUM using a rolling-window fair value.

Fair value source is configurable over recent ticks:
- "mid": rolling mean of mid price
- "micro": rolling mean of microprice
- "median": rolling median of microprice
- "vwap": rolling VWAP from top-of-book sizes
"""

import json
import math
from typing import Dict, List, Tuple

import numpy as np
from datamodel import OrderDepth, Order, Symbol, TradingState


ASH = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80


class Trader:
    # Rolling fair value config
    PRICE_MODE = "vwap"           # one of: "mid", "micro", "median", "vwap"
    FAIR_WINDOW = 21              # rolling tick window for fair value

    # A-S parameters
    GAMMA = 0.07
    KAPPA = 0.18
    MIN_SPREAD = 4.0

    # Volatility estimate (for spread and reservation scaling)
    VAR_WINDOW = 12
    MIN_VAR_OBS = 8

    # Mild asymmetric widening by inventory side
    INV_ASYM = 0.20

    @staticmethod
    def _safe_mid(sells: list[tuple[int, int]], buys: list[tuple[int, int]]) -> float:
        if len(sells) >= 2 and len(buys) >= 2:
            return (sells[1][0] + buys[1][0]) / 2.0
        return (sells[0][0] + buys[0][0]) / 2.0

    @staticmethod
    def _microprice(best_bid: int, best_bid_vol: int, best_ask: int, best_ask_vol: int, fallback: float) -> float:
        bid_q = max(0, int(best_bid_vol))
        ask_q = max(0, int(-best_ask_vol))
        denom = bid_q + ask_q
        if denom <= 0:
            return fallback
        # Opposite-side weighting
        return (best_ask * bid_q + best_bid * ask_q) / float(denom)

    @staticmethod
    def _top_vwap(best_bid: int, best_bid_vol: int, best_ask: int, best_ask_vol: int, fallback: float) -> float:
        bid_q = max(0, int(best_bid_vol))
        ask_q = max(0, int(-best_ask_vol))
        denom = bid_q + ask_q
        if denom <= 0:
            return fallback
        return (best_bid * bid_q + best_ask * ask_q) / float(denom)

    def _trade_ash(self, order_depth: OrderDepth, position: int, data: dict) -> list[Order]:
        orders: list[Order] = []

        sell_orders = sorted(order_depth.sell_orders.items())  # (px, neg qty)
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)  # (px, pos qty)
        if not sell_orders or not buy_orders:
            return orders

        best_ask, best_ask_vol = sell_orders[0]
        best_bid, best_bid_vol = buy_orders[0]
        mid = self._safe_mid(sell_orders, buy_orders)
        micro = self._microprice(best_bid, best_bid_vol, best_ask, best_ask_vol, mid)
        vwap_tick = self._top_vwap(best_bid, best_bid_vol, best_ask, best_ask_vol, mid)

        # Rolling tick histories
        mid_hist: List[float] = data.get("mid_hist", [])
        micro_hist: List[float] = data.get("micro_hist", [])
        vwap_hist: List[float] = data.get("vwap_hist", [])
        mid_hist.append(float(mid))
        micro_hist.append(float(micro))
        vwap_hist.append(float(vwap_tick))

        keep = max(self.FAIR_WINDOW, self.VAR_WINDOW + 1)
        mid_hist = mid_hist[-keep:]
        micro_hist = micro_hist[-keep:]
        vwap_hist = vwap_hist[-keep:]
        data["mid_hist"] = mid_hist
        data["micro_hist"] = micro_hist
        data["vwap_hist"] = vwap_hist

        # Fair value from selected rolling window
        if self.PRICE_MODE == "mid":
            fair = float(np.mean(mid_hist[-self.FAIR_WINDOW:]))
            filt_series = mid_hist
        elif self.PRICE_MODE == "micro":
            fair = float(np.mean(micro_hist[-self.FAIR_WINDOW:]))
            filt_series = micro_hist
        elif self.PRICE_MODE == "median":
            fair = float(np.median(micro_hist[-self.FAIR_WINDOW:]))
            filt_series = micro_hist
        else:  # "vwap" default
            fair = float(np.mean(vwap_hist[-self.FAIR_WINDOW:]))
            filt_series = vwap_hist

        # keep old key for compatibility with existing states
        px_hist: List[float] = data.get("px_hist", [])
        px_hist.append(float(fair))
        px_hist = px_hist[-keep:]
        data["px_hist"] = px_hist

        # Rolling variance on selected fair-source increments
        if len(filt_series) >= self.MIN_VAR_OBS + 1:
            series = np.asarray(filt_series[-(self.VAR_WINDOW + 1):], dtype=float)
            sigma2 = max(float(np.var(np.diff(series))), 1e-6)
        else:
            sigma2 = 1.0

        # A-S reservation and spread
        q = position
        reservation = fair - q * self.GAMMA * sigma2
        spread = self.GAMMA * sigma2 + (2.0 / self.GAMMA) * math.log(1.0 + self.GAMMA / self.KAPPA)
        spread = max(spread, self.MIN_SPREAD)

        # Mild inventory asymmetry around symmetric base spread
        qn = q / max(POSITION_LIMIT, 1)
        half_bid = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, qn))
        half_ask = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, -qn))

        as_bid = math.floor(reservation - half_bid)
        as_ask = math.ceil(reservation + half_ask)

        # Keep quotes marketable but not too far off-book
        mm_bid = min(as_bid, best_bid + 1)
        mm_ask = max(as_ask, best_ask - 1)

        buy_cap = POSITION_LIMIT - position
        sell_cap = -POSITION_LIMIT - position

        if buy_cap > 0:
            orders.append(Order(ASH, int(mm_bid), int(buy_cap)))
        if sell_cap < 0:
            orders.append(Order(ASH, int(mm_ask), int(sell_cap)))

        data["diag"] = {
            "mode": self.PRICE_MODE,
            "mid": round(mid, 3),
            "micro": round(micro, 3),
            "vwap": round(vwap_tick, 3),
            "fair": round(fair, 3),
            "sig2": round(sigma2, 6),
            "res": round(reservation, 3),
            "spr": round(spread, 3),
        }
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        ash_data = data.setdefault("ash", {})

        for product, order_depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == ASH:
                result[product] = self._trade_ash(order_depth, pos, ash_data)
            else:
                result[product] = []

        trader_data = json.dumps(data, separators=(",", ":"))
        return result, conversions, trader_data
