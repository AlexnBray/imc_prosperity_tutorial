"""
ASH-only rolling-mean reversion market maker.

Concept:
- Fair value = rolling mean of mid price.
- Residual = mid - fair value.
- Mean reversion signal = negative residual z-score.
- Around fair: passive market making.
- At large deviations: stronger quote tilt + optional taker toward reversion.
"""

import json
import math
from typing import Dict, List, Tuple

import numpy as np
from datamodel import OrderDepth, Order, Symbol, TradingState


ASH = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80


class Trader:
    # Rolling fair value
    MEAN_WIN = 80
    STD_WIN = 80

    # Avellaneda-Stoikov style execution (simplified)
    GAMMA = 0.06
    KAPPA = 0.20
    MIN_SPREAD = 4.0
    INV_ASYM = 0.30

    # Alpha (mean-reversion quote tilt)
    ALPHA_TO_TICKS = 0.45
    Z_CLIP = 4.0

    # Optional taker behavior on large deviations
    Z_TAKE = 2.2
    TAKE_QTY = 4

    # Vol estimate for spread scaling
    VAR_WIN = 20
    MIN_VAR_OBS = 8
    
    # Safety controls
    MAX_PASSIVE_QTY = 12
    DRIFT_WIN = 12
    DRIFT_TAKE_GUARD = 0.7  # block MR taking if short-horizon trend fights it

    @staticmethod
    def _safe_mid(sells: list[tuple[int, int]], buys: list[tuple[int, int]]) -> float:
        if len(sells) >= 2 and len(buys) >= 2:
            return (sells[1][0] + buys[1][0]) / 2.0
        return (sells[0][0] + buys[0][0]) / 2.0

    def _trade_ash(self, order_depth: OrderDepth, position: int, data: dict) -> list[Order]:
        orders: list[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())  # (px, neg qty)
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)  # (px, pos qty)
        if not sell_orders or not buy_orders:
            return orders

        best_ask, best_ask_vol = sell_orders[0]
        best_bid, best_bid_vol = buy_orders[0]
        mid = self._safe_mid(sell_orders, buy_orders)

        # Rolling fair value
        mids: List[float] = data.get("mids", [])
        mids.append(mid)
        mids = mids[-max(self.MEAN_WIN, self.STD_WIN, self.VAR_WIN + 1) :]
        data["mids"] = mids

        mean_slice = mids[-self.MEAN_WIN :] if len(mids) >= 2 else mids
        fair = float(np.mean(mean_slice))

        resid = mid - fair
        std_slice = mids[-self.STD_WIN :]
        resid_std = float(np.std(std_slice)) if len(std_slice) >= 6 else 1.0
        resid_std = max(resid_std, 1e-6)
        z = max(-self.Z_CLIP, min(self.Z_CLIP, resid / resid_std))

        # Simple rolling variance of price changes
        if len(mids) >= self.MIN_VAR_OBS + 1:
            sigma2 = max(float(np.var(np.diff(mids[-(self.VAR_WIN + 1) :]))), 1e-6)
        else:
            sigma2 = 1.0

        # Alpha is opposite of z (mean reversion)
        alpha = -z
        alpha_shift = self.ALPHA_TO_TICKS * alpha

        # Short-horizon drift guard: don't mean-revert aggressively into strong trend.
        if len(mids) >= self.DRIFT_WIN + 1:
            drift = float(np.mean(np.diff(mids[-(self.DRIFT_WIN + 1) :])))
        else:
            drift = 0.0

        # A-S reservation and spread
        q = position
        reservation = fair - q * self.GAMMA * sigma2 + alpha_shift
        spread = self.GAMMA * sigma2 + (2.0 / self.GAMMA) * math.log(1.0 + self.GAMMA / self.KAPPA)
        spread = max(spread, self.MIN_SPREAD)

        qn = q / max(POSITION_LIMIT, 1)
        half_bid = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, qn))
        half_ask = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, -qn))

        as_bid = math.floor(reservation - half_bid)
        as_ask = math.ceil(reservation + half_ask)

        buy_cap = POSITION_LIMIT - position
        sell_cap = -POSITION_LIMIT - position

        # Optional taker on strong deviations
        allow_take_buy = drift <= self.DRIFT_TAKE_GUARD
        allow_take_sell = drift >= -self.DRIFT_TAKE_GUARD

        if z <= -self.Z_TAKE and buy_cap > 0 and allow_take_buy:
            take = min(-best_ask_vol, buy_cap, self.TAKE_QTY)
            if take > 0:
                orders.append(Order(ASH, best_ask, int(take)))
                buy_cap -= take
        elif z >= self.Z_TAKE and sell_cap < 0 and allow_take_sell:
            take = min(best_bid_vol, -sell_cap, self.TAKE_QTY)
            if take > 0:
                orders.append(Order(ASH, best_bid, -int(take)))
                sell_cap += take

        # Passive quotes near top of book
        mm_bid = min(as_bid, best_bid + 1)
        mm_ask = max(as_ask, best_ask - 1)

        # Size-cap passive quoting to avoid overtrading and inventory whipsaw.
        pass_buy = min(buy_cap, self.MAX_PASSIVE_QTY)
        pass_sell = min(-sell_cap, self.MAX_PASSIVE_QTY)
        if pass_buy > 0:
            orders.append(Order(ASH, int(mm_bid), int(pass_buy)))
        if pass_sell > 0:
            orders.append(Order(ASH, int(mm_ask), -int(pass_sell)))

        data["last"] = {
            "mid": round(mid, 3),
            "fair": round(fair, 3),
            "z": round(z, 3),
            "drift": round(drift, 4),
            "sig2": round(sigma2, 6),
            "spr": round(spread, 3),
            "res": round(reservation, 3),
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
