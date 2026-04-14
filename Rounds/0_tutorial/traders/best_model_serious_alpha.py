import math
import jsonpickle
import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol


class Trader:

    def log_data(self, state: TradingState, product: str, position: int, orders: List[Order], fv: float, effective_fv: float, signal: float):
        bid_map = {}
        ask_map = {}

        for o in orders:
            if o.quantity > 0:
                bid_map[o.price] = bid_map.get(o.price, 0) + o.quantity
            else:
                ask_map[o.price] = ask_map.get(o.price, 0) + o.quantity

        bids_str = ";".join([f"{p}:{q}" for p, q in sorted(bid_map.items(), reverse=True)])
        asks_str = ";".join([f"{p}:{q}" for p, q in sorted(ask_map.items())])

        print(f"[ALGO],{state.timestamp},{product},{position},{fv:.2f},{effective_fv:.2f},{signal:.2f},[{bids_str}],[{asks_str}]")

    def trade_emeralds(self, order_depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = 80
        fv = 10000.0

        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        if not sell_orders or not buy_orders:
            return orders

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]

        initial_pos = position
        buy_capacity = limit - position
        sell_capacity = limit + position

        # 1. Tactical Taking
        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_price < fv:
                take_vol = min(vol, buy_capacity)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", ask_price, take_vol))
                    buy_capacity -= take_vol
            elif math.isclose(ask_price, fv, abs_tol=0.1):
                take_vol = min(vol, buy_capacity, -initial_pos)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", ask_price, take_vol))
                    buy_capacity -= take_vol
                    initial_pos += take_vol

        initial_pos = position
        for bid_price, bid_vol in buy_orders:
            if bid_price > fv:
                take_vol = min(bid_vol, sell_capacity)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", bid_price, -take_vol))
                    sell_capacity -= take_vol
            elif math.isclose(bid_price, fv, abs_tol=0.1):
                take_vol = min(bid_vol, sell_capacity, initial_pos)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", bid_price, -take_vol))
                    sell_capacity -= take_vol
                    initial_pos -= take_vol
        # 2. Market Making Quotes
        min_edge = 1
        my_bid = min(math.floor(fv) - min_edge, best_bid + 1)
        my_ask = max(math.ceil(fv) + min_edge, best_ask - 1)

        if buy_capacity > 0:
            orders.append(Order("EMERALDS", my_bid, buy_capacity))

        if sell_capacity > 0:
            orders.append(Order("EMERALDS", my_ask, -sell_capacity))

        return orders, fv

    def trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        mid_prices: List[float],  # raw wall_mid history — used for variance only
        prev_s: float,            # last fast EMA (alpha=0.4) — AS fair value anchor
        prev_slow: float,         # last slow EMA (alpha=0.05) — drift baseline
        last_cross: int,          # last crossover direction: +1 (bull), -1 (bear), 0 (none)
    ) -> tuple[List[Order], List[float], float, float, int, float, float]:
        """
        Drift detection via dual-EMA crossover last-state.

        Fitted from tomatoe_model_fittin.ipynb on prices_round_0_day_-1/2.csv (N=20,000).
        ARIMA(0,1,1) on wall_mid: MA≈−0.22, confirming I(1) with mean-reverting noise.
        Crossover events: ~280–305 per 10,000 steps (~17 steps avg between flips).

        ── Core idea ────────────────────────────────────────────────────────────
        velocity = s_fast - s_slow and last_cross are 100% equivalent in direction
        (verified empirically). The crossover event is simply the tick where velocity
        changes sign. Carrying last_cross as a persisted ±1 state is cleaner than
        recomputing sign(velocity) each tick and avoids near-zero floating-point edge
        cases when the gap is tiny at the moment of crossing.

        ── Regime ───────────────────────────────────────────────────────────────
        last_cross == +1  → fast EMA crossed above slow: price in upward drift regime.
                            Safe to BUY cheap asks below fair value.
        last_cross == -1  → fast EMA crossed below slow: price in downward drift regime.
                            Safe to SELL into rich bids above fair value. 
        last_cross ==  0  → no crossover has occurred yet (cold start only).
                            No taking; pure market making.

        ── Why this answers the adverse selection question ───────────────────────
        A stale/baiting quote is dangerous when there is NO confirmed drift regime —
        the bot placed a temporarily mispriced L1 quote with no underlying directional
        move. last_cross being in the correct direction means the dual-EMA system has
        already observed the fast average climbing through the slow average, confirming
        the drift is real, not a single-tick anomaly. We only take into quotes that
        are consistent with the established drift direction; quotes in the opposite
        direction are left alone regardless of how attractive they look.

        ── Parameters (data-fitted) ─────────────────────────────────────────────
        FAST_ALPHA = 0.4   (notebook cell 4)
        SLOW_ALPHA = 0.05  (notebook cell 4)
        No velocity magnitude threshold: empirical analysis showed adding a threshold
        on |velocity| does not change which rows are selected (last_cross and
        sign(velocity) agree 100%), and opportunity rate is roughly flat across
        steps-since-crossover bins, so no additional gate improves quality.
        """
        orders: List[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        if not sell_orders or not buy_orders:
            return orders, mid_prices, prev_s, prev_slow, last_cross, 0, 0

        # Guard: need at least L2 on both sides for stable wall_mid
        if len(sell_orders) < 2 or len(buy_orders) < 2:
            return orders, mid_prices, prev_s, prev_slow, last_cross, 0, 0

        POSITION_LIMIT = 80
        gamma = 0.15
        k = 0.18
        T = 1
        lookback = 12
        FAST_ALPHA = 0.45
        SLOW_ALPHA = 0.08

        wall_mid = (sell_orders[1][0] + buy_orders[1][0]) / 2.0

        # ── Fast EMA: AS fair value ───────────────────────────────────────────
        if prev_s == 0.0:
            prev_s = wall_mid
        s = FAST_ALPHA * wall_mid + (1 - FAST_ALPHA) * prev_s

        # ── Slow EMA: long-run drift baseline ─────────────────────────────────
        if prev_slow == 0.0:
            prev_slow = wall_mid
        slow_ema = SLOW_ALPHA * wall_mid + (1 - SLOW_ALPHA) * prev_slow

        # ── Crossover detection: update last_cross on sign change ─────────────
        # prev_s - prev_slow was the gap last tick; s - slow_ema is the gap now.
        prev_gap = prev_s - prev_slow
        curr_gap = s - slow_ema
        if curr_gap > 0 and prev_gap <= 0:
            last_cross = 1    # bullish cross: fast just rose through slow
        elif curr_gap < 0 and prev_gap >= 0:
            last_cross = -1   # bearish cross: fast just fell through slow

        # ── Drift regime gate ─────────────────────────────────────────────────
        # Take only when a drift is confirmed by last_cross AND the current L1
        # quote is mispriced in the drift direction.
        # last_cross == +1: drift is upward → hit rich bids (best_bid > s)
        # last_cross == -1: drift is downward → lift cheap asks (best_ask < s)
        sell_signal = last_cross == -1
        buy_signal  = last_cross == 1

        # ── Raw wall_mid history for variance ─────────────────────────────────
        mid_prices.append(wall_mid)
        mid_prices = mid_prices[-(lookback + 1):]

        if len(mid_prices) < lookback + 1:
            return orders, mid_prices, s, slow_ema, last_cross, 0, 0

        returns = np.diff(mid_prices)
        var = np.var(returns)

        q = position
        r = s - (q * gamma * var * T)

        buy_qty  =  POSITION_LIMIT - position
        sell_qty = -POSITION_LIMIT - position

        # ── 1. Tactical Taking ────────────────────────────────────────────────
        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_price < s + 1.5 and buy_signal:
                take_vol = min(vol, buy_qty)
                if take_vol > 0:
                    orders.append(Order("TOMATOES", ask_price, take_vol))
                    buy_qty -= take_vol

        for bid_price, bid_vol in buy_orders:
            if bid_price > s -1.5 and sell_signal:
                take_vol = min(bid_vol, -sell_qty)
                if take_vol > 0:
                    orders.append(Order("TOMATOES", bid_price, -take_vol))
                    sell_qty += take_vol

        # ── 2. Passive AS Quotes ───────────────────────────────────────────────
        # Tested different delta's to find fill probability to find parameter k
        # (Delta [16,0%)],[15,0%)],[14,0.05%)],[13,0.25%)],[12,1.35%],[10,2.7%],
        #  [8,3.20%],[4,3.5%],[1,5.15%],[-2,5.15%]) K did not match exponential
        # assumption of model so we penny market.
        delta = gamma * var * T + (2 / gamma) * math.log(1 + gamma / k)

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]

        as_bid = math.floor(r - delta / 2)
        as_ask = math.ceil(r + delta / 2)

        new_bid_price = min(as_bid, best_bid + 1)
        new_ask_price = max(as_ask, best_ask - 1)

        if buy_qty > 0:
            orders.append(Order("TOMATOES", int(new_bid_price), buy_qty))
        if sell_qty < 0:
            orders.append(Order("TOMATOES", int(new_ask_price), sell_qty))

        # Return: orders, raw history, fast EMA, slow EMA, last_cross state,
        #         reservation price (effective fv), curr_gap as signal for logging
        return orders, mid_prices, s, slow_ema, last_cross, r, last_cross

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0

        # Parse persistent state
        if state.traderData:
            try:
                data = jsonpickle.decode(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        for product in state.order_depths:
            current_fv = 0.0
            effective_fv = 0.0
            result[product] = []

            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product], current_fv = self.trade_emeralds(order_depth, position)
                effective_fv = current_fv #stationary market
                signal = current_fv #stationary market

            elif product == "TOMATOES":
                prev_prices   = data.get("TOMATOES_mid", [])
                prev_s        = data.get("TOMATOES_s", 0.0)
                prev_slow     = data.get("TOMATOES_slow", 0.0)
                last_cross    = data.get("TOMATOES_last_cross", 0)

                result[product], mid_prices, new_s, new_slow, new_last_cross, effective_fv, signal = self.trade_tomatoes(
                    order_depth, position, prev_prices, prev_s, prev_slow, last_cross
                )

                data["TOMATOES_mid"]        = mid_prices
                data["TOMATOES_s"]          = new_s
                data["TOMATOES_slow"]       = new_slow
                data["TOMATOES_last_cross"] = new_last_cross
                current_fv = new_s  # log the fast EMA fair value

            self.log_data(state, product, position, result[product], current_fv, effective_fv, signal)

        new_trader_data = jsonpickle.encode(data)
        return result, conversions, new_trader_data
