import math
import json
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol


class Trader:

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def log_data(self, state: TradingState, product: str, position: int,
                 orders: List[Order], fv: float, effective_fv: float):
        bid_map = {}
        ask_map = {}
        for o in orders:
            if o.quantity > 0:
                bid_map[o.price] = bid_map.get(o.price, 0) + o.quantity
            else:
                ask_map[o.price] = ask_map.get(o.price, 0) + o.quantity

        bids_str = ";".join([f"{p}:{q}" for p, q in sorted(bid_map.items(), reverse=True)])
        asks_str = ";".join([f"{p}:{q}" for p, q in sorted(ask_map.items())])
        print(f"[ALGO],{state.timestamp},{product},{position},{fv:.2f},{effective_fv:.2f},[{bids_str}],[{asks_str}]")

    # ─────────────────────────────────────────────────────────────────────────
    # EMERALDS — unchanged
    # ─────────────────────────────────────────────────────────────────────────

    def trade_emeralds(self, order_depth: OrderDepth, position: int) -> Tuple[List[Order], float]:
        orders: List[Order] = []
        limit = 80
        fv = 10000.0

        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders  = sorted(order_depth.buy_orders.items(), reverse=True)

        if not sell_orders or not buy_orders:
            return orders, fv

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]

        initial_pos   = position
        buy_capacity  = limit - position
        sell_capacity = limit + position

        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_price < fv:
                take_vol = min(vol, buy_capacity)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", ask_price, take_vol))
                    buy_capacity -= take_vol
            elif math.isclose(ask_price, fv, abs_tol=0.1) and abs(initial_pos) <= 8 and initial_pos < 0:
                take_vol = min(vol, buy_capacity, -initial_pos)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", ask_price, take_vol))
                    buy_capacity -= take_vol
                    initial_pos  += take_vol

        initial_pos = position
        for bid_price, bid_vol in buy_orders:
            if bid_price > fv:
                take_vol = min(bid_vol, sell_capacity)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", bid_price, -take_vol))
                    sell_capacity -= take_vol
            elif math.isclose(bid_price, fv, abs_tol=0.1) and abs(initial_pos) <= 8 and initial_pos > 0:
                take_vol = min(bid_vol, sell_capacity, initial_pos)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", bid_price, -take_vol))
                    sell_capacity -= take_vol
                    initial_pos   -= take_vol

        min_edge = 1
        my_bid = min(math.floor(fv) - min_edge, best_bid + 1)
        my_ask = max(math.ceil(fv)  + min_edge, best_ask - 1)

        if buy_capacity  > 0:
            orders.append(Order("EMERALDS", my_bid,  buy_capacity))
        if sell_capacity > 0:
            orders.append(Order("EMERALDS", my_ask, -sell_capacity))

        return orders, fv

    # ─────────────────────────────────────────────────────────────────────────
    # TOMATOES — EWMA fair value + linreg trend gate + unclamped skew quotes
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _linreg_slope(ys: list) -> float:
        """OLS slope of ys over t = 0..n-1, in price units per tick."""
        n = len(ys)
        if n < 2:
            return 0.0
        xm = (n - 1) / 2.0
        ym = sum(ys) / n
        num = sum((i - xm) * (ys[i] - ym) for i in range(n))
        den = sum((i - xm) ** 2 for i in range(n))
        return num / den if den else 0.0

    def trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        prev_fv: float,
        wall_mid_history: list,   # mutable — mutated in-place, persist via traderData
    ) -> Tuple[List[Order], float, float]:
        """Returns (orders, new_fv, effective_fv)."""
        orders: List[Order] = []

        LIMIT      = 80
        SOFT       = 40    # inventory normaliser for skew + MM quota
        SKEW_F     = 0.5    # eff_fv = fv - (position / SOFT) * SKEW_F
        MIN_EDGE   = 6     # quote offset from eff_fv on each side
        SLOPE_WIN  = 10      # linreg lookback (ticks)
        SLOPE_GATE = 0.08    # |slope| threshold for trend-regime decisions
        MAX_TAKE   = 15      # max units for FV-boundary position-reducing squeeze

        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders  = sorted(order_depth.buy_orders.items(), reverse=True)

        if not sell_orders or not buy_orders:
            return orders, prev_fv, prev_fv

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]

        # L2 mid — more stable than L1 during bot spread compressions
        if len(sell_orders) >= 2 and len(buy_orders) >= 2:
            wall_mid = (sell_orders[1][0] + buy_orders[1][0]) / 2.0
        else:
            wall_mid = (best_ask + best_bid) / 2.0

        # EWMA fair value (ARIMA(0,1,1) equivalent, α = 0.445)
        fv = wall_mid if prev_fv == 0 else 0.445 * wall_mid + 0.555 * prev_fv

        # Linreg slope over rolling window of wall_mid
        wall_mid_history.append(wall_mid)
        if len(wall_mid_history) > SLOPE_WIN:
            wall_mid_history.pop(0)
        slope = self._linreg_slope(wall_mid_history)

        # Inventory skew — shifts eff_fv down when long, up when short.
        # Because TOMATOES fv sits inside the spread (unlike EMERALDS),
        # quotes are NOT clamped to best_bid+1 / best_ask-1 — that would
        # kill the skew entirely. Instead eff_fv ± MIN_EDGE places quotes
        # inside the book; as position grows the ask drifts deeper in
        # (filling faster) while the bid retreats (filling slower).
        skew   = (position / SOFT) * SKEW_F
        eff_fv = fv - skew

        buy_cap  = LIMIT - position
        sell_cap = LIMIT + position
        init_pos = position

        # ── Taking ───────────────────────────────────────────────────────────
        # Slope gate: block directional takes when trend strongly opposes.
        # Position-reducing squeezes at the FV boundary bypass the gate.
        allow_long_takes  = slope >= -SLOPE_GATE
        allow_short_takes = slope <=  SLOPE_GATE

        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_price < eff_fv:
                if allow_long_takes and buy_cap > 0:
                    take = min(vol, buy_cap)
                    orders.append(Order("TOMATOES", ask_price, take))
                    buy_cap -= take
            elif math.isclose(ask_price, eff_fv, abs_tol=0.5) and init_pos < 0 and abs(init_pos) <= MAX_TAKE:
                take = min(vol, buy_cap, -init_pos)
                if take > 0:
                    orders.append(Order("TOMATOES", ask_price, take))
                    buy_cap  -= take
                    init_pos += take

        init_pos = position
        for bid_price, bid_vol in buy_orders:
            if bid_price > eff_fv:
                if allow_short_takes and sell_cap > 0:
                    take = min(bid_vol, sell_cap)
                    orders.append(Order("TOMATOES", bid_price, -take))
                    sell_cap -= take
            elif math.isclose(bid_price, eff_fv, abs_tol=0.5) and init_pos > 0 and abs(init_pos) <= MAX_TAKE:
                take = min(bid_vol, sell_cap, init_pos)
                if take > 0:
                    orders.append(Order("TOMATOES", bid_price, -take))
                    sell_cap  -= take
                    init_pos  -= take

        # ── Market making ─────────────────────────────────────────────────────
        my_bid = math.floor(eff_fv) - MIN_EDGE
        my_ask = math.ceil(eff_fv)  + MIN_EDGE

        buys_placed   = (LIMIT - position) - buy_cap
        sells_placed  = (LIMIT + position) - sell_cap
        pending_long  = position + buys_placed
        pending_short = position - sells_placed

        bid_vol = min(max(0, SOFT - pending_long),  buy_cap)
        ask_vol = min(max(0, pending_short + SOFT),  sell_cap)

        # During strong trends, halve passive volume on the accumulating side
        if slope < -SLOPE_GATE:
            bid_vol = bid_vol // 2
        elif slope > SLOPE_GATE:
            ask_vol = ask_vol // 2

        if bid_vol > 0:
            orders.append(Order("TOMATOES", my_bid,  bid_vol))
        if ask_vol > 0:
            orders.append(Order("TOMATOES", my_ask, -ask_vol))

        return orders, fv, eff_fv

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result     = {}
        conversions = 0

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            position  = state.position.get(product, 0)
            prev_fv   = data.get(f"{product}_fv", 0.0)

            current_fv   = 0.0
            effective_fv = 0.0
            result[product] = []

            if product == "EMERALDS":
                result[product], current_fv = self.trade_emeralds(order_depth, position)
                effective_fv = current_fv
                data[f"{product}_fv"] = current_fv

            elif product == "TOMATOES":
                wall_mid_history = data.get("TOMATOES_wmh", [])
                result[product], current_fv, effective_fv = self.trade_tomatoes(
                    order_depth, position, prev_fv, wall_mid_history
                )
                data[f"{product}_fv"]  = current_fv
                # wall_mid_history was mutated in-place by trade_tomatoes
                data["TOMATOES_wmh"] = wall_mid_history

            self.log_data(state, product, position, result[product], current_fv, effective_fv)

        new_trader_data = json.dumps(data)
        return result, conversions, new_trader_data