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
        mid_prices: List[float],   # raw wall_mid history — used for variance only
        prev_s: float,             # last EMA value — used for fair value only
    ) -> tuple[List[Order], List[float], float, float]:
        """
        s      - EMA fair value (smoothed wall_mid), used for reservation pricing
        prev_s - previous EMA value, stored separately from mid_prices
        mid_prices - raw wall_mid history, used for variance estimation only
        q      - current inventory
        gamma  - inventory risk aversion
        var    - price variance over raw wall_mid returns
        k      - order book shape parameter (fitted: ln(2) / half_spread)
        T      - time horizon
        r      - reservation price
        delta  - optimal full spread
        """
        orders: List[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        if not sell_orders or not buy_orders:
            return orders, mid_prices, prev_s, 0, 0

        # Guard: need at least L2 on both sides
        if len(sell_orders) < 2 or len(buy_orders) < 2:
            return orders, mid_prices, prev_s, 0, 0

        POSITION_LIMIT = 80
        gamma = 0.15
        k = 0.3
        T = 1
        lookback = 10
        alpha = 0.45

        wall_mid = (sell_orders[1][0] + buy_orders[1][0]) / 2.0

        # EMA fair value — prev_s stored separately, not mixed into mid_prices
        if prev_s == 0.0:
            prev_s = wall_mid
        s = alpha * wall_mid + (1-alpha) * prev_s

        # Raw wall_mid history for variance
        mid_prices.append(wall_mid)
        mid_prices = mid_prices[-(lookback + 1):]

        if len(mid_prices) < lookback + 1:
            return orders, mid_prices, s, 0, 0

        returns = np.diff(mid_prices)
        var = np.var(returns)

        q = position
        r = s - (q * gamma * var * T)

        buy_qty = POSITION_LIMIT - position
        sell_qty = -POSITION_LIMIT - position

        buy_signal = False
        sell_signal = False

        # 1. Tactical Taking (Buying)
        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_price < s and buy_signal:
                take_vol = min(vol, buy_qty)
                if take_vol > 0:
                    orders.append(Order("TOMATOES", ask_price, take_vol))
                    buy_qty -= take_vol

        # 2. Tactical Taking (Selling)
        for bid_price, bid_vol in buy_orders:
            if bid_price > s and sell_signal:
                take_vol = min(bid_vol, sell_qty)
                if take_vol > 0:
                    orders.append(Order("TOMATOES", bid_price, take_vol))
                    sell_qty += take_vol

        # Passive quotes
        delta = gamma * var * T + (2 / gamma) * math.log(1 + gamma / k)
    

        best_ask = sell_orders[0][0]
        best_bid= buy_orders[0][0]

        #Tested different delta's to find fill probability to find paramater k (Delta [16,0%)], [15,0%)],[14,0.05%)],[13,0.25%)],[12,1.35%],[10, 2.7%],[8, 3.20%],[4,3.5%],[1,5.15%],[-2,5.15%]) K did not match exponential assumption of model so we can just penny market

        as_bid = math.floor(r - delta / 2)
        as_ask = math.ceil(r + delta / 2)

        new_bid_price = min(as_bid, best_bid + 1)
        new_ask_price = max(as_ask, best_ask - 1)

        if buy_qty > 0:
            orders.append(Order("TOMATOES", int(new_bid_price), buy_qty))
        if sell_qty < 0:
            orders.append(Order("TOMATOES", int(new_ask_price), sell_qty))

        # Return: orders, raw history, updated EMA, reservation price as effective fv
        return orders, mid_prices, s, r, 0

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
                prev_prices = data.get("TOMATOES_mid", [])
                prev_s = data.get("TOMATOES_s", 0.0)

                result[product], mid_prices, new_s, effective_fv, signal = self.trade_tomatoes(
                    order_depth, position, prev_prices, prev_s
                )

                data["TOMATOES_mid"] = mid_prices
                data["TOMATOES_s"] = new_s
                current_fv = new_s  # log the EMA fair value

            self.log_data(state, product, position, result[product], current_fv, effective_fv, signal)

        new_trader_data = jsonpickle.encode(data)
        return result, conversions, new_trader_data