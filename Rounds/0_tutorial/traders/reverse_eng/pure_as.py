import math
import json
import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol


class Trader:

    def log_data(self, state: TradingState, product: str, position: int, orders: List[Order], fv: float, effective_fv: float):
        """
        Comprehensive logger for ALL order levels.
        Format: timestamp, product, position, fv, [Bids_JSON], [Asks_JSON]
        """
        # Group orders by price to handle multiple orders at the same level
        bid_map = {}
        ask_map = {}

        for o in orders:
            if o.quantity > 0:
                bid_map[o.price] = bid_map.get(o.price, 0) + o.quantity
            else:
                ask_map[o.price] = ask_map.get(o.price, 0) + o.quantity

        # Convert to a stable string format (price:qty)
        # We sort them so your Jupyter parser always sees them in order
        bids_str = ";".join([f"{p}:{q}" for p, q in sorted(bid_map.items(), reverse=True)])
        asks_str = ";".join([f"{p}:{q}" for p, q in sorted(ask_map.items())])

        # Final CSV-friendly print
        # Using a semicolon inside the bid/ask strings so the main comma delimiter works
        print(f"[ALGO],{state.timestamp},{product},{position},{fv:.2f},{effective_fv:.2f},0.0,[{bids_str}],[{asks_str}]")

    def trade_emeralds(self, order_depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = 80
        fv = 10000.0
        
        # Sort books: asks ascending (best ask first), bids descending (best bid first)
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
            elif math.isclose(ask_price, fv, abs_tol=0.1): #and abs(initial_pos) <= 8 and initial_pos < 0:
                # Fair-value book clearing capped at neutral (Overshoot protection)
                take_vol = min(vol, buy_capacity, -initial_pos)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", ask_price, take_vol))
                    buy_capacity -= take_vol
                    initial_pos += take_vol 

        # Reset initial_pos to actual position for the sell-side evaluation
        initial_pos = position 
        for bid_price, bid_vol in buy_orders:
            if bid_price > fv:
                take_vol = min(bid_vol, sell_capacity)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", bid_price, -take_vol))
                    sell_capacity -= take_vol
            elif math.isclose(bid_price, fv, abs_tol=0.1): #and abs(initial_pos) <= 8 and initial_pos > 0:
                # Fair-value book clearing capped at neutral (Overshoot protection)
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

    def trade_tomatoes(self, order_depth: OrderDepth, position: int, mid_prices: List[float]) -> tuple[List[Order], float, List[float]]:
        """
            s - mid l2 market price
            q - difference between current size and counterparty order size
            gamma - sensitivity parameter (how much our quote should move in response to inventory changes)
            var - price variance (calculated using mid price over x rolling window)
            T - time horizon (can be set to 1 for simplicity)
            k - order book liquidity density
            r - reservation price
            delta - optimal spread // 2
        """
        orders: List[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        
        if not sell_orders or not buy_orders:
            return orders
        
        POSITION_LIMIT = 80
        s = (sell_orders[1][0] + buy_orders[1][0]) / 2
        q = 0
        gamma = 0.15 #0.2
        var = 0
        k = 0.5 #0.461
        T = 1

        if int(s) != 0:
           mid_prices.append(s)

        lookback = 10

        # Performance Fix: Keep only the necessary history
        mid_prices = mid_prices[-(lookback + 1):]

        if len(mid_prices) < lookback + 1:
            return orders, mid_prices, 0
        
        returns = np.diff(mid_prices[-(lookback + 1):])
        var = np.var(returns)
        q = position

        # Reservation pricing
        r = s - (q * gamma * var * T)

        # Bid ask spread
        delta = (gamma * var * T + (2 / gamma * math.log(1 + (gamma / k))))

        best_ask = sell_orders[0][0]
        best_buy = buy_orders[0][0]
        # Prices to be sent in, compares if we can profit more by pennying the market
        new_bid_price = min(math.floor((r - delta / 2)), best_buy+1)
        new_ask_price = max(math.ceil((r + delta / 2)), best_ask -1)

        buy_qty = POSITION_LIMIT - position
        if buy_qty > 0:
            orders.append(Order("TOMATOES", int(new_bid_price), buy_qty))

        sell_qty = -POSITION_LIMIT - position # Will be a negative number
        if sell_qty < 0:
            orders.append(Order("TOMATOES", int(new_ask_price), sell_qty))

        # Returning current mid and r (reservation price) as the 'effective fv'
        return orders, mid_prices, r

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        
        # 1. Parse persistent state
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except:
                data = {}
        else:
            data = {}

        for product in state.order_depths:
            current_fv = 0.0
            effective_fv = 0.0
            result[product] = []

            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            prev_prices = data.get("TOMATOES", [])
            
            if product == "EMERALDS":
                result[product], current_fv = self.trade_emeralds(order_depth, position)
                effective_fv = current_fv # For logging consistency, even though EMERALDS doesn't use it
            elif product == "TOMATOES":
                result[product], mid_prices, effective_fv = self.trade_tomatoes(order_depth, position, prev_prices)
                data[product] = mid_prices
                current_fv = mid_prices[-1] if mid_prices else 0.0
        
            self.log_data(state, product, position, result[product], current_fv, effective_fv)
        
        # 2. FIX: Serialize the UPDATED data dictionary, not the original string
        new_trader_data = json.dumps(data)
        
        return result, conversions, new_trader_data