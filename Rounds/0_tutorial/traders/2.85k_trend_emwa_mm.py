import math
import statistics
import json
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol

class Trader:

    def log_data(self, state: TradingState, product: str, position: int, orders: List[Order], fv: float, effective_fv: float, slope: float):
        """
        Comprehensive logger for ALL order levels.
        Format: timestamp, product, position, fv, effective_fv, [Bids_JSON], [Asks_JSON]
        """
        bid_map = {}
        ask_map = {}

        for o in orders:
            if o.quantity > 0:
                bid_map[o.price] = bid_map.get(o.price, 0) + o.quantity
            else:
                ask_map[o.price] = ask_map.get(o.price, 0) + o.quantity

        bids_str = ";".join([f"{p}:{q}" for p, q in sorted(bid_map.items(), reverse=True)])
        asks_str = ";".join([f"{p}:{q}" for p, q in sorted(ask_map.items())])

        print(f"[ALGO],{state.timestamp},{product},{position},{fv:.2f},{effective_fv:.2f},{slope:.2f},[{bids_str}],[{asks_str}]")

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

    def trade_tomatoes(self, order_depth: OrderDepth, position: int, fv_history: List[float]) -> tuple[List[Order], List[float], float]:
        orders: List[Order] = []
        limit = 80
        soft_limit = 70 #60
        skew_factor = 0.9 #2
        min_edge = 4 #3
        
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        
        # FIX 1: Ensure we always return the expected 3-tuple, even if books are empty
        if not sell_orders or not buy_orders:
            fallback_fv = fv_history[-1] if fv_history else 0.0
            return orders, fv_history, fallback_fv
            
        wall_mid = (sell_orders[1][0] + buy_orders[1][0]) / 2.0
        prev_fv = fv_history[-1] if fv_history else wall_mid
        fv = 0.445 * wall_mid + 0.555 * prev_fv
        
        fv_history = fv_history + [fv]
        if len(fv_history) > 50: 
            fv_history.pop(0)
        
        # FIX 2: Set slope to 0 instead of returning early and breaking the unpack logic
        if len(fv_history) < 2:
            slope = 0
        else:
            x = range(len(fv_history))
            slope, intercept = statistics.linear_regression(x, fv_history)
        
    
        direction_check = position * slope
        trend_intensity = abs(slope)

        if direction_check < 0:
            # Opposite to trend: Increase skew factor significantly
            dynamic_skew_factor = skew_factor * ((1 + 0.3*trend_intensity)**3) * (position / soft_limit)**2
        else:
            # With trend: Normal skew or slightly dampened
            dynamic_skew_factor = skew_factor

        skew = (position / soft_limit) * dynamic_skew_factor

        effective_fv = fv - skew
        
        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]
        
        initial_pos = position
        buy_capacity = limit - initial_pos
        sell_capacity = limit + initial_pos
        
        # 1. Tactical Taking (Buying)
        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol # Volume is usually negative in order_depth.sell_orders
            
            # Take the ask if it is at or below our predicted fair value
            # Removing the constraint allows us to open LONG positions aggressively
            if ask_price <= effective_fv and slope > -0.05:
                take_vol = min(vol, buy_capacity)
                if take_vol > 0:
                    orders.append(Order("TOMATOES", ask_price, take_vol))
                    buy_capacity -= take_vol

        # 2. Tactical Taking (Selling)
        
        for bid_price, bid_vol in buy_orders:
            # Take the bid if it is at or above our predicted fair value
            # Removing the constraint allows us to open SHORT positions aggressively
            vol = bid_vol
            if bid_price >= effective_fv and slope > 0.05:
                take_vol = min(vol, sell_capacity)
                if take_vol > 0:
                    orders.append(Order("TOMATOES", bid_price, -take_vol))
                    sell_capacity -= take_vol

        # 2. Market Making Quotes 
        my_bid = min(math.floor(effective_fv) - min_edge, best_bid + 1)
        my_ask = max(math.ceil(effective_fv) + min_edge, best_ask - 1)
        
        # Soft limit bounds evaluation
        buys_placed = (limit - position) - buy_capacity
        sells_placed = (limit + position) - sell_capacity
        
        pending_buy_pos = position + buys_placed
        pending_sell_pos = position - sells_placed
        
        # Suppress accumulating-side quotes safely while allowing liquidation quotes
        target_buy_vol = max(0, soft_limit - pending_buy_pos)
        target_sell_vol = max(0, pending_sell_pos - (-soft_limit))
        
        bid_vol = min(target_buy_vol, buy_capacity)
        ask_vol = min(target_sell_vol, sell_capacity)
        
        if bid_vol > 0:
            orders.append(Order("TOMATOES", my_bid, bid_vol))
        if ask_vol > 0:
            orders.append(Order("TOMATOES", my_ask, -ask_vol))

        return orders, fv_history, effective_fv, slope

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except:
                data = {}
        else:
            data = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            history = data.get(product, [])
            
            if product == "EMERALDS":
                result[product], current_fv = self.trade_emeralds(order_depth, position)
                data[product] = current_fv 
                effective_fv = current_fv
                slope = 0
            elif product == "TOMATOES":
                result[product], updated_history, effective_fv, slope = self.trade_tomatoes(order_depth, position, history)
                # FIX 3: Assign the list instead of appending a list to a list
                data[product] = updated_history
                current_fv = updated_history[-1] if updated_history else 0.0
        
            self.log_data(state, product, position, result[product], current_fv, effective_fv, slope)
        
        new_trader_data = json.dumps(data)
        
        return result, conversions, new_trader_data