from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

class Trader:
    
    def run(self, state: TradingState):
        result = {}
        
        # 1. Hard Constraints for TOMATOES
        TARGET_PRODUCT = "TOMATOES"
        POSITION_LIMIT = 80
        gap = 1
        
        for product in state.order_depths:
            # ONLY trade Tomatoes. Skip everything else (like Emeralds).
            if product != TARGET_PRODUCT:
                continue
                
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # 2. Get current position for Tomatoes
            current_pos = state.position.get(product, 0)
            
            # 3. Pull the Order Book Levels
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue
                
            # Sort the book to get the actual Best Bid and Best Ask
            # (In Prosperity, these are dictionaries where keys are prices)
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            # 4. Calculate the "Anti-Spoof" Mid Price
            # We use the raw Mid because we know L1 volume is a ghost.
            mid_price = (best_bid + best_ask) / 2.0
            
            # 5. Inventory Management (The Skew)
            # This pushes our 'Fair Value' down if we are long, up if we are short.
            # Sensitivity of 1.0 means for every 40 units, we shift by 1 tick.
            inventory_skew = (current_pos / POSITION_LIMIT) * 2.0
            acceptable_price = mid_price - inventory_skew
            
            # 6. Quoting Logic (Market Making)
            
            # --- BUY SIDE (Bids) ---
            if current_pos < POSITION_LIMIT:
                # We want to be at the top of the book (best_bid + 1)
                # But we never bid higher than our acceptable_price minus 1 tick spread
                bid_price = int(min(best_bid + gap, acceptable_price - gap))
                
                # Maximize the order size to reach the limit
                buy_quantity = POSITION_LIMIT - current_pos
                orders.append(Order(product, bid_price, buy_quantity))
                print(f"TOMATOES BID: {buy_quantity} @ {bid_price} (Mid: {mid_price})")

            # --- SELL SIDE (Asks) ---
            if current_pos > -POSITION_LIMIT:
                # We want to be at the top of the book (best_ask - 1)
                # But we never sell lower than our acceptable_price plus 1 tick spread
                ask_price = int(max(best_ask - gap, acceptable_price + gap))
                
                # Sell quantity is negative in Prosperity
                sell_quantity = - (POSITION_LIMIT + current_pos)
                orders.append(Order(product, ask_price, sell_quantity))
                print(f"TOMATOES ASK: {sell_quantity} @ {ask_price} (Pos: {current_pos})")

            result[product] = orders

        return result, 0, ""