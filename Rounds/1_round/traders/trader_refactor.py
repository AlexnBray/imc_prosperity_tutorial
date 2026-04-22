import json
import math
import numpy as np
from typing import Any, List, Dict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str, signals: Dict[Symbol, Any]) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                    signals
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                    signals
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out

logger = Logger()

class Trader:  

    @staticmethod
    def get_sell_and_buy_orders(order_depth):
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        return sell_orders, buy_orders
    
    @staticmethod
    def get_best_bid_and_ask(buy_orders: List[tuple[int, int]], sell_orders: List[tuple[int, int]]):
        # Best Bid is the highest price in buy_orders
        best_bid = buy_orders[0][0] if buy_orders else None 
        # Best Ask is the lowest price in sell_orders
        best_ask = sell_orders[0][0] if sell_orders else None
        
        return best_bid, best_ask
    
    @staticmethod
    def calculate_vwap(buy_orders, sell_orders, default, levels=2):
        total_nominal = 0.0
        total_volume = 0.0

        # Process Sell side (Asks)
        for i in range(min(levels, len(sell_orders))):
            price, qty = sell_orders[i]
            total_nominal += price * abs(qty)
            total_volume += abs(qty)

        # Process Buy side (Bids)
        for i in range(min(levels, len(buy_orders))):
            price, qty = buy_orders[i]
            total_nominal += price * abs(qty)
            total_volume += abs(qty)

        if total_volume > 0:
            return total_nominal / total_volume
        
        return default 
    
    @staticmethod
    def calculate_micro_price(buy_orders, sell_orders, default):


        if not buy_orders or not sell_orders:
            return default

        best_bid_price, bid_vol = buy_orders[0]
        best_ask_price, ask_vol = sell_orders[0]
        
        # Note: Using absolute volume for the denominator
        bid_vol = abs(bid_vol)
        ask_vol = abs(ask_vol)

        total_vol = bid_vol + ask_vol
        
        if total_vol > 0:
            # Weighted by the depth of the opposite side
            return (best_bid_price * ask_vol + best_ask_price * bid_vol) / total_vol

        return default

    @staticmethod
    def get_last_valid_entry(time_series: List[tuple[int, Any]], default_price = np.nan) -> tuple[int, Any]:
        return next(
            (p for p in reversed(time_series) if not math.isnan(float(p[1]))), 
            (0, default_price)
        )
    
    def trade_pepper_root(
            self, 
            order_depth: OrderDepth, 
            position: int, 
            price_history: List[tuple[int, float]], 
            current_time:int,
            stored_intercept: float):
        
        # ── General Params ─────────────────────────────────────────────────
        PRODUCT_ID = "INTARIAN_PEPPER_ROOT"
        POSITION_LIMIT = 80
        INTERCEPT_INITILISATION_TICKS = 10 # Ticks used to find intercept
        N_ORACLE = 8 # Ticks ahead of the market is our fair value predicting
        SLOPE = 0.001 # Slope of linear fair value function

        orders: List[Order] = []
        EARLY_EXIT = orders, price_history, None, None, stored_intercept

        # ── A-S Model Params ───────────────────────────────────────────────
        VARIANCE_SAMPLE_SIZE = 10
        gamma = 0.002
        k = 0.14
        position_offset = 6
        MIN_SPREAD = 4

        # ── Fair Value Calculation ─────────────────────────────────────────
        sell_orders, buy_orders = self.get_sell_and_buy_orders(order_depth)
        best_bid, best_ask = self.get_best_bid_and_ask(buy_orders, sell_orders)

        if not price_history and best_bid is None and best_ask is None:
            return EARLY_EXIT  # truly dead tick, nothing we can do

        if not price_history:
            seed_price = (best_bid + best_ask) / 2 if best_bid and best_ask else best_bid or best_ask
            price_history.append((current_time, seed_price))

        fb_t, fb_p = self.get_last_valid_entry(price_history)
        tdelta = current_time - fb_t
        vwap_fallback = fb_p + tdelta * SLOPE

        vwap = self.calculate_vwap(buy_orders, sell_orders, vwap_fallback)

        price_history.append((current_time, vwap))

        # TRUNCATION: Keep only what is strictly necessary as Trader state is limited size
        buffer_limit = max(VARIANCE_SAMPLE_SIZE, INTERCEPT_INITILISATION_TICKS) + 1
        if len(price_history) > buffer_limit:
            price_history = price_history[-buffer_limit:]

        ts, prices = zip(*price_history)
        ts = np.array(ts)
        prices = np.array(prices)

        intercept = stored_intercept
        if intercept is None or not np.isfinite(float(intercept)):
            if len(prices) >= INTERCEPT_INITILISATION_TICKS:
                # Linear regression logic: price = slope * time + intercept
                
                intercepts = prices - (SLOPE * ts)
                intercept = float(np.mean(intercepts))
            else:
                # Still initializing the intercept buy up inventory to long
                if sell_orders: orders.append(Order(PRODUCT_ID, best_ask + 2, -2))
                if buy_orders: orders.append(Order(PRODUCT_ID, best_bid + 2, 15))
                return EARLY_EXIT
        
        s = intercept + SLOPE * (current_time + N_ORACLE * 100)

        # ── Rolling Variance and A-S Reservation Price Calculation ───────────
        if len(price_history) >= VARIANCE_SAMPLE_SIZE:
            variance = max(float(np.var(np.diff(prices))), 1e-6)
        else:
            variance = 1.0

        q = position - position_offset

        r = s - (q * gamma * variance)

        # ── A-S spread (symmetric base) ──────────────────────────────────────
        delta = (gamma * variance + (2 / gamma * math.log(1 + (gamma / k))))
        spread = max(delta, MIN_SPREAD)

        bid_price = math.floor(r - spread/2)
        ask_price = math.ceil(r + spread/2)

        buy_cap = POSITION_LIMIT - position
        sell_cap = -POSITION_LIMIT - position

        if buy_cap > 0 and np.isfinite(bid_price):
            orders.append(Order(PRODUCT_ID, int(bid_price), buy_cap))
        if sell_cap < 0 and np.isfinite(ask_price):
            orders.append(Order(PRODUCT_ID, int(ask_price), sell_cap))
        
        signals = {
            "s": round(s, 2) if not math.isnan(s) else None,
            "r": round(r, 2) if not math.isnan(r) else None,
        }

        return orders, price_history, r, s, intercept, signals

    def trade_osmium(self, order_depth: OrderDepth, position: int):
        orders: List[Order] = []
        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        else:
            data = {
                "pepper_history": [], 
                "pepper_intercept": None
            }

        result = {}
        all_signals = {} # Collect signals for all products here
        conversions = 0

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "INTARIAN_PEPPER_ROOT":
                orders, history, signals = self.trade_pepper_root(
                    order_depth, 
                    position, 
                    data["pepper_history"], 
                    state.timestamp, 
                    data["pepper_intercept"]
                )
                result[product] = orders
                data["pepper_history"] = history
                data["pepper_intercept"] = intercept

        trader_data = json.dumps(data)
        
        # Pass the signals dictionary to the logger
        logger.flush(state, result, conversions, trader_data, all_signals)
        return result, conversions, trader_data