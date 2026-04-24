import json
import math
import numpy as np
from functools import wraps
from typing import Any, List, Dict
from datamodel import Listing, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState




#|||||||||||||||||||GENERAL CONSTANTS||||||||||||||||||#

MEAN_REVERT = "ASH_COATED_OSMIUM"

OPTIONS_SYMBOLS =[
    'penis1',
    'penis2'
]

OPTION_UNDERLYING_SYMBOL = "OPTION"

POS_LIMITS = {
    MEAN_REVERT: 80,
    OPTION_UNDERLYING_SYMBOL: 0
    }


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

def handle_none(default_active=False, default_cache_prev=True, default_fallback=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 1. Grab settings from call-site or decorator defaults
            active = kwargs.pop('_active', default_active)
            cache_prev = kwargs.pop('_cache', default_cache_prev)
            fallback = kwargs.pop('_fallback', default_fallback)

            if not active:
                return func(self, *args, **kwargs)

            # Execute the method
            result = func(self, *args, **kwargs)

            # Check for None or (None, None...)
            def is_none_like(res):
                if res is None: return True
                if isinstance(res, tuple) and all(x is None for x in res): return True
                return False

            # Unique key for this method and product
            cache_key = f"{self.name}_{func.__name__}"

            if is_none_like(result):
                # 2. FAIL CASE: Try to recover from previous tick's data
                if cache_prev:
                    # Look into the dict populated by get_last_traderData()
                    cached_val = self.last_traderData.get(cache_key)
                    if cached_val is not None:
                        # JSON stores tuples as lists; convert back if necessary
                        return tuple(cached_val) if isinstance(cached_val, list) else cached_val
                
                return fallback
            
            else:
                # 3. SUCCESS CASE: Save to new_trader_data for the NEXT tick
                if cache_prev:
                    self.new_trader_data[cache_key] = result
                return result

        return wrapper
    return decorator

class TraderBase:
    def __init__(self, name, state, new_trader_data):

        self.orders = []

        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data

        self.last_traderData = self.get_last_traderData()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0) # position at beginning of round

        self.expected_position = self.initial_position # update this if you expect a certain change in position e.g. to already hedge


        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume() # gets updated when order created
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

    def get_last_traderData(self):
                        
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except: self.log("ERROR", 'td')

        return last_traderData
    
    @handle_none()
    def get_best_bid_ask(self):
        best_bid = best_ask = None

        try:
            if len(self.mkt_buy_orders) > 0:
                best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0:
                best_ask = min(self.mkt_sell_orders.keys())
        except: pass

        return best_bid, best_ask
    
    @handle_none()
    def get_walls(self):

        bid_wall = wall_mid = ask_wall = None

        try: bid_wall = min([x for x,_ in self.mkt_buy_orders.items()])
        except: pass
        
        try: ask_wall = max([x for x,_ in self.mkt_sell_orders.items()])
        except: pass

        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass

        return bid_wall, wall_mid, ask_wall
    
    @handle_none()
    def get_total_market_buy_sell_volume(self):

        market_bid_volume = market_ask_volume = 0

        try:
            market_bid_volume = sum([v for p, v in self.mkt_buy_orders.items()])
            market_ask_volume = sum([v for p, v in self.mkt_sell_orders.items()])
        except: pass

        return market_bid_volume, market_ask_volume
    
    @handle_none()
    def get_max_allowed_volume(self):
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume
    
    @handle_none()
    def get_order_depth(self):

        order_depth, buy_orders, sell_orders = {}, {}, {}

        try: order_depth: OrderDepth = self.state.order_depths[self.name]
        except: pass
        try: buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
        except: pass
        try: sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except: pass

        return buy_orders, sell_orders  
    
    @handle_none()
    def bid(self, price, volume):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    @handle_none()
    def ask(self, price, volume):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -abs_volume)
        self.max_allowed_sell_volume -= abs_volume
        self.orders.append(order)

class PepperTrader(TraderBase):
    def __init__(self, name, state, prints, new_trader_data):
        super().__init__(name, state, prints, new_trader_data)

class OsmiumTrader(TraderBase):
    def __init__(self, name, state, prints, new_trader_data):
        super().__init__(name, state, prints, new_trader_data)

class OptionsTrader:
                    
    def __init__(self, state, new_trader_data):
        self.options = [TraderBase(options, state, new_trader_data)for options in OPTIONS_SYMBOLS]
        self.underlying = TraderBase(OPTION_UNDERLYING_SYMBOL, state, new_trader_data)

    def get_option_orders(self):
        pass

    def get_orders(self):
        
        orders = {
            

        }

class Trader:
    def run(self,state:TradingState):
        result = {} # All orders to send
        new_trader_data = {}
        all_signals = {} # Dictionary to store signals for all products, to be passed to the logger at the end of the tick
        conversions = 0 # Not used for Prosperity 4, but still required as a return
        
        product_traders = {
            OPTION_UNDERLYING_SYMBOL: OptionsTrader
        }

        for symbol, product_trader in product_traders.items(): # Goes through the currently traded items and gets their current order response
            if symbol in state.order_depths: # Ensures the item traded is in order book

                try:
                    trades, signals = product_trader(state, new_trader_data) # Creates a new instance of the class, to store in trader
                    all_signals[symbol] = signals
                    result.update(trades.get_orders()) # Returns a dictionary, updates the dictionary while removing duplicates (.update similar to .extend for lists)

                except: # Safekeeping
                    pass

        try: final_trader_data = json.dumps(new_trader_data)
        except: final_trader_data = ''
        
        # Pass the signals dictionary to the logger
        logger.flush(state, result, conversions, final_trader_data, all_signals)
        return result, conversions, final_trader_data
    