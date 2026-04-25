import json
import math
import numpy as np
from functools import wraps
from typing import Any, List, Dict
from datamodel import Listing, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from statistics import NormalDist

OPTION_UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
HYDROGEL_PACK = "HYDROGEL_PACK"

POS_LIMITS = {
    OPTION_UNDERLYING_SYMBOL: 200,
    HYDROGEL_PACK: 200
    }

##########COINTEGRATION / HYDRO MM CONSTANTS########################################################
# OLS fit from round-3 data (HYDRO ~= alpha + beta * VELVET). Update from notebook when re-calibrated.
COINT_BETA = 0.19762768551131238
COINT_ALPHA = 8953.242130456325

# EWMA memory (in ticks): mean can adapt faster, variance smoother.
SPREAD_MEAN_HALFLIFE_TICKS = 220.0
SPREAD_VAR_HALFLIFE_TICKS = 350.0

# Overlay hysteresis: enter on larger dislocations, exit only after partial mean reversion.
Z_ENTER = 1.2
Z_EXIT = 0.45

# HYDRO quoting / execution defaults (aggressive but still risk-aware).
MIN_SPREAD_TO_QUOTE = 1
BASE_QUOTE_SIZE = 14
MAX_OVERLAY_SIZE = 42
INVENTORY_SOFT_BAND = 120
TAKER_TRIGGER_Z = 1.85
MAX_TAKER_SIZE = 30

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

        try: bid_wall = max([x for x,_ in self.mkt_buy_orders.items()])
        except: pass
        
        try: ask_wall = min([x for x,_ in self.mkt_sell_orders.items()])
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
    
    @staticmethod
    def _alpha_from_halflife(half_life_ticks):
        if half_life_ticks <= 1:
            return 1.0
        return 1.0 - math.exp(math.log(0.5) / half_life_ticks)
    
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))


class HydrogelPack:
    def __init__(self,state,new_trader_data):
        self.state = state
        self.new_trader_data = new_trader_data

        self.hydrogel = TraderBase(HYDROGEL_PACK, state, new_trader_data)
        self.velvetfruit = TraderBase(OPTION_UNDERLYING_SYMBOL, state, new_trader_data)
        # Short aliases used throughout the strategy body.
        self.h = self.hydrogel
        self.v = self.velvetfruit

        try:
            self.last_td = self.h.last_traderData
        except Exception:
            self.last_td = {}


    def _ewma(self, key:str, alpha:float, value:float) -> float:

        prev = float(self.last_td.get(key,value))
        nxt = alpha * value + (1.0 -alpha ) * prev
        self.new_trader_data[key] = nxt

        return nxt
  
    def _calc_spread(self, hydro_mid, velvet_mid):
        return hydro_mid - (COINT_ALPHA + COINT_BETA *velvet_mid)

    # Keep naming compatible with call-sites below.
    def _compute_spread(self, hydro_mid, velvet_mid):
        return self._calc_spread(hydro_mid, velvet_mid)

    def _compute_z(self, spread): 
        a_m = TraderBase._alpha_from_halflife(SPREAD_MEAN_HALFLIFE_TICKS) # slow alpha, mean should remain stable
        a_v = TraderBase._alpha_from_halflife(SPREAD_VAR_HALFLIFE_TICKS) #faster alpha, varaince needs to adapt

        mu = self._ewma("hydro_mu", a_m, spread) # rolling mean of spread

        dev = spread - mu
        var = self._ewma("hydro_var", a_v, dev * dev)
        st = math.sqrt(max(1e-9, var)) # guard against negative/zero variance to avoid erros
        z = (spread - mu) / st
        return z, mu, st


    def _coint_overlay(self, z):

        prev = bool(self.last_td.get("hydro_coint_on", False))
        now = prev

        if not prev and abs(z) >= Z_ENTER:
            now = True
        if prev and abs(z) <= Z_EXIT:
            now = False
        
        self.new_trader_data['hydro_coint_on'] = now
        return now

    # Keep naming compatible with call-sites below.
    def _overlay_active(self, z):
        return self._coint_overlay(z)

    def _inventory_skew_ticks(self, pos):
        if INVENTORY_SOFT_BAND <= 0:
            return 0.0
        return TraderBase._clamp(pos / INVENTORY_SOFT_BAND, -2.0, 2.0)

    def stoikov(self, position):
        pass
    
    def get_orders(self):
        # 1. Validation: Ensure we have data for both legs
        if self.h.wall_mid is None or self.v.wall_mid is None:
            return {}

        # 2. Signal Generation
        hydro_mid = float(self.h.wall_mid)
        velvet_mid = float(self.v.wall_mid)
        spread = self._compute_spread(hydro_mid, velvet_mid)
        z, _, st = self._compute_z(spread)
        
        # Overlay determines if we are currently in an active trade signal
        overlay_on = self._overlay_active(z)

        # 3. Execution Logic (Pure Taker)
        # We only trade if the signal is active (overlay_on)
        if overlay_on and abs(z) >= TAKER_TRIGGER_Z:
            # SPREAD IS TOO HIGH (Z > 0): Short Hydrogel by hitting the Best Bid
            if z > 0:
                quantity = self.h.max_allowed_sell_volume
                if quantity > 0:
                    # Selling at the best bid price ensures an immediate fill
                    self.h.ask(self.h.best_bid, quantity)
            
            # SPREAD IS TOO LOW (Z < 0): Long Hydrogel by hitting the Best Ask
            elif z < 0:
                quantity = self.h.max_allowed_buy_volume
                if quantity > 0:
                    # Buying at the best ask price ensures an immediate fill
                    self.h.bid(self.h.best_ask, quantity)
        return {HYDROGEL_PACK: self.h.orders}

class Trader:
    def run(self,state:TradingState):
        result = {} # All orders to send
        new_trader_data = {}
        all_signals = {} # Dictionary to store signals for all products, to be passed to the logger at the end of the tick
        conversions = 0 # Not used for Prosperity 4, but still required as a return
        
        product_traders = {
            HYDROGEL_PACK: HydrogelPack,
        }

        for symbol, product_trader in product_traders.items(): # Goes through the currently traded items and gets their current order response
            if symbol in state.order_depths: # Ensures the item traded is in order book
                try:
                    trades = product_trader(state, new_trader_data) # Creates a new instance of the class, to store in trader
                    # IMC convention: strategy returns orders dict only; run() returns (orders, conversions, traderData).
                    orders = trades.get_orders()
                    result.update(orders) # Returns a dictionary, updates the dictionary while removing duplicates (.update similar to .extend for lists)
                    all_signals[symbol] = {}
                except: # Safekeeping
                    logger.print(f"ERROR in trader for {symbol}")

        try: final_trader_data = json.dumps(new_trader_data)
        except: final_trader_data = ''
        
        # Pass the signals dictionary to the logger
        logger.flush(state, result, conversions, final_trader_data, all_signals)
        
        return result, conversions, final_trader_data