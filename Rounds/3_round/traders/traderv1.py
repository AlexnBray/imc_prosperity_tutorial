import json
import math
import numpy as np
from functools import wraps
from typing import Any, List, Dict
from datamodel import Listing, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from statistics import NormalDist

_N = NormalDist()




#|||||||||||||||||||GENERAL CONSTANTS||||||||||||||||||#

OPTION_SYMBOLS =[
    #"VEV_4000", 
    #"VEV_4500", 
    "VEV_5000", 
    "VEV_5100", 
    "VEV_5200", 
    "VEV_5300", 
    "VEV_5400", 
    "VEV_5500", 
    #"VEV_6000", 
    #"VEV_6500",
]

MM_SYMBOLS = [
    "VEV_4000", 
    "VEV_4500",  
]

OPTION_UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
HYDROGEL_PACK = "HYDROGEL_PACK"

POS_LIMITS = {
    **{symbol: 300 for symbol in OPTION_SYMBOLS},
    OPTION_UNDERLYING_SYMBOL: 200,
    HYDROGEL_PACK: 200
    }
##########COINTEGRATION############COINTEGRATION############COINTEGRATION############COINTEGRATION############COINTEGRATION###
COINT_BETA = 1.0 # REPLACE WITH YOUR NOTEBOOK OLS/COINTEGRATI
COINT_ALPHA = 0.0

SPREAD_MEAN_HALFLIFE_TICKS = 400.0
SPREAD_VAR_HALFLIFE_TICKS = 600.0

Z_ENTER = 1.6
Z_EXIT = 0.7

####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS  

DAY = 3

DAYS_PER_YEAR = 365

THR_OPEN, THR_CLOSE = 0.5, 0
LOW_VEGA_THR_ADJ = 0.5

THEO_NORM_WINDOW = 20

IV_SCALPING_THR = 0.7
IV_SCALPING_WINDOW = 100

# UNDERLYING
underlying_mean_reversion_thr = 15
underlying_mean_reversion_window = 10

# OPTIONS
options_mean_reversion_thr = 5
options_mean_reversion_window = 30


def session_day()-> int:
    pass

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

        self.hydrogel = TraderBase(HYDROGEL_PACK, state, new_trader_data)
        self.velvetfruit = TraderBase(OPTION_UNDERLYING_SYMBOL, state, new_trader_data)

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

    def stoikov(self, position):
        pass


    def get_orders(self):
        
        if self.h.wall_mid is None or self.v.wall_mid is None:
            return {}

        hydro_mid = float(self.h.wall_mid)
        velvet_mid = float(self.v.wall_mid)

        spread = self._compute_spread(hydro_mid, velvet_mid)
        z, _, st = self._compute_z(spread)
        overlay_on = self._overlay_active(z)

        # Persist diagnostics for logs/analysis.
        self.new_trader_data["hg_spread_last"] = spread
        self.new_trader_data["hg_spread_z"] = z
        self.new_trader_data["hg_spread_st"] = st

        # Start from your preferred setup: touch-improved prices around walls.
        bid_price = int(self.h.bid_wall + 1)
        ask_price = int(self.h.ask_wall - 1)

        # Walk bids: try to overbid meaningful queues while staying anchored near fair.
        for buy_p, buy_v in self.h.mkt_buy_orders.items():
            overbid_price = int(buy_p + 1)
            if buy_v > 1 and overbid_price < self.h.wall_mid:
                bid_price = max(bid_price, overbid_price)
                break
            elif buy_p < self.h.wall_mid:
                bid_price = max(bid_price, int(buy_p))
                break

        # Walk asks: try to undercut meaningful queues while staying anchored near fair.
        for sell_p, sell_v in self.h.mkt_sell_orders.items():
            underbid_price = int(sell_p - 1)
            if sell_v > 1 and underbid_price > self.h.wall_mid:
                ask_price = min(ask_price, underbid_price)
                break
            elif sell_p > self.h.wall_mid:
                ask_price = min(ask_price, int(sell_p))
                break

        # Apply inventory and spread-signal skew after we derive baseline quotes.
        pos = self.state.position.get(HYDROGEL_PACK, 0)
        inv_skew = self._inventory_skew_ticks(pos)
        coint_skew = TraderBase._clamp(z / 1.5, -2.0, 2.0) if overlay_on else 0.0
        total_skew = inv_skew + coint_skew
        bid_px = int(round(bid_price - total_skew))
        ask_px = int(round(ask_price - total_skew))

        # Ensure valid book shape.
        if bid_px >= ask_px:
            mid = 0.5 * (self.h.best_bid + self.h.best_ask)
            bid_px = int(math.floor(mid))
            ask_px = int(math.ceil(mid))
        if ask_px <= bid_px:
            ask_px = bid_px + 1

        # Size model: wider spread => larger base clips.
        book_spread = self.h.best_ask - self.h.best_bid
        spread_bonus = max(0, book_spread - 2)
        base_size_now = BASE_QUOTE_SIZE + 2 * spread_bonus
        buy_sz = min(self.h.max_allowed_buy_volume, base_size_now)
        sell_sz = min(self.h.max_allowed_sell_volume, base_size_now)

        # Overlay sizing: larger dislocation => larger correcting-side size.
        if overlay_on:
            overlay_mag = TraderBase._clamp((abs(z) - Z_ENTER) / 1.2, 0.0, 1.0)
            bump = int(round(overlay_mag * (MAX_OVERLAY_SIZE - base_size_now)))
            if z < 0:
                buy_sz = min(self.h.max_allowed_buy_volume, base_size_now + bump)
            elif z > 0:
                sell_sz = min(self.h.max_allowed_sell_volume, base_size_now + bump)

        # Send passive quotes.
        if buy_sz > 0:
            self.h.bid(bid_px, buy_sz)
        if sell_sz > 0:
            self.h.ask(ask_px, sell_sz)

        # Optional aggressive leg on very large dislocation.
        if overlay_on and abs(z) >= TAKER_TRIGGER_Z:
            taker_mag = TraderBase._clamp((abs(z) - TAKER_TRIGGER_Z) / 1.4, 0.0, 1.0)
            taker_sz = int(round(6 + taker_mag * (MAX_TAKER_SIZE - 6)))
            if z > 0 and self.h.max_allowed_sell_volume > 0:
                self.h.ask(self.h.best_bid, min(self.h.max_allowed_sell_volume, taker_sz))
            elif z < 0 and self.h.max_allowed_buy_volume > 0:
                self.h.bid(self.h.best_ask, min(self.h.max_allowed_buy_volume, taker_sz))

        logger.print(
            f"HG q=({bid_px},{ask_px}) spr={book_spread} z={z:.2f} on={overlay_on} pos={pos}"
        )

        return {HYDROGEL_PACK: self.h.orders}

class MarketMaker:
    def __init__(self, state, new_trader_data):
        self.mm = [TraderBase(mm,state,new_trader_data) for mm in MM_SYMBOLS]

        self.state = state
        self.last_traderData = self.mm.last_traderData
        self.new_trader_data = new_trader_data

        self.indicators = self.calculate_indicators()

class OptionTrader:
                    
    def __init__(self, state, new_trader_data):

        self.options = [TraderBase(os, state, new_trader_data) for os in OPTION_SYMBOLS]
        self.underlying = TraderBase(OPTION_UNDERLYING_SYMBOL, state, new_trader_data)

        self.state = state
        self.last_traderData = self.underlying.last_traderData
        self.new_trader_data = new_trader_data

        self.indicators = self.calculate_indicators()
    def get_option_values(self, S, K, TTE):

        def bs_call(S, K, TTE, s, r=0):        
            d1 = (math.log(S/K) + (r + 0.5 * s**2) * TTE) / (s * TTE**0.5)
            d2 = d1 - s * TTE**0.5
            return S * _N.cdf(d1) - K * math.exp(-r * TTE) * _N.cdf(d2), _N.cdf(d1)

        def bs_vega(S, K, TTE, s, r=0):
            d1 = d1 = (math.log(S/K) + (r + 0.5*s**2) * TTE) / (s * TTE**0.5)
            return S * _N.pdf(d1) * TTE**0.5

        def get_iv(St, K, TTE):
            m_t_k = np.log(K/St) / TTE**0.5
            coeffs = [0.380390, -0.002262, 0.088586]
            iv = np.poly1d(coeffs)(m_t_k)
            return iv
        
        iv = get_iv(S, K, TTE)
        bs_call_value, delta = bs_call(S, K, TTE, iv)
        vega = bs_vega(S, K, TTE, iv)
        return bs_call_value, delta, vega
    

    def calculate_ema(self, td_key, window, value):
        old_mean = self.last_traderData.get(td_key, value)
        alpha = 2/(window+1)
        new_mean = alpha * value + (1 - alpha) * old_mean
        self.new_trader_data[td_key] = new_mean

        return new_mean



    def calculate_indicators(self):

        indicators = {
            'ema_u_dev': None,
            'ema_o_dev': None,
            'mean_theo_diffs': {},
            'current_theo_diffs': {},
            'switch_means': {},
            'deltas': {},
            'vegas': {},
        }


        if self.underlying.wall_mid is not None:

            new_mean_price = self.calculate_ema('ema_u', underlying_mean_reversion_window, self.underlying.wall_mid)
            indicators['ema_u_dev'] = self.underlying.wall_mid - new_mean_price

            new_mean_price = self.calculate_ema('ema_o', options_mean_reversion_window, self.underlying.wall_mid)
            indicators['ema_o_dev'] = self.underlying.wall_mid - new_mean_price


            for option in self.options:

                k = int(option.name.split('_')[-1])

                if option.wall_mid is None:
                    if option.ask_wall is not None:
                        option.wall_mid = option.ask_wall - 0.5
                        option.bid_wall = option.ask_wall - 1
                        option.best_bid = option.ask_wall - 1
                    elif option.bid_wall is not None:
                        option.wall_mid = option.bid_wall + 0.5
                        option.ask_wall = option.bid_wall + 1
                        option.best_ask = option.bid_wall + 1


                if option.wall_mid is not None:

                    tte = 1 - (DAYS_PER_YEAR - 8 + DAY + self.state.timestamp // 100 / 10_000) / DAYS_PER_YEAR
                    underlying = self.underlying.best_bid * 0.5 + self.underlying.best_ask * 0.5
                    option_theo, option_delta, option_vega = self.get_option_values(underlying, k, tte)
                    option_theo_diff = option.wall_mid - option_theo

                    indicators['current_theo_diffs'][option.name] = option_theo_diff
                    indicators['deltas'][option.name] = option_delta
                    indicators['vegas'][option.name] = option_vega


                    new_mean_diff = self.calculate_ema(f'{option.name}_theo_diff', THEO_NORM_WINDOW, option_theo_diff)
                    indicators['mean_theo_diffs'][option.name] = new_mean_diff


                    new_mean_avg_dev = self.calculate_ema(f'{option.name}_avg_devs', IV_SCALPING_WINDOW, abs(option_theo_diff - new_mean_diff))
                    indicators['switch_means'][option.name] = new_mean_avg_dev

        return indicators
    

    def get_iv_scalping_orders(self, options):

        out = {}

        for option in options:

            if option.name in self.indicators['mean_theo_diffs'] and option.name in self.indicators['current_theo_diffs'] and option.name in self.new_switch_mean:

                if self.new_switch_mean[option.name] >= IV_SCALPING_THR:

                    current_theo_diff = self.indicators['current_theo_diffs'][option.name]
                    mean_theo_diff = self.indicators['mean_theo_diffs'][option.name]

                    low_vega_adj = 0
                    if self.vegas.get(option.name, 0) <= 1:
                        low_vega_adj = LOW_VEGA_THR_ADJ


                    if current_theo_diff - option.wall_mid + option.best_bid - mean_theo_diff >= (THR_OPEN + low_vega_adj) and option.max_allowed_sell_volume > 0:
                        option.ask(option.best_bid, option.max_allowed_sell_volume)

                    if current_theo_diff - option.wall_mid + option.best_bid - mean_theo_diff >= THR_CLOSE and option.initial_position > 0:
                        option.ask(option.best_bid, option.initial_position)

                    elif current_theo_diff - option.wall_mid + option.best_ask - mean_theo_diff <= -(THR_OPEN + low_vega_adj) and option.max_allowed_buy_volume > 0:
                        option.bid(option.best_ask, option.max_allowed_buy_volume)
                        
                    if current_theo_diff - option.wall_mid + option.best_ask - mean_theo_diff <= -THR_CLOSE and option.initial_position < 0:
                        option.bid(option.best_ask, -option.initial_position)

                else:

                    if option.initial_position > 0:
                        option.ask(option.best_bid, option.initial_position)
                    elif option.initial_position < 0:
                        option.bid(option.best_ask, -option.initial_position)


            out[option.name] = option.orders

        return out
    
    def get_mr_orders(self, options):

        out = {}

        for option in options:

            if option.name in self.indicators['current_theo_diffs'] and option.name in self.indicators['mean_theo_diffs'] and self.indicators.get('ema_o_dev') is not None:

                current_deviation = self.indicators['ema_o_dev']

                iv_deviation = self.indicators['current_theo_diffs'][option.name] - self.indicators['mean_theo_diffs'][option.name]
                current_deviation += iv_deviation

                if current_deviation > options_mean_reversion_thr and option.max_allowed_sell_volume > 0:
                    option.ask(option.best_bid, option.max_allowed_sell_volume)

                elif current_deviation < -options_mean_reversion_thr and option.max_allowed_buy_volume > 0:
                    option.bid(option.best_ask, option.max_allowed_buy_volume)

                out[option.name] = option.orders

        return out



    def get_option_orders(self):

        if self.state.timestamp / 100 < min([THEO_NORM_WINDOW, underlying_mean_reversion_window, options_mean_reversion_window]): return {}

        iv_scalping_options = self.options
        mr_options = self.options


        out = {
            **self.get_iv_scalping_orders(iv_scalping_options),
            **self.get_mr_orders(mr_options)
        }

        return out
    
    
    def get_underlying_orders(self):

        if self.state.timestamp / 100 < underlying_mean_reversion_window: return {}

        if self.indicators.get('ema_u_dev') is not None:

            current_deviation = self.indicators['ema_o_dev']

            if current_deviation > underlying_mean_reversion_thr and self.underlying.max_allowed_sell_volume > 0:
                self.underlying.ask(self.underlying.bid_wall + 1, self.underlying.max_allowed_sell_volume)

            elif current_deviation < -underlying_mean_reversion_thr and self.underlying.max_allowed_buy_volume > 0:
                self.underlying.bid(self.underlying.ask_wall - 1, self.underlying.max_allowed_buy_volume)


        return {self.underlying.name: self.underlying.orders}


    def get_orders(self):

        orders = {
            **self.get_option_orders(), # order important, first option, then hedge
            #**self.get_underlying_orders()
        }

        return orders

class Trader:
    def run(self,state:TradingState):
        result = {} # All orders to send
        new_trader_data = {}
        all_signals = {} # Dictionary to store signals for all products, to be passed to the logger at the end of the tick
        conversions = 0 # Not used for Prosperity 4, but still required as a return
        
        product_traders = {
            OPTION_UNDERLYING_SYMBOL: OptionTrader,
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