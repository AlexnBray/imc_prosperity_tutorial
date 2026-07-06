import json
import math
import struct
import base64
from collections import deque
from functools import wraps
from typing import Any, List, Dict
from datamodel import Listing, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

DAY = 4

TTE = 8 - int(DAY)


VELVETFRUIT_EXTRACT = "VELVETFRUIT_EXTRACT"
HYDROGEL_PACK = "HYDROGEL_PACK"
VEV_4000 = "VEV_4000"
VEV_4500 = "VEV_4500"
VEV_5000 = "VEV_5000"
VEV_5100 = "VEV_5100"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
VEV_5400 = "VEV_5400"
VEV_5500 = "VEV_5500"
VEV_6000 = "VEV_6000"
VEV_6500 = "VEV_6500"

POS_LIMITS = {
    VELVETFRUIT_EXTRACT: 200,
    HYDROGEL_PACK: 200,
    VEV_4000: 300,
    VEV_4500: 300,
    VEV_5000: 300,
    VEV_5100: 300,
    VEV_5200: 300,
    VEV_5300: 300,
    VEV_5400: 300,
    VEV_5500: 300,
    VEV_6000: 300,
    VEV_6500: 300
    }

ROLLING_WINDOW_VAR = {
    HYDROGEL_PACK: 301,
    VELVETFRUIT_EXTRACT: 280,
    VEV_4000: 175,
    VEV_4500: 226,
    VEV_5000: 316,
    VEV_5100: 326,
    VEV_5200: 283,
    VEV_5300: 221,
    VEV_5400: 220,
    VEV_5500: 131
}

MR_Z_BUY_THRESHOLD = {
    HYDROGEL_PACK: 3.6 ,
    VELVETFRUIT_EXTRACT: 3.25,
    VEV_4000: 4.0,
    VEV_4500: 2.5,
    VEV_5000: 2.5,
    VEV_5100: 2.0,
    VEV_5200: 2.0,
    VEV_5300: 1.5,
    VEV_5400: 3.0,
    VEV_5500: 1.75
}

MR_Z_SELL_THRESHOLD = {
    HYDROGEL_PACK: 2.0 ,
    VELVETFRUIT_EXTRACT: 4.00,
    VEV_4000: 5.0,
    VEV_4500: 4.5 ,
    VEV_5000: 3.0,
    VEV_5100: 4.0,
    VEV_5200: 2.0,
    VEV_5300: 4.5,
    VEV_5400: 3.0,
    VEV_5500: 2.5 
}

GLOBAL_INT = {
    HYDROGEL_PACK: 9991.0,
    VELVETFRUIT_EXTRACT: 5250.0,
    VEV_4000: 1250.0, #d
    VEV_4500: 750.0, #done
    VEV_5000: 253.0, #d
    VEV_5100: 150.0, #d
    VEV_5200: 50.0, #d
    VEV_5300: 1.0, #d
    VEV_5400: 8.0, #d
    VEV_5500: 1.0 #d
}

GLOBAL_SLOPE = {
    HYDROGEL_PACK: 0.0,
    VELVETFRUIT_EXTRACT: 0.0,
    VEV_4000: 0.0 , #d
    VEV_4500: 0.0, # done
    VEV_5000: 0.0, #d
    VEV_5100: 0.006765 , #d
    VEV_5200: 0.020842  , #d
    VEV_5300: 0.022734, #d
    VEV_5400: 0, #d
    VEV_5500: 0 #d
}

GLOBAL_TTE_TRANSLATION = {
    HYDROGEL_PACK: 0,
    VELVETFRUIT_EXTRACT: 0,
    VEV_4000: 0,#d
    VEV_4500: 0, #d
    VEV_5000: 0,#d
    VEV_5100: 1e6,#d
    VEV_5200: 2e6, #d
    VEV_5300: 2.5e6, #d
    VEV_5400: 4e6, #d
    VEV_5500: 4.6e6 #d 
}

# Short key prefixes to minimise JSON payload size
_SHORT_PFX = {
    HYDROGEL_PACK:       "HP",
    VELVETFRUIT_EXTRACT: "VF",
    VEV_4000: "V0",
    VEV_4500: "V1",
    VEV_5000: "V2",
    VEV_5100: "V3",
    VEV_5200: "V4",
    VEV_5300: "V5",
    VEV_5400: "V6",
    VEV_5500: "V7",
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
            active = kwargs.pop('_active', default_active)
            cache_prev = kwargs.pop('_cache', default_cache_prev)
            fallback = kwargs.pop('_fallback', default_fallback)

            if not active:
                return func(self, *args, **kwargs)

            result = func(self, *args, **kwargs)

            def is_none_like(res):
                if res is None: return True
                if isinstance(res, tuple) and all(x is None for x in res): return True
                return False

            cache_key = f"{self.name}_{func.__name__}"

            if is_none_like(result):
                if cache_prev:
                    cached_val = self.last_traderData.get(cache_key)
                    if cached_val is not None:
                        return tuple(cached_val) if isinstance(cached_val, list) else cached_val
                return fallback
            else:
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
        self.initial_position = self.state.position.get(self.name, 0)

        self.expected_position = self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume()
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

    def get_last_traderData(self):
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except: pass
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

class SpikeHedger(TraderBase):
    def __init__(self, name, state, new_trader_data):
        super().__init__(name, state, new_trader_data)

    def get_orders(self):
        self.bid(0, self.max_allowed_buy_volume)
    
        return {self.name: self.orders}
    
class MRTrader(TraderBase):
    def __init__(self, name, state, new_trader_data):
        super().__init__(name, state, new_trader_data)

        self.window = ROLLING_WINDOW_VAR.get(name, 100)
        self.z_buy_threshold  = MR_Z_BUY_THRESHOLD.get(name, 2.5)
        self.z_sell_threshold = MR_Z_SELL_THRESHOLD.get(name, 2.5)
        self.intercept = GLOBAL_INT.get(name, 10)
        self.slope = GLOBAL_SLOPE.get(name, 10)
        self.tte_translation = GLOBAL_TTE_TRANSLATION.get(name, 10)

        # Short key prefix — keeps JSON payload small
        p = _SHORT_PFX.get(name, name[:2])
        self._hk  = f"{p}h"    # eviction buffer (base64-packed deviations)
        self._sxk = f"{p}sx"   # running sum of deviations
        self._s2k = f"{p}s2"   # running sum of squared deviations

        # Restore variance state from previous tick
        raw = self.last_traderData.get(self._hk, "")
        self._buf    = deque(self._unpack_buf(raw) if raw else [], maxlen=self.window)
        self._sum_x  = float(self.last_traderData.get(self._sxk, 0.0))
        self._sum_x2 = float(self.last_traderData.get(self._s2k, 0.0))

    # ------------------------------------------------------------------
    # Buffer packing helpers
    # Deviations from global mean are stored as int16 scaled by ×2,
    # preserving 0.5-precision (wall_mid is always integer or x.5).
    # Range: ±16383 — safe for any realistic deviation from global mean.
    # 2 bytes per element vs ~8 chars per float in JSON → ~4× smaller.
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_buf(buf) -> str:
        if not buf:
            return ""
        scaled = [max(-32768, min(32767, round(v * 2))) for v in buf]
        return base64.b64encode(struct.pack(f">{len(scaled)}h", *scaled)).decode()

    @staticmethod
    def _unpack_buf(s: str) -> list:
        if not s:
            return []
        raw = base64.b64decode(s)
        return [v / 2.0 for v in struct.unpack(f">{len(raw) // 2}h", raw)]

    # ------------------------------------------------------------------
    # O(1) online sliding-window variance
    #
    # Mathematically identical to np.var(window, ddof=1).
    # `value` is already a deviation (caller passes wall_mid - mean),
    # so sum_x and sum_x2 stay small — no catastrophic cancellation.
    #
    # Welford identity:
    #   var = (sum_x2 - sum_x² / n) / (n - 1)
    # ------------------------------------------------------------------

    def _calc_var(self, value: float) -> float:
        buf = self._buf

        if len(buf) == self.window:
            # Steady-state: evict oldest before deque auto-drops it
            evicted     = buf[0]
            self._sum_x  -= evicted
            self._sum_x2 -= evicted * evicted

        buf.append(value)           # deque(maxlen=window) auto-evicts oldest
        self._sum_x  += value
        self._sum_x2 += value * value

        n = len(buf)

        # Persist compactly for next tick
        self.new_trader_data[self._hk]  = self._pack_buf(buf)
        self.new_trader_data[self._sxk] = self._sum_x
        self.new_trader_data[self._s2k] = self._sum_x2

        if n < 2:
            return 1e-8

        var = (self._sum_x2 - (self._sum_x * self._sum_x) / n) / (n - 1)
        return max(var, 1e-8)
    
    def _calc_mean(self, intercept, slope):
        x= self.state.timestamp
        if slope != 0:
            inside = ((TTE * 1e6)- self.tte_translation) - x
            inside = max(inside,0)
            self.mean = intercept + slope * math.sqrt(inside)
        else:
            self.mean = GLOBAL_INT[self.name]
        return self.mean

    def _compute_z(self, value: float, mean: float) -> float:
        # Pass deviation into _calc_var so the running sums stay near zero
        dev = value - mean
        var = self._calc_var(dev)
        return dev / math.sqrt(var)

    def get_orders(self):
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        mean = self._calc_mean(self.intercept, self.slope)   # <-- compute number
        z = self._compute_z(self.wall_mid, mean)

        if z > 0 and abs(z) > self.z_sell_threshold:
            self.ask(self.best_bid, self.max_allowed_sell_volume)
        if z < 0 and abs(z) > self.z_buy_threshold:
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        return {self.name: self.orders}

    def debug(self):
        return self.mean

class MicroSpikeTrader(TraderBase):
    def __init__(self,name,state,new_trader_data):
        super().__init__(name,state,new_trader_data)
        self.mean = GLOBAL_INT.get(self.name,0)

    def get_orders(self):

        if self.best_bid > 2:
            self.ask(self.best_bid, self.max_allowed_sell_volume)

        if self.best_ask <= 2:
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        return {self.name: self.orders}

class ShortTrader(TraderBase):
    def __init__(self,name,state,new_trader_data):
        super().__init__(name,state,new_trader_data)

    def get_orders(self):
        self.ask(self.best_bid,self.max_allowed_sell_volume)

        return {self.name: self.orders}


class Trader:
    def run(self, state: TradingState):
        result         = {}
        new_trader_data = {}
        all_signals    = {}
        conversions    = 0

        product_traders = {
            HYDROGEL_PACK: MRTrader,
            VELVETFRUIT_EXTRACT: MRTrader,
            VEV_4000: MRTrader,
            VEV_4500: MRTrader,
            VEV_5000: MRTrader,
            VEV_5100: MRTrader,
            VEV_5200: MRTrader,
            VEV_5300: MRTrader,
            VEV_5400: MRTrader,
            VEV_5500: MicroSpikeTrader,
            VEV_6000: SpikeHedger,
            VEV_6500: SpikeHedger,
        }

        for symbol, product_trader in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trades = product_trader(symbol, state, new_trader_data)
                    orders = trades.get_orders()
                    result.update(orders)
                    all_signals[symbol] = trades.debug()
                except:
                    logger.print(f"ERROR in trader for {symbol}")

        try: final_trader_data = json.dumps(new_trader_data)
        except: final_trader_data = ''

        logger.flush(state, result, conversions, final_trader_data, all_signals)

        return result, conversions, final_trader_data