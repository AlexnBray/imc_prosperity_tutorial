import json
import math
import struct
import base64
from collections import deque
from functools import wraps
from typing import Any, List, Dict
from dataclasses import dataclass
from datamodel import Listing, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

@dataclass
class ProductBaseConfig:
    symbol: str = ""
    short_code: str = ""
    pos_limit: int = 10
    var_window: int = 50

@dataclass
class MeanReversionConfig(ProductBaseConfig):
    z_buy: float = 2.5
    z_sell: float = 2.5
    mean : float = 0.0

@dataclass
class PairConfig:
    pair1 : MeanReversionConfig
    pair2 : MeanReversionConfig
    spread_window : int = 120
    
@dataclass
class LinearMeanReversionConfig(MeanReversionConfig):
    slope: float = 0.0
    vert_translate: float = 0.0

@dataclass
class HighVolMarketMakerConfig(ProductBaseConfig):
    k = 1.5
    gamma = 0.25

CONFIGS_TEMPLATE = {

    "ROBOT_VACUUMING": HighVolMarketMakerConfig(
        symbol="ROBOT_VACUUMING", short_code="RBVA", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
    "ROBOT_MOPPING": HighVolMarketMakerConfig(
        symbol="ROBOT_MOPPING", short_code="RBMO", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
    "ROBOT_DISHES": HighVolMarketMakerConfig(
        symbol="ROBOT_DISHES", short_code="RBDI", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
    "ROBOT_LAUNDRY": HighVolMarketMakerConfig(
        symbol="ROBOT_LAUNDRY", short_code="RBLA", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
    "ROBOT_IRONING": HighVolMarketMakerConfig(
        symbol="ROBOT_IRONING", short_code="RBIR", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
}

CONFIGS = {

    **CONFIGS_TEMPLATE
}

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
    def __init__(self, name, state, new_trader_data, cfg = None):

        self.orders = []

        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.last_traderData = self.get_last_traderData()

        self.cfg = cfg if cfg is not None else CONFIGS.get(self.name)

        self.position_limit = self.cfg.pos_limit

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

    def _init_variance_state(self, prefix: str, window: int):
        self.window = window
        self._hk = f"{prefix}h"
        self._sxk = f"{prefix}sx"
        self._s2k = f"{prefix}s2"

        raw = self.last_traderData.get(self._hk, "")
        self._buf = deque(self._unpack_buf(raw) if raw else [], maxlen=self.window)
        self._sum_x = float(self.last_traderData.get(self._sxk, 0.0))
        self._sum_x2 = float(self.last_traderData.get(self._s2k, 0.0))

    def _calc_var(self, value: float) -> float:
        buf = self._buf

        if len(buf) == self.window:
            evicted = buf[0]
            self._sum_x -= evicted
            self._sum_x2 -= evicted * evicted

        buf.append(value)
        self._sum_x += value
        self._sum_x2 += value * value

        n = len(buf)

        self.new_trader_data[self._hk] = self._pack_buf(buf)
        self.new_trader_data[self._sxk] = self._sum_x
        self.new_trader_data[self._s2k] = self._sum_x2

        if n < 2:
            return 1e-8

        var = (self._sum_x2 - (self._sum_x * self._sum_x) / n) / (n - 1)
        self.var = max(var, 1e-8)
        return self.var


class HighVolMarketMaker(TraderBase):
    def __init__(self,name, state, new_trader_data, cfg = None):
        super().__init__(name, state, new_trader_data, cfg)
        self.gamma = self.cfg.gamma
        self.k = self.cfg.k
        self._init_variance_state(self.cfg.short_code, self.cfg.var_window)


    def get_orders(self):
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}
        if self.gamma <= 0 or self.k <= 0:
            return {self.name: self.orders}

        s = self.wall_mid
        
        prev_mid = float(self.last_traderData.get(f"{self.cfg.short_code}prev", s))
        self.new_trader_data[f"{self.cfg.short_code}prev"] = s

        price_change = s - prev_mid
        self.var = self._calc_var(price_change)


        r = s - (self.expected_position * self.gamma * self.var)
        delta = self.gamma * self.var + (2 / self.gamma) * math.log(1 + self.gamma / self.k)

        # Penny toward touch without crossing by default.
        bid_price = math.floor(r-delta/ 2)
        ask_price = math.ceil(r + delta / 2)

        if bid_price >= ask_price:
            return {self.name:self.orders}
        
        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}





class BasketTrader():
    pass
            
TRADER_BY_CONFIG_TYPE = {
    HighVolMarketMakerConfig: HighVolMarketMaker,

}

class Trader:
    def run(self, state: TradingState):
        result = {}
        new_trader_data = {}
        conversions = 0

        for symbol, cfg in CONFIGS.items():

            trader_cls = TRADER_BY_CONFIG_TYPE.get(type(cfg))
            if trader_cls is None or symbol not in state.order_depths:
                continue

            try:
                trader = trader_cls(symbol, state, new_trader_data)
                result.update(trader.get_orders())

            except Exception:
                print(f"ERROR in trader for {symbol}")

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ""
        return result, conversions, final_trader_dataimport json
import math
import struct
import base64
from collections import deque
from functools import wraps
from typing import Any, List, Dict
from dataclasses import dataclass
from datamodel import Listing, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

@dataclass
class ProductBaseConfig:
    symbol: str = ""
    short_code: str = ""
    pos_limit: int = 10
    var_window: int = 50

@dataclass
class MeanReversionConfig(ProductBaseConfig):
    z_buy: float = 2.5
    z_sell: float = 2.5
    mean : float = 0.0

@dataclass
class PairConfig:
    pair1 : MeanReversionConfig
    pair2 : MeanReversionConfig
    spread_window : int = 120
    
@dataclass
class LinearMeanReversionConfig(MeanReversionConfig):
    slope: float = 0.0
    vert_translate: float = 0.0

@dataclass
class HighVolMarketMakerConfig(ProductBaseConfig):
    k = 1.5
    gamma = 0.25

CONFIGS_TEMPLATE = {

    "ROBOT_VACUUMING": HighVolMarketMakerConfig(
        symbol="ROBOT_VACUUMING", short_code="RBVA", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
    "ROBOT_MOPPING": HighVolMarketMakerConfig(
        symbol="ROBOT_MOPPING", short_code="RBMO", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
    "ROBOT_DISHES": HighVolMarketMakerConfig(
        symbol="ROBOT_DISHES", short_code="RBDI", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
    "ROBOT_LAUNDRY": HighVolMarketMakerConfig(
        symbol="ROBOT_LAUNDRY", short_code="RBLA", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
    "ROBOT_IRONING": HighVolMarketMakerConfig(
        symbol="ROBOT_IRONING", short_code="RBIR", pos_limit=10, var_window=100, k=1.5, gamma=0.25
    ),
}

CONFIGS = {

    **CONFIGS_TEMPLATE
}

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
    def __init__(self, name, state, new_trader_data, cfg = None):

        self.orders = []

        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.last_traderData = self.get_last_traderData()

        self.cfg = cfg if cfg is not None else CONFIGS.get(self.name)

        self.position_limit = self.cfg.pos_limit

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

    def _init_variance_state(self, prefix: str, window: int):
        self.window = window
        self._hk = f"{prefix}h"
        self._sxk = f"{prefix}sx"
        self._s2k = f"{prefix}s2"

        raw = self.last_traderData.get(self._hk, "")
        self._buf = deque(self._unpack_buf(raw) if raw else [], maxlen=self.window)
        self._sum_x = float(self.last_traderData.get(self._sxk, 0.0))
        self._sum_x2 = float(self.last_traderData.get(self._s2k, 0.0))

    def _calc_var(self, value: float) -> float:
        buf = self._buf

        if len(buf) == self.window:
            evicted = buf[0]
            self._sum_x -= evicted
            self._sum_x2 -= evicted * evicted

        buf.append(value)
        self._sum_x += value
        self._sum_x2 += value * value

        n = len(buf)

        self.new_trader_data[self._hk] = self._pack_buf(buf)
        self.new_trader_data[self._sxk] = self._sum_x
        self.new_trader_data[self._s2k] = self._sum_x2

        if n < 2:
            return 1e-8

        var = (self._sum_x2 - (self._sum_x * self._sum_x) / n) / (n - 1)
        self.var = max(var, 1e-8)
        return self.var


class HighVolMarketMaker(TraderBase):
    def __init__(self,name, state, new_trader_data, cfg = None):
        super().__init__(name, state, new_trader_data, cfg)
        self.gamma = self.cfg.gamma
        self.k = self.cfg.k
        self._init_variance_state(self.cfg.short_code, self.cfg.var_window)


    def get_orders(self):
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}
        if self.gamma <= 0 or self.k <= 0:
            return {self.name: self.orders}

        s = self.wall_mid
        
        prev_mid = float(self.last_traderData.get(f"{self.cfg.short_code}prev", s))
        self.new_trader_data[f"{self.cfg.short_code}prev"] = s

        price_change = s - prev_mid
        self.var = self._calc_var(price_change)


        r = s - (self.expected_position * self.gamma * self.var)
        delta = self.gamma * self.var + (2 / self.gamma) * math.log(1 + self.gamma / self.k)

        # Penny toward touch without crossing by default.
        bid_price = math.floor(r-delta/ 2)
        ask_price = math.ceil(r + delta / 2)

        if bid_price >= ask_price:
            return {self.name:self.orders}
        
        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}





class BasketTrader():
    pass
            
TRADER_BY_CONFIG_TYPE = {
    HighVolMarketMakerConfig: HighVolMarketMaker,

}

class Trader:
    def run(self, state: TradingState):
        result = {}
        new_trader_data = {}
        conversions = 0

        for symbol, cfg in CONFIGS.items():

            trader_cls = TRADER_BY_CONFIG_TYPE.get(type(cfg))
            if trader_cls is None or symbol not in state.order_depths:
                continue

            try:
                trader = trader_cls(symbol, state, new_trader_data)
                result.update(trader.get_orders())

            except Exception:
                print(f"ERROR in trader for {symbol}")

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ""
        return result, conversions, final_trader_data