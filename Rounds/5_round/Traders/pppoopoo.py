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
    mean: float = 0.0

@dataclass
class MarketMarkingConfig(ProductBaseConfig):
    k: float = 1.5
    gamma: float = 0.25

@dataclass
class PairConfig:
    pair1: MeanReversionConfig = None
    pair2: MeanReversionConfig = None
    spread_window: int = 120

@dataclass
class LinearMeanReversionConfig(MeanReversionConfig):
    slope: float = 0.0
    vert_translate: float = 0.0

# ── All product configs in one dict ──────────────────────────────────────────
# CONFIGS_TEMPLATE is the full set. CONFIGS overrides specific products with
# specialised config types (e.g. LinearMeanReversionConfig for snackpacks).
# Trader.run() only iterates CONFIGS so both dicts are merged here.

CONFIGS_TEMPLATE = {
    "PEBBLES_XS": MeanReversionConfig(
        symbol="PEBBLES_XS", short_code="PBXS", pos_limit=10, var_window=120, z_buy=2.2, z_sell=2.2, mean=0.0
    ),
    "PEBBLES_S": MeanReversionConfig(
        symbol="PEBBLES_S", short_code="PBS", pos_limit=10, var_window=120, z_buy=2.2, z_sell=2.2, mean=0.0
    ),
    "PEBBLES_M": MeanReversionConfig(
        symbol="PEBBLES_M", short_code="PBM", pos_limit=10, var_window=120, z_buy=2.2, z_sell=2.2, mean=0.0
    ),
    "PEBBLES_L": MeanReversionConfig(
        symbol="PEBBLES_L", short_code="PBL", pos_limit=10, var_window=120, z_buy=2.2, z_sell=2.2, mean=0.0
    ),
    "PEBBLES_XL": MeanReversionConfig(
        symbol="PEBBLES_XL", short_code="PBXL", pos_limit=10, var_window=120, z_buy=2.2, z_sell=2.2, mean=0.0
    ),
    "SNACKPACK_PISTACHIO": MeanReversionConfig(
        symbol="SNACKPACK_PISTACHIO", short_code="SPPI", pos_limit=10, var_window=120, z_buy=2.2, z_sell=2.2, mean=0.0
    ),
    "SNACKPACK_STRAWBERRY": MeanReversionConfig(
        symbol="SNACKPACK_STRAWBERRY", short_code="SPST", pos_limit=10, var_window=120, z_buy=2.2, z_sell=2.2, mean=0.0
    ),
    "SNACKPACK_RASPBERRY": MeanReversionConfig(
        symbol="SNACKPACK_RASPBERRY", short_code="SPRA", pos_limit=10, var_window=120, z_buy=2.2, z_sell=2.2, mean=0.0
    ),
    "ROBOT_VACUUMING": MeanReversionConfig(
        symbol="ROBOT_VACUUMING", short_code="RBVA", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "ROBOT_MOPPING": MeanReversionConfig(
        symbol="ROBOT_MOPPING", short_code="RBMO", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "ROBOT_DISHES": MeanReversionConfig(
        symbol="ROBOT_DISHES", short_code="RBDI", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "ROBOT_LAUNDRY": MeanReversionConfig(
        symbol="ROBOT_LAUNDRY", short_code="RBLA", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "ROBOT_IRONING": MeanReversionConfig(
        symbol="ROBOT_IRONING", short_code="RBIR", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "OXYGEN_SHAKE_MORNING_BREATH": MeanReversionConfig(
        symbol="OXYGEN_SHAKE_MORNING_BREATH", short_code="OXMB", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "OXYGEN_SHAKE_EVENING_BREATH": MeanReversionConfig(
        symbol="OXYGEN_SHAKE_EVENING_BREATH", short_code="OXEB", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "OXYGEN_SHAKE_MINT": MeanReversionConfig(
        symbol="OXYGEN_SHAKE_MINT", short_code="OXMI", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "OXYGEN_SHAKE_CHOCOLATE": MeanReversionConfig(
        symbol="OXYGEN_SHAKE_CHOCOLATE", short_code="OXCH", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "OXYGEN_SHAKE_GARLIC": MeanReversionConfig(
        symbol="OXYGEN_SHAKE_GARLIC", short_code="OXGA", pos_limit=10, var_window=100, z_buy=2.6, z_sell=2.6, mean=0.0
    ),
    "MICROCHIP_OVAL": MarketMarkingConfig(
        symbol="MICROCHIP_OVAL", short_code="MCOV", pos_limit=10, var_window=80, k=0.0001, gamma=0.005
    ),
    "MICROCHIP_SQUARE": MarketMarkingConfig(
        symbol="MICROCHIP_SQUARE", short_code="MCSQ", pos_limit=10, var_window=80, k=1.5, gamma=0.25
    ),
    "MICROCHIP_RECTANGLE": MarketMarkingConfig(
        symbol="MICROCHIP_RECTANGLE", short_code="MCRE", pos_limit=10, var_window=80, k=1.5, gamma=0.25
    ),
    "MICROCHIP_TRIANGLE": MarketMarkingConfig(
        symbol="MICROCHIP_TRIANGLE", short_code="MCTR", pos_limit=10, var_window=80, k=1.5, gamma=0.25
    ),
    "MICROCHIP_CIRCLE": MarketMarkingConfig(
        symbol="MICROCHIP_CIRCLE", short_code="MCCI", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "GALAXY_SOUNDS_DARK_MATTER": MarketMarkingConfig(
        symbol="GALAXY_SOUNDS_DARK_MATTER", short_code="GSDM", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "GALAXY_SOUNDS_BLACK_HOLES": MarketMarkingConfig(
        symbol="GALAXY_SOUNDS_BLACK_HOLES", short_code="GSBH", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "GALAXY_SOUNDS_PLANETARY_RINGS": MarketMarkingConfig(
        symbol="GALAXY_SOUNDS_PLANETARY_RINGS", short_code="GSPR", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "GALAXY_SOUNDS_SOLAR_WINDS": MarketMarkingConfig(
        symbol="GALAXY_SOUNDS_SOLAR_WINDS", short_code="GSSW", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "GALAXY_SOUNDS_SOLAR_FLAMES": MarketMarkingConfig(
        symbol="GALAXY_SOUNDS_SOLAR_FLAMES", short_code="GSSF", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "SLEEP_POD_SUEDE": MarketMarkingConfig(
        symbol="SLEEP_POD_SUEDE", short_code="SPSU", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "SLEEP_POD_LAMB_WOOL": MarketMarkingConfig(
        symbol="SLEEP_POD_LAMB_WOOL", short_code="SPLW", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "SLEEP_POD_POLYESTER": MarketMarkingConfig(
        symbol="SLEEP_POD_POLYESTER", short_code="SPPO", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "SLEEP_POD_NYLON": MarketMarkingConfig(
        symbol="SLEEP_POD_NYLON", short_code="SPNY", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "SLEEP_POD_COTTON": MarketMarkingConfig(
        symbol="SLEEP_POD_COTTON", short_code="SPCO", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "UV_VISOR_YELLOW": MarketMarkingConfig(
        symbol="UV_VISOR_YELLOW", short_code="UVYE", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "UV_VISOR_AMBER": MarketMarkingConfig(
        symbol="UV_VISOR_AMBER", short_code="UVAM", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "UV_VISOR_ORANGE": MarketMarkingConfig(
        symbol="UV_VISOR_ORANGE", short_code="UVOR", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "UV_VISOR_RED": MarketMarkingConfig(
        symbol="UV_VISOR_RED", short_code="UVRE", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "UV_VISOR_MAGENTA": MarketMarkingConfig(
        symbol="UV_VISOR_MAGENTA", short_code="UVMA", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "TRANSLATOR_SPACE_GRAY": MarketMarkingConfig(
        symbol="TRANSLATOR_SPACE_GRAY", short_code="TRSG", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "TRANSLATOR_ASTRO_BLACK": MarketMarkingConfig(
        symbol="TRANSLATOR_ASTRO_BLACK", short_code="TRAB", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "TRANSLATOR_ECLIPSE_CHARCOAL": MarketMarkingConfig(
        symbol="TRANSLATOR_ECLIPSE_CHARCOAL", short_code="TREC", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "TRANSLATOR_GRAPHITE_MIST": MarketMarkingConfig(
        symbol="TRANSLATOR_GRAPHITE_MIST", short_code="TRGM", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "TRANSLATOR_VOID_BLUE": MarketMarkingConfig(
        symbol="TRANSLATOR_VOID_BLUE", short_code="TRVB", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "PANEL_1X2": MarketMarkingConfig(
        symbol="PANEL_1X2", short_code="PN12", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "PANEL_2X2": MarketMarkingConfig(
        symbol="PANEL_2X2", short_code="PN22", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "PANEL_1X4": MarketMarkingConfig(
        symbol="PANEL_1X4", short_code="PN14", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "PANEL_2X4": MarketMarkingConfig(
        symbol="PANEL_2X4", short_code="PN24", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
    "PANEL_4X4": MarketMarkingConfig(
        symbol="PANEL_4X4", short_code="PN44", pos_limit=10, var_window=80, k=1.0, gamma=0.15
    ),
}

# FIX 4: Merge CONFIGS_TEMPLATE into CONFIGS so all 50 products trade.
# Snackpack overrides use LinearMeanReversionConfig for the linear mean model.
CONFIGS = {
    **CONFIGS_TEMPLATE,
    # Override snackpacks with linear mean reversion
    "SNACKPACK_CHOCOLATE": LinearMeanReversionConfig(
        symbol="SNACKPACK_CHOCOLATE", short_code="SPCH", pos_limit=10, var_window=120,
        z_buy=2.2, z_sell=2.2, slope=0.0, vert_translate=0.0,
    ),
    "SNACKPACK_VANILLA": LinearMeanReversionConfig(
        symbol="SNACKPACK_VANILLA", short_code="SPVA", pos_limit=10, var_window=120,
        z_buy=2.2, z_sell=2.2, slope=0.0, vert_translate=0.0,
    ),
}


def handle_none(default_active=False, default_cache_prev=True, default_fallback=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            active     = kwargs.pop('_active',   default_active)
            cache_prev = kwargs.pop('_cache',    default_cache_prev)
            fallback   = kwargs.pop('_fallback', default_fallback)

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
    def __init__(self, name, state, new_trader_data, cfg=None):

        self.orders = []
        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.last_traderData = self.get_last_traderData()

        self.cfg = cfg if cfg is not None else CONFIGS.get(self.name)

        self.position_limit    = self.cfg.pos_limit
        self.initial_position  = self.state.position.get(self.name, 0)
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
            if len(self.mkt_buy_orders) > 0: best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0: best_ask = min(self.mkt_sell_orders.keys())
        except: pass
        return best_bid, best_ask

    @handle_none()
    def get_walls(self):
        bid_wall = wall_mid = ask_wall = None
        try: bid_wall = max([x for x, _ in self.mkt_buy_orders.items()])
        except: pass
        try: ask_wall = min([x for x, _ in self.mkt_sell_orders.items()])
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
        return (self.position_limit - self.initial_position,
                self.position_limit + self.initial_position)

    @handle_none()
    def get_order_depth(self):
        order_depth = buy_orders = sell_orders = {}
        try: order_depth: OrderDepth = self.state.order_depths[self.name]
        except: pass
        try: buy_orders  = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(),  key=lambda x: x[0], reverse=True)}
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
        if not buf: return ""
        scaled = [max(-32768, min(32767, round(v * 2))) for v in buf]
        return base64.b64encode(struct.pack(f">{len(scaled)}h", *scaled)).decode()

    @staticmethod
    def _unpack_buf(s: str) -> list:
        if not s: return []
        raw = base64.b64decode(s)
        return [v / 2.0 for v in struct.unpack(f">{len(raw) // 2}h", raw)]

    def _init_variance_state(self, prefix: str, window: int):
        self.window  = window
        self._hk     = f"{prefix}h"
        self._sxk    = f"{prefix}sx"
        self._s2k    = f"{prefix}s2"
        raw          = self.last_traderData.get(self._hk, "")
        self._buf    = deque(self._unpack_buf(raw) if raw else [], maxlen=self.window)
        self._sum_x  = float(self.last_traderData.get(self._sxk, 0.0))
        self._sum_x2 = float(self.last_traderData.get(self._s2k, 0.0))

    def _calc_var(self, value: float) -> float:
        buf = self._buf
        if len(buf) == self.window:
            evicted      = buf[0]
            self._sum_x  -= evicted
            self._sum_x2 -= evicted * evicted

        buf.append(value)
        self._sum_x  += value
        self._sum_x2 += value * value
        n = len(buf)

        self.new_trader_data[self._hk]  = self._pack_buf(buf)
        self.new_trader_data[self._sxk] = self._sum_x
        self.new_trader_data[self._s2k] = self._sum_x2

        if n < 2:
            return 1e-8

        var = (self._sum_x2 - (self._sum_x * self._sum_x) / n) / (n - 1)
        self.var = max(var, 1e-8)
        return self.var


class MeanReversionTrader(TraderBase):
    def __init__(self, name, state, new_trader_data, cfg=None):
        # FIX 1: pass cfg through instead of hardcoding None
        super().__init__(name, state, new_trader_data, cfg=cfg)

        self.z_buy_threshold  = self.cfg.z_buy
        self.z_sell_threshold = self.cfg.z_sell
        self.mean = self.cfg.mean
        self._init_variance_state(self.cfg.short_code, self.cfg.var_window)

    def _compute_z(self, value: float, mean: float) -> float:
        dev = value - mean
        var = self._calc_var(dev)
        return dev / math.sqrt(var)

    def get_orders(self):
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        z = self._compute_z(self.wall_mid, self.mean)

        if z > 0 and abs(z) > self.z_sell_threshold:
            self.ask(self.best_bid, self.max_allowed_sell_volume)
        if z < 0 and abs(z) > self.z_buy_threshold:
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        return {self.name: self.orders}

    def debug(self):
        return self.mean


class LinearMeanReversionTrader(MeanReversionTrader):
    def __init__(self, name, state, new_trader_data, cfg=None):
        # FIX 2: pass cfg through instead of hardcoding None
        super().__init__(name, state, new_trader_data, cfg=cfg)

        # Override mean using linear model: mean = slope * timestamp + vert_translate
        time               = self.state.timestamp
        self.mean          = self.cfg.slope * time + self.cfg.vert_translate


class PairsTrader:
    """
    Composes two MeanReversionTraders on a spread signal.
    Does not subclass TraderBase — manages two legs directly.
    """
    def __init__(self, pair_name: str, state: TradingState, new_trader_data: dict):
        self.cfg: PairConfig = CONFIGS.get(pair_name)
        self.pair1 = MeanReversionTrader(
            self.cfg.pair1.symbol, state, new_trader_data, cfg=self.cfg.pair1
        )
        self.pair2 = MeanReversionTrader(
            self.cfg.pair2.symbol, state, new_trader_data, cfg=self.cfg.pair2  # FIX: was pair1
        )
        self.spread_window = self.cfg.spread_window


class ThroupleTrader(TraderBase):
    # FIX 3: removed syntax error ("stat e" → "state")
    def __init__(self, pairs, state, new_trader_data, cfg=None):
        super().__init__(pairs[0] if pairs else "", state, new_trader_data, cfg=cfg)


class SpreadMeanReversionTrader(MeanReversionTrader):
    def __init__(self, name, state, new_trader_data, cfg=None):
        super().__init__(name, state, new_trader_data, cfg=cfg)

    def _calc_mean(self, slope: float, vert_translate: float) -> float:
        x         = self.state.timestamp
        self.mean = slope * x + vert_translate
        return self.mean

    def get_orders(self):
        # FIX: recalculate mean then call super without passing self
        self.mean = self._calc_mean(self.cfg.slope, self.cfg.vert_translate)
        return super().get_orders()


class MarketMarkingTrader(TraderBase):
    def __init__(self, name, state, new_trader_data, cfg=None):
        # FIX 2 (market maker): accept and pass cfg through
        super().__init__(name, state, new_trader_data, cfg=cfg)
        self.gamma = self.cfg.gamma
        self.k     = self.cfg.k
        self._init_variance_state(self.cfg.short_code, self.cfg.var_window)

    def get_orders(self):
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}
        if self.gamma <= 0 or self.k <= 0:
            return {self.name: self.orders}

        # FIX: use price CHANGES not raw price level for variance
        # Raw price (~10000) produces near-zero variance → spread collapses to 0
        prev_mid  = float(self.last_traderData.get(f"{self.cfg.short_code}prev", self.wall_mid))
        self.new_trader_data[f"{self.cfg.short_code}prev"] = self.wall_mid
        price_change = self.wall_mid - prev_mid
        self.var = self._calc_var(price_change)

        s = self.wall_mid
        # Reservation price: skew toward flattening inventory
        r = s - (self.expected_position * self.gamma * self.var)

        # Optimal spread from Avellaneda-Stoikov
        delta = self.gamma * self.var + (2 / self.gamma) * math.log(1 + self.gamma / self.k)

        # Penny toward touch without crossing
        bid_price = max(math.floor(r - delta / 2), self.best_bid + 1)
        ask_price = min(math.ceil(r  + delta / 2), self.best_ask - 1)

        # Ensure quotes don't cross
        if bid_price >= ask_price:
            bid_price = self.best_bid
            ask_price = self.best_ask

        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


class BasketTrader:
    """Placeholder — implement PEBBLES basket arb here."""
    pass


# Maps config type → trader class so Trader.run() dispatches automatically
TRADER_BY_CONFIG_TYPE = {
    MeanReversionConfig:       MeanReversionTrader,
    LinearMeanReversionConfig: LinearMeanReversionTrader,
    MarketMarkingConfig:       MarketMarkingTrader,
}


class Trader:
    def run(self, state: TradingState):
        result          = {}
        new_trader_data = {}
        conversions     = 0

        for symbol, cfg in CONFIGS.items():
            trader_cls = TRADER_BY_CONFIG_TYPE.get(type(cfg))
            if trader_cls is None or symbol not in state.order_depths:
                continue

            try:
                # FIX 3: pass cfg into every trader so the chain never receives None
                trader = trader_cls(symbol, state, new_trader_data, cfg=cfg)
                result.update(trader.get_orders())
            except Exception as e:
                print(f"ERROR in trader for {symbol}: {e}")

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ""

        return result, conversions, final_trader_data