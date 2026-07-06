import json
import math
import struct
import base64
from collections import deque
from functools import wraps
from typing import Any, List, Dict
from dataclasses import dataclass, field
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
class StepMeanReversionConfig(MeanReversionConfig):
    """Chunked running-mean anchors: long + short windows, snap to short when divergence dwarfs residual σ."""
    step_long_ticks: int = 2000
    step_short_ticks: int = 50
    step_k_sd: float = 10.0
    step_sigma_floor: float = 1e-3

@dataclass
class EWMAMeanReversionConfig(MeanReversionConfig):
    """Mid vs EWMA(mid) MR; variance stream on ``mid − fair``."""

    alpha: float = 0.08


# ── All product configs in one dict ──────────────────────────────────────────
# Winners from ``mr_universe_comparison_results.json`` (STEP vs EWMA by Optuna score).
# Individual PEBBLES_* universe legs omitted here (would conflict with any basket arb).

CONFIGS = {
    "GALAXY_SOUNDS_BLACK_HOLES": StepMeanReversionConfig(
        symbol="GALAXY_SOUNDS_BLACK_HOLES",
        short_code="GSBH",
        pos_limit=10,
        var_window=614,
        z_buy=float(3.1),
        z_sell=float(4.1),
        mean=0.0,
        step_long_ticks=578,
        step_short_ticks=59,
        step_k_sd=float(10.0),
        step_sigma_floor=float(0.000879121405786216),
    ),
    "GALAXY_SOUNDS_DARK_MATTER": StepMeanReversionConfig(
        symbol="GALAXY_SOUNDS_DARK_MATTER",
        short_code="GSDM",
        pos_limit=10,
        var_window=377,
        z_buy=float(3.1),
        z_sell=float(1.3),
        mean=0.0,
        step_long_ticks=2128,
        step_short_ticks=122,
        step_k_sd=float(9.0),
        step_sigma_floor=float(0.00012691838167870002),
    ),
    "GALAXY_SOUNDS_PLANETARY_RINGS": EWMAMeanReversionConfig(
        symbol="GALAXY_SOUNDS_PLANETARY_RINGS",
        short_code="GSPR",
        pos_limit=10,
        var_window=172,
        z_buy=float(3.3000000000000003),
        z_sell=float(4.0),
        mean=0.0,
        alpha=float(0.10331020612132799),
    ),
    "GALAXY_SOUNDS_SOLAR_FLAMES": StepMeanReversionConfig(
        symbol="GALAXY_SOUNDS_SOLAR_FLAMES",
        short_code="GSSF",
        pos_limit=10,
        var_window=326,
        z_buy=float(2.3),
        z_sell=float(2.6),
        mean=0.0,
        step_long_ticks=1872,
        step_short_ticks=65,
        step_k_sd=float(4.0),
        step_sigma_floor=float(0.03157734084987153),
    ),
    "GALAXY_SOUNDS_SOLAR_WINDS": EWMAMeanReversionConfig(
        symbol="GALAXY_SOUNDS_SOLAR_WINDS",
        short_code="GSSW",
        pos_limit=10,
        var_window=213,
        z_buy=float(3.5),
        z_sell=float(3.8000000000000003),
        mean=0.0,
        alpha=float(0.010837107636863404),
    ),
    "MICROCHIP_CIRCLE": StepMeanReversionConfig(
        symbol="MICROCHIP_CIRCLE",
        short_code="MCCR",
        pos_limit=10,
        var_window=164,
        z_buy=float(1.6),
        z_sell=float(1.3),
        mean=0.0,
        step_long_ticks=1163,
        step_short_ticks=26,
        step_k_sd=float(2.5),
        step_sigma_floor=float(0.0011087600850205842),
    ),
    "MICROCHIP_OVAL": StepMeanReversionConfig(
        symbol="MICROCHIP_OVAL",
        short_code="MCOV",
        pos_limit=10,
        var_window=716,
        z_buy=float(3.2),
        z_sell=float(1.3),
        mean=0.0,
        step_long_ticks=879,
        step_short_ticks=116,
        step_k_sd=float(6.0),
        step_sigma_floor=float(0.00016907341255436494),
    ),
    "MICROCHIP_RECTANGLE": StepMeanReversionConfig(
        symbol="MICROCHIP_RECTANGLE",
        short_code="MCRT",
        pos_limit=10,
        var_window=669,
        z_buy=float(1.4),
        z_sell=float(2.2),
        mean=0.0,
        step_long_ticks=1746,
        step_short_ticks=156,
        step_k_sd=float(6.5),
        step_sigma_floor=float(0.003124767214055634),
    ),
    "MICROCHIP_SQUARE": StepMeanReversionConfig(
        symbol="MICROCHIP_SQUARE",
        short_code="MCSQ",
        pos_limit=10,
        var_window=370,
        z_buy=float(3.9000000000000004),
        z_sell=float(3.9000000000000004),
        mean=0.0,
        step_long_ticks=1085,
        step_short_ticks=87,
        step_k_sd=float(8.5),
        step_sigma_floor=float(0.0027231411304731077),
    ),
    "MICROCHIP_TRIANGLE": EWMAMeanReversionConfig(
        symbol="MICROCHIP_TRIANGLE",
        short_code="MCTR",
        pos_limit=10,
        var_window=744,
        z_buy=float(2.5),
        z_sell=float(2.4000000000000004),
        mean=0.0,
        alpha=float(0.010302662342055636),
    ),
    "OXYGEN_SHAKE_CHOCOLATE": EWMAMeanReversionConfig(
        symbol="OXYGEN_SHAKE_CHOCOLATE",
        short_code="OXCH",
        pos_limit=10,
        var_window=75,
        z_buy=float(2.4000000000000004),
        z_sell=float(4.0),
        mean=0.0,
        alpha=float(0.019411946720480168),
    ),
    "OXYGEN_SHAKE_EVENING_BREATH": StepMeanReversionConfig(
        symbol="OXYGEN_SHAKE_EVENING_BREATH",
        short_code="OXEB",
        pos_limit=10,
        var_window=166,
        z_buy=float(1.8),
        z_sell=float(1.6),
        mean=0.0,
        step_long_ticks=1089,
        step_short_ticks=174,
        step_k_sd=float(3.75),
        step_sigma_floor=float(0.0024422086388236845),
    ),
    "OXYGEN_SHAKE_GARLIC": StepMeanReversionConfig(
        symbol="OXYGEN_SHAKE_GARLIC",
        short_code="OXGA",
        pos_limit=10,
        var_window=612,
        z_buy=float(2.7),
        z_sell=float(3.8000000000000003),
        mean=0.0,
        step_long_ticks=445,
        step_short_ticks=95,
        step_k_sd=float(11.75),
        step_sigma_floor=float(0.008929576621018318),
    ),
    "OXYGEN_SHAKE_MINT": EWMAMeanReversionConfig(
        symbol="OXYGEN_SHAKE_MINT",
        short_code="OXMI",
        pos_limit=10,
        var_window=711,
        z_buy=float(3.1),
        z_sell=float(3.1),
        mean=0.0,
        alpha=float(0.008148804815626128),
    ),
    "OXYGEN_SHAKE_MORNING_BREATH": StepMeanReversionConfig(
        symbol="OXYGEN_SHAKE_MORNING_BREATH",
        short_code="OXMB",
        pos_limit=10,
        var_window=729,
        z_buy=float(4.5),
        z_sell=float(3.8000000000000003),
        mean=0.0,
        step_long_ticks=488,
        step_short_ticks=78,
        step_k_sd=float(8.5),
        step_sigma_floor=float(0.005945871773934723),
    ),
    "PEBBLES_L": StepMeanReversionConfig(
        symbol="PEBBLES_L",
        short_code="PBBL",
        pos_limit=10,
        var_window=782,
        z_buy=float(3.2),
        z_sell=float(2.3),
        mean=0.0,
        step_long_ticks=582,
        step_short_ticks=97,
        step_k_sd=float(3.75),
        step_sigma_floor=float(0.0010888134478069542),
    ),
    "PEBBLES_M": EWMAMeanReversionConfig(
        symbol="PEBBLES_M",
        short_code="PBBM",
        pos_limit=10,
        var_window=336,
        z_buy=float(2.5),
        z_sell=float(3.0),
        mean=0.0,
        alpha=float(0.020228552524542583),
    ),
    "PEBBLES_S": StepMeanReversionConfig(
        symbol="PEBBLES_S",
        short_code="PBBS",
        pos_limit=10,
        var_window=336,
        z_buy=float(2.2),
        z_sell=float(2.3),
        mean=0.0,
        step_long_ticks=1985,
        step_short_ticks=24,
        step_k_sd=float(10.5),
        step_sigma_floor=float(0.005416854437929823),
    ),
    "PEBBLES_XL": EWMAMeanReversionConfig(
        symbol="PEBBLES_XL",
        short_code="PEXL",
        pos_limit=10,
        var_window=234,
        z_buy=float(1.9),
        z_sell=float(2.4000000000000004),
        mean=0.0,
        alpha=float(0.07147431760746997),
    ),
    "PEBBLES_XS": EWMAMeanReversionConfig(
        symbol="PEBBLES_XS",
        short_code="PEXS",
        pos_limit=10,
        var_window=507,
        z_buy=float(3.3000000000000003),
        z_sell=float(1.6),
        mean=0.0,
        alpha=float(0.00850171445630847),
    ),
    "PANEL_1X2": StepMeanReversionConfig(
        symbol="PANEL_1X2",
        short_code="PN12",
        pos_limit=10,
        var_window=445,
        z_buy=float(2.6),
        z_sell=float(3.9000000000000004),
        mean=0.0,
        step_long_ticks=324,
        step_short_ticks=41,
        step_k_sd=float(8.5),
        step_sigma_floor=float(0.001963984083147008),
    ),
    "PANEL_1X4": StepMeanReversionConfig(
        symbol="PANEL_1X4",
        short_code="PN14",
        pos_limit=10,
        var_window=419,
        z_buy=float(4.4),
        z_sell=float(3.0),
        mean=0.0,
        step_long_ticks=2124,
        step_short_ticks=123,
        step_k_sd=float(4.75),
        step_sigma_floor=float(0.0009159518720085443),
    ),
    "PANEL_2X2": StepMeanReversionConfig(
        symbol="PANEL_2X2",
        short_code="PN22",
        pos_limit=10,
        var_window=490,
        z_buy=float(1.7000000000000002),
        z_sell=float(3.0),
        mean=0.0,
        step_long_ticks=651,
        step_short_ticks=120,
        step_k_sd=float(8.5),
        step_sigma_floor=float(0.0013764540605846847),
    ),
    "PANEL_2X4": StepMeanReversionConfig(
        symbol="PANEL_2X4",
        short_code="PN24",
        pos_limit=10,
        var_window=448,
        z_buy=float(2.1),
        z_sell=float(4.0),
        mean=0.0,
        step_long_ticks=604,
        step_short_ticks=98,
        step_k_sd=float(6.75),
        step_sigma_floor=float(0.005441430002172316),
    ),
    "PANEL_4X4": StepMeanReversionConfig(
        symbol="PANEL_4X4",
        short_code="PN44",
        pos_limit=10,
        var_window=753,
        z_buy=float(3.5),
        z_sell=float(3.7),
        mean=0.0,
        step_long_ticks=2298,
        step_short_ticks=78,
        step_k_sd=float(5.0),
        step_sigma_floor=float(0.0002718863046666178),
    ),
    "ROBOT_DISHES": StepMeanReversionConfig(
        symbol="ROBOT_DISHES",
        short_code="RDIS",
        pos_limit=10,
        var_window=312,
        z_buy=float(1.7000000000000002),
        z_sell=float(3.7),
        mean=0.0,
        step_long_ticks=2118,
        step_short_ticks=151,
        step_k_sd=float(12.0),
        step_sigma_floor=float(0.0008915641324864754),
    ),
    "ROBOT_IRONING": EWMAMeanReversionConfig(
        symbol="ROBOT_IRONING",
        short_code="RIRN",
        pos_limit=10,
        var_window=120,
        z_buy=float(3.4000000000000004),
        z_sell=float(1.8),
        mean=0.0,
        alpha=float(0.014943497924836303),
    ),
    "ROBOT_LAUNDRY": StepMeanReversionConfig(
        symbol="ROBOT_LAUNDRY",
        short_code="RLDY",
        pos_limit=10,
        var_window=101,
        z_buy=float(2.7),
        z_sell=float(3.6),
        mean=0.0,
        step_long_ticks=442,
        step_short_ticks=179,
        step_k_sd=float(5.75),
        step_sigma_floor=float(0.04112842105287768),
    ),
    "ROBOT_MOPPING": StepMeanReversionConfig(
        symbol="ROBOT_MOPPING",
        short_code="RMPP",
        pos_limit=10,
        var_window=797,
        z_buy=float(3.1),
        z_sell=float(3.4000000000000004),
        mean=0.0,
        step_long_ticks=824,
        step_short_ticks=52,
        step_k_sd=float(4.75),
        step_sigma_floor=float(0.0039077876744843835),
    ),
    "ROBOT_VACUUMING": StepMeanReversionConfig(
        symbol="ROBOT_VACUUMING",
        short_code="RVAC",
        pos_limit=10,
        var_window=769,
        z_buy=float(3.4000000000000004),
        z_sell=float(1.3),
        mean=0.0,
        step_long_ticks=2037,
        step_short_ticks=120,
        step_k_sd=float(9.0),
        step_sigma_floor=float(0.004185287373624241),
    ),
    "SLEEP_POD_COTTON": EWMAMeanReversionConfig(
        symbol="SLEEP_POD_COTTON",
        short_code="SLPC",
        pos_limit=10,
        var_window=210,
        z_buy=float(3.6),
        z_sell=float(3.7),
        mean=0.0,
        alpha=float(0.15300269791154542),
    ),
    "SLEEP_POD_LAMB_WOOL": StepMeanReversionConfig(
        symbol="SLEEP_POD_LAMB_WOOL",
        short_code="SLPL",
        pos_limit=10,
        var_window=471,
        z_buy=float(3.5),
        z_sell=float(3.8000000000000003),
        mean=0.0,
        step_long_ticks=481,
        step_short_ticks=20,
        step_k_sd=float(10.25),
        step_sigma_floor=float(0.0004483488676355781),
    ),
    "SLEEP_POD_NYLON": StepMeanReversionConfig(
        symbol="SLEEP_POD_NYLON",
        short_code="SLPN",
        pos_limit=10,
        var_window=203,
        z_buy=float(1.8),
        z_sell=float(2.4000000000000004),
        mean=0.0,
        step_long_ticks=1543,
        step_short_ticks=144,
        step_k_sd=float(10.5),
        step_sigma_floor=float(0.00012934674782510852),
    ),
    "SLEEP_POD_POLYESTER": EWMAMeanReversionConfig(
        symbol="SLEEP_POD_POLYESTER",
        short_code="SLPP",
        pos_limit=10,
        var_window=647,
        z_buy=float(2.8),
        z_sell=float(2.8),
        mean=0.0,
        alpha=float(0.014332677352590202),
    ),
    "SLEEP_POD_SUEDE": StepMeanReversionConfig(
        symbol="SLEEP_POD_SUEDE",
        short_code="SLPS",
        pos_limit=10,
        var_window=629,
        z_buy=float(3.1),
        z_sell=float(3.6),
        mean=0.0,
        step_long_ticks=508,
        step_short_ticks=179,
        step_k_sd=float(11.5),
        step_sigma_floor=float(0.0004055168168039993),
    ),
    "SNACKPACK_CHOCOLATE": StepMeanReversionConfig(
        symbol="SNACKPACK_CHOCOLATE",
        short_code="SPCH",
        pos_limit=10,
        var_window=578,
        z_buy=float(2.8),
        z_sell=float(1.3),
        mean=0.0,
        step_long_ticks=1967,
        step_short_ticks=141,
        step_k_sd=float(6.75),
        step_sigma_floor=float(0.007037013667084517),
    ),
    "SNACKPACK_PISTACHIO": EWMAMeanReversionConfig(
        symbol="SNACKPACK_PISTACHIO",
        short_code="SPPI",
        pos_limit=10,
        var_window=436,
        z_buy=float(3.8000000000000003),
        z_sell=float(3.2),
        mean=0.0,
        alpha=float(0.010743426141510163),
    ),
    "SNACKPACK_RASPBERRY": StepMeanReversionConfig(
        symbol="SNACKPACK_RASPBERRY",
        short_code="SPRA",
        pos_limit=10,
        var_window=526,
        z_buy=float(4.0),
        z_sell=float(3.5),
        mean=0.0,
        step_long_ticks=2281,
        step_short_ticks=114,
        step_k_sd=float(10.5),
        step_sigma_floor=float(0.009129852353765785),
    ),
    "SNACKPACK_STRAWBERRY": StepMeanReversionConfig(
        symbol="SNACKPACK_STRAWBERRY",
        short_code="SPST",
        pos_limit=10,
        var_window=257,
        z_buy=float(3.0),
        z_sell=float(4.0),
        mean=0.0,
        step_long_ticks=1574,
        step_short_ticks=160,
        step_k_sd=float(2.25),
        step_sigma_floor=float(0.0010705876051900246),
    ),
    "SNACKPACK_VANILLA": StepMeanReversionConfig(
        symbol="SNACKPACK_VANILLA",
        short_code="SPVA",
        pos_limit=10,
        var_window=762,
        z_buy=float(2.7),
        z_sell=float(2.9000000000000004),
        mean=0.0,
        step_long_ticks=899,
        step_short_ticks=97,
        step_k_sd=float(3.5),
        step_sigma_floor=float(0.0016274758911983159),
    ),
    "TRANSLATOR_ASTRO_BLACK": EWMAMeanReversionConfig(
        symbol="TRANSLATOR_ASTRO_BLACK",
        short_code="TABK",
        pos_limit=10,
        var_window=500,
        z_buy=float(2.0),
        z_sell=float(1.3),
        mean=0.0,
        alpha=float(0.012582664916859181),
    ),
    "TRANSLATOR_ECLIPSE_CHARCOAL": EWMAMeanReversionConfig(
        symbol="TRANSLATOR_ECLIPSE_CHARCOAL",
        short_code="TECL",
        pos_limit=10,
        var_window=483,
        z_buy=float(2.8),
        z_sell=float(2.8),
        mean=0.0,
        alpha=float(0.01434714576922636),
    ),
    "TRANSLATOR_GRAPHITE_MIST": EWMAMeanReversionConfig(
        symbol="TRANSLATOR_GRAPHITE_MIST",
        short_code="TGRM",
        pos_limit=10,
        var_window=50,
        z_buy=float(4.2),
        z_sell=float(4.4),
        mean=0.0,
        alpha=float(0.051770713184769626),
    ),
    "TRANSLATOR_SPACE_GRAY": StepMeanReversionConfig(
        symbol="TRANSLATOR_SPACE_GRAY",
        short_code="TSPG",
        pos_limit=10,
        var_window=490,
        z_buy=float(4.0),
        z_sell=float(1.6),
        mean=0.0,
        step_long_ticks=1941,
        step_short_ticks=58,
        step_k_sd=float(7.75),
        step_sigma_floor=float(0.0005935046767959667),
    ),
    "TRANSLATOR_VOID_BLUE": StepMeanReversionConfig(
        symbol="TRANSLATOR_VOID_BLUE",
        short_code="TVBL",
        pos_limit=10,
        var_window=410,
        z_buy=float(1.3),
        z_sell=float(2.2),
        mean=0.0,
        step_long_ticks=2045,
        step_short_ticks=97,
        step_k_sd=float(3.25),
        step_sigma_floor=float(0.00021003596928166602),
    ),
    "UV_VISOR_AMBER": StepMeanReversionConfig(
        symbol="UV_VISOR_AMBER",
        short_code="UVAM",
        pos_limit=10,
        var_window=408,
        z_buy=float(3.8000000000000003),
        z_sell=float(2.6),
        mean=0.0,
        step_long_ticks=2498,
        step_short_ticks=49,
        step_k_sd=float(3.25),
        step_sigma_floor=float(0.0004926308925746544),
    ),
    "UV_VISOR_MAGENTA": StepMeanReversionConfig(
        symbol="UV_VISOR_MAGENTA",
        short_code="UVMA",
        pos_limit=10,
        var_window=483,
        z_buy=float(2.7),
        z_sell=float(2.9000000000000004),
        mean=0.0,
        step_long_ticks=1597,
        step_short_ticks=82,
        step_k_sd=float(3.25),
        step_sigma_floor=float(0.001101209996234473),
    ),
    "UV_VISOR_ORANGE": EWMAMeanReversionConfig(
        symbol="UV_VISOR_ORANGE",
        short_code="UVOR",
        pos_limit=10,
        var_window=637,
        z_buy=float(2.7),
        z_sell=float(1.8),
        mean=0.0,
        alpha=float(0.16102777078597785),
    ),
    "UV_VISOR_RED": StepMeanReversionConfig(
        symbol="UV_VISOR_RED",
        short_code="UVRE",
        pos_limit=10,
        var_window=121,
        z_buy=float(1.3),
        z_sell=float(4.5),
        mean=0.0,
        step_long_ticks=1845,
        step_short_ticks=168,
        step_k_sd=float(10.5),
        step_sigma_floor=float(0.004302886481397832),
    ),
    "UV_VISOR_YELLOW": EWMAMeanReversionConfig(
        symbol="UV_VISOR_YELLOW",
        short_code="UVYE",
        pos_limit=10,
        var_window=119,
        z_buy=float(3.5),
        z_sell=float(4.0),
        mean=0.0,
        alpha=float(0.2884841553684487),
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

        # Step-mean mode: two independent chunk accumulators (long / short), O(1), no mid deque.
        lc = getattr(self.cfg, "step_long_ticks", 0)
        sc = getattr(self.cfg, "step_short_ticks", 0)
        if lc > 0 and sc <= 0:
            sc = max(1, lc // 40)
        self._step_mode = lc > 0 and sc > 0
        self._step_short_ticks = sc
        self._step_long_ticks = lc
        prefix = f"{self.cfg.short_code}_sms_"
        self._k_ls_sum = prefix + "ls_sum"
        self._k_ls_n = prefix + "ls_n"
        self._k_ss_sum = prefix + "ss_sum"
        self._k_ss_n = prefix + "ss_n"
        self._ls_sum = float(self.last_traderData.get(self._k_ls_sum, 0.0))
        self._ls_n = int(self.last_traderData.get(self._k_ls_n, 0))
        self._ss_sum = float(self.last_traderData.get(self._k_ss_sum, 0.0))
        self._ss_n = int(self.last_traderData.get(self._k_ss_n, 0))

    def _persist_step_state(self) -> None:
        self.new_trader_data[self._k_ls_sum] = self._ls_sum
        self.new_trader_data[self._k_ls_n] = self._ls_n
        self.new_trader_data[self._k_ss_sum] = self._ss_sum
        self.new_trader_data[self._k_ss_n] = self._ss_n

    def _step_mean_effective(self, mid: float) -> tuple:
        """Chunked running means (sum/count only); σ from `_calc_var(mid - live_long)` (one stream).
        Snap to short anchor when |short−long| > k·σ. Returns (reference_mean, sigma_denom)."""
        cfg = self.cfg
        lc, sc = self._step_long_ticks, self._step_short_ticks

        self._ls_sum += mid
        self._ls_n += 1
        self._ss_sum += mid
        self._ss_n += 1

        live_long = self._ls_sum / self._ls_n
        live_short = self._ss_sum / self._ss_n

        # Rolling variance of (mid − live_long) — same scale as earlier notebook residual σ.
        var = self._calc_var(mid - live_long)
        sigma = max(math.sqrt(var), getattr(cfg, "step_sigma_floor", 1e-3))
        k_sd = getattr(cfg, "step_k_sd", 10.0)

        if abs(live_short - live_long) > k_sd * sigma:
            mu_eff = live_short
        else:
            mu_eff = live_long

        if self._ls_n >= lc:
            self._ls_sum = 0.0
            self._ls_n = 0
        if self._ss_n >= sc:
            self._ss_sum = 0.0
            self._ss_n = 0

        self._persist_step_state()
        return mu_eff, sigma

    def _compute_z(self, value: float, mean: float) -> float:
        dev = value - mean
        var = self._calc_var(dev)
        return dev / math.sqrt(var)

    def get_orders(self):
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        if self._step_mode:
            ref, sigma = self._step_mean_effective(self.wall_mid)
            z = (self.wall_mid - ref) / sigma
        else:
            z = self._compute_z(self.wall_mid, self.mean)

        if z > 0 and abs(z) > self.z_sell_threshold:
            self.ask(self.best_bid, self.max_allowed_sell_volume)
        if z < 0 and abs(z) > self.z_buy_threshold:
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        return {self.name: self.orders}

    def debug(self):
        return self.mean if not self._step_mode else (
            self._ls_sum / max(self._ls_n, 1), self._ss_sum / max(self._ss_n, 1)
        )

class EWMAMeanReversion(MeanReversionTrader):
    def __init__(self,name,state,new_trader_data,cfg = None):
        super().__init__(name, state, new_trader_data, cfg)
        prefix = f"{self.cfg.short_code}"
        self._k_pre_value = prefix + "pv"
        self._k_ewma = prefix + "ew"
        self.prev_value = float(self.last_traderData.get(self._k_pre_value, 0.0))

        self.alpha = self.cfg.alpha

    def _calc_fairvalue(self):
        mid_price = self.wall_mid
        if mid_price is None:
            return self.prev_value
        prev = float(self.prev_value)
        # Cold start: EWMA seeded on first observation.
        ewma = float(mid_price) if prev == 0.0 else (
            float(self.alpha) * float(mid_price) + (1.0 - float(self.alpha)) * prev
        )
        self.prev_value = ewma
        self.new_trader_data[self._k_pre_value] = ewma
        self.new_trader_data[self._k_ewma] = ewma
        return ewma
    
    def get_orders(self):
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        z = self._compute_z(self.wall_mid, self._calc_fairvalue())

        if z > 0 and abs(z) > self.z_sell_threshold:
            self.ask(self.best_bid, self.max_allowed_sell_volume)
        if z < 0 and abs(z) > self.z_buy_threshold:
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        return {self.name: self.orders}

# Maps config type → trader class so Trader.run() dispatches automatically
TRADER_BY_CONFIG_TYPE = {
    StepMeanReversionConfig: MeanReversionTrader,
    EWMAMeanReversionConfig: EWMAMeanReversion,
}

class Trader:
    def run(self, state: TradingState):
        result          = {}
        new_trader_data = {}
        conversions     = 0

        for symbol, cfg in CONFIGS.items():
            trader_cls = TRADER_BY_CONFIG_TYPE.get(type(cfg))
            if trader_cls is None:
                continue

            try:
                trader = trader_cls(symbol, state, new_trader_data, cfg=cfg)
                result.update(trader.get_orders())
            except Exception as e:
                print(f"ERROR in trader for {symbol}: {e}")

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ""

        return result, conversions, final_trader_data



        