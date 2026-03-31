from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import Any
import json
import math


# ── Serialisation helpers ────────────────────────────────────────────────────

class Logger:

    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3_750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
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
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values()
            for t in arr
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        co = {
            p: [
                o.bidPrice, o.askPrice, o.transportFees,
                o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex,
            ]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [
            [o.symbol, o.price, o.quantity]
            for arr in orders.values()
            for o in arr
        ]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()


# ── Position limits ─────────────────────────────────########################################

LIMITS: dict[Symbol, int] = {
    "EMERALDS": 50,
    "TOMATOES": 50,
}
"""
Emeralds are stationary, centred at 10k (reference to rainforest resin in P3) spread is around 16
so put a couple of ticks inside that to get filled while still earning edge
"""
EMERALDS_FAIR_VALUE = 10_000

EMERALDS_SPREAD     = 7      # ticks both sides
EMERALDS_TAKE_EDGE  = 1      # lift up if order book crosses the fair valu by at least this much

"""
tomatoes are drifting price same as P3. 
a short ema can be used 
"""

TOMATOES_EMA_SPAN   = 6.5  
TOMATOES_TAKE_EDGE  = 1    


# ── Utility functions ────────────────────────────────────────────────────────

def best_bid(od: OrderDepth) -> tuple[int, int] | None:
    if not od.buy_orders:
        return None
    p = max(od.buy_orders)
    return p, od.buy_orders[p]


def best_ask(od: OrderDepth) -> tuple[int, int] | None:
    if not od.sell_orders:
        return None
    p = min(od.sell_orders)
    return p, od.sell_orders[p]


def mid_price(od: OrderDepth) -> float | None:
    b = best_bid(od)
    a = best_ask(od)
    if b and a:
        return (b[0] + a[0]) / 2.0
    return None


def buy_capacity(position: int, limit: int) -> int:
    return limit - position


def sell_capacity(position: int, limit: int) -> int:
    return limit + position


# ── Emeralds strategy — pure market-making around a hardcoded fair value ─────

def strategy_emeralds(
    od: OrderDepth,
    position: int,
    limit: int,
) -> list[Order]:
    """
    Emeralds have a stable fair value of 10,000.

    1. Take any ask strictly below (fair - take_edge) — free money.
    2. Hit any bid strictly above (fair + take_edge) — free money.
    3. Post passive bid at (fair - spread) and passive ask at (fair + spread).

    Quote sizes are capped to remaining position capacity.
    """
    orders: list[Order] = []
    fv = EMERALDS_FAIR_VALUE

    # ── Aggressive takes (walk the book) ──────────────────────────────────
    pos = position

    for ask in sorted(od.sell_orders):
        if ask >= fv - EMERALDS_TAKE_EDGE:
            break
        vol = abs(od.sell_orders[ask])
        qty = min(vol, buy_capacity(pos, limit))
        if qty <= 0:
            break
        orders.append(Order("EMERALDS", ask, qty))
        pos += qty

    for bid in sorted(od.buy_orders, reverse=True):
        if bid <= fv + EMERALDS_TAKE_EDGE:
            break
        vol = od.buy_orders[bid]
        qty = min(vol, sell_capacity(pos, limit))
        if qty <= 0:
            break
        orders.append(Order("EMERALDS", bid, -qty))
        pos -= qty

    # ── Passive market-making quotes ──────────────────────────────────────
    passive_buy_qty  = min(buy_capacity(pos, limit),  10)
    passive_sell_qty = min(sell_capacity(pos, limit), 10)

    if passive_buy_qty > 0:
        orders.append(Order("EMERALDS", fv - EMERALDS_SPREAD, passive_buy_qty))
    if passive_sell_qty > 0:
        orders.append(Order("EMERALDS", fv + EMERALDS_SPREAD, -passive_sell_qty))

    return orders


# ── Tomatoes strategy — EMA-based adaptive market-making + taking ─────────────

def strategy_tomatoes(
    od: OrderDepth,
    position: int,
    limit: int,
    ema: float | None,
) -> list[Order]:
    """
    Tomatoes drift over time so we can't use a hardcoded fair value.

    When an EMA estimate is available:
      - Aggressively take orders that cross the EMA by at least TOMATOES_TAKE_EDGE.
      - Post passive quotes either side of the EMA.

    Before we have enough history for the EMA, fall back to quoting around mid.
    """
    orders: list[Order] = []

    fair = ema if ema is not None else mid_price(od)
    if fair is None:
        return orders

    pos = position

    # ── Aggressive takes ───────────────────────────────────────────────────
    for ask in sorted(od.sell_orders):
        if ask >= fair - TOMATOES_TAKE_EDGE:
            break
        vol = abs(od.sell_orders[ask])
        qty = min(vol, buy_capacity(pos, limit))
        if qty <= 0:
            break
        orders.append(Order("TOMATOES", ask, qty))
        pos += qty

    for bid in sorted(od.buy_orders, reverse=True):
        if bid <= fair + TOMATOES_TAKE_EDGE:
            break
        vol = od.buy_orders[bid]
        qty = min(vol, sell_capacity(pos, limit))
        if qty <= 0:
            break
        orders.append(Order("TOMATOES", bid, -qty))
        pos -= qty

    # ── Passive quotes around fair ─────────────────────────────────────────
    # use a slightly wider spread than Emeralds
    spread = 3
    passive_buy_qty  = min(buy_capacity(pos, limit),  8)
    passive_sell_qty = min(sell_capacity(pos, limit), 8)

    if passive_buy_qty > 0:
        orders.append(Order("TOMATOES", math.floor(fair) - spread, passive_buy_qty))
    if passive_sell_qty > 0:
        orders.append(Order("TOMATOES", math.ceil(fair) + spread, -passive_sell_qty))

    return orders


# ── Persistent trader state ───────────────────────────────────────────────────

class TraderState:

    def __init__(self) -> None:
        self.price_history: dict[str, list[float]] = {}

    def update(self, symbol: str, price: float, max_len: int = 50) -> None:
        hist = self.price_history.setdefault(symbol, [])
        hist.append(price)
        if len(hist) > max_len:
            hist.pop(0)

    def ema(self, symbol: str, span: int) -> float | None:
        hist = self.price_history.get(symbol, [])
        if len(hist) < 2:
            return None
        k = 2.0 / (span + 1)
        val = hist[0]
        for p in hist[1:]:
            val = p * k + val * (1 - k)
        return val

    @staticmethod
    def from_json(raw: str) -> "TraderState":
        ts = TraderState()
        if raw:
            try:
                data = json.loads(raw)
                ts.price_history = data.get("price_history", {})
            except Exception:
                pass
        return ts

    def to_json(self) -> str:
        return json.dumps({"price_history": self.price_history})


# ── Main Trader class ─────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        ts = TraderState.from_json(state.traderData)

        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        for symbol, od in state.order_depths.items():
            limit    = LIMITS.get(symbol, 20)
            position = state.position.get(symbol, 0)

            mp = mid_price(od)
            if mp is not None:
                ts.update(symbol, mp)

            if symbol == "EMERALDS":
                # 10,000 hardcoded fair value market making.
                orders = strategy_emeralds(od, position, limit)

            elif symbol == "TOMATOES":
                # EMA
                ema = ts.ema(symbol, TOMATOES_EMA_SPAN)
                orders = strategy_tomatoes(od, position, limit, ema)

            else:
                # claude created fallback for any new product introduced later on
                orders = []
                if mp is not None:
                    spread = 3
                    if buy_capacity(position, limit) > 0:
                        orders.append(Order(symbol, math.floor(mp) - spread, 5))
                    if sell_capacity(position, limit) > 0:
                        orders.append(Order(symbol, math.ceil(mp) + spread, -5))

            if orders:
                result[symbol] = orders

            logger.print(
                f"{symbol:12s}  pos={position:+4d}  mid={mp!s:>8}  "
                f"orders={len(orders)}"
            )

        trader_data = ts.to_json()
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data