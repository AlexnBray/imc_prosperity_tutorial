"""
PEPPER-only buy-and-hold strategy.

Goal:
- Accumulate INTARIAN_PEPPER_ROOT to max long inventory quickly.
- Hold the long position for the rest of the day.

This is intentionally simple to mimic a near-linear upward PnL profile
when PEPPER has persistent upward drift.
"""

import json
from typing import Dict, List, Tuple

from datamodel import OrderDepth, Order, Symbol, TradingState


PEPPER = "INTARIAN_PEPPER_ROOT"
PEPPER_LIMIT = 80


class Trader:
    # Become more aggressive early, then mostly hold.
    AGGRESSIVE_BUILD_TICKS =5000
    TAKE_CHUNK = 80

    def _trade_pepper(self, order_depth: OrderDepth, position: int, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())  # (price, negative vol)
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)  # (price, positive vol)
        if not sell_orders or not buy_orders:
            return orders

        best_ask, best_ask_vol = sell_orders[0]
        best_bid, _best_bid_vol = buy_orders[0]
        buy_cap = PEPPER_LIMIT - position

        # Phase 1: aggressive build to max long.
        if timestamp <= self.AGGRESSIVE_BUILD_TICKS and buy_cap > 0:
            take = min(-best_ask_vol, buy_cap, self.TAKE_CHUNK)
            if take > 0:
                orders.append(Order(PEPPER, best_ask, take))
                buy_cap -= take

        # Phase 2: passive top-up if not yet max long.
        if buy_cap > 0:
            # Improve top bid by 1 tick to keep queue priority while still passive.
            orders.append(Order(PEPPER, best_bid + 1, buy_cap))

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0
        trader_data = "{}"

        for product, order_depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == PEPPER:
                result[product] = self._trade_pepper(order_depth, pos, state.timestamp)
            else:
                result[product] = []

        print(
            json.dumps(
                {
                    "ts": state.timestamp,
                    "pepper_pos": state.position.get(PEPPER, 0),
                    "n_orders": len(result.get(PEPPER, [])),
                }
            )
        )
        return result, conversions, trader_data
