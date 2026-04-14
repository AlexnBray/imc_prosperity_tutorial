"""
Prosperity tutorial trader: EMERALDS stationary MM + TOMATOES Kalman drift.

Tomatoes use a local linear trend Kalman filter on L2 wall_mid (level + drift);
emeralds use fixed fair 10_000 with the original taking / penny-MM logic.
"""
import math
import jsonpickle
import numpy as np
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol


class Trader:
    # Local linear trend Kalman on wall_mid: state = [level, drift_per_tick].
    KF_R_OBS = 1
    KF_Q_LEVEL = 0.3
    KF_Q_DRIFT = 0.015
    KF_DRIFT_EPS = 0.04


    @staticmethod
    def _kalman_local_linear_tick(
        z: float,
        mu: float,
        beta: float,
        p00: float,
        p01: float,
        p10: float,
        p11: float,
        r_obs: float,
        q_level: float,
        q_drift: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """Predict + update one tick; observe price level only."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.array([[q_level, 0.0], [0.0, q_drift]])
        H = np.array([[1.0, 0.0]])
        x = np.array([mu, beta])
        P = np.array([[p00, p01], [p10, p11]])

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        y = z - float(H @ x_pred)
        s = float(H @ P_pred @ H.T) + r_obs
        k = (P_pred @ H.T).flatten() / s
        x_new = x_pred + k * y
        P_new = (np.eye(2) - np.outer(k, H)) @ P_pred
        return (
            float(x_new[0]),
            float(x_new[1]),
            float(P_new[0, 0]),
            float(P_new[0, 1]),
            float(P_new[1, 0]),
            float(P_new[1, 1]),
        )

    def log_data(self, state: TradingState, product: str, position: int, orders: List[Order], fv: float, effective_fv: float, signal: float):
        bid_map = {}
        ask_map = {}

        for o in orders:
            if o.quantity > 0:
                bid_map[o.price] = bid_map.get(o.price, 0) + o.quantity
            else:
                ask_map[o.price] = ask_map.get(o.price, 0) + o.quantity

        bids_str = ";".join([f"{p}:{q}" for p, q in sorted(bid_map.items(), reverse=True)])
        asks_str = ";".join([f"{p}:{q}" for p, q in sorted(ask_map.items())])

        print(f"[ALGO],{state.timestamp},{product},{position},{fv:.2f},{effective_fv:.2f},{signal:.2f},[{bids_str}],[{asks_str}]")

    def trade_emeralds(self, order_depth: OrderDepth, position: int) -> Tuple[List[Order], float]:
        orders: List[Order] = []
        limit = 80
        fv = 10000.0

        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        if not sell_orders or not buy_orders:
            return orders, fv

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]

        initial_pos = position
        buy_capacity = limit - position
        sell_capacity = limit + position

        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_price < fv:
                take_vol = min(vol, buy_capacity)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", ask_price, take_vol))
                    buy_capacity -= take_vol
            elif math.isclose(ask_price, fv, abs_tol=0.1):
                take_vol = min(vol, buy_capacity, -initial_pos)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", ask_price, take_vol))
                    buy_capacity -= take_vol
                    initial_pos += take_vol

        initial_pos = position
        for bid_price, bid_vol in buy_orders:
            if bid_price > fv:
                take_vol = min(bid_vol, sell_capacity)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", bid_price, -take_vol))
                    sell_capacity -= take_vol
            elif math.isclose(bid_price, fv, abs_tol=0.1):
                take_vol = min(bid_vol, sell_capacity, initial_pos)
                if take_vol > 0:
                    orders.append(Order("EMERALDS", bid_price, -take_vol))
                    sell_capacity -= take_vol
                    initial_pos -= take_vol

        min_edge = 1
        my_bid = min(math.floor(fv) - min_edge, best_bid + 1)
        my_ask = max(math.ceil(fv) + min_edge, best_ask - 1)

        if buy_capacity > 0:
            orders.append(Order("EMERALDS", my_bid, buy_capacity))

        if sell_capacity > 0:
            orders.append(Order("EMERALDS", my_ask, -sell_capacity))

        return orders, fv

    def trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        mid_prices: List[float],
        kf_mu: float,
        kf_beta: float,
        kf_p: List[float],
    ) -> tuple[List[Order], List[float], float, float, List[float], float, float]:
        """
        Tomatoes: local linear trend Kalman on L2 wall_mid.

        Filtered level = AS fair anchor; filtered drift gates tactical taking
        (replaces dual-EMA crossover for lower-lag drift read).
        """
        orders: List[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        if not kf_p or len(kf_p) != 4:
            kf_p = [0.0, 0.0, 0.0, 0.0]
        p00, p01, p10, p11 = kf_p

        if not sell_orders or not buy_orders:
            return orders, mid_prices, kf_mu, kf_beta, [p00, p01, p10, p11], 0, 0

        if len(sell_orders) < 2 or len(buy_orders) < 2:
            return orders, mid_prices, kf_mu, kf_beta, [p00, p01, p10, p11], 0, 0

        POSITION_LIMIT = 80
        gamma = 0.07
        k = 0.18
        T = 1
        lookback = 12

        wall_mid = (sell_orders[1][0] + buy_orders[1][0]) / 2.0

        if p00 == p01 == p10 == p11 == 0.0:
            kf_mu = wall_mid
            kf_beta = 0.0
            p00, p01, p10, p11 = 25.0, 0.0, 0.0, 4.0
        else:
            kf_mu, kf_beta, p00, p01, p10, p11 = self._kalman_local_linear_tick(
                wall_mid,
                kf_mu,
                kf_beta,
                p00,
                p01,
                p10,
                p11,
                self.KF_R_OBS,
                self.KF_Q_LEVEL,
                self.KF_Q_DRIFT,
            )

        s = kf_mu
        kf_p_out = [p00, p01, p10, p11]

        sell_signal = kf_beta < -self.KF_DRIFT_EPS
        buy_signal = kf_beta > self.KF_DRIFT_EPS

        mid_prices.append(wall_mid)
        mid_prices = mid_prices[-(lookback + 1):]

        if len(mid_prices) < lookback + 1:
            return orders, mid_prices, kf_mu, kf_beta, kf_p_out, 0, 0

        returns = np.diff(mid_prices)
        var = np.var(returns)

        q = position
        r = s - (q * gamma * var * T)

        buy_qty = POSITION_LIMIT - position
        sell_qty = -POSITION_LIMIT - position

        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_price < s + 1.5 and buy_signal:
                take_vol = min(vol, buy_qty)
                if take_vol > 0:
                    orders.append(Order("TOMATOES", ask_price, take_vol))
                    buy_qty -= take_vol

        for bid_price, bid_vol in buy_orders:
            if bid_price > s - 1.5 and sell_signal:
                take_vol = min(bid_vol, -sell_qty)
                if take_vol > 0:
                    orders.append(Order("TOMATOES", bid_price, -take_vol))
                    sell_qty += take_vol

        delta = gamma * var * T + (2 / gamma) * math.log(1 + gamma / k)

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]

        as_bid = math.floor(r - delta / 2)
        as_ask = math.ceil(r + delta / 2)

        new_bid_price = min(as_bid, best_bid + 1)
        new_ask_price = max(as_ask, best_ask - 1)

        if buy_qty > 0:
            orders.append(Order("TOMATOES", int(new_bid_price), buy_qty))
        if sell_qty < 0:
            orders.append(Order("TOMATOES", int(new_ask_price), sell_qty))

        return orders, mid_prices, kf_mu, kf_beta, kf_p_out, r, kf_beta

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0

        if state.traderData:
            try:
                data = jsonpickle.decode(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        for product in state.order_depths:
            current_fv = 0.0
            effective_fv = 0.0
            signal = 0.0
            result[product] = []

            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product], current_fv = self.trade_emeralds(order_depth, position)
                effective_fv = current_fv
                signal = current_fv

            elif product == "TOMATOES":
                prev_prices = data.get("TOMATOES_mid", [])
                kf_mu = data.get("TOMATOES_s", 0.0)
                kf_beta = data.get("TOMATOES_kf_beta", 0.0)
                kf_p = data.get("TOMATOES_kf_p", [0.0, 0.0, 0.0, 0.0])

                result[product], mid_prices, new_mu, new_beta, new_p, effective_fv, signal = self.trade_tomatoes(
                    order_depth, position, prev_prices, kf_mu, kf_beta, kf_p
                )

                data["TOMATOES_mid"] = mid_prices
                data["TOMATOES_s"] = new_mu
                data["TOMATOES_kf_beta"] = new_beta
                data["TOMATOES_kf_p"] = new_p
                current_fv = new_mu

            self.log_data(state, product, position, result[product], current_fv, effective_fv, signal)

        new_trader_data = jsonpickle.encode(data)
        return result, conversions, new_trader_data
