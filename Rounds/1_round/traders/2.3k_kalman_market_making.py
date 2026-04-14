import math
import jsonpickle
import numpy as np
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol

class Trader:
    # Local linear trend Kalman on micro_price: state = [level, drift_per_tick].
    KF_R_OBS = 200
    KF_Q_LEVEL = 0.2
    KF_Q_DRIFT = 0.01
    KF_DRIFT_EPS = 0.04

    @staticmethod
    def _kalman_local_linear_tick(
        z: float, mu: float, beta: float,
        p00: float, p01: float, p10: float, p11: float,
        r_obs: float, q_level: float, q_drift: float,
    ) -> Tuple[float, float, float, float, float, float]:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.array([[q_level, 0.0], [0.0, q_drift]])
        H = np.array([[1.0, 0.0]])
        x = np.array([mu, beta])
        P = np.array([[p00, p01], [p10, p11]])

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        y = z - float(H @ x_pred)
        s_cov = float(H @ P_pred @ H.T) + r_obs
        k = (P_pred @ H.T).flatten() / s_cov
        
        x_new = x_pred + k * y
        P_new = (np.eye(2) - np.outer(k, H)) @ P_pred
        
        return (
            float(x_new[0]), float(x_new[1]),
            float(P_new[0, 0]), float(P_new[0, 1]),
            float(P_new[1, 0]), float(P_new[1, 1])
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
        print(f"[ALGO],{state.timestamp},{product},{position},{fv:.2f},{effective_fv:.2f},{signal:.4f},[{bids_str}],[{asks_str}]")

    def trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        mid_prices: List[float],
        kf_mu: float,
        kf_beta: float,
        kf_p: List[float],
    ) -> tuple[List[Order], List[float], float, float, List[float], float, float]:
        orders: List[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        if not kf_p or len(kf_p) != 4:
            kf_p = [25.0, 0.0, 0.0, 4.0]
        p00, p01, p10, p11 = kf_p

        if not sell_orders or not buy_orders:
            return orders, mid_prices, kf_mu, kf_beta, [p00, p01, p10, p11], 0, 0

        # Constants
        POSITION_LIMIT = 80
        gamma = 0.001
        k_val = 0.07
        T = 1
        lookback = 100

        # Micro price calculation
        best_ask, ask_vol = sell_orders[0]
        best_bid, bid_vol = buy_orders[0]
        denominator = (abs(bid_vol) + abs(ask_vol))
        micro_price = (best_ask * abs(ask_vol) + best_bid * abs(bid_vol)) / denominator

        # Kalman Update
        if kf_mu == 0:
            kf_mu = micro_price
            kf_beta = 0.0
        else:
            kf_mu, kf_beta, p00, p01, p10, p11 = self._kalman_local_linear_tick(
                micro_price, kf_mu, kf_beta, p00, p01, p10, p11,
                self.KF_R_OBS, self.KF_Q_LEVEL, self.KF_Q_DRIFT
            )

        kf_p_out = [p00, p01, p10, p11]
        mid_prices.append(micro_price)
        mid_prices = mid_prices[-(lookback + 1):]

        # Calculate variance (needed for AS logic)
        if len(mid_prices) > 2:
            returns = np.diff(mid_prices)
            var = max(np.var(returns), 0.0001) # Avoid zero variance
        else:
            var = 0.01 # Initial guess

        # Market Taking Logic (Tactical)
        s = kf_mu
        sell_signal = kf_beta < -self.KF_DRIFT_EPS
        buy_signal = kf_beta > self.KF_DRIFT_EPS

        buy_qty = POSITION_LIMIT - position
        sell_qty = -POSITION_LIMIT - position

        # Aggressive taking
        for ask_price, ask_vol in sell_orders:
            vol = abs(ask_vol)
            if ask_price < s + 1.5 and buy_signal:
                take_vol = min(vol, buy_qty)
                if take_vol > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ask_price, take_vol))
                    buy_qty -= take_vol

        for bid_price, bid_vol in buy_orders:
            vol = abs(bid_vol)
            if bid_price > s - 1.5 and sell_signal:
                take_vol = min(vol, abs(sell_qty))
                if take_vol > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bid_price, -take_vol))
                    sell_qty += take_vol

        # Avellaneda-Stoikov Market Making
        r = s - (position * gamma * var * T)
        delta = (gamma * var * T + (2 / gamma * math.log(1 + (gamma / k_val))))

        as_bid = int(math.floor(r - delta / 2))
        as_ask = int(math.ceil(r + delta / 2))

        # Penny the market if AS prices are too far
        as_bid = max(as_bid, best_bid+1)
        as_ask = min(as_ask, best_ask-1)

        if buy_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", as_bid, buy_qty))
        if sell_qty < 0:
            orders.append(Order("ASH_COATED_OSMIUM", as_ask, sell_qty))

        return orders, mid_prices, kf_mu, kf_beta, kf_p_out, r, kf_beta

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        if state.traderData:
            try:
                data = jsonpickle.decode(state.traderData)
            except:
                data = {}
        else:
            data = {}

        for product in state.order_depths:
            result[product] = []
            if product == "ASH_COATED_OSMIUM":
                order_depth = state.order_depths[product]
                position = state.position.get(product, 0)
                
                prev_prices = data.get("ASH_COATED_OSMIUM_mid", [])
                kf_mu = data.get("ASH_COATED_OSMIUM_s", 0.0)
                kf_beta = data.get("ASH_COATED_OSMIUM_kf_beta", 0.0)
                kf_p = data.get("ASH_COATED_OSMIUM_kf_p", [])

                orders, mid_prices, new_mu, new_beta, new_p, r_val, signal = self.trade_tomatoes(
                    order_depth, position, prev_prices, kf_mu, kf_beta, kf_p
                )

                result[product] = orders
                data["ASH_COATED_OSMIUM_mid"] = mid_prices
                data["ASH_COATED_OSMIUM_s"] = new_mu
                data["ASH_COATED_OSMIUM_kf_beta"] = new_beta
                data["ASH_COATED_OSMIUM_kf_p"] = new_p
                
                self.log_data(state, product, position, orders, new_mu, r_val, signal)

        new_trader_data = jsonpickle.encode(data)
        return result, conversions, new_trader_data