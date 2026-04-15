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

    def trade_pepper_root(self, order_depth: OrderDepth, position: int, current_time: int, price_series: List[tuple[int, float]]) -> tuple[List[Order], List[tuple[int, float]], float, float]:
        orders: List[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        POSITION_LIMIT = 80
        position_offset = 60
        gamma = 0.4
        k_val = 0.15
        lookback = 20  # Increased lookback to smooth out single-tick sweeps
        slope = 0.001
        n_offset = 20
        intercept_initilisation = 15
        MAX_VAR = 2.0  # Cap the variance to prevent 4000+ tick jumps in r

        ts = np.array([p[0] for p in price_series], dtype=float) if price_series else np.array([])
        prices = np.array([p[1] for p in price_series], dtype=float) if price_series else np.array([])

        nominal = 0.0
        volume = 0.0

        for i in range(min(2, len(sell_orders))):
            nominal += sell_orders[i][0] * abs(sell_orders[i][1])
            volume += abs(sell_orders[i][1])
            
        for i in range(min(2, len(buy_orders))):
            nominal += buy_orders[i][0] * abs(buy_orders[i][1])
            volume += abs(buy_orders[i][1])

        # FIX 2: Properly append vwap in BOTH scenarios so the array doesn't freeze
        if volume > 0:
            vwap = nominal / volume
        else:
            if len(prices) > 0:
                time_delta = current_time - ts[-1]
                vwap = prices[-1] + (slope * time_delta)
            else:
                return orders, price_series, 0.0, 0.0

        # Append and trim
        price_series.append((current_time, vwap))
        price_series = price_series[-(lookback + 1):]

        # Recalculate arrays after appending
        ts = np.array([p[0] for p in price_series], dtype=float)
        prices = np.array([p[1] for p in price_series], dtype=float)

        # FIX 1: Cap the variance to prevent the equation from detonating
        if len(prices) > 2:
            returns = np.diff(prices)
            var = np.clip(np.var(returns), 0.0001, MAX_VAR) 
        else:
            var = 0.01

        if len(prices) >= intercept_initilisation and  not intercept:
            intercepts = prices - slope * ts
            intercept = float(np.mean(intercepts))
        else:
            if sell_orders:
                best_ask = sell_orders[0][0]
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask + 10, -1))
            if buy_orders:
                best_bid = buy_orders[0][0]
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid + 1, 10))
            return orders, price_series, 0.0, 0.0

        s = intercept + slope * (current_time + n_offset * 100)

        # r will now only shift by a maximum of: 140 * 0.4 * 2.0 = 112 ticks at max inventory
        r = s - ((position - position_offset) * gamma * var)
        delta = (gamma * var + (2 / gamma * math.log(1 + (gamma / k_val))))

        bid_price = int(math.floor(r - delta / 2))
        ask_price = int(math.ceil(r + delta / 2))

        buy_qty = POSITION_LIMIT - position
        sell_qty = -POSITION_LIMIT - position

        if buy_qty > 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, buy_qty))

        if sell_qty < 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, sell_qty))

        return orders, price_series, s, r
        
    def trade_osmium(
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

        # Constants

        POSITION_LIMIT = 80
        gamma = 0.001
        k_val = 0.07
        T = 1

        if not kf_p or len(kf_p) != 4:
            kf_p = [25.0, 0.0, 0.0, 4.0]
        p00, p01, p10, p11 = kf_p

        # --- STEP 1: ROBUST MICRO-PRICE ---
        if sell_orders and buy_orders:
            best_ask, ask_vol = sell_orders[0]
            best_bid, bid_vol = buy_orders[0]
            denominator = (abs(bid_vol) + abs(ask_vol))
            micro_price = (best_ask * abs(ask_vol) + best_bid * abs(bid_vol)) / denominator
        else:
            # If the book is empty, use our last known 'level' (kf_mu) plus the drift
            # This keeps the Kalman Filter from flatlining.
            micro_price = kf_mu + kf_beta if kf_mu != 0 else 0.0

        # If we have no price and no history, we can't trade.
        if micro_price == 0:
            return orders, mid_prices, kf_mu, kf_beta, kf_p, 0.0, 0.0

        # --- STEP 2: KALMAN UPDATE ---
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
        mid_prices = mid_prices[-(101):]

        # --- STEP 3: AS LOGIC ---
        # (Variance calculation remains the same)
        if len(mid_prices) > 2:
            returns = np.diff(mid_prices)
            var = max(np.var(returns), 0.0001)
        else:
            var = 0.01

        s = kf_mu
        gamma, k_val, T = 0.001, 0.07, 1
        r = s - (position * gamma * var * T)
        delta = (gamma * var * T + (2 / gamma * math.log(1 + (gamma / k_val))))

        # --- STEP 4: SAFE ORDER PLACEMENT ---
        buy_qty = 80 - position
        sell_qty = -80 - position

        # Only execute Market Taking if orders exist
        if sell_orders and buy_orders:
            sell_signal = kf_beta < -self.KF_DRIFT_EPS
            buy_signal = kf_beta > self.KF_DRIFT_EPS

            # Aggressive taking (only if market exists)
            for ask_price, ask_vol in sell_orders:
                if ask_price < s + 1.5 and buy_signal and buy_qty > 0:
                    take_vol = min(abs(ask_vol), buy_qty)
                    orders.append(Order("ASH_COATED_OSMIUM", ask_price, take_vol))
                    buy_qty -= take_vol

            for bid_price, bid_vol in buy_orders:
                if bid_price > s - 1.5 and sell_signal and sell_qty < 0:
                    take_vol = min(abs(bid_vol), abs(sell_qty))
                    orders.append(Order("ASH_COATED_OSMIUM", bid_price, -take_vol))
                    sell_qty += take_vol

            # Market Making (Penny the market safely)
            as_bid = max(int(math.floor(r - delta / 2)), buy_orders[0][0] + 1)
            as_ask = min(int(math.ceil(r + delta / 2)), sell_orders[0][0] - 1)
        else:
            # If market is empty, just place orders at the theoretical AS prices
            as_bid = int(math.floor(r - delta / 2))
            as_ask = int(math.ceil(r + delta / 2))

        if buy_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", as_bid, buy_qty))
        if sell_qty < 0:
            orders.append(Order("ASH_COATED_OSMIUM", as_ask, sell_qty))

        return orders, mid_prices, kf_mu, kf_beta, kf_p_out, r, kf_beta

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        
        # Robust JSON decoding
        try:
            data = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        for product in ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]:
            if product not in state.order_depths:
                continue
            
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            
            if product == "ASH_COATED_OSMIUM":
                prev_mid = data.get("OSM_mid", [])
                kf_mu = data.get("OSM_mu", 0.0)
                kf_beta = data.get("OSM_beta", 0.0)
                kf_p = data.get("OSM_p", [25.0, 0.0, 0.0, 4.0])

                orders, mid_p, n_mu, n_beta, n_p, r_val, sig = self.trade_osmium(
                    order_depth, position, prev_mid, kf_mu, kf_beta, kf_p
                )
                
                result[product] = orders
                data["OSM_mid"], data["OSM_mu"] = mid_p, n_mu
                data["OSM_beta"], data["OSM_p"] = n_beta, n_p
                self.log_data(state, product, position, orders, n_mu, r_val, sig)

            elif product == "INTARIAN_PEPPER_ROOT":
                # Ensure we handle the list of lists vs list of tuples issue
                raw_series = data.get("PEP_series", [])
                price_series = [tuple(p) for p in raw_series] # Force tuple format

                orders, new_series, fv, r_val = self.trade_pepper_root(
                    order_depth, position, state.timestamp, price_series
                )

                result[product] = orders
                data['PEP_series'] = new_series
                
                # Only log if we have a valid Fair Value
                if fv != 0:
                    self.log_data(state, product, position, orders, fv, r_val, 0)

        return result, conversions, jsonpickle.encode(data)