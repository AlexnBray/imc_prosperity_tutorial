"""
Kalman + OU market maker for ASH_COATED_OSMIUM.

Pipeline:
  1. Kalman local-linear-trend filter estimates fair value (mu) and drift (beta).
  2. Residual = observed mid - mu is modeled as an OU process with rolling
     mean-reversion speed theta and volatility sigma.
  3. Avellaneda-Stoikov style quoting uses OU-derived spread and inventory skew.
  4. Drift-gated tactical taking when beta exceeds an epsilon threshold.

Only trades ASH_COATED_OSMIUM. Other products are passed through with no orders.
"""
import math
import json
import numpy as np
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol

POSITION_LIMIT = 80


class Logger:
    """Structured JSON logger for post-run analysis."""

    def __init__(self):
        self._logs: List[dict] = []

    def log(self, entry: dict) -> None:
        self._logs.append(entry)

    def flush(self, state: TradingState) -> str:
        out = json.dumps({
            "timestamp": state.timestamp,
            "entries": self._logs,
        }, default=str)
        self._logs = []
        return out


logger = Logger()


class Trader:
    # Kalman params (MLE grid: Rounds/1_round/data/optimize_kalman_mle.py — keep baselines in sync)
    KF_R_OBS = 1
    KF_Q_LEVEL = 0.3
    KF_Q_DRIFT = 0.015
    KF_DRIFT_EPS = 0.04

    # OU / A-S
    OU_LOOKBACK = 60
    OU_MIN_OBS = 15
    GAMMA = 0.07
    K_LIQ = 0.18
    TAKE_EDGE = 1.5
    MIN_SPREAD = 2
    # PnL-oriented add-ons (OU term was previously unused except in logs)
    OU_SPREAD_BLEND = 0.12          # blend OU sigma/theta into half-spread scale (capped below)
    OU_SPREAD_CAP = 8.0             # cap raw OU adj so outliers do not explode quotes
    RESIDUAL_Z_WIDEN = 0.35         # add to delta per unit |z| above 1 on (mid - fv) / sigma_eps
    RESIDUAL_SKEW_K = 0.018         # pull reservation toward mean-revert: r -= k * (mid - fv)
    RESIDUAL_SKEW_CLIP = 4.0        # max ticks of skew from residual
    INV_SPREAD_ASYM = 0.28        # widen each half-spread when inventory adds risk on that side
    TACTICAL_MAX_EXT = 1.25         # skip lift if mid already this far above fv (buy) / below fv (sell)

    @staticmethod
    def _kalman_tick(
        z: float,
        mu: float, beta: float,
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
        innov = z - float(H @ x_pred)
        S = float(H @ P_pred @ H.T) + r_obs
        K = (P_pred @ H.T).flatten() / S
        x_new = x_pred + K * innov
        P_new = (np.eye(2) - np.outer(K, H)) @ P_pred
        return (
            float(x_new[0]), float(x_new[1]),
            float(P_new[0, 0]), float(P_new[0, 1]),
            float(P_new[1, 0]), float(P_new[1, 1]),
        )

   
    # OU parameter estimation
    @staticmethod
    def _estimate_ou(residuals: List[float]) -> Tuple[float, float]:
        """Estimate OU theta and sigma from recent residuals via AR(1) regression.
        Returns (theta, sigma). theta > 0 implies mean-reversion.
        """
        r = np.array(residuals, dtype=float)
        if len(r) < 5:
            return 0.0, 1.0

        y = r[1:]
        x = r[:-1]

        x_mean = np.mean(x)
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-12:
            return 0.0, max(np.std(r), 1e-6)

        phi = np.sum((x - x_mean) * (y - np.mean(y))) / denom
        phi = max(min(phi, 0.9999), -0.9999)

        theta = -math.log(abs(phi)) if abs(phi) > 1e-6 else 0.0
        if phi <= 0:
            theta = abs(theta)

        eps = y - phi * x
        sigma = max(float(np.std(eps)), 1e-6)
        return theta, sigma

    
    @staticmethod
    def _safe_mid(sell_orders: list, buy_orders: list) -> float:
        if len(sell_orders) >= 2 and len(buy_orders) >= 2:
            return (sell_orders[1][0] + buy_orders[1][0]) / 2.0
        return (sell_orders[0][0] + buy_orders[0][0]) / 2.0

    def trade_osmium(
        self,
        order_depth: OrderDepth,
        position: int,
        data: dict,
    ) -> List[Order]:
        orders: List[Order] = []

        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        if not sell_orders or not buy_orders:
            logger.log({
                "product": "ASH_COATED_OSMIUM",
                "position": position,
                "action": "SKIP",
                "reason": "empty_book",
            })
            return orders

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]
        mid = self._safe_mid(sell_orders, buy_orders)

        # -- Restore Kalman state --
        kf_mu = data.get("kf_mu", 0.0)
        kf_beta = data.get("kf_beta", 0.0)
        kf_p = data.get("kf_p", [0.0, 0.0, 0.0, 0.0])
        if not kf_p or len(kf_p) != 4:
            kf_p = [0.0, 0.0, 0.0, 0.0]
        p00, p01, p10, p11 = kf_p

        # -- Kalman update --
        if p00 == p01 == p10 == p11 == 0.0:
            kf_mu = mid
            kf_beta = 0.0
            p00, p01, p10, p11 = 25.0, 0.0, 0.0, 4.0
        else:
            kf_mu, kf_beta, p00, p01, p10, p11 = self._kalman_tick(
                mid, kf_mu, kf_beta,
                p00, p01, p10, p11,
                self.KF_R_OBS, self.KF_Q_LEVEL, self.KF_Q_DRIFT,
            )

        # -- Save Kalman state --
        data["kf_mu"] = kf_mu
        data["kf_beta"] = kf_beta
        data["kf_p"] = [p00, p01, p10, p11]

        fv = kf_mu

        # -- Build residual history for OU --
        residuals: List[float] = data.get("residuals", [])
        residuals.append(mid - fv)
        residuals = residuals[-self.OU_LOOKBACK:]
        data["residuals"] = residuals

        # -- Collect mid history for variance --
        mid_prices: List[float] = data.get("mid_prices", [])
        mid_prices.append(mid)
        mid_prices = mid_prices[-self.OU_LOOKBACK:]
        data["mid_prices"] = mid_prices

        # -- OU estimation --
        if len(residuals) >= self.OU_MIN_OBS:
            ou_theta, ou_sigma = self._estimate_ou(residuals)
        else:
            ou_theta, ou_sigma = 0.0, 1.0

        half_life = math.log(2) / ou_theta if ou_theta > 1e-6 else float("inf")

        # -- Drift signals --
        buy_signal = kf_beta > self.KF_DRIFT_EPS
        sell_signal = kf_beta < -self.KF_DRIFT_EPS

        # -- Variance for A-S spread --
        if len(mid_prices) > 2:
            var = float(np.var(np.diff(mid_prices)))
        else:
            var = 1.0
        var = max(var, 1e-6)

        residual_now = mid - fv
        inv_scale = abs(position) / max(POSITION_LIMIT, 1)

        # Inventory risk: slightly stronger gamma when near limit
        gamma_eff = self.GAMMA * (1.0 + 0.35 * inv_scale)

        # Reservation: inventory skew + mean-reversion tilt toward fv
        skew_ticks = self.RESIDUAL_SKEW_K * residual_now
        skew_ticks = max(-self.RESIDUAL_SKEW_CLIP, min(self.RESIDUAL_SKEW_CLIP, skew_ticks))
        q = position
        r = fv - (q * gamma_eff * var) - skew_ticks

        ou_spread_adj = min(ou_sigma / max(ou_theta, 0.01), self.OU_SPREAD_CAP)
        base_delta = gamma_eff * var + (2.0 / gamma_eff) * math.log(1.0 + gamma_eff / self.K_LIQ)
        delta = base_delta + self.OU_SPREAD_BLEND * ou_spread_adj

        # Widen when price looks extended vs OU noise (adverse-selection guard)
        sigma_eps = max(ou_sigma, 0.5)
        z_res = abs(residual_now) / sigma_eps if len(residuals) >= self.OU_MIN_OBS else 0.0
        if z_res > 1.0:
            delta += self.RESIDUAL_Z_WIDEN * (z_res - 1.0)

        delta = max(delta, float(self.MIN_SPREAD))

        # Tight market: do not quote inside a one-tick book with huge theoretical half-spread
        market_width = float(best_ask - best_bid)
        if market_width <= 1:
            delta = max(delta, float(self.MIN_SPREAD) + 0.5)

        # Asymmetric half-spread: widen side that would add inventory risk
        qn = position / max(POSITION_LIMIT, 1)
        half_bid = (delta / 2.0) * (1.0 + self.INV_SPREAD_ASYM * max(0.0, qn))
        half_ask = (delta / 2.0) * (1.0 + self.INV_SPREAD_ASYM * max(0.0, -qn))

        as_bid = math.floor(r - half_bid)
        as_ask = math.ceil(r + half_ask)

        buy_qty = POSITION_LIMIT - position
        sell_qty = -POSITION_LIMIT - position

        # Tactical: only lift when not already extended vs fv (reduces buying highs / selling lows)
        allow_aggr_buy = buy_signal and residual_now <= self.TACTICAL_MAX_EXT
        allow_aggr_sell = sell_signal and residual_now >= -self.TACTICAL_MAX_EXT

        for ask_price, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_price < fv + self.TAKE_EDGE and allow_aggr_buy:
                take = min(vol, buy_qty)
                if take > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ask_price, take))
                    buy_qty -= take

        for bid_price, bid_vol in buy_orders:
            if bid_price > fv - self.TAKE_EDGE and allow_aggr_sell:
                take = min(bid_vol, -sell_qty)
                if take > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bid_price, -take))
                    sell_qty += take

        # -- Passive MM quotes --
        mm_bid = min(as_bid, best_bid + 1)
        mm_ask = max(as_ask, best_ask - 1)

        if buy_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", int(mm_bid), buy_qty))
        if sell_qty < 0:
            orders.append(Order("ASH_COATED_OSMIUM", int(mm_ask), sell_qty))

        effective_fv = r

        logger.log({
            "product": "ASH_COATED_OSMIUM",
            "position": position,
            "mid": round(mid, 2),
            "kf_mu": round(kf_mu, 4),
            "kf_beta": round(kf_beta, 6),
            "ou_theta": round(ou_theta, 6),
            "ou_sigma": round(ou_sigma, 6),
            "half_life": round(half_life, 2) if half_life < 1e6 else "inf",
            "var": round(var, 6),
            "fv": round(fv, 2),
            "r": round(effective_fv, 2),
            "delta": round(delta, 4),
            "ou_spread_adj": round(ou_spread_adj, 4),
            "delta_final": round(delta, 4),
            "z_resid": round(z_res, 4),
            "skew_ticks": round(skew_ticks, 4),
            "gamma_eff": round(gamma_eff, 6),
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "allow_aggr_buy": allow_aggr_buy,
            "allow_aggr_sell": allow_aggr_sell,
            "n_orders": len(orders),
        })

        return orders

    # =====================================================================
    # Entry point
    # =====================================================================
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        for product in state.order_depths:
            if product == "ASH_COATED_OSMIUM":
                order_depth = state.order_depths[product]
                position = state.position.get(product, 0)
                result[product] = self.trade_osmium(order_depth, position, data)
            else:
                result[product] = []

        trader_data = json.dumps(data)

        log_output = logger.flush(state)
        print(log_output)

        return result, conversions, trader_data
