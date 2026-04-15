"""
Kalman–OU Avellaneda-Stoikov Market Maker for ASH_COATED_OSMIUM.

Architecture (simplified):
  Layer 1 — Kalman Fair Value: Local linear trend filter -> mu(t), beta(t)
  Layer 2 — OU Residual:      r(t) = mid - mu(t); online theta, sigma estimation
  Layer 3 — A-S Quoting:      inventory-aware reservation price and spread

Notes:
  - No volatility regime switch.
  - No tau term (Prosperity simplification).
"""
import math
import json
import numpy as np
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol

PRODUCT = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Logger:
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Trader:
    # ── Layer 1: Kalman filter (local linear trend) ───────────────────────
    KF_R_OBS = 0.75
    KF_Q_LEVEL = 0.5
    KF_Q_DRIFT = 0.015
    KF_INIT_P_LEVEL = 25.0
    KF_INIT_P_DRIFT = 4.0

    # ── Layer 2: OU residual ──────────────────────────────────────────────
    OU_LOOKBACK = 60
    OU_MIN_OBS = 15
    OU_MR_PULL = 0.015
    OU_MR_CLIP = 5.0

    # ── Layer 3: Avellaneda-Stoikov quoting ───────────────────────────────
    VAR_WINDOW = 12               # tutorial lookback
    MIN_VAR_OBS = 12              # wait for full lookback
    GAMMA_BASE = 0.07             # tutorial gamma
    K_ARRIVAL = 0.18              # tutorial k
    T_HORIZON = 1.0               # tutorial T
    MIN_SPREAD = 4                # tutorial floor

    # ── Tactical taking ──────────────────────────────────────────────────
    DRIFT_EPS = 0.04
    TAKE_EDGE = 1.5
    TAKE_MAX_EXT = 1.25

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 1 — Kalman filter tick (pure scalar, no matrix ops)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @staticmethod
    def _kalman_tick(z, mu, beta, p00, p01, p10, p11,
                     r_obs, q_level, q_drift):
        mu_pred = mu + beta
        beta_pred = beta
        pp00 = p00 + p01 + p10 + p11 + q_level
        pp01 = p01 + p11
        pp10 = p10 + p11
        pp11 = p11 + q_drift

        innov = z - mu_pred
        S = pp00 + r_obs
        k0 = pp00 / S
        k1 = pp10 / S

        mu_new = mu_pred + k0 * innov
        beta_new = beta_pred + k1 * innov
        p00_new = pp00 - k0 * pp00
        p01_new = pp01 - k0 * pp01
        p10_new = pp10 - k1 * pp00
        p11_new = pp11 - k1 * pp01

        return mu_new, beta_new, p00_new, p01_new, p10_new, p11_new

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 2 — OU estimation (AR(1) on residuals)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @staticmethod
    def _estimate_ou(residuals):
        r = np.array(residuals, dtype=float)
        if len(r) < 5:
            return 0.0, 1.0
        y, x = r[1:], r[:-1]
        x_mu = float(np.mean(x))
        denom = float(np.sum((x - x_mu) ** 2))
        if denom < 1e-12:
            return 0.0, max(float(np.std(r)), 1e-6)
        phi = float(np.sum((x - x_mu) * (y - np.mean(y))) / denom)
        phi = max(min(phi, 0.9999), -0.9999)
        theta = -math.log(abs(phi)) if abs(phi) > 1e-6 else 0.0
        if phi <= 0:
            theta = abs(theta)
        sigma = max(float(np.std(y - phi * x)), 1e-6)
        return theta, sigma

    @staticmethod
    def _safe_mid(sells, buys):
        if len(sells) >= 2 and len(buys) >= 2:
            return (sells[1][0] + buys[1][0]) / 2.0
        return (sells[0][0] + buys[0][0]) / 2.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Main trading logic
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def trade_osmium(self, order_depth: OrderDepth, position: int,
                     data: dict, timestamp: int) -> List[Order]:
        orders: List[Order] = []

        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        if not sell_orders or not buy_orders:
            return orders

        best_ask = sell_orders[0][0]
        best_bid = buy_orders[0][0]
        mid = self._safe_mid(sell_orders, buy_orders)

        # ── Layer 1: Kalman fair value ────────────────────────────────────
        kf = data.get("kf")
        if kf is None:
            mu, beta = mid, 0.0
            p00 = self.KF_INIT_P_LEVEL
            p01 = p10 = 0.0
            p11 = self.KF_INIT_P_DRIFT
        else:
            mu, beta, p00, p01, p10, p11 = self._kalman_tick(
                mid, kf["mu"], kf["b"],
                kf["p00"], kf["p01"], kf["p10"], kf["p11"],
                self.KF_R_OBS, self.KF_Q_LEVEL, self.KF_Q_DRIFT,
            )
        data["kf"] = {
            "mu": mu, "b": beta,
            "p00": p00, "p01": p01, "p10": p10, "p11": p11,
        }
        fv = mu

        # ── Layer 2: OU residual ──────────────────────────────────────────
        residual = mid - fv
        resids: List[float] = data.get("resids", [])
        resids.append(residual)
        resids = resids[-self.OU_LOOKBACK:]
        data["resids"] = resids

        if len(resids) >= self.OU_MIN_OBS:
            ou_theta, ou_sigma = self._estimate_ou(resids)
        else:
            ou_theta, ou_sigma = 0.0, 1.0
        half_life = math.log(2) / ou_theta if ou_theta > 1e-6 else float("inf")

        # ── Layer 3: A-S quoting inputs (no regime, no tau) ───────────────
        mids: List[float] = data.get("m", [])
        mids.append(mid)
        mids = mids[-(self.VAR_WINDOW + 1):]
        data["m"] = mids
        if len(mids) >= self.MIN_VAR_OBS + 1:
            sigma2 = max(float(np.var(np.diff(mids))), 1e-6)
        else:
            return orders

        gamma = self.GAMMA_BASE
        pos_limit = POSITION_LIMIT

        # OU-adjusted fair value: lean reservation back toward μ(t)
        ou_pull = self.OU_MR_PULL * residual
        ou_pull = max(-self.OU_MR_CLIP, min(self.OU_MR_CLIP, ou_pull))
        s = fv - ou_pull

        # Tutorial A-S reservation: r = s - q*gamma*sigma2*T
        q = position
        reservation = s - q * gamma * sigma2 * self.T_HORIZON

        # Tutorial A-S spread: delta = gamma*sigma2*T + (2/gamma)*ln(1 + gamma/k)
        spread = (gamma * sigma2 * self.T_HORIZON
                  + (2.0 / gamma) * math.log(1.0 + gamma / self.K_ARRIVAL))
        spread = max(spread, float(self.MIN_SPREAD))

        # Tutorial-style symmetric half-spreads
        half_bid = spread / 2.0
        half_ask = spread / 2.0

        as_bid = math.floor(reservation - half_bid)
        as_ask = math.ceil(reservation + half_ask)

        buy_cap = int(pos_limit) - position
        sell_cap = -int(pos_limit) - position

        # ── Tactical taking (drift-gated) ─────────────────────────────────
        buy_signal = beta > self.DRIFT_EPS
        sell_signal = beta < -self.DRIFT_EPS
        allow_buy = buy_signal and residual <= self.TAKE_MAX_EXT
        allow_sell = sell_signal and residual >= -self.TAKE_MAX_EXT

        for ask_px, ask_vol in sell_orders:
            vol = -ask_vol
            if ask_px < fv + self.TAKE_EDGE and allow_buy and buy_cap > 0:
                take = min(vol, buy_cap)
                if take > 0:
                    orders.append(Order(PRODUCT, ask_px, take))
                    buy_cap -= take

        for bid_px, bid_vol in buy_orders:
            if bid_px > fv - self.TAKE_EDGE and allow_sell and sell_cap < 0:
                take = min(bid_vol, -sell_cap)
                if take > 0:
                    orders.append(Order(PRODUCT, bid_px, -take))
                    sell_cap += take

        # ── Passive MM quotes ─────────────────────────────────────────────
        mm_bid = min(as_bid, best_bid + 1)
        mm_ask = max(as_ask, best_ask - 1)

        if buy_cap > 0:
            orders.append(Order(PRODUCT, int(mm_bid), buy_cap))
        if sell_cap < 0:
            orders.append(Order(PRODUCT, int(mm_ask), sell_cap))

        # ── Logging ───────────────────────────────────────────────────────
        logger.log({
            "p": PRODUCT, "ts": timestamp, "pos": position,
            "mid": round(mid, 2),
            "fv": round(fv, 4), "beta": round(beta, 6),
            "ou_th": round(ou_theta, 6), "ou_sig": round(ou_sigma, 4),
            "hl": round(half_life, 2) if half_life < 1e6 else "inf",
            "gamma": round(gamma, 6), "sig2": round(sigma2, 6),
            "res": round(reservation, 4), "sprd": round(spread, 4),
            "bid": as_bid, "ask": as_ask,
            "plim": round(pos_limit, 1),
            "n": len(orders),
        })

        return orders

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Entry point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        for product in state.order_depths:
            if product == PRODUCT:
                position = state.position.get(product, 0)
                result[product] = self.trade_osmium(
                    state.order_depths[product], position, data, state.timestamp,
                )
            else:
                result[product] = []

        trader_data = json.dumps(data)
        log_output = logger.flush(state)
        print(log_output)

        return result, conversions, trader_data
