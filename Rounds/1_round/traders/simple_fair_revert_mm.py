"""
Full stacked ASH market maker in one file:
1) Fair value (fixed 10k, optional Kalman level model)
2) Residual mean-reversion alpha (AR(1)/OU z-score + half-life)
3) Volatility regime for risk (EWMA fast/slow)
4) Micro-alpha overlay (residual MR + drift beta + OFI + microprice pressure)
5) Quote tilt (A-S reservation + lambda * alpha)
6) Alpha health monitor (rolling IC, hit-rate, slope)
"""

import json
import math
from typing import Dict, List, Tuple

import numpy as np
from datamodel import OrderDepth, Order, Symbol, TradingState


ASH = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80
T_MAX = 999_900


class Trader:
    # 1) Fair value
    USE_KALMAN = False
    FAIR_10K = 10_000.0
    FV_CORR_ALPHA = 0.02
    FV_CORR_CLIP = 100.0
    KF_R = 1.0
    KF_Q_LVL = 0.20
    KF_Q_BETA = 0.01
    KF_P0_LVL = 25.0
    KF_P0_BETA = 4.0

    # 2) Residual MR alpha (OU/AR1)
    RES_WIN = 120
    RES_Z_CLIP = 4.0
    OU_MIN = 20
    W_RES = 0.65

    # 3) Vol regime risk (EWMA fast/slow)
    VAR_FAST_A = 0.30
    VAR_SLOW_A = 0.04
    GAMMA_BASE = 0.05
    GAMMA_STRESS_MULT = 2.5
    POS_LIMIT_STRESS = 35

    # 4) Micro alpha overlay
    W_BETA = 0.15
    W_OFI = 0.10
    W_MICRO = 0.10
    FAC_WIN = 120
    ALPHA_CLIP = 4.0

    # 5) A-S + quote tilt
    KAPPA = 0.40
    MIN_SPREAD = 2.0
    TAU_MIN = 0.005
    INV_ASYM = 0.30
    ALPHA_TO_TICKS = 1.3
    TAKE_ALPHA_TH = 1.0
    TAKE_QTY = 10
    EOD_TAU = 0.10
    EOD_INV_PEN = 4.0

    # 6) Alpha health monitor
    HEALTH_WIN = 150
    IC_ON = 0.01
    IC_OFF = -0.01
    LAM_MIN = 0.0
    LAM_MAX = 1.8

    @staticmethod
    def _safe_mid(sells: list[tuple[int, int]], buys: list[tuple[int, int]]) -> float:
        if len(sells) >= 2 and len(buys) >= 2:
            return (sells[1][0] + buys[1][0]) / 2.0
        return (sells[0][0] + buys[0][0]) / 2.0

    @staticmethod
    def _microprice(best_bid: int, best_bid_vol: int, best_ask: int, best_ask_vol: int, fallback: float) -> float:
        bid_q = max(0, best_bid_vol)
        ask_q = max(0, -best_ask_vol)
        denom = bid_q + ask_q
        if denom <= 0:
            return fallback
        return (best_ask * bid_q + best_bid * ask_q) / float(denom)

    @staticmethod
    def _ewma(prev: float | None, x: float, a: float) -> float:
        return x if prev is None else a * x + (1.0 - a) * prev

    @staticmethod
    def _std(x: list[float], floor: float = 1e-6) -> float:
        return max(float(np.std(x)), floor) if len(x) >= 6 else 1.0

    @staticmethod
    def _corr(a: list[float], b: list[float]) -> float:
        if len(a) < 20 or len(a) != len(b):
            return 0.0
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        return float(np.corrcoef(np.array(a), np.array(b))[0, 1])

    @staticmethod
    def _slope(x: list[float], y: list[float]) -> float:
        if len(x) < 20 or len(x) != len(y):
            return 0.0
        xm = float(np.mean(x))
        denom = float(np.sum((np.array(x) - xm) ** 2))
        if denom < 1e-12:
            return 0.0
        ym = float(np.mean(y))
        return float(np.sum((np.array(x) - xm) * (np.array(y) - ym)) / denom)

    @staticmethod
    def _kalman_tick(z: float, mu: float, beta: float, p00: float, p01: float, p10: float, p11: float,
                     r: float, q_lvl: float, q_beta: float) -> tuple[float, float, float, float, float, float]:
        mu_pred = mu + beta
        beta_pred = beta
        pp00 = p00 + p01 + p10 + p11 + q_lvl
        pp01 = p01 + p11
        pp10 = p10 + p11
        pp11 = p11 + q_beta

        innov = z - mu_pred
        s = pp00 + r
        k0 = pp00 / s
        k1 = pp10 / s

        mu_new = mu_pred + k0 * innov
        beta_new = beta_pred + k1 * innov
        p00_new = pp00 - k0 * pp00
        p01_new = pp01 - k0 * pp01
        p10_new = pp10 - k1 * pp00
        p11_new = pp11 - k1 * pp01
        return mu_new, beta_new, p00_new, p01_new, p10_new, p11_new

    @staticmethod
    def _ou_estimate(res: list[float]) -> tuple[float, float]:
        r = np.array(res, dtype=float)
        if len(r) < 8:
            return 0.0, 1.0
        x = r[:-1]
        y = r[1:]
        xm = float(np.mean(x))
        denom = float(np.sum((x - xm) ** 2))
        if denom < 1e-12:
            return 0.0, max(float(np.std(r)), 1e-6)
        phi = float(np.sum((x - xm) * (y - float(np.mean(y)))) / denom)
        phi = max(min(phi, 0.9999), -0.9999)
        theta = -math.log(max(abs(phi), 1e-6))
        sigma = max(float(np.std(y - phi * x)), 1e-6)
        return theta, sigma

    def _fair_value(self, micro: float, data: dict) -> tuple[float, float]:
        if self.USE_KALMAN:
            kf = data.get("kf")
            if kf is None:
                mu, beta = micro, 0.0
                p00, p01, p10, p11 = self.KF_P0_LVL, 0.0, 0.0, self.KF_P0_BETA
            else:
                mu, beta, p00, p01, p10, p11 = self._kalman_tick(
                    micro, kf["mu"], kf["b"], kf["p00"], kf["p01"], kf["p10"], kf["p11"],
                    self.KF_R, self.KF_Q_LVL, self.KF_Q_BETA,
                )
            data["kf"] = {"mu": mu, "b": beta, "p00": p00, "p01": p01, "p10": p10, "p11": p11}
            return mu, beta

        corr = self._ewma(data.get("fv_corr"), micro - self.FAIR_10K, self.FV_CORR_ALPHA)
        corr = max(-self.FV_CORR_CLIP, min(self.FV_CORR_CLIP, corr))
        data["fv_corr"] = corr
        return self.FAIR_10K + corr, 0.0

    def _trade_ash(self, order_depth: OrderDepth, position: int, data: dict, timestamp: int) -> list[Order]:
        orders: list[Order] = []
        sells = sorted(order_depth.sell_orders.items())
        buys = sorted(order_depth.buy_orders.items(), reverse=True)
        if not sells or not buys:
            return orders

        best_ask, best_ask_vol = sells[0]
        best_bid, best_bid_vol = buys[0]
        mid = self._safe_mid(sells, buys)
        micro = self._microprice(best_bid, best_bid_vol, best_ask, best_ask_vol, mid)

        fv, beta = self._fair_value(micro, data)
        residual = micro - fv

        # Residual / factor histories
        res_hist = data.get("res_hist", [])
        ofi_hist = data.get("ofi_hist", [])
        micro_dev_hist = data.get("micro_dev_hist", [])
        beta_hist = data.get("beta_hist", [])
        alpha_hist = data.get("alpha_hist", [])
        ret_hist = data.get("ret_hist", [])

        bid_q = max(1, best_bid_vol)
        ask_q = max(1, -best_ask_vol)
        ofi = (bid_q - ask_q) / float(bid_q + ask_q)
        micro_dev = micro - mid

        last_micro = data.get("last_micro")
        ret = 0.0 if last_micro is None or last_micro <= 0 else math.log(micro / last_micro)
        data["last_micro"] = micro

        res_hist.append(residual)
        ofi_hist.append(ofi)
        micro_dev_hist.append(micro_dev)
        beta_hist.append(beta)
        res_hist = res_hist[-self.RES_WIN:]
        ofi_hist = ofi_hist[-self.FAC_WIN:]
        micro_dev_hist = micro_dev_hist[-self.FAC_WIN:]
        beta_hist = beta_hist[-self.FAC_WIN:]
        data["res_hist"] = res_hist
        data["ofi_hist"] = ofi_hist
        data["micro_dev_hist"] = micro_dev_hist
        data["beta_hist"] = beta_hist

        # 2) Residual MR alpha + OU diagnostics
        theta, _sigma_ou = self._ou_estimate(res_hist) if len(res_hist) >= self.OU_MIN else (0.0, 1.0)
        hl = (math.log(2.0) / theta) if theta > 1e-6 else float("inf")
        res_z = max(-self.RES_Z_CLIP, min(self.RES_Z_CLIP, residual / self._std(res_hist)))
        alpha_res = -res_z

        # 4) Micro-alpha overlay
        alpha_beta = beta / self._std(beta_hist)
        alpha_ofi = ofi / self._std(ofi_hist)
        alpha_micro = micro_dev / self._std(micro_dev_hist)
        alpha = self.W_RES * alpha_res + self.W_BETA * alpha_beta + self.W_OFI * alpha_ofi + self.W_MICRO * alpha_micro
        alpha = max(-self.ALPHA_CLIP, min(self.ALPHA_CLIP, alpha))

        # 6) Health monitor (IC, hit-rate, slope)
        prev_alpha = data.get("prev_alpha")
        if prev_alpha is not None:
            alpha_hist.append(prev_alpha)
            ret_hist.append(ret)
            alpha_hist = alpha_hist[-self.HEALTH_WIN:]
            ret_hist = ret_hist[-self.HEALTH_WIN:]
            data["alpha_hist"] = alpha_hist
            data["ret_hist"] = ret_hist
        data["prev_alpha"] = alpha

        ic = self._corr(alpha_hist, ret_hist)
        hit = float(np.mean([(a * r) > 0 for a, r in zip(alpha_hist, ret_hist)])) if len(alpha_hist) >= 20 else 0.5
        slope = self._slope(alpha_hist, ret_hist)

        # map health metrics to lambda
        raw_health = 0.50 + 8.0 * ic + 1.5 * (hit - 0.5) + 50.0 * slope
        raw_health = max(0.0, min(1.0, raw_health))
        if ic <= self.IC_OFF:
            lam = self.LAM_MIN
        elif ic >= self.IC_ON:
            lam = self.LAM_MIN + raw_health * (self.LAM_MAX - self.LAM_MIN)
        else:
            x = (ic - self.IC_OFF) / max(1e-6, self.IC_ON - self.IC_OFF)
            lam = (self.LAM_MIN + raw_health * (self.LAM_MAX - self.LAM_MIN)) * x

        # 3) Volatility regime from EWMA fast/slow
        r2 = ret * ret
        v_fast = self._ewma(data.get("v_fast"), r2, self.VAR_FAST_A)
        v_slow = self._ewma(data.get("v_slow"), r2, self.VAR_SLOW_A)
        data["v_fast"] = v_fast
        data["v_slow"] = v_slow
        ratio = 1.0 if v_slow <= 1e-12 else v_fast / v_slow
        stress = max(0.0, min(1.0, (ratio - 1.0) / 3.0))

        # 5) A-S reservation + quote tilt
        tau = max(1.0 - timestamp / T_MAX, self.TAU_MIN)
        gamma = self.GAMMA_BASE * (1.0 + (self.GAMMA_STRESS_MULT - 1.0) * stress)
        sigma2 = max(v_fast, 1e-6)
        pos_limit = POSITION_LIMIT - (POSITION_LIMIT - self.POS_LIMIT_STRESS) * stress

        alpha_shift = self.ALPHA_TO_TICKS * lam * alpha
        reservation = fv - position * gamma * sigma2 * tau + alpha_shift

        if tau < self.EOD_TAU:
            urgency = (self.EOD_TAU - tau) / self.EOD_TAU
            reservation -= self.EOD_INV_PEN * urgency * position / POSITION_LIMIT

        spread = gamma * sigma2 * tau + (2.0 / gamma) * math.log(1.0 + gamma / self.KAPPA)
        spread = max(spread, self.MIN_SPREAD)

        qn = position / max(pos_limit, 1.0)
        hb = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, qn))
        ha = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, -qn))
        bid = math.floor(reservation - hb)
        ask = math.ceil(reservation + ha)
        bid = min(bid, best_bid + 1)
        ask = max(ask, best_ask - 1)

        buy_cap = int(pos_limit) - position
        sell_cap = -int(pos_limit) - position

        # optional taker action when strong alpha agrees with residual direction
        if lam > 0.5 and alpha > self.TAKE_ALPHA_TH and residual < 0 and buy_cap > 0:
            q = min(-best_ask_vol, buy_cap, self.TAKE_QTY)
            if q > 0:
                orders.append(Order(ASH, best_ask, q))
                buy_cap -= q
        elif lam > 0.5 and alpha < -self.TAKE_ALPHA_TH and residual > 0 and sell_cap < 0:
            q = min(best_bid_vol, -sell_cap, self.TAKE_QTY)
            if q > 0:
                orders.append(Order(ASH, best_bid, -q))
                sell_cap += q

        if buy_cap > 0:
            orders.append(Order(ASH, int(bid), buy_cap))
        if sell_cap < 0:
            orders.append(Order(ASH, int(ask), sell_cap))

        print(json.dumps({
            "ts": timestamp, "mid": round(mid, 2), "micro": round(micro, 2), "fv": round(fv, 2),
            "res": round(residual, 3), "z": round(res_z, 3), "hl": "inf" if hl > 1e6 else round(hl, 2),
            "alpha": round(alpha, 3), "lam": round(lam, 3), "ic": round(ic, 4),
            "hit": round(hit, 3), "slope": round(slope, 6),
            "stress": round(stress, 3), "spr": round(spread, 3), "q": position,
            "bid": int(bid), "ask": int(ask), "n": len(orders),
        }))

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        ash_data = data.setdefault("ash", {})

        for product, order_depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == ASH:
                result[product] = self._trade_ash(order_depth, pos, ash_data, state.timestamp)
            else:
                result[product] = []

        trader_data = json.dumps(data, separators=(",", ":"))
        return result, conversions, trader_data
