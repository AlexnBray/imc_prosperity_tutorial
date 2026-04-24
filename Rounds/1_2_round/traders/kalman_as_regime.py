"""
Kalman–OU–Regime Avellaneda-Stoikov Market Maker for ASH_COATED_OSMIUM.

Architecture (4 layers, each feeds the next):
  Layer 1 — Kalman Fair Value:  Local linear trend filter → μ(t), β(t)
  Layer 2 — OU Residual:        r(t) = mid − μ(t); online θ, σ_ou estimation
  Layer 3 — Vol Regime:          max(Δp²) over short window vs slow baseline
  Layer 4 — A-S Quoting:        Regime-scaled reservation price + optimal spread

Data-driven design choices (from Markov switching on historical data):
  - No directional regimes  (both regime means ≈ 0, p > 0.5)
  - Strong volatility regimes (σ²_high / σ²_low ≈ 37×)
  - Regimes are short-lived  (~3-6 ticks) → EWMA is too slow; use raw spike detection
  - Mean-reversion target is Kalman FV, not fixed 10 000
"""
import math
import json
import numpy as np
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol

PRODUCT = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80
T_MAX = 999_900


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
    KF_R_OBS = 1.0
    KF_Q_LEVEL = 0.3
    KF_Q_DRIFT = 0.015
    KF_INIT_P_LEVEL = 25.0
    KF_INIT_P_DRIFT = 4.0

    # ── Layer 2: OU residual ──────────────────────────────────────────────
    OU_LOOKBACK = 60
    OU_MIN_OBS = 15
    OU_MR_PULL = 0.015
    OU_MR_CLIP = 5.0

    # ── Layer 3: Reactive volatility regime ───────────────────────────────
    # Slow EWMA for "normal" baseline variance of Δmid²
    BASELINE_ALPHA = 0.025        # half-life ≈ 27 ticks
    # Short window of raw Δmid² — max over this window is the "current" vol
    SPIKE_WINDOW = 3              # ticks to retain (matches ~3-6 tick regime duration)
    # Linear ramp thresholds: p_hv goes 0 → 1 as worst/baseline goes THRESH → CEIL
    SPIKE_THRESH = 4.0            # ratio to start reacting
    SPIKE_CEIL = 10.0             # ratio for full high-vol

    # ── Layer 4: Avellaneda-Stoikov quoting ───────────────────────────────
    GAMMA_BASE = 0.07
    GAMMA_HIGH_MULT = 2.5         # γ multiplier at p_hv=1
    K_ARRIVAL = 0.35
    TAU_MIN = 0.005
    MIN_SPREAD = 2
    POS_LIMIT_HIGH_VOL = 40
    INV_ASYM_K = 0.30
    # End-of-day inventory urgency (A-S reservation goes flat as τ→0, so
    # we add a τ-independent penalty to actually unwind near close)
    EOD_TAU = 0.10                # last 10% of the day
    EOD_INV_PEN = 3.0             # max ticks of reservation shift at close

    # ── Tactical taking ──────────────────────────────────────────────────
    DRIFT_EPS = 0.04
    TAKE_EDGE = 1.5
    TAKE_MAX_EXT = 1.25
    TAKE_REGIME_GATE = 0.5        # disable taking when p_hv exceeds this

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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 3 — Reactive spike-based regime detection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @staticmethod
    def _ewma(prev, val, alpha):
        return val if prev is None else alpha * val + (1.0 - alpha) * prev

    @staticmethod
    def _spike_regime(worst_var, baseline_var, thresh, ceil):
        """Linear ramp: 0 when ratio <= thresh, 1 when ratio >= ceil."""
        if baseline_var < 1e-15:
            return 0.0
        ratio = worst_var / baseline_var
        if ratio <= thresh:
            return 0.0
        if ratio >= ceil:
            return 1.0
        return (ratio - thresh) / (ceil - thresh)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

        # ── Layer 3: Reactive volatility regime ───────────────────────────
        last_mid = data.get("lm")
        recent_dp2: List[float] = data.get("rdp", [])
        if last_mid is not None:
            dp2 = (mid - last_mid) ** 2
            baseline_var = self._ewma(data.get("bv"), dp2, self.BASELINE_ALPHA)
            recent_dp2.append(dp2)
            recent_dp2 = recent_dp2[-self.SPIKE_WINDOW:]
        else:
            baseline_var = data.get("bv", 1.0)
        data["bv"] = baseline_var
        data["rdp"] = recent_dp2
        data["lm"] = mid

        worst_var = max(recent_dp2) if recent_dp2 else baseline_var
        p_hv = self._spike_regime(
            worst_var, baseline_var, self.SPIKE_THRESH, self.SPIKE_CEIL,
        )

        # ── Layer 4: Avellaneda-Stoikov quoting ───────────────────────────

        tau = max(1.0 - timestamp / T_MAX, self.TAU_MIN)

        # Regime-blended risk aversion
        gamma = self.GAMMA_BASE * (1.0 + (self.GAMMA_HIGH_MULT - 1.0) * p_hv)

        # Regime-blended variance: interpolate between baseline and spike
        sigma2 = max((1.0 - p_hv) * baseline_var + p_hv * worst_var, 1e-6)

        # Regime-scaled position limit
        pos_limit = POSITION_LIMIT - (POSITION_LIMIT - self.POS_LIMIT_HIGH_VOL) * p_hv

        # OU-adjusted fair value: lean reservation back toward μ(t)
        ou_pull = self.OU_MR_PULL * residual
        ou_pull = max(-self.OU_MR_CLIP, min(self.OU_MR_CLIP, ou_pull))
        s = fv - ou_pull

        # A-S reservation price:  r = s − q·γ·σ²·τ
        q = position
        reservation = s - q * gamma * sigma2 * tau

        # End-of-day inventory urgency: A-S reservation penalty vanishes as
        # τ→0, so add a τ-independent term that actively unwinds near close
        if tau < self.EOD_TAU:
            urgency = (self.EOD_TAU - tau) / self.EOD_TAU  # 0→1 as day ends
            eod_shift = self.EOD_INV_PEN * urgency * q / POSITION_LIMIT
            reservation -= eod_shift

        # A-S optimal spread:  δ = γσ²τ + (2/γ)·ln(1 + γ/κ)
        spread = (gamma * sigma2 * tau
                  + (2.0 / gamma) * math.log(1.0 + gamma / self.K_ARRIVAL))
        spread = max(spread, float(self.MIN_SPREAD))

        # Asymmetric half-spreads: widen the side that adds inventory risk
        qn = q / max(pos_limit, 1)
        half_bid = (spread / 2.0) * (1.0 + self.INV_ASYM_K * max(0.0, qn))
        half_ask = (spread / 2.0) * (1.0 + self.INV_ASYM_K * max(0.0, -qn))

        as_bid = math.floor(reservation - half_bid)
        as_ask = math.ceil(reservation + half_ask)

        buy_cap = int(pos_limit) - position
        sell_cap = -int(pos_limit) - position

        # ── Tactical taking (drift-gated, regime-gated) ──────────────────
        buy_signal = beta > self.DRIFT_EPS
        sell_signal = beta < -self.DRIFT_EPS
        regime_safe = p_hv < self.TAKE_REGIME_GATE
        allow_buy = buy_signal and regime_safe and residual <= self.TAKE_MAX_EXT
        allow_sell = sell_signal and regime_safe and residual >= -self.TAKE_MAX_EXT

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
            "p_hv": round(p_hv, 4),
            "worst_v": round(worst_var, 4), "base_v": round(baseline_var, 6),
            "gamma": round(gamma, 6), "sig2": round(sigma2, 6),
            "tau": round(tau, 4),
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
