import json
import math
from typing import Any

import numpy as np
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


ASH = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80
T_MAX = 999_900


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


class Trader:
    # Kalman FV (local linear trend)
    KF_R = 1.0
    KF_Q_LEVEL = 0.30
    KF_Q_DRIFT = 0.015
    KF_P0_LEVEL = 25.0
    KF_P0_DRIFT = 4.0

    # Alpha components
    RESID_WIN = 80
    BETA_WIN = 80
    OFI_WIN = 80
    ALPHA_CLIP = 3.0
    W_RESID = 0.70
    W_BETA = 0.20
    W_OFI = 0.10

    # Alpha health monitor (online IC between alpha(t-1) and return(t))
    IC_WIN = 120
    IC_POS_TH = 0.015
    IC_NEG_TH = -0.010
    LAMBDA_MIN = 0.0
    LAMBDA_MAX = 1.8
    LAMBDA_BASE = 1.0

    # Continuous volatility throttle (risk control, not alpha)
    VAR_ALPHA_FAST = 0.25
    VAR_ALPHA_SLOW = 0.03

    # A-S parameters
    GAMMA_BASE = 0.06
    GAMMA_VOL_MULT = 2.2
    KAPPA = 0.40
    MIN_SPREAD = 2.0
    TAU_MIN = 0.005
    POS_LIMIT_STRESS = 40
    INV_ASYM = 0.30
    EOD_TAU = 0.10
    EOD_INV_PEN = 4.0
    ALPHA_TO_TICKS = 1.4

    @staticmethod
    def _safe_mid(sells: list[tuple[int, int]], buys: list[tuple[int, int]]) -> float:
        if len(sells) >= 2 and len(buys) >= 2:
            return (sells[1][0] + buys[1][0]) / 2.0
        return (sells[0][0] + buys[0][0]) / 2.0

    @staticmethod
    def _microprice(best_bid: int, best_bid_vol: int, best_ask: int, best_ask_vol: int, fallback_mid: float) -> float:
        bid_q = max(0, best_bid_vol)
        ask_q = max(0, -best_ask_vol)
        denom = bid_q + ask_q
        if denom <= 0:
            return fallback_mid
        # Opposite-side weighting: more ask size pulls microprice toward bid (and vice versa).
        return (best_ask * bid_q + best_bid * ask_q) / float(denom)

    @staticmethod
    def _ewma(prev: float | None, value: float, alpha: float) -> float:
        return value if prev is None else alpha * value + (1.0 - alpha) * prev

    @staticmethod
    def _std(x: list[float], floor: float = 1e-6) -> float:
        if len(x) < 5:
            return 1.0
        return max(float(np.std(x)), floor)

    @staticmethod
    def _corr(a: list[float], b: list[float]) -> float:
        if len(a) < 12 or len(a) != len(b):
            return 0.0
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa < 1e-10 or sb < 1e-10:
            return 0.0
        return float(np.corrcoef(np.array(a), np.array(b))[0, 1])

    @staticmethod
    def _kalman_tick(
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
    ) -> tuple[float, float, float, float, float, float]:
        mu_pred = mu + beta
        beta_pred = beta
        pp00 = p00 + p01 + p10 + p11 + q_level
        pp01 = p01 + p11
        pp10 = p10 + p11
        pp11 = p11 + q_drift

        innov = z - mu_pred
        s = pp00 + r_obs
        k0 = pp00 / s
        k1 = pp10 / s

        mu_new = mu_pred + k0 * innov
        beta_new = beta_pred + k1 * innov
        p00_new = pp00 - k0 * pp00
        p01_new = pp01 - k0 * pp01
        p10_new = pp10 - k1 * pp00
        p11_new = pp11 - k1 * pp01
        return mu_new, beta_new, p00_new, p01_new, p10_new, p11_new

    def _trade_ash(self, order_depth: OrderDepth, position: int, data: dict[str, Any], timestamp: int) -> list[Order]:
        orders: list[Order] = []
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        if not sell_orders or not buy_orders:
            return orders

        best_ask, best_ask_vol = sell_orders[0]
        best_bid, best_bid_vol = buy_orders[0]
        mid = self._safe_mid(sell_orders, buy_orders)
        micro = self._microprice(best_bid, best_bid_vol, best_ask, best_ask_vol, mid)

        # Kalman FV update
        kf = data.get("kf")
        if kf is None:
            mu, beta = micro, 0.0
            p00, p01, p10, p11 = self.KF_P0_LEVEL, 0.0, 0.0, self.KF_P0_DRIFT
        else:
            mu, beta, p00, p01, p10, p11 = self._kalman_tick(
                micro,
                kf["mu"],
                kf["b"],
                kf["p00"],
                kf["p01"],
                kf["p10"],
                kf["p11"],
                self.KF_R,
                self.KF_Q_LEVEL,
                self.KF_Q_DRIFT,
            )
        data["kf"] = {"mu": mu, "b": beta, "p00": p00, "p01": p01, "p10": p10, "p11": p11}

        # Feature histories
        residual = micro - mu
        resid_hist = data.get("resid_hist", [])
        beta_hist = data.get("beta_hist", [])
        ofi_hist = data.get("ofi_hist", [])
        ret_hist = data.get("ret_hist", [])
        alpha_hist = data.get("alpha_hist", [])
        realized_hist = data.get("realized_hist", [])

        # OFI proxy from top level book pressure
        ask_q = max(1, -best_ask_vol)
        bid_q = max(1, best_bid_vol)
        ofi = (bid_q - ask_q) / float(bid_q + ask_q)

        last_micro = data.get("last_micro")
        ret = 0.0 if last_micro is None or last_micro <= 0 else math.log(micro / last_micro)
        data["last_micro"] = micro

        resid_hist.append(residual)
        beta_hist.append(beta)
        ofi_hist.append(ofi)
        ret_hist.append(ret)

        resid_hist = resid_hist[-self.RESID_WIN :]
        beta_hist = beta_hist[-self.BETA_WIN :]
        ofi_hist = ofi_hist[-self.OFI_WIN :]
        ret_hist = ret_hist[-self.OFI_WIN :]

        data["resid_hist"] = resid_hist
        data["beta_hist"] = beta_hist
        data["ofi_hist"] = ofi_hist
        data["ret_hist"] = ret_hist

        # Normalized alpha factors
        resid_z = residual / self._std(resid_hist)
        beta_z = beta / self._std(beta_hist)
        ofi_z = ofi / self._std(ofi_hist)

        # Residual mean-reversion, drift-following, order-flow pressure
        alpha_raw = -self.W_RESID * resid_z + self.W_BETA * beta_z + self.W_OFI * ofi_z
        alpha_raw = max(-self.ALPHA_CLIP, min(self.ALPHA_CLIP, alpha_raw))

        # Online IC: correlate alpha(t-1) with realized return(t)
        prev_alpha = data.get("prev_alpha")
        if prev_alpha is not None and last_micro is not None and last_micro > 0:
            alpha_hist.append(prev_alpha)
            realized_hist.append(ret)
            alpha_hist = alpha_hist[-self.IC_WIN :]
            realized_hist = realized_hist[-self.IC_WIN :]
            data["alpha_hist"] = alpha_hist
            data["realized_hist"] = realized_hist
        data["prev_alpha"] = alpha_raw

        ic = self._corr(alpha_hist, realized_hist)

        # Convert IC into directional tilt strength
        if ic <= self.IC_NEG_TH:
            lam = self.LAMBDA_MIN
        elif ic >= self.IC_POS_TH:
            x = min((ic - self.IC_POS_TH) / max(1e-6, 0.05 - self.IC_POS_TH), 1.0)
            lam = self.LAMBDA_BASE + x * (self.LAMBDA_MAX - self.LAMBDA_BASE)
        else:
            lam = self.LAMBDA_BASE * max(0.0, (ic - self.IC_NEG_TH) / (self.IC_POS_TH - self.IC_NEG_TH))

        alpha_shift_ticks = self.ALPHA_TO_TICKS * lam * alpha_raw

        # Continuous volatility throttle (no hard regime switch)
        ret2 = ret * ret
        vf = self._ewma(data.get("v_fast"), ret2, self.VAR_ALPHA_FAST)
        vs = self._ewma(data.get("v_slow"), ret2, self.VAR_ALPHA_SLOW)
        data["v_fast"] = vf
        data["v_slow"] = vs
        vol_ratio = 1.0 if vs <= 1e-12 else vf / vs
        stress = max(0.0, min(1.0, (vol_ratio - 1.0) / 3.0))

        tau = max(1.0 - timestamp / T_MAX, self.TAU_MIN)
        gamma = self.GAMMA_BASE * (1.0 + (self.GAMMA_VOL_MULT - 1.0) * stress)
        sigma2 = max(vf, 1e-6)
        pos_limit = POSITION_LIMIT - (POSITION_LIMIT - self.POS_LIMIT_STRESS) * stress

        # A-S reservation + alpha tilt
        reservation = mu - position * gamma * sigma2 * tau + alpha_shift_ticks
        if tau < self.EOD_TAU:
            urgency = (self.EOD_TAU - tau) / self.EOD_TAU
            reservation -= self.EOD_INV_PEN * urgency * position / POSITION_LIMIT

        spread = gamma * sigma2 * tau + (2.0 / gamma) * math.log(1.0 + gamma / self.KAPPA)
        spread = max(spread, self.MIN_SPREAD)

        qn = position / max(pos_limit, 1.0)
        half_bid = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, qn))
        half_ask = (spread / 2.0) * (1.0 + self.INV_ASYM * max(0.0, -qn))

        quote_bid = math.floor(reservation - half_bid)
        quote_ask = math.ceil(reservation + half_ask)
        quote_bid = min(quote_bid, best_bid + 1)
        quote_ask = max(quote_ask, best_ask - 1)

        buy_cap = int(pos_limit) - position
        sell_cap = -int(pos_limit) - position

        # Small selective taking only when alpha confidence is positive
        if lam > 0.6 and alpha_raw > 0.8 and buy_cap > 0 and best_ask <= quote_bid + 1:
            take = min(-best_ask_vol, buy_cap)
            if take > 0:
                orders.append(Order(ASH, best_ask, take))
                buy_cap -= take
        elif lam > 0.6 and alpha_raw < -0.8 and sell_cap < 0 and best_bid >= quote_ask - 1:
            take = min(best_bid_vol, -sell_cap)
            if take > 0:
                orders.append(Order(ASH, best_bid, -take))
                sell_cap += take

        if buy_cap > 0:
            orders.append(Order(ASH, int(quote_bid), buy_cap))
        if sell_cap < 0:
            orders.append(Order(ASH, int(quote_ask), sell_cap))

        logger.print(
            f"ts={timestamp} mid={mid:.2f} micro={micro:.2f} mu={mu:.2f} res={residual:.3f} "
            f"alpha={alpha_raw:.3f} ic={ic:.4f} lam={lam:.3f} shift={alpha_shift_ticks:.3f} "
            f"stress={stress:.3f} g={gamma:.4f} spr={spread:.3f} "
            f"q={position} lim={pos_limit:.1f} bid={quote_bid} ask={quote_ask}"
        )

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        try:
            trader_data_dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data_dict = {}

        ash_data = trader_data_dict.setdefault("ash", {})

        for symbol, order_depth in state.order_depths.items():
            if symbol == ASH:
                position = state.position.get(symbol, 0)
                result[symbol] = self._trade_ash(order_depth, position, ash_data, state.timestamp)
            else:
                result[symbol] = []

        trader_data = json.dumps(trader_data_dict, separators=(",", ":"))
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
