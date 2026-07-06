"""Parametric test bed for the Round 5 multi-agent stat-arb stack.

This file is loaded by ``rust_backtester`` like ``trader.py`` is, but every
configuration field is overridable through environment variables. It is the
trader entry point used by ``bayesian_optimisation.py`` for Optuna sweeps.

Selecting an agent
------------------
Set ``TARGET_AGENT`` to one of:

* ``SPIKE_MR``      — Run *only* :class:`SpikeMRTrader` on the symbols listed
                      in ``SPIKE_MR_SYMBOLS`` (CSV).
* ``STEP_MR``       — Run *only* :class:`StepMeanReversionTrader` (chunked
                      sum/count mean + σ-snap; same logic as ``clean_trader15``).
* ``EWMA_MR``       — Run *only* :class:`EwmaMeanReversionTrader` (fair = EWMA of
                      mid; z vs rolling σ of ``mid − fair``).
* ``BASKET``        — Run *only* :class:`BasketTrader` (PEBBLES basket).
* ``COMPLEX_PAIR``  — Run *only* :class:`ComplexPairTrader` (SNACKPACK).
* ``LEAD_LAG``      — Run *only* :class:`LeadLagTrader` on configured pairs.
* ``ALL`` (default) — Run every agent that has at least one symbol enabled.

Per-agent overrides
-------------------
STEP_MR (clean_trader15-style stepping mean MR; tune via bayesian_optimisation):
    STEP_MR_SYMBOLS=SNACKPACK_CHOCOLATE,...
    STEP_MR_VAR_WINDOW, STEP_MR_STEP_LONG_TICKS, STEP_MR_STEP_SHORT_TICKS,
    STEP_MR_K_SD, STEP_MR_SIGMA_FLOOR, STEP_MR_Z_BUY, STEP_MR_Z_SELL

EWMA_MR (single-name MR with exponentially weighted fair value):
    EWMA_MR_SYMBOLS=PRODUCT,...
    EWMA_MR_VAR_WINDOW, EWMA_MR_ALPHA, EWMA_MR_Z_BUY, EWMA_MR_Z_SELL

SPIKE_MR:
    SPIKE_MR_SYMBOLS=ROBOT_DISHES,ROBOT_IRONING,...
    SPIKE_MR_VAR_WINDOW   (int)
    SPIKE_MR_Z_IN         (float)
    SPIKE_MR_Z_EXIT       (float)
    SPIKE_MR_Z_STOP       (float)
    SPIKE_MR_TIME_STOP    (int)
    SPIKE_MR_TARGET_SIZE  (int)

BASKET:
    BASKET_VAR_WINDOW, BASKET_Z_IN, BASKET_Z_EXIT, BASKET_Z_STOP,
    BASKET_TARGET_SIZE,
    BASKET_ALPHA_XS, BASKET_ALPHA_S, BASKET_ALPHA_M, BASKET_ALPHA_L

COMPLEX_PAIR:
    CPAIR_TARGET_SIZE, CPAIR_VAR_WINDOW,
    CPAIR_Z_IN, CPAIR_Z_EXIT, CPAIR_Z_STOP,
    CPAIR_BETA_VC, CPAIR_BETA_SR
    (CPAIR_BETA_SP, CPAIR_BETA_RP are read if the optional pairs are enabled
    via CPAIR_INCLUDE_SP=1 / CPAIR_INCLUDE_RP=1 — disabled by default to
    avoid factor double-counting; see plan.md §1B2.)

LEAD_LAG:
    LEAD_LAG_PAIRS=FOLLOWER:LEADER:BETA:GATE,FOLLOWER:LEADER:BETA:GATE,...

All other configuration fields fall back to the defaults baked into the
production ``trader.py``.
"""
from __future__ import annotations

import base64
import json
import math
import os
import struct
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import trader as production
from trader import (
    BasketConfig,
    BasketLegConfig,
    BasketTrader,
    ComplexPairConfig,
    ComplexPairTrader,
    LeadLagConfig,
    LeadLagTrader,
    PairLink,
    ProductBaseConfig,
    SpikeMRConfig,
    SpikeMRTrader,
    TraderBase as ProductionTraderBase,
    _safe_load_trader_data,
)
from datamodel import TradingState


# ─────────────────────────────────────────────────────────────────────
#  ENV HELPERS
# ─────────────────────────────────────────────────────────────────────
def _env(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


TARGET_AGENT = os.getenv("TARGET_AGENT", "ALL").strip().upper() or "ALL"


# ─────────────────────────────────────────────────────────────────────
#  STEP MEAN REVERSION  (literal port of clean_trader15 MeanReversionTrader step path)
# ─────────────────────────────────────────────────────────────────────
def _step_mr_leg_short(sym: str) -> str:
    try:
        cpx = getattr(production, "SNACKPACK_COMPLEX_CONFIG", None)
        if cpx is not None and getattr(cpx, "leg_short_codes", None):
            return cpx.leg_short_codes.get(sym, sym[:4].upper())
    except Exception:
        pass
    return sym[:4].upper()


@dataclass
class StepMeanReversionConfig(ProductBaseConfig):
    """Config for :class:`StepMeanReversionTrader` — mirrors clean ``MeanReversionConfig`` step fields."""

    z_buy: float = 2.5
    z_sell: float = 2.5
    mean: float = 0.0
    step_long_ticks: int = 1000
    step_short_ticks: int = 80
    step_k_sd: float = 5.0
    step_sigma_floor: float = 1e-3


class StepMeanReversionTrader(ProductionTraderBase):
    """Same stepping mean logic as ``clean_trader15.MeanReversionTrader`` (``_step_mode`` branch)."""

    @staticmethod
    def _pack_buf(buf) -> str:
        if not buf:
            return ""
        scaled = [max(-32768, min(32767, round(v * 2))) for v in buf]
        return base64.b64encode(struct.pack(f">{len(scaled)}h", *scaled)).decode()

    @staticmethod
    def _unpack_buf(s: str) -> list:
        if not s:
            return []
        raw = base64.b64decode(s)
        return [v / 2.0 for v in struct.unpack(f">{len(raw) // 2}h", raw)]

    def __init__(
        self,
        name: str,
        state: TradingState,
        new_trader_data: dict,
        last_traderData: dict,
        cfg: StepMeanReversionConfig,
    ):
        super().__init__(name, state, new_trader_data, last_traderData, cfg)
        self.cfg: StepMeanReversionConfig = cfg
        self.z_buy_threshold = cfg.z_buy
        self.z_sell_threshold = cfg.z_sell
        self.mean = cfg.mean

        self._init_variance_state(cfg.short_code, cfg.var_window)

        lc = cfg.step_long_ticks
        sc = cfg.step_short_ticks
        if lc > 0 and sc <= 0:
            sc = max(1, lc // 40)
        self._step_mode = lc > 0 and sc > 0
        self._step_short_ticks = sc
        self._step_long_ticks = lc
        prefix = f"{cfg.short_code}_sms_"
        self._k_ls_sum = prefix + "ls_sum"
        self._k_ls_n = prefix + "ls_n"
        self._k_ss_sum = prefix + "ss_sum"
        self._k_ss_n = prefix + "ss_n"
        lt = last_traderData if last_traderData else {}
        self._ls_sum = float(lt.get(self._k_ls_sum, 0.0))
        self._ls_n = int(lt.get(self._k_ls_n, 0))
        self._ss_sum = float(lt.get(self._k_ss_sum, 0.0))
        self._ss_n = int(lt.get(self._k_ss_n, 0))

    def _init_variance_state(self, prefix: str, window: int) -> None:
        self.window = window
        self._hk = f"{prefix}h"
        self._sxk = f"{prefix}sx"
        self._s2k = f"{prefix}s2"
        lt = self.last_traderData if self.last_traderData else {}
        raw = lt.get(self._hk, "") or ""
        self._buf = deque(self._unpack_buf(raw) if raw else [], maxlen=self.window)
        self._sum_x = float(lt.get(self._sxk, 0.0))
        self._sum_x2 = float(lt.get(self._s2k, 0.0))

    def _calc_var(self, value: float) -> float:
        buf = self._buf
        if len(buf) == self.window:
            evicted = buf[0]
            self._sum_x -= evicted
            self._sum_x2 -= evicted * evicted

        buf.append(value)
        self._sum_x += value
        self._sum_x2 += value * value
        n = len(buf)

        self.new_trader_data[self._hk] = self._pack_buf(buf)
        self.new_trader_data[self._sxk] = self._sum_x
        self.new_trader_data[self._s2k] = self._sum_x2

        if n < 2:
            return 1e-8

        var = (self._sum_x2 - (self._sum_x * self._sum_x) / n) / (n - 1)
        self.var = max(var, 1e-8)
        return self.var

    def _persist_step_state(self) -> None:
        self.new_trader_data[self._k_ls_sum] = self._ls_sum
        self.new_trader_data[self._k_ls_n] = self._ls_n
        self.new_trader_data[self._k_ss_sum] = self._ss_sum
        self.new_trader_data[self._k_ss_n] = self._ss_n

    def _step_mean_effective(self, mid: float) -> tuple:
        lc, sc = self._step_long_ticks, self._step_short_ticks
        self._ls_sum += mid
        self._ls_n += 1
        self._ss_sum += mid
        self._ss_n += 1

        live_long = self._ls_sum / self._ls_n
        live_short = self._ss_sum / self._ss_n

        cfg = self.cfg
        var = self._calc_var(mid - live_long)
        sigma = max(math.sqrt(var), cfg.step_sigma_floor)
        k_sd = cfg.step_k_sd

        if abs(live_short - live_long) > k_sd * sigma:
            mu_eff = live_short
        else:
            mu_eff = live_long

        if self._ls_n >= lc:
            self._ls_sum = 0.0
            self._ls_n = 0
        if self._ss_n >= sc:
            self._ss_sum = 0.0
            self._ss_n = 0

        self._persist_step_state()
        return mu_eff, sigma

    def get_orders(self) -> Dict[str, List]:
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        if self._step_mode:
            ref, sigma = self._step_mean_effective(self.wall_mid)
            z = (self.wall_mid - ref) / sigma
        else:
            dev = self.wall_mid - self.mean
            var = self._calc_var(dev)
            z = dev / math.sqrt(var)

        if z > 0 and abs(z) > self.z_sell_threshold:
            self.ask(self.best_bid, self.max_allowed_sell_volume)
        if z < 0 and abs(z) > self.z_buy_threshold:
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        return {self.name: self.orders}


@dataclass
class EwmaMeanReversionConfig(ProductBaseConfig):
    """Fair value = ``α·mid + (1−α)·fv_prev``; trade z-score of ``mid − fair``."""

    z_buy: float = 2.5
    z_sell: float = 2.5
    alpha: float = 0.08


class EwmaMeanReversionTrader(ProductionTraderBase):
    """EWMA fair with rolling variance on the *deviation* ``mid − fair`` (clean_trader15-style)."""

    def __init__(
        self,
        name: str,
        state: TradingState,
        new_trader_data: dict,
        last_traderData: dict,
        cfg: EwmaMeanReversionConfig,
    ):
        super().__init__(name, state, new_trader_data, last_traderData, cfg)
        self.cfg: EwmaMeanReversionConfig = cfg
        self.z_buy_threshold = cfg.z_buy
        self.z_sell_threshold = cfg.z_sell
        self.alpha = float(cfg.alpha)
        self._init_variance_state(cfg.short_code, cfg.var_window)
        pv = f"{cfg.short_code}ew_pv"
        self._k_pv = pv
        self._fv_prev = float(last_traderData.get(pv, 0.0)) if last_traderData else 0.0

    def _init_variance_state(self, prefix: str, window: int) -> None:
        self.window = window
        self._hk = f"{prefix}h"
        self._sxk = f"{prefix}sx"
        self._s2k = f"{prefix}s2"
        lt = self.last_traderData if self.last_traderData else {}
        raw = lt.get(self._hk, "") or ""
        self._buf = deque(StepMeanReversionTrader._unpack_buf(raw) if raw else [], maxlen=self.window)
        self._sum_x = float(lt.get(self._sxk, 0.0))
        self._sum_x2 = float(lt.get(self._s2k, 0.0))

    def _calc_var(self, value: float) -> float:
        buf = self._buf
        if len(buf) == self.window:
            evicted = buf[0]
            self._sum_x -= evicted
            self._sum_x2 -= evicted * evicted

        buf.append(value)
        self._sum_x += value
        self._sum_x2 += value * value
        n = len(buf)

        self.new_trader_data[self._hk] = StepMeanReversionTrader._pack_buf(buf)
        self.new_trader_data[self._sxk] = self._sum_x
        self.new_trader_data[self._s2k] = self._sum_x2

        if n < 2:
            return 1e-8

        var = (self._sum_x2 - (self._sum_x * self._sum_x) / n) / (n - 1)
        self.var = max(var, 1e-8)
        return self.var

    def _update_fair(self, mid: float) -> float:
        """EWMA(level) anchored on first observation."""
        fv_prev = self._fv_prev
        if fv_prev == 0.0:
            fair = mid
        else:
            fair = self.alpha * mid + (1.0 - self.alpha) * fv_prev
        self._fv_prev = fair
        self.new_trader_data[self._k_pv] = fair
        return fair

    def _compute_z(self, value: float, fair: float) -> float:
        dev = value - fair
        var = self._calc_var(dev)
        return dev / math.sqrt(var)

    def get_orders(self) -> Dict[str, List]:
        if self.wall_mid is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        mid = float(self.wall_mid)
        fair = self._update_fair(mid)
        z = self._compute_z(mid, fair)

        if z > 0 and abs(z) > self.z_sell_threshold:
            self.ask(self.best_bid, self.max_allowed_sell_volume)
        if z < 0 and abs(z) > self.z_buy_threshold:
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        return {self.name: self.orders}


def _build_step_mr_configs() -> Dict[str, StepMeanReversionConfig]:
    if TARGET_AGENT != "STEP_MR":
        return {}

    defaults = getattr(production, "SNACKPACK_COMPLEX_CONFIG", None)
    fallback_pl = (
        defaults.legs["SNACKPACK_CHOCOLATE"]
        if defaults and getattr(defaults, "legs", None) and "SNACKPACK_CHOCOLATE" in defaults.legs
        else 10
    )

    symbols = _env_csv("STEP_MR_SYMBOLS", ["SNACKPACK_CHOCOLATE"])
    vw = _env("STEP_MR_VAR_WINDOW", 500, int)
    lng = max(1, _env("STEP_MR_STEP_LONG_TICKS", 1000, int))
    sht = max(1, _env("STEP_MR_STEP_SHORT_TICKS", 80, int))
    ksd = _env("STEP_MR_K_SD", 5.0, float)
    fl = _env("STEP_MR_SIGMA_FLOOR", 1e-3, float)
    zb = _env("STEP_MR_Z_BUY", 2.5, float)
    zs = _env("STEP_MR_Z_SELL", 2.5, float)

    pos_ov_raw = os.getenv("STEP_MR_POS_LIMIT")
    pos_ov: Optional[int] = None
    if pos_ov_raw is not None and str(pos_ov_raw).strip() != "":
        try:
            pos_ov = int(str(pos_ov_raw).strip())
        except ValueError:
            pos_ov = None

    out: Dict[str, StepMeanReversionConfig] = {}
    for sym in symbols:
        if pos_ov is not None:
            plim = pos_ov
        elif defaults and getattr(defaults, "legs", None) and sym in defaults.legs:
            plim = int(defaults.legs[sym])
        else:
            plim = int(fallback_pl)

        out[sym] = StepMeanReversionConfig(
            symbol=sym,
            short_code=_step_mr_leg_short(sym),
            pos_limit=plim,
            var_window=int(vw),
            z_buy=float(zb),
            z_sell=float(zs),
            mean=0.0,
            step_long_ticks=int(lng),
            step_short_ticks=int(sht),
            step_k_sd=float(ksd),
            step_sigma_floor=float(fl),
        )
    return out


STEP_MR_CONFIGS = _build_step_mr_configs()


def _build_ewma_mr_configs() -> Dict[str, EwmaMeanReversionConfig]:
    if TARGET_AGENT != "EWMA_MR":
        return {}

    defaults = getattr(production, "SNACKPACK_COMPLEX_CONFIG", None)
    fallback_pl = (
        defaults.legs["SNACKPACK_CHOCOLATE"]
        if defaults and getattr(defaults, "legs", None) and "SNACKPACK_CHOCOLATE" in defaults.legs
        else 10
    )

    symbols = _env_csv("EWMA_MR_SYMBOLS", ["SNACKPACK_CHOCOLATE"])
    vw = _env("EWMA_MR_VAR_WINDOW", 500, int)
    alb = float(_env("EWMA_MR_ALPHA", 0.08, float))
    zb = _env("EWMA_MR_Z_BUY", 2.5, float)
    zs = _env("EWMA_MR_Z_SELL", 2.5, float)

    pos_ov_raw = os.getenv("EWMA_MR_POS_LIMIT")
    pos_ov: Optional[int] = None
    if pos_ov_raw is not None and str(pos_ov_raw).strip() != "":
        try:
            pos_ov = int(str(pos_ov_raw).strip())
        except ValueError:
            pos_ov = None

    out: Dict[str, EwmaMeanReversionConfig] = {}
    for sym in symbols:
        if pos_ov is not None:
            plim = pos_ov
        elif defaults and getattr(defaults, "legs", None) and sym in defaults.legs:
            plim = int(defaults.legs[sym])
        else:
            plim = int(fallback_pl)

        out[sym] = EwmaMeanReversionConfig(
            symbol=sym,
            short_code=_step_mr_leg_short(sym),
            pos_limit=plim,
            var_window=int(vw),
            z_buy=float(zb),
            z_sell=float(zs),
            alpha=alb,
        )
    return out


EWMA_MR_CONFIGS = _build_ewma_mr_configs()


# ─────────────────────────────────────────────────────────────────────
#  AGENT A — SPIKE MR  (overrideable per run)
# ─────────────────────────────────────────────────────────────────────
def _build_spike_mr_configs() -> Dict[str, SpikeMRConfig]:
    if TARGET_AGENT not in {"SPIKE_MR", "ALL"}:
        return {}

    default_symbols = list(production.SPIKE_MR_CONFIGS.keys())
    symbols = _env_csv("SPIKE_MR_SYMBOLS", default_symbols)

    def _short(sym: str) -> str:
        return production.SPIKE_MR_CONFIGS[sym].short_code if sym in production.SPIKE_MR_CONFIGS \
            else sym[:4].upper()

    var_window  = _env("SPIKE_MR_VAR_WINDOW", None, int)
    z_in        = _env("SPIKE_MR_Z_IN", None, float)
    z_exit      = _env("SPIKE_MR_Z_EXIT", None, float)
    z_stop      = _env("SPIKE_MR_Z_STOP", None, float)
    time_stop   = _env("SPIKE_MR_TIME_STOP", None, int)
    target_size = _env("SPIKE_MR_TARGET_SIZE", None, int)
    pos_limit   = _env("SPIKE_MR_POS_LIMIT", None, int)

    out: Dict[str, SpikeMRConfig] = {}
    for sym in symbols:
        base = production.SPIKE_MR_CONFIGS.get(sym)
        if base is not None:
            out[sym] = SpikeMRConfig(
                symbol=sym,
                short_code=base.short_code,
                pos_limit=pos_limit if pos_limit is not None else base.pos_limit,
                var_window=var_window if var_window is not None else base.var_window,
                z_in=z_in if z_in is not None else base.z_in,
                z_exit=z_exit if z_exit is not None else base.z_exit,
                z_stop=z_stop if z_stop is not None else base.z_stop,
                time_stop=time_stop if time_stop is not None else base.time_stop,
                target_size=target_size if target_size is not None else base.target_size,
            )
        else:
            # Symbol not in production registry — fall back to global env defaults.
            out[sym] = SpikeMRConfig(
                symbol=sym,
                short_code=_short(sym),
                pos_limit=pos_limit if pos_limit is not None else 10,
                var_window=var_window if var_window is not None else 120,
                z_in=z_in if z_in is not None else 2.5,
                z_exit=z_exit if z_exit is not None else 0.4,
                z_stop=z_stop if z_stop is not None else 4.5,
                time_stop=time_stop if time_stop is not None else 30,
                target_size=target_size if target_size is not None else 8,
            )
    return out


# ─────────────────────────────────────────────────────────────────────
#  AGENT B-1 — BASKET  (overrideable alphas, thresholds, window)
# ─────────────────────────────────────────────────────────────────────
def _build_basket_config() -> BasketConfig | None:
    if TARGET_AGENT not in {"BASKET", "ALL"}:
        return None

    base = production.PEBBLES_BASKET_CONFIG

    alpha_overrides = {
        "PEBBLES_XS": _env("BASKET_ALPHA_XS", None, float),
        "PEBBLES_S":  _env("BASKET_ALPHA_S",  None, float),
        "PEBBLES_M":  _env("BASKET_ALPHA_M",  None, float),
        "PEBBLES_L":  _env("BASKET_ALPHA_L",  None, float),
    }
    new_hedges = []
    for hedge in base.hedges:
        ov = alpha_overrides.get(hedge.symbol)
        new_hedges.append(BasketLegConfig(
            symbol=hedge.symbol,
            short_code=hedge.short_code,
            alpha=ov if ov is not None else hedge.alpha,
            pos_limit=hedge.pos_limit,
        ))

    return BasketConfig(
        name=base.name,
        short_code=base.short_code,
        anchor=base.anchor,
        hedges=new_hedges,
        var_window=_env("BASKET_VAR_WINDOW", base.var_window, int),
        z_in=_env("BASKET_Z_IN", base.z_in, float),
        z_exit=_env("BASKET_Z_EXIT", base.z_exit, float),
        z_stop=_env("BASKET_Z_STOP", base.z_stop, float),
        target_size=_env("BASKET_TARGET_SIZE", base.target_size, int),
        deleverage_threshold=_env("BASKET_DELEVERAGE", base.deleverage_threshold, float),
    )


# ─────────────────────────────────────────────────────────────────────
#  AGENT B-2 — COMPLEX PAIR (overrideable per-pair beta + threshold set)
# ─────────────────────────────────────────────────────────────────────
def _build_complex_pair_config() -> ComplexPairConfig | None:
    if TARGET_AGENT not in {"COMPLEX_PAIR", "ALL"}:
        return None

    base = production.SNACKPACK_COMPLEX_CONFIG

    common_z_in    = _env("CPAIR_Z_IN",    base.pairs[0].z_in,    float)
    common_z_exit  = _env("CPAIR_Z_EXIT",  base.pairs[0].z_exit,  float)
    common_z_stop  = _env("CPAIR_Z_STOP",  base.pairs[0].z_stop,  float)
    common_window  = _env("CPAIR_VAR_WINDOW", base.pairs[0].var_window, int)
    target_size    = _env("CPAIR_TARGET_SIZE", base.target_size, int)
    deleverage     = _env("CPAIR_DELEVERAGE", base.deleverage_threshold, float)

    beta_overrides = {
        "VC": _env("CPAIR_BETA_VC", None, float),
        "SR": _env("CPAIR_BETA_SR", None, float),
        "SP": _env("CPAIR_BETA_SP", None, float),
        "RP": _env("CPAIR_BETA_RP", None, float),
    }
    include_sp = _env_bool("CPAIR_INCLUDE_SP", default=False)
    include_rp = _env_bool("CPAIR_INCLUDE_RP", default=False)

    pairs: List[PairLink] = []
    enabled_codes = {"VC", "SR"}
    if include_sp:
        enabled_codes.add("SP")
    if include_rp:
        enabled_codes.add("RP")

    # Use base pairs as templates for VC and SR; build SP/RP from defaults.
    base_by_code = {p.short_code: p for p in base.pairs}
    template_defaults = {
        "VC": ("SNACKPACK_VANILLA",    "SNACKPACK_CHOCOLATE", -0.883),
        "SR": ("SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY", -0.875),
        "SP": ("SNACKPACK_STRAWBERRY", "SNACKPACK_PISTACHIO", +1.260),
        "RP": ("SNACKPACK_RASPBERRY",  "SNACKPACK_PISTACHIO", -1.211),
    }
    for code in enabled_codes:
        sym1, sym2, beta_default = template_defaults[code]
        existing = base_by_code.get(code)
        if existing is not None:
            sym1, sym2 = existing.sym1, existing.sym2
            beta_default = existing.beta
        beta = beta_overrides[code]
        pairs.append(PairLink(
            sym1=sym1, sym2=sym2,
            beta=beta if beta is not None else beta_default,
            z_in=common_z_in, z_exit=common_z_exit, z_stop=common_z_stop,
            var_window=common_window, short_code=code,
        ))

    if not pairs:
        return None

    return ComplexPairConfig(
        name=base.name,
        short_code=base.short_code,
        legs=dict(base.legs),
        leg_short_codes=dict(base.leg_short_codes),
        pairs=pairs,
        target_size=target_size,
        deleverage_threshold=deleverage,
    )


# ─────────────────────────────────────────────────────────────────────
#  AGENT C — LEAD-LAG  (env-driven pair list; off by default)
# ─────────────────────────────────────────────────────────────────────
def _build_lead_lag_configs() -> Dict[str, LeadLagConfig]:
    if TARGET_AGENT not in {"LEAD_LAG", "ALL"}:
        return {}

    raw = os.getenv("LEAD_LAG_PAIRS", "").strip()
    if not raw:
        return {}

    pos_limit   = _env("LEAD_LAG_POS_LIMIT", 10, int)
    var_window  = _env("LEAD_LAG_VAR_WINDOW", 50, int)
    target_size = _env("LEAD_LAG_TARGET_SIZE", 4, int)

    out: Dict[str, LeadLagConfig] = {}
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        parts = piece.split(":")
        if len(parts) != 4:
            continue
        follower, leader, beta_s, gate_s = parts
        try:
            beta, gate = float(beta_s), float(gate_s)
        except ValueError:
            continue
        out[follower] = LeadLagConfig(
            symbol=follower,
            short_code=follower[:4].upper(),
            pos_limit=pos_limit,
            var_window=var_window,
            leader=leader,
            follower=follower,
            beta=beta,
            gate=gate,
            target_size=target_size,
            enabled=True,
        )
    return out


# ─────────────────────────────────────────────────────────────────────
#  REGISTRIES (built once per process)
# ─────────────────────────────────────────────────────────────────────
SPIKE_MR_CONFIGS = _build_spike_mr_configs()
BASKET_CONFIG = _build_basket_config()
COMPLEX_PAIR_CONFIG = _build_complex_pair_config()
LEAD_LAG_CONFIGS = _build_lead_lag_configs()


# ─────────────────────────────────────────────────────────────────────
#  TRADER
# ─────────────────────────────────────────────────────────────────────
class Trader:
    def run(self, state: TradingState):
        result: Dict[str, list] = {}
        new_trader_data: dict = {}
        last_traderData = _safe_load_trader_data(getattr(state, "traderData", "") or "")

        if TARGET_AGENT == "STEP_MR":
            for sym, cfg in STEP_MR_CONFIGS.items():
                if sym not in state.order_depths:
                    continue
                try:
                    sm = StepMeanReversionTrader(sym, state, new_trader_data, last_traderData, cfg)
                    for s, orders in sm.get_orders().items():
                        if orders:
                            result.setdefault(s, []).extend(orders)
                except Exception as exc:
                    print(f"[testbed/StepMR] {sym}: {exc!r}")
            try:
                final_trader_data = json.dumps(new_trader_data, separators=(",", ":"))
            except Exception:
                final_trader_data = ""
            return result, 0, final_trader_data

        if TARGET_AGENT == "EWMA_MR":
            for sym, cfg in EWMA_MR_CONFIGS.items():
                if sym not in state.order_depths:
                    continue
                try:
                    em = EwmaMeanReversionTrader(sym, state, new_trader_data, last_traderData, cfg)
                    for s, orders in em.get_orders().items():
                        if orders:
                            result.setdefault(s, []).extend(orders)
                except Exception as exc:
                    print(f"[testbed/EwmaMR] {sym}: {exc!r}")
            try:
                final_trader_data = json.dumps(new_trader_data, separators=(",", ":"))
            except Exception:
                final_trader_data = ""
            return result, 0, final_trader_data

        for sym, cfg in SPIKE_MR_CONFIGS.items():
            if sym not in state.order_depths:
                continue
            try:
                trader_obj = SpikeMRTrader(sym, state, new_trader_data, last_traderData, cfg)
                for s, orders in trader_obj.get_orders().items():
                    if orders:
                        result.setdefault(s, []).extend(orders)
            except Exception as exc:
                print(f"[testbed/SpikeMR] {sym}: {exc!r}")

        if BASKET_CONFIG is not None:
            try:
                basket = BasketTrader(BASKET_CONFIG, state, new_trader_data, last_traderData)
                for s, orders in basket.get_orders().items():
                    if orders:
                        result.setdefault(s, []).extend(orders)
            except Exception as exc:
                print(f"[testbed/Basket]: {exc!r}")

        if COMPLEX_PAIR_CONFIG is not None:
            try:
                cpx = ComplexPairTrader(COMPLEX_PAIR_CONFIG, state, new_trader_data, last_traderData)
                for s, orders in cpx.get_orders().items():
                    if orders:
                        result.setdefault(s, []).extend(orders)
            except Exception as exc:
                print(f"[testbed/ComplexPair]: {exc!r}")

        for sym, cfg in LEAD_LAG_CONFIGS.items():
            if sym not in state.order_depths:
                continue
            try:
                trader_obj = LeadLagTrader(sym, state, new_trader_data, last_traderData, cfg)
                for s, orders in trader_obj.get_orders().items():
                    if orders:
                        result.setdefault(s, []).extend(orders)
            except Exception as exc:
                print(f"[testbed/LeadLag] {sym}: {exc!r}")

        try:
            final_trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        except Exception:
            final_trader_data = ""
        return result, 0, final_trader_data
