"""Optuna driver for the Round 5 multi-agent stat-arb stack.

The driver runs ``rust_backtester`` against ``test_bed.py``, varying that
file's environment-variable overrides per trial, and scores a *composite*
objective (range, dispersion, and worst day — see env vars below) across the
requested days. Choose which agent to tune with the ``TARGET_AGENT``
constant below (or via the ``TARGET_AGENT`` environment variable):

* ``SPIKE_MR``      — tune Agent A on a single symbol (``SPIKE_MR_SYMBOL``).
* ``BASKET``        — tune Agent B-1 (PEBBLES residual: alphas + thresholds).
* ``COMPLEX_PAIR``  — tune Agent B-2 (SNACKPACK pair betas + thresholds).
* ``LEAD_LAG``      — tune Agent C on a configured leader/follower pair.
* ``STEP_MR``       — tune stepping mean MR (clean_trader15-style).
                      Set ``STEP_MR_SYMBOL`` for one product, **or**
                      ``STEP_MR_SNACKPACK_ALL=1`` to optimise snackpack legs
                      (one study each by default, or **linked legs** —
                      ``STEP_MR_BATCH_LINK_GROUPS``, e.g. ``CHOCOLATE+VANILLA`` —
                      shares one hyperparameter vector across a pair in ``test_bed``’s
                      CSV ``STEP_MR_SYMBOLS``).

                      Dedicated SQLite file ``optuna_step_mr_snackpack.db`` unless overridden.

                      Or ``STEP_MR_GALAXY_ALL=1`` to optimise **all** ``GALAXY_SOUNDS_*``
                      legs (SQLite ``optuna_step_mr_galaxy.db`` by default).

* ``EWMA_MR``       — tune single-name MR with EWMA fair (``test_bed`` only),
                      same objective as ``STEP_MR``. Set ``EWMA_MR_SYMBOL``.

* ``MR_UNIVERSE_ALL=1`` — run **both** ``STEP_MR`` and ``EWMA_MR`` Optuna studies
                      for **every** Round 5 product (50 names), write a comparison
                      JSON (see ``run_mr_universe_comparison_batch``). Heavy: set
                      ``OPT_N_TRIALS`` and ``MR_UNIVERSE_ONLY`` for scoped runs.

The composite score maximised by Optuna is (weights from env with defaults that
prefer **stable** day curves — big swings across days hurt the score):

    score = Σ day_pnl − OPT_STABILITY_PENALTY·(max_day − min_day)
            − OPT_DAY_STDDEV_PENALTY·stdev(days)
            − OPT_NEG_DAY_LOSS_WEIGHT·max(0, −min_day)
            − OPT_CV_PENALTY·(stdev(days) / max(|mean_day|, floor))

Optional ``OPT_CV_PENALTY`` (defaults to ``0`` except snackpack-batch setdefaults)
adds scale-free dispersion vs average daily profit.

Very different PnL day-to-day is a practical **sign of overfitting** to one
session’s microstructure; the penalties down-weight “lottery” trials that
only work on a subset of days. Set any weight to ``0`` to disable that term.

``STEP_MR_SNACKPACK_ALL`` runs apply **stricter** defaults if those variables
are unset (see ``run_step_mr_snackpack_batch``).

The objective is computed across every day in ``DAYS_TO_TEST`` and parsed from
``rust_backtester`` summary output (one ``D+<set>   <day>   …   <pnl>   …`` row per
day). By default **one** subprocess runs **all** dataset days (``--day`` omitted).
Set ``OPT_BT_SEPARATE_DAY_RUNS=1`` for the older mode (one subprocess per day).

By default each trial writes backtest output under a **temporary** directory
(``--artifact-mode none`` = small ``metrics.json`` only) and deletes it after
parsing PnL, so ``prosperity_rust_backtester/runs/`` is not filled during sweeps.
Set ``OPT_BT_KEEP_RUNS_OUTPUT=1`` to restore the old behaviour (logs under the
Rust default ``runs/`` tree — can exhaust disk on large studies).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import statistics
import subprocess
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import optuna
from sqlalchemy import create_engine, text
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRADER_PATH  = "/home/dansp/projects/imc_prosperity_tutorial/Rounds/5_round/Traders/test_bed.py"
DATASET_PATH = "/home/dansp/projects/imc_prosperity_tutorial/prosperity_rust_backtester/datasets/round5"

SNACKPACK_STEP_MR_SYMBOLS = [
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_RASPBERRY",
]

GALAXY_STEP_MR_SYMBOLS = [
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
]

STEP_MR_BATCH_DB = os.getenv("STEP_MR_BATCH_DB", "optuna_step_mr_snackpack.db")
STEP_MR_BATCH_DB_PATH = f"sqlite:///{STEP_MR_BATCH_DB}?timeout=60"

STEP_MR_GALAXY_DB = os.getenv("STEP_MR_GALAXY_DB", "optuna_step_mr_galaxy.db")
STEP_MR_GALAXY_DB_PATH = f"sqlite:///{STEP_MR_GALAXY_DB}?timeout=60"

DAYS_TO_TEST = [2, 3, 4]

N_TRIALS = int(os.getenv("OPT_N_TRIALS", "200"))
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)
# Default objective: high total PnL but penalise volatile / one-sided day curves.
# Literal fallbacks — objective reads getenv each trial so batch setdefault wins in workers.
STABILITY_PENALTY = float(os.getenv("OPT_STABILITY_PENALTY", "1.0"))
DAY_STDDEV_PENALTY_DEFAULT = "0.6"
NEG_DAY_LOSS_WEIGHT_DEFAULT = "0.45"

# Pick the agent whose parameters this run will tune.
TARGET_AGENT = os.getenv("TARGET_AGENT", "SPIKE_MR").strip().upper()

# Used by SPIKE_MR / LEAD_LAG / STEP_MR symbol selection.
SPIKE_MR_SYMBOL = os.getenv("SPIKE_MR_SYMBOL", "ROBOT_DISHES").strip()
STEP_MR_SYMBOL = os.getenv("STEP_MR_SYMBOL", "SNACKPACK_CHOCOLATE").strip()
EWMA_MR_SYMBOL = os.getenv("EWMA_MR_SYMBOL", "SNACKPACK_CHOCOLATE").strip()

LEAD_LAG_FOLLOWER = os.getenv("LEAD_LAG_FOLLOWER", "").strip()
LEAD_LAG_LEADER   = os.getenv("LEAD_LAG_LEADER", "").strip()

# Full Round 5 product universe (IMC Prosperity 2025 Round 5, days 2–4 dataset).
ROUND5_ALL_MR_PRODUCTS: tuple[str, ...] = (
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "PANEL_1X2",
    "PANEL_1X4",
    "PANEL_2X2",
    "PANEL_2X4",
    "PANEL_4X4",
    "PEBBLES_L",
    "PEBBLES_M",
    "PEBBLES_S",
    "PEBBLES_XL",
    "PEBBLES_XS",
    "ROBOT_DISHES",
    "ROBOT_IRONING",
    "ROBOT_LAUNDRY",
    "ROBOT_MOPPING",
    "ROBOT_VACUUMING",
    "SLEEP_POD_COTTON",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_SUEDE",
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_VANILLA",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_VOID_BLUE",
    "UV_VISOR_AMBER",
    "UV_VISOR_MAGENTA",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
)

MR_UNIVERSE_DB = os.getenv("MR_UNIVERSE_DB", "optuna_mr_universe.db")
MR_UNIVERSE_DB_PATH = f"sqlite:///{MR_UNIVERSE_DB}?timeout=60"

DB_FILE = "optuna_parallel.db"
DB_PATH = f"sqlite:///{DB_FILE}?timeout=60"
STUDY_NAME = os.getenv("OPT_STUDY_NAME", f"r5_{TARGET_AGENT.lower()}_opt")


# Summary table rows: ``D+<set>   <day>   <ticks>   <own_trades>   <pnl>   <run_dir>``
# (``run_dir`` may be ``runs/...`` or a temp path when using isolated backtest output).
_PNL_SUMMARY_ROW_RE = re.compile(
    r"^D\+\d+\s+(\d+)\s+\d+\s+\d+\s+([\-\d,.]+)\s+\S",
    re.MULTILINE,
)


def _parse_day_pnls_from_stdout(stdout: str, wanted_days: list[int]) -> list[float] | None:
    """Return PnL for each day in ``wanted_days`` order from multiday backtester output."""
    want = list(wanted_days)
    want_set = frozenset(want)
    by_day: dict[int, float] = {}
    for day_s, pnl_s in _PNL_SUMMARY_ROW_RE.findall(stdout):
        d = int(day_s)
        if d not in want_set:
            continue
        if d in by_day:
            return None
        try:
            by_day[d] = float(pnl_s.replace(",", ""))
        except ValueError:
            return None
    out = []
    for d in want:
        if d not in by_day:
            return None
        out.append(by_day[d])
    return out


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────────
#  AGENT-SPECIFIC SEARCH SPACES + ENV BUILDERS
# ─────────────────────────────────────────────────────────────────────
def _spike_mr_params(trial: optuna.Trial) -> dict:
    return {
        "TARGET_AGENT": "SPIKE_MR",
        "SPIKE_MR_SYMBOLS": SPIKE_MR_SYMBOL,
        "SPIKE_MR_VAR_WINDOW":  str(trial.suggest_int("var_window", 40, 300)),
        "SPIKE_MR_Z_IN":        str(trial.suggest_float("z_in", 1.5, 4.0, step=0.1)),
        "SPIKE_MR_Z_EXIT":      str(trial.suggest_float("z_exit", 0.0, 1.2, step=0.05)),
        "SPIKE_MR_Z_STOP":      str(trial.suggest_float("z_stop", 3.0, 7.0, step=0.25)),
        "SPIKE_MR_TIME_STOP":   str(trial.suggest_int("time_stop", 5, 100)),
        "SPIKE_MR_TARGET_SIZE": str(trial.suggest_int("target_size", 2, 10)),
    }


def _basket_params(trial: optuna.Trial) -> dict:
    return {
        "TARGET_AGENT": "BASKET",
        "BASKET_VAR_WINDOW":   str(trial.suggest_int("var_window", 80, 400)),
        "BASKET_Z_IN":         str(trial.suggest_float("z_in", 1.2, 4.0, step=0.1)),
        "BASKET_Z_EXIT":       str(trial.suggest_float("z_exit", 0.0, 1.2, step=0.05)),
        "BASKET_Z_STOP":       str(trial.suggest_float("z_stop", 3.0, 8.0, step=0.25)),
        "BASKET_TARGET_SIZE":  str(trial.suggest_int("target_size", 2, 10)),
        "BASKET_ALPHA_XS":     str(trial.suggest_float("alpha_xs", 0.20, 0.90, step=0.02)),
        "BASKET_ALPHA_S":      str(trial.suggest_float("alpha_s",  0.30, 1.10, step=0.02)),
        "BASKET_ALPHA_M":      str(trial.suggest_float("alpha_m",  0.40, 1.20, step=0.02)),
        "BASKET_ALPHA_L":      str(trial.suggest_float("alpha_l",  0.40, 1.20, step=0.02)),
    }


def _complex_pair_params(trial: optuna.Trial) -> dict:
    env = {
        "TARGET_AGENT":         "COMPLEX_PAIR",
        "CPAIR_VAR_WINDOW":     str(trial.suggest_int("var_window", 80, 400)),
        "CPAIR_Z_IN":           str(trial.suggest_float("z_in", 1.2, 4.0, step=0.1)),
        "CPAIR_Z_EXIT":         str(trial.suggest_float("z_exit", 0.0, 1.2, step=0.05)),
        "CPAIR_Z_STOP":         str(trial.suggest_float("z_stop", 3.0, 8.0, step=0.25)),
        "CPAIR_TARGET_SIZE":    str(trial.suggest_int("target_size", 1, 6)),
        "CPAIR_BETA_VC":        str(trial.suggest_float("beta_vc", -1.10, -0.50, step=0.01)),
        "CPAIR_BETA_SR":        str(trial.suggest_float("beta_sr", -1.10, -0.50, step=0.01)),
    }
    # Optionally enable the SP / RP pairs and tune their betas. Disabled by
    # default in the trader because they double-count the strawberry triangle
    # (see plan.md §1B2). Uncomment below to expose them to Optuna.
    # env["CPAIR_INCLUDE_SP"] = "1"
    # env["CPAIR_BETA_SP"] = str(trial.suggest_float("beta_sp", 0.80, 1.60, step=0.01))
    return env


def _lead_lag_params(trial: optuna.Trial) -> dict:
    if not LEAD_LAG_FOLLOWER or not LEAD_LAG_LEADER:
        raise ValueError("Set LEAD_LAG_FOLLOWER and LEAD_LAG_LEADER env vars before optimising LEAD_LAG.")
    beta = trial.suggest_float("beta", -2.0, 2.0, step=0.05)
    gate = trial.suggest_float("gate_bps", 1.0, 50.0, step=1.0) / 1e4   # gate in fractional return
    return {
        "TARGET_AGENT": "LEAD_LAG",
        "LEAD_LAG_PAIRS":       f"{LEAD_LAG_FOLLOWER}:{LEAD_LAG_LEADER}:{beta}:{gate}",
        "LEAD_LAG_VAR_WINDOW":  str(trial.suggest_int("var_window", 20, 200)),
        "LEAD_LAG_TARGET_SIZE": str(trial.suggest_int("target_size", 2, 8)),
    }


def _step_mr_params_for_symbol(symbol: str, trial: optuna.Trial) -> dict:
    return {
        "TARGET_AGENT": "STEP_MR",
        "STEP_MR_SYMBOLS": symbol.strip(),
        "STEP_MR_VAR_WINDOW": str(trial.suggest_int("var_window", 100, 800)),
        "STEP_MR_STEP_LONG_TICKS": str(trial.suggest_int("step_long_ticks", 200, 2500)),
        "STEP_MR_STEP_SHORT_TICKS": str(trial.suggest_int("step_short_ticks", 20, 200)),
        "STEP_MR_K_SD": str(trial.suggest_float("step_k_sd", 2.0, 12.0, step=0.25)),
        "STEP_MR_SIGMA_FLOOR": str(trial.suggest_float("sigma_floor", 1e-4, 0.05, log=True)),
        "STEP_MR_Z_BUY": str(trial.suggest_float("z_buy", 1.0, 4.5, step=0.1)),
        "STEP_MR_Z_SELL": str(trial.suggest_float("z_sell", 1.0, 4.5, step=0.1)),
    }


def _step_mr_params(trial: optuna.Trial) -> dict:
    return _step_mr_params_for_symbol(STEP_MR_SYMBOL, trial)


def _ewma_mr_params_for_symbol(symbol: str, trial: optuna.Trial) -> dict:
    return {
        "TARGET_AGENT": "EWMA_MR",
        "EWMA_MR_SYMBOLS": symbol.strip(),
        "EWMA_MR_VAR_WINDOW": str(trial.suggest_int("var_window", 50, 800)),
        "EWMA_MR_ALPHA": str(trial.suggest_float("alpha", 0.008, 0.45, log=True)),
        "EWMA_MR_Z_BUY": str(trial.suggest_float("z_buy", 1.0, 4.5, step=0.1)),
        "EWMA_MR_Z_SELL": str(trial.suggest_float("z_sell", 1.0, 4.5, step=0.1)),
    }


def _ewma_mr_params(trial: optuna.Trial) -> dict:
    return _ewma_mr_params_for_symbol(EWMA_MR_SYMBOL, trial)


SEARCH_SPACES = {
    "SPIKE_MR":     _spike_mr_params,
    "BASKET":       _basket_params,
    "COMPLEX_PAIR": _complex_pair_params,
    "LEAD_LAG":     _lead_lag_params,
    "STEP_MR":      _step_mr_params,
    "EWMA_MR":      _ewma_mr_params,
}


# ─────────────────────────────────────────────────────────────────────
#  SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────
def run_simulation(env_overrides: dict) -> dict | None:
    """Execute ``rust_backtester`` across ``DAYS_TO_TEST`` days and aggregate PnL.

    By default one subprocess runs **all** days in the dataset (omit ``--day``).

    Unless ``OPT_BT_KEEP_RUNS_OUTPUT`` is set, uses a temp ``--output-root`` and
    ``--artifact-mode none``, then removes that tree so sweeps do not flood
    ``runs/``.

    ``OPT_BT_SEPARATE_DAY_RUNS=1`` forces one subprocess per day (legacy).
    """
    base_env = os.environ.copy()
    base_env.update(env_overrides)

    keep_runs = _env_truthy("OPT_BT_KEEP_RUNS_OUTPUT")
    tmp_root: str | None = None
    if not keep_runs:
        tmp_root = tempfile.mkdtemp(prefix="r5_opt_bt_")

    # Wall-clock per trial: multiday Rust run scales ~linearly with day count.
    _to = max(300, 120 * max(2, len(DAYS_TO_TEST)))

    try:
        day_pnls: list[float] | None

        if _env_truthy("OPT_BT_SEPARATE_DAY_RUNS"):
            accumulated: list[float] = []
            for day in DAYS_TO_TEST:
                env = base_env.copy()
                env["DAY"] = str(day)
                cmd = [
                    "rust_backtester",
                    "--trader", TRADER_PATH,
                    "--dataset", DATASET_PATH,
                    "--day", str(day),
                    "--products", "off",
                ]
                if tmp_root is not None:
                    run_id = f"opt_p{os.getpid()}_d{day}_{secrets.token_hex(4)}"
                    cmd.extend(
                        [
                            "--artifact-mode", "none",
                            "--output-root", tmp_root,
                            "--run-id", run_id,
                        ]
                    )
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True, env=env, timeout=180
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    return None
                dp = _parse_day_pnls_from_stdout(result.stdout, [day])
                if dp is None or len(dp) != 1:
                    return None
                accumulated.append(dp[0])
            day_pnls = accumulated
        else:
            cmd = [
                "rust_backtester",
                "--trader", TRADER_PATH,
                "--dataset", DATASET_PATH,
                "--products", "off",
            ]
            if tmp_root is not None:
                run_id = f"opt_p{os.getpid()}_multi_{secrets.token_hex(6)}"
                cmd.extend(
                    [
                        "--artifact-mode", "none",
                        "--output-root", tmp_root,
                        "--run-id", run_id,
                    ]
                )
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True, env=base_env, timeout=_to
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None
            day_pnls = _parse_day_pnls_from_stdout(result.stdout, DAYS_TO_TEST)

        if not day_pnls:
            return None
        pnl_spread = max(day_pnls) - min(day_pnls) if len(day_pnls) > 1 else 0.0
        pnl_std = statistics.pstdev(day_pnls) if len(day_pnls) > 1 else 0.0
        mean_day_pnl = statistics.fmean(day_pnls)
        min_day_pnl = min(day_pnls)
        return {
            "final_pnl":     sum(day_pnls),
            "mean_day_pnl":  mean_day_pnl,
            "pnl_spread":    pnl_spread,
            "pnl_std":       pnl_std,
            "min_day_pnl":   min_day_pnl,
            "day_pnls":      day_pnls,
        }
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────
#  OPTUNA OBJECTIVE
# ─────────────────────────────────────────────────────────────────────
def _objective_agent_kind(trial: optuna.Trial) -> str:
    """Batch studies set ``optimize_agent`` so worker processes don’t rely on a stale ``TARGET_AGENT``."""
    raw = trial.study.user_attrs.get("optimize_agent")
    if raw is not None and str(raw).strip():
        return str(raw).strip().upper()
    return TARGET_AGENT


def _mr_csv_for_step(trial: optuna.Trial) -> str:
    ua = trial.study.user_attrs
    if ua.get("mr_focus_symbol"):
        return str(ua["mr_focus_symbol"]).strip()
    raw = ua.get("step_mr_symbol", STEP_MR_SYMBOL)
    return str(raw).strip()


def _mr_csv_for_ewma(trial: optuna.Trial) -> str:
    ua = trial.study.user_attrs
    if ua.get("mr_focus_symbol"):
        return str(ua["mr_focus_symbol"]).strip()
    raw = ua.get("ewma_mr_symbol", EWMA_MR_SYMBOL)
    return str(raw).strip()


def objective(trial: optuna.Trial) -> float:
    trial.set_user_attr("final_pnl", 0.0)
    trial.set_user_attr("mean_day_pnl", 0.0)
    trial.set_user_attr("pnl_spread", 0.0)
    trial.set_user_attr("pnl_std", 0.0)
    trial.set_user_attr("min_day_pnl", 0.0)
    trial.set_user_attr("day_pnls", [])
    trial.set_user_attr("objective_score", 0.0)

    agent = _objective_agent_kind(trial)

    if agent == "STEP_MR":
        mr_csv = _mr_csv_for_step(trial)
        trial.set_user_attr("step_mr_csv", mr_csv)
        env_overrides = _step_mr_params_for_symbol(mr_csv, trial)
    elif agent == "EWMA_MR":
        mr_csv = _mr_csv_for_ewma(trial)
        trial.set_user_attr("ewma_mr_csv", mr_csv)
        env_overrides = _ewma_mr_params_for_symbol(mr_csv, trial)
    else:
        builder = SEARCH_SPACES[agent]
        env_overrides = builder(trial)

    res = run_simulation(env_overrides)
    if res is None:
        return -1e9

    # Read each trial so batch-mode setdefault(...) in the parent applies to workers.
    stab = float(os.getenv("OPT_STABILITY_PENALTY", str(STABILITY_PENALTY)))
    std_w = float(os.getenv("OPT_DAY_STDDEV_PENALTY", DAY_STDDEV_PENALTY_DEFAULT))
    neg_w = float(os.getenv("OPT_NEG_DAY_LOSS_WEIGHT", NEG_DAY_LOSS_WEIGHT_DEFAULT))
    cv_w = float(os.getenv("OPT_CV_PENALTY", "0"))

    loss_tail = max(0.0, float(-res["min_day_pnl"]))
    # CV denominator floor: avoids huge CV when |μ_day|≈0; scales weakly with |total PnL|.
    fa = abs(float(res["final_pnl"]))
    mean_floor = max(25.0, 0.02 * fa) if fa > 1e-9 else 25.0
    mean_abs = max(abs(float(res["mean_day_pnl"])), mean_floor)
    pnl_cv = float(res["pnl_std"]) / mean_abs if len(res["day_pnls"]) > 1 else 0.0

    score = (
        res["final_pnl"]
        - stab * res["pnl_spread"]
        - std_w * res["pnl_std"]
        - neg_w * loss_tail
        - cv_w * pnl_cv
    )

    trial.set_user_attr("final_pnl",     res["final_pnl"])
    trial.set_user_attr("mean_day_pnl",  res["mean_day_pnl"])
    trial.set_user_attr("pnl_spread",    res["pnl_spread"])
    trial.set_user_attr("pnl_std",       res["pnl_std"])
    trial.set_user_attr("pnl_cv",        pnl_cv)
    trial.set_user_attr("min_day_pnl",   res["min_day_pnl"])
    trial.set_user_attr("day_pnls",      res["day_pnls"])
    trial.set_user_attr("objective_score", score)

    return score


def worker_task(study_name: str, db_path: str) -> None:
    try:
        study = optuna.load_study(study_name=study_name, storage=db_path)
        study.optimize(objective, n_trials=1)
    except Exception:
        # Workers swallow errors so one failure doesn't kill the pool.
        pass


def _canonical_snackpack_name(piece: str) -> str | None:
    """Map a shorthand token (``CHOCOLATE``, ``snackpack_vanilla``) to ``SNACKPACK_*``."""
    pool = SNACKPACK_STEP_MR_SYMBOLS
    p = piece.strip().upper().replace("-", "_")
    if not p:
        return None
    if p in pool:
        return p
    if p.startswith("SNACKPACK_") and p in pool:
        return p
    cand = "SNACKPACK_" + p
    if cand in pool:
        return cand
    return next((s for s in pool if s.endswith("_" + p)), None)


def _filtered_snackpack_symbols() -> list[str]:
    raw = os.getenv("STEP_MR_SNACKPACK_ONLY", "").strip()
    if not raw:
        return list(SNACKPACK_STEP_MR_SYMBOLS)
    uniq: list[str] = []
    for piece in raw.split(","):
        full = _canonical_snackpack_name(piece)
        if full is not None and full not in uniq:
            uniq.append(full)
    return uniq


def _canonical_galaxy_name(piece: str) -> str | None:
    """Map shorthand (``DARK_MATTER``, ``galaxy_sounds_solar_flames``) to full symbol."""
    pool = GALAXY_STEP_MR_SYMBOLS
    p = piece.strip().upper().replace("-", "_")
    if not p:
        return None
    if p in pool:
        return p
    if p.startswith("GALAXY_SOUNDS_") and p in pool:
        return p
    cand = "GALAXY_SOUNDS_" + p
    if cand in pool:
        return cand
    return next((s for s in pool if s.endswith("_" + p)), None)


def _filtered_galaxy_symbols() -> list[str]:
    raw = os.getenv("STEP_MR_GALAXY_ONLY", "").strip()
    if not raw:
        return list(GALAXY_STEP_MR_SYMBOLS)
    uniq: list[str] = []
    for piece in raw.split(","):
        full = _canonical_galaxy_name(piece)
        if full is not None and full not in uniq:
            uniq.append(full)
    return uniq


def _study_name_galaxy(sym: str) -> str:
    suf = sym.replace("GALAXY_SOUNDS_", "").strip().lower()
    return f"r5_step_mr_gs_{suf}"


def _study_name_step_mr(sym: str) -> str:
    suf = sym.replace("SNACKPACK_", "").strip().lower()
    return f"r5_step_mr_{suf}"


def _snackpack_batch_study_specs(filtered_symbols: list[str]) -> list[tuple[str, str]]:
    """Build (Optuna study name, ``STEP_MR_SYMBOLS`` CSV) for the snackpack batch.

    ``STEP_MR_BATCH_LINK_GROUPS`` uses ``+`` to tie legs that share one parameter
    set (comma-separated CSV in ``test_bed``, e.g. ``CHOCOLATE+VANILLA``).
    Separate groups with ``;``. Order within ``filtered_symbols`` is preserved
    for ungrouped singletons.

    Mirrors (e.g. chocolate / vanilla) are good candidates — same MR knobs reduce
    per-leg overfitting; microstructure differences are still plausible, so grouping
    is optional.
    """
    filt_set = set(filtered_symbols)
    raw_groups = os.getenv("STEP_MR_BATCH_LINK_GROUPS", "").strip()
    used: set[str] = set()
    specs: list[tuple[str, str]] = []

    if raw_groups:
        for block in raw_groups.split(";"):
            block = block.strip()
            if not block:
                continue
            intersection: list[str] = []
            seen_mem: set[str] = set()
            for piece in block.split("+"):
                full = _canonical_snackpack_name(piece.strip())
                if full is None:
                    raise SystemExit(
                        "Unknown snackpack token in STEP_MR_BATCH_LINK_GROUPS: "
                        f"{piece.strip()!r}"
                    )
                if full in filt_set and full not in seen_mem:
                    seen_mem.add(full)
                    intersection.append(full)

            if len(intersection) >= 2:
                slug = "_".join(s.replace("SNACKPACK_", "").lower() for s in intersection)
                study_nm = f"r5_step_mr_linked_{slug}"
                specs.append((study_nm, ",".join(intersection)))
                used.update(intersection)

    for sym in filtered_symbols:
        if sym not in used:
            specs.append((_study_name_step_mr(sym), sym))
    return specs


def _batch_wal_pragma(db_fname: str) -> None:
    engine = create_engine(f"sqlite:///{db_fname}")
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.commit()


def run_step_mr_snackpack_batch() -> None:
    """Snackpack batch: one Optuna study per leg or per ``STEP_MR_BATCH_LINK_GROUPS``."""
    global TARGET_AGENT

    os.environ["TARGET_AGENT"] = "STEP_MR"
    TARGET_AGENT = "STEP_MR"
    symbols = _filtered_snackpack_symbols()
    if not symbols:
        raise SystemExit("No snackpack symbols after filtering STEP_MR_SNACKPACK_ONLY.")

    study_specs = _snackpack_batch_study_specs(symbols)

    fname = STEP_MR_BATCH_DB
    stor = STEP_MR_BATCH_DB_PATH
    resume = _env_truthy("STEP_MR_BATCH_RESUME")

    if _env_truthy("STEP_MR_SNACKPACK_FRESH"):
        try:
            if os.path.isfile(fname):
                os.remove(fname)
        except OSError:
            pass

    _batch_wal_pragma(fname)

    # Stricter defaults for batch runs if unset (parent + workers read via getenv in objective).
    os.environ.setdefault("OPT_STABILITY_PENALTY", "1.15")
    os.environ.setdefault("OPT_DAY_STDDEV_PENALTY", "0.85")
    os.environ.setdefault("OPT_NEG_DAY_LOSS_WEIGHT", "0.55")
    os.environ.setdefault("OPT_CV_PENALTY", "350")

    stab_e = float(os.environ["OPT_STABILITY_PENALTY"])
    std_e = float(os.environ["OPT_DAY_STDDEV_PENALTY"])
    neg_e = float(os.environ["OPT_NEG_DAY_LOSS_WEIGHT"])
    cv_e = float(os.environ["OPT_CV_PENALTY"])

    summaries: list[tuple[str, float, dict]] = []

    print("\n🔭 Round 5 — STEP_MR snackpack sweep (one study per leg or linked group)")
    print(f"   SQLite             : {fname}")
    print(f"   Symbols            : {', '.join(symbols)}")
    print(f"   Resume studies    : {resume}")
    print(f"   Workers            : {MAX_WORKERS}")
    print(f"   Trials per study   : {N_TRIALS}")
    print(f"   Days               : {DAYS_TO_TEST}")
    print(f"   OPT_STABILITY_PENALTY      : {stab_e}")
    print(f"   OPT_DAY_STDDEV_PENALTY     : {std_e}")
    print(f"   OPT_NEG_DAY_LOSS_WEIGHT    : {neg_e}")
    print(f"   OPT_CV_PENALTY             : {cv_e}")

    link_raw = os.getenv("STEP_MR_BATCH_LINK_GROUPS", "").strip()
    if link_raw:
        print(f"   Link groups       : {link_raw}")
        print(f"   Studies (& CSV)   : {[(n, c) for n, c in study_specs]}")

    for study_nm, mr_csv in study_specs:
        if not resume:
            try:
                optuna.delete_study(study_name=study_nm, storage=stor)
            except Exception:
                pass

        sampler = optuna.samplers.TPESampler(multivariate=True)
        study = optuna.create_study(
            study_name=study_nm,
            storage=stor,
            direction="maximize",
            sampler=sampler,
            load_if_exists=resume,
        )
        study.set_user_attr("step_mr_symbol", mr_csv)
        study.set_user_attr("optimize_agent", "STEP_MR")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(worker_task, study_nm, stor) for _ in range(N_TRIALS)]
            short = "+".join(x.replace("SNACKPACK_", "") for x in mr_csv.split(","))
            for _ in tqdm(as_completed(futures), total=N_TRIALS, desc=short):
                pass

        st = optuna.load_study(study_name=study_nm, storage=stor)
        if not st.trials:
            print(f"❌ {study_nm}: no trials recorded — skipping.")
            continue

        bt = st.best_trial
        summaries.append((mr_csv, float(st.best_value), dict(st.best_params)))
        print("─" * 60)
        print(
            f" ✓ Best {mr_csv}  score={st.best_value:,.2f}  "
            f"raw_pnl={bt.user_attrs.get('final_pnl', 0):,.2f}  "
            f"spread={bt.user_attrs.get('pnl_spread', 0):,.2f}  "
            f"σ_days={bt.user_attrs.get('pnl_std', 0):,.2f}  "
            f"cv={bt.user_attrs.get('pnl_cv', 0):.3f}  "
            f"min_day={bt.user_attrs.get('min_day_pnl', 0):,.2f}"
        )
        print(f"   Params: {st.best_params}")

    print("\n" + "═" * 60)
    print(f" {'ALL SNACKPACK STEP_MR STUDIES DONE':^58}")
    print("═" * 60)
    for tag, score, bp in summaries:
        print(f"  {tag:40}  best_score {score:10,.2f}    {bp}")
    print("═" * 60)


def run_step_mr_galaxy_batch() -> None:
    """One Optuna study per ``GALAXY_SOUNDS_*`` leg; storage ``STEP_MR_GALAXY_DB``."""
    global TARGET_AGENT

    os.environ["TARGET_AGENT"] = "STEP_MR"
    TARGET_AGENT = "STEP_MR"
    symbols = _filtered_galaxy_symbols()
    if not symbols:
        raise SystemExit("No GALAXY_SOUNDS symbols after filtering STEP_MR_GALAXY_ONLY.")

    fname = STEP_MR_GALAXY_DB
    stor = STEP_MR_GALAXY_DB_PATH
    resume = _env_truthy("STEP_MR_GALAXY_RESUME")

    if _env_truthy("STEP_MR_GALAXY_FRESH"):
        try:
            if os.path.isfile(fname):
                os.remove(fname)
        except OSError:
            pass

    _batch_wal_pragma(fname)

    os.environ.setdefault("OPT_STABILITY_PENALTY", "1.15")
    os.environ.setdefault("OPT_DAY_STDDEV_PENALTY", "0.85")
    os.environ.setdefault("OPT_NEG_DAY_LOSS_WEIGHT", "0.55")
    os.environ.setdefault("OPT_CV_PENALTY", "350")

    stab_e = float(os.environ["OPT_STABILITY_PENALTY"])
    std_e = float(os.environ["OPT_DAY_STDDEV_PENALTY"])
    neg_e = float(os.environ["OPT_NEG_DAY_LOSS_WEIGHT"])
    cv_e = float(os.environ["OPT_CV_PENALTY"])

    summaries: list[tuple[str, float, dict, list]] = []

    print("\n🔭 Round 5 — STEP_MR GALAXY_SOUNDS sweep (one study per leg)")
    print(f"   SQLite             : {fname}")
    print(f"   Symbols            : {', '.join(symbols)}")
    print(f"   Resume studies     : {resume}")
    print(f"   Workers            : {MAX_WORKERS}")
    print(f"   Trials per study   : {N_TRIALS}")
    print(f"   Days               : {DAYS_TO_TEST}")
    print(f"   OPT_STABILITY_PENALTY      : {stab_e}")
    print(f"   OPT_DAY_STDDEV_PENALTY     : {std_e}")
    print(f"   OPT_NEG_DAY_LOSS_WEIGHT    : {neg_e}")
    print(f"   OPT_CV_PENALTY             : {cv_e}")

    for sym in symbols:
        study_nm = _study_name_galaxy(sym)
        mr_csv = sym

        if not resume:
            try:
                optuna.delete_study(study_name=study_nm, storage=stor)
            except Exception:
                pass

        sampler = optuna.samplers.TPESampler(multivariate=True)
        study = optuna.create_study(
            study_name=study_nm,
            storage=stor,
            direction="maximize",
            sampler=sampler,
            load_if_exists=resume,
        )
        study.set_user_attr("step_mr_symbol", mr_csv)
        study.set_user_attr("optimize_agent", "STEP_MR")

        short = sym.replace("GALAXY_SOUNDS_", "")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(worker_task, study_nm, stor) for _ in range(N_TRIALS)]
            for _ in tqdm(as_completed(futures), total=N_TRIALS, desc=short):
                pass

        st = optuna.load_study(study_name=study_nm, storage=stor)
        if not st.trials:
            print(f"❌ {sym}: no trials recorded — skipping.")
            continue

        bt = st.best_trial
        day_pnls = list(bt.user_attrs.get("day_pnls", []) or [])
        summaries.append((sym, float(st.best_value), dict(st.best_params), day_pnls))
        print("─" * 60)
        print(
            f" ✓ Best {sym}  score={st.best_value:,.2f}  "
            f"raw_pnl={bt.user_attrs.get('final_pnl', 0):,.2f}  "
            f"spread={bt.user_attrs.get('pnl_spread', 0):,.2f}  "
            f"σ_days={bt.user_attrs.get('pnl_std', 0):,.2f}  "
            f"cv={bt.user_attrs.get('pnl_cv', 0):.3f}  "
            f"min_day={bt.user_attrs.get('min_day_pnl', 0):,.2f}"
        )
        print(f"   Params: {st.best_params}")

    print("\n" + "═" * 60)
    print(f" {'ALL GALAXY_SOUNDS STEP_MR STUDIES DONE':^58}")
    print("═" * 60)
    for sym, score, bp, dp in summaries:
        print(f"  {sym:40}  best_score {score:10,.2f}    {bp}")
    print("═" * 60)

    summary_path = os.getenv(
        "STEP_MR_GALAXY_SUMMARY_JSON",
        "step_mr_galaxy_best_trials.json",
    )
    out = []
    for sym, score, bp, dp in summaries:
        out.append(
            {
                "symbol": sym,
                "best_objective_score": score,
                "best_params_env": {
                    "STEP_MR_VAR_WINDOW": str(bp.get("var_window")),
                    "STEP_MR_STEP_LONG_TICKS": str(bp.get("step_long_ticks")),
                    "STEP_MR_STEP_SHORT_TICKS": str(bp.get("step_short_ticks")),
                    "STEP_MR_K_SD": str(bp.get("step_k_sd")),
                    "STEP_MR_SIGMA_FLOOR": str(bp.get("sigma_floor")),
                    "STEP_MR_Z_BUY": str(bp.get("z_buy")),
                    "STEP_MR_Z_SELL": str(bp.get("z_sell")),
                },
                "best_params_numeric": bp,
                "days_tested": list(DAYS_TO_TEST),
                "day_pnls": dp,
            }
        )
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nWrote {summary_path!r} ({len(out)} legs).")
    except OSError as e:
        print(f"\n⚠️  Could not write {summary_path!r}: {e}")


def _filtered_mr_universe_symbols() -> list[str]:
    raw = os.getenv("MR_UNIVERSE_ONLY", "").strip()
    allowed = list(ROUND5_ALL_MR_PRODUCTS)
    universe_set = set(allowed)
    if not raw:
        return allowed
    out: list[str] = []
    for piece in raw.split(","):
        s = piece.strip()
        if not s:
            continue
        if s not in universe_set:
            raise SystemExit(
                "MR_UNIVERSE_ONLY lists unknown symbol "
                f"{s!r} (must be one of the 50 ROUND5_ALL_MR_PRODUCTS)."
            )
        if s not in out:
            out.append(s)
    return out


def _mr_universe_study_name(agent: str, sym: str) -> str:
    ag = agent.strip().lower().replace("_", "")
    suf = sym.strip().lower().replace(" ", "_")[:100]
    return f"r5_univ_{ag}__{suf}"


def _pack_step_best_env(bp: dict) -> dict:
    return {
        "STEP_MR_VAR_WINDOW": str(bp.get("var_window")),
        "STEP_MR_STEP_LONG_TICKS": str(bp.get("step_long_ticks")),
        "STEP_MR_STEP_SHORT_TICKS": str(bp.get("step_short_ticks")),
        "STEP_MR_K_SD": str(bp.get("step_k_sd")),
        "STEP_MR_SIGMA_FLOOR": str(bp.get("sigma_floor")),
        "STEP_MR_Z_BUY": str(bp.get("z_buy")),
        "STEP_MR_Z_SELL": str(bp.get("z_sell")),
    }


def _pack_ewma_best_env(bp: dict) -> dict:
    return {
        "EWMA_MR_VAR_WINDOW": str(bp.get("var_window")),
        "EWMA_MR_ALPHA": str(bp.get("alpha")),
        "EWMA_MR_Z_BUY": str(bp.get("z_buy")),
        "EWMA_MR_Z_SELL": str(bp.get("z_sell")),
    }


def _study_best_record(study: optuna.Study) -> dict | None:
    if not study.trials:
        return None
    try:
        score = float(study.best_value)
    except ValueError:
        return None
    bt = study.best_trial
    return {
        "best_objective_score": score,
        "best_params": dict(bt.params),
        "final_pnl": float(bt.user_attrs.get("final_pnl", 0.0)),
        "mean_day_pnl": float(bt.user_attrs.get("mean_day_pnl", 0.0)),
        "pnl_spread": float(bt.user_attrs.get("pnl_spread", 0.0)),
        "pnl_std": float(bt.user_attrs.get("pnl_std", 0.0)),
        "min_day_pnl": float(bt.user_attrs.get("min_day_pnl", 0.0)),
        "day_pnls": list(bt.user_attrs.get("day_pnls", []) or []),
    }


def _run_symbol_agent_study(
    sym: str,
    agent: str,
    storage_uri: str,
    n_trials: int,
    resume: bool,
) -> optuna.Study:
    study_nm = _mr_universe_study_name(agent, sym)
    if not resume:
        try:
            optuna.delete_study(study_name=study_nm, storage=storage_uri)
        except Exception:
            pass
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(
        study_name=study_nm,
        storage=storage_uri,
        direction="maximize",
        sampler=sampler,
        load_if_exists=resume,
    )
    study.set_user_attr("optimize_agent", agent.upper())
    study.set_user_attr("mr_focus_symbol", sym)

    desc = f"{agent[:8]} {sym[:22]}"
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker_task, study_nm, storage_uri) for _ in range(n_trials)]
        for _ in tqdm(as_completed(futures), total=n_trials, desc=desc):
            pass
    return optuna.load_study(study_name=study_nm, storage=storage_uri)


def run_mr_universe_comparison_batch() -> None:
    """Run STEP_MR vs EWMA_MR Optuna studies for every Round 5 product.

    Env:
      * ``MR_UNIVERSE_ALL`` must be truthy when invoked via ``bayesian_optimisation`` CLI.
      * ``MR_UNIVERSE_ONLY`` — optional comma-separated subset of symbols.
      * ``MR_UNIVERSE_TRIALS_PER_STUDY`` — trials per agent per symbol (default: same as ``OPT_N_TRIALS``, i.e. ``200`` unless overridden).
      * ``OPT_N_TRIALS`` — fallback if ``MR_UNIVERSE_TRIALS_PER_STUDY`` is unset.
      * ``MR_UNIVERSE_RESUME`` — keep existing SQLite studies.
      * ``MR_UNIVERSE_FRESH`` — delete the universe SQLite DB before starting.
      * ``MR_UNIVERSE_SUMMARY_JSON`` — output path (default ``mr_universe_comparison_results.json``).

    Interpretation notes
    --------------------
    EWMA_MR fair is seeded with the **first observation** each run (cold start).

    Confirmation for ``clean_trader15`` is **not applied automatically**. After you
    review the JSON summary, selectively promote STEP params into ``CONFIGS``, or wire
    the EWMA class / prod trader if EWMA_MR wins consistently on a cohort.
    """
    symbols = _filtered_mr_universe_symbols()
    fname = MR_UNIVERSE_DB
    stor = MR_UNIVERSE_DB_PATH
    resume = _env_truthy("MR_UNIVERSE_RESUME")
    n_trials = int(os.getenv("MR_UNIVERSE_TRIALS_PER_STUDY", str(N_TRIALS)))

    if _env_truthy("MR_UNIVERSE_FRESH"):
        try:
            if os.path.isfile(fname):
                os.remove(fname)
        except OSError:
            pass

    _batch_wal_pragma(fname)

    json_path = os.getenv("MR_UNIVERSE_SUMMARY_JSON", "mr_universe_comparison_results.json")

    print("\n⚖️  Round 5 — STEP_MR vs EWMA_MR universe comparison")
    print(f"   Symbols            : {len(symbols)}")
    print(f"   SQLite             : {fname}")
    print(f"   Trials / study    : {n_trials}")
    print(f"   Resume             : {resume}")
    print(f"   Workers            : {MAX_WORKERS}")
    print(f"   Days               : {DAYS_TO_TEST}")
    print(f"   Output JSON       : {json_path}")

    out_rows: list[dict] = []

    for sym in symbols:
        st_study = _run_symbol_agent_study(sym, "STEP_MR", stor, n_trials, resume)
        ew_study = _run_symbol_agent_study(sym, "EWMA_MR", stor, n_trials, resume)

        bp_step = _study_best_record(st_study)
        bp_ew = _study_best_record(ew_study)

        recommendation = "inspect"
        delta_score = None
        delta_pnl = None
        if bp_step is not None and bp_ew is not None:
            ds = bp_step["best_objective_score"] - bp_ew["best_objective_score"]
            dp = bp_step["final_pnl"] - bp_ew["final_pnl"]
            delta_score = ds
            delta_pnl = dp
            if ds > 1e-6:
                recommendation = "STEP_MR"
            elif ds < -1e-6:
                recommendation = "EWMA_MR"
            else:
                recommendation = "tie"

        row = {
            "symbol": sym,
            "step_mr_best": bp_step,
            "ewma_mr_best": bp_ew,
            "comparison": {
                "better_by_optuna_score": (
                    recommendation if recommendation != "inspect" else None
                ),
                "delta_best_objective_score": delta_score,
                "delta_final_pnl_best_trial": delta_pnl,
            },
            "proposed_clean_trader_pick": recommendation,
            "implement_step_params_env": (
                _pack_step_best_env(bp_step["best_params"]) if bp_step else None
            ),
            "implement_ewma_params_env": (
                _pack_ewma_best_env(bp_ew["best_params"]) if bp_ew else None
            ),
        }
        out_rows.append(row)

        print("═" * 72)
        print(f" {sym}")
        print("─" * 72)
        if bp_step:
            print(
                f"  STEP_MR score={bp_step['best_objective_score']:,.2f}  "
                f"PnL={bp_step['final_pnl']:,.2f}"
            )
        else:
            print("  STEP_MR  (no usable trials)")
        if bp_ew:
            print(
                f"  EWMA_MR score={bp_ew['best_objective_score']:,.2f}  "
                f"PnL={bp_ew['final_pnl']:,.2f}"
            )
        else:
            print("  EWMA_MR (no usable trials)")
        print(f"  → proposed pick : {row['proposed_clean_trader_pick']}")

    payload = {
        "days_tested": list(DAYS_TO_TEST),
        "trials_per_study": n_trials,
        "n_symbols": len(symbols),
        "results": out_rows,
    }
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n✓ Wrote {json_path!r}")
    except OSError as e:
        print(f"\n⚠️  Could not write {json_path!r}: {e}")


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if _env_truthy("MR_UNIVERSE_ALL"):
        run_mr_universe_comparison_batch()
        raise SystemExit(0)

    if TARGET_AGENT not in SEARCH_SPACES:
        raise SystemExit(
            f"Unknown TARGET_AGENT={TARGET_AGENT!r}. "
            f"Choose from {sorted(SEARCH_SPACES)}.")

    galaxy_all = TARGET_AGENT == "STEP_MR" and _env_truthy("STEP_MR_GALAXY_ALL")
    if galaxy_all:
        run_step_mr_galaxy_batch()
        raise SystemExit(0)

    snack_all = TARGET_AGENT == "STEP_MR" and _env_truthy("STEP_MR_SNACKPACK_ALL")
    if snack_all:
        run_step_mr_snackpack_batch()
        raise SystemExit(0)

    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )

    if TARGET_AGENT == "STEP_MR":
        study.set_user_attr("optimize_agent", "STEP_MR")
        study.set_user_attr("step_mr_symbol", STEP_MR_SYMBOL)
    elif TARGET_AGENT == "EWMA_MR":
        study.set_user_attr("optimize_agent", "EWMA_MR")
        study.set_user_attr("ewma_mr_symbol", EWMA_MR_SYMBOL)

    engine = create_engine(f"sqlite:///{DB_FILE}")
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.commit()

    print("\n🔭 Round 5 Bayesian Optimisation")
    print(f"   Target agent      : {TARGET_AGENT}")
    if TARGET_AGENT == "SPIKE_MR":
        print(f"   Spike MR symbol   : {SPIKE_MR_SYMBOL}")
    elif TARGET_AGENT == "STEP_MR":
        print(f"   Step MR symbol    : {STEP_MR_SYMBOL}")
    elif TARGET_AGENT == "EWMA_MR":
        print(f"   EWMA MR symbol    : {EWMA_MR_SYMBOL}")
    elif TARGET_AGENT == "LEAD_LAG":
        print(f"   Leader / Follower : {LEAD_LAG_LEADER} → {LEAD_LAG_FOLLOWER}")
    print(f"   Workers           : {MAX_WORKERS}")
    print(f"   Trials            : {N_TRIALS}")
    print(f"   Days              : {DAYS_TO_TEST}")
    stab_e = float(os.getenv("OPT_STABILITY_PENALTY", str(STABILITY_PENALTY)))
    std_e = float(os.getenv("OPT_DAY_STDDEV_PENALTY", DAY_STDDEV_PENALTY_DEFAULT))
    neg_e = float(os.getenv("OPT_NEG_DAY_LOSS_WEIGHT", NEG_DAY_LOSS_WEIGHT_DEFAULT))
    cv_e = float(os.getenv("OPT_CV_PENALTY", "0"))
    print(f"   OPT_STABILITY_PENALTY   : {stab_e}")
    print(f"   OPT_DAY_STDDEV_PENALTY  : {std_e}")
    print(f"   OPT_NEG_DAY_LOSS_WEIGHT : {neg_e}")
    print(f"   OPT_CV_PENALTY          : {cv_e}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker_task, STUDY_NAME, DB_PATH)
                   for _ in range(N_TRIALS)]
        for _ in tqdm(as_completed(futures), total=N_TRIALS, desc="Optimising"):
            pass

    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
    except Exception as e:
        print(f"❌ Could not load study: {e}")
        raise SystemExit(1)

    if not study.trials:
        print("❌ No trials recorded.")
        raise SystemExit(1)

    best = study.best_trial
    print("\n" + "═" * 60)
    print(f" {'OPTIMIZATION COMPLETE':^58}")
    print("═" * 60)
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k:18} : {v:,.6f}")
        else:
            print(f"  {k:18} : {v}")
    print("-" * 60)
    print(f"  Best objective score : {study.best_value:,.2f}")
    print(f"  Raw final PnL        : {best.user_attrs.get('final_pnl', 0):,.2f}")
    print(f"  PnL spread (range)   : {best.user_attrs.get('pnl_spread', 0):,.2f}")
    print(f"  Mean day PnL         : {best.user_attrs.get('mean_day_pnl', 0):,.2f}")
    print(f"  PnL stdev (days)     : {best.user_attrs.get('pnl_std', 0):,.2f}")
    print(f"  PnL CV (σ/|μ|)       : {best.user_attrs.get('pnl_cv', 0):.4f}")
    print(f"  Worst day PnL        : {best.user_attrs.get('min_day_pnl', 0):,.2f}")
    day_pnls = best.user_attrs.get("day_pnls", [])
    if day_pnls:
        formatted = ", ".join(f"D+{d}={p:,.0f}" for d, p in zip(DAYS_TO_TEST, day_pnls))
        print(f"  Per-day PnL         : {formatted}")
    print("═" * 60)
