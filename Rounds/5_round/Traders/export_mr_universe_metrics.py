#!/usr/bin/env python3
"""Export Optuna MR-universe metrics + best params (STEP vs EWMA) from SQLite.

Use when ``mr_universe_comparison_results.json`` is not ready yet, or as a
parallel machine-readable dump.

Usage (from ``Traders/``)::

    uv run --with optuna python export_mr_universe_metrics.py \\
        --db optuna_mr_universe.db \\
        --out /path/to/mr_universe_metrics_export.json

Columns (per completed best trial)
---------------------------------
* **objective_score** — Optuna maximised composite (penalised).
* **final_pnl** — raw Σ day PnL over ``DAYS_TO_TEST``.
* **pnl_std** — population stdev of per-day PnLs.
* **pnl_spread** — max(day PnL) − min(day PnL).
* **mean_day_pnl**, **min_day_pnl**, **day_pnls** — from the same backtest.
* **best_params** — Optuna hyperparameters for that strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
from optuna.trial import TrialState

# Same ordering as universe batch (import after path setup).
_TRADERS_DIR = Path(__file__).resolve().parent
if str(_TRADERS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRADERS_DIR))

import bayesian_optimisation as bo  # noqa: E402


def _slug(sym: str) -> str:
    return sym.strip().lower().replace(" ", "_")[:100]


def _symbol_from_study_suffix(suffix: str) -> str | None:
    for s in bo.ROUND5_ALL_MR_PRODUCTS:
        if _slug(s) == suffix:
            return s
    return None


def _parse_study_name(name: str) -> tuple[str | None, str | None]:
    """Return (\"STEP_MR\"|\"EWMA_MR\"|None, symbol|None)."""
    if not name.startswith("r5_univ_"):
        return None, None
    rest = name[len("r5_univ_") :]
    if "__" not in rest:
        return None, None
    ag_slug, sym_slug = rest.split("__", 1)
    if ag_slug == "stepmr":
        agent = "STEP_MR"
    elif ag_slug == "ewmamr":
        agent = "EWMA_MR"
    else:
        agent = None
    sym = _symbol_from_study_suffix(sym_slug)
    return agent, sym


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--db",
        type=Path,
        default=Path("optuna_mr_universe.db"),
        help="Optuna SQLite file (default: ./optuna_mr_universe.db)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write full JSON report (default: print Markdown to stdout only)",
    )
    ap.add_argument(
        "--expected-trials",
        type=int,
        default=None,
        help="If set, flag studies with fewer COMPLETE trials as incomplete",
    )
    args = ap.parse_args()

    if not args.db.is_file():
        print(f"No database: {args.db.resolve()}", file=sys.stderr)
        return 1

    storage = f"sqlite:///{args.db.resolve()}?timeout=60"
    summaries = optuna.study.get_all_study_summaries(storage)

    rows: dict[str, dict[str, dict]] = {}
    incomplete: list[str] = []

    for summary in summaries:
        agent_sym = _parse_study_name(summary.study_name)
        agent, sym = agent_sym
        if agent is None or sym is None:
            continue

        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        complete_n = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        if complete_n == 0:
            incomplete.append(f"{sym}::{agent} (no complete trials)")
            continue

        if args.expected_trials is not None and complete_n < args.expected_trials:
            incomplete.append(
                f"{sym}::{agent} ({complete_n}/{args.expected_trials} trials complete)"
            )

        try:
            bt = study.best_trial
        except ValueError:
            incomplete.append(f"{sym}::{agent} (no best_trial)")
            continue

        ua = bt.user_attrs
        rec = {
            "strategy": agent,
            "symbol": sym,
            "study_name": summary.study_name,
            "n_trials_complete": complete_n,
            "objective_score": float(bt.value),
            "objective_score_user_attr": float(ua.get("objective_score", bt.value)),
            "final_pnl_raw": float(ua.get("final_pnl", 0.0)),
            "mean_day_pnl": float(ua.get("mean_day_pnl", 0.0)),
            "pnl_std_across_days": float(ua.get("pnl_std", 0.0)),
            "pnl_spread_range": float(ua.get("pnl_spread", 0.0)),
            "min_day_pnl": float(ua.get("min_day_pnl", 0.0)),
            "day_pnls": list(ua.get("day_pnls", []) or []),
            "days_tested": list(bo.DAYS_TO_TEST),
            "best_params": dict(bt.params),
        }
        rows.setdefault(sym, {})[agent] = rec

    # Build symmetric report: only symbols with at least one strategy
    report = {
        "metadata": {
            "database": str(args.db.resolve()),
            "days_tested": list(bo.DAYS_TO_TEST),
            "symbols_with_data": sorted(rows.keys()),
            "expected_trials_per_study": args.expected_trials,
            "incomplete_notes": incomplete,
        },
        "by_symbol": rows,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out.resolve()}", file=sys.stderr)

    md: list[str] = []
    md.append("# MR universe — metrics + optimal params\n\n")
    md.append("| Symbol | Strategy | Objective | Raw ΣPnL | σ(days) | Spread | Mean/day | Worst day | Params |\n")
    md.append("|--------|----------|-----------|----------|---------|--------|----------|-----------|--------|\n")
    for sym in sorted(rows.keys()):
        for strat in ("STEP_MR", "EWMA_MR"):
            rec = rows[sym].get(strat)
            if not rec:
                md.append(
                    f"| {sym} | {strat} | — | — | — | — | — | — | *(no study data)* |\n"
                )
                continue
            p = json.dumps(rec["best_params"], separators=(",", ":"))
            if len(p) > 120:
                p = p[:117] + "..."
            md.append(
                f"| {sym} | {strat} | {rec['objective_score']:,.2f} | "
                f"{rec['final_pnl_raw']:,.2f} | "
                f"{rec['pnl_std_across_days']:,.2f} | "
                f"{rec['pnl_spread_range']:,.2f} | "
                f"{rec['mean_day_pnl']:,.2f} | "
                f"{rec['min_day_pnl']:,.2f} | `{p}` |\n"
            )
    if incomplete:
        md.append("\n## Incomplete / partial\n\n")
        for line in incomplete:
            md.append(f"- {line}\n")

    print("".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
