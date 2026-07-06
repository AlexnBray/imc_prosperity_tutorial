#!/usr/bin/env python3
"""Turn ``mr_universe_comparison_results.json`` into review artifacts.

Reads the JSON emitted by ``bayesian_optimisation.run_mr_universe_comparison_batch``.
Writes Markdown + CSV beside the JSON unless ``--stdout-only``.

Usage::

    cd Rounds/5_round/Traders
    python3 summarize_mr_universe_json.py /path/to/mr_universe_comparison_results.json

    python3 summarize_mr_universe_json.py   # resolves default repo path below

What to confirm before editing ``clean_trader15.CONFIGS``
---------------------------------------------------------
1. ``proposed_clean_trader_pick`` uses **Optuna composite score**, not purely raw PnL.
2. **Single-name MR** only fits products you actually run as standalone MR legs.
3. ``short_code`` in snippets — verify against ``test_bed`` / ``trader.py`` leg registry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


def _f(x: object) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
        if math.isnan(v):
            return float("nan")
        return v
    except (TypeError, ValueError):
        return float("nan")


def _short_code_placeholder(symbol: str) -> str:
    return symbol[:4].upper()


def env_to_step_snippet(sym: str, env: dict | None, short_hint: str) -> str:
    if not env:
        return ""
    vw = env.get("STEP_MR_VAR_WINDOW", "?")
    sl = env.get("STEP_MR_STEP_LONG_TICKS", "?")
    ss = env.get("STEP_MR_STEP_SHORT_TICKS", "?")
    ksd = env.get("STEP_MR_K_SD", "?")
    sf = env.get("STEP_MR_SIGMA_FLOOR", "?")
    zb = env.get("STEP_MR_Z_BUY", "?")
    zs = env.get("STEP_MR_Z_SELL", "?")
    return (
        f'    "{sym}": StepMeanReversionConfig(\n'
        f'        symbol="{sym}",\n'
        f'        short_code="{short_hint}",  # VERIFY leg code\n'
        "        pos_limit=10,\n"
        f"        var_window={vw},\n"
        f"        z_buy=float({zb}),\n"
        f"        z_sell=float({zs}),\n"
        "        mean=0.0,\n"
        f"        step_long_ticks=int({sl}),\n"
        f"        step_short_ticks=int({ss}),\n"
        f"        step_k_sd=float({ksd}),\n"
        f"        step_sigma_floor=float({sf}),\n"
        "    ),"
    )


def env_to_ewma_snippet(sym: str, env: dict | None, short_hint: str) -> str:
    if not env:
        return ""
    vw = env.get("EWMA_MR_VAR_WINDOW", "?")
    al = env.get("EWMA_MR_ALPHA", "?")
    zb = env.get("EWMA_MR_Z_BUY", "?")
    zs = env.get("EWMA_MR_Z_SELL", "?")
    return (
        f'    "{sym}": EWMAMeanReversionConfig(\n'
        f'        symbol="{sym}",\n'
        f'        short_code="{short_hint}",  # VERIFY leg code\n'
        "        pos_limit=10,\n"
        f"        var_window={vw},\n"
        f"        z_buy=float({zb}),\n"
        f"        z_sell=float({zs}),\n"
        "        mean=0.0,\n"
        f"        alpha=float({al}),\n"
        "    ),"
    )


def default_json_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        .parent
        .joinpath("mr_universe_comparison_results.json")
    )


def flat_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for r in payload.get("results") or []:
        sym = r.get("symbol") or ""
        st = r.get("step_mr_best") or {}
        ew = r.get("ewma_mr_best") or {}
        comp = r.get("comparison") or {}
        ds = _f(comp.get("delta_best_objective_score"))
        rows.append(
            {
                "symbol": sym,
                "better_by_optuna_score": comp.get("better_by_optuna_score") or "",
                "proposed_pick": r.get("proposed_clean_trader_pick") or "",
                "delta_score": ds,
                "delta_pnl": _f(comp.get("delta_final_pnl_best_trial")),
                "step_score": _f(st.get("best_objective_score")),
                "step_pnl": _f(st.get("final_pnl")),
                "step_min_day": _f(st.get("min_day_pnl")),
                "ewma_score": _f(ew.get("best_objective_score")),
                "ewma_pnl": _f(ew.get("final_pnl")),
                "ewma_min_day": _f(ew.get("min_day_pnl")),
            }
        )
    rows.sort(
        key=lambda x: (-abs(float(x["delta_score"]))) if math.isfinite(x["delta_score"]) else 0.0
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=os.getenv(
            "MR_UNIVERSE_SUMMARY_JSON",
            str(default_json_path()),
        ),
    )
    ap.add_argument("--stdout-only", action="store_true", help="print Markdown to stdout only")
    args = ap.parse_args()

    jpath = Path(args.json_path)
    if not jpath.is_file():
        print(f"Missing JSON: {jpath}", file=sys.stderr)
        print("Optimisation still running or path wrong.", file=sys.stderr)
        return 1

    payload = json.loads(jpath.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    trials = payload.get("trials_per_study")
    days = payload.get("days_tested")

    md_lines: list[str] = []
    md_lines.append("# MR universe — STEP_MR vs EWMA_MR\n")
    md_lines.append(f"- JSON: `{jpath.resolve()}`\n")
    md_lines.append(f"- Days evaluated: `{days}`\n")
    md_lines.append(f"- Trials per study (per agent, per symbol): `{trials}`\n")
    md_lines.append(
        "- **Recommendation column** prefers higher **Optuna score** "
        "(stability‑penalised), not necessarily higher raw ΣPnL.\n"
    )

    counts: dict[str, int] = {}
    wins_step = wins_ewma = ties = inconclusive = 0
    for r in results:
        p = str(r.get("proposed_clean_trader_pick") or "").upper()
        counts[p] = counts.get(p, 0) + 1
        if p == "STEP_MR":
            wins_step += 1
        elif p == "EWMA_MR":
            wins_ewma += 1
        elif p == "TIE":
            ties += 1
        else:
            inconclusive += 1

    md_lines.append("\n## Headline counts\n")
    md_lines.append(f"| STEP_MR wins (by score) | {wins_step} |\n")
    md_lines.append(f"| EWMA_MR wins (by score) | {wins_ewma} |\n")
    md_lines.append(f"| Ties                    | {ties} |\n")
    md_lines.append(f"| Inconclusive / missing  | {inconclusive} |\n")

    md_lines.append("\n## Per-symbol summary (sorted by |Δ score|)\n")
    md_lines.append(
        "| Symbol | Pick (score) | Δ score | Δ PnL(best) | "
        "STEP score | STEP ΣPnL | EWMA score | EWMA ΣPnL |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )

    rows = flat_rows(payload)
    for row in rows:
        sym = row["symbol"]
        ds = row["delta_score"]
        dsp = ds if isinstance(ds, (int, float)) and math.isfinite(ds) else ds
        md_lines.append(
            f"| {sym} | {row['proposed_pick']} | {dsp:,.4g} "
            f"| {_format_num(row['delta_pnl']) } "
            f"| {_format_num(row['step_score'])} | {_format_num(row['step_pnl'])} "
            f"| {_format_num(row['ewma_score'])} | {_format_num(row['ewma_pnl'])} |\n"
        )

    md_lines.append("\n## Generated ``CONFIG`` snippets (winner only)\n")
    md_lines.append("Paste into ``clean_trader15.py`` **after manual review**. Import types as needed.\n")
    md_lines.append("```python\n# --- STEP_MR winners ---\n")
    for r in results:
        if r.get("proposed_clean_trader_pick") != "STEP_MR":
            continue
        sym = r["symbol"]
        env = r.get("implement_step_params_env")
        md_lines.append(env_to_step_snippet(sym, env, _short_code_placeholder(sym)) + "\n")
    md_lines.append("\n# --- EWMA_MR winners ---\n")
    for r in results:
        if r.get("proposed_clean_trader_pick") != "EWMA_MR":
            continue
        sym = r["symbol"]
        env = r.get("implement_ewma_params_env")
        md_lines.append(env_to_ewma_snippet(sym, env, _short_code_placeholder(sym)) + "\n")
    md_lines.append("```\n")

    md_lines.append("\n## Full nested JSON pointers\n")
    md_lines.append(
        "Each ``results[]`` entry has ``step_mr_best.best_params``, "
        "``ewma_mr_best.best_params``, ``day_pnls``, and ``implement_*_params_env`` "
        "(env strings aligned with ``test_bed.py``).\n"
    )

    text = "".join(md_lines)
    if args.stdout_only:
        print(text)
    else:
        md_path = jpath.with_suffix(".analysis.md")
        csv_path = jpath.with_suffix(".analysis.csv")
        md_path.write_text(text, encoding="utf-8")
        write_csv(csv_path, rows)
        print(f"Wrote {md_path}")
        print(f"Wrote {csv_path}")
        print(text)
    return 0


def _format_num(x: float) -> str:
    if not isinstance(x, (int, float)) or math.isnan(x):
        return "—"
    return f"{float(x):,.2f}"

if __name__ == "__main__":
    sys.exit(main())
