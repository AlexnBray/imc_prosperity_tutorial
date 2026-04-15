"""
Estimate Kalman R, Q_level, Q_drift by maximum likelihood on historical
ASH_COATED_OSMIUM mids (Round 1 CSVs in this folder).

Baseline / optimizer start point must match the live trader:
  Rounds/1_round/traders/kalman_fv.py  →  class Trader (KF_R_OBS, KF_Q_LEVEL, KF_Q_DRIFT)

Run from repo root (recommended):
  .venv\\Scripts\\python.exe Rounds/1_round/data/optimize_kalman_mle.py

Or from this folder:
  cd Rounds/1_round/data
  python optimize_kalman_mle.py

Options:
  --walk-forward          MLE per day + pooled
  --max-ticks N           Use only first N mids (pooled / each segment)
  --day-max-ticks N       With --walk-forward: cap each day (default 4000)
  --out FILE.csv          Save per-segment results to CSV

Requires: scipy, pandas, numpy (your project venv).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PRODUCT = "ASH_COATED_OSMIUM"

# --- Keep in sync with kalman_fv.Trader Kalman constants ---
BASELINE_KF_R_OBS = 200
BASELINE_KF_Q_LEVEL = 0.3
BASELINE_KF_Q_DRIFT = 10**-6

# Log-space bounds for L-BFGS-B (match optimize_mle)
LOG_BOUNDS = (
    (math.log(1e-4), math.log(100.0)),
    (math.log(1e-6), math.log(10.0)),
    (math.log(1e-8), math.log(1.0)),
)


def load_mids(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("prices_round_1_day_*.csv")):
        df = pd.read_csv(path, sep=";")
        sub = df[df["product"] == PRODUCT].copy()
        if sub.empty:
            continue
        if "mid_price" in sub.columns and sub["mid_price"].notna().any():
            sub["mid"] = pd.to_numeric(sub["mid_price"], errors="coerce")
        else:
            b1 = pd.to_numeric(sub["bid_price_1"], errors="coerce")
            a1 = pd.to_numeric(sub["ask_price_1"], errors="coerce")
            b2 = pd.to_numeric(sub.get("bid_price_2"), errors="coerce")
            a2 = pd.to_numeric(sub.get("ask_price_2"), errors="coerce")
            sub["mid"] = np.where(
                b2.notna() & a2.notna(),
                (b2 + a2) / 2.0,
                (b1 + a1) / 2.0,
            )
        sub = sub.dropna(subset=["mid"])
        sub["mid"] = sub["mid"].replace(0.0, np.nan).dropna()
        stem = path.stem
        day_part = stem.split("day_")[-1] if "day_" in stem else "0"
        try:
            day = int(day_part)
        except ValueError:
            day = 0
        sub["day"] = day
        rows.append(sub[["day", "timestamp", "mid"]])
    if not rows:
        raise FileNotFoundError(f"No price CSVs with {PRODUCT} in {data_dir}")
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["day", "timestamp"]).reset_index(drop=True)


def _kalman_step(
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
) -> Tuple[float, float, float, float, float, float, float, float]:
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[q_level, 0.0], [0.0, q_drift]])
    H = np.array([[1.0, 0.0]])
    x = np.array([mu, beta])
    P = np.array([[p00, p01], [p10, p11]])

    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    innov = float(z) - float(np.squeeze(H @ x_pred))
    S = float(np.squeeze(H @ P_pred @ H.T)) + r_obs
    if S <= 1e-18:
        S = 1e-18
    K = (P_pred @ H.T).flatten() / S
    x_new = x_pred + K * innov
    P_new = (np.eye(2) - np.outer(K, H)) @ P_pred
    return (
        float(x_new[0]),
        float(x_new[1]),
        float(P_new[0, 0]),
        float(P_new[0, 1]),
        float(P_new[1, 0]),
        float(P_new[1, 1]),
        innov,
        S,
    )


def innovation_log_likelihood(z: np.ndarray, r_obs: float, q_level: float, q_drift: float) -> float:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 3:
        return -1e18

    mu, beta = float(z[0]), 0.0
    p00, p01, p10, p11 = 25.0, 0.0, 0.0, 4.0

    ll = 0.0
    for t in range(1, len(z)):
        mu, beta, p00, p01, p10, p11, innov, S = _kalman_step(
            float(z[t]), mu, beta, p00, p01, p10, p11, r_obs, q_level, q_drift
        )
        ll += -0.5 * math.log(2.0 * math.pi * S) - 0.5 * (innov * innov) / S
    return ll


def neg_ll_from_log_params(u: np.ndarray, z: np.ndarray) -> float:
    r = math.exp(float(u[0]))
    q_l = math.exp(float(u[1]))
    q_d = math.exp(float(u[2]))
    ll = innovation_log_likelihood(z, r, q_l, q_d)
    if not math.isfinite(ll):
        return 1e12
    return -ll


def optimize_mle(z: np.ndarray, x0_log: np.ndarray | None = None) -> dict[str, Any]:
    from scipy.optimize import minimize

    if x0_log is None:
        x0_log = np.array([math.log(1.0), math.log(0.3), math.log(0.015)])

    res = minimize(
        neg_ll_from_log_params,
        x0_log,
        args=(z,),
        method="L-BFGS-B",
        bounds=list(LOG_BOUNDS),
        options={"maxiter": 80, "ftol": 1e-7},
    )

    r = math.exp(res.x[0])
    q_l = math.exp(res.x[1])
    q_d = math.exp(res.x[2])
    ll = innovation_log_likelihood(z, r, q_l, q_d)
    return {
        "segment": "pooled",
        "KF_R_OBS": r,
        "KF_Q_LEVEL": q_l,
        "KF_Q_DRIFT": q_d,
        "log_likelihood": ll,
        "n_obs": int(len(z)),
        "success": bool(res.success),
        "message": str(res.message),
    }


def _near_bound(val: float, lo: float, hi: float, pct: float = 0.05) -> bool:
    return val <= lo * (1.0 + pct) or val >= hi / (1.0 + pct)


def print_bounds_note(out: dict[str, Any]) -> None:
    lo_r, hi_r = math.exp(LOG_BOUNDS[0][0]), math.exp(LOG_BOUNDS[0][1])
    lo_ql, hi_ql = math.exp(LOG_BOUNDS[1][0]), math.exp(LOG_BOUNDS[1][1])
    lo_qd, hi_qd = math.exp(LOG_BOUNDS[2][0]), math.exp(LOG_BOUNDS[2][1])
    if _near_bound(out["KF_R_OBS"], lo_r, hi_r):
        print("Note: KF_R_OBS near optimization bound.", flush=True)
    if _near_bound(out["KF_Q_LEVEL"], lo_ql, hi_ql):
        print("Note: KF_Q_LEVEL near optimization bound.", flush=True)
    if _near_bound(out["KF_Q_DRIFT"], lo_qd, hi_qd):
        print(
            "Note: KF_Q_DRIFT near a bound (often lower: MLE likes smooth drift). "
            "Consider a trading floor above 1e-8 for regime changes.",
            flush=True,
        )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})
    print(f"Wrote {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="MLE Kalman R, Q_level, Q_drift from Round 1 CSVs")
    ap.add_argument("--walk-forward", action="store_true", help="MLE per day + pooled")
    ap.add_argument("--max-ticks", type=int, default=0, help="Cap mids (0 = all)")
    ap.add_argument("--day-max-ticks", type=int, default=4000, help="Per-day cap with --walk-forward (0 = all)")
    ap.add_argument("--out", type=str, default="", help="Write results CSV to this path")
    args = ap.parse_args()

    df = load_mids(DATA_DIR)
    print(f"Loaded {len(df)} rows for {PRODUCT} from {DATA_DIR}", flush=True)

    results: List[dict[str, Any]] = []
    z_all = df["mid"].values.astype(float)
    if args.max_ticks > 0:
        z_all = z_all[: args.max_ticks]

    if args.walk_forward:
        for d in sorted(df["day"].unique()):
            z = df.loc[df["day"] == d, "mid"].values.astype(float)
            if args.day_max_ticks > 0:
                z = z[: args.day_max_ticks]
            if args.max_ticks > 0:
                z = z[: args.max_ticks]
            if len(z) < 50:
                print(f"  day {d}: skip (n={len(z)})", flush=True)
                continue
            print(f"  day {d}: optimizing (n={len(z)})...", flush=True)
            out = optimize_mle(z)
            out["segment"] = f"day_{d}"
            results.append(out)
            print(
                f"  day {d}: R={out['KF_R_OBS']:.6g} Q_level={out['KF_Q_LEVEL']:.6g} "
                f"Q_drift={out['KF_Q_DRIFT']:.6g} LL={out['log_likelihood']:.2f} n={out['n_obs']}",
                flush=True,
            )

        print("\nPooled: optimizing...", flush=True)
        out_pooled = optimize_mle(z_all)
        out_pooled["segment"] = "pooled"
        results.append(out_pooled)
        out = out_pooled
    else:
        out = optimize_mle(z_all)
        out["segment"] = "pooled"
        results.append(out)

    print(
        f"\n=== Baseline (current kalman_fv.Trader) ===\n"
        f"  KF_R_OBS = {BASELINE_KF_R_OBS}\n"
        f"  KF_Q_LEVEL = {BASELINE_KF_Q_LEVEL}\n"
        f"  KF_Q_DRIFT = {BASELINE_KF_Q_DRIFT}",
        flush=True,
    )
    print("\n=== MLE Kalman parameters (optional paste into kalman_fv.py) ===", flush=True)
    print(f"  KF_R_OBS = {out['KF_R_OBS']:.12g}", flush=True)
    print(f"  KF_Q_LEVEL = {out['KF_Q_LEVEL']:.12g}", flush=True)
    print(f"  KF_Q_DRIFT = {out['KF_Q_DRIFT']:.12g}", flush=True)
    print(f"\nlog_likelihood = {out['log_likelihood']:.4f}  (n={out['n_obs']})", flush=True)
    print(f"success = {out['success']}  ({out['message']})", flush=True)
    print_bounds_note(out)

    base_ll = innovation_log_likelihood(
        z_all,
        BASELINE_KF_R_OBS,
        BASELINE_KF_Q_LEVEL,
        BASELINE_KF_Q_DRIFT,
    )
    print(
        f"\nBaseline LL (kalman_fv: R={BASELINE_KF_R_OBS}, "
        f"Q_level={BASELINE_KF_Q_LEVEL}, Q_drift={BASELINE_KF_Q_DRIFT}) = {base_ll:.4f}",
        flush=True,
    )
    print(f"Delta LL (MLE - baseline) = {out['log_likelihood'] - base_ll:.4f}", flush=True)

    if args.out:
        for row in results:
            row["baseline_KF_R_OBS"] = BASELINE_KF_R_OBS
            row["baseline_KF_Q_LEVEL"] = BASELINE_KF_Q_LEVEL
            row["baseline_KF_Q_DRIFT"] = BASELINE_KF_Q_DRIFT
        write_csv(Path(args.out), results)


if __name__ == "__main__":
    main()
