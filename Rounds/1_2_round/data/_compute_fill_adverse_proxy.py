import numpy as np
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent
DAYS = [-2, -1, 0]
PRODUCT = "ASH_COATED_OSMIUM"
HORIZONS = [1, 3, 5, 10]


def load_day(day: int):
    p = pd.read_csv(BASE / f"prices_round_1_day_{day}.csv", sep=";")
    t = pd.read_csv(BASE / f"trades_round_1_day_{day}.csv", sep=";")

    p = p[p["product"] == PRODUCT].copy()
    p = p.sort_values("timestamp").reset_index(drop=True)
    p["mid"] = pd.to_numeric(p["mid_price"], errors="coerce").replace(0, np.nan)
    p = p.dropna(subset=["mid", "bid_price_1", "ask_price_1"]).reset_index(drop=True)

    t = t[t["symbol"] == PRODUCT].copy()
    t = t.sort_values("timestamp").reset_index(drop=True)
    t["price"] = pd.to_numeric(t["price"], errors="coerce")
    t = t.dropna(subset=["price"]).reset_index(drop=True)
    return p, t


def has_fill(window_trades: pd.DataFrame, px: float, side: str):
    # side == "bid": passive buy at bid, proxy fill when trade prints exactly at bid
    # side == "ask": passive sell at ask, proxy fill when trade prints exactly at ask
    if side == "bid":
        w = window_trades[window_trades["price"] == px]
    else:
        w = window_trades[window_trades["price"] == px]
    return not w.empty


def compute_day_metrics(day: int):
    p, t = load_day(day)

    out_rows = []
    ts_tr = t["timestamp"].values if len(t) else np.array([], dtype=float)

    for h in HORIZONS:
        n = len(p)
        usable = n - h
        if usable <= 0:
            continue

        bid_fill_count = 0
        ask_fill_count = 0
        bid_total_markouts = []
        ask_total_markouts = []
        bid_adverse_costs = []
        ask_adverse_costs = []

        for i in range(usable):
            ts0 = p.at[i, "timestamp"]
            ts1 = p.at[i + h, "timestamp"]
            mid_0 = float(p.at[i, "mid"])
            mid_h = float(p.at[i + h, "mid"])
            bid = float(p.at[i, "bid_price_1"])
            ask = float(p.at[i, "ask_price_1"])

            if len(ts_tr):
                l = np.searchsorted(ts_tr, ts0, side="left")
                r = np.searchsorted(ts_tr, ts1, side="right")
                w = t.iloc[l:r]
            else:
                w = t.iloc[0:0]

            bid_hit = has_fill(w, bid, side="bid")
            if bid_hit:
                bid_fill_count += 1
                # total markout vs fill price
                bid_total_markouts.append(mid_h - bid)
                # adverse selection uses post-fill move only (exclude spread capture)
                bid_adverse_costs.append(-(mid_h - mid_0))

            ask_hit = has_fill(w, ask, side="ask")
            if ask_hit:
                ask_fill_count += 1
                ask_total_markouts.append(ask - mid_h)
                ask_adverse_costs.append(mid_h - mid_0)

        bid_fill_prob = bid_fill_count / usable
        ask_fill_prob = ask_fill_count / usable
        both_fill_prob = (bid_fill_count + ask_fill_count) / (2 * usable)

        bid_markout = float(np.mean(bid_total_markouts)) if bid_total_markouts else np.nan
        ask_markout = float(np.mean(ask_total_markouts)) if ask_total_markouts else np.nan
        avg_markout = float(np.nanmean([bid_markout, ask_markout]))
        bid_adv = float(np.mean(bid_adverse_costs)) if bid_adverse_costs else np.nan
        ask_adv = float(np.mean(ask_adverse_costs)) if ask_adverse_costs else np.nan
        avg_adv_cost = float(np.nanmean([bid_adv, ask_adv]))

        out_rows.append(
            {
                "day": day,
                "horizon_ticks": h,
                "samples": usable,
                "bid_fill_prob": bid_fill_prob,
                "ask_fill_prob": ask_fill_prob,
                "avg_fill_prob": both_fill_prob,
                "bid_markout_ticks": bid_markout,
                "ask_markout_ticks": ask_markout,
                "avg_total_markout_ticks": avg_markout,
                "bid_adverse_cost_ticks": bid_adv,
                "ask_adverse_cost_ticks": ask_adv,
                "avg_adverse_selection_cost_ticks": avg_adv_cost,
                "expected_markout_per_quote": both_fill_prob * avg_markout,
            }
        )

    return pd.DataFrame(out_rows)


def main():
    all_rows = []
    for d in DAYS:
        all_rows.append(compute_day_metrics(d))

    res = pd.concat(all_rows, ignore_index=True)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    print("=== Per-day proxy fill + adverse selection ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    agg = (
        res.groupby("horizon_ticks", as_index=False)[
            [
                "bid_fill_prob",
                "ask_fill_prob",
                "avg_fill_prob",
                "avg_total_markout_ticks",
                "avg_adverse_selection_cost_ticks",
                "expected_markout_per_quote",
            ]
        ]
        .mean()
        .sort_values("horizon_ticks")
    )
    print("\n=== Average across days ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
