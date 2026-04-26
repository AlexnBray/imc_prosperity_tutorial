import subprocess
import re
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import optuna
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRADER_PATH  = "/home/dansp/projects/imc_prosperity_tutorial/Rounds/3_round/traders/test_bed.py"
DATASET_PATH = "/home/dansp/projects/imc_prosperity_tutorial/prosperity_rust_backtester/datasets/round3"

# Search bounds
MR_Z_BUY_RANGE  = (1.0, 8.0)
MR_Z_SELL_RANGE = (2.0, 8.0)
STEP            = 0.25

# Optimization settings
N_TRIALS         = 80
N_STARTUP_TRIALS = 15
SPREAD_PENALTIES = [0.0, 0.5, 1.0, 2.0, 3.5]

# Parallelism settings
MAX_WORKERS = max(1, os.cpu_count() - 1)
DB_PATH     = "sqlite:///optuna_optimizer.db"
OUTPUT_CSV  = "bayes_results.csv"
OUTPUT_PNG  = "bayes_analysis.png"

# ─────────────────────────────────────────────
#  SIMULATION ENGINE
# ─────────────────────────────────────────────
def run_simulation(mr_z_buy: float, mr_z_sell: float) -> dict | None:
    env = os.environ.copy()
    env["MR_Z_BUY"]  = str(mr_z_buy)
    env["MR_Z_SELL"] = str(mr_z_sell)
    
    cmd = ["rust_backtester", "--trader", TRADER_PATH, "--dataset", DATASET_PATH]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        stdout = result.stdout

        day_pnls = [float(p.replace(",", "")) for p in re.findall(r"D[=+]\d\s+\d+\s+\d+\s+\d+\s+([\d,.\-]+)", stdout)]
        total_match = re.search(r"TOTAL\s+.*?\s+([\d,.\-]+)\s*-?\s*$", stdout, re.MULTILINE)
        
        if not total_match: return None

        final_pnl  = float(total_match.group(1).replace(",", ""))
        pnl_spread = max(day_pnls) - min(day_pnls) if len(day_pnls) > 1 else 0.0

        out = {"final_pnl": final_pnl, "pnl_spread": pnl_spread}
        for i, v in enumerate(day_pnls):
            out[f"day_{i}_pnl"] = v
        return out
    except:
        return None

def optimize_task(penalty: float, study_name: str):
    """Worker task: Runs on a separate process to avoid GIL bottlenecks."""
    study = optuna.load_study(study_name=study_name, storage=DB_PATH)
    trial = study.ask()
    
    raw_buy  = trial.suggest_float("MR_Z_BUY",  *MR_Z_BUY_RANGE)
    raw_sell = trial.suggest_float("MR_Z_SELL", *MR_Z_SELL_RANGE)
    
    buy  = round(round(raw_buy  / STEP) * STEP, 4)
    sell = round(round(raw_sell / STEP) * STEP, 4)

    if sell < buy:
        study.tell(trial, float("-inf"))
        return {"penalty": penalty, "score": float("-inf")}

    res = run_simulation(buy, sell)
    if res is None:
        study.tell(trial, float("-inf"))
        return {"penalty": penalty, "score": float("-inf")}

    score = res["final_pnl"] - penalty * res["pnl_spread"]
    study.tell(trial, score)
    
    return {"penalty": penalty, "score": score, "MR_Z_BUY": buy, "MR_Z_SELL": sell, **res}

# ─────────────────────────────────────────────
#  OUTPUT & VISUALIZATION
# ─────────────────────────────────────────────
def print_optimal_summary(df: pd.DataFrame):
    print("\n" + "═" * 85)
    print(f"{'TOP 5 CONFIGURATIONS PER PENALTY LEVEL':^85}")
    print("═" * 85)
    for penalty in SPREAD_PENALTIES:
        print(f"\n>>> PENALTY λ = {penalty}")
        print(f"{'Rank':>4} | {'BUY':>8} | {'SELL':>8} | {'Score':>12} | {'Final PnL':>12} | {'Spread':>8}")
        print("-" * 85)
        sub = df[df["penalty"] == penalty].sort_values(by="score", ascending=False).head(5)
        for i, (_, row) in enumerate(sub.iterrows(), 1):
            print(f"{i:>4} | {row['MR_Z_BUY']:>8.2f} | {row['MR_Z_SELL']:>8.2f} | "
                  f"{row['score']:>12,.0f} | {row['final_pnl']:>12,.0f} | {row['pnl_spread']:>8,.0f}")

def build_plots(df: pd.DataFrame):
    penalties = SPREAD_PENALTIES
    n_pen = len(penalties)
    colors = ["#00D4FF", "#7B61FF", "#FF6B6B", "#FFB347", "#4CAF50", "#FF4DA6"][:n_pen]
    
    fig = plt.figure(figsize=(24, 30), facecolor="#0D0F14")
    fig.suptitle("Bayesian Optimisation Analysis", fontsize=26, fontweight="bold", color="white", y=0.98)
    gs = gridspec.GridSpec(5, n_pen, figure=fig, hspace=0.45, wspace=0.3, top=0.94, bottom=0.04, left=0.05, right=0.95)

    # Row 0: Convergence
    for col, (penalty, color) in enumerate(zip(penalties, colors)):
        ax = fig.add_subplot(gs[0, col], facecolor="#161921")
        sub = df[df["penalty"] == penalty].reset_index()
        if not sub.empty:
            valid = sub["score"].replace(-np.inf, sub[sub["score"] > -np.inf]["score"].min() if any(sub["score"] > -np.inf) else 0)
            ax.plot(sub.index, valid, "o", color=color, alpha=0.3, ms=3)
            ax.plot(sub.index, np.maximum.accumulate(valid), color=color, lw=2)
        ax.set_title(f"λ = {penalty} Convergence", color=color)

    # Row 1: Heatmaps
    for col, (penalty, color) in enumerate(zip(penalties, colors)):
        ax = fig.add_subplot(gs[1, col], facecolor="#161921")
        sub = df[(df["penalty"] == penalty) & (df["score"] > -np.inf)]
        sc = ax.scatter(sub["MR_Z_BUY"], sub["MR_Z_SELL"], c=sub["score"], cmap=LinearSegmentedColormap.from_list("c", ["#1A1D28", color]), s=40)
        ax.set_title(f"λ = {penalty} Heatmap", color=color)

    # Row 2: 3D Surface
    for col, (penalty, color) in enumerate(zip(penalties, colors)):
        ax = fig.add_subplot(gs[2, col], projection='3d', facecolor="#0D0F14")
        sub = df[(df["penalty"] == penalty) & (df["score"] > -np.inf)]
        ax.scatter(sub["MR_Z_BUY"], sub["MR_Z_SELL"], sub["score"], c=sub["score"], cmap=LinearSegmentedColormap.from_list("c", ["#444444", color]), s=15)
        ax.set_title(f"3D λ = {penalty}", color=color)
        ax.xaxis.set_pane_color((0,0,0,0)); ax.yaxis.set_pane_color((0,0,0,0)); ax.zaxis.set_pane_color((0,0,0,0))
        ax.tick_params(colors="#8A8FA8", labelsize=7)

    # Row 3: Marginals
    ax_buy = fig.add_subplot(gs[3, :n_pen//2 + 1], facecolor="#161921")
    ax_sell = fig.add_subplot(gs[3, n_pen//2 + 1:], facecolor="#161921")
    for penalty, color in zip(penalties, colors):
        sub = df[(df["penalty"] == penalty) & (df["score"] > -np.inf)]
        ax_buy.plot(sub.groupby("MR_Z_BUY")["score"].mean().sort_index(), "o-", color=color, label=f"λ={penalty}", alpha=0.6)
        ax_sell.plot(sub.groupby("MR_Z_SELL")["score"].mean().sort_index(), "o-", color=color, label=f"λ={penalty}", alpha=0.6)
    ax_buy.set_title("Marginal: Z_BUY", color="white"); ax_sell.set_title("Marginal: Z_SELL", color="white")
    ax_buy.legend(fontsize=7, facecolor="#1E2130", labelcolor="white")

    # Row 4: Frontier
    ax_f = fig.add_subplot(gs[4, :], facecolor="#161921")
    for penalty, color in zip(penalties, colors):
        sub = df[df["penalty"] == penalty]
        ax_f.scatter(sub["pnl_spread"], sub["final_pnl"], c=color, s=30, alpha=0.5, label=f"λ={penalty}")
    ax_f.set_title("Risk-Reward Frontier", color="white")
    ax_f.set_xlabel("Volatility (Spread)"); ax_f.set_ylabel("PnL")

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="#0D0F14")
    plt.show()

# ─────────────────────────────────────────────
#  EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists("optuna_optimizer.db"): os.remove("optuna_optimizer.db")
    all_records = []

    print(f"Starting Bayesian Optimization with {MAX_WORKERS} workers...")
    for penalty in SPREAD_PENALTIES:
        study_name = f"penalty_{penalty}"
        study = optuna.create_study(study_name=study_name, storage=DB_PATH, direction="maximize",
                                    sampler=optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True))

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(optimize_task, penalty, study_name) for _ in range(N_TRIALS)]
            for f in tqdm(as_completed(futures), total=N_TRIALS, desc=f"Optimizing λ={penalty}"):
                res = f.result()
                if res: all_records.append(res)

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print_optimal_summary(df)
    build_plots(df)