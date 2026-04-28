import subprocess
import re
import pandas as pd
import numpy as np
import os
import concurrent.futures
from itertools import product
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TRADER_PATH = "/home/dansp/projects/imc_prosperity_tutorial/Rounds/4_round/traders/fast_traderv5.py"
DATASET_PATH = "/home/dansp/projects/imc_prosperity_tutorial/prosperity_rust_backtester/datasets/round4"
1
param_grid = {
    'MR_Z_BUY': [1, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
    'MR_Z_SELL': [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
}

def run_simulation(params):
    """Runs backtester and extracts PnL for each day + total."""
    env_vars = os.environ.copy()
    env_vars["MR_Z_BUY"] = str(params['MR_Z_BUY'])
    env_vars["MR_Z_SELL"] = str(params['MR_Z_SELL'])

    cmd = [
        "rust_backtester",
        "--trader", TRADER_PATH,
        "--dataset", DATASET_PATH,
    ]

    try:
        # result.stdout is captured, not printed to terminal
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env_vars)
        
        # Extract individual day PnLs (D=0, D+1, D+2)
        # Matches the PnL column in the "SET DAY TICKS..." table
        day_matches = re.findall(r"D[=|+]\d\s+\d+\s+\d+\s+\d+\s+([\d,\.-]+)", result.stdout)
        day_pnls = [float(p.replace(',', '')) for p in day_matches]
        
        # Search for the line starting with TOTAL and grab the numeric value at the end
        # This regex looks for 'TOTAL', skips characters until it finds a number with potential decimals/commas
        total_match = re.search(r"TOTAL\s+.*?\s+([\d,\.-]+)\s+-?$", result.stdout, re.MULTILINE)

        if total_match:
            final_pnl = float(total_match.group(1).replace(',', ''))
        else:
            final_pnl = np.nan

        # Calculate Spread (Stability metric)
        pnl_spread = max(day_pnls) - min(day_pnls) if len(day_pnls) > 1 else 0
        
        # Consistency Score: Higher is better (High PnL and Low Spread)
        consistency_score = final_pnl - pnl_spread

        res_dict = {
            **params, 
            'final_pnl': final_pnl, 
            'pnl_spread': pnl_spread,
            'consistency_score': consistency_score
        }
        
        # Dynamically add day columns (day_0, day_1, etc.)
        for i, val in enumerate(day_pnls):
            res_dict[f'day_{i}_pnl'] = val
            
        return res_dict

    except subprocess.CalledProcessError as e:
        return None

if __name__ == "__main__":
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    
    print(f"Starting Grid Search (Silent Mode)...")
    print(f"Total combinations: {len(combinations)} | Cores: {os.cpu_count()}")

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_params = {executor.submit(run_simulation, p): p for p in combinations}
        
        for count, future in enumerate(concurrent.futures.as_completed(future_to_params)):
            res = future.result()
            if res:
                results.append(res)
            if (count + 1) % 10 == 0:
                print(f"Progress: {count + 1}/{len(combinations)} simulations done.")

    if results:
        df = pd.DataFrame(results)
        
        # Sort by consistency_score (PnL vs Spread)
        df = df.sort_values('consistency_score', ascending=False)
        df.to_csv("grid_search_results.csv", index=False)
        
        # Terminal Summary
        print("\n--- TOP 5 STABLE & PROFITABLE SETS ---")
        # Selecting key columns for clean terminal display
        display_cols = ['MR_Z_BUY', 'MR_Z_SELL', 'final_pnl', 'pnl_spread', 'consistency_score']
        print(df[display_cols].head(5).to_string(index=False))

        # Re-plot using the new consistency score
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, param in zip(axes, ['MR_Z_BUY', 'MR_Z_SELL']):
            avg_perf = df.groupby(param)['consistency_score'].mean()
            ax.plot(avg_perf.index, avg_perf.values, marker='o', color='green')
            ax.set_title(f"{param} vs Avg Consistency Score")
            ax.set_ylabel("Consistency (PnL - Spread)")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("stability_summary.png")
        print("\nResults saved to grid_search_results.csv and stability_summary.png")