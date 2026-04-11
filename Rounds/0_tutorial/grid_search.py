import subprocess
import re
import pandas as pd
import os
import concurrent.futures
from itertools import product
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# --- CONFIGURATION ---
BASE = r"C:\Users\dansp\OneDrive\Desktop\imc_prosperity_tutorial"

TRADER_PATH = rf"{BASE}\rounds\0_tutorial\traders\stable_emwa_mm.py"
PYTHON_BIN  = rf"{BASE}\.venv\Scripts\python.exe"
SIM_CMD     = rf"{BASE}\.venv\Scripts\prosperity4mcbt.exe"

# Define ranges to test
param_grid = {
    'skew_cf': [0.5, 1.5, 2, 3, 4, 5, 6, 7, 8],
    'min_edge': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'drop_edge': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
}

def run_simulation(params):
    """Runs simulation by passing parameters through Environment Variables."""
    # Create a unique environment for this specific worker
    env_vars = os.environ.copy()
    env_vars["SKEW_CF"] = str(params['skew_cf'])
    env_vars["MIN_EDGE"] = str(params['min_edge'])
    env_vars["DROP_EDGE"] = str(params['drop_edge'])

    cmd = [
        SIM_CMD, TRADER_PATH,
        "--quick"
    ]
    
    try:
        # env=env_vars is the magic that passes your grid_search.env data
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env_vars)
        
        mean_match = re.search(r"Mean total PnL:\s*([\d,\.-]+)", result.stdout)
        std_match = re.search(r"Std total PnL:\s*([\d,\.-]+)", result.stdout)
        
        if mean_match and std_match:
            mean = float(mean_match.group(1).replace(',', ''))
            std = float(std_match.group(1).replace(',', ''))
            sharpe = mean / std if std != 0 else 0
            return {**params, 'mean_pnl': mean, 'std_pnl': std, 'sharpe': sharpe}
    except subprocess.CalledProcessError as e:
        print(f"Sim failed for {params}. Error: {e.stderr}")
    return None

if __name__ == "__main__":
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    
    print(f"Starting Parallel Grid Search (Environment Variable Mode)...")
    print(f"Total combinations: {len(combinations)} | Cores: {os.cpu_count()}")

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_params = {executor.submit(run_simulation, p): p for p in combinations}
        
        for count, future in enumerate(concurrent.futures.as_completed(future_to_params)):
            res = future.result()
            if res:
                results.append(res)
            if (count + 1) % 10 == 0:
                print(f"Completed {count + 1}/{len(combinations)}...")

    if results:
        df = pd.DataFrame(results)
        df.to_csv("grid_search_results.csv", index=False)
        
        # Plotting - one subplot per parameter, showing its effect on mean PnL
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Grid Search: Parameter Effects on Mean PnL", fontsize=14)
        
        params_to_plot = [
            ('skew_cf', 'Skew CF'),
            ('min_edge', 'Min Edge'),
            ('drop_edge', 'Drop Edge'),
        ]
        
        for ax, (param, label) in zip(axes, params_to_plot):
            avg_perf = df.groupby(param)['mean_pnl'].mean()
            ax.plot(avg_perf.index, avg_perf.values, marker='o')
            ax.set_title(f"{label} vs Avg PnL")
            ax.set_xlabel(label)
            ax.set_ylabel("Mean PnL")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("optimization_summary.png")
        plt.show()
        
        print("\n--- TOP 3 STABLE SETS (By Sharpe) ---")
        print(df.sort_values('sharpe', ascending=False).head(3))

        print("\n--- TOP 3 SETS (By PNL) ---")
        print(df.sort_values('mean_pnl', ascending=False).head(3))