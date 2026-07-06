import subprocess
import re
import os
import warnings
import optuna
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Update these paths as necessary for your local environment
TRADER_PATH  = "/home/dansp/projects/imc_prosperity_tutorial/Rounds/5_round/Traders/test_bed.py"
DATASET_PATH = "/home/dansp/projects/imc_prosperity_tutorial/prosperity_rust_backtester/datasets/round5"

# Days to iterate over
DAYS_TO_TEST = [1,2,3] 

# Optimization Settings
N_TRIALS = 200          
MAX_WORKERS = max(1, os.cpu_count() - 1)
STABILITY_PENALTY = 0.8
STEP = 0.1

DB_FILE = "optuna_parallel.db"
DB_PATH = f"sqlite:///{DB_FILE}?timeout=60"
OPT_PRODUCT = os.getenv("OPT_PRODUCT", "PEBBLES_XS")
STUDY_NAME = f"{OPT_PRODUCT.lower()}_round5_opt"

# ─────────────────────────────────────────────
#  SIMULATION ENGINE
# ─────────────────────────────────────────────
def run_simulation(params: dict) -> dict | None:
    """Executes the rust backtester day by day, aggregating results."""
    base_env = os.environ.copy()
    
    # Static parameters used across all trials
    base_env["TEST_PRODUCTS"] = OPT_PRODUCT
    base_env["P1_TYPE"] = os.getenv("OPT_PRODUCT_TYPE", "MR")
    base_env["P1_SHORT_CODE"] = os.getenv("OPT_SHORT_CODE", "OPT1")
    base_env["P1_POS_LIMIT"] = os.getenv("OPT_POS_LIMIT", "10")
    base_env["P1_VAR_WINDOW"] = str(params["ROLLING_WINDOW"])
    base_env["P1_MEAN"] = os.getenv("OPT_GLOBAL_INT", "1250")
    base_env["P1_SLOPE"] = os.getenv("OPT_GLOBAL_SLOPE", "0")
    base_env["P1_TTE_TRANSLATION"] = os.getenv("OPT_GLOBAL_TTE_TRANSLATION", "0")
    base_env["P1_Z_BUY"] = str(params["MR_Z_BUY"])
    base_env["P1_Z_SELL"] = str(params["MR_Z_SELL"])

    day_pnls = []
    total_pnl = 0.0

    for day in DAYS_TO_TEST:
        env = base_env.copy()
        
        # FIXED: Changed "CURRENT_DAY" to "DAY" to match your trader's os.getenv("DAY")
        env["DAY"] = str(day) 
        
        cmd = [
            "rust_backtester", 
            "--trader", TRADER_PATH, 
            "--dataset", DATASET_PATH,
            "--day", str(day) 
        ]
        
        try:
            # Timeout at 120s to prevent hanging trials
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env, timeout=120)
            stdout = result.stdout

            # FIXED REGEX: Matches the PnL column in the Rust backtester's summary table
            # Line format: D+1   1   10000   102   19650.00   runs/backtest-...
            total_match = re.search(r"D\+\d+\s+\d+\s+\d+\s+\d+\s+([\d,.\-]+)\s+runs/", stdout)
            
            if not total_match:
                # Fallback: Search for the specific product breakdown line if the summary row is missing
                product_name = env.get("TEST_PRODUCT", OPT_PRODUCT)
                total_match = re.search(rf"{product_name}\s+([\d,.\-]+)", stdout)

            if not total_match: 
                return None

            day_pnl = float(total_match.group(1).replace(",", ""))
            day_pnls.append(day_pnl)
            total_pnl += day_pnl

        except Exception:
            return None # Fail the whole trial if one day crashes

    # Calculate spread across the evaluated days for stability scoring
    pnl_spread = max(day_pnls) - min(day_pnls) if len(day_pnls) > 1 else 0.0

    return {"final_pnl": total_pnl, "pnl_spread": pnl_spread}

def objective(trial):
    """Optuna objective function for a single trial."""
    trial.set_user_attr("final_pnl", 0.0)
    trial.set_user_attr("pnl_spread", 0.0)

    params = {
        "MR_Z_BUY":       round(trial.suggest_float("MR_Z_BUY", 1.0, 8.0) / STEP) * STEP,
        "MR_Z_SELL":      round(trial.suggest_float("MR_Z_SELL", 1.0, 8.0) / STEP) * STEP,
        "ROLLING_WINDOW": round(trial.suggest_int("ROLLING_WINDOW", 10, 500))
        
    }

    # Early exit for logically invalid parameters (e.g., selling cheaper than buying)
    if params["MR_Z_SELL"] < params["MR_Z_BUY"]:
        return -9999999.0

    res = run_simulation(params)
    if res is None:
        return -9999999.0

    # Store successful trial metrics for final reporting
    trial.set_user_attr("final_pnl", res["final_pnl"])
    trial.set_user_attr("pnl_spread", res["pnl_spread"])
    
    # Score = Profit - (Risk Penalty * Spread)
    return res["final_pnl"] - (STABILITY_PENALTY * res["pnl_spread"])

def worker_task(study_name, db_path):
    """Execution wrapper for worker processes."""
    try:
        study = optuna.load_study(study_name=study_name, storage=db_path)
        study.optimize(objective, n_trials=1)
    except Exception:
        pass 

# ─────────────────────────────────────────────
#  EXECUTION BLOCK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Clean start: delete the old database if it exists
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(multivariate=True)
    )

    # Optimization for SQLite performance in parallel
    engine = create_engine(f"sqlite:///{DB_FILE}")
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.commit()
    
    print(f"🚀 Starting parallel optimization (WAL Mode Enabled)...")
    print(f"   Workers: {MAX_WORKERS} | Trials: {N_TRIALS} | Days: {DAYS_TO_TEST}")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit batches of work
        futures = [executor.submit(worker_task, STUDY_NAME, DB_PATH) for _ in range(N_TRIALS)]
        for _ in tqdm(as_completed(futures), total=N_TRIALS, desc="Optimizing"):
            pass

    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
        if len(study.trials) == 0:
            print("❌ No successful trials recorded. Check your TRADER_PATH or regex.")
        else:
            best_trial = study.best_trial
            print("\n" + "═"*60)
            print(f" {'OPTIMIZATION COMPLETE':^58}")
            print("═"*60)
            
            for k, v in study.best_params.items():
                print(f" {k:18} : {v:,.6f}" if isinstance(v, float) else f" {k:18} : {v}")
            
            print("-" * 60)
            print(f" Best Adjusted Score : {study.best_value:,.2f}")
            print(f" Raw Final PnL       : {best_trial.user_attrs.get('final_pnl', 0):,.2f}")
            print(f" PnL Spread (Risk)   : {best_trial.user_attrs.get('pnl_spread', 0):,.2f}")
            print("═"*60)
    except Exception as e:
        print(f"❌ Could not load final results: {e}")