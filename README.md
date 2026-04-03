# Getting Started

**Quick Setup**

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone and sync dependencies:
   ```bash
   git clone <repository>
   cd <project>
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

That's it—`uv sync` creates the venv automatically and installs all dependencies from `pyproject.toml` and `uv.lock`.

Fix your system path Alex: C:\Users\alexn\AppData\Local\Python\pythoncore-3.14-64\Scripts\uv.exe sync

## Project structure

- `algorithms.py` — sample strategy library and utilities.
- `stat_utils.py` — statistics helpers used by analyzers.
- `backtests/` — stores log files from the official IMC backtester and logs from [prosperity4btest](https://github.com/nabayansaha/imc-prosperity-4-backtester).
- `montecarlo_backtester/` — modular Monte Carlo backtesting framework for Prosperity 4.
- `rounds` — stores all code and data for each round 
   - `0-tutorial` — example for how all future rounds should be setup
      - `data` — contains Prosperity 4 data packages
      - `traders` — has python scripts for backtesting and submission
      - `explore.ipynb` — jupiter notebook to call helper functions from stats and algorithms to visualise and manipulate data for that round

## Built-in backtesters and visualisers

1. `log_visualiser`
   - Visualises official Prosperity 4 logs and log files from prosperity4btest [prosperity4btest](https://github.com/nabayansaha/imc-prosperity-4-backtester) a fork of [jmerle's backtester for Prosperity 3](https://github.com/jmerle/imc-prosperity-3-backtester)
   - To use just run `log_visualiser/dashapp.py`

2. `montecarlo_backtester/imc-prosperity-4`
   - Uses 1st Place USA [Chris Roberts backtester](https://github.com/chrispyroberts/imc-prosperity-4)
   - Full Monte Carlo backtester + data model (`prosperity4mcbt/`) for stress-testing strategies.
   - Can implement models for assets for new rounds or clone from original repo if Cris Roberts keep updating
   - Videos on the topic
      - Backtester: https://www.youtube.com/watch?v=Mi-vVCZ0Vo4
      - Prosperity Breakdown: https://www.youtube.com/watch?v=PI2lJ063sJ8

## Monte Carlo quick start (Rust + npm)

1. Install Rust toolchain (Cargo):
   - Windows: `winget install --id Rustlang.Rustup -e --source winget` or https://www.rust-lang.org/tools/install
   - macOS/Linux: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

2. Install Node/npm:
   - Windows: `winget install --id OpenJS.NodeJS` or https://nodejs.org/
   - macOS/Linux: `brew install node` or package manager of choice

3. Run the Monte Carlo runner with a trader sample:
   ```bash
   prosperity4mcbt example_trader.py --quick --vis --out /tmp/mc_run/dashboard.json
   ```

5. Read through backtester README for more info


