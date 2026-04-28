import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

DATA_DIR = Path(__file__).resolve().parent / 'data'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

OPTION_SYMBOLS = {4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500}
UNDERLYING_SYMBOL = 'VELVETFRUIT_EXTRACT'
OPTION_PREFIX = 'VEV_'
EXPIRY_DAY = 5
DAYS_PER_YEAR = 365


def bs_call(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def implied_volatility_call(market_price, S, K, T, r=0.0, tol=1e-6, maxiter=100):
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    intrinsic = max(0.0, S - K * math.exp(-r * T))
    if market_price < intrinsic or market_price > S:
        return np.nan

    def objective(sigma):
        return bs_call(S, K, T, sigma, r) - market_price

    low, high = 1e-6, 5.0
    try:
        # brentq returns 'low' if the root is at the boundary
        implied_vol = brentq(objective, low, high, xtol=tol, maxiter=maxiter)
        return implied_vol
    except Exception:
        return np.nan


def load_price_data():
    frames = []
    for csv_path in sorted(DATA_DIR.glob('prices_round_3_day_*.csv')):
        df = pd.read_csv(csv_path, sep=';')
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_iv_dataset(prices):
    if prices.empty:
        return pd.DataFrame()

    underlying = prices[prices['product'] == UNDERLYING_SYMBOL][['day', 'timestamp', 'mid_price']].copy()
    underlying.rename(columns={'mid_price': 'S'}, inplace=True)

    options = prices[prices['product'].str.startswith(OPTION_PREFIX)].copy()
    options['strike'] = options['product'].str.replace(f'{OPTION_PREFIX}', '', regex=False).astype(int)
    options = options[options['strike'].isin(OPTION_SYMBOLS)] 

    merged = options.merge(underlying, on=['day', 'timestamp'], how='left')
    merged = merged.rename(columns={'mid_price': 'option_mid'})

    # TTE calculation based on your provided logic
    merged['tte'] = (8 - merged['day'] - (merged['timestamp'] // 100) / 10000.0) / DAYS_PER_YEAR
    merged['tte'] = merged['tte'].clip(lower=1e-6)
    merged = merged[merged['S'] > 0].copy()

    merged['m_t_k'] = np.log(merged['strike'] / merged['S']) / np.sqrt(merged['tte'])
    merged['iv'] = merged.apply(
        lambda row: implied_volatility_call(row['option_mid'], row['S'], row['strike'], row['tte']),
        axis=1,
    )
    
    # 1. DROP NaNs and Infs
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['iv'])
    
    # 2. FILTER OUT BOUNDARY ZEROS 
    # Brentq defaults to 1e-6 when it fails to find a root. 
    # We filter slightly above that to catch all "flat-line" failures.
    merged = merged[merged['iv'] > 1e-4] 
    
    return merged


def fit_iv_smile(data, degree=2):
    X = np.vstack([data['m_t_k']**i for i in range(degree + 1)]).T
    coeffs, residuals, *_ = np.linalg.lstsq(X, data['iv'], rcond=None)
    return coeffs, residuals[0] if len(residuals) > 0 else 0.0


def plot_smiles(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    for day, group in data.groupby('day'):
        # Using median to smooth out individual noise/outliers per strike
        sample = group.groupby('strike')['iv'].median().sort_index()
        ax.plot(sample.index, sample.values, marker='o', label=f'Day {day}')

    ax.set_title('Option IV Smile by Day (Filtered)')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Volatility')
    ax.grid(True, alpha=0.35)
    ax.legend()
    path = OUTPUT_DIR / 'iv_smile_by_day.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_surface(data):
    data = data.copy()
    data['time_fraction'] = data['day'] + data['timestamp'] / 10000.0
    pivot = data.pivot_table(index='strike', columns='time_fraction', values='iv', aggfunc='median')
    strikes = pivot.index.values
    times = pivot.columns.values
    z = pivot.values

    # 3. MASK INVALID DATA
    # This prevents the 3D surface from "plunging" to zero where data is missing
    z_masked = np.ma.masked_invalid(z)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    T, K = np.meshgrid(times, strikes)
    
    ax.plot_surface(T, K, z_masked, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title('Cleaned IV Surface (Strike vs Time)')
    ax.set_xlabel('Day + timestamp/10000')
    ax.set_ylabel('Strike')
    ax.set_zlabel('Implied Volatility')
    
    path = OUTPUT_DIR / 'iv_surface_strike_time.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_moneyness_scatter(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = data.sample(min(len(data), 5000), random_state=42)
    sc = ax.scatter(sample['m_t_k'], sample['iv'], c=sample['day'], cmap='coolwarm', s=12, alpha=0.6)
    fig.colorbar(sc, label='Day')
    ax.set_title('IV vs Normalized Moneyness (Filtered)')
    ax.set_xlabel('m_t_k = ln(K/S) / sqrt(TTE)')
    ax.set_ylabel('Implied Volatility')
    ax.grid(True, alpha=0.35)
    path = OUTPUT_DIR / 'iv_vs_moneyness.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def print_recommendation(coeffs, residual):
    print('\n=== Recommended get_iv polynomial coefficients ===')
    formula = ' + '.join(f'{c:.6f}*x^{i}' for i, c in enumerate(coeffs))
    print(f'Polynomial: iv(m_t_k) = {formula}')
    print(f'Residual sum of squares: {residual:.6e}')


def main():
    print("Loading data...")
    prices = load_price_data()
    
    print("Building IV dataset and filtering noise...")
    iv_data = build_iv_dataset(prices)
    
    if iv_data.empty:
        print("Error: No valid IV data found after filtering.")
        return

    coeffs, residual = fit_iv_smile(iv_data, degree=2)
    print_recommendation(coeffs, residual)
    
    print("Generating plots...")
    plot_smiles(iv_data)
    plot_surface(iv_data)
    plot_moneyness_scatter(iv_data)

    out_path = OUTPUT_DIR / 'iv_data_summary_cleaned.csv'
    iv_data.to_csv(out_path, index=False)
    print(f'Process complete. Outputs saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()