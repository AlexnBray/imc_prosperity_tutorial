import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OPTION_SYMBOLS = [
    #"VEV_4000", 
    #"VEV_4500", 
    "VEV_5000", 
    "VEV_5100", 
    "VEV_5200", 
    "VEV_5300", 
    "VEV_5400", 
    "VEV_5500", 
   # "VEV_6000", 
   # "VEV_6500",
]

DAY = 3
DAYS_PER_YEAR = 365

def bs_call(S, K, T, sigma, r=0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol(S, K, T, price, r=0):
    def f(sigma):
        return bs_call(S, K, T, sigma, r) - price
    try:
        return newton(f, 0.2, tol=1e-6, maxiter=100)
    except:
        return np.nan

def load_data(data_dir):
    print("Starting load_data")
    ivs = []
    for day in [0, 1, 2]:
        file = f'prices_round_3_day_{day}.csv'
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep=';')
        print(f"Loaded {len(df)} rows for day {day}")
        timestamps = df.timestamp.unique()
        print(f"Unique timestamps: {len(timestamps)}")
        for ts in timestamps[::10]:
            data = df[df.timestamp == ts]
            S_row = data[data['product'] == 'VELVETFRUIT_EXTRACT']
            if S_row.empty:
                continue
            S = S_row.mid_price.iloc[0]
            tte = 1 - (DAYS_PER_YEAR - 8 + DAY + ts // 100 / 10000) / DAYS_PER_YEAR
            if tte <= 0:
                continue
            for prod in OPTION_SYMBOLS:
                row = data[data['product'] == prod]
                if row.empty:
                    continue
                price = row.mid_price.iloc[0]
                if price <= 0:
                    continue
                K = int(prod.split('_')[1])
                iv = implied_vol(S, K, tte, price)
                if not np.isnan(iv) and iv > 0:
                    m_t_k = np.log(K / S) / np.sqrt(tte)
                    ivs.append((m_t_k, iv))
    return pd.DataFrame(ivs, columns=['moneyness', 'iv'])

if __name__ == '__main__':
    data_dir = 'Rounds/3_round/data'
    ivs_df = load_data(data_dir)
    print(f"Collected {len(ivs_df)} IV points")
    if len(ivs_df) > 10:
        coeffs = np.polyfit(ivs_df.moneyness, ivs_df.iv, 4)
        print("Fitted coefficients (a, b, c, d, e for a*x^4 + b*x^3 + c*x^2 + d*x + e):", coeffs)
        
        # Plot
        import matplotlib.pyplot as plt
        plt.scatter(ivs_df.moneyness, ivs_df.iv, alpha=0.5, label='Data')
        m_sorted = np.sort(ivs_df.moneyness)
        plt.plot(m_sorted, np.polyval(coeffs, m_sorted), color='red', label='4th degree fit')
        plt.xlabel('Moneyness (log(K/S) / sqrt(TTE))')
        plt.ylabel('Implied Volatility')
        plt.title('IV Smile')
        plt.legend()
        plt.savefig('iv_smile.png')
        print("Plot saved to iv_smile.png")
    else:
        print("Not enough data points")