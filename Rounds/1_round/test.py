import sys
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import re

# --- Helper Functions for Indicators ---

def calculate_book_vwap(df):
    nominal = 0
    volume = 0
    for i in range(1, 4):
        nominal += (df[f'bid_price_{i}'] * df[f'bid_volume_{i}']).fillna(0)
        nominal += (df[f'ask_price_{i}'] * df[f'ask_volume_{i}']).fillna(0)
        volume += df[f'bid_volume_{i}'].fillna(0)
        volume += df[f'ask_volume_{i}'].fillna(0)
    return nominal / volume

def calculate_ewma(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def apply_kalman_filter(series, process_variance=1e-5, measurement_variance=1e-3):
    """
    Simple 1D Kalman Filter (Local Level Model).
    High measurement_variance (R) = Smoother/Slower.
    High process_variance (Q) = More reactive/Jittery.
    """
    n_iter = len(series)
    xhat = np.zeros(n_iter)      # posterior estimate
    P = np.zeros(n_iter)         # posterior error estimate
    xhatminus = np.zeros(n_iter) # prior estimate
    Pminus = np.zeros(n_iter)    # prior error estimate
    K = np.zeros(n_iter)         # Kalman gain

    xhat[0] = series.iloc[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # Predict
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + process_variance
        # Update
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
        xhat[k] = xhatminus[k] + K[k] * (series.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat

# --- Data Loading ---

files = [
    'rounds/1_round/data/prices_round_1_day_-1.csv',
    'rounds/1_round/data/prices_round_1_day_-2.csv',
    'rounds/1_round/data/prices_round_1_day_0.csv'
]

prices = []
for file in files:
    try:
        df = pd.read_csv(file, sep=';')
        day_match = re.search(r'day_(-?\d+)', file)
        if day_match:
            df['day'] = int(day_match.group(1))
        prices.append(df)
    except FileNotFoundError:
        print(f"Warning: {file} not found.")

df_total = pd.concat(prices, ignore_index=True)
df_total = df_total.sort_values(by=['day', 'timestamp']).reset_index(drop=True)

# --- Visualization Logic ---

fig = go.Figure()
products = df_total['product'].unique()
days = sorted(df_total['day'].unique())
buttons = []
current_trace_idx = 0

for product in products:
    for day in days:
        subset = df_total[(df_total['product'] == product) & (df_total['day'] == day)].copy()
        if subset.empty:
            continue
            
        traces_in_this_group = 0
        
        # Calculate Indicators
        subset['VWAP'] = calculate_book_vwap(subset)
        subset['MID'] = (subset['bid_price_1'] + subset['ask_price_1']) / 2
        
        # Apply Filters to VWAP
        subset['EWMA_VWAP'] = calculate_ewma(subset['VWAP'], span=40)
        subset['KALMAN_VWAP'] = apply_kalman_filter(subset['VWAP'], measurement_variance=0.05)
        
        # Apply Filters to MID
        subset['EWMA_MID'] = calculate_ewma(subset['MID'], span=40)
        subset['KALMAN_MID'] = apply_kalman_filter(subset['MID'], measurement_variance=0.05)

        # 1. Main Price Traces
        fig.add_trace(go.Scatter(x=subset['timestamp'], y=subset['VWAP'], 
                      name='VWAP', line=dict(color='white', width=1.5), visible=False))
        traces_in_this_group += 1

        fig.add_trace(go.Scatter(x=subset['timestamp'], y=subset['KALMAN_VWAP'], 
                      name='Kalman (VWAP)', line=dict(color='#FFD700', width=2), visible=False))
        traces_in_this_group += 1

        fig.add_trace(go.Scatter(x=subset['timestamp'], y=subset['EWMA_VWAP'], 
                      name='EWMA (VWAP)', line=dict(color='cyan', width=2, dash='dot'), visible=False))
        traces_in_this_group += 1

        # 2. Book Depth (Bids/Asks)
        colors = {'bid': ['#225522', '#338833', '#44BB44'], 
                  'ask': ['#552222', '#883333', '#BB4444']}
        
        for side in ['bid', 'ask']:
            for level in ['1', '2', '3']:
                col_name = f'{side}_price_{level}'
                if col_name in subset.columns:
                    fig.add_trace(go.Scatter(
                        x=subset['timestamp'], y=subset[col_name],
                        name=f'{side.capitalize()} {level}',
                        line=dict(color=colors[side][int(level)-1], width=1),
                        visible=False, opacity=0.4
                    ))
                    traces_in_this_group += 1

        # Dropdown Mask Logic
        group_start = current_trace_idx
        group_end = current_trace_idx + traces_in_this_group
        
        buttons.append(dict(
            label=f"{product} (D{day})",
            method="update",
            args=[{"visible": (group_start, group_end)}, 
                  {"title": f"Order Book Analysis: {product} - Day {day}"}]
        ))
        current_trace_idx += traces_in_this_group

# Finalize Visibility Masks
total_traces = len(fig.data)
for btn in buttons:
    start, end = btn['args'][0]['visible']
    mask = [False] * total_traces
    for j in range(start, end):
        mask[j] = True
    btn['args'][0]['visible'] = mask

# Default View
if total_traces > 0:
    first_group_size = sum(buttons[0]['args'][0]['visible'])
    for i in range(first_group_size):
        fig.data[i].visible = True

fig.update_layout(
    updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.15, xanchor='left')],
    xaxis_title="Timestamp",
    yaxis_title="Price",
    template="plotly_dark",
    hovermode="x unified",
    height=800
)

fig.show()