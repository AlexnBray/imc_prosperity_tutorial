import pandas as pd
import json
import io
import numpy as np

def parse_prosperity_log(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 1. Parse Market Data
    csv_data = data.get('activitiesLog', '')
    df_market = pd.read_csv(io.StringIO(csv_data), sep=';')

    # 2. Parse Algorithm Logs
    algo_records = []
    for log_entry in data.get('logs', []):
        lambda_log = log_entry.get('lambdaLog', '')
        for line in lambda_log.split('\n'):
            if line.startswith('[ALGO]'):
                parts = line.split(',')
                if len(parts) >= 7:
                    record = {
                        'timestamp': int(parts[1]),
                        'product': parts[2],
                        'position': int(parts[3]),
                        'fv': float(parts[4]),
                        'effective_fv': float(parts[5]),
                        'slope': float(parts[6])
                    }
                    bids = parts[7].strip('[]').split(';') if parts[7].strip('[]') else []
                    asks = parts[8].strip('[]').split(';') if parts[8].strip('[]') else []
                    for i in range(3):
                        if i < len(bids):
                            p, v = bids[i].split(':')
                            record[f'algo_bid_price_{i+1}'] = float(p)
                            record[f'algo_bid_volume_{i+1}'] = abs(int(v))
                        if i < len(asks):
                            p, v = asks[i].split(':')
                            record[f'algo_ask_price_{i+1}'] = float(p)
                            record[f'algo_ask_volume_{i+1}'] = abs(int(v))
                    algo_records.append(record)
    df_algo_logs = pd.DataFrame(algo_records)

    # Ensure all expected algo columns exist to avoid KeyErrors later
    expected_algo_cols = []
    for i in range(1, 4):
        expected_algo_cols.extend([f'algo_bid_price_{i}', f'algo_bid_volume_{i}', 
                                   f'algo_ask_price_{i}', f'algo_ask_volume_{i}'])
    
    for col in expected_algo_cols:
        if col not in df_algo_logs.columns:
            df_algo_logs[col] = np.nan

    # 3. Vectorized Trade Processing
    trades = data.get('tradeHistory', [])
    if not trades:
        # If no trades, just merge market and algo logs
        df_final = df_market.copy()
        if not df_algo_logs.empty:
            df_final = pd.merge(df_final, df_algo_logs, on=['timestamp', 'product'], how='left')
        return df_final.fillna(0).set_index('timestamp').sort_index()

    df_trades = pd.DataFrame(trades).rename(columns={'symbol': 'product'})
    
    # Pre-merge Market and Algo logs to create a lookup table
    df_lookup = df_market.copy()
    if not df_algo_logs.empty:
        df_lookup = pd.merge(df_lookup, df_algo_logs, on=['timestamp', 'product'], how='left')

    # Merge trades with the lookup table
    df_trades = pd.merge(df_trades, df_lookup, on=['timestamp', 'product'], how='left')

    # Determine Role
    df_trades['is_algo'] = (df_trades['buyer'] == 'SUBMISSION') | (df_trades['seller'] == 'SUBMISSION')
    df_trades['role'] = np.where(df_trades['is_algo'], "algo", "market")

    # Vectorized Level/Side Detection
    # Logic: If algo trade, check 'algo_' columns; if market trade, check standard columns
    df_trades['level'] = "3" # Default
    df_trades['side_type'] = np.where(df_trades['buyer'] == 'SUBMISSION', "ask", "bid") # Initial guess

    for i in [1, 2, 3]:
        # Conditions for Bid matches
        is_bid_match = (
            ((df_trades['role'] == "algo") & (df_trades['price'] == df_trades[f'algo_bid_price_{i}'])) |
            ((df_trades['role'] == "market") & (df_trades['price'] == df_trades[f'bid_price_{i}']))
        )
        df_trades.loc[is_bid_match, 'level'] = str(i)
        df_trades.loc[is_bid_match, 'side_type'] = "bid"

        # Conditions for Ask matches
        is_ask_match = (
            ((df_trades['role'] == "algo") & (df_trades['price'] == df_trades[f'algo_ask_price_{i}'])) |
            ((df_trades['role'] == "market") & (df_trades['price'] == df_trades[f'ask_price_{i}']))
        )
        df_trades.loc[is_ask_match, 'level'] = str(i)
        df_trades.loc[is_ask_match, 'side_type'] = "ask"

    # Create Fill Columns
    df_trades['fill_col'] = df_trades['role'] + "_" + df_trades['side_type'] + "_fill_" + df_trades['level']
    
    # Pivot trades to get volume and price columns per level
    # We use pivot_table to aggregate multiple fills at the same timestamp
    df_vol = df_trades.pivot_table(index=['timestamp', 'product'], columns='fill_col', values='quantity', aggfunc='sum')
    df_vol.columns = [f"{c}_volume" for c in df_vol.columns]
    
    df_px = df_trades.pivot_table(index=['timestamp', 'product'], columns='fill_col', values='price', aggfunc='mean')
    df_px.columns = [f"{c}_price" for c in df_px.columns]

    df_trades_agg = pd.concat([df_vol, df_px], axis=1).reset_index()

    expected_fills = []
    for role in ['algo', 'market']:
        for side in ['bid', 'ask']:
            for lvl in ['1', '2', '3']:
                expected_fills.append(f"{role}_{side}_fill_{lvl}")

    # Ensure every expected volume and price column exists
    for fill in expected_fills:
        vol_col = f"{fill}_volume"
        px_col = f"{fill}_price"
        
        if vol_col not in df_trades_agg.columns:
            df_trades_agg[vol_col] = 0.0 # if not trades occured volume is 0
        if px_col not in df_trades_agg.columns:
            df_trades_agg[px_col] = np.nan # price is not 0 though if no trades occured

    df_final = pd.merge(df_lookup, df_trades_agg, on=['timestamp', 'product'], how='left')
    
    df_final.fillna(0, inplace=True)
    df_final.set_index('timestamp', inplace=True)
    df_final.sort_index(inplace=True)

    return df_final