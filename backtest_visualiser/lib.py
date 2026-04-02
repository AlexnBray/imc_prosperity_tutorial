# lib.pyamethu

from dash import Dash, html, Input, Output, dash_table, dcc, State, ctx
import pandas as pd
import json
import io
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import threading
import re


def parse_product_list(log_file):
    with open(log_file, "r") as file:
        # Check if it's a JSON file or Text file
        first_char = file.read(1).strip()
        file.seek(0)

        # CASE 1: JSON Wrapper (e.g., 37993.log)
        if first_char == "{":
            try:
                data = json.load(file)
                activities_log = data.get("activitiesLog", "")
                df = pd.read_csv(io.StringIO(activities_log), sep=";", header=0)
                return df["product"].unique()
            except Exception as e:
                print(f"Error parsing JSON product list: {e}")
                return []

        # CASE 2: Text Marker (e.g., 2026 log)
        else:
            file_content = file.read()
            # Safety check to ensure the markers exist
            if "Sandbox logs:" not in file_content or "Activities log:" not in file_content:
                return []
                
            sections = file_content.split("Sandbox logs:")[1].split("Activities log:")
            activities_log = sections[1].split("Trade History:")[0]
            df = pd.read_csv(io.StringIO(activities_log), sep=";", header=0)
            return df["product"].unique()


def calculate_fair(df):
    for product in df["product"].unique():
        if product == "AMETHYSTS":
            df.loc[df["product"] == product, "fair"] = 10000
        elif product == "STARFRUIT":
            df["mm_bid"] = df.apply(
                lambda x: (
                    x["bid_price_1"]
                    if x["bid_volume_1"] > 15
                    else (
                        x["bid_price_2"] if x["bid_volume_2"] > 15 else x["bid_price_3"]
                    )
                ),
                axis=1,
            )
            df["mm_ask"] = df.apply(
                lambda x: (
                    x["ask_price_1"]
                    if x["ask_volume_1"] > 15
                    else (
                        x["ask_price_2"] if x["ask_volume_2"] > 15 else x["ask_price_3"]
                    )
                ),
                axis=1,
            )
            df.loc[df["product"] == product, "fair"] = (df["mm_bid"] + df["mm_ask"]) / 2
            # Rolling Avg
            # df.loc[df['product'] == product, 'fair'] = df.loc[df['product'] == product, 'mid_price'].rolling(window=10, min_periods=1).mean()
        else:
            df.loc[df["product"] == product, "fair"] = (
                df["bid_price_1"] + df["ask_price_1"]
            ) / 2  # mid

def parse_huge_trade_history(file_path):
    # Regex to find a comma followed by a closing brace/bracket, ignoring whitespace
    # This fixes the "non-standard" JSON issue
    trailing_comma_regex = re.compile(r',\s*([\]}])')

    trade_data_str = ""
    found_section = False
    
    with open(file_path, 'r') as f:
        for line in f:
            if not found_section:
                if "Trade History:" in line:
                    found_section = True
                    # Grab everything after the label if it's on the same line
                    parts = line.split("Trade History:", 1)
                    if len(parts) > 1:
                        trade_data_str += parts[1]
                continue
            
            # Once in the section, collect the lines
            trade_data_str += line

    if not trade_data_str.strip():
        return "{}"

    # 1. Quick Regex fix for trailing commas
    fixed_json = trailing_comma_regex.sub(r'\1', trade_data_str)
    
    # 2. Parse and Normalize
    try:
        # Using pd.read_json with lines=False or json.loads
        # json.loads is usually faster for a single large blob once cleaned
        data = json.loads(fixed_json)
        return pd.json_normalize(data).to_json(orient='records')
    except Exception as e:
        print(f"Error parsing Trade History: {e}")
        return "{}"

# Pre-compile regex for speed
trailing_comma_cleanup = re.compile(r',\s*([\]}])')

def parse_log_file(log_file, product):
    activities_lines = []
    trade_history_lines = []
    sandbox_logs = []
    
    with open(log_file, "r") as file:
        first_char = file.read(1).strip()
        file.seek(0)
        
        # --- CASE 1: JSON Wrapper Format (e.g., 37993.log) ---
        if first_char == "{":
            try:
                full_data = json.load(file)
                # Extract activities
                activities_str = full_data.get("activitiesLog", "")
                df = pd.read_csv(io.StringIO(activities_str), sep=";", header=0)
                
                # Extract Trades
                trades_data = full_data.get("tradeHistory", [])
                trade_json = pd.json_normalize(trades_data).to_json(orient='records')
                
                # Extract Sandbox
                sandbox_logs = full_data.get("sandboxLogs", [])
                # If sandboxLogs is a string of JSONs, we may need to parse it
                if isinstance(sandbox_logs, str):
                    sandbox_logs = [json.loads(line) for line in sandbox_logs.strip().split('\n') if line.strip()]
            except Exception as e:
                print(f"Error parsing JSON format: {e}")
                return pd.DataFrame(), "{}", "{}"

        # --- CASE 2: Text Marker Format (e.g., 2026-04-03.log) ---
        else:
            current_section = None
            for line in file:
                if "Sandbox logs:" in line:
                    current_section = "sandbox"
                    continue
                elif "Activities log:" in line:
                    current_section = "activities"
                    continue
                elif "Trade History:" in line:
                    current_section = "trades"
                    parts = line.split("Trade History:", 1)
                    if len(parts) > 1 and parts[1].strip():
                        trade_history_lines.append(parts[1])
                    continue

                if current_section == "sandbox":
                    line_s = line.strip()
                    if line_s.startswith("{"):
                        try:
                            sandbox_logs.append(json.loads(line_s))
                        except: pass
                elif current_section == "activities":
                    activities_lines.append(line)
                elif current_section == "trades":
                    trade_history_lines.append(line)

            # Process Activities
            df = pd.read_csv(io.StringIO("".join(activities_lines)), sep=";", header=0)
            
            # Process Trades with regex fix
            trade_str = "".join(trade_history_lines).strip()
            fixed_trade_json = trailing_comma_cleanup.sub(r'\1', trade_str)
            try:
                trades_data = json.loads(fixed_trade_json)
                trade_json = pd.json_normalize(trades_data).to_json(orient='records')
            except:
                trade_json = "{}"

    # --- COMMON POST-PROCESSING ---
    if "fair" not in df.columns:
        calculate_fair(df)

    df_sandbox = pd.DataFrame(sandbox_logs)
    if not df_sandbox.empty and "lambdaLog" in df_sandbox.columns:
        # Extract patterns
        patterns = {
            'implied_bid': r'IMPLIED_BID: (\d+\.\d+)',
            'implied_ask': r'IMPLIED_ASK: (\d+\.\d+)',
            'orchids_position': r'ORCHIDS_?POSITION: (-?\d+)',
            'foreign_bid': r'FOREIGN_BID: (\d+\.\d+)',
            'foreign_ask': r'FOREIGN_ASK: (\d+\.\d+)',
            'import_tariff': r'IMPORT_TARIFF: (-?\d+\.\d+)',
            'transport_fees': r'TRANSPORT_FEES: (\d+\.\d+)'
        }

        for col, regex in patterns.items():
            # Check if keyword exists in any log entry
            keyword = col.upper().replace('ORCHIDS_', 'ORCHIDS ')
            if df_sandbox["lambdaLog"].str.contains(col.upper()[:8]).any(): 
                df_sandbox[col] = df_sandbox['lambdaLog'].apply(
                    lambda x: float(re.search(regex, x).group(1)) if isinstance(x, str) and re.search(regex, x) else None
                )
                # Merge into main df
                subset = df_sandbox.dropna(subset=[col])[['timestamp', col]].copy()
                subset['product'] = "ORCHIDS" # Ensure product matches for the merge
                df = df.merge(subset, on=['timestamp', 'product'], how='left')

        # Algo logs
        for side in ['BID', 'ASK']:
            col_name = f'algo_{side.lower()}'
            reg = fr'ALGO_{side}: (\d+\.?\d*)'
            if df_sandbox["lambdaLog"].str.contains(f"ALGO_{side}").any():
                df_sandbox[col_name] = df_sandbox['lambdaLog'].apply(
                    lambda x: float(re.search(reg, x).group(1)) if isinstance(x, str) and re.search(reg, x) else None
                )
                subset = df_sandbox.dropna(subset=[col_name])[['timestamp', col_name]].copy()
                subset['product'] = "ORCHIDS"
                df = df.merge(subset, on=['timestamp', 'product'], how='left')

    return df, trade_json, df_sandbox.to_json()