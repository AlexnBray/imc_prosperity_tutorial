import os
import glob
import tkinter as tk
from tkinter import filedialog

import dash
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from log_parser import parse_prosperity_log


# ---------------------------------------------------------
# 1. Configuration & Styles
# ---------------------------------------------------------

COLORS = {
    "mkt_bid":  {1: "#022c22", 2: "#059669", 3: "#10b981"},
    "mkt_ask":  {1: "#390808", 2: "#931616", 3: "#ff4b4b"},
    "algo_bid": {1: "#38bdf8", 2: "#0ea5e9", 3: "#075985"},
    "algo_ask": {1: "#fbbf24", 2: "#d97706", 3: "#78350f"},
    "fill_buy":  "#ffffff",
    "fill_sell": "#ffffff",
    "mid":       "#64748b",
    "fv":        "#ffffff",
    "effective_fv": "#ff00d4",
    "secondary_fv":"#00ccff",
    "pos_buy":   "#10b981",   # Green  – cumulative gross buys
    "pos_sell":  "#f43f5e",   # Rose   – cumulative gross sells
    "pos_net":   "#6366f1",   # Indigo – net position
    "pnl":       "#10b981",
    "drawdown":  "#f43f5e",
}

LEVEL_PARAMS = {
    1: {"width": 2, "opacity": 1, "dash": "solid"},
    2: {"width": 1, "opacity": 1, "dash": "solid"},
    3: {"width": 0, "opacity": 1, "dash": "solid"},
}

ADVERSE_WINDOW = 5   # timesteps forward for adverse selection measurement


# ---------------------------------------------------------
# 2. Log File Discovery
# ---------------------------------------------------------

def scan_log_directory(directory: str) -> list[dict]:
    """Return sorted dbc.Select option dicts for all .log files in a directory."""
    files = sorted(glob.glob(os.path.join(directory, "*.log")))
    return [{"label": os.path.basename(f), "value": f} for f in files]


def open_file_dialog() -> tuple[str | None, list[dict]]:
    """
    Open a Tk file-picker so the user can choose a .log file.
    The whole sibling directory is then scanned so the dropdown is pre-populated.
    Falls back gracefully if Tk is unavailable (e.g. headless server).
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select a Prosperity log file",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")],
        )
        root.destroy()
        if not selected:
            return None, []
        directory = os.path.dirname(selected)
        options = scan_log_directory(directory)
        if not options:
            options = [{"label": os.path.basename(selected), "value": selected}]
        return selected, options
    except Exception:
        return None, []


# On startup: look for .log files beside this script; prompt via picker if none found.
_script_dir     = os.path.dirname(os.path.abspath(__file__))
_initial_options = scan_log_directory(_script_dir)

if _initial_options:
    _initial_log = _initial_options[0]["value"]
else:
    _initial_log, _initial_options = open_file_dialog()

if _initial_log and os.path.exists(_initial_log):
    full_df = parse_prosperity_log(_initial_log).reset_index()
else:
    full_df = pd.DataFrame()

AVAILABLE_PRODUCTS = full_df["product"].dropna().unique().tolist() if not full_df.empty else []


# ---------------------------------------------------------
# 3. Metrics Computation
# ---------------------------------------------------------

def compute_fill_series(dff: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Accumulate gross buy and gross sell volume from algo fills over time.
    Returns (cumulative_buys, cumulative_sells) aligned to dff.index.
    Buy fills = algo filled vs ask (we bought). Sell fills = algo filled vs bid (we sold).
    """
    buy_vol  = pd.Series(0.0, index=dff.index)
    sell_vol = pd.Series(0.0, index=dff.index)
    for level in [1, 2, 3]:
        buy_col  = f"algo_ask_fill_{level}_volume"
        sell_col = f"algo_bid_fill_{level}_volume"
        if buy_col in dff.columns:
            buy_vol  += dff[buy_col].fillna(0)
        if sell_col in dff.columns:
            sell_vol += dff[sell_col].fillna(0)
    return buy_vol.cumsum(), sell_vol.cumsum()


def compute_metrics(dff: pd.DataFrame) -> dict:
    """Compute all scalar performance metrics from a filtered product DataFrame."""
    m = {}

    pnl = dff["profit_and_loss"].ffill().fillna(0)
    mid = dff["mid_price"].ffill()
    pos = dff["position"].ffill().fillna(0)

    pnl_returns = pnl.diff().fillna(0)
    mean_ret    = pnl_returns.mean()
    std_ret     = pnl_returns.std()

    # Sharpe (scaled to full episode length as a proxy for annualisation)
    m["sharpe"]     = round(mean_ret / std_ret * np.sqrt(len(pnl_returns)), 4) if std_ret > 0 else 0.0
    m["volatility"] = round(float(std_ret), 4)
    m["final_pnl"]  = round(float(pnl.iloc[-1]), 2) if not pnl.empty else 0.0

    # Drawdown
    running_max       = pnl.cummax()
    drawdown          = pnl - running_max
    m["max_drawdown"] = round(float(drawdown.min()), 2)

    # Position skew (time-weighted mean signed position)
    m["position_skew"] = round(float(pos.mean()), 2)

    # Accumulate fill volumes and vwap prices
    buy_vol      = pd.Series(0.0, index=dff.index)
    sell_vol     = pd.Series(0.0, index=dff.index)
    buy_px_vwap  = pd.Series(0.0, index=dff.index)
    sell_px_vwap = pd.Series(0.0, index=dff.index)

    for level in [1, 2, 3]:
        bv_col = f"algo_ask_fill_{level}_volume"
        sv_col = f"algo_bid_fill_{level}_volume"
        bp_col = f"algo_ask_fill_{level}_price"
        sp_col = f"algo_bid_fill_{level}_price"
        if bv_col in dff.columns:
            bv = dff[bv_col].fillna(0)
            buy_vol += bv
            if bp_col in dff.columns:
                buy_px_vwap += dff[bp_col].fillna(0) * bv
        if sv_col in dff.columns:
            sv = dff[sv_col].fillna(0)
            sell_vol += sv
            if sp_col in dff.columns:
                sell_px_vwap += dff[sp_col].fillna(0) * sv

    total_buy  = buy_vol.sum()
    total_sell = sell_vol.sum()
    m["total_buy_vol"]  = int(total_buy)
    m["total_sell_vol"] = int(total_sell)
    m["total_volume"]   = int(total_buy + total_sell)

    any_fill        = (buy_vol + sell_vol) > 0
    m["fill_rate"]  = round(float(any_fill.mean()) * 100, 2)
    m["turn_count"] = int(any_fill.sum())

    # Execution edge vs mid
    avg_buy_px  = buy_px_vwap.sum()  / total_buy  if total_buy  > 0 else np.nan
    avg_sell_px = sell_px_vwap.sum() / total_sell if total_sell > 0 else np.nan
    avg_mid_buy  = mid[buy_vol  > 0].mean() if (buy_vol  > 0).any() else np.nan
    avg_mid_sell = mid[sell_vol > 0].mean() if (sell_vol > 0).any() else np.nan

    m["buy_edge"]  = round(avg_mid_buy  - avg_buy_px,        4) if not np.isnan(avg_buy_px)  else None
    m["sell_edge"] = round(avg_sell_px  - avg_mid_sell,      4) if not np.isnan(avg_sell_px) else None

    # Adverse selection: avg mid-price change N ticks after each fill
    mid_arr    = mid.values
    buy_mask   = (buy_vol  > 0).values
    sell_mask  = (sell_vol > 0).values
    n          = len(mid_arr)
    buy_adv, sell_adv = [], []
    for i in np.where(buy_mask)[0]:
        if i + ADVERSE_WINDOW < n:
            buy_adv.append(mid_arr[i + ADVERSE_WINDOW] - mid_arr[i])
    for i in np.where(sell_mask)[0]:
        if i + ADVERSE_WINDOW < n:
            sell_adv.append(mid_arr[i + ADVERSE_WINDOW] - mid_arr[i])

    m["adverse_buy"]  = round(float(np.mean(buy_adv)),  4) if buy_adv  else None
    m["adverse_sell"] = round(float(np.mean(sell_adv)), 4) if sell_adv else None

    return m


def _metric_card(title: str, value: str, subtitle: str = "", color: str = "#f8fafc") -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.P(title,    className="small text-muted mb-1", style={"fontSize": "11px"}),
                html.H5(value,   style={"fontFamily": "monospace", "color": color, "marginBottom": "0"}),
                html.P(subtitle, className="small text-muted mb-0", style={"fontSize": "10px"}),
            ], className="py-2 px-3"),
            className="h-100",
        ),
        xs=6, sm=4, md=3, lg=2, className="mb-2",
    )


def build_metrics_cards(m: dict) -> list:
    def fmt(val, spec=".2f"):
        return f"{val:{spec}}" if val is not None else "N/A"

    def sign_color(val):
        if val is None: return "#94a3b8"
        return "#10b981" if val >= 0 else "#f43f5e"

    def adverse_color(val, is_buy):
        if val is None: return "#94a3b8"
        good = (val >= 0) if is_buy else (val <= 0)
        return "#10b981" if good else "#f43f5e"

    sharpe_color = "#10b981" if m["sharpe"] >= 1 else ("#fbbf24" if m["sharpe"] >= 0 else "#f43f5e")

    return [
        _metric_card("Final PnL",        f"{m['final_pnl']:+.2f}",
                     "XIRECS ",
                     sign_color(m["final_pnl"])),

        _metric_card("Sharpe Ratio",     fmt(m["sharpe"]),
                     f"σ={fmt(m['volatility'])}",
                     sharpe_color),

        _metric_card("Max Drawdown",     fmt(m["max_drawdown"], "+.2f"),
                     "peak → trough PnL",
                     "#f43f5e"),

        _metric_card("PnL Volatility",   fmt(m["volatility"]),
                     "std of tick PnL Δ"),

        _metric_card("Position Skew",    fmt(m["position_skew"]),
                     "time-avg signed pos",
                     "#fbbf24" if abs(m["position_skew"]) > 5 else "#10b981"),

        _metric_card("Fill Rate",        f"{fmt(m['fill_rate'])} %",
                     f"{m['turn_count']} turns"),

        _metric_card("Total Volume",     str(m["total_volume"]),
                     f"B {m['total_buy_vol']}  /  S {m['total_sell_vol']}"),

        _metric_card("Buy Edge",         fmt(m["buy_edge"]),
                     "mid − avg buy px",
                     sign_color(m["buy_edge"])),

        _metric_card("Sell Edge",        fmt(m["sell_edge"]),
                     "avg sell px − mid",
                     sign_color(m["sell_edge"])),

        _metric_card("Adverse Sel. (B)", fmt(m["adverse_buy"]),
                     f"mid Δ +{ADVERSE_WINDOW}t after buy",
                     adverse_color(m["adverse_buy"],  is_buy=True)),

        _metric_card("Adverse Sel. (S)", fmt(m["adverse_sell"]),
                     f"mid Δ +{ADVERSE_WINDOW}t after sell",
                     adverse_color(m["adverse_sell"], is_buy=False)),
    ]


# ---------------------------------------------------------
# 4. App Layout
# ---------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    id="app-wrapper",
    style={"transition": "background-color 0.3s ease"},
    children=[
        dcc.Store(id="full-data-store",    data=full_df.to_dict("records") if not full_df.empty else []),
        dcc.Store(id="filtered-data-store"),
        dcc.Store(id="log-dir-store",      data=os.path.dirname(_initial_log) if _initial_log else ""),

        dbc.Container(fluid=True, children=[

            # ---- Header Row ----
            dbc.Row([
                dbc.Col(
                    html.H3("Prosperity 4 · Quant Visualizer", className="mt-2 mb-0",
                            style={"fontFamily": "monospace", "whiteSpace": "nowrap"}),
                    width="auto",
                ),
                dbc.Col(
                    html.Div([
                        dbc.Select(
                            id="log-file-dropdown",
                            options=_initial_options,
                            value=_initial_log,
                            style={"minWidth": "180px", "maxWidth": "340px"},
                        ),
                        dbc.Button("📂 Browse", id="browse-btn", color="secondary",
                                   outline=True, size="sm", className="ms-2 text-nowrap"),
                    ], className="d-flex align-items-center mt-2"),
                    width=True,
                ),
                dbc.Col(
                    html.Div([
                        dbc.Checklist(
                            options=[{"label": "Liquidity Markers", "value": "show"}],
                            value=[], id="marker-toggle", switch=True, inline=True,
                        ),
                        dbc.Checklist(
                            options=[{"label": "Dark Mode", "value": "dark"}],
                            value=["dark"], id="theme-switch", switch=True, inline=True,
                        ),
                    ], className="d-flex justify-content-end align-items-center mt-3 gap-3"),
                    width="auto",
                ),
            ], className="border-bottom mb-3 pb-2", align="center"),

            # ---- Main Content Row ----
            dbc.Row([
                dbc.Col([
                    dbc.Select(
                        id="product-dropdown",
                        options=[{"label": p, "value": p} for p in AVAILABLE_PRODUCTS],
                        value=AVAILABLE_PRODUCTS[0] if AVAILABLE_PRODUCTS else None,
                        className="mb-2",
                    ),
                    dcc.Graph(id="price-graph",              style={"height": "600px"}),
                    dcc.Graph(id="pos-pnl-graph",            style={"height": "280px"}),
                    dcc.Graph(id="drawdown-graph", style={"height": "180px"}),

                ], width=9),

                dbc.Col([
                    html.Div(id="sync-status", className="badge bg-secondary w-100 mb-2",
                             style={"fontSize": "12px"}),

                    html.P("Market Depth", className="small fw-bold mb-1"),
                    dash_table.DataTable(
                        id="market-book-table",
                        style_table={"height": "150px", "minHeight": "150px"},
                        fixed_rows={"headers": True},
                    ),

                    html.P("Algo Quotes", className="small fw-bold mt-3 mb-1"),
                    dash_table.DataTable(
                        id="algo-book-table",
                        style_table={"height": "150px", "minHeight": "150px"},
                        fixed_rows={"headers": True},
                    ),

                    html.P("Recent Algo Fills", className="small fw-bold mt-3 mb-1"),
                    dash_table.DataTable(
                        id="fill-history-table",
                        page_size=5,
                        style_table={"height": "200px", "minHeight": "200px"},
                    ),
                    # ---- Metrics Section ----
                    html.Hr(className="my-3"),
                    html.H6("Performance Metrics", className="fw-bold mb-2",
                            style={"fontFamily": "monospace", "letterSpacing": "0.05em"}),
                    dbc.Row(id="metrics-cards", className="g-2"),
                ], width=3),
            ]),
        ]),
    ],
)


# ---------------------------------------------------------
# 5. Server Callbacks
# ---------------------------------------------------------

@app.callback(
    Output("log-file-dropdown", "options"),
    Output("log-file-dropdown", "value"),
    Output("log-dir-store",     "data"),
    Input("browse-btn", "n_clicks"),
    prevent_initial_call=True,
)
def browse_for_log(_n_clicks):
    selected, options = open_file_dialog()
    if not selected:
        raise dash.exceptions.PreventUpdate
    return options, selected, os.path.dirname(selected)


@app.callback(
    Output("full-data-store",    "data"),
    Output("product-dropdown",   "options"),
    Output("product-dropdown",   "value"),
    Input("log-file-dropdown",   "value"),
)
def load_log(log_path):
    if not log_path or not os.path.exists(log_path):
        raise dash.exceptions.PreventUpdate
    df       = parse_prosperity_log(log_path).reset_index()
    products = df["product"].dropna().unique().tolist()
    options  = [{"label": p, "value": p} for p in products]
    return df.to_dict("records"), options, products[0] if products else None


@app.callback(
    Output("filtered-data-store", "data"),
    Input("product-dropdown",     "value"),
    State("full-data-store",      "data"),
)
def filter_by_product(product, data):
    if not product or not data:
        return []
    dff = pd.DataFrame(data)
    dff = dff[dff["product"] == product].copy()
    for col in [c for c in dff.columns if "price" in c or "fv" in c]:
        dff[col] = dff[col].replace(0, np.nan)
    return dff.to_dict("records")


@app.callback(
    Output("price-graph",              "figure"),
    Output("pos-pnl-graph",            "figure"),
    Output("drawdown-graph",           "figure"),
    Output("metrics-cards",            "children"),
    Output("app-wrapper",              "style"),
    Input("filtered-data-store", "data"),
    Input("theme-switch",        "value"),
    Input("marker-toggle",       "value"),
)
def update_visuals(data, theme, markers):
    is_dark      = "dark" in (theme   or [])
    show_markers = "show" in (markers or [])
    bg_color     = "#0f172a" if is_dark else "#ffffff"
    text_color   = "#f8fafc" if is_dark else "#1e293b"
    grid_color   = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.06)"
    template     = "plotly_dark" if is_dark else "plotly_white"

    wrapper_style = {"backgroundColor": bg_color, "color": text_color, "minHeight": "100vh"}
    empty         = go.Figure()

    if not data:
        return empty, empty, empty, empty, [], wrapper_style

    dff = pd.DataFrame(data)

    base_layout = dict(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, r=10, b=10, l=10),
    )

    # ---- Price Chart ----
    vol_cols     = [c for c in dff.columns if "volume" in c]
    max_vol      = dff[vol_cols].max().max()
    base_sizeref = 2.0 * max_vol / (40 ** 2)
    l3_fallback  = 8

    fig_price = go.Figure()
    fig_price.add_trace(go.Scattergl(
        x=dff["timestamp"], y=dff["mid_price"],
        name="Mid Price", line=dict(color=COLORS["mid"], width=1),
    ))
    fig_price.add_trace(go.Scattergl(
        x=dff["timestamp"], y=dff["fv"],
        name="Fair Value", line=dict(color=COLORS["fv"], width=1.5),
    ))
    fig_price.add_trace(go.Scattergl(
        x=dff["timestamp"], y=dff["effective_fv"],
        name="Effective Fair Value", line=dict(color=COLORS["effective_fv"], width=1.5),
    ))
    fig_price.add_trace(go.Scattergl(
        x=dff["timestamp"], y=dff["secondary_fv"],
        name="signal", line=dict(color=COLORS["secondary_fv"], width=1.5),
        yaxis="y2"
    ))


    for side in ["bid", "ask"]:
        for level, style in LEVEL_PARAMS.items():
            mode = "markers" if level == 3 else ("lines+markers" if show_markers else "lines")

            mkt_sz = l3_fallback if (level == 3 and not show_markers) else dff[f"{side}_volume_{level}"]
            fig_price.add_trace(go.Scattergl(
                x=dff["timestamp"], y=dff[f"{side}_price_{level}"],
                name=f"Mkt {side.upper()} L{level}",
                legendgroup="Market", legendgrouptitle_text="Market Depth",
                mode=mode, connectgaps=False,
                line=dict(width=style["width"], color=COLORS[f"mkt_{side}"][level], dash=style["dash"]),
                marker=dict(size=mkt_sz, sizemode="area", sizeref=base_sizeref,
                            color=COLORS[f"mkt_{side}"][level], opacity=style["opacity"]),
            ))

            algo_sz = l3_fallback if (level == 3 and not show_markers) else dff[f"algo_{side}_volume_{level}"]
            fig_price.add_trace(go.Scattergl(
                x=dff["timestamp"], y=dff[f"algo_{side}_price_{level}"],
                name=f"Algo {side.upper()} L{level}",
                legendgroup="Algo", legendgrouptitle_text="Algo Quotes",
                mode=mode, connectgaps=False,
                line=dict(width=style["width"], color=COLORS[f"algo_{side}"][level],
                          dash="dot" if level < 3 else "solid"),
                marker=dict(size=algo_sz, sizemode="area", sizeref=base_sizeref,
                            symbol="circle-open", color=COLORS[f"algo_{side}"][level],
                            opacity=style["opacity"]),
            ))

    for prefix, label_prefix in [("market", "Mkt"), ("algo", "Algo")]:
        for side, label in [("bid", "Buy"), ("ask", "Sell")]:
            # Determine triangle direction based on the side
            # 'triangle-up' for Buy/Bid, 'triangle-down' for Sell/Ask
            custom_symbol = "triangle-up" if side == "bid" else "triangle-down"
            
            for level in [1, 2, 3]:
                col_p = f"{prefix}_{side}_fill_{level}_price"
                col_v = f"{prefix}_{side}_fill_{level}_volume"
                
                if col_p in dff.columns:
                    fig_price.add_trace(go.Scattergl(
                        x=dff["timestamp"], 
                        y=dff[col_p],
                        name=f"{label_prefix} {label} L{level}",
                        legendgroup="Fills", 
                        legendgrouptitle_text="Trade Fills",
                        mode="markers",
                        marker=dict(
                            size=dff[col_v], 
                            sizemode="area", 
                            sizeref=base_sizeref / 4,
                            symbol=custom_symbol, 
                            color=COLORS[f"fill_{label.lower()}"],
                            # Add black outline here
                            line=dict(
                                width=1,
                                color='black'
                            )
                        ),
                        # Optional: helps performance if you have many points
                        hoverinfo='all'
                    ))

    fig_price.update_layout(
        **base_layout,
        hovermode="x unified",
        legend=dict(groupclick="toggleitem", bgcolor=bg_color, font=dict(color=text_color)),
        uirevision=dff["product"].iloc[0] if not dff.empty else True,
        yaxis2=dict(
            title="Position",
            overlaying="y", 
            side="right", 
            showgrid=False
        )
    )

    # ---- PnL / Net Position ----
    fig_pp = go.Figure()

    # 1. PnL (Left Axis)
    fig_pp.add_trace(go.Scatter(
        x=dff["timestamp"], y=dff["profit_and_loss"],
        name="PnL", 
        line=dict(color=COLORS["pnl"]),
        showlegend=True  # Explicitly ensure it shows
    ))

    # 2. Net Position (Right Axis 1)
    fig_pp.add_trace(go.Scatter(
        x=dff["timestamp"], y=dff["position"],
        name="Net Position", 
        yaxis="y2",
        line_shape="hv", 
        line=dict(color=COLORS["pos_net"]),
        showlegend=True
    ))
    """
    # 3. Slope (Right Axis 2 - shifted or overlaid)
    fig_pp.add_trace(go.Bar(
        x=dff["timestamp"], y=dff["slope"],
        name="Slope", 
        yaxis="y3",             
        marker_color=[
            'rgba(0, 200, 100, 0.5)' if val >= 0 else 'rgba(255, 50, 50, 0.5)' 
            if pd.notnull(val) else 'rgba(0,0,0,0)' 
            for val in dff['slope']],
        showlegend=True
    ))
    """
    fig_pp.update_layout(
        **base_layout,
        hovermode="x unified",
        legend=dict(groupclick="toggleitem", bgcolor=bg_color, font=dict(color=text_color)),
        # Primary y-axis
        yaxis=dict(title="PnL"),
        # Secondary y-axis
        yaxis2=dict(
            title="Position",
            overlaying="y", 
            side="right", 
            showgrid=False
        ))
    
    """
    # Tertiary y-axis
    yaxis3=dict(
        title="Slope",
        overlaying="y", # Should overlay the primary 'y', not 'y2'
        side="left", 
        showgrid=False,
        anchor="free",  # Prevents it from overlapping the yaxis2 text
        autoshift=True
    """
  

    # ---- Position Breakdown: gross buys / gross sells / net ----
    cum_buys, cum_sells = compute_fill_series(dff)
    net_pos = dff["position"].ffill().fillna(0)

    # ---- Drawdown ----
    pnl_series  = dff["profit_and_loss"].ffill().fillna(0)
    drawdown    = pnl_series - pnl_series.cummax()

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scattergl(
        x=dff["timestamp"], y=drawdown,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color=COLORS["drawdown"], width=1),
        fillcolor="rgba(244,63,94,0.15)",
    ))
    fig_dd.add_hline(y=0, line_color="rgba(148,163,184,0.3)", line_width=1)
    fig_dd.update_layout(
        **base_layout,
        title=dict(text="PnL Drawdown", font=dict(size=12, color=text_color), x=0.01),
        hovermode="x unified",
        yaxis=dict(gridcolor=grid_color),
        xaxis=dict(gridcolor=grid_color),
    )

    # ---- Metrics Cards ----
    metrics = compute_metrics(dff)
    cards   = build_metrics_cards(metrics)

    return fig_price, fig_pp, fig_dd, cards, wrapper_style


# ---------------------------------------------------------
# 6. Clientside Callbacks
# ---------------------------------------------------------

app.clientside_callback(
    """
    function(theme) {
        const isDark = theme && theme.length > 0;
        const wrapper = document.getElementById('app-wrapper');
        if (wrapper) {
            wrapper.setAttribute('data-bs-theme', isDark ? 'dark' : 'light');
            wrapper.style.setProperty('--bs-body-bg',      isDark ? '#0f172a' : '#ffffff');
            wrapper.style.setProperty('--bs-body-color',   isDark ? '#f8fafc' : '#1e293b');
            wrapper.style.setProperty('--bs-tertiary-bg',  isDark ? '#1e293b' : '#f1f5f9');
            wrapper.style.setProperty('--bs-border-color', isDark ? '#334155' : '#dee2e6');
        }
        const style = {
            header: { backgroundColor: isDark ? '#1e293b' : '#f1f5f9', color: isDark ? 'white'   : 'black',   fontSize: '11px' },
            data:   { backgroundColor: isDark ? '#0f172a' : 'white',   color: isDark ? '#cbd5e1' : '#334155', fontSize: '11px' }
        };
        return [style.header, style.data, style.header, style.data, style.header, style.data];
    }
    """,
    [
        Output("market-book-table",  "style_header"), Output("market-book-table",  "style_data"),
        Output("algo-book-table",    "style_header"), Output("algo-book-table",    "style_data"),
        Output("fill-history-table", "style_header"), Output("fill-history-table", "style_data"),
    ],
    Input("theme-switch", "value"),
)

app.clientside_callback(
    """
    function(hoverData, data) {
        if (!hoverData || !data) return [[], [], [], "No Data"];
        const ts  = hoverData.points[0].x;
        const row = data.find(r => r.timestamp === ts);
        if (!row) return [[], [], [], "Searching..."];

        const mkt = [1,2,3].map(i => ({
            bid_vol: row[`bid_volume_${i}`]      || 0, bid_px: row[`bid_price_${i}`]         || '',
            ask_px:  row[`ask_price_${i}`]       || '', ask_vol: row[`ask_volume_${i}`]       || 0
        }));
        const algo = [1,2,3].map(i => ({
            bid_vol: row[`algo_bid_volume_${i}`] || 0, bid_px: row[`algo_bid_price_${i}`]    || '',
            ask_px:  row[`algo_ask_price_${i}`]  || '', ask_vol: row[`algo_ask_volume_${i}`] || 0
        }));

        const fills = [];
        const idx = data.indexOf(row);
        for (let j = idx; j > Math.max(0, idx - 40); j--) {
            const r = data[j];
            [1,2,3].forEach(l => {
                if (r[`algo_ask_fill_${l}_volume`] > 0)
                    fills.push({timestamp: r.timestamp, side: 'Algo Sell',  qty: r[`algo_ask_fill_${l}_volume`], price: r[`algo_ask_fill_${l}_price`]});
                if (r[`algo_bid_fill_${l}_volume`] > 0)
                    fills.push({timestamp: r.timestamp, side: 'Algo Buy', qty: r[`algo_bid_fill_${l}_volume`], price: r[`algo_bid_fill_${l}_price`]});
            });
        }
        return [mkt, algo, fills.slice(0, 8), `T = ${ts}`];
    }
    """,
    [
        Output("market-book-table",  "data"),
        Output("algo-book-table",    "data"),
        Output("fill-history-table", "data"),
        Output("sync-status",        "children"),
    ],
    Input("price-graph", "hoverData"),
    State("filtered-data-store", "data"),
)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)