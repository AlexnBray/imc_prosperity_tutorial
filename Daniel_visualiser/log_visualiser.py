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
    # Brightest/Saturated for L1 -> Darker/Muted for L3
    "mkt_bid":  {1: "#022c22", 2: "#059669", 3: "#10b981"},  # Emerald Green
    "mkt_ask":  {1: "#390808", 2: "#931616", 3: "#ff4b4b"},   # Bright Red
    "algo_bid": {1: "#38bdf8", 2: "#0ea5e9", 3: "#075985"},   # Sky Blue
    "algo_ask": {1: "#fbbf24", 2: "#d97706", 3: "#78350f"},   # Amber
    "fill_buy":  "#22d3ee",
    "fill_sell": "#e879f9",
    "mid": "#64748b",  # Slightly darker mid-price to stay in background
    "fv":  "#ffffff",  # White / high-contrast for Fair Value
}

# Level-specific rendering config
LEVEL_PARAMS = {
    1: {"width": 2, "opacity": 1, "dash": "solid"},
    2: {"width": 1, "opacity": 1, "dash": "solid"},
    3: {"width": 0, "opacity": 1, "dash": "solid"},
}


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
_script_dir = os.path.dirname(os.path.abspath(__file__))
_initial_options = scan_log_directory(_script_dir)

if _initial_options:
    _initial_log = _initial_options[0]["value"]
else:
    _initial_log, _initial_options = open_file_dialog()

# Load initial data (may be empty if user cancelled or no logs exist)
if _initial_log and os.path.exists(_initial_log):
    full_df = parse_prosperity_log(_initial_log).reset_index()
else:
    full_df = pd.DataFrame()

AVAILABLE_PRODUCTS = full_df["product"].dropna().unique().tolist() if not full_df.empty else []


# ---------------------------------------------------------
# 3. App Layout
# ---------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    id="app-wrapper",
    style={"transition": "background-color 0.3s ease"},
    children=[
        # Stores
        dcc.Store(id="full-data-store", data=full_df.to_dict("records") if not full_df.empty else []),
        dcc.Store(id="filtered-data-store"),
        dcc.Store(id="log-dir-store", data=os.path.dirname(_initial_log) if _initial_log else ""),

        dbc.Container(fluid=True, children=[

            # ---- Header Row ----
            dbc.Row([
                # Title
                dbc.Col(
                    html.H3(
                        "Prosperity 4 · Quant Visualizer",
                        className="mt-2 mb-0",
                        style={"fontFamily": "monospace", "whiteSpace": "nowrap"},
                    ),
                    width="auto",
                ),

                # Log file selector + browse button
                dbc.Col(
                    html.Div([
                        dbc.Select(
                            id="log-file-dropdown",
                            options=_initial_options,
                            value=_initial_log,
                            style={"minWidth": "180px", "maxWidth": "340px"},
                        ),
                        dbc.Button(
                            "📂 Browse",
                            id="browse-btn",
                            color="secondary",
                            outline=True,
                            size="sm",
                            className="ms-2 text-nowrap",
                        ),
                    ], className="d-flex align-items-center mt-2"),
                    width=True,
                ),

                # Controls
                dbc.Col(
                    html.Div([
                        dbc.Checklist(
                            options=[{"label": "Liquidity Markers", "value": "show"}],
                            value=[],
                            id="marker-toggle",
                            switch=True,
                            inline=True,
                        ),
                        dbc.Checklist(
                            options=[{"label": "Dark Mode", "value": "dark"}],
                            value=["dark"],
                            id="theme-switch",
                            switch=True,
                            inline=True,
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
                    dcc.Graph(id="price-graph", style={"height": "600px"}),
                    dcc.Graph(id="pos-pnl-graph", style={"height": "300px"}),
                ], width=9),

                dbc.Col([
                    html.Div(
                        id="sync-status",
                        className="badge bg-secondary w-100 mb-2",
                        style={"fontSize": "12px"},
                    ),

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
                ], width=3),
            ]),
        ]),
    ],
)


# ---------------------------------------------------------
# 4. Server Callbacks
# ---------------------------------------------------------

@app.callback(
    Output("log-file-dropdown", "options"),
    Output("log-file-dropdown", "value"),
    Output("log-dir-store", "data"),
    Input("browse-btn", "n_clicks"),
    prevent_initial_call=True,
)
def browse_for_log(_n_clicks):
    """Open OS file picker, scan the chosen directory, update the log dropdown."""
    selected, options = open_file_dialog()
    if not selected:
        raise dash.exceptions.PreventUpdate
    new_dir = os.path.dirname(selected)
    return options, selected, new_dir


@app.callback(
    Output("full-data-store", "data"),
    Output("product-dropdown", "options"),
    Output("product-dropdown", "value"),
    Input("log-file-dropdown", "value"),
)
def load_log(log_path):
    """Parse the selected log file and repopulate the product dropdown."""
    if not log_path or not os.path.exists(log_path):
        raise dash.exceptions.PreventUpdate
    df = parse_prosperity_log(log_path).reset_index()
    products = df["product"].dropna().unique().tolist()
    options = [{"label": p, "value": p} for p in products]
    return df.to_dict("records"), options, products[0] if products else None


@app.callback(
    Output("filtered-data-store", "data"),
    Input("product-dropdown", "value"),
    State("full-data-store", "data"),
)
def filter_by_product(product, data):
    if not product or not data:
        return []
    dff = pd.DataFrame(data)
    dff = dff[dff["product"] == product].copy()
    price_cols = [c for c in dff.columns if "price" in c or "fv" in c]
    for col in price_cols:
        dff[col] = dff[col].replace(0, np.nan)
    return dff.to_dict("records")


@app.callback(
    Output("price-graph", "figure"),
    Output("pos-pnl-graph", "figure"),
    Output("app-wrapper", "style"),
    Input("filtered-data-store", "data"),
    Input("theme-switch", "value"),
    Input("marker-toggle", "value"),
)
def update_visuals(data, theme, markers):
    is_dark = "dark" in (theme or [])
    show_markers = "show" in (markers or [])
    bg_color = "#0f172a" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#1e293b"
    template = "plotly_dark" if is_dark else "plotly_white"

    wrapper_style = {"backgroundColor": bg_color, "color": text_color, "minHeight": "100vh"}

    if not data:
        return go.Figure(), go.Figure(), wrapper_style

    dff = pd.DataFrame(data)
    vol_cols = [c for c in dff.columns if "volume" in c]
    max_vol = dff[vol_cols].max().max()
    base_sizeref = 2.0 * max_vol / (40 ** 2)
    l3_fallback_size = 8  # L3 is sparse so can't use a continuous line; fixed fallback size

    fig_price = go.Figure()

    # Reference lines
    fig_price.add_trace(go.Scattergl(
        x=dff["timestamp"], y=dff["mid_price"],
        name="Mid Price",
        line=dict(color=COLORS["mid"], width=1),
    ))
    fig_price.add_trace(go.Scattergl(
        x=dff["timestamp"], y=dff["fv"],
        name="Fair Value",
        line=dict(color=COLORS["fv"], width=1.5),
    ))

    # Market depth & algo quotes
    for side in ["bid", "ask"]:
        for level, style in LEVEL_PARAMS.items():
            # L3 always uses markers (gapped series); L1/L2 follow the toggle
            mode = "markers" if level == 3 else ("lines+markers" if show_markers else "lines")

            mkt_size = (
                l3_fallback_size if (level == 3 and not show_markers)
                else dff[f"{side}_volume_{level}"]
            )
            fig_price.add_trace(go.Scattergl(
                x=dff["timestamp"], y=dff[f"{side}_price_{level}"],
                name=f"Mkt {side.upper()} L{level}",
                legendgroup="Market", legendgrouptitle_text="Market Depth",
                mode=mode, connectgaps=False,
                line=dict(width=style["width"], color=COLORS[f"mkt_{side}"][level], dash=style["dash"]),
                marker=dict(
                    size=mkt_size, sizemode="area", sizeref=base_sizeref,
                    color=COLORS[f"mkt_{side}"][level], opacity=style["opacity"],
                ),
            ))

            algo_size = (
                l3_fallback_size if (level == 3 and not show_markers)
                else dff[f"algo_{side}_volume_{level}"]
            )
            fig_price.add_trace(go.Scattergl(
                x=dff["timestamp"], y=dff[f"algo_{side}_price_{level}"],
                name=f"Algo {side.upper()} L{level}",
                legendgroup="Algo", legendgrouptitle_text="Algo Quotes",
                mode=mode, connectgaps=False,
                line=dict(
                    width=style["width"],
                    color=COLORS[f"algo_{side}"][level],
                    dash="dot" if level < 3 else "solid",
                ),
                marker=dict(
                    size=algo_size, sizemode="area", sizeref=base_sizeref,
                    symbol="circle-open",
                    color=COLORS[f"algo_{side}"][level], opacity=style["opacity"],
                ),
            ))

    # Trade fills
    for prefix, symbol, label_prefix in [("market", "diamond", "Mkt"), ("algo", "x", "Algo")]:
        for side, label in [("ask", "Buy"), ("bid", "Sell")]:
            for level in [1, 2, 3]:
                col_p = f"{prefix}_{side}_fill_{level}_price"
                col_v = f"{prefix}_{side}_fill_{level}_volume"
                if col_p in dff.columns:
                    fig_price.add_trace(go.Scattergl(
                        x=dff["timestamp"], y=dff[col_p],
                        name=f"{label_prefix} {label} L{level}",
                        legendgroup="Fills", legendgrouptitle_text="Trade Fills",
                        mode="markers",
                        marker=dict(
                            size=dff[col_v], sizemode="area", sizeref=base_sizeref / 4,
                            symbol=symbol, color=COLORS[f"fill_{label.lower()}"],
                        ),
                    ))

    fig_price.update_layout(
        template=template,
        margin=dict(t=30, r=10, b=10, l=10),
        hovermode="x unified",
        legend=dict(groupclick="toggleitem", bgcolor=bg_color, font=dict(color=text_color)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # uirevision tied to product so zoom survives theme/marker toggles
        uirevision=dff["product"].iloc[0] if not dff.empty else True,
    )

    # PnL / Position graph
    fig_pp = go.Figure()
    fig_pp.add_trace(go.Scattergl(
        x=dff["timestamp"], y=dff["profit_and_loss"],
        name="PnL", line=dict(color="#10b981"),
    ))
    fig_pp.add_trace(go.Scattergl(
        x=dff["timestamp"], y=dff["position"],
        name="Position", yaxis="y2",
        line_shape="hv", line=dict(color="#6366f1"),
    ))
    fig_pp.update_layout(
        template=template,
        yaxis2=dict(overlaying="y", side="right"),
        margin=dict(t=10, r=10, b=10, l=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig_price, fig_pp, wrapper_style


# ---------------------------------------------------------
# 5. Clientside Callbacks
# ---------------------------------------------------------

app.clientside_callback(
    """
    function(theme) {
        const isDark = theme && theme.length > 0;
        const wrapper = document.getElementById('app-wrapper');
        wrapper.setAttribute('data-bs-theme', isDark ? 'dark' : 'light');
        // Override Bootstrap's dark theme surface colours to match your slate palette
        wrapper.style.setProperty('--bs-body-bg',         isDark ? '#0f172a' : '#ffffff');
        wrapper.style.setProperty('--bs-body-color',       isDark ? '#f8fafc' : '#1e293b');
        wrapper.style.setProperty('--bs-tertiary-bg',      isDark ? '#1e293b' : '#f1f5f9');
        wrapper.style.setProperty('--bs-border-color',     isDark ? '#334155' : '#dee2e6');
        wrapper.style.setProperty('--bs-form-select-bg',   isDark ? '#1e293b' : '#ffffff')
        const style = {
            header: { backgroundColor: isDark ? '#1e293b' : '#f1f5f9', color: isDark ? 'white' : 'black', fontSize: '11px' },
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
                    fills.push({timestamp: r.timestamp, side: 'Algo Buy',  qty: r[`algo_ask_fill_${l}_volume`], price: r[`algo_ask_fill_${l}_price`]});
                if (r[`algo_bid_fill_${l}_volume`] > 0)
                    fills.push({timestamp: r.timestamp, side: 'Algo Sell', qty: r[`algo_bid_fill_${l}_volume`], price: r[`algo_bid_fill_${l}_price`]});
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