import sys
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

import pandas as pd
import re

files = [
    'rounds/1_round/data/prices_round_1_day_-1.csv',
    'rounds/1_round/data/prices_round_1_day_-2.csv',
    'rounds/1_round/data/prices_round_1_day_0.csv'
]

prices = []

for file in files:
    # 1. Load the data
    df = pd.read_csv(file, sep=';')
    
    # 2. Extract the day from the filename using a simple split or Regex
    # This looks for a minus sign (optional) followed by digits at the end of the string
    day_match = re.search(r'day_(-?\d+)', file)
    if day_match:
        day_val = int(day_match.group(1))
        # Add/Overwrite the day column to ensure it's correct
        df['day'] = day_val
        
    prices.append(df)

# 3. Combine everything into one master DataFrame
df_total = pd.concat(prices, ignore_index=True)

# 4. Sort by Day and Timestamp so technical indicators (like SMAs) work correctly
df_total = df_total.sort_values(by=['day', 'timestamp']).reset_index(drop=True)



df_plot = df_total.copy()

# 2. Initialize the figure
fig = go.Figure()

# Get unique products and days to build the dropdown
products = df_plot['product'].unique()
days = df_plot['day'].unique()

# 3. Add a trace for every combination of Product and Day
trace_metadata = []
for product in products:
    for day in days:
        subset = df_plot[(df_plot['product'] == product) & (df_plot['day'] == day)]
        
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['timestamp'],
                y=subset['mid_price'],
                mode='lines+markers',
                name=f"{product} | Day {day}",
                visible=False  # Hide all initially
            ))
            trace_metadata.append((product, day))

# Show the first trace by default
if fig.data:
    fig.data[0].visible = True

# 4. Create the Dropdown Menu (updatemenus)
buttons = []
for i, (prod, d) in enumerate(trace_metadata):
    # Visibility list: True for the current index, False for all others
    visibility = [False] * len(fig.data)
    visibility[i] = True
    
    button = dict(
        label=f"{prod} (Day {d})",
        method="update",
        args=[
            {"visible": visibility}, # Update visibility of traces
            {"title": f"Mid Price: {prod} - Day {d}"} # Update plot title
        ]
    )
    buttons.append(button)

# 5. Finalize Layout
fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.0,
        xanchor="left",
        y=1.15,
        yanchor="top"
    )],
    title=f"Price Visualization: {trace_metadata[0][0]} - Day {trace_metadata[0][1]}",
    xaxis_title="Timestamp",
    yaxis_title="Mid Price",
    template="plotly_white",
    hovermode="x unified"
)

# Display the plot
fig.show()
          
#price_df = load_csv(files[0])
#price_df.head(5)


df = prices_obj[2]['ASH_COATED_OSMIUM']



'''

price_graph = go.Figure()
price_graph.add_trace(go.Scatter(x=ash_coated_osmium_df.index, y=ash_coated_osmium_df['mid_price'], mode='lines', name='Mid Price'))
price_graph.add_trace(go.Scatter(x=ash_coated_osmium_df.index, y=ash_coated_osmium_df['micro_price'], mode='lines', name='Micro Price'))
#price_graph.add_trace(go.Scatter(x=tomatoe_price_df.index, y=tomatoe_price_df['l3_vwap_15'], mode='lines', name='L3 VWAP (15)'))

price_graph.add_trace(go.Scatter(x=ash_coated_osmium_df.index, y=ash_coated_osmium_df['bid_price_1'], mode='lines', name='Best Bid Price'))
price_graph.add_trace(go.Scatter(x=ash_coated_osmium_df.index, y=ash_coated_osmium_df['bid_price_2'], mode='lines', name='Level 2 Bid Price'))
price_graph.add_trace(go.Scatter(x=ash_coated_osmium_df.index, y=ash_coated_osmium_df['bid_price_3'], mode='markers', name='Level 3 Bid Price'))

price_graph.add_trace(go.Scatter(x=ash_coated_osmium_df.index, y=ash_coated_osmium_df['ask_price_1'], mode='lines', name='Best Ask Price'))
price_graph.add_trace(go.Scatter(x=ash_coated_osmium_df.index, y=ash_coated_osmium_df['ask_price_2'], mode='lines', name='Level 2 Ask Price'))
price_graph.add_trace(go.Scatter(x=ash_coated_osmium_df.index, y=ash_coated_osmium_df['ask_price_3'], mode='markers', name='Level 3 Ask Price'))
price_graph.show()

'''