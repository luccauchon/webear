import pickle
import plotly.graph_objects as go
from constants import FYAHOO__OUTPUTFILENAME_DAY

# Load cached data
with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
    data_cache = pickle.load(f)

TICKER = "^GSPC"
data = data_cache[TICKER]

# Extract and rename OHLC columns
df = data[[('Open', TICKER), ('High', TICKER), ('Low', TICKER), ('Close', TICKER)]].copy()
df.columns = ['Open', 'High', 'Low', 'Close']
df.index.name = 'Date'  # Ensure index is datetime for Plotly
df.sort_index(inplace=True)  # Ensure chronological order

# Optional: limit to last N points for performance (uncomment if needed)
df = df.iloc[-200:]

# Create interactive candlestick chart
fig = go.Figure(data=go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
))

fig.update_layout(
    title=f'{TICKER} Interactive Candlestick Chart',
    yaxis_title='Price (USD)',
    xaxis_rangeslider_visible=False,  # Disable default range slider if desired
    xaxis_type='date',  # Optional: use if dates are too dense; or keep as 'date' for proper time axis
    dragmode='zoom',
    hovermode='x unified'
)

# Show in browser (interactive)
fig.show()