import mplfinance as mpf
import pickle
from constants import FYAHOO__OUTPUTFILENAME_DAY

with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
    data_cache = pickle.load(f)

TICKER = "^GSPC"
data = data_cache[TICKER]

# Extract and rename the relevant columns for mplfinance
df = data[[('Open', TICKER), ('High', TICKER), ('Low', TICKER), ('Close', TICKER)]].copy()
df.columns = ['Open', 'High', 'Low', 'Close']
#df = df.iloc[-100:]
# Define custom market colors: green for up, red for down
market_colors = mpf.make_marketcolors(
    up='g',        # or 'green'
    down='r',      # or 'red'
    inherit=True   # ensures volume, wick, edge colors follow candle color
)

# Create a custom style using those colors
custom_style = mpf.make_mpf_style(marketcolors=market_colors)

# Plot candlestick chart with red/green colors
mpf.plot(
    df,
    type='candle',
    title=f'{TICKER} Candlestick Chart (Red/Green)',
    ylabel='Price (USD)',
    style=custom_style,
    figratio=(12, 6),
    figscale=1.0,
    show_nontrading=False
)