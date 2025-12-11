import argparse
try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import pickle
import copy
import plotly.graph_objects as go
import pandas as pd
from utils import get_filename_for_dataset, DATASET_AVAILABLE


def main(args):
    TICKER = args.ticker
    LIMIT = args.limit
    DATASET = args.dataset

    # Load cached data
    filename = get_filename_for_dataset(DATASET)
    with open(filename, 'rb') as f:
        data_cache = pickle.load(f)

    if TICKER not in data_cache:
        raise KeyError(f"Ticker '{TICKER}' not found in {filename}")

    data = data_cache[TICKER]

    # Extract and rename OHLC columns
    df = copy.deepcopy(data[[('Open', TICKER), ('High', TICKER), ('Low', TICKER), ('Close', TICKER)]])
    df.columns = ['Open', 'High', 'Low', 'Close']
    df.index.name = 'Date'
    df.sort_index(inplace=True)  # Ensure chronological order

    # Limit to last N points
    df = copy.deepcopy(df.iloc[-LIMIT:])

    # Compute direction: compare close with previous close
    df['PrevClose'] = df['Close'].shift(1)
    df['UpDay'] = df['Close'] > df['PrevClose']

    # Split into up and down days
    df_up   = copy.deepcopy(df[df['UpDay']])
    df_down = copy.deepcopy(df[~df['UpDay']])

    # Create figure
    fig = go.Figure()

    # Add up candles (green)
    if not df_up.empty:
        fig.add_trace(go.Candlestick(
            x=df_up.index,
            open=df_up['Open'],
            high=df_up['High'],
            low=df_up['Low'],
            close=df_up['Close'],
            increasing_line_color='green',
            decreasing_line_color='green',
            name='Up',
            showlegend=False
        ))

    # Add down candles (red)
    if not df_down.empty:
        fig.add_trace(go.Candlestick(
            x=df_down.index,
            open=df_down['Open'],
            high=df_down['High'],
            low=df_down['Low'],
            close=df_down['Close'],
            increasing_line_color='red',
            decreasing_line_color='red',
            name='Down',
            showlegend=False
        ))

    fig.update_layout(
        title=f'{TICKER} Candlestick (Green: Close↑ vs Prev, Red: Close↓ vs Prev) — {DATASET.capitalize()} data',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        xaxis_type='date',
        dragmode='zoom',
        hovermode='x unified'
    )

    # --- Add light black rectangles for weekends ---
    min_date = df.index.min()
    max_date = df.index.max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    weekend_dates = all_dates[all_dates.weekday >= 5]

    # Group weekends: find Saturdays and draw rect from Sat to end of Sunday
    saturdays = weekend_dates[weekend_dates.weekday == 5]
    for sat in saturdays:
        sun = sat + pd.Timedelta(days=1)
        fig.add_vrect(
            x0=sat,
            x1=sun + pd.Timedelta(days=1),
            fillcolor="black",
            opacity=0.1,
            layer="below",
            line_width=0,
        )

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot custom candlestick chart based on OHLC data.")
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol (default: ^GSPC)')
    parser.add_argument('--limit', type=int, default=50, help='Number of most recent data points to plot (default: 50)')
    parser.add_argument('--dataset', choices=DATASET_AVAILABLE, default='day',
                        help='Dataset frequency: day, week, or month (default: day)')
    args = parser.parse_args()
    main(args)
