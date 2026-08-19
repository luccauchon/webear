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
import os
import copy
import plotly.graph_objects as go
import pandas as pd
from utils import get_next_step, factory_load_data, get_and_clean_stub_dir
from runners.streak_probability import new_main as streak_probability
from argparse import Namespace
from datetime import datetime


def main(args):
    TICKER = args.ticker
    LIMIT = args.limit
    DATASET_ID = args.dataset_id
    REALTIME = args.realtime
    GENERATE_IMAGE = args.generate_image
    GENERATE_HTML = args.generate_html
    # Load cached data
    data = factory_load_data(_dataset_id=DATASET_ID, _ticker=TICKER, _args={"realtime": REALTIME})

    # Extract and rename OHLC columns
    df = copy.deepcopy(data[[('Open', TICKER), ('High', TICKER), ('Low', TICKER), ('Close', TICKER)]])
    df.columns = ['Open', 'High', 'Low', 'Close']
    df.index.name = 'Date'
    df.sort_index(inplace=True)  # Ensure chronological order
    assert df.index.is_monotonic_increasing
    assert df.index[-1] > df.index[0]

    # Compute direction: compare close with previous close
    # NOTE: Computed BEFORE limiting to last N points so the first plotted candle has a valid previous close
    df['PrevClose'] = df['Close'].shift(1)
    df['UpDay'] = df['Close'] > df['PrevClose']

    # Limit to last N points
    df = copy.deepcopy(df.iloc[-LIMIT:])

    # Split into up and down days
    df_up = copy.deepcopy(df[df['UpDay']])
    df_down = copy.deepcopy(df[~df['UpDay']])

    sp_pos_config = Namespace(ticker=TICKER, frequency=DATASET_ID, direction="pos", max_n=10, min_n=0, delta=0, verbose=False, debug_verify_speeding=False, forward_steps=1, epsilon=0.00005)
    sp_pos = streak_probability(sp_pos_config)
    sp_neg_config = Namespace(ticker=TICKER, frequency=DATASET_ID, direction="neg", max_n=10, min_n=0, delta=0, verbose=False, debug_verify_speeding=False, forward_steps=1, epsilon=0.00005)
    sp_neg = streak_probability(sp_neg_config)

    # After 0 positive bar,  proability of having a positive bar is sp_pos[0]['prob']
    # After 1 positive bar,  proability of having a second positive bar is sp_pos[1]['prob']
    # After 2 positive bars, proability of having a third positive bar is sp_pos[2]['prob']
    # And so on
    # After 0 negative bar,  proability of having a negative bar is sp_neg[0]['prob']
    # After 1 negative bar,  proability of having a second negative bar is sp_neg[1]['prob']
    # After 2 negative bars, proability of having a third negative bar is sp_neg[2]['prob']
    # And so on

    # --- Compute streak probabilities for each candle ---
    pos_streak = 0
    neg_streak = 0
    prob_texts = []

    for i in range(len(df)):
        # The first row has no previous close to compare against, so we skip streak tracking for it
        if pd.isna(df['PrevClose'].iloc[i]):
            prob_texts.append("")
            continue

        if df['UpDay'].iloc[i]:
            idx = pos_streak
            if idx < len(sp_pos):
                prob_texts.append(f"{sp_pos[idx]['prob']:.2%}")
            else:
                prob_texts.append("N/A")

            # Update streak counters
            pos_streak += 1
            neg_streak = 0
        else:
            idx = neg_streak
            if idx < len(sp_neg):
                prob_texts.append(f"{sp_neg[idx]['prob']:.2%}")
            else:
                prob_texts.append("N/A")

            # Update streak counters
            neg_streak += 1
            pos_streak = 0

    df['prob_text'] = prob_texts

    # --- Add Next Day Candle ---
    last_row = df.iloc[-1]
    last_close = last_row['Close']

    # Calculate next date (simple 1 day increment)
    try:
        next_date = get_next_step(the_date=last_row.name, dataset_id=DATASET_ID, nn=1)
    except Exception:
        # Fallback in case of unusual index types
        next_date = str(last_row.name) + " (Next)"

    # Determine probabilities for the next day based on current streaks
    pos_idx = pos_streak if pos_streak < len(sp_pos) else len(sp_pos) - 1
    neg_idx = neg_streak if neg_streak < len(sp_neg) else len(sp_neg) - 1

    pos_prob = sp_pos[pos_idx]['prob'] if pos_idx < len(sp_pos) else 0.0
    neg_prob = sp_neg[neg_idx]['prob'] if neg_idx < len(sp_neg) else 0.0

    # Probability texts for the next (future) candle:
    # - Up arrow (↑) on the positive probability: positive means a higher close
    # - Down arrow (↓) on the negative probability: negative means a lower close
    next_pos_prob_text = f"{pos_prob:.1%}↑"
    next_neg_prob_text = f"{neg_prob:.1%}↓"

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

    # Add next day candle (blue)
    fig.add_trace(go.Candlestick(
        x=[next_date],
        open=[last_close],
        high=[last_close],
        low=[last_close],
        close=[last_close],
        increasing_line_color='blue',
        decreasing_line_color='blue',
        name='Next Day',
        showlegend=False
    ))

    # Print the probability on each candle visually using a scatter text trace
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['High'],  # Position the text at the highest wick of each candle
        text=df['prob_text'],
        mode='text',
        textposition='top center',
        textfont=dict(size=12, color='black'),  # Increased font size (was 9)
        showlegend=False,
        hoverinfo='skip'
    ))

    # Print the probability on the next day candle
    # Positive probability (chance the future candle closes higher) shown ABOVE the candle with an up arrow
    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[last_close],
        text=[next_pos_prob_text],
        mode='text',
        textposition='top center',
        textfont=dict(size=16, color='green'),  # Increased font size (was 9)
        showlegend=False,
        hoverinfo='skip'
    ))

    # Negative probability (chance the future candle closes lower) shown BELOW the candle with a down arrow
    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[last_close],
        text=[next_neg_prob_text],
        mode='text',
        textposition='bottom center',
        textfont=dict(size=16, color='red'),  # Increased font size (was 9)
        showlegend=False,
        hoverinfo='skip'
    ))
    now_str = ""
    if REALTIME:
        now_str = f" | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    fig.update_layout(
        title=f'{TICKER} Candlestick (Green: Close↑ vs Prev, Red: Close↓ vs Prev) | {DATASET_ID.capitalize()} data{now_str}',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        xaxis_type='date',
        dragmode='zoom',
        hovermode='x unified',
        margin=dict(l=60, r=100, t=100, b=60),
    )

    # --- Add light black rectangles for weekends ---
    min_date = df.index.min()
    max_date = next_date  # Include the next day in the weekend shading range
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
    output_filename = None
    # Generate a PNG of the plot if the option is specified
    if GENERATE_IMAGE:
        output_dir = get_and_clean_stub_dir(local_dir="plotly_close_oriented")
        output_filename = os.path.join(output_dir, f"{TICKER}_plot_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        fig.write_image(output_filename, width=1600, height=1000)
    if GENERATE_HTML:
        fig.show()
    return {"output_filename": output_filename}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot custom candlestick chart based on OHLC data.")
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol (default: ^GSPC)')
    parser.add_argument('--limit', type=int, default=6, help='Number of most recent data points to plot (default: 6)')
    parser.add_argument('--dataset-id', default='day', help='Dataset frequency (default: day)')
    parser.add_argument('--generate-image', action='store_true', default=False, help='Generate a PNG of the plot (default: False)')
    parser.add_argument('--realtime', action='store_true', default=False, help='')
    parser.add_argument('--generate-html', action='store_true', default=True, help='')
    args = parser.parse_args()
    main(args)