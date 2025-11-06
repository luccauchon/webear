try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # print(parent_dir)
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from constants import FYAHOO_TICKER__OUTPUTFILENAME, TOP10_SP500_TICKERS


def analyze_earnings_surprise(ticker, period='1y'):
    """
    Analyze earnings surprise vs. stock price movement

    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Data period (e.g., '1y', '2y', '5y')

    Returns:
    pd.DataFrame: Earnings analysis with price changes
    """
    # Load pre-fetched data
    with open(FYAHOO_TICKER__OUTPUTFILENAME, 'rb') as f:
        loaded_data = pickle.load(f)
    stock = loaded_data[ticker]

    # Get earnings dates
    earnings_dates = stock['earnings_dates']

    # Get today's date
    today = datetime.now().date()
    next_earnings_date = earnings_dates.index[0].date()
    days_to_next_earnings = (next_earnings_date - today).days
    print(f"\n\n\n")
    print(f"🔍 Analyzing earnings for: {ticker.upper()}")
    print(f"📅 Next earnings date: {next_earnings_date} ({days_to_next_earnings} days away)")
    print("-" * 80)

    # Fetch price history
    try:
        hist = stock['history']
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return None

    if earnings_dates is None or hist.empty:
        print(f"⚠️ No earnings or price data available for {ticker}")
        return None

    # Prepare earnings data
    earnings = earnings_dates.copy()
    earnings.index.name = 'Earnings Date'
    earnings = earnings.reset_index()
    earnings = earnings.dropna(subset=['EPS Estimate', 'Reported EPS'])

    # Calculate surprise
    earnings['Surprise'] = earnings['Reported EPS'] - earnings['EPS Estimate']
    earnings['Surprise %'] = (earnings['Surprise'] / earnings['EPS Estimate']) * 100

    # Initialize price change columns
    earnings['Close Before'] = np.nan
    earnings['Close After'] = np.nan
    earnings['1D Change %'] = np.nan
    earnings['2D Change %'] = np.nan

    # Match earnings dates with price data
    for idx, row in earnings.iterrows():
        earn_date = row['Earnings Date'].date()

        # Close before earnings
        before_dates = hist[hist.index.date < earn_date].index
        if len(before_dates) > 0:
            last_before = before_dates[-1]
            close_before = hist.loc[last_before, 'Close']
            earnings.loc[idx, 'Close Before'] = close_before
        else:
            continue

        # Close after earnings
        after_dates = hist[hist.index.date > earn_date].index
        if len(after_dates) > 0:
            first_after = after_dates[0]
            close_after = hist.loc[first_after, 'Close']
            earnings.loc[idx, 'Close After'] = close_after
            earnings.loc[idx, '1D Change %'] = ((close_after - close_before) / close_before) * 100

            if len(after_dates) > 1:
                second_after = after_dates[1]
                close_2d = hist.loc[second_after, 'Close']
                earnings.loc[idx, '2D Change %'] = ((close_2d - close_before) / close_before) * 100

    # Final clean-up
    result = earnings[[
        'Earnings Date', 'EPS Estimate', 'Reported EPS',
        'Surprise', 'Surprise %', 'Close Before', 'Close After',
        '1D Change %', '2D Change %'
    ]].copy()

    result = result.sort_values('Earnings Date', ascending=False).reset_index(drop=True)
    return result


if __name__ == "__main__":
    positive_surprise_count, positive_surprise_stocks = 0, []
    negative_surprise_count, negative_surprise_stocks = 0, []
    valid_tickers = 0

    for ticker in TOP10_SP500_TICKERS:
        results = analyze_earnings_surprise(ticker, period='2y')

        if results is not None:
            # Pretty-print the DataFrame
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.float_format', lambda x: f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}")

            print("\n📊 Earnings Surprise vs. Price Reaction")
            print("=" * 100)
            print(results.to_string(index=False, justify='right'))
            print("=" * 100)

            # Correlation analysis
            clean_data = results.dropna(subset=['Surprise %', '1D Change %'])
            if len(clean_data) > 1:
                correlation = clean_data['Surprise %'].corr(clean_data['1D Change %'])
                print(f"\n📈 Correlation (Surprise % vs. 1D Price Change): {correlation:.3f}")
            else:
                print("\n⚠️ Not enough data to compute correlation.")

            # Get the most recent earnings report (first row after sorting by date descending)
            latest = results.iloc[0]
            surprise_pct = latest['Surprise %']
            # Only count if surprise % is not NaN
            if pd.notna(surprise_pct):
                valid_tickers += 1
                if surprise_pct > 0:
                    positive_surprise_count += 1
                    positive_surprise_stocks.append(ticker)
                elif surprise_pct < 0:
                    negative_surprise_count += 1
                    negative_surprise_stocks.append(ticker)
                # Note: surprise_pct == 0 is ignored (rare, but possible)
        else:
            print("❌ Analysis failed or no data returned.")

    print("\n" + "=" * 60)
    print(f"✅ Total tickers with valid latest earnings surprise: {valid_tickers}")
    print(f"🟢 Stocks with positive earnings surprise: {positive_surprise_count}  {positive_surprise_stocks}")
    print(f"🔴 Stocks with negative earnings surprise: {negative_surprise_count}  {negative_surprise_stocks}")
    print("=" * 60)
