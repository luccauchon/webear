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
import argparse
import pickle
from constants import FYAHOO__OUTPUTFILENAME_MONTH

def main(ticker: str, year_threshold: int):
    col = "Close"
    colname = (col, ticker)
    filepath = FYAHOO__OUTPUTFILENAME_MONTH

    # Load data
    with open(filepath, 'rb') as f:
        data_cache = pickle.load(f)
    df = data_cache[ticker].copy()
    df = df.sort_index()

    # The list of years you want to keep
    years_to_keep = [yy for yy in range(year_threshold, 9999)]

    # Create a boolean mask: True for rows to KEEP, False for rows to drop
    # The .dt accessor is used to access datetime properties like 'year'
    mask = df.index.year.isin(years_to_keep)

    # Apply the mask with the negation operator (~) to select rows in the list
    df = df[mask].copy()

    # Compute monthly returns
    return_col = ('Return', ticker)
    df[return_col] = df[colname].pct_change()
    df = df.dropna(subset=[return_col])

    # Add month name
    month_name_col = ('MonthName', ticker)
    df[month_name_col] = df.index.strftime('%B')

    # --- 1. Direction Summary (Positive/Negative Count) ---
    sign_col = ('Sign', ticker)
    df[sign_col] = df[return_col].apply(lambda x: 'Positive' if x > 0 else 'Negative')

    direction_summary = df.groupby([month_name_col, sign_col]).size().unstack(fill_value=0)
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    direction_summary = direction_summary.reindex(month_order, fill_value=0)
    direction_summary['Total'] = direction_summary.sum(axis=1)

    # --- 2. Historical Monthly Return Stats ---
    # 🔧 FIX: Wrap return_col in a LIST to avoid ValueError
    grouped = df.groupby(month_name_col)[[return_col]]  # Note the double brackets!

    mean_return = grouped.mean()[return_col]
    median_return = grouped.median()[return_col]
    positive_pct = grouped.apply(lambda x: (x[return_col] > 0).mean())

    stats_df = pd.DataFrame({
        'Mean Return': mean_return,
        'Median Return': median_return,
        'Positive %': positive_pct
    })
    stats_df = stats_df.reindex(month_order)

    # Format as percentages
    stats_df['Mean Return'] = stats_df['Mean Return'].apply(lambda x: f"{x:+.2%}")
    stats_df['Median Return'] = stats_df['Median Return'].apply(lambda x: f"{x:+.2%}")
    stats_df['Positive %'] = stats_df['Positive %'].apply(lambda x: f"{x:.0%}")

    # --- Output ---
    print(f"\n📈 Historical Monthly Return Analysis for: {ticker}")
    print("=" * 60)

    print("\n📊 Frequency of Positive vs Negative Months:")
    print(direction_summary.to_string())

    print("\n📈 Average Monthly Returns by Calendar Month:")
    print(stats_df.to_string())

    # Optional: Show last 12 actual monthly returns
    print(f"\n📆 Last 12 Monthly Returns:")
    print("-" * 30)
    sample = df[[return_col]].tail(12).copy()
    sample.columns = ['Return']
    sample['Return'] = sample['Return'].apply(lambda x: f"{x:+.2%}")
    sample.index.name = 'Month'
    print(sample.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze historical monthly returns by calendar month.")
    parser.add_argument(
        "--ticker",
        type=str,
        default="^GSPC",
        help="Ticker symbol (default: ^GSPC)"
    )
    parser.add_argument(
        "--year",
        type=str,
        default="1950",
        help="Keep only the years equals and above"
    )
    args = parser.parse_args()
    main(args.ticker, year_threshold=int(args.year))