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
from constants import FYAHOO__OUTPUTFILENAME_MONTH
import pandas as pd
import pickle

if __name__ == "__main__":
    ticker_name = '^GSPC'
    close_col = ('Close', ticker_name)
    open_col = ('Open', ticker_name)
    negative_col = ('negative', ticker_name)          # Changed from 'positive'
    groupid_col = ('group_id', ticker_name)
    monthlyreturn_col = ('monthly_return', ticker_name)

    # Load cached monthly data
    with open(FYAHOO__OUTPUTFILENAME_MONTH, 'rb') as f:
        data_cache = pickle.load(f)

    df_full = data_cache[ticker_name]
    print(f"📉 Analyzing monthly data for {ticker_name}")
    print(f"   Period: {df_full.index[0].strftime('%Y-%m')} → {df_full.index[-1].strftime('%Y-%m')}")
    print("-" * 60)

    # Precompute all negative streaks once (for denominator)
    df_base = df_full.copy()
    df_base[negative_col] = df_base[close_col] < df_base[open_col]  # Negative month: close < open
    df_base[groupid_col] = (df_base[negative_col] != df_base[negative_col].shift(1)).cumsum()
    all_negative_groups = df_base.loc[df_base[negative_col]].groupby(groupid_col).size()
    total_negative_streaks = len(all_negative_groups)

    for NN in range(2, 10):
        df = df_base.copy()

        # Identify long negative streaks (≥ NN consecutive negative months)
        long_streak_group_ids = all_negative_groups[all_negative_groups >= NN].index

        if long_streak_group_ids.empty:
            prob = 0.0
        else:
            # Compute monthly returns (still useful for magnitude)
            df[monthlyreturn_col] = (df[close_col] - df[open_col]) / df[open_col]

            # Filter rows in long negative streaks
            long_streak_rows = df[df[groupid_col].isin(long_streak_group_ids) & df[negative_col]]

            # Summarize each streak
            grouped = long_streak_rows.groupby(groupid_col)
            streak_summary = grouped.agg(
                start_date=(groupid_col, lambda x: x.index.min()),
                end_date=(groupid_col, lambda x: x.index.max()),
                length=(groupid_col, 'size'),
                start_open=(open_col, 'first'),
                end_close=(close_col, 'last'),
                avg_monthly_return=(monthlyreturn_col, 'mean'),
                return_volatility=(monthlyreturn_col, 'std'),
            ).reset_index(drop=True)

            # Total return over the streak (%)
            streak_summary['total_return_%'] = (
                (streak_summary['end_close'] - streak_summary['start_open']) /
                streak_summary['start_open'] * 100
            ).round(2)

            # Convert to percentages
            streak_summary['avg_monthly_return_%'] = (streak_summary['avg_monthly_return'] * 100).round(2)
            streak_summary['volatility_%'] = (
                streak_summary['return_volatility'].fillna(0) * 100
            ).round(2)

            num_long_streaks = len(streak_summary)
            prob = num_long_streaks / total_negative_streaks

        print(f"• Streaks ≥{NN:2d} months: {prob:>6.2%} ({int(prob * total_negative_streaks)} out of {total_negative_streaks})")

    print("-" * 60)
    print("✅ Negative streak analysis complete.")