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
from datetime import datetime
import pandas as pd
import pickle
from tabulate import tabulate

if __name__ == "__main__":
    ticker_name = '^GSPC'
    close_col = ('Close', ticker_name)
    open_col = ('Open', ticker_name)
    positive_col = ('positive', ticker_name)
    groupid_col = ('group_id', ticker_name)
    monthlyreturn_col = ('monthly_return', ticker_name)

    # Load cached monthly data
    with open(FYAHOO__OUTPUTFILENAME_MONTH, 'rb') as f:
        data_cache = pickle.load(f)

    df_full = data_cache[ticker_name]
    print(f"📈 Analyzing monthly data for {ticker_name}")
    print(f"   Period: {df_full.index[0].strftime('%Y-%m')} → {df_full.index[-1].strftime('%Y-%m')}")
    print("-" * 60)

    # Precompute all positive streaks once for denominator
    df_base = df_full.copy()
    df_base[positive_col] = df_base[close_col] > df_base[open_col]
    df_base[groupid_col] = (df_base[positive_col] != df_base[positive_col].shift(1)).cumsum()
    all_positive_groups = df_base.loc[df_base[positive_col]].groupby(groupid_col).size()
    total_positive_streaks = len(all_positive_groups)

    for NN in range(2, 10):
        df = df_base.copy()

        # Identify valid long streaks (≥ NN consecutive positive months)
        long_streak_group_ids = all_positive_groups[all_positive_groups >= NN].index

        if long_streak_group_ids.empty:
            prob = 0.0
        else:
            # Compute monthly returns
            df[monthlyreturn_col] = (df[close_col] - df[open_col]) / df[open_col]

            # Filter rows in long streaks
            long_streak_rows = df[df[groupid_col].isin(long_streak_group_ids) & df[positive_col]]

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

            # Compute total return (%)
            streak_summary['total_return_%'] = (
                (streak_summary['end_close'] - streak_summary['start_open']) /
                streak_summary['start_open'] * 100
            ).round(2)

            # Convert to percentages and round
            streak_summary['avg_monthly_return_%'] = (streak_summary['avg_monthly_return'] * 100).round(2)
            streak_summary['volatility_%'] = (
                streak_summary['return_volatility'].fillna(0) * 100
            ).round(2)

            num_long_streaks = len(streak_summary)
            prob = num_long_streaks / total_positive_streaks

            # Optional: print top streaks for NN=2 or 3 (comment out if not needed)
            # if NN <= 3:
            #     print(f"\n🔍 Top streaks with ≥{NN} positive months:")
            #     print(tabulate(
            #         streak_summary[
            #             ['start_date', 'end_date', 'length', 'total_return_%', 'avg_monthly_return_%']
            #         ].sort_values('total_return_%', ascending=False).head(5),
            #         headers='keys', tablefmt='simple', floatfmt=".2f"
            #     ))

        print(f"• Streaks ≥{NN:2d} months: {prob:>6.2%} ({int(prob * total_positive_streaks)} out of {total_positive_streaks})")

    print("-" * 60)
    print("✅ Analysis complete.")
