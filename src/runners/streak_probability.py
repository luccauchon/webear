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

from constants import FYAHOO__OUTPUTFILENAME_MONTH
import pandas as pd
import pickle

def main(direction: str):
    ticker_name = '^GSPC'
    close_col = ('Close', ticker_name)
    open_col = ('Open', ticker_name)
    groupid_col = ('group_id', ticker_name)
    monthlyreturn_col = ('monthly_return', ticker_name)

    # Load cached monthly data
    with open(FYAHOO__OUTPUTFILENAME_MONTH, 'rb') as f:
        data_cache = pickle.load(f)

    df_full = data_cache[ticker_name]
    direction_label = "Negative" if direction == "neg" else "Positive"
    emoji = "📉" if direction == "neg" else "📈"
    print(f"{emoji} Analyzing monthly data for {ticker_name}   (USING OPEN/CLOSE VALUES OF A MONTH TO IDENTIFY {direction_label.upper()} ONES)")
    print(f"   Period: {df_full.index[0].strftime('%Y-%m')} → {df_full.index[-1].strftime('%Y-%m')}")
    print("-" * 60)

    # Define condition column and name
    condition_col = ('positive', ticker_name) if direction == "pos" else ('negative', ticker_name)
    df_base = df_full.copy()
    if direction == "pos":
        df_base[condition_col] = df_base[close_col] > df_base[open_col]
    else:
        df_base[condition_col] = df_base[close_col] < df_base[open_col]

    df_base[groupid_col] = (df_base[condition_col] != df_base[condition_col].shift(1)).cumsum()
    all_groups = df_base.loc[df_base[condition_col]].groupby(groupid_col).size()
    total_streaks = len(all_groups)

    for NN in range(2, 10):
        long_streak_group_ids = all_groups[all_groups >= NN].index

        if long_streak_group_ids.empty:
            prob = 0.0
        else:
            df_base[monthlyreturn_col] = (df_base[close_col] - df_base[open_col]) / df_base[open_col]
            long_streak_rows = df_base[df_base[groupid_col].isin(long_streak_group_ids) & df_base[condition_col]]

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

            streak_summary['total_return_%'] = (
                (streak_summary['end_close'] - streak_summary['start_open']) /
                streak_summary['start_open'] * 100
            ).round(2)

            streak_summary['avg_monthly_return_%'] = (streak_summary['avg_monthly_return'] * 100).round(2)
            streak_summary['volatility_%'] = (
                streak_summary['return_volatility'].fillna(0) * 100
            ).round(2)

            num_long_streaks = len(streak_summary)
            prob = num_long_streaks / total_streaks

        count = int(prob * total_streaks)
        print(f"• Streaks ≥{NN:2d} months: {prob:>6.2%} ({count} out of {total_streaks})")

    print("-" * 60)
    print(f"✅ {direction_label} streak analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze positive or negative monthly streaks for ^GSPC.")
    parser.add_argument(
        "-d", "--direction",
        choices=["pos", "neg"],
        default="pos",
        help="Direction of streaks to analyze: 'pos' for positive, 'neg' for negative (default: neg)"
    )
    args = parser.parse_args()
    main(args.direction)