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

from constants import FYAHOO__OUTPUTFILENAME_MONTH, FYAHOO__OUTPUTFILENAME_DAY, FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_YEAR
import pandas as pd
import pickle
import copy
from utils import get_filename_for_dataset, DATASET_AVAILABLE


def main(direction: str, method: str, older_dataset: str, bold: int, frequency: str, delta: float, ticker_name: str, verbose: bool, bring_my_own_df=None):
    if delta != 0.0 and method != "prev_close":
        raise ValueError("The --delta option is only supported with --method prev_close")
    if verbose:
        print(f"The proportion (or count) of streaks that are at least N periods long.")
    close_col = ('Close', ticker_name)
    open_col = ('Open', ticker_name)
    groupid_col = ('group_id', ticker_name)
    monthlyreturn_col = ('monthly_return', ticker_name)
    if bring_my_own_df is None:
        one_dataset_filename = get_filename_for_dataset(frequency, older_dataset)
        if verbose:
            print(f"Using {one_dataset_filename}")
        with open(one_dataset_filename, 'rb') as f:
            data_cache = pickle.load(f)
        df_full = data_cache[ticker_name]
        df_full = df_full.sort_index()
    else:
        df_full = bring_my_own_df[ticker_name]
        df_full = df_full.sort_index()
    direction_label = "Negative" if direction == "neg" else "Positive"
    emoji = "📉" if direction == "neg" else "📈"

    if method == "open_close":
        method_desc = f"using intra-step (Open/Close) to identify {direction_label.upper()} ones"
    elif method == "prev_close":
        method_desc = f"Using inter-step (Close-to-Close) to identify {direction_label.upper()} ones"
    else:
        raise ValueError(f"Unknown method: {method}")
    if verbose:
        print(f"{emoji} Analyzing {frequency} data for {ticker_name}   ({method_desc})")
        print(f"   Period: {df_full.index[0].strftime('%Y-%m')} → {df_full.index[-1].strftime('%Y-%m')}")
        print("-" * 60)

    # Define condition column
    condition_col = ('positive', ticker_name) if direction == "pos" else ('negative', ticker_name)
    df_base = copy.deepcopy(df_full)

    if method == "open_close":
        if direction == "pos":
            df_base[condition_col] = df_base[close_col] > df_base[open_col]
        else:
            df_base[condition_col] = df_base[close_col] < df_base[open_col]
    elif method == "prev_close":
        prev_close = df_base[close_col].shift(1)
        if direction == "pos":
            df_base[condition_col] = df_base[close_col] > prev_close
        else:
            df_base[condition_col] = df_base[close_col] < prev_close

    # Remove the first row if using prev_close
    if method == "prev_close":
        df_base = df_base.dropna(subset=[close_col])

    # Assign group IDs for consecutive streaks
    df_base[groupid_col] = (df_base[condition_col] != df_base[condition_col].shift(1)).cumsum()
    all_groups = df_base.loc[df_base[condition_col]].groupby(groupid_col).size()
    total_streaks = len(all_groups)
    returned_results = {}
    for NN in range(2, 13):
        long_streak_group_ids = all_groups[all_groups >= NN].index

        if long_streak_group_ids.empty:
            prob = 0.0
        else:
            # Compute periodic returns
            if method == "open_close":
                df_base[monthlyreturn_col] = (df_base[close_col] - df_base[open_col]) / df_base[open_col]
            else:  # prev_close
                prev_close = df_base[close_col].shift(1)
                df_base[monthlyreturn_col] = (df_base[close_col] - prev_close) / prev_close

            long_streak_rows = df_base[df_base[groupid_col].isin(long_streak_group_ids) & df_base[condition_col]]

            grouped = long_streak_rows.groupby(groupid_col)

            valid_streak_ids = []
            for gid, group in grouped:
                if len(group) < NN:
                    continue
                group_sorted = group.sort_index()
                if len(group_sorted) < 2:
                    continue

                last_idx = group_sorted.index[-1]
                prev_idx = group_sorted.index[-2]

                close_last = df_base.loc[last_idx, close_col]
                close_prev = df_base.loc[prev_idx, close_col]

                if direction == "pos":
                    if close_last > close_prev * (1 + delta):
                        valid_streak_ids.append(gid)
                else:  # neg
                    if close_last < close_prev * (1 - delta):
                        valid_streak_ids.append(gid)

            if not valid_streak_ids:
                prob = 0.0
            else:
                valid_rows = df_base[df_base[groupid_col].isin(valid_streak_ids) & df_base[condition_col]]
                valid_grouped = valid_rows.groupby(groupid_col)

                streak_summary = valid_grouped.agg(
                    start_date=(groupid_col, lambda x: x.index.min()),
                    end_date=(groupid_col, lambda x: x.index.max()),
                    length=(groupid_col, 'size'),
                    start_open=(open_col if method == "open_close" else close_col, 'first'),
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
        if NN == bold:
            if verbose:
                print(f"\033[1m• Streaks ≥{NN:2d} {frequency}s: {prob:>6.2%} ({count} out of {total_streaks}) *\033[0m")
        else:
            if verbose:
                print(f"• Streaks ≥{NN:2d} {frequency}s: {prob:>6.2%} ({count} out of {total_streaks})")
        returned_results.update({NN: {'NN': NN, 'frequency': frequency, 'prob': prob, 'count': count, 'total_streaks': total_streaks, 'delta': delta}})
    if verbose:
        print("-" * 60)
        print(f"✅ {direction_label} streak analysis complete.")
    return returned_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze positive or negative streaks for ^GSPC.")
    parser.add_argument(
        "-d", "--direction",
        choices=["pos", "neg"],
        default="pos",
        help="Direction of streaks to analyze: 'pos' for positive, 'neg' for negative (default: pos)"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["open_close", "prev_close"],
        default="prev_close",
        help="Method to determine sign: 'open_close' uses Close vs Open of same period; 'prev_close' uses Close vs previous period Close (default: prev_close)"
    )
    parser.add_argument("-o","--older_dataset", type=str, default="")
    parser.add_argument(
        "-b", "--bold",
        default=2,
        type=int,
        help="Display in bold the element of interest in the sequence"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="day",
        choices=DATASET_AVAILABLE
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="Minimum percentage change (as a decimal) required for the FINAL step of a streak (e.g., 0.02 = 2%). Only used with --method prev_close."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default='^GSPC',
        help=""
    )
    args = parser.parse_args()
    main(args.direction, args.method, args.older_dataset, args.bold, args.frequency, args.delta, args.ticker, True)