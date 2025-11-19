#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze S&P 500 (^GSPC) monthly data for:
1. Long positive streaks (≥ N consecutive months with Close > Open)
2. Monthly return statistics during streaks
3. Probability metrics
4. Seasonal green-month analysis (by calendar month)
5. Short-term return pattern: P positive → Q negative → next-day outcome

Author: Luc Cauchon (based on user context)
"""
try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from constants import FYAHOO__OUTPUTFILENAME_MONTH, FYAHOO__OUTPUTFILENAME_DAY, FYAHOO__OUTPUTFILENAME_WEEK
from datetime import datetime
import pandas as pd
import numpy as np
from tabulate import tabulate
import pickle
import argparse
from utils import transform_path


def main(P, Q, older_dataset, frequency, detailed, quiet):
    # ----------------------------
    # Configuration
    # ----------------------------
    TICKER = "^GSPC"
    # ----------------------------
    # Fetch Data
    # ----------------------------
    if frequency == 'monthly':
        one_dataset_filename = FYAHOO__OUTPUTFILENAME_MONTH if older_dataset == "" else transform_path(FYAHOO__OUTPUTFILENAME_MONTH, older_dataset)
    elif frequency == 'daily':
        one_dataset_filename = FYAHOO__OUTPUTFILENAME_DAY if older_dataset == "" else transform_path(FYAHOO__OUTPUTFILENAME_DAY, older_dataset)
    elif frequency == 'weekly':
        one_dataset_filename = FYAHOO__OUTPUTFILENAME_WEEK if older_dataset == "" else transform_path(FYAHOO__OUTPUTFILENAME_WEEK, older_dataset)
    else:
        assert False, f"{frequency=}"
    if not quiet:
        print(f"Using {one_dataset_filename}")
    with open(one_dataset_filename, 'rb') as f:
        data_cache = pickle.load(f)
    data = data_cache[TICKER]
    if not quiet:
        print(f"📅 Data range: {data.index[0].strftime('%Y-%m')} → {data.index[-1].strftime('%Y-%m')} "
              f"({len(data)} , {one_dataset_filename})\n")

    # Column references (multi-index)
    close_col = ('Close', TICKER)
    open_col = ('Open', TICKER)
    positive_col = ('positive', TICKER)
    groupid_col = ('group_id', TICKER)
    monthlyreturn_col = ('monthly_return', TICKER)

    # ----------------------------
    # Prepare DataFrame
    # ----------------------------
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")

    df[positive_col] = df[close_col] > df[open_col]
    df[monthlyreturn_col] = (df[close_col] - df[open_col]) / df[open_col]
    df[groupid_col] = (df[positive_col] != df[positive_col].shift(1)).cumsum()

    # ----------------------------
    # Pattern Analysis: P↑ → Q↓ → Next Month?
    # ----------------------------
    if not quiet:
        print("\n" + "=" * 60)
        print(f"🔍 Pattern Analysis: {P} positive → {Q} negative moves → next close direction (using Close-to-Close values)")
    results_to_return = {"price_based":{'total_patterns': 0, 'positive_outcomes': 0, 'prob': 0},
                         "intra_step":{'total_patterns': 0, 'positive_outcomes': 0, 'prob': 0}}
    close_values_month = data[close_col].dropna()
    min_required = P + Q + 1
    if len(close_values_month) < min_required:
        print(f"⚠️ Not enough monthly data for pattern (need at least {min_required} months).")
    else:
        returns = close_values_month.diff()
        up_months = returns > 0
        down_months = returns < 0

        pattern_results = []
        for i in range(len(close_values_month) - (P + Q + 1) + 1):
            up_seq = up_months.iloc[i: i + P]
            down_seq = down_months.iloc[i + P: i + P + Q]

            if up_seq.all() and down_seq.all():
                end_idx = i + P + Q - 1
                next_idx = end_idx + 1
                if next_idx >= len(close_values_month):
                    continue

                end_date = close_values_month.index[end_idx]
                next_date = close_values_month.index[next_idx]
                end_close = close_values_month.iloc[end_idx]
                next_close = close_values_month.iloc[next_idx]
                next_return = (next_close - end_close) / end_close
                direction = "↑" if next_return > 0 else "↓" if next_return < 0 else "="

                pattern_results.append({
                    "pattern_end": end_date.strftime('%Y-%m-%d'),
                    "next_date": next_date.strftime('%Y-%m-%d'),
                    "end_close": end_close,
                    "next_close": next_close,
                    "next_return_%": next_return * 100,
                    "direction": direction
                })

        if pattern_results:
            results_df = pd.DataFrame(pattern_results)
            total_patterns = len(results_df)
            positive_outcomes = (results_df["next_return_%"] > 0).sum()
            prob = positive_outcomes / total_patterns
            results_to_return['price_based'].update({'total_patterns': total_patterns, 'positive_outcomes': positive_outcomes, 'prob': prob})
            if not quiet:
                print(f"Total patterns found: {total_patterns}")
                print(f"Next positive:        {positive_outcomes}")
                print(f"Probability:          {prob:.2%}\n")
            if detailed and not quiet:
                print("📋 Detailed Pattern Outcomes:")
                print(tabulate(
                    results_df[["pattern_end", "next_date", "end_close", "next_close", "next_return_%", "direction"]],
                    headers=["Pattern End", "Next Date", "End Close", "Next Close", "Next Return (%)", "Dir"],
                    tablefmt="simple",
                    floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f", "")
                ))
        else:
            if not quiet:
                print("No occurrences of this pattern in the data.")

    # ----------------------------
    # Pattern Analysis: P↑ → Q↓ → Next Month? (Using Open/Close within month)
    # ----------------------------
    if not quiet:
        print("\n\n" + "=" * 60)
        print(f"🔍 Pattern Analysis: {P} positive → {Q} negative moves → next close direction (using Open/Close values)")

    open_monthly = data[open_col].dropna()
    close_monthly = data[close_col].dropna()
    common_index = open_monthly.index.intersection(close_monthly.index)
    open_monthly = open_monthly.loc[common_index]
    close_monthly = close_monthly.loc[common_index]

    if len(close_monthly) < P + Q + 1:
        print(f"⚠️ Not enough monthly data for pattern (need at least {P + Q + 1} months).")
    else:
        monthly_positive = close_monthly > open_monthly
        monthly_negative = close_monthly < open_monthly

        pattern_results = []
        n = len(monthly_positive)

        for i in range(n - (P + Q + 1) + 1):
            pos_seq = monthly_positive.iloc[i: i + P]
            neg_seq = monthly_negative.iloc[i + P: i + P + Q]

            if pos_seq.all() and neg_seq.all():
                end_idx = i + P + Q - 1
                next_idx = end_idx + 1
                if next_idx >= n:
                    continue

                end_date = close_monthly.index[end_idx]
                next_date = close_monthly.index[next_idx]
                end_close = close_monthly.iloc[end_idx]
                next_close = close_monthly.iloc[next_idx]
                next_return = (next_close - end_close) / end_close
                direction = "↑" if next_return > 0 else "↓" if next_return < 0 else "="

                pattern_results.append({
                    "pattern_end": end_date.strftime('%Y-%m-%d'),
                    "next_date": next_date.strftime('%Y-%m-%d'),
                    "end_close": end_close,
                    "next_close": next_close,
                    "next_return_%": next_return * 100,
                    "direction": direction
                })

        if pattern_results:
            results_df = pd.DataFrame(pattern_results)
            total_patterns = len(results_df)
            positive_outcomes = (results_df["next_return_%"] > 0).sum()
            prob = positive_outcomes / total_patterns
            results_to_return['intra_step'].update({'total_patterns': total_patterns, 'positive_outcomes': positive_outcomes, 'prob': prob})
            if not quiet:
                print(f"Total patterns found: {total_patterns}")
                print(f"Next positive:        {positive_outcomes}")
                print(f"Probability:          {prob:.2%}\n")
            if detailed and not quiet:
                print("📋 Detailed Pattern Outcomes:")
                print(tabulate(
                    results_df[["pattern_end", "next_date", "end_close", "next_close", "next_return_%", "direction"]],
                    headers=["Pattern End", "Next Date", "End Close", "Next Close", "Next Return (%)", "Dir"],
                    tablefmt="simple",
                    floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f", "")
                ))
        else:
            if not quiet:
                print("No occurrences of this pattern in the data.")
    return results_to_return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze S&P 500 streaks of P↑→Q↓ patterns.")
    parser.add_argument('--P', type=int, default=6, help='Number of consecutive positive months (default: 6)')
    parser.add_argument('--Q', type=int, default=1, help='Number of consecutive negative months after P positives (default: 1)')
    parser.add_argument("--older_dataset", type=str, default="")
    parser.add_argument("--frequency", type=str, default="monthly", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--detailed", type=bool, default=False)
    parser.add_argument("--quiet", type=bool, default=False)
    args = parser.parse_args()

    main(P=args.P, Q=args.Q, older_dataset=args.older_dataset, frequency=args.frequency, detailed=args.detailed, quiet=args.quiet)