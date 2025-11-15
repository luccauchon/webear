#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze S&P 500 (^GSPC) monthly data for:
1. Long positive streaks
2. Monthly return stats
3. Probability metrics
4. Seasonal green-month analysis
5. Pattern: P1↑ → Q↓ → P2↑ → next month outcome
   - Supports two definitions of "↑/↓":
        a) Intra-month: Close > Open
        b) Inter-month: Close[t] > Close[t-1] (price-based return)

Author: Luc Cauchon
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
from constants import FYAHOO__OUTPUTFILENAME_MONTH, FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_DAY
import pandas as pd
import numpy as np
from tabulate import tabulate
import pickle
import argparse
from utils import transform_path


def _find_patterns(monthly_up, monthly_down, close_series, P1, Q, P2, mode_name, detailed):
    """
    Generic pattern finder.
    monthly_up: boolean Series (True = positive month)
    monthly_down: boolean Series (True = negative month)
    close_series: price series aligned with the boolean masks
    """
    n = len(monthly_up)
    min_len = P1 + Q + P2 + 1
    if n < min_len:
        print(f"⚠️ Not enough data for {mode_name} pattern (need ≥{min_len} points).")
        return

    pattern_results = []
    for i in range(n - (P1 + Q + P2 + 1) + 1):
        pos1 = monthly_up.iloc[i: i + P1]
        neg = monthly_down.iloc[i + P1: i + P1 + Q]
        pos2 = monthly_up.iloc[i + P1 + Q: i + P1 + Q + P2]

        if pos1.all() and neg.all() and pos2.all():
            next_idx = i + P1 + Q + P2
            if next_idx >= n:
                continue

            end_idx = next_idx - 1
            end_close = close_series.iloc[end_idx]
            next_close = close_series.iloc[next_idx]
            next_return = (next_close - end_close) / end_close
            direction = "↑" if next_return > 0 else "↓" if next_return < 0 else "="

            pattern_results.append({
                "pattern_end": close_series.index[end_idx].strftime('%Y-%m-%d'),
                "next_date": close_series.index[next_idx].strftime('%Y-%m-%d'),
                "end_close": end_close,
                "next_close": next_close,
                "next_return_%": next_return * 100,
                "direction": direction
            })

    if pattern_results:
        results_df = pd.DataFrame(pattern_results)
        total = len(results_df)
        pos_next = (results_df["next_return_%"] > 0).sum()
        prob = pos_next / total

        print(f"\n✅ {mode_name} Pattern Results:")
        print(f"Total patterns found: {total}")
        print(f"Next month positive:  {pos_next}")
        print(f"Probability:          {prob:.2%}\n")
        if detailed:
            print("📋 Outcomes:")
            print(tabulate(
                results_df[["pattern_end", "next_date", "end_close", "next_close", "next_return_%", "direction"]],
                headers=["Pattern End", "Next Date", "End Close", "Next Close", "Next Return (%)", "Dir"],
                tablefmt="simple",
                floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f", "")
            ))
    else:
        print(f"\n❌ No {mode_name} patterns found.")


def main(P1, Q, P2, older_dataset, frequency, detailed):
    TICKER = "^GSPC"
    if frequency == 'monthly':
        one_dataset_filename = FYAHOO__OUTPUTFILENAME_MONTH if older_dataset == "" else transform_path(FYAHOO__OUTPUTFILENAME_MONTH, older_dataset)
    elif frequency == 'weekly':
        one_dataset_filename = FYAHOO__OUTPUTFILENAME_WEEK if older_dataset == "" else transform_path(FYAHOO__OUTPUTFILENAME_WEEK, older_dataset)
    elif frequency == 'daily':
        one_dataset_filename = FYAHOO__OUTPUTFILENAME_DAY if older_dataset == "" else transform_path(FYAHOO__OUTPUTFILENAME_DAY, older_dataset)
    else:
        assert False

    with open(one_dataset_filename, 'rb') as f:
        data_cache = pickle.load(f)
    data = data_cache[TICKER]

    print(f"📅 Data range: {data.index[0].strftime('%Y-%m')} → {data.index[-1].strftime('%Y-%m')} "
          f"({len(data)} {frequency.upper()})\n")

    close_col = ('Close', TICKER)
    open_col = ('Open', TICKER)

    # Extract and align Close & Open
    close_monthly = data[close_col].dropna()
    open_monthly = data[open_col].dropna()
    common_idx = close_monthly.index.intersection(open_monthly.index)
    close_monthly = close_monthly.loc[common_idx]
    open_monthly = open_monthly.loc[common_idx]

    # =============================================
    # Mode 1: Intra-month (Close vs Open)
    # =============================================
    print("=" * 70)
    print(f"🔍 Pattern: {P1}↑ → {Q}↓ → {P2}↑ → next {frequency.upper()}")
    print(f"   Mode 1: Intra-{frequency.upper()} direction (Close > Open = ↑)")

    intra_up = close_monthly > open_monthly
    intra_down = close_monthly < open_monthly
    _find_patterns(intra_up, intra_down, close_monthly, P1, Q, P2, "Intra-month", detailed=detailed)

    # =============================================
    # Mode 2: Inter-month (Close-to-Close returns)
    # =============================================
    print("\n" + "=" * 70)
    print("   Mode 2: Price-based returns (Close[t] > Close[t-1] = ↑)")

    # Compute month-over-month returns
    close_series = close_monthly.sort_index()  # ensure chronological order
    returns = close_series.pct_change()
    inter_up = returns > 0
    inter_down = returns < 0

    # Note: first value is NaN → drop it
    inter_up = inter_up.iloc[1:]
    inter_down = inter_down.iloc[1:]
    close_aligned = close_series.iloc[1:]

    _find_patterns(inter_up, inter_down, close_aligned, P1, Q, P2, "Price-based (Close-to-Close)", detailed=detailed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze S&P 500: P1↑ → Q↓ → P2↑ → next.")
    parser.add_argument('--P1', type=int, default=2, help='Initial positive (default: 2)')
    parser.add_argument('--Q', type=int, default=1, help='Negative months (default: 1)')
    parser.add_argument('--P2', type=int, default=2, help='Trailing positive (default: 2)')
    parser.add_argument("--older_dataset", type=str, default="")
    parser.add_argument("--frequency", type=str, default="monthly")
    parser.add_argument("--detailed", type=bool, default=False)
    args = parser.parse_args()

    main(P1=args.P1, Q=args.Q, P2=args.P2, older_dataset=args.older_dataset, frequency=args.frequency, detailed=args.detailed)