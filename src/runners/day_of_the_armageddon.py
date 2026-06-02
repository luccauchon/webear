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
import pickle
from constants import FYAHOO__OUTPUTFILENAME_DAY
import copy

def main():
    # Load data
    with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
        master_data_cache = pickle.load(f)
    TICKER = "^GSPC"
    spx_df = copy.deepcopy(master_data_cache[TICKER])
    close_col = ('Close', TICKER)

    # Ensure DataFrame is sorted by date
    spx_df = spx_df.sort_index()

    # Filter out weekends (Saturday=5, Sunday=6)
    spx_df = spx_df[spx_df.index.dayofweek < 5]  # Keep Mon=0 to Fri=4 only

    # Calculate daily percent change (based on Close)
    spx_df['Daily_Pct_Change'] = spx_df[close_col].pct_change() * 100  # in percent

    # Identify drops (negative returns)
    spx_df['Is_Drop'] = spx_df['Daily_Pct_Change'] < 0
    spx_df['Abs_Drop'] = -spx_df['Daily_Pct_Change'].where(spx_df['Is_Drop'], 0)

    # Add weekday info (0=Monday, ..., 4=Friday)
    spx_df['DayOfWeek'] = spx_df.index.dayofweek
    spx_df['DayName'] = spx_df.index.day_name()

    # Define drop thresholds
    thresholds = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.]

    for thresh in thresholds:
        drop_days = spx_df[spx_df['Abs_Drop'] >= thresh]

        if drop_days.empty:
            print(f"\n=== Drops >= {thresh}% ===")
            print("No such drops found.")
            continue

        # Count and average drop by weekday (Mon–Fri only)
        count_by_day = drop_days['DayOfWeek'].value_counts().reindex(range(5), fill_value=0)
        avg_drop_by_day = drop_days.groupby('DayOfWeek')['Abs_Drop'].mean().reindex(range(5), fill_value=0)

        # Map to day names (Mon–Fri)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        count_by_day.index = day_names
        avg_drop_by_day.index = day_names

        print(f"\n=== Drops >= {thresh}% ===")
        print(f"Total events: {len(drop_days)}")
        print("Counts by weekday:")
        print(count_by_day)
        print("Average drop size by weekday:")
        print(avg_drop_by_day)
        print(f"Day with most frequent drops: {count_by_day.idxmax()} ({count_by_day.max()} times)")
        print(f"Day with largest average drop: {avg_drop_by_day.idxmax()} ({avg_drop_by_day.max():.2f}%)")

    # Worst single-day drop (among weekdays only)
    if not spx_df.empty:
        worst_drop_idx = spx_df['Abs_Drop'].idxmax()
        worst_drop = spx_df.loc[worst_drop_idx]

        print(f"\n=== Worst Single-Day Drop (Mon–Fri) ===")
        print(f"Date: {worst_drop_idx.date()}")
        print(f"Drop: {worst_drop['Abs_Drop'].item():.2f}%")
        print(f"Day of week: {worst_drop['DayName']}")

if __name__ == "__main__":
    main()