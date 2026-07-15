try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import yfinance as yf
import pandas as pd
import argparse
import pickle
from utils import get_filename_for_dataset, get_next_step
from fetchers.serialize_fyahoo import get_realtime_dataset


def compute_break_even_credit(win_rate, spread_width):
    # Core Math
    loss_rate = 1.0 - win_rate
    breakeven_credit = loss_rate * spread_width
    return breakeven_credit


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dataset-id", type=str, default="day", choices=["day", "week", "month"])
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    # Changed epsilon to be a percentage (fraction). e.g., 0.01 = 1%
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Margin as a percentage/fraction (e.g., 0.01 for 1%%) to avoid close values')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output')

    return parser


def calculate_sp500_probabilities(args):
    open_col = ('Open', args.ticker)
    close_col = ('Close', args.ticker)
    high_col = ('High', args.ticker)
    low_col = ('Low', args.ticker)

    # Extract epsilon for use in comparisons
    epsilon = args.epsilon
    _master_data_cache = get_realtime_dataset(dataset_id=args.dataset_id)
    df_ticker = _master_data_cache[args.ticker].sort_index().copy()

    close_t_1_col = ('Close_t-1', args.ticker)
    low_t_1_col = ('Low_t-1', args.ticker)
    high_t_1_col = ('High_t-1', args.ticker)

    # 2. Create lagged columns for t-1
    df_ticker[close_t_1_col] = df_ticker[close_col].shift(1)
    df_ticker[low_t_1_col] = df_ticker[low_col].shift(1)
    df_ticker[high_t_1_col] = df_ticker[high_col].shift(1)

    # Drop the first row which contains NaNs due to the shift
    df_ticker.dropna(inplace=True)

    total_bars = len(df_ticker)
    print(f"Successfully loaded {total_bars} trading bars.\n")

    # ==========================================
    # PROBABILITY 1
    # Condition: Open_t > Close_t-1 * (1 + epsilon) (Gap Up)
    # Target: Close_t > Low_t-1 * (1 + epsilon)
    # ==========================================
    cond1 = df_ticker[open_col] > (df_ticker[close_t_1_col] * (1 + epsilon))
    target1 = df_ticker[close_col] > (df_ticker[low_t_1_col] * (1 + epsilon))

    # Calculate probability (mean of boolean series) and sample size
    prob1 = target1[cond1].mean()
    count1 = cond1.sum()

    # ==========================================
    # PROBABILITY 2
    # Condition: Open_t < Close_t-1 * (1 - epsilon) (Gap Down)
    # Target: Close_t < High_t-1 * (1 - epsilon)
    # ==========================================
    cond2 = df_ticker[open_col] < (df_ticker[close_t_1_col] * (1 - epsilon))
    target2 = df_ticker[close_col] < (df_ticker[high_t_1_col] * (1 - epsilon))

    # Calculate probability and sample size
    prob2 = target2[cond2].mean()
    count2 = cond2.sum()

    # ==========================================
    # PROBABILITY 3
    # Condition: Open_t < Close_t-1 * (1 - epsilon) (Gap Down)
    # Target: Close_t < Low_t-1 * (1 - epsilon) (Breaks Previous Low)
    # ==========================================
    cond3 = df_ticker[open_col] < (df_ticker[close_t_1_col] * (1 - epsilon))
    target3 = df_ticker[close_col] < (df_ticker[low_t_1_col] * (1 - epsilon))

    # Calculate probability and sample size
    prob3 = target3[cond3].mean()
    count3 = cond3.sum()

    # ==========================================
    # PROBABILITY 4
    # Condition: Open_t > Close_t-1 * (1 + epsilon) (Gap Up)
    # Target: Close_t > High_t-1 * (1 + epsilon) (Breaks Previous High)
    # ==========================================
    cond4 = df_ticker[open_col] > (df_ticker[close_t_1_col] * (1 + epsilon))
    target4 = df_ticker[close_col] > (df_ticker[high_t_1_col] * (1 + epsilon))

    # Calculate probability and sample size
    prob4 = target4[cond4].mean()
    count4 = cond4.sum()

    # ==========================================
    # PROBABILITY 5
    # Condition: Open_t > Close_t-1 * (1 + epsilon) (Gap Up)
    # Target: Close_t > Close_t-1 * (1 + epsilon)
    # ==========================================
    cond5 = df_ticker[open_col] > (df_ticker[close_t_1_col] * (1 + epsilon))
    target5 = df_ticker[close_col] > (df_ticker[close_t_1_col] * (1 + epsilon))

    # Calculate probability and sample size
    prob5 = target5[cond5].mean()
    count5 = cond5.sum()

    # ==========================================
    # PROBABILITY 6
    # Condition: Open_t < Close_t-1 * (1 + epsilon)
    # Target: Close_t < Close_t-1 * (1 + epsilon)
    # ==========================================
    cond6 = df_ticker[open_col] < (df_ticker[close_t_1_col] * (1 + epsilon))
    target6 = df_ticker[close_col] < (df_ticker[close_t_1_col] * (1 + epsilon))

    # Calculate probability and sample size
    prob6 = target6[cond6].mean()
    count6 = cond6.sum()

    # ==========================================
    # PROBABILITY 7
    # Condition: Open_t > High_t-1
    # Target: Close_t > Close_t-1
    # ==========================================
    cond7 = df_ticker[open_col] > df_ticker[high_t_1_col]
    target7 = df_ticker[close_col] > df_ticker[close_t_1_col]

    # Calculate probability and sample size
    prob7 = target7[cond7].mean()
    count7 = cond7.sum()

    # ==========================================
    # PROBABILITY 8
    # Condition: Open_t < Low_t-1
    # Target: Close_t < Close_t-1
    # ==========================================
    cond8 = df_ticker[open_col] < df_ticker[low_t_1_col]
    target8 = df_ticker[close_col] < df_ticker[close_t_1_col]

    # Calculate probability and sample size
    prob8 = target8[cond8].mean()
    count8 = cond8.sum()

    # Get today's Open and yesterday's Close
    latest_open = df_ticker[open_col].iloc[-1]
    yesterday_close = df_ticker[close_col].iloc[-2]  # iloc[-2] gets the previous day

    # Format epsilon as a percentage string for readability
    eps_pct_str = f"{epsilon * 100:.2f}%"

    # Determine which "Given That" we are currently in
    if latest_open > (yesterday_close * (1 + epsilon)):
        current_condition = f"[1]/[4]/[5] GIVEN THAT Open_t > Close_t-1 + {eps_pct_str} (Gap Up)"
        active_probs = [1, 4, 5]
    elif latest_open < (yesterday_close * (1 - epsilon)):
        current_condition = f"[2]/[3]/[6] GIVEN THAT Open_t < Close_t-1 - {eps_pct_str} (Gap Down)"
        active_probs = [2, 3, 6]
    else:
        current_condition = f"[X] No Significant Gap (Open_t ~ Close_t-1)"
        active_probs = []

    # Add new conditions to active probs based on yesterday's High/Low
    if latest_open > df_ticker[high_col].iloc[-2]:
        active_probs.append(7)
        current_condition += " | [7] Open_t > High_t-1"
    if latest_open < df_ticker[low_col].iloc[-2]:
        active_probs.append(8)
        current_condition += " | [8] Open_t < Low_t-1"

    close_t__str = f"{df_ticker[close_col].iloc[-1]:.0f}$"
    open_t__str = f"{df_ticker[open_col].iloc[-1]:.0f}$"
    low_t_1__str = f"{df_ticker[low_col].iloc[-2]:.0f}$"
    high_t_1__str = f"{df_ticker[high_col].iloc[-2]:.0f}$"
    close_t_1__str = f"{df_ticker[close_col].iloc[-2]:.0f}$"

    # ==========================================
    # OUTPUT RESULTS
    # ==========================================

    # ANSI escape codes for terminal bold text
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print("-" * 50)
    print(f"RESULTS FOR S&P 500 ({str(args.dataset_id).upper()} Candles)")
    print(f"Epsilon (Margin): {eps_pct_str}")
    print("-" * 50)

    # Print the newly identified current state
    print(f"\nCURRENT STATE:")
    print(f"    -> Today we are in: {current_condition}")
    print(f"    -> Today's Open: {latest_open:.2f}  ({df_ticker.index[-1].strftime('%Y-%m-%d')})")
    print(f"    -> Yesterday's Close: {yesterday_close:.2f}  ({df_ticker.index[-2].strftime('%Y-%m-%d')})")

    b1, e1 = (BOLD, RESET) if 1 in active_probs else ("", "")
    print(f"{b1}\n[1] Probability that Close_t({close_t__str}) > Low_t-1({low_t_1__str}) + {eps_pct_str}  (Put Credit Spread)")
    print(f"    GIVEN THAT Open_t({open_t__str}) > Close_t-1({close_t_1__str}) + {eps_pct_str} (Gap Up)")
    print(f"    -> Sample Size (Occurrences): {count1}  ({(count1 / total_bars):.2%} :: total of {total_bars} bars)")
    print(f"    -> Probability: {prob1:.2%}  , Break Even Premium (with 500$ max loss): {compute_break_even_credit(prob1, 500):.2f}${e1}")

    b2, e2 = (BOLD, RESET) if 2 in active_probs else ("", "")
    print(f"{b2}\n[2] Probability that Close_t({close_t__str}) < High_t-1({high_t_1__str}) - {eps_pct_str}  (Call Credit Spread)")
    print(f"    GIVEN THAT Open_t({open_t__str}) < Close_t-1({close_t_1__str}) - {eps_pct_str} (Gap Down)")
    print(f"    -> Sample Size (Occurrences): {count2}  ({(count2 / total_bars):.2%} :: total of {total_bars} bars)")
    print(f"    -> Probability: {prob2:.2%}  , Break Even Premium (with 500$ max loss): {compute_break_even_credit(prob2, 500):.2f}${e2}")

    b3, e3 = (BOLD, RESET) if 3 in active_probs else ("", "")
    print(f"{b3}\n[3] Probability that Close_t({close_t__str}) < Low_t-1({low_t_1__str}) - {eps_pct_str}  (Breaks Previous Low :: Put Credit Spread)")
    print(f"    GIVEN THAT Open_t({open_t__str}) < Close_t-1({close_t_1__str}) - {eps_pct_str} (Gap Down)")
    print(f"    -> Sample Size (Occurrences): {count3}  ({(count3 / total_bars):.2%} :: total of {total_bars} bars)")
    print(f"    -> Probability: {prob3:.2%}  , Break Even Premium (with 500$ max loss): {compute_break_even_credit(prob3, 500):.2f}${e3}")

    b4, e4 = (BOLD, RESET) if 4 in active_probs else ("", "")
    print(f"{b4}\n[4] Probability that Close_t({close_t__str}) > High_t-1({high_t_1__str}) + {eps_pct_str}  (Breaks Previous High :: Call Credit Spread)")
    print(f"    GIVEN THAT Open_t({open_t__str}) > Close_t-1({close_t_1__str}) + {eps_pct_str} (Gap Up)")
    print(f"    -> Sample Size (Occurrences): {count4}  ({(count4 / total_bars):.2%} :: total of {total_bars} bars)")
    print(f"    -> Probability: {prob4:.2%}  , Break Even Premium (with 500$ max loss): {compute_break_even_credit(prob4, 500):.2f}${e4}")

    b5, e5 = (BOLD, RESET) if 5 in active_probs else ("", "")
    print(f"{b5}\n[5] Probability that Close_t({close_t__str}) > Close_t-1({close_t_1__str}) + {eps_pct_str}  (Closes Higher Than Yesterday's Close)")
    print(f"    GIVEN THAT Open_t({open_t__str}) > Close_t-1({close_t_1__str}) + {eps_pct_str} (Gap Up)")
    print(f"    -> Sample Size (Occurrences): {count5}  ({(count5 / total_bars):.2%} :: total of {total_bars} bars)")
    print(f"    -> Probability: {prob5:.2%}  , Break Even Premium (with 500$ max loss): {compute_break_even_credit(prob5, 500):.2f}${e5}")

    b6, e6 = (BOLD, RESET) if 6 in active_probs else ("", "")
    print(f"{b6}\n[6] Probability that Close_t({close_t__str}) < Close_t-1({close_t_1__str}) + {eps_pct_str}  (Closes Lower Than Yesterday's Close + Epsilon)")
    print(f"    GIVEN THAT Open_t({open_t__str}) < Close_t-1({close_t_1__str}) + {eps_pct_str}")
    print(f"    -> Sample Size (Occurrences): {count6}  ({(count6 / total_bars):.2%} :: total of {total_bars} bars)")
    print(f"    -> Probability: {prob6:.2%}  , Break Even Premium (with 500$ max loss): {compute_break_even_credit(prob6, 500):.2f}${e6}")

    b7, e7 = (BOLD, RESET) if 7 in active_probs else ("", "")
    print(f"{b7}\n[7] Probability that Close_t({close_t__str}) > Close_t-1({close_t_1__str})")
    print(f"    GIVEN THAT Open_t({open_t__str}) > High_t-1({high_t_1__str})")
    print(f"    -> Sample Size (Occurrences): {count7}  ({(count7 / total_bars):.2%} :: total of {total_bars} bars)")
    print(f"    -> Probability: {prob7:.2%}  , Break Even Premium (with 500$ max loss): {compute_break_even_credit(prob7, 500):.2f}${e7}")

    b8, e8 = (BOLD, RESET) if 8 in active_probs else ("", "")
    print(f"{b8}\n[8] Probability that Close_t({close_t__str}) < Close_t-1({close_t_1__str})")
    print(f"    GIVEN THAT Open_t({open_t__str}) < Low_t-1({low_t_1__str})")
    print(f"    -> Sample Size (Occurrences): {count8}  ({(count8 / total_bars):.2%} :: total of {total_bars} bars)")
    print(f"    -> Probability: {prob8:.2%}  , Break Even Premium (with 500$ max loss): {compute_break_even_credit(prob8, 500):.2f}${e8}")

    print("-" * 50)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    calculate_sp500_probabilities(args)