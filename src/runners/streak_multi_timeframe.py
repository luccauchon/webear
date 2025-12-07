try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import matplotlib.pyplot as plt
from runners.streak_probability_lot import main as streak_probability_lot
from argparse import Namespace
import argparse
from utils import get_filename_for_dataset
import pickle


def main(args):
    direction = args.direction
    ticker_name = args.ticker
    nn_day = args.nn_day
    nn_week = args.nn_week
    nn_month = args.nn_month
    ticker_streak_data = {'month': {'probabilities': [], 'price_levels': []},
                          'day': {'probabilities': [], 'price_levels': []},
                          'week': {'probabilities': [], 'price_levels': []}}

    one_dataset_filename = get_filename_for_dataset('day', older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        data_cache = pickle.load(f)
    # Extract close_value early so it's available in the delta loop
    close_col = ('Close', ticker_name)
    last_close_value = data_cache[ticker_name][close_col].iloc[-1]
    last_close_date  = data_cache[ticker_name][close_col].index[-1]

    # --- Step 1: Organize the data into a structured dictionary ---
    # Day
    configuration = Namespace(
        direction=direction,
        method="prev_close",
        frequency='day',
        ticker=ticker_name,
        deltas=[0, 0.1, 0.25, 0.50, 0.75, 0.9, 1.0, 1.25, 1.33, 1.5, 1.75, 2.0, 2.25, 2.5],
        display_only_nn=None,
    )
    day_results = streak_probability_lot(configuration)
    for delta, prob, count, total_streaks, close_value in day_results[nn_day]:
        ticker_streak_data['day']['probabilities'].append(prob*100.)
        ticker_streak_data['day']['price_levels'].append(close_value)

    # Week
    configuration = Namespace(
        direction=direction,
        method="prev_close",
        frequency='week',
        ticker=ticker_name,
        deltas=[0, 1., 2., 3., 4.],
        display_only_nn=None,
    )
    week_results = streak_probability_lot(configuration)
    for delta, prob, count, total_streaks, close_value in week_results[nn_week]:
        ticker_streak_data['week']['probabilities'].append(prob*100.)
        ticker_streak_data['week']['price_levels'].append(close_value)

    # Month
    configuration = Namespace(
        direction=direction,
        method="prev_close",
        frequency='month',
        ticker=ticker_name,
        deltas=[0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00],
        display_only_nn=None,
    )
    month_results = streak_probability_lot(configuration)
    for delta, prob, count, total_streaks, close_value in month_results[nn_month]:
        ticker_streak_data['month']['probabilities'].append(prob*100.)
        ticker_streak_data['month']['price_levels'].append(close_value)

    # --- Step 2: Find the point closest to 5% for each frequency ---
    def find_closest_to_q(freq_data, q=5.0):
        probs = freq_data["probabilities"]
        prices = freq_data["price_levels"]
        # Find index where |prob - 5| is minimized
        diffs = [abs(p - q) for p in probs]
        idx = diffs.index(min(diffs))
        return prices[idx], probs[idx]

    month_q = find_closest_to_q(ticker_streak_data["month"], q=5.0)
    week_q  = find_closest_to_q(ticker_streak_data["week"], q=5.0)
    day_q   = find_closest_to_q(ticker_streak_data["day"], q=5.0)

    # --- Step 3: Plotting ---
    plt.figure(figsize=(12, 7))

    # Plot each series
    plt.plot(
        ticker_streak_data["month"]["price_levels"],
        ticker_streak_data["month"]["probabilities"],
        marker='o', linestyle='-', color='red', label=f'Month (≥{nn_month} streaks)'
    )
    plt.plot(
        ticker_streak_data["week"]["price_levels"],
        ticker_streak_data["week"]["probabilities"],
        marker='s', linestyle='-', color='green', label=f'Week (≥{nn_week} streaks)'
    )
    plt.plot(
        ticker_streak_data["day"]["price_levels"],
        ticker_streak_data["day"]["probabilities"],
        marker='^', linestyle='-', color='blue', label=f'Day (≥{nn_day} streaks)'
    )

    # Highlight ~q% points
    plt.scatter(*month_q, color='red', s=150, zorder=5, edgecolor='black')
    plt.scatter(*week_q, color='green', s=150, zorder=5, edgecolor='black')
    plt.scatter(*day_q, color='blue', s=150, zorder=5, edgecolor='black')

    # Annotate q% points
    plt.annotate(f'{month_q[1]:.2f}% @ {month_q[0]:.1f}', xy=month_q, xytext=(10, 10),
                 textcoords='offset points', color='red', fontweight='bold')
    plt.annotate(f'{week_q[1]:.2f}% @ {week_q[0]:.1f}', xy=week_q, xytext=(10, 10),
                 textcoords='offset points', color='green', fontweight='bold')
    plt.annotate(f'{day_q[1]:.2f}% @ {day_q[0]:.1f}', xy=day_q, xytext=(10, 10),
                 textcoords='offset points', color='blue', fontweight='bold')

    # Labels and title
    plt.xlabel('SPX Price Level')
    plt.ylabel('Probability of Streak ≥ N (%)')
    plt.title('SPX Positive Streak Probabilities by Time Frequency\n(Superimposed: Month, Week, Day)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Optional: Add horizontal line at q%
    plt.axhline(y=5.0, color='gray', linestyle='--', linewidth=1, label=f'{5}% Reference')
    plt.legend()

    # --- Add last close reference line and date label ---
    plt.axvline(x=last_close_value, color='black', linestyle='-', linewidth=2,
                label=f'Last Close: {last_close_date.strftime("%Y-%m-%d")}')

    # Annotate the date near the top of the plot
    ylims = plt.ylim()
    plt.text(last_close_value, ylims[1] * 0.95,
             last_close_date.strftime("%Y-%m-%d"),
             rotation=90, verticalalignment='top', horizontalalignment='right',
             color='black', fontweight='bold', fontsize=9)
    plt.text(last_close_value, ylims[1] * 0.75,
             f'{last_close_value:.1f}',
             rotation=90, verticalalignment='top', horizontalalignment='right',
             color='black', fontweight='bold', fontsize=9)

    # Show plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="📊 Compute streak probabilities multi timeframe for financial time series."
    )
    parser.add_argument(
        "--direction", choices=["pos", "neg"], default="pos",
        help="Direction of streak: 'pos' (positive) or 'neg' (negative)"
    )
    parser.add_argument(
        "--ticker", type=str, default="^GSPC",
        help="Ticker symbol (e.g., '^GSPC' for S&P 500)"
    )
    parser.add_argument(
        "--nn_day", type=int, default=2,
        help=""
    )
    parser.add_argument(
        "--nn_week", type=int, default=2,
        help=""
    )
    parser.add_argument(
        "--nn_month", type=int, default=2,
        help=""
    )

    args = parser.parse_args()

    # Nicely print the parsed arguments
    print("🔧 Configuration:")
    print("-" * 50)
    for arg, value in vars(args).items():
        if arg == "deltas":
            val_str = "[" + ", ".join(f"{v:.2f}" for v in value) + "]"
            print(f"    {arg.replace('_', '-'):.<30} {val_str}")
        else:
            print(f"    {arg.replace('_', '-'):.<30} {value}")
    print("-" * 50)
    main(args)