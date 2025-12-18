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
import mplcursors
from runners.streak_probability_lot import main as streak_probability_lot
from argparse import Namespace
import argparse
from utils import get_filename_for_dataset, str2bool
import pickle


def main(args):
    direction_day   = args.direction_day
    direction_week  = args.direction_week
    direction_month = args.direction_month
    ticker_name = args.ticker
    nn_day   = args.nn_day
    nn_week  = args.nn_week
    nn_month = args.nn_month
    skip_last_day   = args.skip_last_day
    skip_last_week  = args.skip_last_week
    skip_last_month = args.skip_last_month

    # Store direction mapping for tooltip use
    directions = {
        'day': direction_day,
        'week': direction_week,
        'month': direction_month
    }

    # Store full point metadata per frequency
    ticker_streak_data = {
        'day': [],
        'week': [],
        'month': []
    }

    # Load data to get last close
    one_dataset_filename = get_filename_for_dataset('day', older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        data_cache = pickle.load(f)
    # Extract close_value early so it's available in the delta loop
    close_col = ('Close', ticker_name)
    local_last_close_value = data_cache[ticker_name][close_col].iloc[-1]
    local_last_close_date  = data_cache[ticker_name][close_col].index[-1]

    # --- Step 1: Organize the data into a structured dictionary ---
    # Day
    configuration = Namespace(
        direction=direction_day,
        method="prev_close",
        frequency='day',
        ticker=ticker_name,
        deltas=[0, 0.1, 0.25, 0.50, 0.75, 0.9, 1.0, 1.25, 1.33, 1.5, 1.75, 2.0, 2.25, 2.5],
        display_only_nn=None,
        remove_last_element=skip_last_day,
        verbose=False,
    )
    day_results = streak_probability_lot(configuration)
    for delta, prob, count, total_streaks, close_value, last_close_value, last_date in day_results[nn_day]:
        ticker_streak_data['day'].append({
            'delta': delta,
            'prob': prob * 100.0,
            'count': count,
            'total': total_streaks,
            'price': close_value,
            'last_price': last_close_value,
            'last_date': last_date,
        })

    # --- Week ---
    configuration = Namespace(
        direction=direction_week,
        method="prev_close",
        frequency='week',
        ticker=ticker_name,
        deltas=[0, 1., 2., 3., 4.],
        display_only_nn=None,
        remove_last_element=skip_last_week,
        verbose=False,
    )
    week_results = streak_probability_lot(configuration)
    for delta, prob, count, total_streaks, close_value, last_close_value, last_date in week_results[nn_week]:
        ticker_streak_data['week'].append({
            'delta': delta,
            'prob': prob * 100.0,
            'count': count,
            'total': total_streaks,
            'price': close_value,
            'last_price': last_close_value,
            'last_date': last_date,
        })

    # --- Month ---
    configuration = Namespace(
        direction=direction_month,
        method="prev_close",
        frequency='month',
        ticker=ticker_name,
        deltas=[0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00],
        display_only_nn=None,
        remove_last_element=skip_last_month,
        verbose=False,
    )
    month_results = streak_probability_lot(configuration)
    for delta, prob, count, total_streaks, close_value, last_close_value, last_date in month_results[nn_month]:
        ticker_streak_data['month'].append({
            'delta': delta,
            'prob': prob * 100.0,
            'count': count,
            'total': total_streaks,
            'price': close_value,
            'last_price': last_close_value,
            'last_date': last_date,
        })

    the_col = 'price'

    # --- Find closest to 5% for annotation (optional) ---
    def find_closest_to_q(freq_data, q=5.0, col=the_col):
        best = min(freq_data, key=lambda p: abs(p['prob'] - q))
        return best[col], best['prob']

    month_q = find_closest_to_q(ticker_streak_data["month"], q=5.0, col=the_col)
    week_q  = find_closest_to_q(ticker_streak_data["week"], q=5.0, col=the_col)
    day_q   = find_closest_to_q(ticker_streak_data["day"], q=5.0, col=the_col)

    # --- Step 3: Plotting ---
    plt.figure(figsize=(12, 7))

    # Plot lines and keep references
    line_month, = plt.plot(
        [p[the_col] for p in ticker_streak_data["month"]],
        [p['prob'] for p in ticker_streak_data["month"]],
        marker='o', linestyle='-', color='red', label=f'Month (≥{nn_month+1} streaks)'
    )
    line_week, = plt.plot(
        [p[the_col] for p in ticker_streak_data["week"]],
        [p['prob'] for p in ticker_streak_data["week"]],
        marker='s', linestyle='-', color='green', label=f'Week (≥{nn_week+1} streaks)'
    )
    line_day, = plt.plot(
        [p[the_col] for p in ticker_streak_data["day"]],
        [p['prob'] for p in ticker_streak_data["day"]],
        marker='^', linestyle='-', color='blue', label=f'Day (≥{nn_day+1} streaks)'
    )
    # --- Add semi-transparent lines for 'last_price' ---
    if not skip_last_month:
        plt.plot(
            [p['last_price'] for p in ticker_streak_data["month"]],
            [p['prob'] for p in ticker_streak_data["month"]],
            marker='o', linestyle='-', color='red', alpha=0.5
        )
    if not skip_last_week:
        plt.plot(
            [p['last_price'] for p in ticker_streak_data["week"]],
            [p['prob'] for p in ticker_streak_data["week"]],
            marker='s', linestyle='-', color='green', alpha=0.5
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
    ending_day   = ticker_streak_data["day"][0]['last_date']
    ending_week  = ticker_streak_data["week"][0]['last_date']
    ending_month = ticker_streak_data["month"][0]['last_date']
    plt.xlabel(f'{ticker_name} Price Level')
    plt.ylabel('Probability of Streak ≥ N (%)')
    plt.title(f'{ticker_name} Streak Probabilities by Time Frequency\n'
              f'(Superimposed: Month {"POS" if direction_month=="pos" else "NEG"} ({ending_month.strftime("%Y-%m-%d")}), '
              f'Week {"POS" if direction_week=="pos" else "NEG"} ({ending_week.strftime("%Y-%m-%d")}), '
              f'Day {"POS" if direction_day=="pos" else "NEG"} ({ending_day.strftime("%Y-%m-%d")}))')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(y=5.0, color='gray', linestyle='--', linewidth=1)

    # Last close line
    plt.axvline(x=local_last_close_value, color='black', linestyle='-', linewidth=2)
    ylims = plt.ylim()
    plt.text(local_last_close_value, ylims[1] * 0.95,
             local_last_close_date.strftime("%Y-%m-%d"),
             rotation=90, verticalalignment='top', horizontalalignment='right',
             color='black', fontweight='bold', fontsize=9)
    plt.text(local_last_close_value, ylims[1] * 0.75,
             f'{local_last_close_value:.1f}',
             rotation=90, verticalalignment='top', horizontalalignment='right',
             color='black', fontweight='bold', fontsize=9)

    plt.legend()

    # --- Tooltip with mplcursors ---
    def format_tooltip(sel):
        # Map artist to data + direction
        mapping = {
            line_month: ('Month', nn_month, ticker_streak_data['month'], directions['month']),
            line_week: ('Week', nn_week, ticker_streak_data['week'], directions['week']),
            line_day: ('Day', nn_day, ticker_streak_data['day'], directions['day']),
        }
        if sel.artist not in mapping:
            return

        freq_name, n_streak, data_list, direction = mapping[sel.artist]

        # Determine "above" or "below" based on direction
        close_text = "below" if direction == "neg" else "above"

        lines = [f"Streaks ≥ {n_streak+1} ({freq_name.lower()}-frequency):"]
        for p in data_list:
            prob_str = f"{p['prob']:.2f}%"
            count_str = f"{int(p['count']):>3}"
            total_str = f"{int(p['total']):>4}"
            delta_str = f"{p['delta']:>4.1f}%"
            price_str = f"{p['price']:>9,.1f}"
            line = f"     {prob_str} ({count_str} / {total_str}) → Close {close_text} {price_str} (Δ = {delta_str})"
            lines.append(line)

        tooltip_text = "\n".join(lines)
        sel.annotation.set_text(tooltip_text)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.95, boxstyle="round,pad=0.6")
        sel.annotation.set_fontsize(9)

    # Enable hover tooltips
    cursor = mplcursors.cursor([line_month, line_week, line_day], hover=True)
    cursor.connect("add", format_tooltip)

    plt.tight_layout()
    # Show plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="📊 Compute streak probabilities multi timeframe for financial time series."
    )
    parser.add_argument(
        "--direction_day", choices=["pos", "neg"], default="pos",
        help="Direction of streak: 'pos' (positive) or 'neg' (negative)"
    )
    parser.add_argument(
        "--direction_week", choices=["pos", "neg"], default="pos",
        help="Direction of streak: 'pos' (positive) or 'neg' (negative)"
    )
    parser.add_argument(
        "--direction_month", choices=["pos", "neg"], default="pos",
        help="Direction of streak: 'pos' (positive) or 'neg' (negative)"
    )
    parser.add_argument(
        "--ticker", type=str, default="^GSPC",
        help="Ticker symbol (e.g., '^GSPC' for S&P 500)"
    )
    parser.add_argument(
        "--nn_day", type=int, default=5,
        help="Minimum streak length for daily data"
    )
    parser.add_argument(
        "--nn_week", type=int, default=3,
        help="Minimum streak length for weekly data"
    )
    parser.add_argument(
        "--nn_month", type=int, default=7,
        help="Minimum streak length for monthly data"
    )
    parser.add_argument(
        "--skip_last_month", type=str2bool, default=True,
        help="Remove last month of dataframe"
    )
    parser.add_argument(
        "--skip_last_week", type=str2bool, default=True,
        help="Remove last week of dataframe"
    )
    parser.add_argument(
        "--skip_last_day", type=str2bool, default=True,
        help="Remove last day of dataframe"
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Display verbose output"
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