try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from runners.streak_probability import main as streak_probability
from tqdm import tqdm
from collections import defaultdict
from utils import get_filename_for_dataset, str2bool
import pickle
import argparse
import copy
from datetime import datetime
import time


def main(args):
    direction = args.direction
    method = args.method
    frequency = args.frequency
    ticker_name = args.ticker
    deltas = args.deltas
    display_only_nn = args.display_only_nn
    remove_last_element = args.remove_last_element
    verbose = args.verbose
    one_dataset_filename = get_filename_for_dataset(frequency, older_dataset=None)
    if verbose:
        print(f"📁 Using dataset: {one_dataset_filename}")
    with open(one_dataset_filename, 'rb') as f:
        data_cache = pickle.load(f)

    # Extract close_value early so it's available in the delta loop
    df        = data_cache[ticker_name]
    df        = df.sort_index()
    if remove_last_element:
        df = df.iloc[:-1]
    df        = copy.deepcopy(df)
    close_col = ('Close', ticker_name)
    close_value        = df[close_col].iloc[-1]
    before_close_value = df[close_col].iloc[-2]
    last_date          = df[close_col].index[-1]
    # Group results by NN: {NN: [(delta, prob, count, total_streaks), ...]}
    grouped_results = defaultdict(list)
    for delta in tqdm(deltas, desc="🌀 Processing deltas") if verbose else deltas:
        result = streak_probability(
            direction=direction,
            method=method,
            older_dataset='',
            bold=-1,
            frequency=frequency,
            delta=delta / 100.0,
            ticker_name=ticker_name,
            verbose=False,
            bring_my_own_df=df,
        )
        for k, v in result.items():
            NN = v['NN']
            prob = v['prob']
            count = v['count']
            total_streaks = v['total_streaks']
            vl_dl        = close_value * (1 - delta / 100.0)        if direction == 'neg' else close_value * (1 + delta / 100.0)
            before_vl_dl = before_close_value * (1 - delta / 100.0) if direction == 'neg' else before_close_value * (1 + delta / 100.0)
            grouped_results[NN].append((delta, prob, count, total_streaks, vl_dl, before_vl_dl, last_date))

    # Sort NN values
    sorted_NNs = sorted(grouped_results.keys())
    now = datetime.now()
    # Pretty header
    direction_label = "Negative" if direction == "neg" else "Positive"
    emoji = "📉" if direction == "neg" else "📈"
    if verbose:
        print("\n" + "="*70)
        print(f"{emoji}  Streak Probability Analysis — {direction_label} Price Moves")
        print(f"📊 Ticker: {ticker_name} | Last Close: {close_value:,.2f} | Last Date in {frequency} DF: {last_date.strftime("%Y-%m-%d")} | Now: {now.strftime('%Y-%m-%d')}")
        print("="*70)

    for NN in sorted_NNs:
        if display_only_nn is not None and display_only_nn != NN:
            continue
        if verbose:
            tmp = 'First occurence' if 0 == NN else f'Already {NN} occurences'
            print(f"\n🔹 Streaks ≥ {NN+1} ({frequency}-frequency) ({tmp}):")
        entries = sorted(grouped_results[NN], key=lambda x: x[0])  # sort by delta
        for delta, prob, count, total_streaks, vl_dl, before_vl_dl, last_date in entries:
            cl_dir = "below" if direction == "neg" else "above"
            if verbose:
                print(
                    f"    {prob:>7.2%} ({count:>4d} / {total_streaks:>4d}) → "
                    f"Close {cl_dir} {vl_dl:,.1f} "
                    f"(Δ = {delta:>4.1f}%)"
                )
    if verbose:
        print("\n✅ Done.\n")
    return grouped_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="📊 Compute streak probabilities for financial time series."
    )
    parser.add_argument(
        "--direction", choices=["pos", "neg"], default="pos",
        help="Direction of streak: 'pos' (positive) or 'neg' (negative)"
    )
    parser.add_argument(
        "--method", type=str, default="prev_close",
        help="Method used for comparison (e.g., 'prev_close')"
    )
    parser.add_argument(
        "--frequency", type=str, default="day",
        help="Data frequency (e.g., 'day', 'hour')"
    )
    parser.add_argument(
        "--ticker", type=str, default="^GSPC",
        help="Ticker symbol (e.g., '^GSPC' for S&P 500)"
    )
    parser.add_argument(
        "--deltas", nargs="+", type=float,
        default=[0, 0.1, 0.25, 0.50, 0.75, 0.9, 1.0, 1.25, 1.33, 1.5, 1.75, 2.0, 2.25, 2.5],
        help="List of delta percentages"
    )
    parser.add_argument(
        "--display-only-nn", type=int, default=None,
        help="Only display results for streaks of this exact length NN (e.g., 5)"
    )
    parser.add_argument(
        "--remove_last_element", type=str2bool, default=False,
        help="Remove last element in df since it is incomplete"
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