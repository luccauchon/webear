try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
import pickle
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool
import copy
import numpy as np
from tqdm import tqdm
from runners.MMI_realtime import main as MMI_realtime
from argparse import Namespace
import pandas as pd


def main(args):
    if args.verbose:
        # --- Nicely print the arguments ---
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            if 'master_data_cache' in arg:
                print(f"    {arg:.<40} {value.index[0].strftime('%Y-%m-%d')} to {value.index[-1].strftime('%Y-%m-%d')} ({args.one_dataset_filename})")
                continue
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)
    assert args.look_ahead >= 1.
    one_dataset_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = copy.deepcopy(master_data_cache[args.ticker])
    master_data_cache = copy.deepcopy(master_data_cache.sort_index())
    results = []
    for step_back in range(1, args.step_back_range + 1) if not args.verbose else tqdm(range(1, args.step_back_range + 1)):
        past_df   = master_data_cache.iloc[:-step_back]
        future_df = master_data_cache.iloc[-step_back:]
        # print(f"{past_df.index[0]}/{past_df.index[-1]} --> {future_df.index[0]}/{future_df.index[-1]}")
        assert 0 == len(past_df.index.intersection(future_df.index))
        if args.look_ahead > len(future_df) or 0 == len(past_df):
            continue
        configuration = Namespace(
            master_data_cache=copy.deepcopy(past_df),
            ticker=args.ticker, col=args.col,
            mmi_period=args.mmi_period,
            mmi_trend_max=args.mmi_trend_max,
            sma_period=args.sma_period,
            return_threshold=args.return_threshold,
            verbose=False,
        )
        _result       = MMI_realtime(configuration)

        close_col     = (args.col, args.ticker)
        close_prices  = past_df[close_col]
        # print(f"{past_df.index[-1]} --> {future_df.index[args.look_ahead - 1]}")
        future_price  = future_df[close_col].iloc[args.look_ahead - 1]
        current_price = close_prices.iloc[-1]
        future_return = (future_price / current_price) - 1
        if future_return > args.return_threshold:
            gt = "Bull"
        elif future_return < -args.return_threshold:
            gt = "Bear"
        else:
            gt = "Choppy"
        _result.update({"ground_truth": gt,"return": future_return,})
        results.append(_result)
    results_df = pd.DataFrame(results)
    if 0 == len(results_df):
        return 0, results_df
    accuracy = (results_df.signal == results_df.ground_truth).mean()
    return accuracy, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
    parser.add_argument("--look_ahead", type=int, default=1)
    parser.add_argument("--mmi_period", type=int, default=100)
    parser.add_argument("--mmi_trend_max", type=int, default=70)
    parser.add_argument("--sma_period", type=int, default=50)
    parser.add_argument("--return_threshold", type=float, default=0.01)
    parser.add_argument('--step-back-range', type=int, default=5,
                        help="Number of historical time windows to simulate (rolling backtest depth).")
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    accuracy, results_df = main(args)
    if args.verbose:
        print(accuracy)
        print(results_df)