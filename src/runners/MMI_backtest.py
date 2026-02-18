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
    vix__master_data_cache = master_data_cache['^VIX']
    vix__master_data_cache = copy.deepcopy(vix__master_data_cache.sort_index()[('Close', '^VIX')])
    master_data_cache      = copy.deepcopy(master_data_cache[args.ticker])
    master_data_cache      = copy.deepcopy(master_data_cache.sort_index())
    results = []
    for step_back in range(1, args.step_back_range + 1) if not args.verbose else tqdm(range(1, args.step_back_range + 1)):
        past_df   = master_data_cache.iloc[:-step_back]
        future_df = master_data_cache.iloc[-step_back:]
        vix_df    = vix__master_data_cache.iloc[:-step_back]
        if args.look_ahead > len(future_df) or 0 == len(past_df):
            continue
        if args.use_vix:
            if 0 == len(vix_df):
                continue
            assert vix_df.index[-1].strftime('%Y-%m-%d') == past_df.index[-1].strftime('%Y-%m-%d')
        # print(f"{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} --> {future_df.index[0].strftime('%Y-%m-%d')}/{future_df.index[-1].strftime('%Y-%m-%d')}")
        assert 0 == len(past_df.index.intersection(future_df.index))
        configuration = Namespace(
            master_data_cache=copy.deepcopy(past_df),
            ticker=args.ticker, col=args.col,
            mmi_period=args.mmi_period,
            mmi_trend_max=args.mmi_trend_max,
            sma_period=args.sma_period,
            return_threshold=args.return_threshold,
            use_ema=args.use_ema,
            keep_last_step=True,
            verbose=False,
        )
        _result       = MMI_realtime(configuration)
        close_col     = (args.col, args.ticker)
        close_prices  = past_df[close_col]
        # print(f"{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} :: {future_df.index[args.look_ahead - 1].strftime('%Y-%m-%d')}\n\n")
        # continue
        future_price  = future_df[close_col].iloc[args.look_ahead - 1]
        current_price = close_prices.iloc[-1]
        future_return = (future_price / current_price) - 1
        if future_return > args.return_threshold:
            gt = "Bull"
        elif future_return < -args.return_threshold:
            gt = "Bear"
        else:
            gt = "Choppy"
        if args.use_vix:
            _result.update({"ground_truth": gt, "return": future_return, "vix": vix_df.iloc[-1]})
        else:
            _result.update({"ground_truth": gt, "return": future_return, })
        results.append(_result)
    _results_df = pd.DataFrame(results)
    if 0 == len(_results_df):
        return {'overall_accuracy': 0,'bull_accuracy': 0,'bear_accuracy': 0,'n_bull': 0,'n_bear': 0,}, _results_df
    # Overall accuracy (unchanged)
    overall_accuracy = (_results_df.signal == _results_df.ground_truth).mean()
    if args.use_vix:
        # Sanity check
        assert overall_accuracy == _results_df[_results_df.vix < 9999].apply(lambda x: x.signal == x.ground_truth, axis=1).mean()
        # Filter for low volatility and then calculate accuracy
        for vix_value in (9, 10, 20, 30, 40, 60, 99):
            overall_accuracy = _results_df[_results_df.vix < vix_value].apply(lambda x: x.signal == x.ground_truth, axis=1).mean()
            if np.isnan(overall_accuracy):
                continue
            print(f"\tVIX below {vix_value} -> {overall_accuracy:.4f}")
    # --- Accuracy when Ground Truth is "Bull" ---
    bull_mask = _results_df.ground_truth == "Bull"
    if bull_mask.any():
        bull_accuracy = (_results_df.loc[bull_mask, 'signal'] == "Bull").mean()
    else:
        bull_accuracy = 0.0

    # --- Accuracy when Ground Truth is "Bear" ---
    bear_mask = _results_df.ground_truth == "Bear"
    if bear_mask.any():
        bear_accuracy = (_results_df.loc[bear_mask, 'signal'] == "Bear").mean()
    else:
        bear_accuracy = 0.0

    # Optional: print or return detailed metrics
    if args.verbose:
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({len(_results_df)}) samples")
        print(f"Bull Accuracy (when GT=Bull): {bull_accuracy:.4f} ({bull_mask.sum()} samples)")
        print(f"Bear Accuracy (when GT=Bear): {bear_accuracy:.4f} ({bear_mask.sum()} samples)")

    # Return a dict or tuple with all metrics
    metrics = {
        'overall_accuracy': overall_accuracy,
        'bull_accuracy': bull_accuracy,
        'bear_accuracy': bear_accuracy,
        'n_bull': bull_mask.sum(),
        'n_bear': bear_mask.sum(),
    }

    return metrics, _results_df


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
    parser.add_argument('--use_ema', type=str2bool, default=False)
    parser.add_argument("--return_threshold", type=float, default=0.01)
    parser.add_argument('--step-back-range', type=int, default=5,
                        help="Number of historical time windows to simulate (rolling backtest depth).")
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--use_vix', action='store_true',
                        help="Save the vix value per run in backtest")
    args = parser.parse_args()

    metrics, results_df = main(args)
    print(metrics)
    print("\nFinal Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("\nResults DataFrame:")
    print(results_df)