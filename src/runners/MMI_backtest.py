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
            filter_open_gaps=args.filter_open_gaps,
        )
        _result       = MMI_realtime(configuration)
        close_col     = (args.col, args.ticker)
        close_prices  = past_df[close_col]
        # open_prices   = past_df[("Open", args.ticker)]
        # high_prices   = past_df[("High", args.ticker)]
        # low_prices    = past_df[("Low", args.ticker)]
        # print(f"{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} :: {future_df.index[args.look_ahead - 1].strftime('%Y-%m-%d')}\n\n")
        # continue
        future_price      = future_df[close_col].iloc[args.look_ahead - 1]
        # future_open_price = future_df[("Open", args.ticker)].iloc[args.look_ahead - 1]
        # future_high_price = future_df[("High", args.ticker)].iloc[args.look_ahead - 1]
        # future_low_price  = future_df[("Low", args.ticker)].iloc[args.look_ahead - 1]
        current_price = close_prices.iloc[-1]
        future_return = (future_price / current_price) - 1
        if future_return > args.return_threshold:
            gt = "Bull"
        elif future_return < -args.return_threshold:
            gt = "Bear"
        else:
            gt = "Choppy"

        # --- Check if Open is inside Precedent Day's Range ---
        # Precedent Day = Last row of past_df
        # Current Day   = First row of future_df (Execution day)
        prec_low = past_df[("Low", args.ticker)].iloc[-1]
        prec_high = past_df[("High", args.ticker)].iloc[-1]
        curr_open = future_df[("Open", args.ticker)].iloc[args.look_ahead - 1]

        is_inside = (curr_open >= prec_low) and (curr_open <= prec_high)
        _result['is_inside_open'] = is_inside
        # ---------------------------------------------------------------

        if args.use_vix:
            _result.update({"ground_truth": gt, "return": future_return, "vix": vix_df.iloc[-1]})
        else:
            _result.update({"ground_truth": gt, "return": future_return, })
        results.append(_result)
    _results_df = pd.DataFrame(results)
    if 0 == len(_results_df):
        return {'overall_accuracy': 0, 'bull_accuracy': 0, 'bear_accuracy': 0, 'n_bull': 0, 'n_bear': 0, }, _results_df

    # --- Filter DataFrame based on new option ---
    eval_df = _results_df
    length_of_eval_before_filtering = len(eval_df)
    if args.filter_inside_open:
        eval_df = _results_df[_results_df['is_inside_open'] == True]

    if 0 == len(eval_df):
        # If filtering removes all data, return zeros but indicate sample count is 0
        return {'overall_accuracy': 0, 'bull_accuracy': 0, 'bear_accuracy': 0, 'n_bull': 0, 'n_bear': 0, }, _results_df

    # Overall accuracy (calculated on filtered eval_df if option is set)
    overall_accuracy = (eval_df.signal == eval_df.ground_truth).mean()

    if args.use_vix:
        # Sanity check (applied to filtered df)
        # Note: We assume vix column exists if args.use_vix is True
        assert overall_accuracy == eval_df[eval_df.vix < 9999].apply(lambda x: x.signal == x.ground_truth, axis=1).mean()
        # Filter for low volatility and then calculate accuracy
        for vix_value in (9, 10, 20, 30, 40, 60, 99):
            # Calculate accuracy on the subset of eval_df that meets VIX criteria
            vix_subset = eval_df[eval_df.vix < vix_value]
            if len(vix_subset) > 0:
                overall_accuracy = vix_subset.apply(lambda x: x.signal == x.ground_truth, axis=1).mean()
                if not np.isnan(overall_accuracy):
                    print(f"\tVIX below {vix_value} -> {overall_accuracy:.4f} (on inside-open samples)")

    # --- Accuracy when Ground Truth is "Bull" ---
    bull_mask = eval_df.ground_truth == "Bull"
    if bull_mask.any():
        bull_accuracy = (eval_df.loc[bull_mask, 'signal'] == "Bull").mean()
    else:
        bull_accuracy = 0.0

    # --- Accuracy when Ground Truth is "Bear" ---
    bear_mask = eval_df.ground_truth == "Bear"
    if bear_mask.any():
        bear_accuracy = (eval_df.loc[bear_mask, 'signal'] == "Bear").mean()
    else:
        bear_accuracy = 0.0

    # Optional: print or return detailed metrics
    if args.verbose:
        filter_note = " (Filtered: Inside Open)" if args.filter_inside_open else ""
        _tmp_str_3 = f' (down from {length_of_eval_before_filtering})' if args.filter_inside_open else ""
        print(f"Overall Accuracy{filter_note}: {overall_accuracy:.4f} ({len(eval_df)}) samples{_tmp_str_3}")
        print(f"Bull Accuracy (when GT=Bull){filter_note}: {bull_accuracy:.4f} ({bull_mask.sum()} samples)")
        print(f"Bear Accuracy (when GT=Bear){filter_note}: {bear_accuracy:.4f} ({bear_mask.sum()} samples)")
        if args.filter_inside_open:
            print(f"\tTotal samples before filter: {len(_results_df)}")
            print(f"\tTotal samples after filter:  {len(eval_df)}")

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
    parser.add_argument('--use_vix', type=str2bool,
                        help="Save the vix value per run in backtest")
    parser.add_argument('--filter_open_gaps', type=str2bool, default=False,
                        help="Remove rows where Open > Prev High or Open < Prev Low")
    parser.add_argument('--filter_inside_open', type=str2bool,
                        help="Compute accuracy only if Current Open is between Precedent Day's Low and High")
    # -------------------
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