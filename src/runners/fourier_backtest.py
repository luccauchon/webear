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
from optimizers.fourier_decomposition import entry as entry_of__fourier_decomposition
import pickle
import copy
from constants import FYAHOO__OUTPUTFILENAME_WEEK, OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from datetime import datetime
import argparse
from argparse import Namespace
import json
import os
from utils import str2bool, transform_path, get_filename_for_dataset, DATASET_AVAILABLE
from runners.fourier_realtime import main as fourier_realtime
from tqdm import tqdm
import time


def main(args):
    ticker   = args.ticker
    col      = args.col
    col_name = (col, ticker)
    dataset_id = args.dataset_id
    n_forecast_length = args.n_forecast_length
    n_forecast_length_in_training = args.n_forecast_length_in_training
    number_of_step_back = args.step_back_range
    scale_forecast = args.scale_forecast
    assert 0.95 <= scale_forecast <= 1.05
    scale_factor_for_ground_truth = args.scale_factor_for_ground_truth
    assert scale_factor_for_ground_truth >= 0.
    success_for_put_credit_spread = args.success_if_pred_lt_gt
    success_for_call_credit_spread = args.success_if_pred_gt_gt
    verbose = args.verbose
    assert (success_for_put_credit_spread and not success_for_call_credit_spread) or (not success_for_put_credit_spread and success_for_call_credit_spread)
    one_dataset_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = copy.deepcopy(master_data_cache[ticker])
    master_data_cache = copy.deepcopy(master_data_cache.sort_index())
    if verbose:
        # --- Parameter Summary ---
        print("\n" + "=" * 50)
        print("BACKTESTING PARAMETERS SUMMARY".center(50))
        print("=" * 50)
        print(f"Ticker               : {ticker}")
        print(f"Column               : {col}")
        print(f"Dataset Frequency    : {dataset_id}")
        print(f"Forecast Length      : {n_forecast_length}")
        print(f"Forecast Length Train: {n_forecast_length_in_training}")
        print(f"Step-Back Range      : {number_of_step_back}")
    results = {}
    for step_back in range(1, number_of_step_back + 1) if not verbose else tqdm(range(1, number_of_step_back + 1)):
        t1 = time.time()
        if len(master_data_cache) < step_back + n_forecast_length + n_forecast_length_in_training:
            continue
        # Create the "Now" dataframe
        df = copy.deepcopy(master_data_cache.iloc[:-step_back])
        data_cache_for_parameter_extraction = copy.deepcopy(df.iloc[:-n_forecast_length])
        data_cache_for_forecasting = copy.deepcopy(df.iloc[-n_forecast_length:])
        # Sanity check
        assert n_forecast_length == len(data_cache_for_forecasting)
        assert data_cache_for_parameter_extraction.index.intersection(data_cache_for_forecasting.index).empty
        # print(f'[{step_back}] {df.index[0].strftime("%Y-%m-%d")}:{df.index[-1].strftime("%Y-%m-%d")}')
        # All data except the last `step_back` rows → for parameter extraction
        data_cache_for_parameter_extraction = copy.deepcopy(df.iloc[:-n_forecast_length])
        # Rows at position `-step_back` → for forecasting
        data_cache_for_forecasting = copy.deepcopy(df.iloc[-n_forecast_length:])
        # print(f'{data_cache_for_parameter_extraction.index[0].strftime("%Y-%m-%d")}:{data_cache_for_parameter_extraction.index[-1].strftime("%Y-%m-%d")} --> {data_cache_for_forecasting.index}')
        assert n_forecast_length == len(data_cache_for_forecasting), f"{len(data_cache_for_forecasting)}"
        assert data_cache_for_parameter_extraction.index.intersection(data_cache_for_forecasting.index).empty, "Indices must be disjoint"
        output_dir = rf"../../stubs/fourier_backtesting_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}/__{step_back}/"
        os.makedirs(output_dir, exist_ok=True)
        the_ground_truth = data_cache_for_forecasting[col_name].values
        assert len(the_ground_truth) == n_forecast_length
        configuration = Namespace(
            algorithms_to_run="0,1,2,3,4,5,6,7,8,9",
            col=col,
            dataset_id=dataset_id,
            length_step_back=n_forecast_length_in_training,
            length_prediction_for_the_future=n_forecast_length,
            n_forecasts=args.n_forecasts,
            q_max_filter=97,
            q_min_filter=3,
            quiet=True,
            older_dataset=None,
            plot_graph=False,
            ticker=ticker,
            use_this_df=data_cache_for_parameter_extraction,
            verbose=False,
        )
        forecasts, mean_forecast = fourier_realtime(configuration)
        assert mean_forecast.shape == the_ground_truth.shape
        results.update({step_back: {'train_t1': data_cache_for_parameter_extraction.index[0].strftime("%Y-%m-%d"),
                                    'train_t2': data_cache_for_parameter_extraction.index[-1].strftime("%Y-%m-%d"),
                                    'gt': the_ground_truth,
                                    'mean_forecast': mean_forecast, 'forecasts': forecasts}})
    step_back_success = []
    for the_step_back, one_result in results.items():
        gt       = one_result['gt'][-1]
        gt_lower = gt * (1 - scale_factor_for_ground_truth)
        gt_upper = gt * (1 + scale_factor_for_ground_truth)
        pred     = one_result['mean_forecast'][-1]*scale_forecast
        if success_for_put_credit_spread:
            if gt_lower <= pred < gt:
                step_back_success.append(the_step_back)
        if success_for_call_credit_spread:
            if gt < pred <= gt_upper:
                step_back_success.append(the_step_back)

    # --- Final Results Summary ---
    total_steps = len(results)
    successful_steps = len(step_back_success)
    success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
    if verbose:
        print("\n" + "=" * 60)
        print("BACKTESTING RESULTS SUMMARY".center(60))
        print("=" * 60)
        print(f"Total backtest windows evaluated   : {total_steps}")
        print(f"Successful predictions (pred < gt) : {successful_steps}")
        print(f"Success rate                       : {success_rate:.1f}%")

        if step_back_success:
            print(f"\nSuccessful step-backs           : {sorted(step_back_success)}")
        else:
            print("\nNo successful predictions.")
    if args.save_to_disk:
        output_summary_file = os.path.join(OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST,
                                           f"backtest_summary_{ticker}_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        serializable_results = {
            step: {
                'train_t1': res['train_t1'],
                'train_t2': res['train_t2'],
                'gt': res['gt'].tolist(),
                'mean_forecast': res['mean_forecast'].tolist()
            }
            for step, res in results.items()
        }
        with open(output_summary_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\nFull results saved to: {output_summary_file}")
    if verbose:
        print("=" * 60)
    return {'success_rate': success_rate}


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Fourier-based stock backtest.")
    parser.add_argument('--ticker', type=str, default='^GSPC',
                        help="Yahoo Finance ticker symbol (default: ^GSPC)")
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                        help="Price column to use (default: Close)")
    parser.add_argument('--dataset_id', type=str, default='month',
                        choices=DATASET_AVAILABLE[1:],
                        help="Dataset frequency: 'week', ... (default: month)")
    parser.add_argument("--n_forecasts", type=int, default=19)
    parser.add_argument('--n_forecast_length', type=int, default=1,
                        help="Number of future steps to forecast (default: 1)")
    parser.add_argument("--n_forecast_length_in_training", type=int, default=4)
    parser.add_argument("--save_to_disk", type=str2bool, default=False)
    parser.add_argument("--scale_forecast", type=float, default=1.)
    parser.add_argument("--scale_factor_for_ground_truth", type=float, default=0.04)
    parser.add_argument("--success_if_pred_lt_gt", type=str2bool, default=True)
    parser.add_argument("--success_if_pred_gt_gt", type=str2bool, default=False)
    parser.add_argument('--step-back-range', type=int, default=5,
                        help="Number of past steps to backtest (default: 5)")
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()
    main(args)