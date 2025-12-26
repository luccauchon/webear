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
from utils import str2bool, transform_path, get_filename_for_dataset, DATASET_AVAILABLE, IS_RUNNING_ON_CASIR
from runners.fourier_backtest import main as fourier_backtest
from tqdm import tqdm
import time
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import numpy as np
import time
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import signal
import sys

# Custom exception for timeout
class TimeExceededError(Exception):
    pass


def main(args):
    ticker = args.ticker
    col = args.col
    dataset_id = args.dataset_id
    n_forecast_length = args.n_forecast_length
    number_of_step_back = args.step_back_range
    show_n_top_configurations = 5
    verbose = args.verbose

    # Define search space
    space = [
        Integer(1, 299, name='n_forecast_length_in_training'),
        Integer(1, 299, name='n_forecasts')  # adjust upper bound as needed
    ]
    n_calls = int(0.1 * 299*299)  # 10% of the space ?
    # Keep track of results
    results = []

    # Set time limit (e.g., 600 seconds = 10 minutes)
    time_limit_seconds = 84600
    start_time = time.time()

    @use_named_args(space)
    def objective(n_forecast_length_in_training, n_forecasts):
        # Check if time limit exceeded
        elapsed = time.time() - start_time
        if elapsed > time_limit_seconds and IS_RUNNING_ON_CASIR:
            print(f"\n⏰ Time limit ({time_limit_seconds}s) exceeded. Stopping optimization.")
            raise TimeExceededError()

        configuration = Namespace(
            col=col,
            dataset_id=dataset_id,
            n_forecast_length=n_forecast_length,
            n_forecast_length_in_training=n_forecast_length_in_training,
            n_forecasts=n_forecasts,
            step_back_range=number_of_step_back,
            save_to_disk=False,
            scale_forecast=0.98,
            success_if_pred_lt_gt=True,
            success_if_pred_gt_gt=False,
            ticker=ticker,
            verbose=verbose
        )
        result = fourier_backtest(configuration)
        success_rate = result['success_rate']

        # Store full result for later analysis
        results.append({
            'config': copy.deepcopy(configuration),
            'result': result,
            'success_rate': success_rate
        })

        # skopt *minimizes*, so return negative success rate
        return -success_rate
    if IS_RUNNING_ON_CASIR:
        print(f"🚀 Starting Bayesian optimization (time limit: {time_limit_seconds}s)...")

    try:
        res2 = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=42,
            verbose=False
        )
    except TimeExceededError:
        # Optimization stopped due to time limit — that's OK
        pass

    # Proceed to report results even if time-limited
    if not results:
        print("❌ No evaluations completed within time limit.")
        return

    # Sort top results by success rate (descending)
    top_results = sorted(results, key=lambda x: x['success_rate'], reverse=True)[:show_n_top_configurations]

    # Nice output
    print("\n" + "=" * 60)
    print(f"🏆 TOP {show_n_top_configurations} CONFIGURATIONS BY SUCCESS RATE")
    print("=" * 60)
    for i, res in enumerate(top_results, 1):
        cfg = res['config']
        sr = res['success_rate'] / 100.0
        print(f"{i}. Success Rate: {sr:.2%}")
        print(f"   • Forecast Length (training): {cfg.n_forecast_length_in_training}")
        print(f"   • Forecast Length: {n_forecast_length}")
        print(f"   • Number of Forecasts: {cfg.n_forecasts}")
        print("-" * 60)


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
    parser.add_argument('--n_forecast_length', type=int, default=1,
                        help="Number of future steps to forecast (default: 1)")
    parser.add_argument('--step-back-range', type=int, default=300,
                        help="Number of past steps to backtest (default: 300)")
    parser.add_argument("--scale_forecast", type=float, default=0.98)
    parser.add_argument("--success_if_pred_lt_gt", type=str2bool, default=True)
    parser.add_argument("--success_if_pred_gt_gt", type=str2bool, default=False)
    parser.add_argument('--verbose', type=str2bool, default=False)
    args = parser.parse_args()
    main(args)