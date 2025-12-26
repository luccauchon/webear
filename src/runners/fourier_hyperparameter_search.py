# Standard imports
import pickle
import copy
import json
import os
import sys
import time
import signal
from datetime import datetime
from multiprocessing import freeze_support
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import argparse
# Third-party optimization library
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# Local custom modules
try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

# Import domain-specific components
from optimizers.fourier_decomposition import entry as entry_of__fourier_decomposition
from constants import FYAHOO__OUTPUTFILENAME_WEEK, OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST
from utils import (
    str2bool,
    transform_path,
    get_filename_for_dataset,
    DATASET_AVAILABLE,
    IS_RUNNING_ON_CASIR
)
from runners.fourier_backtest import main as fourier_backtest
from tqdm import tqdm


# ==============================
# Custom Exception for Timeouts
# ==============================
class TimeExceededError(Exception):
    """Raised when the optimization exceeds the allowed time limit."""
    pass


# ==============================
# Main Optimization Routine
# ==============================
def main(args):
    """
    Runs Bayesian optimization over Fourier-based stock forecasting configurations
    to maximize the success rate of options-like strategies (credit spreads).
    """
    # Extract CLI arguments
    ticker = args.ticker
    col = args.col
    dataset_id = args.dataset_id
    n_forecast_length = args.n_forecast_length  # Fixed forecast horizon for evaluation
    number_of_step_back = args.step_back_range  # How far back to simulate
    show_n_top_configurations = 5
    verbose = args.verbose
    scale_factor_for_ground_truth = args.scale_factor_for_ground_truth

    # Strategy flags: must choose exactly one (call or put credit spread)
    sell_call_credit_spread = args.sell_call_credit_spread
    sell_put_credit_spread = args.sell_put_credit_spread
    assert (sell_call_credit_spread and not sell_put_credit_spread) or \
           (not sell_call_credit_spread and sell_put_credit_spread), \
           "Exactly one of sell_call_credit_spread or sell_put_credit_spread must be True."

    # Define Bayesian optimization search space:
    # - n_forecast_length_in_training: how many steps the model uses during training
    # - n_forecasts: number of forecast attempts per backtest window
    space = [
        Integer(1, 99, name='n_forecast_length_in_training'),
        Integer(1, 99, name='n_forecasts'),
        Real(-2., 2., name='scale_factor'),
    ]

    # Number of evaluations: ~10% of full grid (299x299 ≈ 89k → ~8.9k calls → too high!)
    # ⚠️ Consider reducing this or using a smarter n_calls strategy
    n_calls = int(0.1 * 299 * 299)

    # Storage for all evaluated configurations and results
    results = []

    # Time limit handling
    time_limit_seconds = args.time_limit_seconds
    start_time = time.time()

    # Objective function for Bayesian optimization (to be minimized)
    @use_named_args(space)
    def objective(n_forecast_length_in_training, n_forecasts, scale_factor):
        """Evaluates a configuration by running a Fourier backtest and returning -success_rate."""
        # Enforce time limit
        elapsed = time.time() - start_time
        if time_limit_seconds != -1 and elapsed > time_limit_seconds:
            print(f"\n⏰ Time limit ({time_limit_seconds}s) exceeded. Stopping optimization.")
            raise TimeExceededError()

        # Build configuration namespace for the backtester
        configuration = Namespace(
            col=col,
            dataset_id=dataset_id,
            n_forecast_length=n_forecast_length,  # fixed from CLI
            n_forecast_length_in_training=n_forecast_length_in_training,
            n_forecasts=n_forecasts,
            step_back_range=number_of_step_back,
            save_to_disk=False,
            scale_forecast=scale_factor,  # adjusts prediction thresholds
            scale_factor_for_ground_truth=scale_factor_for_ground_truth,
            success_if_pred_lt_gt=sell_put_credit_spread,   # success if pred < actual (for puts)
            success_if_pred_gt_gt=sell_call_credit_spread,  # success if pred > actual (for calls)
            ticker=ticker,
            verbose=verbose
        )

        # Run backtest
        result = fourier_backtest(configuration)
        success_rate = result['success_rate']

        # Save full result for post-analysis
        results.append({
            'config': copy.deepcopy(configuration),
            'result': result,
            'success_rate': success_rate
        })

        # skopt minimizes → return negative success rate to maximize it
        return -success_rate

    # Launch optimization
    if time_limit_seconds != -1:
        print(f"🚀 Starting Bayesian optimization (time limit: {time_limit_seconds}s)...")
    else:
        print(f"🚀 Starting Bayesian optimization (no time limit)...")

    try:
        res2 = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=42,
            verbose=False  # tqdm progress is handled via objective calls
        )
    except TimeExceededError:
        # Graceful exit on timeout — results collected so far are still valid
        pass

    # Handle case where no evaluations completed
    if not results:
        print("❌ No evaluations completed within time limit.")
        return

    # Sort and display top configurations
    top_results = sorted(results, key=lambda x: x['success_rate'], reverse=True)[:show_n_top_configurations]

    print("\n" + "=" * 60)
    print(f"🏆 TOP {show_n_top_configurations} CONFIGURATIONS BY SUCCESS RATE")
    print("=" * 60)
    for i, res in enumerate(top_results, 1):
        cfg = res['config']
        sr = res['success_rate'] / 100.0  # convert from percentage (e.g., 85 → 0.85)
        print(f"{i}. Success Rate: {sr:.2%}")
        print(f"   • Forecast Length (training): {cfg.n_forecast_length_in_training}")
        print(f"   • Forecast Length (evaluation): {n_forecast_length}")  # fixed value
        print(f"   • Number of Forecasts: {cfg.n_forecasts}")
        print("-" * 60)


# ==============================
# CLI Argument Parsing
# ==============================
if __name__ == "__main__":
    freeze_support()  # Required for multiprocessing on Windows

    parser = argparse.ArgumentParser(description="Run Bayesian optimization on Fourier-based stock backtest.")

    # Core data parameters
    parser.add_argument('--ticker', type=str, default='^GSPC',
                        help="Yahoo Finance ticker symbol (default: ^GSPC for S&P 500)")

    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                        help="Price column to forecast (default: 'Close')")

    parser.add_argument('--dataset_id', type=str, default='month',
                        choices=DATASET_AVAILABLE[1:],  # assumes first entry is invalid/reserved
                        help="Dataset frequency: e.g., 'week', 'month' (default: 'month')")

    # Forecasting setup
    parser.add_argument('--n_forecast_length', type=int, default=1,
                        help="Number of future time steps to predict in evaluation (default: 1)")

    parser.add_argument('--step-back-range', type=int, default=300,
                        help="Number of historical windows to backtest over (default: 300)")

    # Options strategy parameters
    parser.add_argument("--scale_factor_for_ground_truth", type=float, default=0.04,
                        help="Defines the zone where the prediction becomes a success.")

    parser.add_argument("--sell_call_credit_spread", type=str2bool, default=True,
                        help="Optimize for call credit spread strategy (default: True)")

    parser.add_argument("--sell_put_credit_spread", type=str2bool, default=False,
                        help="Optimize for put credit spread strategy (default: False). "
                             "Exactly one of call/put must be True.")

    # Runtime control
    parser.add_argument('--time_limit_seconds', type=int, default=-1,
                        help="Maximum time (in seconds) to spend optimizing. "
                             "Use -1 for no limit (default: -1)")

    parser.add_argument('--verbose', type=str2bool, default=False,
                        help="Enable detailed logging during backtests (default: False)")

    # Parse and run
    args = parser.parse_args()
    main(args)