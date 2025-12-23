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
from runners.fourier_backtest import main as fourier_backtest
from tqdm import tqdm
import time


def main(args):
    ticker   = args.ticker
    col      = args.col
    col_name = (col, ticker)
    dataset_id = args.dataset_id
    n_forecast_length = args.n_forecast_length
    number_of_step_back = args.step_back_range
    show_n_top_configurations = 5
    verbose = args.verbose

    experiences, results = [], []
    for n_forecast_length_in_training in (1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,):
        for n_forecasts in (19,):  # Nombre de modèles à conserver
            experiences.append({'n_forecast_length_in_training': n_forecast_length_in_training,
                                'n_forecasts': n_forecasts})
    for one_experience in tqdm(experiences):
        configuration = Namespace(
            col=col,
            dataset_id=dataset_id,
            n_forecast_length=n_forecast_length,
            n_forecast_length_in_training=one_experience['n_forecast_length_in_training'],
            n_forecasts=one_experience['n_forecasts'],
            step_back_range=number_of_step_back,
            save_to_disk=False,
            ticker=ticker,
            verbose=verbose)
        result = fourier_backtest(configuration)
        # Save config + result
        results.append({
            'config': configuration,
            'result': result,
            'success_rate': result['success_rate']
        })
    # Sort by success_rate descending
    top_results = sorted(results, key=lambda x: x['success_rate'], reverse=True)[:show_n_top_configurations]

    # Nice output
    print("\n" + "=" * 60)
    print(f"🏆 TOP {show_n_top_configurations} CONFIGURATIONS BY SUCCESS RATE")
    print("=" * 60)
    for i, res in enumerate(top_results, 1):
        cfg = res['config']
        sr = res['success_rate'] / 100.
        print(f"{i}. Success Rate: {sr:.2%}")
        print(f"   • Forecast Length (training): {cfg.n_forecast_length_in_training}")
        print(f"   • Forecast Length : {n_forecast_length}")
        print(f"   • Number of Forecasts (# models used to aggregate): {cfg.n_forecasts}")
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
    parser.add_argument('--verbose', type=str2bool, default=False)
    args = parser.parse_args()
    main(args)