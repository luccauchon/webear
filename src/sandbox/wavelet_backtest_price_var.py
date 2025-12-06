try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import os
from datetime import datetime
import json
import numpy as np
import random
import argparse
from multiprocessing import freeze_support, Lock, Process, Queue, Value
from optimizers.wavelet_opt import main as wavelet_optimizer_entry_point
from argparse import Namespace
import matplotlib.pyplot as plt
import pickle
from constants import FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_DAY
from tqdm import tqdm
from runners.wavelet_realtime import main as wavelet_realtime_entry_point
from utils import format_execution_time, is_friday, is_monday, is_thursday
import pandas as pd
import time
import math


def main(args):
    ticker = args.ticker
    col = args.col
    col_name = (col, ticker)
    dataset_id = args.dataset_id
    n_forecast_length = args.n_forecast_length
    n_forecast_length_in_training = args.n_forecast_length_in_training
    thresholds_ep = args.thresholds_ep
    n_models_to_keep = args.n_models_to_keep
    number_of_step_back = args.step_back_range
    start_of_back_range = args.start_of_back_range
    exit_strategy = args.strategy_for_exit
    threshold_for_shape_similarity = args.threshold_for_shape_similarity
    verbose = args.verbose
    use_last_week_only=args.use_last_week_only
    floor_and_ceil = 5.0
    assert start_of_back_range>0 and start_of_back_range<number_of_step_back
    if dataset_id == 'day':
        df_filename = FYAHOO__OUTPUTFILENAME_DAY
    elif dataset_id == 'week':
        df_filename = FYAHOO__OUTPUTFILENAME_WEEK
    with open(df_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = master_data_cache[ticker].copy()
    all__n_forecast_length_in_training = range(1, n_forecast_length_in_training)
    # --- Parameter Summary ---
    print("\n" + "="*50)
    print("BACKTESTING PARAMETERS SUMMARY".center(50))
    print("="*50)
    print(f"Ticker               : {ticker}")
    print(f"Column               : {col}")
    print(f"Dataset Frequency    : {dataset_id}")
    print(f"Forecast Length      : {n_forecast_length}")
    print(f"* Forecast Length Train: {all__n_forecast_length_in_training}")
    print(f"N models to keep     : {n_models_to_keep}")
    print(f"T Shape Similarity   : {threshold_for_shape_similarity}")
    print(f"Thresholds (EP)      : {thresholds_ep}")
    print(f"Step-Back Range      : {number_of_step_back}")
    print(f"Start Step-Back Range: {start_of_back_range}")
    print(f"Data File            : {df_filename}")
    print(f"Exit strategy        : {exit_strategy}")
    print(f"Last week of month   : {use_last_week_only}")
    print(f"Verbose              : {verbose}")
    print("="*50)
    suive_performance = {}
    tt1 = time.time()
    for __n_forecast_length_in_training in all__n_forecast_length_in_training:
        tt2 = time.time()
        performance = {'rmse':[], 'slope':[], 'slope_success':[]}
        for step_back in range(start_of_back_range, number_of_step_back + 1) if verbose else (range(start_of_back_range, number_of_step_back + 1)):
            t1 = time.time()
            # Create the "Now" dataframe
            df = master_data_cache.iloc[:-step_back].copy()
            # print(f'{df.index[0].strftime("%Y-%m-%d")}:{df.index[-1].strftime("%Y-%m-%d")}')

            # All data except the last `step_back` rows → for parameter extraction
            data_cache_for_parameter_extraction = df.iloc[:-n_forecast_length].copy()

            if use_last_week_only:
                last_date = data_cache_for_parameter_extraction.index[-1]
                month_end = last_date + pd.offsets.MonthEnd(0)
                is_in_last_week = last_date >= (month_end - pd.Timedelta(days=6))

                if not is_in_last_week:
                    continue

            # Rows at position `-step_back` → for forecasting
            data_cache_for_forecasting = df.iloc[-n_forecast_length: ].copy()
            # print(f'{data_cache_for_parameter_extraction.index[0].strftime("%Y-%m-%d")}:{data_cache_for_parameter_extraction.index[-1].strftime("%Y-%m-%d")} --> {data_cache_for_forecasting.index}')
            day_of_the_trade = data_cache_for_parameter_extraction.index[-1]
            price_paid_for_the_trade = data_cache_for_parameter_extraction[col_name].values[-1]
            #print(price_paid_for_the_trade)
            #continue
            if 2 == n_forecast_length and dataset_id == 'day':
                if is_friday(day_of_the_trade) or is_thursday(day_of_the_trade):
                    continue
            assert n_forecast_length == len(data_cache_for_forecasting), f"{len(data_cache_for_forecasting)}"
            assert data_cache_for_parameter_extraction.index.intersection(data_cache_for_forecasting.index).empty, "Indices must be disjoint"
            output_dir = rf"../../stubs/wavelet_backtesting_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}/__{step_back}/"
            os.makedirs(output_dir, exist_ok=True)
            args = Namespace(
                master_data_cache=data_cache_for_parameter_extraction.copy(),
                ticker=ticker, col=col,
                output_dir=output_dir,
                dataset_id=dataset_id,
                n_forecast_length=n_forecast_length,
                n_forecast_length_in_training=__n_forecast_length_in_training,
                thresholds_ep=thresholds_ep,
                plot_graph=False,
                use_given_gt_truth=data_cache_for_forecasting[col_name].values,
                display_tqdm=False,
                strategy_for_exit=args.strategy_for_exit,
                n_models_to_keep=n_models_to_keep,
                threshold_for_shape_similarity=threshold_for_shape_similarity,
                verbose=False,
            )
            user_instruction, misc_returned = wavelet_realtime_entry_point(args)
            mean_forecast = misc_returned['mean_forecast']
            th2, th1 = None, None
            gt = data_cache_for_forecasting[col_name].values.copy()
            assert mean_forecast.shape == gt.shape, f"{mean_forecast.shape=}   {gt.shape=}"
            rmse = np.mean((mean_forecast - gt) ** 2) ** 0.5
            performance['rmse'].append(rmse)
            the_dates = [f'{dd.strftime("%Y-%m-%d")}' for dd in data_cache_for_forecasting.index]
            errors = ((mean_forecast - gt) ** 2)**0.5
            errors_pct = ' '.join(f'{(e/g*100):.2f}%' if g else 'N/A' for e, g in zip(errors, gt))
            mslope_pred = np.polyfit(np.arange(len(mean_forecast)), mean_forecast, 1)[0]
            mslope_gt   = np.polyfit(np.arange(len(gt)), gt, 1)[0]
            ddslope = ('+' if mslope_gt > 0 else '-') + ('+' if mslope_pred > 0 else '-')
            performance['slope'].append(ddslope)
            threshold_up_ep, threshold_down_ep = float(eval(thresholds_ep)[0]), float(eval(thresholds_ep)[1])
            upper_line = price_paid_for_the_trade * (1. + threshold_up_ep)  # th1
            upper_line = math.ceil(upper_line / floor_and_ceil) * floor_and_ceil
            lower_line = price_paid_for_the_trade * (1. - threshold_down_ep)  # th2
            lower_line = math.floor(lower_line / floor_and_ceil) * floor_and_ceil

            success = True
            if ddslope not in ['++','--']:  # GT,PRED
                if ddslope == '+-':
                    if 0 == np.count_nonzero(gt[1:]<upper_line):
                        success = False
                if ddslope == '-+':
                    if 0 == np.count_nonzero(gt[1:]>lower_line):
                        success = False
            performance['slope_success'].append(success)
            if not success or verbose:
                print(f'{ddslope}  RMSE:{rmse.astype(int)}  {errors.astype(int)} -> {errors_pct}      {data_cache_for_parameter_extraction.index[0].strftime("%Y-%m-%d")}:{data_cache_for_parameter_extraction.index[-1].strftime("%Y-%m-%d")} --> {the_dates}', flush=True)
        print(f"Mean RMSE: {np.mean(performance['rmse']):0.1f}")
        print(f"STD RMSE: {np.std(performance['rmse']):0.1f}")
        for p in [5, 95]:
            print(f"{p}th Percentile RMSE: {np.percentile(performance['rmse'], p):0.1f}")
        total_slope = len(performance['slope'])
        total_slope_pos = len([ppp for ppp in performance['slope'] if ppp in ['++','--']])
        total_slope_success = len([ppp for ppp in performance['slope_success'] if ppp])
        # --- Slope Success Summary ---
        print("\n" + "="*50)
        print(f"SLOPE SUCCESS ANALYSIS : {__n_forecast_length_in_training=}   {format_execution_time(time.time() - tt2)}".center(50))
        print("="*50)
        print(f"Total backtest steps                : {total_slope}")
        print(f"Same-direction forecasts (++/--)    : {total_slope_pos} ({100 * total_slope_pos / total_slope:.1f}%)")
        print(f"Successful slope-based outcomes     : {total_slope_success} ({100 * total_slope_success / total_slope:.1f}%)")
        print("Interpretation:")
        print("- 'Slope success' = correct directional alignment OR, when directions differ,")
        print("  price still respected exit thresholds (e.g., didn’t breach stop-loss/take-profit)")
        print("="*50)
        suive_performance.update({__n_forecast_length_in_training: {'++/--': total_slope_pos / total_slope, 'slope-based': total_slope_success / total_slope}})
    # --- Final Performance Summary Across Training Forecast Lengths ---
    print("\n" + "="*60)
    print(f"OVERALL SLOPE PERFORMANCE SUMMARY   {format_execution_time(time.time() - tt1)}".center(60))
    print("="*60)
    print(f"{'Train Len':<10} {'Consistent Dir (%)':<22} {'Slope Success (%)':<22}")
    print("-" * 60)
    for train_len in sorted(suive_performance.keys()):
        cons_dir = suive_performance[train_len]['++/--'] * 100
        slope_success = suive_performance[train_len]['slope-based'] * 100
        print(f"{train_len:<10} {cons_dir:<22.1f} {slope_success:<22.1f}")
    print("="*60)
    print("Notes:")
    print("- 'Consistent Dir': % of forecasts where predicted & actual slopes matched (++, --)")
    print("- 'Slope Success': % of forecasts deemed successful by directional + threshold logic")
    print("="*60)


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Wavelet-based stock backtesting.")

    parser.add_argument('--ticker', type=str, default='^GSPC',
                        help="Yahoo Finance ticker symbol (default: ^GSPC)")
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                        help="Price column to use (default: Close)")
    parser.add_argument('--dataset_id', type=str, default='day',
                        choices=['day', 'week'],
                        help="Dataset frequency: 'day' or 'week' (default: day)")
    parser.add_argument('--n_forecast_length', type=int, default=2,
                        help="Number of future steps to forecast (default: 2)")
    parser.add_argument("--n_forecast_length_in_training", type=int, default=16)
    parser.add_argument("--n_models_to_keep", type=int, default=60)
    parser.add_argument("--threshold_for_shape_similarity", type=float, default=0)
    parser.add_argument('--thresholds_ep', type=str, default="(0.0125, 0.0125)",
                        help="Thresholds for entry/exit as a string tuple (default: '(0.0125, 0.0125)')")
    parser.add_argument('--step_back_range', type=int, default=5,
                        help="Number of past steps to backtest (default: 5)")
    parser.add_argument('--start_of_back_range', type=int, default=1)
    parser.add_argument('--strategy_for_exit', type=str, default="hold_until_the_end_with_roll",
                        choices=['hold_until_the_end', 'hold_until_the_end_with_roll'],
                        help="")
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--use_last_week_only', type=bool, default=False)

    args = parser.parse_args()

    main(args)