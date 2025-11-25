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
from utils import format_execution_time
import pandas as pd
import time


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
    exit_strategy = args.strategy_for_exit
    threshold_for_shape_similarity = args.threshold_for_shape_similarity
    verbose = args.verbose
    use_last_week_only=args.use_last_week_only
    if dataset_id == 'day':
        df_filename = FYAHOO__OUTPUTFILENAME_DAY
    elif dataset_id == 'week':
        df_filename = FYAHOO__OUTPUTFILENAME_WEEK
    with open(df_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = master_data_cache[ticker].copy()
    # --- Parameter Summary ---
    print("\n" + "="*50)
    print("BACKTESTING PARAMETERS SUMMARY".center(50))
    print("="*50)
    print(f"Ticker               : {ticker}")
    print(f"Column               : {col}")
    print(f"Dataset Frequency    : {dataset_id}")
    print(f"Forecast Length      : {n_forecast_length}")
    print(f"Forecast Length Train: {n_forecast_length_in_training}")
    print(f"N models to keep     : {n_models_to_keep}")
    print(f"T Shape Similarity   : {threshold_for_shape_similarity}")
    print(f"Thresholds (EP)      : {thresholds_ep}")
    print(f"Step-Back Range      : {number_of_step_back}")
    print(f"Data File            : {df_filename}")
    print(f"Exit strategy        : {exit_strategy}")
    print(f"Last week of month   : {use_last_week_only}")
    print("="*50)
    # input("Press Enter to start backtesting...")
    performance, put_credit_spread_performance, call_credit_spread_performance = {}, {}, {}
    for step_back in range(1, number_of_step_back + 1):
        t1 = time.time()
        # Create the "Now" dataframe
        df = master_data_cache.iloc[:-step_back].copy()
        #print(f'{df.index[0].strftime("%Y-%m-%d")}:{df.index[-1].strftime("%Y-%m-%d")}')
        if use_last_week_only:
            assert 'week' == dataset_id
            last_date = df.index[-1]
            month_end = last_date + pd.offsets.MonthEnd(0)
            #print("Month end:", month_end)

            # Check if last_date is in the last week of the month
            last_week_start = month_end - pd.Timedelta(days=6)
            is_in_last_week = last_date >= last_week_start
            #print("Is in last week of month:", is_in_last_week)
            if not is_in_last_week:
                continue
        # All data except the last `step_back` rows → for parameter extraction
        data_cache_for_parameter_extraction = df.iloc[:-n_forecast_length].copy()
        # Rows at position `-step_back` → for forecasting
        data_cache_for_forecasting = df.iloc[-n_forecast_length: ].copy()
        # print(f'{data_cache_for_parameter_extraction.index[0].strftime("%Y-%m-%d")}:{data_cache_for_parameter_extraction.index[-1].strftime("%Y-%m-%d")} --> {data_cache_for_forecasting.index}')
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
            n_forecast_length_in_training=n_forecast_length_in_training,
            thresholds_ep=thresholds_ep,
            plot_graph=False,
            use_given_gt_truth=data_cache_for_forecasting[col_name].values,
            display_tqdm=False,
            strategy_for_exit=args.strategy_for_exit,
            n_models_to_keep=n_models_to_keep,
            threshold_for_shape_similarity=threshold_for_shape_similarity,
            verbose=verbose,
        )
        user_instruction, misc_returned = wavelet_realtime_entry_point(args)
        operation_data = user_instruction['op']
        operation_request, operation_success, operation_missed_threshold = operation_data['action'], False, 0
        operation_aborted = False
        if operation_request == 'iron_condor':
            if 1 == n_forecast_length:
                assert 1 == len(data_cache_for_forecasting[col_name].values)
                real_value = data_cache_for_forecasting[col_name].values[0]
                low, high = operation_data['sell1'], operation_data['sell2']
                if low < real_value < high:
                    operation_success = True
                else:
                    operation_missed_threshold = low - real_value if real_value < low else real_value - high
            else:
                assert n_forecast_length == len(data_cache_for_forecasting[col_name].values)
                real_values = data_cache_for_forecasting[col_name].values
                last_real_value = real_values[-1]
                low, high = operation_data['sell1'], operation_data['sell2']
                if low < last_real_value < high:
                    operation_success = True
                else:
                    operation_missed_threshold = low - last_real_value if last_real_value < low else last_real_value - high
        if operation_request == 'vertical_put':
            if 1 == n_forecast_length:
                assert 1 == len(data_cache_for_forecasting[col_name].values)
                real_value = data_cache_for_forecasting[col_name].values[0]
                low = operation_data['sell1']
                if real_value > low:
                    operation_success = True
                else:
                    operation_missed_threshold = low - real_value
            else:
                assert n_forecast_length == len(data_cache_for_forecasting[col_name].values)
                real_values = data_cache_for_forecasting[col_name].values
                last_real_value = real_values[-1]
                low = operation_data['sell1']
                if last_real_value > low:
                    operation_success = True
                else:
                    operation_missed_threshold = low - last_real_value
        if operation_request == 'vertical_call':
            if 1 == n_forecast_length:
                assert 1 == len(data_cache_for_forecasting[col_name].values)
                real_value = data_cache_for_forecasting[col_name].values[0]
                high = operation_data['sell1']
                if real_value < high:
                    operation_success = True
                else:
                    operation_missed_threshold = real_value - high
            else:
                assert n_forecast_length == len(data_cache_for_forecasting[col_name].values)
                real_values = data_cache_for_forecasting[col_name].values
                last_real_value = real_values[-1]
                high = operation_data['sell1']
                if last_real_value < high:
                    operation_success = True
                else:
                    operation_missed_threshold = last_real_value - high
        if operation_request == 'do_nothing':
            operation_aborted = True
        performance.update({step_back: {}})
        performance[step_back].update({'data_cache_for_forecasting': data_cache_for_forecasting,
                                       'operation_request': operation_request,
                                       'operation_success': operation_success,
                                       'operation_aborted': operation_aborted, 'operation_missed_threshold': operation_missed_threshold})
        if operation_request == 'vertical_put':
            put_credit_spread_performance.update({step_back: {}})
            put_credit_spread_performance[step_back].update({'data_cache_for_forecasting': data_cache_for_forecasting,
                                                             'operation_request': operation_request,
                                                             'operation_success': operation_success,
                                                             'operation_aborted': operation_aborted, 'operation_missed_threshold': operation_missed_threshold})
        if operation_request == 'vertical_call':
            call_credit_spread_performance.update({step_back: {}})
            call_credit_spread_performance[step_back].update({'data_cache_for_forecasting': data_cache_for_forecasting,
                                                              'operation_request': operation_request,
                                                              'operation_success': operation_success,
                                                              'operation_aborted': operation_aborted, 'operation_missed_threshold': operation_missed_threshold})
        t2 = time.time()
        print(f"[{step_back}/{number_of_step_back}] used {format_execution_time(t2-t1)}")
        if operation_aborted:
            print(f"\tOn the day {data_cache_for_forecasting.index[0].strftime('%Y-%m-%d')} , no operation was taken")
        else:
            if operation_success:
                print(f"\tOn the day {data_cache_for_forecasting.index[0].strftime('%Y-%m-%d')} , the {operation_request} was successful")
            else:
                print(f"\tOn the day {data_cache_for_forecasting.index[0].strftime('%Y-%m-%d')} , the {operation_request} was failed by {operation_missed_threshold:0.1f}")
        print(f"\t\t{user_instruction['description']}")
    # --- Summary Report ---
    total_runs = len(performance)
    successes = sum(1 for v in performance.values() if v['operation_success'])
    failures  = sum(1 for v in performance.values() if not v['operation_success'] and not v['operation_aborted'])
    skipped   = sum(1 for v in performance.values() if v['operation_aborted'])

    print("\n" + "="*50)
    print("BACKTESTING PERFORMANCE SUMMARY".center(50))
    print("="*50)
    print(f"Total Backtest Windows     : {total_runs}")
    print(f"Successful Trades          : {successes} ({successes/(successes+failures)*100:.1f}%)")
    print(f"Failed Trades              : {failures} ({failures/(successes+failures)*100:.1f}%)")
    print(f"Skipped / No Action        : {skipped} ({skipped/total_runs*100:.1f}%)")
    print("="*50)

    # --- Put Credit Spread Summary ---
    put_runs = len(put_credit_spread_performance)
    if put_runs > 0:
        put_successes = sum(1 for v in put_credit_spread_performance.values() if v['operation_success'])
        put_failures = put_runs - put_successes
        print(f"\nPut Credit Spreads ({put_runs} trades):")
        print(f"  Successes: {put_successes} ({put_successes/put_runs*100:.1f}%)")
        print(f"  Failures : {put_failures} ({put_failures/put_runs*100:.1f}%)")

    # --- Call Credit Spread Summary ---
    call_runs = len(call_credit_spread_performance)
    if call_runs > 0:
        call_successes = sum(1 for v in call_credit_spread_performance.values() if v['operation_success'])
        call_failures = call_runs - call_successes
        print(f"\nCall Credit Spreads ({call_runs} trades):")
        print(f"  Successes: {call_successes} ({call_successes/call_runs*100:.1f}%)")
        print(f"  Failures : {call_failures} ({call_failures/call_runs*100:.1f}%)")


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
    parser.add_argument("--n_forecast_length_in_training", type=int, default=4)
    parser.add_argument("--n_models_to_keep", type=int, default=60)
    parser.add_argument("--threshold_for_shape_similarity", type=float, default=0)
    parser.add_argument('--thresholds_ep', type=str, default="(0.0125, 0.0125)",
                        help="Thresholds for entry/exit as a string tuple (default: '(0.0125, 0.0125)')")
    parser.add_argument('--step-back-range', type=int, default=5,
                        help="Number of past steps to backtest (default: 5)")
    parser.add_argument('--strategy_for_exit', type=str, default="hold_until_the_end_with_roll",
                        choices=['hold_until_the_end', 'hold_until_the_end_with_roll'],
                        help="")
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--use_last_week_only', type=bool, default=False)

    args = parser.parse_args()

    main(args)