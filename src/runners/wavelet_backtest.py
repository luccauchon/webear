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
import copy
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
from utils import format_execution_time, DATASET_AVAILABLE, get_filename_for_dataset
import pandas as pd
import time
import numpy as np


def compute_and_print_stats_for_fomo_strategy(data, using_quantized=False):
    # Define the multiplier labels in the order they appear
    multipliers = [0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    keys_m = [f'last_value_forecasted__m{m:.1f}'.replace('.', '_') for m in multipliers]
    keys_p = [f'last_value_forecasted__p{m:.1f}'.replace('.', '_') for m in multipliers]

    # Collect truth and forecasted values
    truths = []
    forecasts_m = {key: [] for key in keys_m}
    forecasts_p = {key: [] for key in keys_p}

    for step, values in data.items():
        truths.append(values['last_value_truth'])
        for key in keys_m:
            forecasts_m[key].append(values[key])
        for key in keys_p:
            forecasts_p[key].append(values[key])

    truths = np.array(truths)
    tmp1 = f'Truth > Forecast_{"q10" if using_quantized else "m"} (%)'
    tmp2 = f'Truth < Forecast_{"q90" if using_quantized else "m"} (%)'
    print(f"{'Multiplier':<12} {tmp1:<25} {tmp2:<25} {'Truth in interval (%)':<25} {'Count in / Total'}")
    print("-" * 110)

    for m, key_m, key_p in zip(multipliers, keys_m, keys_p):
        forecast_m = np.array(forecasts_m[key_m])
        forecast_p = np.array(forecasts_p[key_p])

        # Truth > lower band (your original metric)
        comparison_gt = truths > forecast_m
        count_gt = np.sum(comparison_gt)
        pct_gt = (count_gt / len(truths)) * 100

        # Truth < upper band (new metric you asked for)
        comparison_lt = truths < forecast_p
        count_lt = np.sum(comparison_lt)
        pct_lt = (count_lt / len(truths)) * 100

        # Truth between lower and upper band (new metric)
        in_band = (truths > forecast_m) & (truths < forecast_p)
        count_in = np.sum(in_band)
        pct_in = (count_in / len(truths)) * 100

        print(f"{m:<12} {pct_gt:<25.2f} {pct_lt:<25.2f} {pct_in:<25.2f} {count_in} / {len(truths)}")


# --- Main Backtesting Function ---
def main(args):
    """
    Execute the main backtesting loop based on the selected strategy.

    Loads historical data, iteratively simulates trading decisions over a rolling window,
    and aggregates performance metrics (success rates, coverage, etc.).

    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # Unpack arguments for clarity
    backtest_strategy = args.backtest_strategy
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
    warrior_spread = args.warrior_spread
    warrior_gt_range_for_success = args.warrior_gt_range_for_success
    assert 0 <= warrior_gt_range_for_success <= 0.2
    warrior_pred_scale_factor = args.warrior_pred_scale_factor
    assert -0.15 <= warrior_pred_scale_factor <= 0.15
    use_last_week_only=args.use_last_week_only
    use_vix = args.use_vix
    one_dataset_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    vix__master_data_cache = master_data_cache['^VIX']
    vix__master_data_cache = copy.deepcopy(vix__master_data_cache.sort_index()[('Close', '^VIX')])
    master_data_cache = copy.deepcopy(master_data_cache[ticker])
    master_data_cache = copy.deepcopy(master_data_cache.sort_index())
    # --- Parameter Summary ---
    if verbose:
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
        print(f"Data File            : {one_dataset_filename}")
        print(f"Exit strategy        : {exit_strategy}")
        print(f"Last week of month   : {use_last_week_only}")
        print(f"Backtest strategy    : {backtest_strategy}")
        if backtest_strategy in ['warrior']:
            print(f"Warrior spread       : {warrior_spread}")
            print(f"Warrior GT range     : {warrior_gt_range_for_success}")
            print(f"Warrior Pred X       : {warrior_pred_scale_factor}")
        print("="*50)
    performance, put_credit_spread_performance, call_credit_spread_performance, iron_condor_performance = {}, {}, {}, {}
    results_for_warrior = {}
    for step_back in range(1, number_of_step_back + 1) if not verbose else tqdm(range(1, number_of_step_back + 1)):
        t1 = time.time()
        if len(master_data_cache) < step_back + n_forecast_length + n_forecast_length_in_training:
            continue
        # Create the "Now" dataframe
        df                                  = copy.deepcopy(master_data_cache.iloc[:-step_back])
        data_cache_for_parameter_extraction = copy.deepcopy(df.iloc[:-n_forecast_length])
        data_cache_for_forecasting          = copy.deepcopy(df.iloc[-n_forecast_length:])
        assert n_forecast_length == len(data_cache_for_forecasting)
        assert data_cache_for_parameter_extraction.index.intersection(data_cache_for_forecasting.index).empty
        df_vix                              = copy.deepcopy(vix__master_data_cache.iloc[:-step_back])
        vix_for_prediction                  = copy.deepcopy(df_vix[:-n_forecast_length])
        vix_future                          = copy.deepcopy(df_vix[-n_forecast_length:])
        if use_vix:
            assert n_forecast_length > 0
            if len(vix_for_prediction) < 1:
                continue
            assert n_forecast_length == len(vix_future)
            assert vix_future.index.intersection(vix_for_prediction.index).empty, "Indices must be disjoint"
            assert len(vix_future) == len(data_cache_for_forecasting)
            # print(f'[{step_back}] VIX {vix_for_prediction.index[0].strftime("%Y-%m-%d")}:{vix_for_prediction.index[-1].strftime("%Y-%m-%d")}')
            assert vix_future.index[0]  == data_cache_for_forecasting.index[0]
            assert vix_future.index[-1] == data_cache_for_forecasting.index[-1]
        if use_last_week_only:
            assert 'week' == dataset_id
            last_date = df.index[-1]
            month_end = last_date + pd.offsets.MonthEnd(0)
            #print("Month end:", month_end)

            # Check if last_date is in the last week of the month
            last_week_start = month_end - pd.Timedelta(days=5)
            is_in_last_week = last_date >= last_week_start
            #print("Is in last week of month:", is_in_last_week)
            if not is_in_last_week:
                continue
        # print(f'[{step_back}] {col_name} {data_cache_for_parameter_extraction.index[0].strftime("%Y-%m-%d")}:{data_cache_for_parameter_extraction.index[-1].strftime("%Y-%m-%d")} --> {data_cache_for_forecasting.index}')
        assert n_forecast_length == len(data_cache_for_forecasting), f"{len(data_cache_for_forecasting)}"
        assert data_cache_for_parameter_extraction.index.intersection(data_cache_for_forecasting.index).empty, "Indices must be disjoint"
        output_dir = rf"../../stubs/wavelet_backtesting_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}/__{step_back}/"
        os.makedirs(output_dir, exist_ok=True)
        the_ground_truth = data_cache_for_forecasting[col_name].values
        assert len(the_ground_truth) == n_forecast_length
        configuration = Namespace(
            master_data_cache=copy.deepcopy(data_cache_for_parameter_extraction),
            ticker=ticker, col=col,
            output_dir=output_dir,
            dataset_id=dataset_id,
            n_forecast_length=n_forecast_length,
            n_forecast_length_in_training=n_forecast_length_in_training,
            thresholds_ep=thresholds_ep,
            plot_graph=False,
            use_given_gt_truth=the_ground_truth,
            display_tqdm=False,
            strategy_for_exit=exit_strategy,
            n_models_to_keep=n_models_to_keep,
            q_min_filter=3,
            q_max_filter=97,
            threshold_for_shape_similarity=threshold_for_shape_similarity,
            verbose=verbose,
        )

        # Run wavelet forecasting and get trading instruction
        user_instruction, misc_returned = wavelet_realtime_entry_point(configuration)

        # --- Strategy-Specific Evaluation ---
        if backtest_strategy == 'stratego':
            # Parse operation from wavelet model
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
            if operation_request == 'iron_condor':
                iron_condor_performance.update({step_back: {}})
                iron_condor_performance[step_back].update({'data_cache_for_forecasting': data_cache_for_forecasting,
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
            if verbose:
                print(f"[{step_back}/{number_of_step_back}] used {format_execution_time(t2-t1)}")
                if operation_aborted:
                    print(f"\tOn the day {data_cache_for_forecasting.index[0].strftime('%Y-%m-%d')} , no operation was taken")
                else:
                    if operation_success:
                        print(f"\tOn the day {data_cache_for_forecasting.index[0].strftime('%Y-%m-%d')} , the {operation_request} was successful")
                    else:
                        print(f"\tOn the day {data_cache_for_forecasting.index[0].strftime('%Y-%m-%d')} , the {operation_request} was failed by {operation_missed_threshold:0.1f}")
                print(f"\t\t{user_instruction['description']}")
        if backtest_strategy in ['fomo', 'fomo_quantilized']:
            mean_forecast, q10_forecast, q90_forecast = misc_returned['mean_forecast'], misc_returned['q10_forecast'], misc_returned['q90_forecast']
            assert len(mean_forecast) == n_forecast_length and len(mean_forecast) == len(the_ground_truth)
            assert len(q10_forecast) == n_forecast_length and len(mean_forecast) == len(the_ground_truth)
            assert len(q90_forecast) == n_forecast_length and len(mean_forecast) == len(the_ground_truth)
            last_value_forecasted_upper_side = mean_forecast[-1]
            last_value_forecasted_lower_side = mean_forecast[-1]
            last_value_truth                 = the_ground_truth[-1]
            if backtest_strategy == 'fomo_quantilized':
                last_value_forecasted_upper_side = q90_forecast[-1]
                last_value_forecasted_lower_side = q10_forecast[-1]
            performance.update({step_back: {'last_value_truth': last_value_truth,
                                            'last_value_forecasted__p0_0': last_value_forecasted_upper_side,
                                            'last_value_forecasted__p0_5': last_value_forecasted_upper_side * 1.005,
                                            'last_value_forecasted__p1_0': last_value_forecasted_upper_side * 1.010,
                                            'last_value_forecasted__p1_5': last_value_forecasted_upper_side * 1.015,
                                            'last_value_forecasted__p2_0': last_value_forecasted_upper_side * 1.020,
                                            'last_value_forecasted__p2_5': last_value_forecasted_upper_side * 1.025,
                                            'last_value_forecasted__p3_0': last_value_forecasted_upper_side * 1.030,
                                            'last_value_forecasted__p3_5': last_value_forecasted_upper_side * 1.035,
                                            'last_value_forecasted__p4_0': last_value_forecasted_upper_side * 1.040,
                                            'last_value_forecasted__p5_0': last_value_forecasted_upper_side * 1.050,
                                            'last_value_forecasted__m0_0': last_value_forecasted_lower_side,
                                            'last_value_forecasted__m0_5': last_value_forecasted_lower_side*0.995,
                                            'last_value_forecasted__m1_0': last_value_forecasted_lower_side*0.990,
                                            'last_value_forecasted__m1_5': last_value_forecasted_lower_side*0.985,
                                            'last_value_forecasted__m2_0': last_value_forecasted_lower_side*0.980,
                                            'last_value_forecasted__m2_5': last_value_forecasted_lower_side*0.975,
                                            'last_value_forecasted__m3_0': last_value_forecasted_lower_side*0.970,
                                            'last_value_forecasted__m3_5': last_value_forecasted_lower_side*0.965,
                                            'last_value_forecasted__m4_0': last_value_forecasted_lower_side*0.960,
                                            'last_value_forecasted__m5_0': last_value_forecasted_lower_side*0.950,},})
        if backtest_strategy in ['warrior']:
            mean_forecast, q10_forecast, q90_forecast = misc_returned['mean_forecast'], misc_returned['q10_forecast'], misc_returned['q90_forecast']
            assert len(mean_forecast) == n_forecast_length and len(mean_forecast) == len(the_ground_truth)
            assert len(q10_forecast) == n_forecast_length and len(mean_forecast) == len(the_ground_truth)
            assert len(q90_forecast) == n_forecast_length and len(mean_forecast) == len(the_ground_truth)
            win = False
            pred     = mean_forecast[-1] * (1 + warrior_pred_scale_factor)
            gt       = the_ground_truth[-1]
            gt_lower = gt * (1 - warrior_gt_range_for_success)
            gt_upper = gt * (1 + warrior_gt_range_for_success)
            if warrior_spread == 'call':  # Call Credit Spread
                assert warrior_pred_scale_factor >= 0
                if gt < pred <= gt_upper:
                    win = True
            if warrior_spread == 'put':  # Put Credit Spread
                assert warrior_pred_scale_factor <= 0
                if  gt_lower <= pred < gt:
                    win = True
            results_for_warrior.update({step_back: {'status': win}})
            if use_vix:
                vix_value = vix_for_prediction.iloc[-1]
                results_for_warrior[step_back].update({'vix': vix_value})

    # --- Summary Report ---
    if backtest_strategy == 'stratego':
        total_runs = len(performance)
        successes = sum(1 for v in performance.values() if v['operation_success'])
        failures  = sum(1 for v in performance.values() if not v['operation_success'] and not v['operation_aborted'])
        skipped   = sum(1 for v in performance.values() if v['operation_aborted'])

        print("\n" + "="*50)
        print("BACKTESTING PERFORMANCE SUMMARY".center(50))
        print("="*50)
        if total_runs > 0:
            print(f"Total Backtest Windows     : {total_runs}")
            print(f"Successful Trades          : {successes} ({successes/(successes+failures)*100:.1f}%)")
            print(f"Failed Trades              : {failures} ({failures/(successes+failures)*100:.1f}%)")
            print(f"Skipped / No Action        : {skipped} ({skipped/total_runs*100:.1f}%)")
        print("="*50)

        # --- Iron Condor Summary ---
        iron_condor_runs = len(iron_condor_performance)
        if iron_condor_runs > 0:
            iron_condor_successes = sum(1 for v in iron_condor_performance.values() if v['operation_success'])
            iron_condor_failures = iron_condor_runs - iron_condor_successes
            print(f"\nIron Condor ({iron_condor_runs} trades):")
            print(f"  Successes: {iron_condor_successes} ({iron_condor_successes / iron_condor_runs * 100:.1f}%)")
            print(f"  Failures : {iron_condor_failures} ({iron_condor_failures / iron_condor_runs * 100:.1f}%)")

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
    if backtest_strategy in ['fomo', 'fomo_quantilized']:
        compute_and_print_stats_for_fomo_strategy(performance, using_quantized=backtest_strategy=='fomo_quantilized')
    if backtest_strategy in ['warrior']:
        nb_success, nb_total = len([k for k, v in results_for_warrior.items() if v['status']]), len(results_for_warrior)
        success_rate = round(nb_success / nb_total * 100, 2) if nb_total > 0 else 0
        if verbose:
            print(f"\nWarrior Strategy Success Rate: {success_rate}% ({nb_success}/{nb_total})")
            if use_vix:
                vix_threshold=30.1
                nb_success, nb_total = len([k for k, v in results_for_warrior.items() if v['status'] and v['vix'] < vix_threshold]), len([k for k, v in results_for_warrior.items() if v['vix'] < vix_threshold])
                success_rate = round(nb_success / nb_total * 100, 2) if nb_total > 0 else 0
                print(f"\n[VIX < {vix_threshold}] Warrior Strategy Success Rate: {success_rate}% ({nb_success}/{nb_total})")
        return {'success_rate': success_rate}

    return None


# --- CLI Entry Point ---
if __name__ == "__main__":
    freeze_support()  # Required for multiprocessing on Windows

    parser = argparse.ArgumentParser(description="Run Wavelet-based Stock Backtesting.")

    parser.add_argument('--backtest_strategy', type=str, default='stratego',
                        choices=['stratego', 'fomo', 'fomo_quantilized', 'warrior'],
                        help="Strategy to backtest: 'stratego' (options trading), "
                             "'fomo' (point forecast), 'fomo_quantilized' (quantile-based), "
                             "or 'warrior' (tolerance-band trading).")

    parser.add_argument('--ticker', type=str, default='^GSPC',
                        help="Yahoo Finance ticker symbol (e.g., '^GSPC' for S&P 500).")

    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                        help="Price/volume column to forecast.")

    parser.add_argument('--dataset_id', type=str, default='day',
                        choices=DATASET_AVAILABLE,
                        help="Data frequency: e.g., 'day' for daily, 'week' for weekly.")

    parser.add_argument('--n_forecast_length', type=int, default=2,
                        help="Number of future time steps to predict (e.g., 2 days ahead).")

    parser.add_argument("--n_forecast_length_in_training", type=int, default=4,
                        help="Number of recent steps used during model training.")

    parser.add_argument("--n_models_to_keep", type=int, default=60,
                        help="Maximum number of wavelet models to retain during optimization.")

    parser.add_argument("--threshold_for_shape_similarity", type=float, default=0.0,
                        help="Minimum similarity threshold to select historical patterns (0 = no filter).")

    parser.add_argument('--thresholds_ep', type=str, default="(0.0125, 0.0125)",
                        help="Entry/exit sensitivity thresholds as a tuple string, e.g., '(0.01, 0.02)'.")

    parser.add_argument('--step-back-range', type=int, default=5,
                        help="Number of historical time windows to simulate (rolling backtest depth).")

    parser.add_argument('--strategy_for_exit', type=str, default="hold_until_the_end_with_roll",
                        choices=['hold_until_the_end', 'hold_until_the_end_with_roll'],
                        help="How to manage position exit: with or without re-rolling.")

    parser.add_argument('--verbose', action='store_true',
                        help="Enable detailed per-step logging (disables progress bar).")

    parser.add_argument('--warrior_spread', type=str, default='call', choices=['call', 'put'],
                        help="For 'warrior' strategy: type of credit spread to simulate.")

    parser.add_argument('--warrior_gt_range_for_success', type=float, default=0.04,
                        help="Tolerance band around ground truth (determine the upper/lower bound).")

    parser.add_argument('--warrior_pred_scale_factor', type=float, default=0.,
                        help="Multiply the prediction with this factor.")

    parser.add_argument('--use_last_week_only', action='store_true',
                        help="Only run backtests during the last week of each month (requires weekly data).")

    parser.add_argument('--use_vix', action='store_true',
                        help="Save the vix value per run in backtest")

    args = parser.parse_args()
    main(args)