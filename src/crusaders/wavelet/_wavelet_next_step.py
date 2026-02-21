try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
from argparse import Namespace
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool, next_week, next_day, get_next_week_range, get_next_month_range, next_month, next_weekday, get_next_day_range
import copy
import numpy as np
import os
from datetime import datetime, timedelta
from runners.wavelet_realtime import main as wavelet_realtime_entry_point
import pickle


def main(args):
    now = datetime.now()
    if args.step_type == "week":
        step__name = "week"
        step__next_step_fct = next_week
        step__next_step_range_fct = get_next_week_range
    elif args.step_type == "month":
        step__name = "month"
        step__next_step_fct = next_month
        step__next_step_range_fct = get_next_month_range
    elif args.step_type == "day":
        step__name = "day"
        step__next_step_fct = next_weekday
        step__next_step_range_fct = get_next_day_range
    else:
        assert False

    output_dir = rf"../../../stubs/wavelet_next_{step__name}_at_Xp_{now.strftime('%Y_%m_%d__%H_%M_%S')}/"
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    one_dataset_filename = get_filename_for_dataset(step__name, older_dataset=None if args.older_dataset == "None" else args.older_dataset)
    if args.verbose:
        print(f"Using data from {one_dataset_filename}")
        # --- Nicely print the arguments ---
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)
        print("\n" + "=" * 80)
        if args.put_side and args.call_side and args.percentage is not None:
            assert len(args.percentage) == len(args.lower_performance) == len(args.upper_performance)
            for ppp, lower_performance, upper_performance in zip(args.percentage, args.lower_performance, args.upper_performance):
                print(f"[@{ppp}%] Historical performance of {lower_performance * 100:.2f}% (lower:GroundTruth>Forecast) and {upper_performance * 100:0.2f}% (upper:GroundTruth<Forecast)")
        print("=" * 80)
    col_name = ("Close", "^GSPC")
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    vix__master_data_cache = master_data_cache['^VIX']
    vix__master_data_cache = copy.deepcopy(vix__master_data_cache.sort_index()[('Close', '^VIX')])
    master_data_cache = copy.deepcopy(master_data_cache[args.ticker])
    master_data_cache = master_data_cache.sort_index()
    if not args.keep_last_step:
        master_data_cache      = master_data_cache.iloc[:-1]
        vix__master_data_cache = vix__master_data_cache.iloc[:-1]

    # Lookup the VIX value
    vix_close_value_in_last_day = vix__master_data_cache.iloc[-1]
    if master_data_cache[col_name].index[-1].strftime('%Y_%m_%d') != vix__master_data_cache.index[-1].strftime('%Y_%m_%d'):
        vix_ok = False
        for uu in range(2, 13):
            vix_close_value_in_last_day = vix__master_data_cache.iloc[-uu]
            if master_data_cache[col_name].index[-1].strftime('%Y_%m_%d') == vix__master_data_cache.index[-uu].strftime('%Y_%m_%d'):
                vix_ok = True
                break
        assert vix_ok

    _next_step        = step__next_step_fct(master_data_cache[col_name].index[-1])
    _str, __next_step = step__next_step_range_fct(master_data_cache[col_name].index[-1])
    if args.n_forecast_length > 1:
        for uu in range(1, args.n_forecast_length):
            _tmp = _next_step
            _next_step = step__next_step_fct(_tmp)
            _str, __next_step = step__next_step_range_fct(_tmp)
    assert _next_step.strftime('%Y-%m-%d') == __next_step.strftime('%Y-%m-%d')
    print(f"{_str} , actual price of {col_name} is {master_data_cache[col_name].iloc[-1]:.0f} ({master_data_cache[col_name].index[-1].strftime('%Y-%m-%d')})")
    configuration = Namespace(
        master_data_cache=copy.deepcopy(master_data_cache),
        ticker=args.ticker, col=args.col,
        output_dir=output_dir,
        dataset_id=step__name,
        n_forecast_length=args.n_forecast_length,
        n_forecast_length_in_training=int(args.n_forecast_length_in_training),
        thresholds_ep="(0.0125,0.0125)",
        plot_graph=False,
        use_given_gt_truth=None,
        display_tqdm=False,
        strategy_for_exit=None,
        n_models_to_keep=int(args.n_models_to_keep),
        q_min_filter=3,
        q_max_filter=97,
        threshold_for_shape_similarity=0,
        verbose=False,
    )
    # Run wavelet forecasting and get trading instruction
    user_instruction, misc_returned = wavelet_realtime_entry_point(configuration)
    mean_forecast, q10_forecast, q90_forecast = misc_returned['mean_forecast'], misc_returned['q10_forecast'], misc_returned['q90_forecast']
    assert len(mean_forecast) == args.n_forecast_length and len(mean_forecast)
    assert len(q10_forecast) == args.n_forecast_length and len(mean_forecast)
    assert len(q90_forecast) == args.n_forecast_length and len(mean_forecast)
    if args.which_output_to_use == "last_of_mean_forecast":
        last_value_forecasted_upper_side = mean_forecast[-1]
        last_value_forecasted_lower_side = mean_forecast[-1]
    else:
        assert False
    if (args.put_side and args.call_side) and args.percentage is not None:
        assert len(args.percentage) == len(args.lower_multiplier) == len(args.upper_multiplier)
        for ppp, lower_multiplier, upper_multiplier, lower_performance, upper_performance in zip(args.percentage, args.lower_multiplier, args.upper_multiplier, args.lower_performance,args.upper_performance):
            last_value_forecasted__px_0 = last_value_forecasted_upper_side * float(upper_multiplier)
            last_value_forecasted__mx_0 = last_value_forecasted_lower_side * float(lower_multiplier)
            print(f"[@{ppp}%] For {step__name} ending {_next_step.strftime('%Y_%m_%d')} , lower value shall be {last_value_forecasted__mx_0:.0f} ({lower_performance * 100:.2f}%)  "
                  f"and upper value {last_value_forecasted__px_0:.0f} ({upper_performance * 100:.2f}%)")
    if (args.put_side or args.call_side) and args.vix_modulation is not None:
        assert args.put_side or args.call_side
        multiplier = args.upper_multiplier
        if args.put_side:
            multiplier = args.lower_multiplier
        assert multiplier is not None
        performance_modulated_by_vix = args.vix_modulation[min([k for k in args.vix_modulation if vix_close_value_in_last_day <= k])]
        assert last_value_forecasted_upper_side == last_value_forecasted_lower_side
        for m in multiplier:
            last_value_forecasted = last_value_forecasted_lower_side * float(m)
            str1 = "lower" if args.put_side else "upper"
            print(f"For {step__name} ending {_next_step.strftime('%Y_%m_%d')} , {str1} value shall be {last_value_forecasted:.0f} ({performance_modulated_by_vix * 100:.2f}%) , VIX={vix_close_value_in_last_day:.1f} ")
