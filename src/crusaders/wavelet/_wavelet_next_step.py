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
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool, next_week, next_day, get_next_week_range, get_next_month_range, next_month
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
        assert len(args.percentage) == len(args.lower_performance) == len(args.upper_performance)
        for ppp, lower_performance, upper_performance in zip(args.percentage, args.lower_performance, args.upper_performance):
            print(f"[@{ppp}%] Historical performance of {lower_performance * 100:.2f}% (lower:GroundTruth>Forecast) and {upper_performance * 100:0.2f}% (upper:GroundTruth<Forecast)")
        print("=" * 80)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = copy.deepcopy(master_data_cache[args.ticker])
    master_data_cache = master_data_cache.sort_index()
    if not args.keep_last_step:
        master_data_cache = master_data_cache.iloc[:-1]
    _next_step = step__next_step_fct(master_data_cache[("Close", "^GSPC")].index[-1])
    _str, __next_step = step__next_step_range_fct(master_data_cache[("Close", "^GSPC")].index[-1])
    assert _next_step == __next_step
    print(_str)
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
    last_value_forecasted_upper_side = mean_forecast[-1]
    last_value_forecasted_lower_side = mean_forecast[-1]
    assert len(args.percentage) == len(args.lower_multiplier) == len(args.upper_multiplier)
    for ppp, lower_multiplier, upper_multiplier, lower_performance, upper_performance in zip(args.percentage, args.lower_multiplier, args.upper_multiplier, args.lower_performance,args.upper_performance):
        last_value_forecasted__px_0 = last_value_forecasted_upper_side * float(upper_multiplier)
        last_value_forecasted__mx_0 = last_value_forecasted_lower_side * float(lower_multiplier)
        print(f"[@{ppp}%] For {step__name} ending {_next_step.strftime('%Y_%m_%d')} , lower value shall be {last_value_forecasted__mx_0:.0f} ({lower_performance * 100:.2f}%)  "
              f"and upper value {last_value_forecasted__px_0:.0f} ({upper_performance * 100:.2f}%)")