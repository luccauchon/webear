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
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool, next_month, next_week, next_day
import copy
import numpy as np
import os
from datetime import datetime, timedelta
from runners.wavelet_realtime import main as wavelet_realtime_entry_point
import pickle


def main(args):
    lower_performance, upper_performance = 0.8350, 0.7233
    if args.verbose:
        print("\n" + "=" * 80)
        print(f"Historical performance of {lower_performance*100:.2f}% (lower) and {upper_performance*100:0.2f}% (upper)")
        print("=" * 80)
    now = datetime.now()
    output_dir = rf"../../../stubs/wavelet_next_month_at_3p_{now.strftime('%Y_%m_%d__%H_%M_%S')}/"
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    one_dataset_filename = get_filename_for_dataset("month", older_dataset=None if args.older_dataset == "None" else args.older_dataset)
    print(f"Using data from {one_dataset_filename}")
    # --- Nicely print the arguments ---
    print("🔧 Arguments:")
    for arg, value in vars(args).items():
        print(f"    {arg:.<40} {value}")
    print("-" * 80, flush=True)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = copy.deepcopy(master_data_cache[args.ticker])
    master_data_cache = master_data_cache.sort_index()
    if not args.keep_last_step:
        master_data_cache = master_data_cache.iloc[:-1]
    n_forecast_length = 1
    _next_month = next_month(master_data_cache[("Close", "^GSPC")].index[-1])
    print(f"Starting prediction for month ending {_next_month.strftime('%Y_%m_%d')} ...")
    configuration = Namespace(
        master_data_cache=copy.deepcopy(master_data_cache),
        ticker="^GSPC", col="Close",
        output_dir=output_dir,
        dataset_id="month",
        n_forecast_length=n_forecast_length,
        n_forecast_length_in_training=6,
        thresholds_ep="(0.0125,0.0125)",
        plot_graph=False,
        use_given_gt_truth=None,
        display_tqdm=False,
        strategy_for_exit=None,
        n_models_to_keep=330,
        q_min_filter=3,
        q_max_filter=97,
        threshold_for_shape_similarity=0,
        verbose=False,
    )
    # Run wavelet forecasting and get trading instruction
    user_instruction, misc_returned = wavelet_realtime_entry_point(configuration)
    mean_forecast, q10_forecast, q90_forecast = misc_returned['mean_forecast'], misc_returned['q10_forecast'], misc_returned['q90_forecast']
    assert len(mean_forecast) == n_forecast_length and len(mean_forecast)
    assert len(q10_forecast) == n_forecast_length and len(mean_forecast)
    assert len(q90_forecast) == n_forecast_length and len(mean_forecast)
    last_value_forecasted_upper_side = mean_forecast[-1]
    last_value_forecasted_lower_side = mean_forecast[-1]
    last_value_forecasted__p3_0 = last_value_forecasted_upper_side * 1.030
    last_value_forecasted__m3_0 = last_value_forecasted_lower_side * 0.970
    print(f"For month ending {_next_month.strftime('%Y_%m_%d')} , lower value shall be {last_value_forecasted__m3_0} ({lower_performance*100:.2f}%)  "
          f"and upper value {last_value_forecasted__p3_0} ({upper_performance*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument('--keep_last_step', type=str2bool, default=True)
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)