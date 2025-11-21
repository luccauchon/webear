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


def main(args):
    ticker = '^GSPC'
    col = 'Close'
    col_name = (col, ticker)
    dataset_id = 'day'
    n_forecast_length   = 1
    thresholds_ep = "(0.0125, 0.0125)"

    if dataset_id == 'day':
        df_filename = FYAHOO__OUTPUTFILENAME_DAY
    elif dataset_id == 'week':
        df_filename = FYAHOO__OUTPUTFILENAME_WEEK
    with open(df_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = master_data_cache[ticker].copy()
    performance = {}
    number_of_step_back=999
    for step_back in range(1, number_of_step_back + 1):
        # All data except the last `step_back` rows → for parameter extraction
        data_cache_for_parameter_extraction = master_data_cache.iloc[:-step_back].copy()
        performance.update({step_back: {}})
        # The single row at position `-step_back` → for forecasting
        data_cache_for_forecasting = master_data_cache.iloc[[-step_back]].copy()
        assert n_forecast_length == len(data_cache_for_forecasting)
        output_dir = rf"../../stubs/wavelet_backtesting_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}/__{step_back}/"
        os.makedirs(output_dir, exist_ok=True)
        args = Namespace(
            master_data_cache=data_cache_for_parameter_extraction.copy(),
            ticker=ticker, col=col,
            output_dir=output_dir,
            dataset_id=dataset_id,
            n_forecast_length=n_forecast_length,
            thresholds_ep=thresholds_ep,
            plot_graph=False,
            use_given_gt_truth=data_cache_for_forecasting[col_name].values,
            display_tqdm=False,
        )
        user_instruction, misc_returned = wavelet_realtime_entry_point(args)
        operation_data = user_instruction['op']
        operation_request, operation_success, operation_missed_threshold = operation_data['action'], False, 0
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
                assert False
        if operation_request == 'vertical_put':
            if 1 == n_forecast_length:
                assert 1 == len(data_cache_for_forecasting[col_name].values)
                real_value = data_cache_for_forecasting[col_name].values[0]
                low = operation_data['sell1']
                if real_value > low:
                    operation_success = True
            else:
                assert False
        if operation_request == 'vertical_call':
            if 1 == n_forecast_length:
                assert 1 == len(data_cache_for_forecasting[col_name].values)
                real_value = data_cache_for_forecasting[col_name].values[0]
                high = operation_data['sell1']
                if real_value < high:
                    operation_success = True
            else:
                assert False
        performance[step_back].update({})
        if operation_success:
            print(f"\tOn the day {data_cache_for_forecasting.index[0]} , the {operation_request} was successful")
        else:
            print(f"\tOn the day {data_cache_for_forecasting.index[0]} , the {operation_request} was failed by {operation_missed_threshold:0.1f}")


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Wavelet-based stock real time estimator.")
    args = parser.parse_args()

    main(args)