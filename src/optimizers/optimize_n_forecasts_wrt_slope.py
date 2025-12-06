from argparse import Namespace

from utils import format_execution_time, format_duration
from tqdm import tqdm
from runners.fourier_realtime import main as fourier_forecast_with_slope
from constants import FYAHOO__OUTPUTFILENAME, FYAHOO__OUTPUTFILENAME_DAY, FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_MONTH, NB_WORKERS
import pickle
import numpy as np
import pandas as pd
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import time

def main():
    pass

if __name__ == "__main__":
    freeze_support()
    t1 = time.time()
    ticker = "^GSPC"
    col = "Close"
    colname = (col, ticker)
    one_dataset_filename = FYAHOO__OUTPUTFILENAME_DAY
    with open(one_dataset_filename, 'rb') as f:
        data_cache_daily = pickle.load(f)
    data_cache_daily = data_cache_daily[ticker].copy()
    data_cache_weekly = data_cache_daily.resample('W-FRI').agg({
        ('Open', ticker): 'first',
        ('High', ticker): 'max',
        ('Low', ticker): 'min',
        ('Close', ticker): 'last',
        ('Volume', ticker): 'sum'
    }).copy()
    playback_a_specific_year            = None
    percentage_of_edge_for_strike_price = 0.025
    max_length_for_training             = 666
    length_prediction_for_the_future    = 4
    number_of_step_back                 = 1024
    n_forcasts                          = 99
    results_per = {}
    for idx_fr in range(0, n_forcasts):
        results_per.update({idx_fr: {2: [], 4: []}})
    for step_back in tqdm(range(0, number_of_step_back)):
        idx1      = length_prediction_for_the_future + step_back
        df_past   = data_cache_weekly[colname].iloc[-max_length_for_training-idx1:-idx1].copy()
        df_future = data_cache_weekly[colname].iloc[-idx1:len(data_cache_weekly)+-idx1+length_prediction_for_the_future].copy()
        assert len(df_future) == length_prediction_for_the_future
        assert df_future.index[0] > df_past.index[-1]
        assert len(df_past) == max_length_for_training
        if playback_a_specific_year is not None:
            if playback_a_specific_year != df_past.index[-1].year:
                continue
        # Define all parameters explicitly
        args = Namespace(
            ticker=ticker,
            col=col,
            older_dataset=None,
            dataset_id='week',
            length_step_back=4,
            length_prediction_for_the_future=length_prediction_for_the_future,
            algorithms_to_run="0,1",
            n_forecasts=n_forcasts,
            use_this_df=df_past.values.astype(np.float64).copy(),
            plot_graph=False,
            quiet=True,
            print_result=False,
        )
        #######################################################################
        # Call the function
        #######################################################################
        forecasts, mean_forecast = fourier_forecast_with_slope(args)
        assert len(mean_forecast) == length_prediction_for_the_future
        real_values = df_future.values.astype(np.float64).copy()
        for idx_fr in range(0, len(forecasts)):
            one_forecast = forecasts[idx_fr]
            pred_values  = one_forecast.copy()
            for number_of_points in [2, 4]:
                mslope_prediction, _ = np.polyfit(np.arange(number_of_points), pred_values[0:number_of_points], 1)
                slope_gt, _          = np.polyfit(np.arange(number_of_points), real_values[0:number_of_points], 1)
                if slope_gt < 0:
                    if mslope_prediction < 0:
                        results_per[idx_fr][number_of_points].append(1)
                    else:
                        results_per[idx_fr][number_of_points].append(0)
                if slope_gt > 0:
                    if mslope_prediction > 0:
                        results_per[idx_fr][number_of_points].append(1)
                    else:
                        results_per[idx_fr][number_of_points].append(0)
    # --- After the loop ---
    best_setup = None
    best_accuracy = -1.0
    summary = []

    print("\n" + "="*80)
    print("SUMMARY OF ACCURACIES (trend direction prediction)")
    print("="*80)

    for idx_fr in results_per:
        for number_of_points in [2, 4]:
            outcomes = results_per[idx_fr][number_of_points]
            if len(outcomes) == 0:
                accuracy = 0.0
            else:
                accuracy = np.mean(outcomes)
            summary.append((idx_fr, number_of_points, accuracy, len(outcomes)))

            # Track best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_setup = (idx_fr, number_of_points)

            print(f"idx_fr={idx_fr:2d}, n_points={number_of_points}: accuracy = {accuracy:.2%} ({len(outcomes)} samples)")

    print("-" * 80)
    if best_setup:
        idx_best, npoints_best = best_setup
        print(f"✅ BEST SETUP: idx_fr = {idx_best}, number_of_points = {npoints_best}")
        print(f"   Accuracy = {best_accuracy:.2%} based on {len(results_per[idx_best][npoints_best])} samples")
    else:
        print("⚠️  No valid results to evaluate.")

    t2 = time.time()
    print(f"\nTotal execution time: {format_execution_time(t2 - t1)}")

