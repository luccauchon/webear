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
from constants import FYAHOO__OUTPUTFILENAME_WEEK, OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST
import numpy as np
import matplotlib.pyplot as plt
from utils import transform_path
from multiprocessing import freeze_support
from datetime import datetime
import argparse


def get_prediction_for_the_future_with_fourier_algo(_best_setup, _data_cache, _col, _ticker, _length_prediction):
    prices = _data_cache[_ticker][(_col, _ticker)].values.astype(np.float64).copy()
    length_train_data = _best_setup['length_train_data']
    length_step_back = 0
    n_pred = _length_prediction
    energy_threshold = _best_setup['energy_threshold']
    the_algo = _best_setup['the_algo']

    import algorithms.fourier
    the_function = getattr(algorithms.fourier, the_algo)

    assert len(prices) > length_train_data + length_step_back + n_pred
    x_series = prices[len(prices) - length_train_data - length_step_back:]
    assert len(x_series) == length_train_data
    y_series = prices[len(prices) - length_step_back:len(prices) - length_step_back + n_pred]
    assert len(y_series) == 0  # because length_step_back=0

    forecast, lower, upper, diag = the_function(x_series, n_predict=n_pred, energy_threshold=energy_threshold, conf_level=0.95)

    prediction = forecast[-n_pred:].copy()
    assert len(prediction) == n_pred
    if lower is not None:
        assert len(upper) == n_pred
        assert len(lower) == n_pred
    x = forecast[:-n_pred].copy()
    assert len(x) == length_train_data
    return prediction, x, lower, upper


def main():
    parser = argparse.ArgumentParser(description="Run Fourier-based stock forecast.")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="2025.10.31")  # e.g., "2025.10.31" or "None"
    parser.add_argument("--dataset_id", type=str, default='week')
    parser.add_argument("--length_step_back", type=int, default=4)
    parser.add_argument("--length_prediction_for_the_future", type=int, default=4)
    parser.add_argument("--algorithms_to_run", type=str, default="0,1,2")
    args = parser.parse_args()

    # --- Nicely print the arguments ---
    print("🔧 Arguments:")
    for arg, value in vars(args).items():
        print(f"    {arg:.<40} {value}")
    print("-" * 80, flush=True)

    # Handle older_dataset = "None" as actual None
    older_dataset = None if args.older_dataset == "None" else args.older_dataset

    one_dataset_filename = FYAHOO__OUTPUTFILENAME_WEEK if older_dataset is None else transform_path(FYAHOO__OUTPUTFILENAME_WEEK, older_dataset)

    close_results, data_cache, _ = entry_of__fourier_decomposition(
        multi_threaded=True,
        ticker=args.ticker,
        col=args.col,
        fast_result=False,
        length_step_back=args.length_step_back,
        one_dataset_filename=one_dataset_filename,
        one_dataset_id=args.dataset_id,
        selected_algo=[int(num) for num in args.algorithms_to_run.split(",")]
    )

    best_prediction = min(close_results.values(), key=lambda v: v['error'])

    with open(one_dataset_filename, 'rb') as f:
        data_cache = pickle.load(f)

    close_prediction, close_values, close_prediction_lower, close_prediction_upper = get_prediction_for_the_future_with_fourier_algo(
        _best_setup=best_prediction,
        _data_cache=data_cache,
        _col=args.col,
        _length_prediction=args.length_prediction_for_the_future,
        _ticker=args.ticker
    )

    # --- Plotting ---
    plt.figure(figsize=(14, 7))
    n_train_plot = min(args.length_prediction_for_the_future, len(close_values))
    plot_start_idx = len(close_values) - n_train_plot
    train_indices_full = np.arange(len(close_values))
    train_indices_plot = train_indices_full[plot_start_idx:]
    close_values_plot = close_values[plot_start_idx:]
    pred_indices = np.arange(len(close_values), len(close_values) + len(close_prediction))

    y_values = list(close_values_plot) + list(close_prediction)
    if close_prediction_lower is not None:
        y_values += list(close_prediction_lower) + list(close_prediction_upper)

    y_min = min(y_values)
    y_max = max(y_values)
    margin = (y_max - y_min) * 0.02
    y_min -= margin
    y_max += margin

    plt.plot(train_indices_plot, close_values_plot, label=f'Historical {args.col}', color='blue',
             marker='s', markersize=8, linestyle='-', linewidth=1)
    plt.plot(pred_indices, close_prediction, label=f'Predicted {args.col}', color='red',
             linestyle='--', marker='s', markersize=10, linewidth=1)

    for x, y in zip(train_indices_plot, close_values_plot):
        plt.text(x, y, f'{y:.2f}', fontsize=10, ha='left', va='bottom', color='darkblue')
    for x, y in zip(pred_indices, close_prediction):
        plt.text(x, y, f'{y:.2f}', fontsize=10, ha='left', va='bottom', color='darkred')

    if close_prediction_lower is not None:
        plt.fill_between(pred_indices, close_prediction_lower, close_prediction_upper,
                         color='orange', alpha=0.3, label='95% Confidence Interval')

    sep_x = len(close_values) - 0.5
    plt.axvline(x=sep_x, color='black', linewidth=3, linestyle='-', label='Train / Prediction Boundary')

    plt.ylim(y_min, y_max)
    today_str = datetime.today().strftime('%Y-%m-%d')
    title = f'[{args.dataset_id}][train error:{best_prediction["error"]:0.2f}]{args.col} Forecast for {args.ticker} ({best_prediction["the_algo"]}) – Last {n_train_plot} Training Points'
    plt.title(title, fontsize=14)
    plt.xlabel('Time Index (Relative)')
    plt.ylabel(f'{args.col} Price')
    plt.legend()
    plt.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
    plt.minorticks_on()
    plt.tight_layout()

    os.makedirs(os.path.join(OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST, args.older_dataset), exist_ok=True)
    figure_filename = os.path.join(OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST, args.older_dataset, f"forecast_{args.ticker}_{args.dataset_id}_{today_str}.png")
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()