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
from runners.fourier_forecast import get_prediction_for_the_future_with_fourier_algo


def main():
    parser = argparse.ArgumentParser(description="Run Fourier-based stock forecast.")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="2025.10.31")  # e.g., "2025.10.31" or "None"
    parser.add_argument("--dataset_id", type=str, default='week')
    parser.add_argument("--length_step_back", type=int, default=4)
    parser.add_argument("--length_prediction_for_the_future", type=int, default=4)
    parser.add_argument("--algorithms_to_run", type=str, default="0,1,2")
    parser.add_argument("--n_forecasts", type=int, default=19)
    args = parser.parse_args()

    # --- Nicely print the arguments ---
    print("🔧 Arguments:")
    for arg, value in vars(args).items():
        print(f"    {arg:.<40} {value}")
    print("-" * 80, flush=True)

    # Handle older_dataset = "None" as actual None
    older_dataset = None if args.older_dataset == "None" else args.older_dataset

    one_dataset_filename = FYAHOO__OUTPUTFILENAME_WEEK if older_dataset is None else transform_path(FYAHOO__OUTPUTFILENAME_WEEK, older_dataset)

    close_results, _, all_results = entry_of__fourier_decomposition(
        multi_threaded=True,
        ticker=args.ticker,
        col=args.col,
        fast_result=False,
        length_step_back=args.length_step_back,
        one_dataset_filename=one_dataset_filename,
        one_dataset_id=args.dataset_id,
        selected_algo=[int(num) for num in args.algorithms_to_run.split(",")]
    )

    # Load full data_cache from file (needed for get_prediction_for_the_future)
    with open(one_dataset_filename, 'rb') as f:
        data_cache = pickle.load(f)

    # Get base historical series from the top forecast
    all_sorted = list(all_results.values())
    if len(all_sorted) == 0:
        raise ValueError("No results found in all_results.")
    if args.n_forecasts > len(all_sorted):
        print(f"Warning: requested {args.n_forecasts} forecasts, but only {len(all_sorted)} available. Using all.")
        n_forecasts = len(all_sorted)

    first_best = all_sorted[0]
    _, base_close_values, _, _ = get_prediction_for_the_future_with_fourier_algo(
        _best_setup=first_best,
        _data_cache=data_cache.copy(),
        _col=args.col,
        _length_prediction=args.length_prediction_for_the_future,
        _ticker=args.ticker
    )

    # Generate N forecasts
    forecasts = []
    pred_indices = None

    for i in range(args.n_forecasts):
        setup = all_sorted[i]
        pred, _, _, _ = get_prediction_for_the_future_with_fourier_algo(
            _best_setup=setup,
            _data_cache=data_cache.copy(),
            _col=args.col,
            _length_prediction=args.length_prediction_for_the_future,
            _ticker=args.ticker
        )
        forecasts.append(pred)
        if pred_indices is None:
            pred_indices = np.arange(len(base_close_values), len(base_close_values) + len(pred))

    forecasts = np.array(forecasts)  # Shape: (n_forecasts, n_pred)
    mean_forecast = np.mean(forecasts, axis=0)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(16, 9))

    # Historical data
    n_train_plot = min(30, len(base_close_values))
    start_idx = len(base_close_values) - n_train_plot
    train_idx = np.arange(start_idx, len(base_close_values))
    hist_vals = base_close_values[start_idx:]

    ax.plot(train_idx, hist_vals, label='Historical Close', color='black',
            marker='o', markersize=6, linewidth=2, zorder=10)

    # Individual forecasts (faint)
    for f in forecasts:
        ax.plot(pred_indices, f, color='red', linestyle='--', linewidth=1, alpha=0.4)

    # Mean tendency (bold)
    ax.plot(pred_indices, mean_forecast, color='blue', linestyle='-', linewidth=3,
            marker='o', markersize=7, label=f'Mean Forecast (n={args.n_forecasts})', zorder=20)

    # Vertical separator
    ax.axvline(x=len(base_close_values) - 0.5, color='gray', linestyle='--',
               linewidth=2, label='Train / Prediction Boundary')

    # Y-limits
    all_ys = np.concatenate([hist_vals, mean_forecast, forecasts.flatten()])
    y_margin = (all_ys.max() - all_ys.min()) * 0.03
    ax.set_ylim(all_ys.min() - y_margin, all_ys.max() + y_margin)

    # Labels
    ax.set_title(f'Mean Tendency of Top {args.n_forecasts} Forecasts for {args.ticker} ({args.dataset_id})', fontsize=16)
    ax.set_xlabel('Time Index (Relative)')
    ax.set_ylabel('Close Price')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # --- Slope annotation ---
    if len(mean_forecast) > 1:
        slope = (mean_forecast[-1] - mean_forecast[0]) / (len(mean_forecast) - 1)
        total_change = mean_forecast[-1] - mean_forecast[0]
        slope_text = (
            f"Slope per step: {slope:+.4f}\n"
            f"Total Δ over {len(mean_forecast)} steps: {total_change:+.4f}"
        )
        print(f"\n{slope_text}")
        # Place nicely on the plot (bottom-left, inside a box)
        ax.text(
            0.02, 0.02, slope_text,
            transform=fig.transFigure,
            fontsize=11,
            ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='navy', boxstyle='round,pad=0.4')
        )
    else:
        msg = "Prediction horizon = 1 → slope undefined"
        print(f"\n{msg}")
        ax.text(
            0.02, 0.02, msg,
            transform=fig.transFigure,
            fontsize=11,
            ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.4')
        )

    plt.tight_layout()

    # Save and show AFTER adding text
    today_str = datetime.today().strftime('%Y-%m-%d')
    os.makedirs(os.path.join(OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST, args.older_dataset), exist_ok=True)
    figure_filename = os.path.join(OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST, args.older_dataset, f"forecast_mean_slope_{args.ticker}_{args.dataset_id}__.png")
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()