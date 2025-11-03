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
from constants import FYAHOO__OUTPUTFILENAME, FYAHOO__OUTPUTFILENAME_DAY, FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_MONTH
import pickle
import matplotlib.pyplot as plt
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import numpy as np
from tqdm import tqdm
from algorithms.fourier import fourier_forecast_log_returns_with_confidence, fourier_extrapolation_auto


def entry():
    ticker, colname = '^GSPC', ('Close', '^GSPC')
    best_result = {}
    length_step_back = 4
    length_prediction = 4
    min_train = 25
    one_dataset_filename, one_dataset_id = FYAHOO__OUTPUTFILENAME_WEEK, 'week'
    show_graphics = False
    # for one_dataset_filename, one_dataset_id in zip([FYAHOO__OUTPUTFILENAME_DAY], ['day']):
    with open(one_dataset_filename, 'rb') as f:
        data_cache = pickle.load(f)
    # Search
    for the_algo in [fourier_forecast_log_returns_with_confidence, fourier_extrapolation_auto]:
        print(f"Using algo: {the_algo.__name__}", flush=True)
        tmp = {
            'error': float('inf'),
            'one_length_train_data': None,
            'energy_threshold': None,
            'y_true': None,
            'y_pred': None
        }
        best_result.update({f'{the_algo.__name__}': tmp})
        prices_2 = data_cache[ticker][colname].values.astype(np.float64).copy()

        max_train = len(prices_2) - length_step_back - length_prediction

        # Precompute energy thresholds to avoid repeated np.arange calls
        energy_thresholds = np.arange(0.5, 1.0, 0.01)

        # Wrap outer loop with tqdm
        for one_length_train_data in tqdm(
                range(min_train, max_train, 1),
                desc=f"Training length ({one_dataset_id})",
                leave=True
        ):
            for energy_threshold in energy_thresholds:
                # Reload data (or better: reuse prices_2 sliced appropriately)
                prices = prices_2.copy()  # Avoid reloading from disk in inner loop if possible!
                assert len(prices) > one_length_train_data + length_step_back + length_prediction
                xi1, xi2 = len(prices) - one_length_train_data - length_step_back, -length_step_back
                yi1, yi2 = len(prices) - length_step_back, len(prices) - length_step_back + length_prediction
                assert len(prices)+xi2 ==  yi1  # La fin du X correspond au début du Y
                x_series = prices[len(prices) - one_length_train_data - length_step_back: -length_step_back]
                y_series = prices[len(prices) - length_step_back: len(prices) - length_step_back + length_prediction]
                assert len(x_series) == one_length_train_data
                assert len(y_series) == length_prediction
                #print(f"{data_cache[ticker].index[xi1:xi2][0]} --> {data_cache[ticker].index[xi1:xi2][-1]}")
                #print(f"{data_cache[ticker].index[yi1:yi2][0]} --> {data_cache[ticker].index[yi1:yi2][-1]}")
                # Forecast
                forecast, lower, upper, diag = the_algo(
                    prices=x_series,
                    n_predict=length_prediction,
                    energy_threshold=round(energy_threshold, 2),  # mitigate FP errors
                    conf_level=0.95)

                y_pred = forecast[-length_prediction:]
                y_true = y_series
                assert len(y_pred) == len(y_true) == length_prediction
                error = np.sqrt(np.mean((y_true - y_pred) ** 2))

                # Update best result
                if error < best_result[f'{the_algo.__name__}']['error']:
                    best_result[f'{the_algo.__name__}'].update({
                        'error': error,
                        'length_train_data': one_length_train_data,
                        'energy_threshold': round(energy_threshold, 2),
                        'y_true': y_true.copy(),
                        'y_pred': y_pred.copy(),
                        'x_series': x_series.copy(),
                        'n_predict': length_prediction,
                        'forecast': forecast.copy(),
                        'lower': lower,
                        'upper': upper,
                        'diag': diag,
                    })
    # Final report
    for the_algo in [fourier_forecast_log_returns_with_confidence, fourier_extrapolation_auto]:
        print(f"\n✅ Best setup found for [{the_algo.__name__}]:")
        print(f"  RMSE: {best_result[f'{the_algo.__name__}']['error']:.4f}")
        print(f"  Training length: {best_result[f'{the_algo.__name__}']['one_length_train_data']}")
        print(f"  Energy threshold: {best_result[f'{the_algo.__name__}']['energy_threshold']:.2f}")
        print(f"  True: {best_result[f'{the_algo.__name__}']['y_true']}")
        print(f"  Pred: {best_result[f'{the_algo.__name__}']['y_pred']}")

        prices   = best_result[f'{the_algo.__name__}']['x_series']
        n_pred   = best_result[f'{the_algo.__name__}']['n_predict']
        forecast = best_result[f'{the_algo.__name__}']['forecast']
        y_true   = best_result[f'{the_algo.__name__}']['y_true']
        error    = best_result[f'{the_algo.__name__}']['error']
        lower    = best_result[f'{the_algo.__name__}']['lower']
        upper    = best_result[f'{the_algo.__name__}']['upper']
        diag     = best_result[f'{the_algo.__name__}']['diag']
        # Plot
        plt.figure(figsize=(14, 7))
        obs_t = np.arange(len(prices))
        future_t = np.arange(len(prices), len(prices) + n_pred)

        plt.plot(obs_t, prices, color='black', label='Observed SPX Close', linewidth=1.2)
        if length_step_back > 0:
            assert len(forecast[-n_pred:]) == len(y_true)
        plt.plot(future_t, forecast[-n_pred:], color='red', marker='o', label='Fourier Forecast (log + exp detrend)', linewidth=2)
        if length_step_back > 0:
            plt.plot(future_t, y_true, color='blue', marker='+', label='Ground Truth', linewidth=2)
        plt.fill_between(future_t, lower, upper, color='red', alpha=0.2, label=f'{int(diag["conf_level"] * 100)}% Confidence Band')

        # Optional: show full reconstruction for diagnostics
        plt.plot(np.arange(len(forecast)), forecast, color='blue', alpha=0.4, label='Full Reconstruction + Forecast')

        plt.axvline(len(prices) - 1, color='gray', linestyle='--')
        if length_step_back > 0:
            plt.title(f'S&P 500 Forecast: Log Returns + Exponential Detrending + Confidence Bands ({diag["n_harm"]} harmonics)  mean error:{error:0.2f}')
        else:
            plt.title(f'S&P 500 Forecast: Log Returns + Exponential Detrending + Confidence Bands ({diag["n_harm"]} harmonics)  IN THE FUTURE')
        plt.xlabel(f'Time Step ({one_dataset_id})')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        if show_graphics:
            plt.show()
    # Projection
    length_step_back = 0
    for the_algo in [fourier_forecast_log_returns_with_confidence, fourier_extrapolation_auto]:
        length_prediction = best_result[f'{the_algo.__name__}']['n_predict']
        energy_threshold  = best_result[f'{the_algo.__name__}']['energy_threshold']
        length_train_data = best_result[f'{the_algo.__name__}']['length_train_data']
        with open(one_dataset_filename, 'rb') as f:
            data_cache = pickle.load(f)
        prices = data_cache['^GSPC'][('Close', '^GSPC')].values.astype(np.float64)

        if length_step_back > 0:
            assert length_prediction <= length_step_back
        assert len(prices) > length_train_data + length_step_back + length_prediction
        if length_step_back > 0:
            x_series = prices[len(prices) - length_train_data - length_step_back:-length_step_back]
        else:
            x_series = prices[len(prices) - length_train_data - length_step_back:]
        assert len(x_series) == length_train_data
        y_series = prices[len(prices) - length_step_back:len(prices) - length_step_back + length_prediction]
        if length_step_back > 0:
            assert len(y_series) == length_prediction
        else:
            assert 0 == len(y_series)
        prices = x_series
        n_pred = length_prediction

        # Forecast
        forecast, lower, upper, diag = the_algo(
            prices=prices,
            n_predict=n_pred,
            energy_threshold=energy_threshold,
            conf_level=0.95
        )
        if length_step_back > 0:
            assert len(forecast[-n_pred:]) == len(y_series)

            def mse(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)

            y_true, y_pred = y_series, forecast[-n_pred:]
            error = mse(forecast[-n_pred:], y_series) ** 0.5
            print(f"Error: {error:0.2f}   {y_true} ? {y_pred}")

        # Plot
        plt.figure(figsize=(14, 7))
        obs_t = np.arange(len(prices))
        future_t = np.arange(len(prices), len(prices) + n_pred)

        plt.plot(obs_t, prices, color='black', label='Observed SPX Close', linewidth=1.2)
        if length_step_back > 0:
            assert len(forecast[-n_pred:]) == len(y_true)
        plt.plot(future_t, forecast[-n_pred:], color='red', marker='o', label='Fourier Forecast (log + exp detrend)', linewidth=2)
        if length_step_back > 0:
            plt.plot(future_t, y_true, color='blue', marker='+', label='Ground Truth', linewidth=2)
        plt.fill_between(future_t, lower, upper, color='red', alpha=0.2, label=f'{int(diag["conf_level"] * 100)}% Confidence Band')

        # Optional: show full reconstruction for diagnostics
        plt.plot(np.arange(len(forecast)), forecast, color='blue', alpha=0.4, label='Full Reconstruction + Forecast')

        plt.axvline(len(prices) - 1, color='gray', linestyle='--')
        if length_step_back > 0:
            plt.title(f'S&P 500 Forecast: {the_algo} ({diag["n_harm"]} harmonics)  mean error:{error:0.2f}')
        else:
            plt.title(f'S&P 500 Forecast: {the_algo} ({diag["n_harm"]} harmonics)  IN THE FUTURE')
        plt.xlabel(f'Time Step ({one_dataset_id})')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        if show_graphics:
            plt.show()

    return best_result


if __name__ == "__main__":
    freeze_support()
    entry()