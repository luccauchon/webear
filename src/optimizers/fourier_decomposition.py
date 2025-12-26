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
from constants import NB_WORKERS
import pickle
import time
import psutil
import matplotlib.pyplot as plt
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import numpy as np
from tqdm import tqdm
from utils import transform_path
import os
import copy

import warnings

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,
    module="sklearn.utils.deprecation"
)

###############################################################################
# NE JAMAIS CHANGE L'ORDRE
# JUSTE AJOUTER DES METHODES
###############################################################################
# Define a registry mapping algo names (strings) to actual functions
from algorithms.fourier import fourier_forecast_log_returns_with_confidence, fourier_extrapolation_auto, fourier_extrapolation_hybrid
FOURIER_ALGO_REGISTRY = [fourier_forecast_log_returns_with_confidence,  # 0
                         fourier_extrapolation_auto,  # 1
                         # fourier_extrapolation_hybrid,  # 2   Identique à #0. À voir .
                        ]
    # 'sarimax_auto_forecast': sarimax_auto_forecast,
###############################################################################
###############################################################################


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared):
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)
    # print(f"[{os.getpid()}] Hello world!", flush=True)
    best_result, all_results_from_worker, NNN = {}, {}, 500
    while True:
        try:
            use_case = use_cases__shared.get(timeout=0.1)
        except:
            break  # Terminé
        the_algo_name   = use_case['the_algo_name']
        the_algo_index  = use_case['the_algo_index']
        algo_func       = FOURIER_ALGO_REGISTRY[the_algo_index]
        x_series        = use_case['prices']
        y_series        = use_case['y_series']
        length_prediction = use_case['n_predict']
        energy_threshold  = use_case['energy_threshold']
        length_train_data = use_case['length_train_data']
        t1 = time.time()
        assert length_train_data == len(x_series)
        # print(f"[{os.getpid()}] {x_series}", flush=True)
        forecast, lower, upper, diag = algo_func(prices=x_series, n_predict=length_prediction, energy_threshold=round(energy_threshold, 2), conf_level=0.95)
        y_pred = forecast[-length_prediction:]
        y_true = y_series
        assert len(y_pred) == len(y_true) == length_prediction
        error = np.sqrt(np.mean((y_true - y_pred) ** 2))
        # print(f"[{os.getpid()}] <<{the_algo_name}>>  {length_train_data}  {int(error):.0f}  {time.time()-t1:.2f} seconds")
        # Update best result
        if the_algo_name not in best_result:
            best_result.update({the_algo_name: {'error': 9999999}})
        if error < best_result[f'{the_algo_name}']['error']:
            best_result[f'{the_algo_name}'].update({
            'error': error,
            'length_train_data': length_train_data,
            'energy_threshold': round(energy_threshold, 2),
            'y_true': copy.deepcopy(y_true),
            'y_pred': copy.deepcopy(y_pred),
            'x_series': copy.deepcopy(x_series),
            'n_predict': length_prediction,
            'forecast': copy.deepcopy(forecast),
            'lower': lower,
            'upper': upper,
            'diag': diag,
            'the_algo': the_algo_name,
        })
        # Keep all of them
        all_results_from_worker.update({error: {
            'error': error,
            'length_train_data': length_train_data,
            'energy_threshold': round(energy_threshold, 2),
            'y_true': copy.deepcopy(y_true),
            'y_pred': copy.deepcopy(y_pred),
            'x_series': copy.deepcopy(x_series),
            'n_predict': length_prediction,
            'forecast': copy.deepcopy(forecast),
            'lower': lower,
            'upper': upper,
            'diag': diag,
            'the_algo': the_algo_name,
        }})
    # print(f"[{os.getpid()}] Hello world!", flush=True)
    all_results_from_worker = dict(sorted(all_results_from_worker.items()))
    all_results_from_worker = {k: v for i, (k, v) in enumerate(all_results_from_worker.items()) if i < NNN}
    out__shared.put((best_result, all_results_from_worker))


def entry(one_dataset_filename=None, one_dataset_id=None, length_prediction_for_forecast=None, ticker= '^GSPC', col='Close', show_plots=False, selected_algo=(0,1,2),
          save_graphics = False, fast_result=False, print_result=False, length_step_back=4, multi_threaded=False, use_this_df=None):
    colname = (col, ticker)
    best_result = {}
    min_train = 20
    _nb_workers = NB_WORKERS
    length_prediction_in_training = length_step_back
    algorithms_to_run = [e for e, el in enumerate(FOURIER_ALGO_REGISTRY) if e in selected_algo]
    if use_this_df is None or 0 == len(use_this_df):
        assert one_dataset_filename is not None
        with open(one_dataset_filename, 'rb') as f:
            data_cache = pickle.load(f)
        prices_2 = copy.deepcopy(data_cache[ticker][colname].values.astype(np.float64))
    else:
        data_cache = None
        one_dataset_filename = None
        prices_2 = copy.deepcopy(use_this_df[colname].values)
    # Precompute energy thresholds to avoid repeated np.arange calls
    energy_thresholds = np.arange(0.95, 1.0, 0.1) if fast_result else np.arange(0.5, 1.0, 0.01)
    max_train = len(prices_2) - length_step_back - length_prediction_in_training
    if max_train > 4000:
        print(f"*****************************  Setting max_train to 4000")
        max_train = 4000
    # print(f"{max_train=}")
    max_train = range(min_train, min_train + 10) if fast_result else range(min_train, max_train)
    # print(f"Computing {len(algorithms_to_run)*len(max_train)*len(energy_thresholds)} configuration")
    all_results_collected = {}
    if multi_threaded:
        # Generate use cases
        use_cases = []
        for the_algo_index in algorithms_to_run:
            if the_algo_index >= len(FOURIER_ALGO_REGISTRY):
                continue
            the_algo = FOURIER_ALGO_REGISTRY[the_algo_index]
            for one_length_train_data in max_train:
                for energy_threshold in energy_thresholds:
                    # Reload data (or better: reuse prices_2 sliced appropriately)
                    prices = copy.deepcopy(prices_2)  # Avoid reloading from disk in inner loop if possible!
                    assert len(prices) > one_length_train_data + length_step_back + length_prediction_in_training
                    xi1, xi2 = len(prices) - one_length_train_data - length_step_back, -length_step_back
                    yi1, yi2 = len(prices) - length_step_back, len(prices) - length_step_back + length_prediction_in_training
                    assert len(prices) + xi2 == yi1  # La fin du X correspond au début du Y
                    x_series = prices[len(prices) - one_length_train_data - length_step_back: -length_step_back]
                    y_series = prices[len(prices) - length_step_back: len(prices) - length_step_back + length_prediction_in_training]
                    assert len(x_series) == one_length_train_data
                    assert len(y_series) == length_prediction_in_training
                    use_cases.append({'the_algo_name': the_algo.__name__, 'the_algo_index': the_algo_index,
                                      'prices': x_series, 'y_series': y_series, 'n_predict': length_prediction_in_training,
                                      'energy_threshold': round(energy_threshold, 2), 'length_train_data': one_length_train_data})
        # use_cases = use_cases[:999]
        # print(f"Got {len(use_cases)} use_cases")
        # _nb_workers =1
        # Search
        use_cases__shared, master_cmd__shared = Queue(len(use_cases)), Value("i", 0)
        out__shared = [Queue(1) for k in range(0, _nb_workers)]
        # Lancement des workers
        for k in range(0, _nb_workers):
            p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k],))
            p.start()
            pid = p.pid
            p_obj = psutil.Process(pid)
            # p_obj.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        # Envoie les informations aux workers pour traitement
        for use_case in use_cases:
            use_cases__shared.put(use_case)
        # Autoriser les workers à traiter
        with master_cmd__shared.get_lock():
            master_cmd__shared.value = 1
        # Récupération des résultats
        data_from_workers = []
        for k in range(0, _nb_workers):
            data_from_workers.append(out__shared[k].get())
        # Conserve les meilleurs
        best_result = {FOURIER_ALGO_REGISTRY[the_algo_index].__name__: {'error': 9999999} for the_algo_index in algorithms_to_run}
        for the_algo_index in algorithms_to_run:
            the_algo = FOURIER_ALGO_REGISTRY[the_algo_index]
            for tmp_payload in data_from_workers:
                one_of_the_best, all_results_from_worker = tmp_payload[0], tmp_payload[1]
                all_results_collected |= all_results_from_worker
                if the_algo.__name__ not in one_of_the_best:
                    continue
                if one_of_the_best[the_algo.__name__]['error'] < best_result[the_algo.__name__]['error']:
                    best_result[the_algo.__name__] = one_of_the_best[the_algo.__name__]
    else:
        # Search
        for the_algo_index in algorithms_to_run:
            if the_algo_index >= len(FOURIER_ALGO_REGISTRY):
                continue
            the_algo = FOURIER_ALGO_REGISTRY[the_algo_index]
            tmp = {
                'error': float('inf'),
                'length_train_data': None,
                'energy_threshold': None,
                'y_true': None,
                'y_pred': None
            }
            best_result.update({f'{the_algo.__name__}': tmp})
            # Wrap outer loop with tqdm
            for one_length_train_data in tqdm(max_train,desc=f"Training length ({one_dataset_id})",leave=True):
                for energy_threshold in energy_thresholds:
                    # Reload data (or better: reuse prices_2 sliced appropriately)
                    prices = copy.deepcopy(prices_2)  # Avoid reloading from disk in inner loop if possible!
                    assert len(prices) > one_length_train_data + length_step_back + length_prediction_in_training
                    xi1, xi2 = len(prices) - one_length_train_data - length_step_back, -length_step_back
                    yi1, yi2 = len(prices) - length_step_back, len(prices) - length_step_back + length_prediction_in_training
                    assert len(prices)+xi2 ==  yi1  # La fin du X correspond au début du Y
                    assert yi2-yi1 == length_prediction_in_training
                    x_series = prices[len(prices) - one_length_train_data - length_step_back: -length_step_back]
                    y_series = prices[len(prices) - length_step_back: len(prices) - length_step_back + length_prediction_in_training]
                    assert len(x_series) == one_length_train_data
                    assert len(y_series) == length_prediction_in_training
                    # Forecast
                    forecast, lower, upper, diag = the_algo(
                        prices=x_series,
                        n_predict=length_prediction_in_training,
                        energy_threshold=round(energy_threshold, 2),  # mitigate FP errors
                        conf_level=0.95)
                    y_pred = forecast[-length_prediction_in_training:]
                    y_true = y_series
                    assert len(y_pred) == len(y_true) == length_prediction_in_training
                    error = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    all_results_collected.update({error: {
                            'error': error,
                            'length_train_data': one_length_train_data,
                            'energy_threshold': round(energy_threshold, 2),
                            'y_true': copy.deepcopy(y_true),
                            'y_pred': copy.deepcopy(y_pred),
                            'x_series': copy.deepcopy(x_series),
                            'n_predict': length_prediction_in_training,
                            'forecast': copy.deepcopy(forecast),
                            'lower': lower,
                            'upper': upper,
                            'diag': diag,
                            'the_algo': the_algo.__name__,
                        }})
                    # Update best result
                    if error < best_result[f'{the_algo.__name__}']['error']:
                        best_result[f'{the_algo.__name__}'].update({
                            'error': error,
                            'length_train_data': one_length_train_data,
                            'energy_threshold': round(energy_threshold, 2),
                            'y_true': copy.deepcopy(y_true),
                            'y_pred': copy.deepcopy(y_pred),
                            'x_series': copy.deepcopy(x_series),
                            'n_predict': length_prediction_in_training,
                            'forecast': copy.deepcopy(forecast),
                            'lower': lower,
                            'upper': upper,
                            'diag': diag,
                            'the_algo': the_algo.__name__,
                        })
    # Wipe not good result
    for the_algo_index in algorithms_to_run:
        the_algo = FOURIER_ALGO_REGISTRY[the_algo_index]
        if best_result[f'{the_algo.__name__}']['error'] > 9999998.:
            del best_result[f'{the_algo.__name__}']
    # Final report
    for the_algo_index in algorithms_to_run:
        the_algo = FOURIER_ALGO_REGISTRY[the_algo_index]
        if print_result:
            print(f"\n✅ [{ticker}][{colname}] Best setup found for [{the_algo.__name__}]:")
            print(f"  RMSE: {best_result[f'{the_algo.__name__}']['error']:.4f}")
            print(f"  Training length: {best_result[f'{the_algo.__name__}']['length_train_data']}")
            print(f"  Energy threshold: {best_result[f'{the_algo.__name__}']['energy_threshold']:.2f}")
            print(f"  True: {best_result[f'{the_algo.__name__}']['y_true']}")
            print(f"  Pred: {best_result[f'{the_algo.__name__}']['y_pred']}")
        if f'{the_algo.__name__}' not in best_result:
            continue
        l_train  = best_result[f'{the_algo.__name__}']['length_train_data']
        prices   = best_result[f'{the_algo.__name__}']['x_series']
        n_pred   = best_result[f'{the_algo.__name__}']['n_predict']
        forecast = best_result[f'{the_algo.__name__}']['forecast']
        y_true   = best_result[f'{the_algo.__name__}']['y_true']
        error    = best_result[f'{the_algo.__name__}']['error']
        lower    = best_result[f'{the_algo.__name__}']['lower']
        upper    = best_result[f'{the_algo.__name__}']['upper']
        diag     = best_result[f'{the_algo.__name__}']['diag']
        # Plot
        if save_graphics or show_plots:
            plt.figure(figsize=(14, 7))
            obs_t = np.arange(len(prices))
            future_t = np.arange(len(prices), len(prices) + n_pred)

            plt.plot(obs_t, prices, color='black', label='Observed SPX Close', linewidth=1.2)
            if length_step_back > 0:
                assert len(forecast[-n_pred:]) == len(y_true)
            plt.plot(future_t, forecast[-n_pred:], color='red', marker='o', label=f'{the_algo.__name__}', linewidth=2)
            if length_step_back > 0:
                plt.plot(future_t, y_true, color='blue', marker='+', label='Ground Truth', linewidth=2)
            if lower is not None and upper is not None:
                plt.fill_between(future_t, lower, upper, color='red', alpha=0.2, label=f'{int(diag["conf_level"] * 100)}% Confidence Band')

            # Optional: show full reconstruction for diagnostics
            plt.plot(np.arange(len(forecast)), forecast, color='blue', alpha=0.4, label='Full Reconstruction + Forecast')

            plt.axvline(len(prices) - 1, color='gray', linestyle='--')
            if length_step_back > 0:
                plt.title(f'S&P 500 Training: {the_algo.__name__} ({diag["n_harm"]} harmonics, {l_train} training points)  mean error:{error:0.2f}')
            else:
                plt.title(f'S&P 500 Training: {the_algo.__name__} ({diag["n_harm"]} harmonics)  IN THE FUTURE')
            plt.xlabel(f'Time Step ({one_dataset_id})')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            if save_graphics:
                # Save figure instead of showing
                safe_algo_name = the_algo.__name__.replace("/", "_").replace("\\", "_")  # sanitize if needed
                filename = f"Training_{ticker}_{colname}_{safe_algo_name}.png"
                plt.savefig(filename, dpi=150)
            if not show_plots:
                plt.close()  # Important: frees memory and prevents figure accumulation
    # Projection
    length_step_back = 0
    for the_algo_index in algorithms_to_run:
        if length_prediction_for_forecast is None:
            continue
        assert length_prediction_for_forecast > 0
        the_algo = FOURIER_ALGO_REGISTRY[the_algo_index]
        energy_threshold  = best_result[f'{the_algo.__name__}']['energy_threshold']
        length_train_data = best_result[f'{the_algo.__name__}']['length_train_data']
        prices = copy.deepcopy(prices_2)
        assert len(prices) > length_train_data + length_step_back + length_prediction_for_forecast
        x_series = prices[len(prices) - length_train_data - length_step_back:]
        assert len(x_series) == length_train_data
        # Forecast
        forecast, lower, upper, diag = the_algo(
            prices=x_series,
            n_predict=length_prediction_for_forecast,
            energy_threshold=energy_threshold,
            conf_level=0.95
        )
        assert len(forecast) == len(x_series) + length_prediction_for_forecast
        # Plot
        if save_graphics or show_plots:
            plt.figure(figsize=(14, 7))
            obs_t = np.arange(len(prices))
            future_t = np.arange(len(prices), len(prices) + length_prediction_for_forecast)

            plt.plot(obs_t, prices, color='black', label='Observed SPX Close', linewidth=1.2)
            plt.plot(future_t, forecast[-length_prediction_for_forecast:], color='red', marker='o', label=f'{the_algo.__name__}', linewidth=2)
            if lower is not None and upper is not None:
                plt.fill_between(future_t, lower, upper, color='red', alpha=0.2, label=f'{int(diag["conf_level"] * 100)}% Confidence Band')

            # Optional: show full reconstruction for diagnostics
            # plt.plot(np.arange(len(forecast)), forecast, color='blue', alpha=0.4, label='Full Reconstruction + Forecast')

            plt.axvline(len(prices) - 1, color='gray', linestyle='--')
            plt.title(f'S&P 500 Forecast: {the_algo} ({diag["n_harm"]} harmonics)  IN THE FUTURE')
            plt.xlabel(f'Time Step ({one_dataset_id})')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            if save_graphics:
                # Save figure instead of showing
                safe_algo_name = the_algo.__name__.replace("/", "_").replace("\\", "_")  # sanitize if needed
                filename = f"Forecast_{ticker}_{colname}_{safe_algo_name}.png"
                plt.savefig(filename, dpi=150)
            if not show_plots:
                plt.close()  # Important: frees memory and prevents figure accumulation
    # Show all plots at once if requested
    if show_plots:
        plt.show()  # This will display all open figures simultaneously (backend-dependent)
    all_results_collected = dict(sorted(all_results_collected.items()))
    return best_result, copy.deepcopy(data_cache) if data_cache is not None else None, all_results_collected