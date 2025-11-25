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
import argparse
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pickle
from constants import NB_WORKERS, IS_RUNNING_ON_CASIR
import psutil
import json
import math
import warnings
warnings.filterwarnings("ignore", message="Level value.*too high.*")
# Or suppress only specific RuntimeWarnings from NumPy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
import pandas as pd
import numpy as np
import pywt
import pyarrow as pa
from pyarrow import ipc
from utils import next_weekday, transform_path


###############################################################################
# NE JAMAIS CHANGE L'ORDRE
# JUSTE AJOUTER DES METHODES
###############################################################################
from algorithms.wavelet import wavelet_forecast__version_1, wavelet_multi_forecast__version_2
# Define a registry mapping algo names (strings) to actual functions
WAVELET_ALGO_REGISTRY = [wavelet_multi_forecast__version_2,  # 0
                         wavelet_forecast__version_1,  # 1
                        ]
###############################################################################
###############################################################################

def serialize_dict_of_dfs(df_dict):
    """
    Convert a dict of pandas DataFrames to a dict of Arrow IPC bytes.
    """
    serialized = {}
    for key, df in df_dict.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Value for key '{key}' is not a pandas DataFrame")
        table = pa.Table.from_pandas(df)
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        serialized[key] = sink.getvalue().to_pybytes()
    return serialized

def deserialize_dict_of_dfs(serialized_dict):
    """
    Convert a dict of Arrow IPC bytes back to a dict of pandas DataFrames.
    """
    df_dict = {}
    for key, data_bytes in serialized_dict.items():
        reader = ipc.open_stream(pa.BufferReader(data_bytes))
        table = reader.read_all()
        df_dict[key] = table.to_pandas()
    return df_dict


def _worker_processor_backtesting(use_cases__shared, master_cmd__shared, out__shared, the_filename_for_worker):
    with open(the_filename_for_worker, 'rb') as f:
        data = pickle.load(f)

    NNN = 88
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)
    best_rmse = {'rmse': 9999}
    all_results_computed = []
    while True:
        use_case_batch = []
        while len(use_case_batch) < 10:
            try:
                item = use_cases__shared.get(timeout=0.1)
                use_case_batch.append(item)
            except:
                break  # Queue is empty or no more items within timeout
        if 0 == len(use_case_batch):
            break
        for use_case in use_case_batch:
            n_train_length    = use_case['n_train_length']
            level             = use_case['level']
            wavlet_type       = use_case['wavlet_type']
            n_forecast_length = use_case['n_forecast_length']
            close_col         = use_case['close_col']
            RMSE_TOL          = use_case['RMSE_TOL']
            step_back         = use_case['step_back']
            index_of_algo     = use_case['index_of_algo']
            if 0 == step_back:
                df = data[close_col].copy()
            else:
                df = data[close_col][:-step_back].copy()
            assert len(df) == len(data[close_col]) - step_back, f"{len(df)=}  {len(data[close_col])=}  {step_back=}"
            train_prices = df[-n_train_length - n_forecast_length:-n_forecast_length]
            gt_prices    = df[-n_forecast_length:]
            assert train_prices.index.is_unique and gt_prices.index.is_unique, "Indices must be unique"
            assert train_prices.index.intersection(gt_prices.index).empty, "Indices must be disjoint"
            t1, t2, t3, t4 = train_prices.index[0], train_prices.index[-1], gt_prices.index[0], gt_prices.index[-1]
            train_prices = train_prices.values.astype(np.float64).copy()
            gt_prices    = gt_prices.values.astype(np.float64).copy()
            try:
                algo = WAVELET_ALGO_REGISTRY[index_of_algo]
                forecast_values = algo(
                    prices=train_prices,
                    forecast_steps=n_forecast_length,
                    wavelet=wavlet_type,
                    level=level,
                )
            except Exception as e:
                continue
            assert len(forecast_values) == n_train_length + n_forecast_length
            pred_values = forecast_values[-n_forecast_length:]
            assert len(pred_values) == len(gt_prices) == n_forecast_length
            # After computing pred_values and gt_prices
            rmse = np.sqrt(np.mean((pred_values - gt_prices) ** 2))

            # Compute Directional Accuracy (DA)
            if n_forecast_length > 1:
                true_diff = np.diff(gt_prices)
                pred_diff = np.diff(pred_values)
                # Handle zero differences by treating them as no change (direction = 0)
                true_direction = np.sign(true_diff)
                pred_direction = np.sign(pred_diff)
                directional_accuracy = np.mean(true_direction == pred_direction)
            else:
                # Not enough points to compute direction (only 1-step forecast)
                directional_accuracy = np.nan

            # Determine if new candidate is better
            is_better = False

            if rmse < best_rmse['rmse'] - RMSE_TOL:
                # Clearly better RMSE
                is_better = True
            elif abs(rmse - best_rmse['rmse']) <= RMSE_TOL:
                # RMSE tied — break tie with directional accuracy
                current_da = best_rmse.get('directional_accuracy', -1)
                if directional_accuracy > current_da:
                    is_better = True

            if is_better:
                best_rmse['rmse'] = rmse
                best_rmse['directional_accuracy'] = directional_accuracy
                best_rmse['level'] = level
                best_rmse['wavlet_type'] = wavlet_type
                best_rmse['n_train_length'] = n_train_length
                best_rmse['prices'] = train_prices.copy()
                best_rmse['gt_prices'] = gt_prices.copy()
                best_rmse['pred_values'] = pred_values.copy()
                best_rmse['n_forecast_length'] = n_forecast_length
                best_rmse['times'] = (t1, t2, t3, t4)
                best_rmse['index_of_algo'] = index_of_algo
            tmp = {}
            tmp['rmse'] = rmse
            tmp['directional_accuracy'] = directional_accuracy
            tmp['level'] = level
            tmp['wavlet_type'] = wavlet_type
            tmp['n_train_length'] = n_train_length
            tmp['prices'] = train_prices.copy()
            tmp['gt_prices'] = gt_prices.copy()
            tmp['pred_values'] = pred_values.copy()
            tmp['n_forecast_length'] = n_forecast_length
            tmp['times'] = (t1, t2, t3, t4)
            tmp['index_of_algo'] = index_of_algo
            all_results_computed.append(tmp)
    all_results_computed.sort(key=lambda x: x['rmse'])  # ascending: best (lowest RMSE) first
    # Get top NNN results (as a list)
    top_results = all_results_computed[:NNN]
    # If you really need a dictionary (e.g., indexed by rank), you can do:
    all_results_from_worker = {i: result for i, result in enumerate(top_results)}
    out__shared.put((best_rmse, all_results_from_worker))


def _worker_processor_realtime(use_cases__shared, master_cmd__shared, out__shared, the_filename_for_worker):
    with open(the_filename_for_worker, 'rb') as f:
        data = pickle.load(f)

    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)

    all_results_computed = []
    while True:
        use_case_batch = []
        while len(use_case_batch) < 10:
            try:
                item = use_cases__shared.get(timeout=0.1)
                use_case_batch.append(item)
            except:
                break  # Queue is empty or no more items within timeout
        if 0 == len(use_case_batch):
            break
        for use_case in use_case_batch:
            n_train_length    = use_case['n_train_length']
            level             = use_case['level']
            wavlet_type       = use_case['wavlet_type']
            n_forecast_length = use_case['n_forecast_length']
            close_col         = use_case['close_col']
            index_of_algo     = use_case['index_of_algo']
            dataset_id        = use_case['dataset_id']

            df = data[close_col].copy()
            train_prices = df[-n_train_length:]

            assert train_prices.index.is_unique

            t1, t2, t3, t4 = train_prices.index[0], train_prices.index[-1], None, None

            assert dataset_id in ['day', 'week']
            freq = 'W'
            if dataset_id == 'day':
                freq = 'D'

            # Map freq to DateOffset keyword
            freq_to_offset = {
                'D': 'days',
                'W': 'weeks',
                'H': 'hours',
                'min': 'minutes',
                's': 'seconds',
                # add more if needed
            }

            n = n_forecast_length

            if freq in ['D', 'H', 'min', 's']:
                t3 = next_weekday(t2)
                t4 = next_weekday(t2)
                for ppp in range(1, n):
                    t4 = next_weekday(t4)
            else:
                offset_key = freq_to_offset[freq]  # e.g., 'W' → 'weeks'
                t3 = t2 + pd.DateOffset(**{offset_key: 1})
                t4 = t2 + pd.DateOffset(**{offset_key: n})
            # print(f"[{t1.strftime('%Y%m%d')}-{t2.strftime('%m%d')}→{t3.strftime('%Y%m%d')}-{t4.strftime('%m%d')}]")
            train_prices = train_prices.values.astype(np.float64).copy()

            try:
                algo = WAVELET_ALGO_REGISTRY[index_of_algo]
                forecast_values = algo(
                    prices=train_prices,
                    forecast_steps=n_forecast_length,
                    wavelet=wavlet_type,
                    level=level,
                )
            except Exception as e:
                continue
            assert len(forecast_values) == n_train_length + n_forecast_length
            pred_values = forecast_values[-n_forecast_length:]
            assert len(pred_values) == n_forecast_length

            directional_accuracy = np.nan

            tmp = {}
            tmp['rmse'] = 9999.
            tmp['directional_accuracy'] = directional_accuracy
            tmp['level'] = level
            tmp['wavlet_type'] = wavlet_type
            tmp['n_train_length'] = n_train_length
            tmp['prices'] = train_prices.copy()
            tmp['gt_prices'] = None
            tmp['pred_values'] = pred_values.copy()
            tmp['n_forecast_length'] = n_forecast_length
            tmp['times'] = (t1, t2, t3, t4)
            tmp['index_of_algo'] = index_of_algo
            all_results_computed.append(tmp)

    all_results_computed.sort(key=lambda x: x['n_train_length'])
    # Get top NNN results (as a list)
    top_results = all_results_computed
    # If you really need a dictionary (e.g., indexed by rank), you can do:
    all_results_from_worker = {i: result for i, result in enumerate(top_results)}
    out__shared.put((None, all_results_from_worker))


def main(args):
    ticker = args.ticker
    plot_graph   = args.plot_graph
    show_graph   = args.show_graph
    save_to_disk = args.save_graph
    index_of_algo = 0
    use_given_gt_truth = args.use_given_gt_truth if isinstance(args.use_given_gt_truth, np.ndarray) else None
    verbose      = not args.quiet
    close_col = ('Close', ticker)
    n_forecast_length = int(args.n_forecast_length)
    n_forecast_length_in_training = int(args.n_forecast_length_in_training)
    n_models_to_keep = int(args.n_models_to_keep)
    real_time = args.real_time
    if not real_time:
        assert use_given_gt_truth is None
    _nb_workers = _nb_workers = 8 if real_time else NB_WORKERS
    performance_tracking     = {'put':[], 'call': [], '?': []}
    performance_tracking_xtm = {'put':  {'success': [], 'failure': []},
                                'call': {'success': [], 'failure': []}}
    performance_tracking_1_point_prediction = {'iron_condor_0DTE':   {'success': [], 'failure': []},
                                               'put_credit_spread':  {'success': [], 'failure': []},
                                               'call_credit_spread': {'success': [], 'failure': []}}
    thresholds_ep = eval(args.thresholds_ep)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    threshold_for_shape_similarity = 0.6
    number_of_step_back = int(args.number_of_step_back)
    sequence_for_train_length = range(int(eval(args.maxmin_sequence_for_train_length)[0]), int(eval(args.maxmin_sequence_for_train_length)[1]))
    sequence_for_level = range(1, 16)
    floor_and_ceil = args.floor_and_ceil
    maintenance_margin = args.maintenance_margin
    description_of_what_user_shall_do = {}  # Trace of what user shall do to execute the trade
    parameters_best_models= {}
    close_graph = not args.do_not_close_graph
    if real_time:
        assert 1 == number_of_step_back
        number_of_step_back = 1

    # Serialize the data locally for our processing.
    assert args.master_data_cache is not None
    the_filename_for_worker = args.temp_filename
    try:
        os.remove(the_filename_for_worker)
    except:
        pass
    with open(the_filename_for_worker, 'wb') as f:
        pickle.dump(args.master_data_cache, f)

    strategy_for_exit = args.strategy_for_exit
    misc_returned = {}
    # Iterate through time
    for step_back in tqdm(range(0, number_of_step_back), disable=not args.display_tqdm):
        description_of_what_user_shall_do.update({step_back: {}})
        parameters_best_models.update({step_back: []})
        misc_returned.update({step_back: {}})
        use_cases = []
        if real_time:
            use_cases =  args.real_time_use_cases
            # In real time, the use cases are given. Just add information for forecasting.
            for a_use_case in use_cases:
                a_use_case.update({'n_forecast_length': n_forecast_length, 'close_col': close_col, 'index_of_algo': index_of_algo, 'dataset_id': args.dataset_id})
        else:
            data = args.master_data_cache.copy()
            this_year = data[:-(step_back+1)].index[-1].year
            typical_price = data[data.index.year==this_year][close_col].mean()
            RMSE_TOL = max(0.5, 0.0001 * typical_price)
            for n_train_length in sequence_for_train_length:
                for level in sequence_for_level:
                    for wavlet_type in pywt.wavelist(kind='discrete'):
                        use_cases.append({'n_train_length': n_train_length, 'level': level, 'wavlet_type': wavlet_type, 'n_forecast_length': n_forecast_length_in_training,
                                          'close_col': close_col, 'RMSE_TOL': RMSE_TOL, 'step_back': step_back, 'index_of_algo': index_of_algo})
        use_cases__shared, master_cmd__shared = Queue(len(use_cases)), Value("i", 0)
        out__shared = [Queue(1) for k in range(0, _nb_workers)]
        # print(f"{len(use_cases)} use_cases to evaluate")
        # Lancement des workers
        for k in range(0, _nb_workers):
            if real_time:
                p = Process(target=_worker_processor_realtime, args=(use_cases__shared, master_cmd__shared, out__shared[k], the_filename_for_worker))
            else:
                p = Process(target=_worker_processor_backtesting, args=(use_cases__shared, master_cmd__shared, out__shared[k], the_filename_for_worker))
            p.start()
            pid = p.pid
            p_obj = psutil.Process(pid)
            p_obj.nice(10 if IS_RUNNING_ON_CASIR else psutil.BELOW_NORMAL_PRIORITY_CLASS)
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
        # Triage
        compilation = []
        for item in data_from_workers:
            compilation.extend(list(item[1].values()))
        sorted_data_from_workers = sorted(compilation, key=lambda x: x['rmse'])
        assert n_models_to_keep < len(sorted_data_from_workers)

        # Keep all the parameters needed to run the real time processor
        if not real_time:
            for a_good_example in sorted_data_from_workers:
                parameters_best_models[step_back].append({'level': a_good_example['level'], 'wavlet_type': a_good_example['wavlet_type'], 'n_train_length': a_good_example['n_train_length']})

        best_rmse = sorted_data_from_workers[0]
        n_train_length = best_rmse['n_train_length']
        train_prices = best_rmse['prices']
        gt_prices = best_rmse['gt_prices']
        n_forecast_length = best_rmse['n_forecast_length']
        algo_name = WAVELET_ALGO_REGISTRY[best_rmse['index_of_algo']].__name__
        # ---------------------------
        # Plotting All Forecasts + Mean
        # ---------------------------
        if plot_graph:
            plt.figure(figsize=(12, 6))

            # Plot training data
            plt.plot(range(0, n_train_length), train_prices, label='Training Data', color='blue')

        # Plot actual future values
        future_indices = np.arange(n_train_length, n_train_length + n_forecast_length)

        # Collect forecasts from top N models
        top_n = min(n_models_to_keep, len(sorted_data_from_workers))
        all_forecasts = []
        for i in range(top_n):
            pred = sorted_data_from_workers[i]['pred_values']
            all_forecasts.append(pred)
            if plot_graph:
                # Light gray lines for individual forecasts
                plt.plot(future_indices, pred, color='lightgray', alpha=0.6, linewidth=1)

        # Compute and plot mean forecast
        all_forecasts = np.array(all_forecasts)  # shape: (top_n, n_forecast_length)
        mean_forecast = np.mean(all_forecasts, axis=0)
        if real_time:
            if use_given_gt_truth is None:
                gt_prices = mean_forecast.copy()
            else:
                gt_prices = use_given_gt_truth
            misc_returned[step_back]['mean_forecast'] = mean_forecast.copy()
        # Plot actual future values
        future_indices = np.arange(n_train_length, n_train_length + n_forecast_length)
        if plot_graph:
            plt.plot(future_indices, gt_prices, label='Actual', marker='o', color='green', linewidth=2)
            # Add vertical line separating training and forecast
            plt.axvline(x=n_train_length, color='black', linestyle='--', linewidth=1, label='Train / Forecast Boundary')

        # ==============================
        # ENHANCED ROBUSTNESS METRICS
        # ==============================

        # 1. Forecast dispersion (normalized std)
        forecast_std = np.std(all_forecasts, axis=0)
        mean_std = np.mean(forecast_std)
        normalized_std = mean_std / (np.mean(np.abs(mean_forecast)) + 1e-8)

        # 2. RMSE consistency within top-N
        best_rmse_val = sorted_data_from_workers[0]['rmse']
        nth_rmse_val = sorted_data_from_workers[top_n - 1]['rmse']
        relative_rmse_gap = (nth_rmse_val - best_rmse_val) / (best_rmse_val + 1e-8)
        rmse_consistency = 1.0 / (1.0 + relative_rmse_gap)

        # 3. Directional (slope) consensus
        if n_forecast_length > 1:
            ensemble_slope = np.polyfit(np.arange(n_forecast_length), mean_forecast, 1)[0]
            individual_slopes = np.array([
                np.polyfit(np.arange(n_forecast_length), pred, 1)[0]
                for pred in all_forecasts
            ])
            directional_consensus = np.mean(np.sign(individual_slopes) == np.sign(ensemble_slope))
        else:
            directional_consensus = 1.0

        # # 4. Decision consensus (using YOUR rules)
        # decision_labels = []
        # for pred in all_forecasts:
        #     pred_slope = np.polyfit(np.arange(len(pred)), pred, 1)[0]
        #     pred_is_between = (pred > lower_line).all() and (pred < upper_line).all()
        #
        #     # Call conditions (mirroring your logic)
        #     call_cond = (
        #             (pred[0] > lower_line and np.any(pred < lower_line) and pred_slope < 0) or
        #             (pred[0] > lower_line > pred[-1] and pred_slope < 0) or
        #             (pred[0] < lower_line and not np.any(pred > lower_line) and pred_slope < 0) or
        #             (pred_is_between and pred_slope < 0 and shape_similarity > threshold_for_shape_similarity)
        #     )
        #
        #     # Put conditions
        #     put_cond = (
        #             (pred_slope > 0 and pred_is_between) or
        #             (pred_slope > 0 and pred[0] > lower_line and pred[-1] > upper_line) or
        #             (pred_slope > 0 and pred[-1] > upper_line) or
        #             (pred_is_between and pred_slope > 0 and shape_similarity > threshold_for_shape_similarity)
        #     )
        #
        #     if call_cond:
        #         decision_labels.append('call')
        #     elif put_cond:
        #         decision_labels.append('put')
        #     else:
        #         decision_labels.append('?')
        #
        # # Determine current ensemble decision (reuse your existing flags)
        # ensemble_decision = 'call' if sell__call_credit_spread else ('put' if sell__put_credit_spread else '?')
        # decision_consensus = np.mean([d == ensemble_decision for d in decision_labels])
        if plot_graph:
            plt.plot(future_indices, mean_forecast, label='Mean Forecast', marker='p', color='red', linewidth=2.5)
        assert len(mean_forecast) == len(gt_prices)

        # >>> SHAPE SIMILARITY METRIC <<<
        if top_n > 1:
            corr_matrix = np.corrcoef(all_forecasts)
            triu_indices = np.triu_indices_from(corr_matrix, k=1)
            mean_pairwise_corr = np.mean(corr_matrix[triu_indices]) if len(triu_indices[0]) > 0 else 1.0
        else:
            mean_pairwise_corr = 1.0
        shape_similarity = mean_pairwise_corr
        shape_similarity = 0. if np.isnan(shape_similarity) else shape_similarity
        # 5. Composite robustness score (0–1)
        norm_shape_sim = np.clip(shape_similarity, 0.0, 1.0)
        norm_low_disp = np.clip(1.0 - normalized_std, 0.0, 1.0)
        norm_rmse_cons = np.clip(rmse_consistency, 0.0, 1.0)
        norm_dir_cons = np.clip(directional_consensus, 0.0, 1.0)
        # norm_dec_cons = np.clip(decision_consensus, 0.0, 1.0)

        robustness_score = np.mean([
            norm_shape_sim,
            norm_low_disp,
            norm_rmse_cons,
            norm_dir_cons,
            # norm_dec_cons
        ])
        # Update plot title to show robustness
        robustness_str = f", Robust: {robustness_score:.2f}"

        # Add horizontal lines based on last training price
        threshold_up_ep, threshold_down_ep = thresholds_ep[0], thresholds_ep[1]
        last_train_price = train_prices[-1]
        upper_line = last_train_price * (1. + threshold_up_ep)  # th1
        upper_line = math.ceil(upper_line / floor_and_ceil) * floor_and_ceil
        lower_line = last_train_price * (1. - threshold_down_ep)  # th2
        lower_line = math.floor(lower_line / floor_and_ceil) * floor_and_ceil
        if plot_graph:
            plt.axhline(y=upper_line, color='orange', linestyle='--', alpha=0.6, linewidth=1, label=f"+{threshold_up_ep*100:.2f}% / -{threshold_down_ep*100:.2f}%")
            plt.axhline(y=lower_line, color='orange', linestyle='--', alpha=0.6, linewidth=1)

            # Add bold, black text labels for upper_line and lower_line
            plt.text(
                x=n_train_length + n_forecast_length - 4.5,  # slightly left of right edge
                y=upper_line,
                s=f"{upper_line:.2f}",
                fontsize=20,
                fontweight='bold',
                color='black',
                verticalalignment='center',
                horizontalalignment='right'
            )
            plt.text(
                x=n_train_length + n_forecast_length - 9.5,
                y=lower_line,
                s=f"{lower_line:.2f}",
                fontsize=20,
                fontweight='bold',
                color='black',
                verticalalignment='center',
                horizontalalignment='right'
            )

        th1, th2, entry_price = upper_line, lower_line, last_train_price
        assert th1 > entry_price > th2 and th1 > th2
        sell__call_credit_spread, sell__put_credit_spread, nope__= False, False, False
        mslope_pred = np.polyfit(np.arange(len(mean_forecast)), mean_forecast, 1)[0] if n_forecast_length > 1 else 0.
        is_between_th1_and_th2 = len(mean_forecast) == np.count_nonzero(mean_forecast < th1) and len(mean_forecast) == np.count_nonzero(mean_forecast > th2)
        is_slope_neg            = mslope_pred < 0
        is_slope_pos            = mslope_pred > 0
        is_no_slope             = mslope_pred == 0
        is_shape_sim_high       = shape_similarity > threshold_for_shape_similarity
        if n_forecast_length > 1:
            #######################################################################
            # Rules to Sell Call Credit Spread
            #######################################################################
            # starts above th2 + goes below th2 + ends above th2 + neg slope
            r1 = mean_forecast[0] > th2 and 0 != np.count_nonzero(mean_forecast < th2) and is_slope_neg
            # starts above th2 + ends below th2 + neg slope
            r2 = mean_forecast[0] > th2 > mean_forecast[-1] and is_slope_neg
            # starts below th2, stays below th2 + neg slope
            r3 = mean_forecast[0] < th2 and 0 == np.count_nonzero(mean_forecast > th2) and is_slope_neg
            # stays between th1 and th2 + neg slope + high sim
            r4 = is_between_th1_and_th2 and is_slope_neg and is_shape_sim_high
            if r1 or r2 or r3 or r4:
                sell__call_credit_spread = True
            #######################################################################
            # Rules to Sell Put Credit Spread
            #######################################################################
            # stays between th1 and th2, with positive slope
            s1 = is_slope_pos and is_between_th1_and_th2
            # starts above th2, ends above th1, with positive slope
            s2 = is_slope_pos and mean_forecast[0]  > th2 and mean_forecast[-1] > th1
            # starts above th1, ends above th1, with positive slope
            s3 = is_slope_pos and mean_forecast[-1] > th1 and mean_forecast[-1] > th1
            # stays between th1 and th2 + pos slope + high sim
            s4 = is_between_th1_and_th2 and is_slope_pos and is_shape_sim_high
            if s1 or s2 or s3 or s4:
                sell__put_credit_spread = True
            #######################################################################
            # Not sure
            #######################################################################
            z1 = not sell__call_credit_spread and not sell__put_credit_spread
            if z1:
                nope__ = True
            assert ((sell__call_credit_spread and not sell__put_credit_spread and not nope__) or
                    (sell__put_credit_spread and not sell__call_credit_spread and not nope__) or
                    (nope__ and not sell__call_credit_spread and not sell__put_credit_spread))
            if not sell__put_credit_spread and not sell__call_credit_spread:
                description_of_what_user_shall_do[step_back]['op'] = {'action': 'do_nothing'}
                description_of_what_user_shall_do[step_back]['description'] = f"Do no trade this setup"
            else:
                if sell__put_credit_spread:
                    th2_rounded = math.floor(th2 / floor_and_ceil) * floor_and_ceil
                    distance_for_protection = maintenance_margin // 100
                    description_of_what_user_shall_do[step_back]['description'] = f"Do a Vertical Put Credit Spread:"
                    description_of_what_user_shall_do[step_back]['description'] += f" Write Put @{th2_rounded}$ with protection @{th2_rounded - distance_for_protection}$"
                    description_of_what_user_shall_do[step_back]['op'] = {'action': 'vertical_put', 'sell1': th2_rounded, 'buy1': th2_rounded - distance_for_protection}
                if sell__call_credit_spread:
                    th1_ceiled = math.ceil(th1 / floor_and_ceil) * floor_and_ceil
                    distance_for_protection = maintenance_margin // 100
                    description_of_what_user_shall_do[step_back]['description'] = f"Do a Vertical Call Credit Spread:"
                    description_of_what_user_shall_do[step_back]['description'] += f" Write Call @{th1_ceiled}$ with protection @{th1_ceiled + distance_for_protection}$"
                    description_of_what_user_shall_do[step_back]['op'] = {'action': 'vertical_call', 'sell1': th1_ceiled, 'buy1': th1_ceiled + distance_for_protection}
        else: # Just have a point.
            assert 1 == len(mean_forecast) and 1 == len(gt_prices) and th1 > th2
            # Prediction is between th1 and th2 --> Iron Condor
            if th2 < mean_forecast[0] < th1:
                # For an iron condor to work, closing price shall be between th1 and th2
                if th2 < gt_prices[0] < th1:
                    performance_tracking_1_point_prediction['iron_condor_0DTE']['success'].append(step_back)
                else:
                    performance_tracking_1_point_prediction['iron_condor_0DTE']['failure'].append(step_back)
                th2_rounded = math.floor(th2 / floor_and_ceil) * floor_and_ceil
                th1_ceiled  = math.ceil(th1 / floor_and_ceil) * floor_and_ceil
                distance_for_protection = maintenance_margin // 2 // 100
                description_of_what_user_shall_do[step_back]['description']  = f"Do an Iron Condor:"
                description_of_what_user_shall_do[step_back]['description'] += f" Write Put @{th2_rounded}$ with protection @{th2_rounded - distance_for_protection}$"
                description_of_what_user_shall_do[step_back]['description'] += f" Write Call @{th1_ceiled}$ with protection @{th1_ceiled  + distance_for_protection}$"
                description_of_what_user_shall_do[step_back]['op'] = {'action': 'iron_condor',
                                                                      'sell1': th2_rounded, 'buy1': th2_rounded - distance_for_protection,
                                                                      'sell2': th1_ceiled,  'buy2': th1_ceiled + distance_for_protection}
            # Prediction is above th1 --> sell an Credit Put Spread @ th2
            if mean_forecast[0] > th1:
                if gt_prices[0] > th2:
                    performance_tracking_1_point_prediction['put_credit_spread']['success'].append(step_back)
                else:
                    performance_tracking_1_point_prediction['put_credit_spread']['failure'].append(step_back)
                th2_rounded = math.floor(th2 / floor_and_ceil) * floor_and_ceil
                distance_for_protection = maintenance_margin // 100
                description_of_what_user_shall_do[step_back]['description'] = f"Do a Vertical Put Credit Spread:"
                description_of_what_user_shall_do[step_back]['description'] += f" Write Put @{th2_rounded}$ with protection @{th2_rounded - distance_for_protection}$"
                description_of_what_user_shall_do[step_back]['op'] = {'action': 'vertical_put', 'sell1': th2_rounded, 'buy1': th2_rounded - distance_for_protection}
            # Prediction is below th2 --> sell an Call Put Spread @ th1
            if mean_forecast[0] < th2:
                if gt_prices[0] < th1:
                    performance_tracking_1_point_prediction['call_credit_spread']['success'].append(step_back)
                else:
                    performance_tracking_1_point_prediction['call_credit_spread']['failure'].append(step_back)
                th1_ceiled = math.ceil(th1 / floor_and_ceil) * floor_and_ceil
                distance_for_protection = maintenance_margin // 100
                description_of_what_user_shall_do[step_back]['description'] = f"Do a Vertical Call Credit Spread:"
                description_of_what_user_shall_do[step_back]['description'] += f" Write Call @{th1_ceiled}$ with protection @{th1_ceiled + distance_for_protection}$"
                description_of_what_user_shall_do[step_back]['op'] = {'action': 'vertical_call', 'sell1': th1_ceiled, 'buy1': th1_ceiled + distance_for_protection}
        assert gt_prices.shape == mean_forecast.shape
        #######################################################################
        # Compute statistic based on the real price of the asset
        #######################################################################
        if sell__call_credit_spread:
            condition_call = (gt_prices - th1) > 0
            indices_call = np.where(condition_call)[0]  # or np.nonzero(condition_call)[0]
            number_of_times_real_price_goes_above_call_strike_price = len(indices_call)
            performance_tracking['call'].append((step_back, number_of_times_real_price_goes_above_call_strike_price, [int(b) for b in indices_call]))
            if strategy_for_exit == 'hold_until_the_end':
                # if last price is OTM, it is a success
                if gt_prices[-1] < th1:
                    performance_tracking_xtm['call']['success'].append((step_back, gt_prices[-1], th1))
                else:
                    performance_tracking_xtm['call']['failure'].append((step_back, gt_prices[-1], th1))
            if strategy_for_exit == 'hold_until_the_end_with_roll':
                # Look at t-2, if we decide to roll the position
                is_roll_being_requested = True if gt_prices[-2] > th1 else False
                if is_roll_being_requested:
                    # Roll it
                    price_on_tm2 = gt_prices[-2]
                    new_th1 = price_on_tm2 * (1. + threshold_up_ep)  # th1
                    new_th1 = math.ceil(new_th1 / floor_and_ceil) * floor_and_ceil
                    if gt_prices[-1] < new_th1:
                        performance_tracking_xtm['call']['success'].append((step_back, gt_prices[-1], th1, new_th1))
                    else:
                        performance_tracking_xtm['call']['failure'].append((step_back, gt_prices[-1], th1, new_th1))
                else:
                    if gt_prices[-1] < th1:
                        performance_tracking_xtm['call']['success'].append((step_back, gt_prices[-1], th1, None))
                    else:
                        performance_tracking_xtm['call']['failure'].append((step_back, gt_prices[-1], th1, None))
        if sell__put_credit_spread:
            condition_put = (th2 - gt_prices) > 0
            indices_put = np.where(condition_put)[0]  # or np.nonzero(condition_put)[0]
            number_of_times_real_price_goes_below_put_strike_price = len(indices_put)
            performance_tracking['put'].append((step_back, number_of_times_real_price_goes_below_put_strike_price, [int(b) for b in indices_put]))
            if strategy_for_exit == 'hold_until_the_end':
                # if last price is OTM, it is a success
                if gt_prices[-1] > th2:
                    performance_tracking_xtm['put']['success'].append((step_back, gt_prices[-1], th2))
                else:
                    performance_tracking_xtm['put']['failure'].append((step_back, gt_prices[-1], th2))
            if strategy_for_exit == 'hold_until_the_end_with_roll':
                # Look at t-2, if we decide to roll the position
                is_roll_being_requested = True if gt_prices[-2] < th2 else False
                if is_roll_being_requested:
                    # Roll it
                    price_on_tm2 = gt_prices[-2]
                    new_th2 = price_on_tm2 * (1. - threshold_down_ep)  # th2
                    new_th2 = math.floor(new_th2 / floor_and_ceil) * floor_and_ceil
                    if gt_prices[-1] > new_th2:
                        performance_tracking_xtm['put']['success'].append((step_back, gt_prices[-1], th1, new_th2))
                    else:
                        performance_tracking_xtm['put']['failure'].append((step_back, gt_prices[-1], th1, new_th2))
                else:
                    if gt_prices[-1] > th2:
                        performance_tracking_xtm['put']['success'].append((step_back, gt_prices[-1], th2, None))
                    else:
                        performance_tracking_xtm['put']['failure'].append((step_back, gt_prices[-1], th2, None))
        if nope__:
            performance_tracking['?'].append((step_back, 9999, []))

        mean_rmse = np.sqrt(np.mean((mean_forecast - gt_prices) ** 2))
        true_diff = np.diff(gt_prices)
        pred_diff = np.diff(mean_forecast)
        # Handle zero differences by treating them as no change (direction = 0)
        true_direction = np.sign(true_diff)
        pred_direction = np.sign(pred_diff)
        mean_directional_accuracy = np.mean(true_direction == pred_direction)

        t1, t2, t3, t4 = best_rmse['times']  # It all the same time frame for all "n_models_to_keep" models
        # Format dates compactly (e.g., "YYMMDD")
        time_str = f"[{t1.strftime('%Y%m%d')}-{t2.strftime('%m%d')}→{t3.strftime('%Y%m%d')}-{t4.strftime('%m%d')}]"
        # Title info (using best model for metadata)
        da_str = f", DA: {mean_directional_accuracy:.2%}"
        if plot_graph:
            plt.title(f'{ticker} Forecast ({args.dataset_id}) — '
                      f'Mean RMSE: {mean_rmse:.2f}{da_str} | '
                      f'Shape Sim: {shape_similarity:.2f} | Robust: {robustness_score:.2f} |\n'
                      f'{top_n} Models Shown   Step Back:{step_back}   {time_str}  {algo_name}', fontsize=12)
            plt.xlabel('Time Index')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

        if plot_graph and show_graph:
            plt.show()
        if save_to_disk:
            my_output_dir = os.path.join(output_dir, "images")
            os.makedirs(my_output_dir, exist_ok=True)
            if plot_graph:
                plt.savefig(os.path.join(my_output_dir, f'{step_back}___E{mean_rmse:.2f}_D{mean_directional_accuracy:.2f}___{ticker}_ensemble_forecast_plot.png'), dpi=300)
        try:
            if close_graph:
                if plot_graph and show_graph:
                    plt.close()
        except:
            pass
    # === Performance Tracking Summary ===
    if n_forecast_length > 1:
        # with open(os.path.join(output_dir, "breach.txt"), "w") as f:
        #     json.dump(performance_tracking, f, indent=2)
        # call_events = performance_tracking['call']
        # put_events  = performance_tracking['put']
        nope_events = performance_tracking['?']
        # total_call_signals = len(call_events)
        # total_put_signals  = len(put_events)
        # total_nope_signals = len(nope_events)
        # call_breaches = sum(1 for _, breaches, idx in call_events if breaches > 0)
        # put_breaches  = sum(1 for _, breaches, idx in put_events if breaches > 0)
        # print("\n" + "=" * 60)
        # print("PERFORMANCE TRACKING SUMMARY".center(60))
        # print("=" * 60)
        # print(f"Call Credit Spread Signals : {total_call_signals}")
        # if 0 != total_call_signals:
        #     print(f"  → Price breached call strike : {call_breaches} times ({call_breaches / total_call_signals:.1%})")
        # print()
        # print(f"Put  Credit Spread Signals : {total_put_signals}")
        # if 0 != total_put_signals:
        #     print(f"  → Price breached put strike  : {put_breaches} times ({put_breaches / total_put_signals:.1%})")
        # print()
        # print(f"Nope  Signals : {total_nope_signals}")
        # print()
        # total_signals = total_call_signals + total_put_signals
        # total_breaches = call_breaches + put_breaches
        # sdfg = f"({total_breaches / total_signals:.1%})" if total_signals > 0 else ""
        # print(f"Overall breach rate          : {total_breaches}/{total_signals} {sdfg}")
        # print("=" * 60)
        # # Identify out_of_breach step_back values
        # call_out_of_breach = [step for step, breaches, idx in call_events if breaches != 0]
        # put_out_of_breach  = [step for step, breaches, idx in put_events  if breaches != 0]
        nope_out_of_breach = [step for step, _, _ in nope_events]  # all are "out of breach" by definition
        if verbose:
            print("\n" + "=" * 60)
            print("OUT-OF-BREACH step_back VALUES".center(60))
            print("=" * 60)
            # print(f"Call  (breach)       : {call_out_of_breach}")
            # print(f"Put   (breach)       : {put_out_of_breach}")
            print(f"?     (skipped trade): {nope_out_of_breach}")
            # print("=" * 60)
            # out_of_breach_summary = {
            #     'call_out_of_breach': call_out_of_breach,
            #     'put_out_of_breach': put_out_of_breach,
            #     'nope_steps': nope_out_of_breach
            # }
            # with open(os.path.join(output_dir, "out_of_breach.txt"), "w") as f:
            #     json.dump(out_of_breach_summary, f, indent=2)

        # Identify puts and calls that ended OTM
        call_otm = [step[0] for step in performance_tracking_xtm['call']['success']]
        put_otm  = [step[0] for step in performance_tracking_xtm['put']['success']]
        call_itm = [step[0] for step in performance_tracking_xtm['call']['failure']]
        put_itm  = [step[0] for step in performance_tracking_xtm['put']['failure']]
        call_rolled = [step for step in performance_tracking_xtm['call']['success'] if len(step) > 3 and step[3] is not None]
        put_rolled  = [step for step in performance_tracking_xtm['put']['success']  if len(step) > 3 and step[3] is not None]
        if verbose:
            print("\n" + "=" * 60)
            print("OUT-OF-MONEY CALLs and PUTs".center(60))
            print("=" * 60)
            print(f"Call  (OTM): {call_otm if len(call_otm) < 25 else f'#{len(call_otm)}'}")
            print(f"Put   (OTM): {put_otm if len(put_otm) < 25 else f'#{len(put_otm)}'}")
            print(f"Call  (ITM): {call_itm if len(call_itm) < 25 else f'#{len(call_itm)}'}")
            print(f"Put   (ITM): {put_itm if len(put_itm) < 25 else f'#{len(put_itm)}'}")
            print(f"Call Rolled: {len(call_rolled)}")
            print(f"Put  Rolled: {len(put_rolled)}")
            print("=" * 60)
        otm_summary = {
            'call_otm': call_otm,
            'put_otm': put_otm,
        }
        with open(os.path.join(output_dir, "otm.txt"), "w") as f:
            json.dump(otm_summary, f, indent=2)

        if verbose:
            print("\n" + "=" * 60)
            print(f"List of fails - based on strategy '{strategy_for_exit}'".center(60))
            print("=" * 60)
        failed_trades = []
        for step_back in range(0, number_of_step_back):
            if step_back in nope_out_of_breach:
                continue  # No trade
            if step_back in call_otm or step_back in put_otm:
                continue  # Trade success
            failed_trades.append(step_back)
        if verbose:
            print(f"Failed: {failed_trades}")
            print("=" * 60)
        failed_summary = {'failed_trades': failed_trades,}
        with open(os.path.join(output_dir, "failed_trades.txt"), "w") as f:
            json.dump(failed_summary, f, indent=2)

        total_number_of_possible_trade = number_of_step_back
        total_number_of_winning_trade  = number_of_step_back - len(failed_trades)
        total_number_of_loosing_trade  = len(failed_trades)
        if verbose:
            print("\n" + "=" * 60)
            print(f"Succes rate is {total_number_of_winning_trade/total_number_of_possible_trade*100:0.1f}%".center(60))
            print("=" * 60)
    else:
        if verbose:
            # Nicely formatted output for 1-point prediction performance
            print("\n" + "=" * 70)
            print("1-POINT PREDICTION PERFORMANCE SUMMARY".center(70))
            print("=" * 70)
        total_number_of_winning_trade = 0
        total_number_of_loosing_trade = 0
        for strategy, results in performance_tracking_1_point_prediction.items():
            success = results['success']
            failure = results['failure']
            total = len(success) + len(failure)
            success_rate = (len(success) / total * 100) if total > 0 else 0.0
            total_number_of_winning_trade += len(success)
            total_number_of_loosing_trade += len(failure)
            if verbose:
                print(f"\n{strategy.replace('_', ' ').title()}:")
                print(f"  ✅ Success: {len(success):3d} {success}")
                print(f"  ❌ Failure: {len(failure):3d} {failure}")
                print(f"  🎯 Success Rate: {success_rate:5.1f}%  (out of {total} trades)")
        if verbose:
            print("\n" + "=" * 70)
        total_number_of_possible_trade = number_of_step_back
    return total_number_of_possible_trade, total_number_of_winning_trade, total_number_of_loosing_trade, description_of_what_user_shall_do, parameters_best_models, misc_returned


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Wavelet-based stock forecast.")
    parser.add_argument("--ticker", type=str, default='^GSPC', choices=['^GSPC'])  # Need to modify maintenance margin and so on if we change the stock
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="")
    parser.add_argument("--dataset_id", type=str, default='day', choices=['day', 'week', 'month'])
    parser.add_argument("--output_dir", type=str, default=r"../../stubs/wavelet_opt/")
    parser.add_argument("--number_of_step_back", type=int, default=2605)
    parser.add_argument("--n_forecast_length", type=int, default=2)
    parser.add_argument("--n_forecast_length_in_training", type=int, default=2)
    parser.add_argument("--maxmin_sequence_for_train_length", type=str, default="(4,64)")
    parser.add_argument("--floor_and_ceil", type=float, default=5.)
    parser.add_argument("--maintenance_margin", type=float, default=2000)
    parser.add_argument("--algorithms_to_run", type=str, default="0,1,2")
    parser.add_argument("--n_forecasts", type=int, default=19)
    parser.add_argument("--n_models_to_keep", type=int, default=60)
    parser.add_argument("--use_this_df", type=json.loads, default={})
    parser.add_argument("--plot_graph", type=bool, default=False)
    parser.add_argument("--show_graph", type=bool, default=True)
    parser.add_argument("--save_graph", type=bool, default=True)
    parser.add_argument("--real_time", type=bool, default=False)
    parser.add_argument("--display_tqdm", type=bool, default=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--do_not_close_graph", type=bool, default=False)
    parser.add_argument("--thresholds_ep", type=str, default="(0.025, 0.02)")
    parser.add_argument("--use_given_gt_truth", type=str, default=None)
    parser.add_argument("--temp_filename", type=str, required=True)
    parser.add_argument("--strategy_for_exit", type=str, default=r"hold_until_the_end_with_roll")
    args = parser.parse_args()

    from constants import FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_DAY, FYAHOO__OUTPUTFILENAME_MONTH
    if args.dataset_id == 'day':
        df_filename = FYAHOO__OUTPUTFILENAME_DAY
    elif args.dataset_id == 'week':
        df_filename = FYAHOO__OUTPUTFILENAME_WEEK
    elif args.dataset_id == 'month':
        df_filename = FYAHOO__OUTPUTFILENAME_MONTH
    older_dataset = None if args.older_dataset == "None" else args.older_dataset
    one_dataset_filename = None
    if args.use_this_df is None or 0 == len(args.use_this_df):
        one_dataset_filename = df_filename if older_dataset is None else transform_path(df_filename, older_dataset)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    args.master_data_cache = master_data_cache.copy()

    main(args)