import os
import argparse
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from constants import FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_DAY
import pickle
import pywt
from constants import NB_WORKERS, IS_RUNNING_ON_CASIR
import psutil
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import json
from itertools import chain
import warnings
warnings.filterwarnings("ignore", message="Level value.*too high.*")
# Or suppress only specific RuntimeWarnings from NumPy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import numpy as np
import pywt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


def forecast_coeff_series(coeff, forecast_len, lag=10):
    """
    Forecast a 1D coefficient series using recursive linear regression.
    """
    coeff = np.array(coeff, dtype=np.float64)
    n = len(coeff)

    lag = min(lag, max(3, n // 3))  # adaptive lag

    if n <= lag:
        # too short → extend with last value
        return np.concatenate([coeff, np.full(forecast_len, coeff[-1])])

    # Build lagged features
    X = np.array([coeff[j - lag:j] for j in range(lag, n)])
    y = coeff[lag:]

    model = LinearRegression()
    model.fit(X, y)

    # Recursive prediction
    forecast = []
    current = coeff[-lag:].copy()
    for _ in range(forecast_len):
        next_val = model.predict(current.reshape(1, -1))[0]
        forecast.append(next_val)
        current = np.append(current[1:], next_val)

    return np.concatenate([coeff, forecast])


def wavelet_multi_forecast__version_2(
        prices,
        forecast_steps=5,
        wavelet='db4',
        level=3,
        forecast_detail_levels=(3, 2)  # which detail levels to forecast: cD3, cD2
):
    prices = np.asarray(prices)
    n = len(prices)

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Wavelet decomposition
    coeffs = pywt.wavedec(scaled, wavelet, level=level)
    # coeffs = [cA, cD3, cD2, cD1]

    # Determine extension for each level
    # Lower frequency → fewer points needed
    ext_sizes = []
    for i in range(level + 1):
        # i=0 => cA
        ext_sizes.append(int(np.ceil(forecast_steps / (2 ** i))))

    forecasted_coeffs = []

    for i, c in enumerate(coeffs):
        if i == 0:
            # Approximation (always forecast)
            new_c = forecast_coeff_series(c, ext_sizes[i])
            forecasted_coeffs.append(new_c)

        elif (level - i + 1) in forecast_detail_levels:
            # Forecast selected detail levels
            new_c = forecast_coeff_series(c, ext_sizes[i])
            forecasted_coeffs.append(new_c)

        else:
            # Skip forecasting noisy high-frequency cD1 etc.
            forecasted_coeffs.append(c)

    # Align coeff lengths using a dummy signal
    dummy = np.zeros(n + forecast_steps)
    dummy_coeffs = pywt.wavedec(dummy, wavelet, level=level)

    aligned = []
    for fc, dc in zip(forecasted_coeffs, dummy_coeffs):
        if len(fc) < len(dc):
            fc = np.pad(fc, (0, len(dc) - len(fc)), mode='edge')
        else:
            fc = fc[:len(dc)]
        aligned.append(fc)

    # Reconstruct forecasted series
    forecasted = pywt.waverec(aligned, wavelet)

    expected_len = n + forecast_steps
    if len(forecasted) < expected_len:
        forecasted = np.pad(forecasted, (0, expected_len - len(forecasted)), mode='edge')

    forecasted = forecasted[:expected_len]

    # Denormalize
    forecasted = scaler.inverse_transform(forecasted.reshape(-1, 1)).flatten()

    return forecasted


def wavelet_forecast__version_1(
    prices,
    forecast_steps: int = 5,
    wavelet: str = 'db4',
    level: int = 3,
    forecast_detail_levels=None,
) -> np.ndarray:
    n = len(prices)

    # Normalize
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Wavelet decomposition
    coeffs = pywt.wavedec(scaled_prices, wavelet, level=level)

    # Determine how many points to add to approximation coefficients
    approx_extension = int(np.ceil(forecast_steps / (2 ** level)))

    forecasted_coeffs = []
    for i, coeff in enumerate(coeffs):
        coeff = np.array(coeff, dtype=np.float64)
        if i == 0:  # Approximation coefficients (trend)
            length = len(coeff)
            lag = min(10, max(1, length // 3))

            if length <= lag:
                # Not enough data: extend with last value
                extended = np.full(approx_extension, coeff[-1])
                new_coeff = np.concatenate([coeff, extended])
            else:
                # Build lagged features
                X = np.array([coeff[j - lag:j] for j in range(lag, length)])
                y = coeff[lag:]
                model = LinearRegression()
                model.fit(X, y)

                # Recursive forecasting
                current_input = coeff[-lag:].copy()
                forecast = []
                for _ in range(approx_extension):
                    next_val = model.predict(current_input.reshape(1, -1))[0]
                    forecast.append(next_val)
                    current_input = np.append(current_input[1:], next_val)
                new_coeff = np.concatenate([coeff, forecast])
            forecasted_coeffs.append(new_coeff)
        else:
            # Detail coefficients: DO NOT extend (keep original length)
            forecasted_coeffs.append(coeff)

    # After building forecasted_coeffs with extended cA and original cD's:
    # We need to make sure waverec won't fail due to length mismatch.

    # Create a dummy signal of desired length to get correct coeff lengths
    dummy_signal = np.zeros(n + forecast_steps)
    dummy_coeffs = pywt.wavedec(dummy_signal, wavelet, level=level)

    # Pad or truncate each coefficient to match dummy
    final_coeffs = []
    for orig_c, dummy_c in zip(forecasted_coeffs, dummy_coeffs):
        if len(orig_c) < len(dummy_c):
            padded = np.pad(orig_c, (0, len(dummy_c) - len(orig_c)), mode='edge')
        else:
            padded = orig_c[:len(dummy_c)]
        final_coeffs.append(padded)

    # Now reconstruct
    forecasted_signal = pywt.waverec(final_coeffs, wavelet)

    # Truncate or pad to exact desired length
    expected_length = n + forecast_steps
    if len(forecasted_signal) < expected_length:
        forecasted_signal = np.pad(
            forecasted_signal,
            (0, expected_length - len(forecasted_signal)),
            mode='edge'
        )
    forecasted_signal = forecasted_signal[:expected_length]

    # Denormalize
    forecast_original_scale = scaler.inverse_transform(
        forecasted_signal.reshape(-1, 1)
    ).flatten()

    return forecast_original_scale


###############################################################################
# NE JAMAIS CHANGE L'ORDRE
# JUSTE AJOUTER DES METHODES
###############################################################################
# Define a registry mapping algo names (strings) to actual functions
WAVELET_ALGO_REGISTRY = [wavelet_multi_forecast__version_2,  # 0
                         wavelet_forecast__version_1,  # 1
                        ]
###############################################################################
###############################################################################

def _worker_processor(use_cases__shared, master_cmd__shared, out__shared, ticker, data_filename):
    with open(data_filename, 'rb') as f:
        data_cache = pickle.load(f)
    data = data_cache[ticker]
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


def main(args):
    ticker = args.ticker
    if args.dataset_id == 'day':
        df_filename = FYAHOO__OUTPUTFILENAME_DAY
    elif args.dataset_id == 'week':
        df_filename = FYAHOO__OUTPUTFILENAME_WEEK
    _nb_workers = NB_WORKERS
    plot         = False
    save_to_disk = True
    index_of_algo = 0
    close_col = ('Close', ticker)
    n_forecast_length = int(args.n_forecast_length)
    n_models_to_keep = int(args.n_models_to_keep)
    performance_tracking     = {'put':[], 'call': [], '?': []}
    performance_tracking_xtm = {'put': [], 'call': []}
    thresholds_ep = eval(args.thresholds_ep)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    threshold_for_shape_similarity = 0.6
    number_of_step_back = int(args.number_of_step_back)
    sequence_for_train_length = range(8, 32)
    sequence_for_level = range(1, 8)
    for step_back in tqdm(range(0, number_of_step_back)):
        use_cases = []
        with open(df_filename, 'rb') as f:
            data_cache = pickle.load(f)
        data = data_cache[ticker]
        this_year = data[:-(step_back+1)].index[-1].year
        typical_price = data[data.index.year==this_year][close_col].mean()
        RMSE_TOL = max(0.5, 0.0001 * typical_price)
        for n_train_length in sequence_for_train_length:
            for level in sequence_for_level:
                for wavlet_type in pywt.wavelist(kind='discrete'):
                    use_cases.append({'n_train_length': n_train_length, 'level': level, 'wavlet_type': wavlet_type, 'n_forecast_length': n_forecast_length,
                                      'close_col': close_col, 'RMSE_TOL': RMSE_TOL, 'step_back': step_back, 'index_of_algo': index_of_algo})
        # print(f"[{step_back=}] Evaluating {len(use_cases)} experiences with {_nb_workers} workers...")
        use_cases__shared, master_cmd__shared = Queue(len(use_cases)), Value("i", 0)
        out__shared = [Queue(1) for k in range(0, _nb_workers)]
        # Lancement des workers
        for k in range(0, _nb_workers):
            p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k], ticker, df_filename))
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

        best_rmse = sorted_data_from_workers[0]

        n_train_length = best_rmse['n_train_length']
        train_prices = best_rmse['prices']
        gt_prices = best_rmse['gt_prices']
        n_forecast_length = best_rmse['n_forecast_length']
        algo_name = WAVELET_ALGO_REGISTRY[best_rmse['index_of_algo']].__name__
        # ---------------------------
        # Plotting All Forecasts + Mean
        # ---------------------------
        plt.figure(figsize=(12, 6))

        # Plot training data
        plt.plot(range(0, n_train_length), train_prices, label='Training Data', color='blue')

        # Plot actual future values
        future_indices = np.arange(n_train_length, n_train_length + n_forecast_length)
        plt.plot(future_indices, gt_prices, label='Actual', marker='o', color='green', linewidth=2)
        # Add vertical line separating training and forecast
        plt.axvline(x=n_train_length, color='black', linestyle='--', linewidth=1, label='Train / Forecast Boundary')

        # Collect forecasts from top 19 models
        top_n = min(n_models_to_keep, len(sorted_data_from_workers))
        all_forecasts = []
        for i in range(top_n):
            pred = sorted_data_from_workers[i]['pred_values']
            all_forecasts.append(pred)
            # Light gray lines for individual forecasts
            plt.plot(future_indices, pred, color='lightgray', alpha=0.6, linewidth=1)

        # Compute and plot mean forecast
        all_forecasts = np.array(all_forecasts)  # shape: (top_n, n_forecast_length)
        mean_forecast = np.mean(all_forecasts, axis=0)

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
        plt.plot(future_indices, mean_forecast, label='Mean Forecast', color='red', linewidth=2.5)
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
        upper_line = last_train_price * (1. + threshold_up_ep)
        lower_line = last_train_price * (1. - threshold_down_ep)
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
            x=n_train_length + n_forecast_length - 6.5,
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
        mslope_pred, _ = np.polyfit(np.arange(len(mean_forecast)), mean_forecast, 1)
        is_between_th1_and_th2 = len(mean_forecast) == np.count_nonzero(mean_forecast < th1) and len(mean_forecast) == np.count_nonzero(mean_forecast > th2)
        is_slope_neg            = mslope_pred < 0
        is_slope_pos            = mslope_pred > 0
        is_shape_sim_high       = shape_similarity > threshold_for_shape_similarity
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
        assert gt_prices.shape == mean_forecast.shape

        #######################################################################
        # Compute statistic based on the real price of the asset
        #######################################################################
        if sell__call_credit_spread:
            condition_call = (gt_prices - th1) > 0
            indices_call = np.where(condition_call)[0]  # or np.nonzero(condition_call)[0]
            number_of_times_real_price_goes_above_call_strike_price = len(indices_call)
            performance_tracking['call'].append((step_back, number_of_times_real_price_goes_above_call_strike_price, [int(b) for b in indices_call]))
            # if last price is OTM, it is a success
            if gt_prices[-1] < th1:
                performance_tracking_xtm['call'].append((step_back, gt_prices[-1], th1))
        if sell__put_credit_spread:
            condition_put = (th2 - gt_prices) > 0
            indices_put = np.where(condition_put)[0]  # or np.nonzero(condition_put)[0]
            number_of_times_real_price_goes_below_put_strike_price = len(indices_put)
            performance_tracking['put'].append((step_back, number_of_times_real_price_goes_below_put_strike_price, [int(b) for b in indices_put]))
            # if last price is OTM, it is a success
            if gt_prices[-1] > th2:
                performance_tracking_xtm['put'].append((step_back, gt_prices[-1], th2))
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
        plt.title(f'{ticker} Forecast ({Path(df_filename).stem.upper()}) — '
                  f'Mean RMSE: {mean_rmse:.2f}{da_str} | '
                  f'Shape Sim: {shape_similarity:.2f} | Robust: {robustness_score:.2f} |\n'
                  f'{top_n} Models Shown   Step Back:{step_back}   {time_str}  {algo_name}', fontsize=12)
        plt.xlabel('Time Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        if plot:
            plt.show()
        if save_to_disk:
            my_output_dir = os.path.join(output_dir, "images")
            os.makedirs(my_output_dir, exist_ok=True)
            plt.savefig(os.path.join(my_output_dir, f'{step_back}___E{mean_rmse:.2f}_D{mean_directional_accuracy:.2f}___{ticker}_ensemble_forecast_plot.png'), dpi=300)
        try:
            plt.close()
        except:
            pass
    # === Performance Tracking Summary ===
    with open(os.path.join(output_dir, "breach.txt"), "w") as f:
        json.dump(performance_tracking, f, indent=2)
    call_events = performance_tracking['call']
    put_events  = performance_tracking['put']
    nope_events = performance_tracking['?']
    total_call_signals = len(call_events)
    total_put_signals  = len(put_events)
    total_nope_signals = len(nope_events)
    call_breaches = sum(1 for _, breaches, idx in call_events if breaches > 0)
    put_breaches  = sum(1 for _, breaches, idx in put_events if breaches > 0)
    print("\n" + "=" * 60)
    print("PERFORMANCE TRACKING SUMMARY".center(60))
    print("=" * 60)
    print(f"Call Credit Spread Signals : {total_call_signals}")
    if 0 != total_call_signals:
        print(f"  → Price breached call strike : {call_breaches} times ({call_breaches / total_call_signals:.1%})")
    print()
    print(f"Put  Credit Spread Signals : {total_put_signals}")
    if 0 != total_put_signals:
        print(f"  → Price breached put strike  : {put_breaches} times ({put_breaches / total_put_signals:.1%})")
    print()
    print(f"Nope  Signals : {total_nope_signals}")
    print()
    total_signals = total_call_signals + total_put_signals
    total_breaches = call_breaches + put_breaches
    sdfg = f"({total_breaches / total_signals:.1%})" if total_signals > 0 else ""
    print(f"Overall breach rate          : {total_breaches}/{total_signals} {sdfg}")
    print("=" * 60)
    # Identify out_of_breach step_back values
    call_out_of_breach = [step for step, breaches, idx in call_events if breaches != 0]
    put_out_of_breach  = [step for step, breaches, idx in put_events  if breaches != 0]
    nope_out_of_breach = [step for step, _, _ in nope_events]  # all are "out of breach" by definition
    print("\n" + "=" * 60)
    print("OUT-OF-BREACH step_back VALUES".center(60))
    print("=" * 60)
    print(f"Call  (breach)       : {call_out_of_breach}")
    print(f"Put   (breach)       : {put_out_of_breach}")
    print(f"?     (skipped trade): {nope_out_of_breach}")
    print("=" * 60)
    out_of_breach_summary = {
        'call_out_of_breach': call_out_of_breach,
        'put_out_of_breach': put_out_of_breach,
        'nope_steps': nope_out_of_breach
    }
    with open(os.path.join(output_dir, "out_of_breach.txt"), "w") as f:
        json.dump(out_of_breach_summary, f, indent=2)

    # Identify puts and calls that ended OTM
    call_otm = [step[0] for step in performance_tracking_xtm['call']]
    put_otm  = [step[0] for step in performance_tracking_xtm['put']]
    print("\n" + "=" * 60)
    print("OUT-OF-MONEY CALLs and PUTs".center(60))
    print("=" * 60)
    print(f"Call  (OTM): {call_otm}")
    print(f"Put   (OTM): {put_otm}")
    print("=" * 60)
    otm_summary = {
        'call_otm': call_otm,
        'put_otm': put_otm,
    }
    with open(os.path.join(output_dir, "otm.txt"), "w") as f:
        json.dump(otm_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("List of fails - based on that the user keeps the trade open until the end".center(60))
    print("=" * 60)
    failed_trades = []
    for step_back in range(0, number_of_step_back):
        if step_back in nope_out_of_breach:
            continue  # No trade
        if step_back in call_otm or step_back in put_otm:
            continue  # Trade success
        failed_trades.append(step_back)
    print(f"Failed: {failed_trades}")
    print("=" * 60)
    failed_summary = {'failed_trades': failed_trades,}
    with open(os.path.join(output_dir, "failed_trades.txt"), "w") as f:
        json.dump(failed_summary, f, indent=2)

    total_number_of_possible_trade = number_of_step_back
    total_number_of_winning_trade  = number_of_step_back - len(failed_trades)
    total_number_of_loosing_trade  = len(failed_trades)
    print("\n" + "=" * 60)
    print(f"Succes rate is {total_number_of_winning_trade/total_number_of_possible_trade*100:0.1}%".center(60))
    print("=" * 60)
    return total_number_of_possible_trade, total_number_of_winning_trade, total_number_of_loosing_trade


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Wavelet-based stock forecast.")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="")
    parser.add_argument("--dataset_id", type=str, default='day', choices=['day', 'week'])
    parser.add_argument("--output_dir", type=str, default=r"../../stubs/wavelet_2/")
    parser.add_argument("--number_of_step_back", type=int, default=2605)
    parser.add_argument("--n_forecast_length", type=int, default=2)
    parser.add_argument("--algorithms_to_run", type=str, default="0,1,2")
    parser.add_argument("--n_forecasts", type=int, default=19)
    parser.add_argument("--n_models_to_keep", type=int, default=60)
    parser.add_argument("--use_this_df", type=json.loads, default={})
    parser.add_argument("--plot_graph", type=bool, default=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--thresholds_ep", type=str, default="(0.025, 0.02)")
    args = parser.parse_args()
    main(args)