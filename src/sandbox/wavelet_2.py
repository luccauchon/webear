import os
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from constants import FYAHOO__OUTPUTFILENAME_WEEK
import pickle
import pywt
from constants import NB_WORKERS
import psutil
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from itertools import chain
import warnings

warnings.filterwarnings("ignore", message="Level value.*too high.*")


def wavelet_forecast(
    prices,
    forecast_steps: int = 5,
    wavelet: str = 'db4',
    level: int = 3
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
                forecast_values = wavelet_forecast(
                    prices=train_prices,
                    forecast_steps=n_forecast_length,
                    wavelet=wavlet_type,
                    level=level
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
            all_results_computed.append(tmp)
    all_results_computed.sort(key=lambda x: x['rmse'])  # ascending: best (lowest RMSE) first
    # Get top NNN results (as a list)
    top_results = all_results_computed[:NNN]
    # If you really need a dictionary (e.g., indexed by rank), you can do:
    all_results_from_worker = {i: result for i, result in enumerate(top_results)}
    out__shared.put((best_rmse, all_results_from_worker))


def main():
    ticker = "^GSPC"
    df_filename = FYAHOO__OUTPUTFILENAME_WEEK
    _nb_workers = NB_WORKERS
    plot=False
    save_to_disk = True
    close_col = ('Close', ticker)
    n_forecast_length = 4
    n_models_to_keep = 60
    for step_back in tqdm(range(0, 2000)):
        use_cases = []
        with open(df_filename, 'rb') as f:
            data_cache = pickle.load(f)
        data = data_cache[ticker]
        this_year = data[:-(step_back+1)].index[-1].year
        print(this_year)
    sys.exit(0)
    for step_back in tqdm(range(0, 2000)):
        use_cases = []
        with open(df_filename, 'rb') as f:
            data_cache = pickle.load(f)
        data = data_cache[ticker]
        this_year = data[:-(step_back+1)].index[-1].year
        typical_price = data[data.index.year==this_year][close_col].mean()
        RMSE_TOL = max(0.5, 0.0001 * typical_price)
        for n_train_length in range(8, 32):
            for level in range(1, 8):
                for wavlet_type in pywt.wavelist(kind='discrete'):
                    use_cases.append({'n_train_length': n_train_length, 'level': level, 'wavlet_type': wavlet_type, 'n_forecast_length': n_forecast_length,
                                      'close_col': close_col, 'RMSE_TOL': RMSE_TOL, 'step_back': step_back})
        # print(f"[{step_back=}] Evaluating {len(use_cases)} experiences with {_nb_workers} workers...")
        use_cases__shared, master_cmd__shared = Queue(len(use_cases)), Value("i", 0)
        out__shared = [Queue(1) for k in range(0, _nb_workers)]
        # Lancement des workers
        for k in range(0, _nb_workers):
            p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k], ticker, df_filename))
            p.start()
            pid = p.pid
            p_obj = psutil.Process(pid)
            p_obj.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
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
        if plot or save_to_disk:
            best_rmse = sorted_data_from_workers[0]

            n_train_length = best_rmse['n_train_length']
            train_prices = best_rmse['prices']
            gt_prices = best_rmse['gt_prices']
            n_forecast_length = best_rmse['n_forecast_length']

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

            # Add ±2.5% horizontal lines based on last training price
            threshold_ep = 0.025
            last_train_price = train_prices[-1]
            upper_line = last_train_price * (1. + threshold_ep)
            lower_line = last_train_price * (1. - threshold_ep)
            plt.axhline(y=upper_line, color='orange', linestyle='--', alpha=0.6, linewidth=1, label=f"+/-{2.5}%")
            plt.axhline(y=lower_line, color='orange', linestyle='--', alpha=0.6, linewidth=1)
            th1, th2, etry_price = upper_line, lower_line, last_train_price
            # mslope_pred, _ = np.polyfit(np.arange(len(mean_forecast)), mean_forecast, 1)
            # Rules to Sell Call Credit Spread
            # sell__call_credit_spread = False
            # starts above th2 + goes below th2 + ends above th2 + neg slope
            # r1 = mean_forecast[0] > th2 and 0 != np.count_nonzero(mean_forecast < th2) and mslope_pred < 0
            # starts above th2 + ends below th2 + neg slope
            # r2 = mean_forecast[0] > th2 and mean_forecast[-1] < th2 and mslope_pred < 0:
            # starts below th2, stays below th2, with neg slope
            # r3 = mean_forecast[0] < th2 and 0 == np.count_nonzero(mean_forecast > th2)
            #
            # Not sure
            # stays between th1 and th2, with negative slope
            # len(mean_forecast) == np.count_nonzero(th1 < mean_forecast < th2) and mslope_pred < 0
            #
            #
            # Rules to Sell Put Credit Spread
            # if mslope_pred > 0 and len(mean_forecast) == np.count_nonzero(th1 < mean_forecast < th2)
            # if mslope_pred > 0 and mean_forecast[0]  > th2 and mean_forecast[-1] > th1
            # if mslope_pred > 0 and mean_forecast[-1] > th1 and mean_forecast[-1] > th1

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
            plt.title(f'{ticker} Forecast ({Path(FYAHOO__OUTPUTFILENAME_WEEK).stem.upper()}) — '
                      f'Mean RMSE: {mean_rmse:.2f}{da_str} | '
                      f'Shape Sim: {shape_similarity:.2f} | \n'
                      f'{top_n} Models Shown   Step Back:{step_back}   {time_str}', fontsize=12)
            plt.xlabel('Time Index')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            if plot:
                plt.show()
            if save_to_disk:
                output_dir = "images_2"
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f'{step_back}___E{mean_rmse:.2f}_D{mean_directional_accuracy:.2f}___{ticker}_ensemble_forecast_plot.png'), dpi=300)
            try:
                plt.close()
            except:
                pass

if __name__ == "__main__":
    freeze_support()
    main()