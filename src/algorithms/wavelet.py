import numpy as np
import pywt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import copy


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
    current = copy.deepcopy(coeff[-lag:])
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
                current_input = copy.deepcopy(coeff[-lag:])
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