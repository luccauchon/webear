import numpy as np
from scipy.stats import norm


def fourier_extrapolation_auto(prices, n_predict=20, energy_threshold=0.95, conf_level=None):
    """
    Fourier extrapolation with automatic harmonic selection and linear detrending.

    Parameters:
    - x: 1D array of real-valued signal (e.g., SPX closing prices)
    - n_predict: number of future steps to predict
    - energy_threshold: fraction of total spectral energy to retain (e.g., 0.95 = 95%)

    Returns:
    - full_forecast: original + predicted values (with trend restored)
    """
    n = len(prices)
    t = np.arange(n)

    # --- Step 1: Linear detrending ---
    # Fit y = a*t + b
    coeffs = np.polyfit(t, prices, 1)
    trend = coeffs[0] * t + coeffs[1]
    x_detrended = prices - trend

    # --- Step 2: FFT on detrended signal ---
    xf = np.fft.fft(x_detrended)
    power = np.abs(xf) ** 2  # power spectrum

    # Total energy (Parseval's theorem: sum |x|^2 = sum |X|^2 / n)
    total_energy = np.sum(power) / n

    # --- Step 3: Select harmonics to capture energy_threshold ---
    # Only consider non-negative frequencies (0 to n//2)
    # DC (k=0) and Nyquist (if n even, k=n//2) are unique; others come in pairs.
    cum_energy = 0.0
    n_harm = 0
    energy_target = energy_threshold * total_energy

    # Start with DC component
    cum_energy += power[0] / n
    if cum_energy >= energy_target:
        n_harm = 0
    else:
        # Add symmetric pairs (k and n-k) for k=1,2,...
        for k in range(1, n // 2 + 1):
            if k == n - k:  # Nyquist frequency (only when n even)
                pair_power = power[k]
            else:
                pair_power = power[k] + power[n - k]
            cum_energy += pair_power / n
            n_harm = k
            if cum_energy >= energy_target:
                break

    # --- Step 4: Reconstruct using selected harmonics ---
    xf_trunc = np.zeros(n, dtype=complex)
    xf_trunc[0] = xf[0]  # DC

    if n_harm > 0:
        xf_trunc[1:n_harm + 1] = xf[1:n_harm + 1]
        if n_harm < n // 2 or (n % 2 == 1):
            xf_trunc[-n_harm:] = xf[-n_harm:]
        elif n % 2 == 0 and n_harm == n // 2:
            # Include Nyquist if needed
            xf_trunc[n // 2] = xf[n // 2]

    # --- Step 5: Extrapolate in time ---
    t_extended = np.arange(n + n_predict)
    forecast_detrended = np.zeros(n + n_predict, dtype=np.float64)

    # DC term
    forecast_detrended += xf_trunc[0].real / n

    # Add each harmonic
    for k in range(1, n_harm + 1):
        amp_pos = xf_trunc[k]
        k_neg = n - k
        amp_neg = xf_trunc[k_neg] if k != n - k else np.conj(amp_pos)  # ensure conjugate symmetry

        phase = 2 * np.pi * k * t_extended / n
        forecast_detrended += (amp_pos * np.exp(1j * phase) + amp_neg * np.exp(-1j * phase)).real / n

    # --- Step 6: Add trend back ---
    # Extend the linear trend into the future
    trend_extended = coeffs[0] * t_extended + coeffs[1]
    full_forecast = forecast_detrended + trend_extended
    diagnostics = {
        'n_harm': n_harm,
        'energy_captured': cum_energy / total_energy,
        'sigma_log': 0,
        'trend_slope': coeffs[0],  # daily log-return trend
        'conf_level': 0,
        'z_score': 0
    }
    lower_band, upper_band = None, None
    return full_forecast, lower_band, upper_band, diagnostics


def fourier_forecast_log_returns_with_confidence(
        prices,
        n_predict=20,
        energy_threshold=0.95,
        conf_level=0.95  # for confidence bands
):
    """
    Forecast future prices using:
      - Log returns
      - Exponential detrending (via log-linear fit)
      - Fourier extrapolation on detrended log returns
      - Confidence bands from in-sample reconstruction error

    Returns:
      - forecast_prices: array of length len(prices) + n_predict
      - lower_band, upper_band: confidence intervals for forecast portion
      - diagnostics: dict with n_harm, sigma, etc.
    """
    n = len(prices)
    if n < 20:
        raise ValueError("Need at least 20 data points.")

    t = np.arange(n)

    # --- Step 1: Work in log space ---
    log_p = np.log(prices)

    # --- Step 2: Fit exponential trend (log-linear) ---
    # log(P_t) ≈ a * t + b  =>  P_t ≈ exp(b) * exp(a*t)
    coeffs = np.polyfit(t, log_p, 1)  # [a, b]
    log_trend = coeffs[0] * t + coeffs[1]
    detrended_log = log_p - log_trend  # stationary-ish residual

    # --- Step 3: FFT on detrended log returns (actually log deviations) ---
    xf = np.fft.fft(detrended_log)
    power = np.abs(xf) ** 2
    total_energy = np.sum(power) / n

    # --- Step 4: Auto-select harmonics ---
    cum_energy = power[0] / n
    n_harm = 0
    target = energy_threshold * total_energy

    if cum_energy >= target:
        n_harm = 0
    else:
        for k in range(1, n // 2 + 1):
            if k == n - k:
                pair_power = power[k]
            else:
                pair_power = power[k] + power[n - k]
            cum_energy += pair_power / n
            n_harm = k
            if cum_energy >= target:
                break

    # --- Step 5: Reconstruct full timeline (past + future) ---
    t_ext = np.arange(n + n_predict)
    recon_detrended_ext = np.zeros(n + n_predict)

    # Truncate spectrum
    xf_trunc = np.zeros(n, dtype=complex)
    xf_trunc[0] = xf[0]
    if n_harm > 0:
        xf_trunc[1:n_harm + 1] = xf[1:n_harm + 1]
        if n_harm < n // 2 or (n % 2 == 1):
            xf_trunc[-n_harm:] = xf[-n_harm:]
        elif n % 2 == 0 and n_harm == n // 2:
            xf_trunc[n // 2] = xf[n // 2]

    # Reconstruct
    recon_detrended_ext += xf_trunc[0].real / n
    for k in range(1, n_harm + 1):
        amp_pos = xf_trunc[k]
        amp_neg = xf_trunc[n - k] if k != n - k else np.conj(amp_pos)
        phase = 2 * np.pi * k * t_ext / n
        recon_detrended_ext += (amp_pos * np.exp(1j * phase) + amp_neg * np.exp(-1j * phase)).real / n

    # --- Step 6: Add trend back in log space, then exponentiate ---
    log_trend_ext = coeffs[0] * t_ext + coeffs[1]
    log_forecast = recon_detrended_ext + log_trend_ext
    forecast_prices = np.exp(log_forecast)

    # --- Step 7: Compute confidence bands ---
    # In-sample reconstruction (for error estimation)
    recon_insample = recon_detrended_ext[:n] + log_trend  # log scale
    residuals_log = log_p - recon_insample
    sigma_log = np.std(residuals_log)

    # Convert to multiplicative error in price space
    # Since log(P) ~ N(μ, σ²) → P has log-normal error
    # Approximate symmetric bands in log space, then exponentiate
    from scipy.stats import norm
    z = norm.ppf(0.5 + conf_level / 2)  # e.g., 1.96 for 95%

    # Forecast portion only
    log_forecast_future = log_forecast[n:]
    lower_log = log_forecast_future - z * sigma_log
    upper_log = log_forecast_future + z * sigma_log

    lower_band = np.exp(lower_log)
    upper_band = np.exp(upper_log)

    diagnostics = {
        'n_harm': n_harm,
        'energy_captured': cum_energy / total_energy,
        'sigma_log': sigma_log,
        'trend_slope': coeffs[0],  # daily log-return trend
        'conf_level': conf_level,
        'z_score': z
    }

    return forecast_prices, lower_band, upper_band, diagnostics


def fourier_extrapolation_hybrid(
        prices,
        n_predict=20,
        energy_threshold=0.95,
        conf_level=0.95
):
    """
    Hybrid Fourier extrapolation combining:
      - Log-space exponential detrending (better for financial data)
      - Automatic harmonic selection by spectral energy
      - Confidence bands from in-sample log-reconstruction error

    Parameters:
    - prices: 1D array of positive real prices (e.g., SPX)
    - n_predict: number of future steps to forecast
    - energy_threshold: fraction of spectral energy to retain (0 < th <= 1)
    - conf_level: confidence level for prediction bands (e.g., 0.95); if None, bands are None

    Returns:
    - full_forecast: array of length len(prices) + n_predict
    - lower_band: array of length n_predict (or None)
    - upper_band: array of length n_predict (or None)
    - diagnostics: dict with metadata
    """
    prices = np.asarray(prices, dtype=np.float64)
    if np.any(prices <= 0):
        raise ValueError("Prices must be strictly positive for log transformation.")
    n = len(prices)
    if n < 20:
        raise ValueError("Need at least 20 data points.")

    t = np.arange(n)
    t_ext = np.arange(n + n_predict)

    # --- Step 1: Log-transform ---
    log_p = np.log(prices)

    # --- Step 2: Exponential (log-linear) detrending ---
    coeffs = np.polyfit(t, log_p, 1)  # log_p ≈ a*t + b
    log_trend = coeffs[0] * t + coeffs[1]
    detrended = log_p - log_trend

    # --- Step 3: FFT on detrended log signal ---
    xf = np.fft.fft(detrended)
    power = np.abs(xf) ** 2
    total_energy = np.sum(power) / n

    # --- Step 4: Select harmonics to meet energy threshold ---
    cum_energy = power[0] / n
    n_harm = 0
    target_energy = energy_threshold * total_energy

    if cum_energy < target_energy:
        for k in range(1, n // 2 + 1):
            if k == n - k:  # Nyquist (n even)
                pair_power = power[k]
            else:
                pair_power = power[k] + power[n - k]
            cum_energy += pair_power / n
            n_harm = k
            if cum_energy >= target_energy:
                break

    # --- Step 5: Build truncated spectrum with conjugate symmetry ---
    xf_trunc = np.zeros(n, dtype=complex)
    xf_trunc[0] = xf[0]  # DC

    if n_harm > 0:
        xf_trunc[1:n_harm + 1] = xf[1:n_harm + 1]
        if n % 2 == 0 and n_harm == n // 2:
            # Nyquist term is its own conjugate
            xf_trunc[n // 2] = xf[n // 2]
        else:
            xf_trunc[-n_harm:] = xf[-n_harm:]

    # --- Step 6: Reconstruct detrended signal over extended timeline ---
    recon_ext = np.full(n + n_predict, xf_trunc[0].real / n)

    for k in range(1, n_harm + 1):
        amp_pos = xf_trunc[k]
        if k == n - k:
            amp_neg = np.conj(amp_pos)  # enforce symmetry at Nyquist
        else:
            amp_neg = xf_trunc[n - k]
        phase = 2 * np.pi * k * t_ext / n
        recon_ext += (amp_pos * np.exp(1j * phase) + amp_neg * np.exp(-1j * phase)).real / n

    # --- Step 7: Add back log-trend and exponentiate ---
    log_trend_ext = coeffs[0] * t_ext + coeffs[1]
    log_forecast = recon_ext + log_trend_ext
    full_forecast = np.exp(log_forecast)

    # --- Step 8: Compute confidence bands (if requested) ---
    lower_band, upper_band = None, None
    sigma_log = 0.0
    z_score = 0.0

    if conf_level is not None and 0 < conf_level < 1:
        # Reconstruct *in-sample* detrended signal for error estimation
        recon_insample_detrended = np.full(n, xf_trunc[0].real / n)
        for k in range(1, n_harm + 1):
            amp_pos = xf_trunc[k]
            amp_neg = np.conj(amp_pos) if k == n - k else xf_trunc[n - k]
            phase_in = 2 * np.pi * k * t / n
            recon_insample_detrended += (amp_pos * np.exp(1j * phase_in) + amp_neg * np.exp(-1j * phase_in)).real / n

        log_recon_insample = recon_insample_detrended + log_trend
        residuals_log = log_p - log_recon_insample
        sigma_log = np.std(residuals_log, ddof=1)  # unbiased estimate

        z_score = norm.ppf(0.5 + conf_level / 2)
        log_future = log_forecast[n:]
        lower_log = log_future - z_score * sigma_log
        upper_log = log_future + z_score * sigma_log
        lower_band = np.exp(lower_log)
        upper_band = np.exp(upper_log)

    diagnostics = {
        'n_harm': n_harm,
        'energy_captured': min(cum_energy / total_energy, 1.0),
        'sigma_log': sigma_log,
        'trend_slope': coeffs[0],  # daily log-return drift
        'conf_level': conf_level,
        'z_score': z_score
    }

    return full_forecast, lower_band, upper_band, diagnostics


def fourier_extrapolation_loess_bootstrap(
        prices,
        n_predict=20,
        energy_threshold=0.95,  # kept for API compatibility (unused)
        conf_level=0.95
):
    """
    Forecast using LOESS (local polynomial regression) + block bootstrap for CIs.
    Addresses Fourier limitations: no stationarity, no cycles, no FFT edge effects.

    Parameters:
    - prices: 1D array of positive real prices
    - n_predict: forecast horizon
    - energy_threshold: ignored (for API compatibility)
    - conf_level: confidence level for prediction bands

    Returns:
    - full_forecast: original + predicted prices
    - lower_band, upper_band: arrays of length n_predict
    - diagnostics: dict
    """
    from scipy.interpolate import interp1d
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    if n < 20:
        raise ValueError("Need at least 20 data points.")
    if np.any(prices <= 0):
        raise ValueError("Prices must be positive.")

    t = np.arange(n)
    t_future = np.arange(n, n + n_predict)
    t_full = np.concatenate([t, t_future])

    # --- Step 1: Fit LOESS-like smoother using LOWESS approximation
    # Since we avoid external libs, use rolling polynomial fit with adaptive window
    def local_poly_forecast(t_train, y_train, t_target, frac=0.3, order=1):
        """Local linear regression at each target point."""
        y_pred = np.empty_like(t_target, dtype=np.float64)
        window_size = max(10, int(frac * len(t_train)))
        for i, t0 in enumerate(t_target):
            # Compute distances and weights (tri-cube kernel)
            distances = np.abs(t_train - t0)
            max_dist = np.partition(distances, min(window_size - 1, len(distances) - 1))[window_size - 1]
            if max_dist == 0:
                weights = np.ones_like(distances)
            else:
                rel_dist = distances / max_dist
                weights = (1 - rel_dist ** 3) ** 3
                weights[rel_dist > 1] = 0

            # Fit weighted polynomial
            try:
                coeffs = np.polyfit(t_train, y_train, order, w=np.sqrt(weights + 1e-12))
                y_pred[i] = np.polyval(coeffs, t0)
            except np.linalg.LinAlgError:
                # Fallback to last observed value
                y_pred[i] = y_train[-1]
        return y_pred

    # Work in log space for stability
    log_prices = np.log(prices)
    log_forecast_future = local_poly_forecast(t, log_prices, t_future, frac=0.3, order=1)
    log_full = np.concatenate([log_prices, log_forecast_future])
    full_forecast = np.exp(log_full)

    lower_band, upper_band = None, None
    sigma_log = 0.0
    trend_slope = 0.0

    if conf_level is not None and 0 < conf_level < 1:
        # --- Step 2: Block bootstrap for prediction intervals ---
        # Use overlapping blocks to preserve temporal dependence
        block_size = min(10, n // 3)
        n_boot = 500
        boot_forecasts = np.empty((n_boot, n_predict))

        residuals = log_prices - local_poly_forecast(t, log_prices, t, frac=0.3, order=1)
        # Remove NaNs
        residuals = residuals[np.isfinite(residuals)]

        if len(residuals) < block_size:
            # Fallback to Gaussian if not enough residuals
            sigma_log = np.std(log_prices[1:] - log_prices[:-1])
            z = norm.ppf(0.5 + conf_level / 2)
            lower_log = log_forecast_future - z * sigma_log * np.sqrt(np.arange(1, n_predict + 1))
            upper_log = log_forecast_future + z * sigma_log * np.sqrt(np.arange(1, n_predict + 1))
        else:
            for b in range(n_boot):
                # Resample blocks
                boot_resid = np.empty(n)
                i = 0
                while i < n:
                    start = np.random.randint(0, len(residuals) - block_size + 1)
                    chunk = residuals[start:start + block_size]
                    end = min(i + block_size, n)
                    boot_resid[i:end] = chunk[:end - i]
                    i += block_size

                # Add bootstrapped noise to original fit
                boot_log = log_prices + boot_resid
                boot_future = local_poly_forecast(t, boot_log, t_future, frac=0.3, order=1)
                boot_forecasts[b, :] = boot_future

            # Compute quantiles
            alpha = (1 - conf_level) / 2
            lower_log = np.quantile(boot_forecasts, alpha, axis=0)
            upper_log = np.quantile(boot_forecasts, 1 - alpha, axis=0)

        lower_band = np.exp(lower_log)
        upper_band = np.exp(upper_log)
        sigma_log = np.std(residuals) if len(residuals) > 1 else 0.0

    # Estimate trend from last segment
    if n >= 5:
        recent_trend = np.polyfit(t[-5:], log_prices[-5:], 1)[0]
        trend_slope = recent_trend
    else:
        trend_slope = 0.0

    diagnostics = {
        'n_harm': -1,  # not applicable
        'energy_captured': 0.0,  # not applicable
        'sigma_log': sigma_log,
        'trend_slope': trend_slope,
        'conf_level': conf_level,
        'z_score': norm.ppf(0.5 + conf_level / 2) if conf_level else 0
    }

    return full_forecast, lower_band, upper_band, diagnostics


def sarimax_auto_forecast(
    prices,
    n_predict=20,
    energy_threshold=0.95,  # kept for API compatibility (unused)
    conf_level=0.95
):
    """
    Forecast using SARIMAX with automatic order selection (via pmdarima).
    Falls back to ARIMA(1,1,1) if pmdarima is not available.

    Parameters:
    - prices: 1D array of positive real prices
    - n_predict: forecast horizon
    - energy_threshold: ignored (for API compatibility)
    - conf_level: confidence level for prediction bands

    Returns:
    - full_forecast: original + predicted prices
    - lower_band, upper_band: arrays of length n_predict
    - diagnostics: dict
    """
    import numpy as np
    from scipy.stats import norm

    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    if n < 20:
        raise ValueError("Need at least 20 data points.")
    if np.any(prices <= 0):
        raise ValueError("Prices must be positive.")

    # Try to use pmdarima for auto SARIMAX; fall back if not available
    try:
        import pmdarima as pm
        auto_model = pm.auto_arima(
            prices,
            seasonal=False,        # disable seasonality unless you have strong reason
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=5,
            max_d=2,
            max_q=5,
            m=1,                   # non-seasonal
            trace=False,
            n_jobs=1
        )
        fitted_model = auto_model.fit(prices)
        forecast_result = fitted_model.predict(
            n_periods=n_predict,
            return_conf_int=True,
            alpha=1 - conf_level if conf_level else 0.05
        )
        if conf_level is not None:
            forecast_vals, conf_int = forecast_result
            lower_band = conf_int[:, 0]
            upper_band = conf_int[:, 1]
        else:
            forecast_vals = forecast_result
            lower_band = upper_band = None

        # Extract model order for diagnostics
        order = fitted_model.order
        seasonal_order = fitted_model.seasonal_order
        sigma = np.sqrt(fitted_model.model_result.mse) if hasattr(fitted_model, 'model_result') else np.std(np.diff(prices))
    except ImportError:
        # Fallback: simple ARIMA(1,1,1) using statsmodels if pmdarima not available
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(prices, order=(1, 1, 1))
            fitted = model.fit()
            forecast_obj = fitted.get_forecast(steps=n_predict)
            forecast_vals = forecast_obj.predicted_mean
            if conf_level is not None:
                conf_int = forecast_obj.conf_int(alpha=1 - conf_level)
                lower_band = conf_int[:, 0]
                upper_band = conf_int[:, 1]
            else:
                lower_band = upper_band = None
            order = (1, 1, 1)
            seasonal_order = (0, 0, 0, 0)
            sigma = np.sqrt(fitted.mse)
        except Exception as e:
            # Ultimate fallback: naive forecast
            last_val = prices[-1]
            forecast_vals = np.full(n_predict, last_val)
            lower_band = upper_band = None
            order = (0, 0, 0)
            seasonal_order = (0, 0, 0, 0)
            sigma = np.std(prices)

    full_forecast = np.concatenate([prices, forecast_vals])

    # Estimate linear trend in log space for diagnostics
    t = np.arange(n)
    log_p = np.log(prices)
    coeffs = np.polyfit(t, log_p, 1)
    trend_slope = coeffs[0]

    diagnostics = {
        'n_harm': -2,  # code for SARIMAX
        'energy_captured': 0.0,  # not applicable
        'sigma_log': sigma / np.mean(prices),  # normalized for comparability (approx)
        'trend_slope': trend_slope,
        'conf_level': conf_level,
        'z_score': norm.ppf(0.5 + conf_level / 2) if conf_level else 0,
        'sarimax_order': order,
        'sarimax_seasonal_order': seasonal_order
    }

    return full_forecast, lower_band, upper_band, diagnostics