import numpy as np


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