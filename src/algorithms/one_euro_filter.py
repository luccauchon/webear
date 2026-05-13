import numpy as np
import pandas as pd
import math


def one_euro_filter(data, period_min=10, factor=0.2):
    """
    One-Euro Filter implementation in Python

    Adaptive low-pass filter that adjusts smoothing based on rate of change.
    Based on the Zorro Trader implementation from Traders Tips 12/2025.

    Parameters:
    -----------
    data : array-like
        Input time series (e.g., df['Close'].values)
    period_min : float, default=10
        Minimum cutoff period for the filter
    factor : float, default=0.2
        Sensitivity factor: higher = more responsive to rate of change

    Returns:
    --------
    np.ndarray
        Filtered/smoothed output series (same length as input)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n == 0:
        return np.array([])

    # Output arrays
    smoothed = np.zeros(n)
    smoothed_dx = np.zeros(n)  # EMA of delta (rate of change)

    # Initialize first values
    smoothed[0] = data[0]
    smoothed_dx[0] = 0.0

    # Fixed alpha for smoothing the delta (rate of change)
    # Based on fixed cutoff of 10 as in original code
    alpha_dx = 2 * math.pi / (4 * math.pi + 10)

    for i in range(1, n):
        # Compute delta (rate of change)
        delta = data[i] - data[i - 1]

        # EMA the Delta Price (smooth the rate of change)
        smoothed_dx[i] = alpha_dx * delta + (1 - alpha_dx) * smoothed_dx[i - 1]

        # Adjust cutoff period based on magnitude of rate of change
        cutoff = period_min + factor * abs(smoothed_dx[i])

        # Compute adaptive alpha for main smoothing
        alpha = 2 * math.pi / (4 * math.pi + cutoff)

        # Apply adaptive exponential smoothing
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed