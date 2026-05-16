import numpy as np
import pandas as pd
import math
from numba import njit
import time


def one_euro_filter_slow(data, period_min=10, factor=0.2):
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


@njit(cache=True)
def _one_euro_filter_core(data, period_min, factor):
    n = len(data)
    smoothed = np.empty(n, dtype=np.float64)
    smoothed_dx = np.empty(n, dtype=np.float64)

    smoothed[0] = data[0]
    smoothed_dx[0] = 0.0

    # Precompute constants to eliminate repeated operations inside the loop
    alpha_dx = 2.0 * np.pi / (4.0 * np.pi + 10.0)
    inv_alpha_dx = 1.0 - alpha_dx
    two_pi = 2.0 * np.pi
    four_pi = 4.0 * np.pi

    for i in range(1, n):
        delta = data[i] - data[i - 1]

        # EMA of rate of change
        smoothed_dx[i] = alpha_dx * delta + inv_alpha_dx * smoothed_dx[i - 1]

        # Adaptive cutoff & alpha
        cutoff = period_min + factor * abs(smoothed_dx[i])
        alpha = two_pi / (four_pi + cutoff)

        # Main adaptive smoothing
        smoothed[i] = alpha * data[i] + (1.0 - alpha) * smoothed[i - 1]

    return smoothed


def one_euro_filter(data, period_min=10.0, factor=0.2):
    """
    Super-fast One-Euro Filter using Numba JIT compilation.
    Matches the original implementation exactly but runs at C-speed.
    """
    # Ensure contiguous float64 array for optimal cache performance
    data_arr = np.ascontiguousarray(data, dtype=np.float64)
    return _one_euro_filter_core(data_arr, float(period_min), float(factor))


if __name__ == "__main__":
    np.random.seed(42)

    for u in range(0, 100):
        prices = np.cumsum(np.random.randn(100000)) + 100

        # Mesure de la version rapide
        t1 = time.time()
        fast = one_euro_filter(prices, period_min=10, factor=0.2)
        t2 = time.time()
        t_fast = t2 - t1

        # Mesure de la version lente
        t3 = time.time()
        normal = one_euro_filter_slow(prices, period_min=10, factor=0.2)
        t4 = time.time()
        t_slow = t4 - t3

        # Vérification de l'exactitude
        assert np.allclose(fast, normal), "Les résultats diffèrent !"

        # Affichage du gain réel
        try:
            gain_multiplicateur = t_slow / t_fast
            print(f"Iteration {u:02d} | Version rapide {gain_multiplicateur:.1f}x plus rapide")
        except ZeroDivisionError:
            pass