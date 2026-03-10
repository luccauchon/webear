import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Setup & Data Loading (Done once before optimization)
# ---------------------------------------------------------
try:
    from utils import DATASET_AVAILABLE, get_filename_for_dataset, str2bool, add_vwap_with_bands

    one_dataset_filename = get_filename_for_dataset("day", older_dataset=None)
    import pickle

    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
except ImportError:
    print("ERROR: Could not import 'utils'. Please ensure utils.py is in your directory.")
    print("For testing purposes, you can uncomment the lines below to download data directly.")
    # --- Fallback for testing if utils is missing ---
    # ticker = '^GSPC'
    # df = yf.download(ticker, start="2010-01-01", end="2024-01-01")
    # master_data_cache = {ticker: df}
    # ----------------------------------------------
    raise SystemExit

ticker = '^GSPC'
df_base = master_data_cache[ticker].copy()

# Clean columns if MultiIndex
if isinstance(df_base.columns, pd.MultiIndex):
    df_base.columns = df_base.columns.get_level_values(0)

# Ensure no NaNs in critical columns before starting
df_base.dropna(subset=['Close'], inplace=True)
df_base.reset_index(drop=True, inplace=True)

print(f"Data loaded: {len(df_base)} rows. Starting Optimization...")


# ---------------------------------------------------------
# 2. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    # --- Suggest Parameters ---
    # RSI Parameters
    rsi_len = trial.suggest_int('rsi_len', 10, 20)
    rsi_thresh = trial.suggest_int('rsi_thresh', 60, 75)
    rsi_lookback = trial.suggest_int('rsi_lookback', 3, 10)

    # SMA Parameters
    sma_len = trial.suggest_int('sma_len', 20, 100)

    # MACD Parameters (NEW)
    macd_fast = trial.suggest_int('macd_fast', 8, 15)
    # Ensure slow is always larger than fast by setting min to fast + 5
    macd_slow = trial.suggest_int('macd_slow', macd_fast + 5, 40)
    macd_signal = trial.suggest_int('macd_signal', 5, 15)

    # Cluster Parameters
    cluster_window = trial.suggest_int('cluster_window', 20, 60)
    cluster_threshold = trial.suggest_int('cluster_threshold', 5, 10)

    # Fixed/Other Parameters
    forward_days = 20
    drop_threshold = -0.03

    # --- Work on a Copy ---
    df = df_base.copy()

    # --- 1. Calculate Indicators FIRST ---
    df['RSI'] = ta.rsi(df['Close'], length=rsi_len)

    # Calculate MACD with dynamic parameters
    # pandas_ta returns a DataFrame with columns named based on params (e.g. MACD_12_26_9)
    # We use iloc to avoid KeyError when param names change
    macd_df = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df['MACD_Line'] = macd_df.iloc[:, 0]  # MACD Line
    df['MACD_Sig'] = macd_df.iloc[:, 1]  # Signal Line

    df['SMA'] = ta.sma(df['Close'], length=sma_len)

    # --- 2. Calculate Future Target BEFORE Filtering ---
    # This must happen before creating 'clusters' so the column exists
    df['Future_Min'] = df['Close'].rolling(window=forward_days).min().shift(-forward_days)

    # --- 3. Generate Signals ---
    cond_price = df['Close'] > df['SMA']
    df['RSI_Max'] = df['RSI'].rolling(window=rsi_lookback).max()
    cond_rsi = df['RSI_Max'] > rsi_thresh

    # Bearish MACD crossover (Line crosses below Signal)
    cond_macd = df['MACD_Line'] < df['MACD_Sig']

    df['Signal'] = (cond_price & cond_rsi & cond_macd).astype(int)
    df['Signal'].fillna(0, inplace=True)

    # --- 4. Cluster Logic ---
    df['Omen_Count'] = df['Signal'].rolling(window=cluster_window).sum()
    df['Prev_Count'] = df['Omen_Count'].shift(1)
    df['Is_Cluster'] = (df['Omen_Count'] >= cluster_threshold) & (df['Prev_Count'] < cluster_threshold)

    # --- 5. Filter to Clusters (Now Future_Min exists in this subset) ---
    clusters = df[df['Is_Cluster']].copy()

    # --- 6. Validation ---
    if len(clusters) < 5:
        return 0.0

    # Drop rows where we don't have future data (end of dataset)
    clusters = clusters.dropna(subset=['Future_Min'])

    if len(clusters) < 5:
        return 0.0

    # --- 7. Calculate Win Rate ---
    clusters['Pct_Drop'] = (clusters['Future_Min'] - clusters['Close']) / clusters['Close']
    hits = clusters[clusters['Pct_Drop'] <= drop_threshold]

    win_rate = (len(hits) / len(clusters)) * 100

    return win_rate


# ---------------------------------------------------------
# 3. Run Optimization
# ---------------------------------------------------------
if __name__ == "__main__":
    # Create Study
    study = optuna.create_study(direction='maximize',
                                study_name='GSPC_Omen_Optimization',
                                load_if_exists=True)  # Loads if you stop and restart

    # Optimize
    # n_trials: How many combinations to try. 50-100 is usually good for start.
    # Increased timeout slightly to account for more complex parameter space
    study.optimize(objective, n_trials=9999999, timeout=120000, show_progress_bar=True, n_jobs=1)

    # ---------------------------------------------------------
    # 4. Report Results
    # ---------------------------------------------------------
    print("\n" + "=" * 30)
    print("OPTIMIZATION COMPLETE")
    print("=" * 30)
    print(f"Best Win Rate: {study.best_value:.2f}%")
    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nTotal Clusters found with best params: (Re-run script with these params to verify)")

    # Optional: Plotting (Requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        plot_optimization_history(study).show()
        plot_param_importances(study).show()
        print("\nTip: Use optuna.visualization.plot_param_importances(study) to see which params mattered most.")
    except:
        pass