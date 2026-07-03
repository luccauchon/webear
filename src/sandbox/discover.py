try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import optuna
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import time
from curl_cffi import requests
import urllib3
from constants import FRED_API_KEY, IS_RUNNING_IREQ

# Suppress Optuna & pandas debug logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None

# OPTIMIZATION: Try to use Numba for a 100x speedup on rolling percentile rank
try:
    from numba import njit


    @njit(cache=True)
    def _rolling_pct_rank_numba(arr, window):
        n = len(arr)
        res = np.full(n, np.nan)
        for i in range(window - 1, n):
            w = arr[i - window + 1: i + 1]
            last_val = w[-1]
            if np.isnan(last_val):
                continue
            count_less = 0
            count_equal = 0
            count_valid = 0
            for val in w:
                if not np.isnan(val):
                    count_valid += 1
                    if val < last_val:
                        count_less += 1
                    elif val == last_val:
                        count_equal += 1
            if count_valid > 0:
                res[i] = (count_less + 0.5 * count_equal + 0.5) / count_valid
        return res


    def rolling_percentile_rank(series, window):
        arr = series.to_numpy()
        res = _rolling_pct_rank_numba(arr, window)
        return pd.Series(res, index=series.index)
except ImportError:
    # Fallback: Highly optimized pure NumPy version (5x-10x faster than original)
    def rolling_percentile_rank(series, window):
        def _rank(x):
            last_val = x[-1]
            if np.isnan(last_val):
                return np.nan
            # np.count_nonzero is much faster than np.sum for boolean arrays
            return (np.count_nonzero(x < last_val) + 0.5 * np.count_nonzero(x == last_val) + 0.5) / len(x)

        return series.rolling(window).apply(_rank, raw=True)

# OPTIMIZATION 3: Removed duplicate features to save memory and computation
__FEATURES__ = [
    'spx_pct_rank', 'spx_zscore',
    'spx_vol_pct_rank', 'spx_vol_zscore',
    'vix_pct_rank', 'vix_zscore',
    'hyg_pct_rank', 'hyg_zscore',
    'lqd_pct_rank', 'lqd_zscore',
    'curve_pct_rank', 'curve_zscore',
    'curve_3m_pct_rank', 'curve_3m_zscore',
    'baa10y_pct_rank', 'baa10y_zscore',
    'nfci_pct_rank', 'nfci_zscore',
    'walcl_pct_rank', 'walcl_zscore',
    'wm2ns_pct_rank', 'wm2ns_zscore',
    'dfii10_pct_rank', 'dfii10_zscore',
    'usphci_pct_rank', 'usphci_zscore',
    'breakeven_pct_rank', 'breakeven_zscore',
    'rrp_pct_rank', 'rrp_zscore',
    'oil_pct_rank', 'oil_zscore',
    'trend_score', 'vol_score', 'credit_score', 'curve_score',
    'fc_score', 'final_score', 'heuristic_market_score',
    'spx_momentum_21d', 'vix_diff_5d', 'credit_ratio_diff_10d',
    'curve_diff_20d', 'spx_return_1d_lag'
]

# Base columns needed for feature engineering (avoids copying the whole DF)
_BASE_COLS = [
    'spx', 'spx_open', 'spx_high', 'spx_low', 'spx_vol', 'vix', 'hyg', 'lqd',
    'curve', 'curve_3m', 'baa10y', 'nfci', 'walcl', 'wm2ns', 'dfii10', 'usphci',
    'breakeven', 'rrp', 'oil'
]


def _calculate_metrics(y_true, y_pred, params: dict) -> tuple:
    """Helper to calculate R2, Win Rate, and Sharpe for a given true/pred pair."""
    index = getattr(y_true, 'index', None)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert "epsilon" in params
    epsilon = params.get("epsilon", 0.001)

    actual_up = y_true > epsilon
    actual_down = y_true < -epsilon
    actual_neutral = (y_true >= -epsilon) & (y_true <= epsilon)

    pred_up = y_pred > epsilon
    pred_down = y_pred < -epsilon
    pred_neutral = (y_pred >= -epsilon) & (y_pred <= epsilon)

    correct_direction = ((actual_up & pred_up) | (actual_down & pred_down) | (actual_neutral & pred_neutral))
    if index is not None:
        correct_direction = pd.Series(correct_direction, index=index)

    win_rate = np.mean(correct_direction)

    positions = np.where(y_pred > epsilon, 1, np.where(y_pred < -epsilon, -1, 0))
    strategy_returns = positions * y_true  # Optimized: removed .values

    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns) + 1e-8

    # Adjust annualization factor based on timeframe
    timeframe = params.get("timeframe", "day")
    if timeframe == "week":
        periods_per_year = 52
    elif timeframe == "month":
        periods_per_year = 12
    else:
        periods_per_year = 252

    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(periods_per_year / params["lookahead_bars"])
    assert "density" in params
    density = params.get("density", 0.15)
    active_days = np.sum(positions != 0)
    activity_ratio = active_days / len(y_true)
    penalty = max(0, density - activity_ratio)
    sharpe_ratio -= penalty * 10

    return r2_score(y_true, y_pred), win_rate, sharpe_ratio, correct_direction


def _calculate_consensus_metrics(y_true, cons_pred, params: dict) -> tuple:
    """Helper to calculate Win Rate and Sharpe for the Consensus strategy."""
    index = getattr(y_true, 'index', None)
    y_true = np.asarray(y_true)
    cons_pred = np.asarray(cons_pred)

    assert "epsilon" in params
    epsilon = params.get("epsilon", 0.001)

    actual_up = y_true > epsilon
    actual_down = y_true < -epsilon
    actual_neutral = (y_true >= -epsilon) & (y_true <= epsilon)

    # Consensus thresholds match real-time logic (BULLISH >= 30, BEARISH <= -30)
    pred_up = cons_pred >= 30
    pred_down = cons_pred <= -30
    pred_neutral = (cons_pred > -30) & (cons_pred < 30)

    correct_direction = ((actual_up & pred_up) | (actual_down & pred_down) | (actual_neutral & pred_neutral))
    if index is not None:
        correct_direction = pd.Series(correct_direction, index=index)

    win_rate = np.mean(correct_direction)

    # Positions for Sharpe calculation
    positions = np.where(cons_pred >= 30, 1, np.where(cons_pred <= -30, -1, 0))
    strategy_returns = positions * y_true  # Optimized: removed .values

    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns) + 1e-8

    timeframe = params.get("timeframe", "day")
    if timeframe == "week":
        periods_per_year = 52
    elif timeframe == "month":
        periods_per_year = 12
    else:
        periods_per_year = 252

    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(periods_per_year / params["lookahead_bars"])
    assert "density" in params
    density = params.get("density", 0.15)
    active_days = np.sum(positions != 0)
    activity_ratio = active_days / len(y_true)
    penalty = max(0, density - activity_ratio)
    sharpe_ratio -= penalty * 10

    return 0.0, win_rate, sharpe_ratio, correct_direction


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Market Prediction Model with Optuna Optimization and Walk-Forward Validation"
    )

    parser.add_argument("--timeframe", type=str, default="day", choices=["day", "week", "month"])
    parser.add_argument("--model", type=str, default="ridge",
                        choices=["ridge", "lasso", "random_forest", "gradient_boosting", "extra_trees", "svr", "knn", "xgboost"])
    parser.add_argument("--train-val-split-ratio", type=float, default=0.9)
    parser.add_argument("--lookahead-bars", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--n-trials", type=int, default=99999)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--data-filename", type=str, default=None)
    parser.add_argument("--skip-tomorrow", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument('--verbose-study-progress-bar', action=argparse.BooleanOptionalAction, default=False, help='Verbose output')

    # OPTIMIZATION: Added n-fold argument to reduce CV overhead
    parser.add_argument("--n-fold", type=int, default=10,
                        help="Number of folds for TimeSeriesSplit (default: 10). Lower is much faster.")

    # FEATURE SELECTION ARGUMENTS
    parser.add_argument("--min-features", type=int, default=-1,
                        help="Minimum number of features to select (default: -1). -1 will set it to max-features.")
    parser.add_argument("--max-features", type=int, default=None,
                        help="Maximum number of features to select (default: all available).")

    # OPTUNA PERSISTENCE AND SAMPLER ARGUMENTS
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., 'sqlite:///optuna.db') to persist the study.")
    parser.add_argument("--study-name", type=str, default="market_prediction_study",
                        help="Name of the Optuna study (used with --storage).")
    parser.add_argument("--random-sampler", action="store_true",
                        help="Use RandomSampler instead of the default TPESampler.")

    return parser


def load_data(filename: str = None, timeframe: str = 'day') -> pd.DataFrame:
    """Loads market data from disk or fetches it from Yahoo Finance and FRED."""
    if filename is None:
        current_date_str = datetime.now().strftime("%Y%m%d")
        filename = f"market_data_{timeframe}_{current_date_str}.csv"

    fetch_data = False
    if os.path.exists(filename):
        print(f"File '{filename}' exists. Reloading data from disk...")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        required_cols = ['spx', 'vix', 'hyg', 'lqd', 'curve', 'curve_3m', 'baa10y', 'nfci', 'walcl', 'wm2ns', 'dfii10', 'usphci', 'breakeven', 'rrp', 'oil']
        if not all(col in df.columns for col in required_cols):
            print(f"File '{filename}' is missing new macroeconomic columns. Refetching data...")
            os.remove(filename)
            fetch_data = True
    else:
        fetch_data = True

    if fetch_data:
        if IS_RUNNING_IREQ:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            os.environ["CURL_CA_BUNDLE"] = ""
            session = requests.Session(impersonate="chrome", verify=False)

            spx_data = yf.download("^GSPC", start="2000-01-01", auto_adjust=True, session=session)
            spx_open = spx_data["Open"].squeeze()
            spx_high = spx_data["High"].squeeze()
            spx_low = spx_data["Low"].squeeze()
            spx = spx_data["Close"].squeeze()
            spx_vol = spx_data["Volume"].squeeze()
            vix = yf.download("^VIX", start="2000-01-01", auto_adjust=True, session=session)["Close"].squeeze()
            hyg = yf.download("HYG", start="2007-01-01", auto_adjust=True, session=session)["Close"].squeeze()
            lqd = yf.download("LQD", start="2007-01-01", auto_adjust=True, session=session)["Close"].squeeze()
        else:
            spx_data = yf.download("^GSPC", start="2000-01-01", auto_adjust=True)
            spx_open = spx_data["Open"].squeeze()
            spx_high = spx_data["High"].squeeze()
            spx_low = spx_data["Low"].squeeze()
            spx = spx_data["Close"].squeeze()
            spx_vol = spx_data["Volume"].squeeze()
            vix = yf.download("^VIX", start="2000-01-01", auto_adjust=True)["Close"].squeeze()
            hyg = yf.download("HYG", start="2007-01-01", auto_adjust=True)["Close"].squeeze()
            lqd = yf.download("LQD", start="2007-01-01", auto_adjust=True)["Close"].squeeze()

        print(f"File '{filename}' not found. Fetching new data...")
        fred = Fred(api_key=FRED_API_KEY)

        curve = fred.get_series("T10Y2Y")
        curve_3m = fred.get_series("T10Y3M")
        baa10y = fred.get_series("BAA10Y")
        nfci = fred.get_series("NFCI")
        walcl = fred.get_series("WALCL")
        wm2ns = fred.get_series("WM2NS")
        dfii10 = fred.get_series("DFII10")
        usphci = fred.get_series("USPHCI")
        breakeven = fred.get_series("T10YIE")
        rrp = fred.get_series("RRPONTSYD")
        oil = fred.get_series("DCOILWTICO")

        df = pd.DataFrame(index=spx.index)
        df["spx_open"] = spx_open
        df["spx_high"] = spx_high
        df["spx_low"] = spx_low
        df["spx"] = spx
        df["spx_vol"] = spx_vol
        df["vix"] = vix
        df["hyg"] = hyg
        df["lqd"] = lqd
        df["curve"] = curve
        df["curve_3m"] = curve_3m
        df["baa10y"] = baa10y
        df["nfci"] = nfci
        df["walcl"] = walcl
        df["wm2ns"] = wm2ns
        df["dfii10"] = dfii10
        df["usphci"] = usphci
        df["breakeven"] = breakeven
        df["rrp"] = rrp
        df["oil"] = oil

        df = df.loc["2008-01-01":].ffill()

        # OPTIMIZATION: Convert to float32 to save memory and speed up computations
        numeric_cols = df.select_dtypes(include=[np.float64, np.int64]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

        if timeframe != 'day':
            if timeframe == 'week':
                rule = 'W-FRI'
            else:
                try:
                    pd.Series([1], index=pd.date_range('2020-01-01', periods=1)).resample('ME').sum()
                    rule = 'ME'
                except ValueError:
                    rule = 'M'

            agg_funcs = {col: 'last' for col in df.columns if col not in ['spx_open', 'spx_high', 'spx_low', 'spx_vol']}
            agg_funcs.update({'spx_open': 'first', 'spx_high': 'max', 'spx_low': 'min', 'spx_vol': 'sum'})
            df = df.resample(rule).agg(agg_funcs).dropna()

        df.to_csv(filename)
        print(f"Data successfully saved to '{filename}'")

    return df


def rolling_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def compute_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Computes all feature scores and new momentum/diff features for a given dataset."""
    # OPTIMIZATION: Only copy the necessary base columns to save memory and time
    df = df[_BASE_COLS].copy()

    lookahead_bars = params["lookahead_bars"]
    ma50_win = params["value_of_ma50"]
    ma200_win = params["value_of_ma200"]
    w1 = params["trend_raw_weight_1"]
    w2 = params["trend_raw_weight_2"]
    vol_win = params["value_of_roll_volatility"]
    credit_win = params["value_of_roll_credit_score"]
    curve_win = params["value_of_roll_yield_curve"]
    fc_win = params["value_of_roll_fc"]
    final_win = params["value_of_roll_final_score"]

    trend_score_weight = params["trend_score_weight"]
    volume_score_weight = params["volume_score_weight"]
    credit_score_weight = params["credit_score_weight"]
    curve_score_weight = params["curve_score_weight"]
    fc_score_weight = params["fc_score_weight"]

    rank_window = params["rank_window"]
    zscore_window = params["zscore_window"]

    df["ground_truth_spx"] = df["spx"].shift(-lookahead_bars)

    # 1. Trend Score
    ma50 = df["spx"].rolling(ma50_win).mean()
    ma200 = df["spx"].rolling(ma200_win).mean()
    trend_raw = w1 * (df["spx"] > ma200) + w2 * (ma50 > ma200)
    trend_score = -trend_raw

    # 2. Volatility Score
    vix_mean = df["vix"].rolling(vol_win).mean()
    vix_std = df["vix"].rolling(vol_win).std()
    vol_score = -(df["vix"] - vix_mean) / vix_std

    # 3. Credit Score
    hyg_lqd = df["hyg"] / df["lqd"]
    baa_mean = df["baa10y"].rolling(credit_win).mean()
    baa_std = df["baa10y"].rolling(credit_win).std()
    hl_mean = hyg_lqd.rolling(credit_win).mean()
    hl_std = hyg_lqd.rolling(credit_win).std()
    credit_score = ((df["baa10y"] - baa_mean) / baa_std) - ((hyg_lqd - hl_mean) / hl_std)

    # 4. Yield Curve Score
    curve_mean = df["curve"].rolling(curve_win).mean()
    curve_std = df["curve"].rolling(curve_win).std()
    curve_score = -((df["curve"] - curve_mean) / curve_std)

    # 5. Financial Conditions Score
    nfci_mean = df["nfci"].rolling(fc_win).mean()
    nfci_std = df["nfci"].rolling(fc_win).std()
    fc_score = (df["nfci"] - nfci_mean) / nfci_std

    raw_score = (trend_score_weight * trend_score +
                 volume_score_weight * vol_score +
                 credit_score_weight * credit_score +
                 curve_score_weight * curve_score +
                 fc_score_weight * fc_score)

    final_score = (raw_score - raw_score.rolling(final_win).mean()) / raw_score.rolling(final_win).std()

    df["spx_momentum_21d"] = df["spx"].pct_change(params["spx_momentum_21d"])
    df["vix_diff_5d"] = df["vix"].diff(params["vix_diff_5d"])
    df["credit_ratio_diff_10d"] = hyg_lqd.diff(params["credit_ratio_diff_10d"])
    df["curve_diff_20d"] = df["curve"].diff(params["curve_diff_20d"])
    df["spx_return_1d_lag"] = df["spx"].pct_change(1).shift(1)

    df['trend_score'] = trend_score
    df['vol_score'] = vol_score
    df['credit_score'] = credit_score
    df['curve_score'] = curve_score
    df['fc_score'] = fc_score
    df['final_score'] = final_score
    df['heuristic_market_score'] = np.clip(-final_score * 40, -100, 100)

    # Regime-normalized market and macro features
    df["spx_pct_rank"] = rolling_percentile_rank(df["spx"], rank_window)
    df["spx_zscore"] = rolling_zscore(df["spx"], zscore_window)
    df["spx_vol_pct_rank"] = rolling_percentile_rank(df["spx_vol"], rank_window)
    df["spx_vol_zscore"] = rolling_zscore(df["spx_vol"], zscore_window)
    df["vix_pct_rank"] = rolling_percentile_rank(df["vix"], rank_window)
    df["vix_zscore"] = rolling_zscore(df["vix"], zscore_window)
    df["hyg_pct_rank"] = rolling_percentile_rank(df["hyg"], rank_window)
    df["hyg_zscore"] = rolling_zscore(df["hyg"], zscore_window)
    df["lqd_pct_rank"] = rolling_percentile_rank(df["lqd"], rank_window)
    df["lqd_zscore"] = rolling_zscore(df["lqd"], zscore_window)
    df["curve_pct_rank"] = rolling_percentile_rank(df["curve"], rank_window)
    df["curve_zscore"] = rolling_zscore(df["curve"], zscore_window)
    df["baa10y_pct_rank"] = rolling_percentile_rank(df["baa10y"], rank_window)
    df["baa10y_zscore"] = rolling_zscore(df["baa10y"], zscore_window)
    df["nfci_pct_rank"] = rolling_percentile_rank(df["nfci"], rank_window)
    df["nfci_zscore"] = rolling_zscore(df["nfci"], zscore_window)
    df["walcl_pct_rank"] = rolling_percentile_rank(df["walcl"], rank_window)
    df["walcl_zscore"] = rolling_zscore(df["walcl"], zscore_window)
    df["wm2ns_pct_rank"] = rolling_percentile_rank(df["wm2ns"], rank_window)
    df["wm2ns_zscore"] = rolling_zscore(df["wm2ns"], zscore_window)
    df["dfii10_pct_rank"] = rolling_percentile_rank(df["dfii10"], rank_window)
    df["dfii10_zscore"] = rolling_zscore(df["dfii10"], zscore_window)
    df["usphci_pct_rank"] = rolling_percentile_rank(df["usphci"], rank_window)
    df["usphci_zscore"] = rolling_zscore(df["usphci"], zscore_window)
    df["curve_3m_pct_rank"] = rolling_percentile_rank(df["curve_3m"], rank_window)
    df["curve_3m_zscore"] = rolling_zscore(df["curve_3m"], zscore_window)
    df["breakeven_pct_rank"] = rolling_percentile_rank(df["breakeven"], rank_window)
    df["breakeven_zscore"] = rolling_zscore(df["breakeven"], zscore_window)
    df["rrp_pct_rank"] = rolling_percentile_rank(df["rrp"], rank_window)
    df["rrp_zscore"] = rolling_zscore(df["rrp"], zscore_window)
    df["oil_pct_rank"] = rolling_percentile_rank(df["oil"], rank_window)
    df["oil_zscore"] = rolling_zscore(df["oil"], zscore_window)

    return df.dropna().copy()


def get_model_and_scaler(model_type: str, params: dict):
    """Factory function to return the appropriate model and scaler based on the specified model type."""
    if model_type == "ridge":
        model = Ridge(alpha=params.get("ml_alpha", 1.0))
        scaler = RobustScaler()
    elif model_type == "lasso":
        model = Lasso(alpha=params.get("ml_alpha", 0.1), max_iter=10000)
        scaler = RobustScaler()
    elif model_type == "elastic_net":
        model = ElasticNet(alpha=params.get("ml_alpha", 0.1), l1_ratio=params.get("ml_l1_ratio", 0.5), max_iter=10000)
        scaler = RobustScaler()
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1
        )
        scaler = None
    elif model_type == "gradient_boosting":
        # OPTIMIZATION: HistGradientBoostingRegressor is 10x-100x faster than standard GradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42
        )
        scaler = StandardScaler()
    elif model_type == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1
        )
        scaler = None
    elif model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 1),
            gamma=params.get("gamma", 0.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
        scaler = None
    elif model_type == "svr":
        model = SVR(
            C=params.get("svr_c", 1.0),
            epsilon=params.get("svr_epsilon", 0.1),
            kernel=params.get("svr_kernel", "rbf")
        )
        scaler = RobustScaler()
    elif model_type == "knn":
        model = KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("knn_weights", "distance"),
            n_jobs=-1
        )
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, scaler


def evaluate_datasets(df_train: pd.DataFrame, df_test: pd.DataFrame, params: dict, features: list, model_type: str) -> tuple:
    """Fits model on TRAIN set and evaluates on BOTH train and TEST sets properly."""
    df_train_feat = compute_features(df_train, params)
    df_train_feat['forward_return'] = (df_train_feat["ground_truth_spx"] / df_train_feat["spx"]) - 1.0
    valid_train = df_train_feat.dropna(subset=features + ['forward_return'])

    df_test_feat = compute_features(df_test, params)
    df_test_feat['forward_return'] = (df_test_feat["ground_truth_spx"] / df_test_feat["spx"]) - 1.0
    valid_test = df_test_feat.dropna(subset=features + ['forward_return'])

    # OPTIMIZATION: Convert features to NumPy, but KEEP targets as Series to preserve index!
    X_train_np = valid_train[features].to_numpy(dtype=np.float32)
    y_train_np = valid_train['forward_return']
    X_test_np = valid_test[features].to_numpy(dtype=np.float32)
    y_test_np = valid_test['forward_return']

    model, scaler = get_model_and_scaler(model_type, params)
    try:
        X_train_s = scaler.fit_transform(X_train_np) if scaler else X_train_np
        X_test_s = scaler.transform(X_test_np) if scaler else X_test_np

        model.fit(X_train_s, y_train_np)

        y_pred_train = model.predict(X_train_s)
        train_r2, train_wr, train_sharpe, train_correct_direction = _calculate_metrics(y_train_np, y_pred_train, params)

        p95_train = np.percentile(y_train_np, 95)
        p05_train = np.percentile(y_train_np, 5)
        max_abs_ret_train = max(abs(p95_train), abs(p05_train))
        if max_abs_ret_train > 1e-6:
            ml_train_scaled = np.clip((y_pred_train / max_abs_ret_train) * 100, -100, 100)
        else:
            ml_train_scaled = np.zeros_like(y_pred_train)

        heur_train = valid_train['heuristic_market_score'].values
        cons_pred_train = (ml_train_scaled + heur_train) / 2.0
        train_cons_r2, train_cons_wr, train_cons_sharpe, train_cons_correct_direction = _calculate_consensus_metrics(y_train_np, cons_pred_train, params)

        y_pred_test = model.predict(X_test_s)
        test_r2, test_wr, test_sharpe, test_correction_direction = _calculate_metrics(y_test_np, y_pred_test, params)

        if max_abs_ret_train > 1e-6:
            ml_test_scaled = np.clip((y_pred_test / max_abs_ret_train) * 100, -100, 100)
        else:
            ml_test_scaled = np.zeros_like(y_pred_test)

        heur_test = valid_test['heuristic_market_score'].values
        cons_pred_test = (ml_test_scaled + heur_test) / 2.0
        test_cons_r2, test_cons_wr, test_cons_sharpe, test_cons_correction_direction = _calculate_consensus_metrics(y_test_np, cons_pred_test, params)

        return (train_r2, train_wr, train_sharpe, train_correct_direction,
                test_r2, test_wr, test_sharpe, test_correction_direction,
                train_cons_r2, train_cons_wr, train_cons_sharpe, train_cons_correct_direction,
                test_cons_r2, test_cons_wr, test_cons_sharpe, test_cons_correction_direction)
    except Exception:
        return (-1e9, 0.0, -1e9, 0.0, -1e9, 0.0, -1e9, 0.0, -1e9, 0.0, -1e9, 0.0, -1e9, 0.0, -1e9, 0.0)


def objective(trial: optuna.Trial, df: pd.DataFrame, model_type: str, n_fold: int, **kwargs) -> float:
    """Optuna objective with conditional hyperparameters for the specified model."""
    max_window = max(10, len(df) // 2)
    max_ma = max(5, len(df) // 4)
    max_ma200 = min(max_ma * 2, len(df) - 2)
    if max_ma200 <= max_ma:
        max_ma200 = max_ma + 1
    max_diff = max(1, len(df) // 10)

    is_day = kwargs["timeframe"] == "day"

    if is_day:
        rank_low, rank_high = 50, min(1000, max_window)
        zscore_low, zscore_high = 20, min(500, max_window)
        ma50_low, ma50_high = 10, min(100, max_ma)
        ma200_low, ma200_high = min(100, max_ma) + 1, min(300, max_ma200)
        roll_low, roll_high = 20, min(500, max_window)
        mom_21d_low, mom_21d_high = min(21, max_diff), min(21, max_diff)
        vix_5d_low, vix_5d_high = min(5, max_diff), min(5, max_diff)
        credit_10d_low, credit_10d_high = min(10, max_diff), min(10, max_diff)
        curve_20d_low, curve_20d_high = min(20, max_diff), min(20, max_diff)
    else:
        rank_low, rank_high = 3, max_window
        zscore_low, zscore_high = 3, max_window
        ma50_low, ma50_high = 2, max_ma
        ma200_low, ma200_high = max_ma + 1, max_ma200
        roll_low, roll_high = 3, max_window
        mom_21d_low, mom_21d_high = 1, min(21, max_diff)
        vix_5d_low, vix_5d_high = 1, min(5, max_diff)
        credit_10d_low, credit_10d_high = 1, min(10, max_diff)
        curve_20d_low, curve_20d_high = 1, min(20, max_diff)

    # --- FEATURE SELECTION LOGIC ---
    min_features = kwargs.get("min_features", 10)
    max_features = kwargs.get("max_features", len(__FEATURES__))

    # Ensure bounds are valid
    min_features = max(1, min(min_features, len(__FEATURES__)))
    max_features = max(min_features, min(max_features, len(__FEATURES__)))

    num_features = trial.suggest_int("num_features", min_features, max_features)

    feature_weights = {}
    for feat in __FEATURES__:
        feature_weights[feat] = trial.suggest_float(f"fw_{feat}", 0.0, 1.0, step=0.01)

    # Select the top features based on the weights suggested by Optuna
    selected_features = sorted(feature_weights, key=feature_weights.get, reverse=True)[:num_features]
    # -----------------------------

    params = {
        "timeframe": kwargs["timeframe"], "model_type": model_type,
        "rank_window": trial.suggest_int("rank_window", rank_low, rank_high),
        "zscore_window": trial.suggest_int("zscore_window", zscore_low, zscore_high),
        "value_of_ma50": trial.suggest_int("value_of_ma50", ma50_low, ma50_high),
        "value_of_ma200": trial.suggest_int("value_of_ma200", ma200_low, ma200_high),
        "trend_raw_weight_1": trial.suggest_float("trend_raw_weight_1", 0.0, 1.0, step=0.01),
        "trend_raw_weight_2": trial.suggest_float("trend_raw_weight_2", 0.0, 1.0, step=0.01),
        "value_of_roll_volatility": trial.suggest_int("value_of_roll_volatility", roll_low, roll_high),
        "value_of_roll_credit_score": trial.suggest_int("value_of_roll_credit_score", roll_low, roll_high),
        "value_of_roll_yield_curve": trial.suggest_int("value_of_roll_yield_curve", roll_low, roll_high),
        "value_of_roll_fc": trial.suggest_int("value_of_roll_fc", roll_low, roll_high),
        "value_of_roll_final_score": trial.suggest_int("value_of_roll_final_score", roll_low, roll_high),
        "trend_score_weight": trial.suggest_float("trend_score_weight", 0.65, 0.65),
        "volume_score_weight": trial.suggest_float("volume_score_weight", 0.45, 0.45),
        "credit_score_weight": trial.suggest_float("credit_score_weight", 0.15, 0.15),
        "curve_score_weight": trial.suggest_float("curve_score_weight", 0.10, 0.10),
        "fc_score_weight": trial.suggest_float("fc_score_weight", 0.05, 0.05),
        "lookahead_bars": trial.suggest_int("lookahead_bars", kwargs['lookahead_bars'], kwargs['lookahead_bars']),
        "epsilon": trial.suggest_float("epsilon", kwargs['epsilon'], kwargs['epsilon']),
        "density": trial.suggest_float("density", kwargs['density'], kwargs['density']),
        "spx_momentum_21d": trial.suggest_int("spx_momentum_21d", mom_21d_low, mom_21d_high),
        "vix_diff_5d": trial.suggest_int("vix_diff_5d", vix_5d_low, vix_5d_high),
        "credit_ratio_diff_10d": trial.suggest_int("credit_ratio_diff_10d", credit_10d_low, credit_10d_high),
        "curve_diff_20d": trial.suggest_int("curve_diff_20d", curve_20d_low, curve_20d_high),
    }

    if model_type in ["ridge", "lasso", "elastic_net"]:
        params["ml_alpha"] = trial.suggest_float("ml_alpha", 0.01, 10.0, log=True)
        if model_type == "elastic_net":
            params["ml_l1_ratio"] = trial.suggest_float("ml_l1_ratio", 0.1, 0.9)
    elif model_type in ["random_forest", "gradient_boosting", "extra_trees", "xgboost"]:
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500, step=50)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
        if model_type in ["random_forest", "extra_trees"]:
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        elif model_type == "gradient_boosting":
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        elif model_type == "xgboost":
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
            params["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
    elif model_type == "svr":
        params["svr_c"] = trial.suggest_float("svr_c", 0.1, 10.0, log=True)
        params["svr_epsilon"] = trial.suggest_float("svr_epsilon", 0.01, 1.0, log=True)
        params["svr_kernel"] = trial.suggest_categorical("svr_kernel", ["rbf", "linear", "poly"])
    elif model_type == "knn":
        params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 20)
        params["knn_weights"] = trial.suggest_categorical("knn_weights", ["uniform", "distance"])

    # Use the dynamically selected features
    features = selected_features

    df_feat = compute_features(df, params)
    df_feat['forward_return'] = (df_feat["ground_truth_spx"] / df_feat["spx"]) - 1.0
    valid_df = df_feat.dropna(subset=features + ['forward_return'])

    if len(valid_df) < n_fold + 1:
        return -1e9

    # OPTIMIZATION: Convert to NumPy ONCE to avoid pandas overhead in the CV loop
    X_all = valid_df[features].to_numpy(dtype=np.float32)
    y_all = valid_df['forward_return'].to_numpy(dtype=np.float32)

    tscv = TimeSeriesSplit(n_splits=n_fold)
    sharpe_scores = []

    for step, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        if len(X_train) < 3 or len(X_val) < 3:
            continue

        model, scaler = get_model_and_scaler(model_type, params)
        try:
            X_train_s = scaler.fit_transform(X_train) if scaler else X_train
            X_val_s = scaler.transform(X_val) if scaler else X_val

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)

            _, _, sharpe, _ = _calculate_metrics(y_val, y_pred, params)
            sharpe_scores.append(sharpe)

            # OPTIMIZATION: Optuna Pruning to skip bad trials early
            trial.report(np.mean(sharpe_scores), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        except Exception:
            continue

    return np.mean(sharpe_scores) if sharpe_scores else -1e9


def predict_latest_score(df: pd.DataFrame, params: dict, model_type: str, features: list, predict_next_period: bool = False) -> tuple:
    """Trains the final ML model on the ENTIRE dataset and outputs a scaled market score."""
    df_feat = compute_features(df, params)
    df_feat['forward_return'] = (df_feat["ground_truth_spx"] / df_feat["spx"]) - 1.0
    valid_df = df_feat.dropna(subset=features + ['forward_return'])

    X_train = valid_df[features].to_numpy(dtype=np.float32)
    y_train = valid_df['forward_return'].to_numpy(dtype=np.float32)

    model, scaler = get_model_and_scaler(model_type, params)
    X_train_scaled = scaler.fit_transform(X_train) if scaler else X_train
    model.fit(X_train_scaled, y_train)

    if predict_next_period:
        last_row = df_feat.iloc[[-1]]
        X_last = last_row[features].ffill().bfill()
        X_last_scaled = scaler.transform(X_last) if scaler else X_last.values
        pred_return = model.predict(X_last_scaled)[0]
        latest_heuristic_score = last_row['heuristic_market_score'].iloc[0]

        timeframe = params.get("timeframe", "day")
        if timeframe == "week":
            next_date = last_row.index[-1] + pd.Timedelta(weeks=1)
        elif timeframe == "month":
            next_date = last_row.index[-1] + pd.DateOffset(months=1)
        else:
            next_date = last_row.index[-1] + pd.Timedelta(days=1)
            if next_date.weekday() == 5:
                next_date += pd.Timedelta(days=2)
            elif next_date.weekday() == 6:
                next_date += pd.Timedelta(days=1)

        latest_date = next_date.strftime('%Y-%m-%d')
    else:
        last_valid_idx = valid_df.index[-1]
        X_last = X_train[[-1]]  # Optimized: direct numpy indexing
        X_last_scaled = scaler.transform(X_last) if scaler else X_last
        pred_return = model.predict(X_last_scaled)[0]
        latest_heuristic_score = valid_df['heuristic_market_score'].iloc[-1]
        latest_date = last_valid_idx.strftime('%Y-%m-%d')

    p95 = np.percentile(y_train, 95)
    p05 = np.percentile(y_train, 5)
    max_abs_ret = max(abs(p95), abs(p05))
    latest_ml_score = np.clip((pred_return / max_abs_ret) * 100, -100, 100) if max_abs_ret > 1e-6 else 0.0

    return latest_date, latest_heuristic_score, latest_ml_score


def entry(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    timeframe = args.timeframe

    if args.real_time:
        if not args.model_path:
            raise ValueError("--model-path is required when using --real-time")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}")

        print(f"Loading model from {args.model_path}...")
        model_data = joblib.load(args.model_path)
        timeframe = model_data['params']['timeframe']
        model = model_data["model"]
        scaler = model_data["scaler"]
        params = model_data["params"]
        model_type = model_data["model_type"]
        features = model_data["features"]
        max_abs_ret = model_data["max_abs_ret"]

        print("\n" + "=" * 73)
        print("MODEL PERFORMANCE SUMMARY (Train vs Test)")
        print("=" * 73)
        print(f"{'Metric':<22} | {'Train':<18} | {'Test':<18} | {'All':<18}")
        print("-" * 73)

        def fmt(val, is_pct=False):
            if val is None: return "N/A"
            return f"{val:.2%}" if is_pct else f"{val:.4f}"

        print(f"{'R-squared':<22} | {fmt(model_data.get('train_r2')):<18} | {fmt(model_data.get('test_r2')):<18} | {fmt(model_data.get('all_r2')):<18}")
        print(f"{'ML Win Rate':<22} | {fmt(model_data.get('train_wr'), True):<18} | {fmt(model_data.get('test_wr'), True):<18} | {fmt(model_data.get('all_wr'), True):<18}")
        print(f"{'ML Simulated Sharpe':<22} | {fmt(model_data.get('train_sharpe')):<18} | {fmt(model_data.get('test_sharpe')):<18} | {fmt(model_data.get('all_sharpe')):<18}")
        print("-" * 73)
        print(f"{'Consensus Win Rate':<22} | {fmt(model_data.get('train_cons_wr'), True):<18} | {fmt(model_data.get('test_cons_wr'), True):<18} | {fmt(model_data.get('all_cons_wr'), True):<18}")
        print(f"{'Consensus Sharpe':<22} | {fmt(model_data.get('train_cons_sharpe')):<18} | {fmt(model_data.get('test_cons_sharpe')):<18} | {fmt(model_data.get('all_cons_sharpe')):<18}")
        print("=" * 73 + "\n")

        print(f"{model_data.get('test_str_keeped')}")
        print(f"{model_data.get('test_str_keeped_cons')}")
        print("Fetching latest data...")
        df = load_data(filename=args.data_filename, timeframe=timeframe)

        print("Computing features for the latest datapoint...")
        df_feat = compute_features(df, params)

        last_row = df_feat.iloc[[-1]]
        X_last = last_row[features].ffill().bfill()
        X_last_scaled = scaler.transform(X_last) if scaler else X_last.values

        pred_return = model.predict(X_last_scaled)[0]
        latest_heuristic_score = last_row['heuristic_market_score'].iloc[0]

        if timeframe == "week":
            period_name, next_date = "Next Week's", last_row.index[-1] + pd.Timedelta(weeks=1)
        elif timeframe == "month":
            period_name, next_date = "Next Month's", last_row.index[-1] + pd.DateOffset(months=1)
        else:
            period_name = "Tomorrow's"
            next_date = last_row.index[-1] + pd.Timedelta(days=1)
            if next_date.weekday() == 5:
                next_date += pd.Timedelta(days=2)
            elif next_date.weekday() == 6:
                next_date += pd.Timedelta(days=1)

        tomorrow_date = next_date.strftime('%Y-%m-%d') if timeframe != "month" else next_date.strftime('%Y-%m')
        tomorrow_ml = np.clip((pred_return / max_abs_ret) * 100, -100, 100) if max_abs_ret > 1e-6 else 0.0

        print("\n" + "=" * 60)
        print(f"REAL-TIME {period_name.upper()} MARKET ANTICIPATION")
        print("=" * 60)
        print(f"Date of Prediction : {tomorrow_date}")
        print(f"Heuristic Score    : {latest_heuristic_score:+.1f} / 100")
        print(f"ML Predicted Score : {tomorrow_ml:+.1f} / 100  (Scaled {model_type.upper()} prediction)")
        tomorrow_consensus = (latest_heuristic_score + tomorrow_ml) / 2
        print(f"Consensus Score    : {tomorrow_consensus:+.1f} / 100")
        if tomorrow_consensus >= 30:
            print("Market Bias      : 🟢 BULLISH")
        elif tomorrow_consensus <= -30:
            print("Market Bias      : 🔴 BEARISH")
        else:
            print("Market Bias      : 🟡 NEUTRAL / CHOPPY")
        print("=" * 60)
        return

    # --- STANDARD OPTIMIZATION MODE ---
    train_val_split_ratio = float(args.train_val_split_ratio)
    lookahead_bars = int(args.lookahead_bars)
    epsilon = float(args.epsilon)
    density = float(args.density)
    n_trials = int(args.n_trials)
    timeout = int(args.timeout)
    model_type = args.model
    n_fold = int(args.n_fold)  # OPTIMIZATION: Use configurable n-fold

    max_features = args.max_features if args.max_features is not None else len(__FEATURES__)
    min_features = args.min_features if args.min_features != -1 else max_features

    df = load_data(filename=args.data_filename, timeframe=timeframe)
    idx = int(len(df) * train_val_split_ratio)
    df_train = df.iloc[:idx].copy()
    df_test = df.iloc[idx:].copy()

    print(f"Total Data    : {df.index[0].strftime('%Y-%m-%d')} :: {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} rows)")
    print(f"Train&Val Set : {df_train.index[0].strftime('%Y-%m-%d')} :: {df_train.index[-1].strftime('%Y-%m-%d')} ({len(df_train)} rows)")
    print(f"Test Set      : {df_test.index[0].strftime('%Y-%m-%d')} :: {df_test.index[-1].strftime('%Y-%m-%d')} ({len(df_test)} rows)")
    print(f"LA:{lookahead_bars} | Epsilon:{epsilon} | Density:{density} | Timeframe:{timeframe}")
    print(f"Features pool : {len(__FEATURES__)} available features")
    print(f"Feature selection: {min_features} to {max_features} features")
    print(f"\n>>> Optimizing {model_type.upper()} Model with Walk-Forward Validation (Maximizing Sharpe Ratio) <<<\n")

    # OPTIMIZATION: Add MedianPruner to skip poorly performing trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # Setup Sampler
    sampler = optuna.samplers.RandomSampler() if args.random_sampler else None

    # Create or load study
    study_kwargs = {
        "study_name": args.study_name,
        "direction": "maximize",
        "pruner": pruner,
        "sampler": sampler
    }
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)

    # If persisted, print best stats and candidates before running
    if args.storage and len(study.trials) > 0:
        print("\n" + "=" * 60)
        print("RESUMING PERSISTED STUDY")
        print("=" * 60)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            print(f"Total completed trials found: {len(completed_trials)}")
            print(f"Previous Best CV Mean Sharpe Ratio : {study.best_value:.4f}")
            print("Previous Best parameters:")
            for k, v in study.best_params.items():
                print(f"  {k}: {v}")
        else:
            print("Study exists in storage but has no completed trials yet.")
        print("=" * 60 + "\n")

    print(f"Starting Optuna optimization on Training Set ({n_fold}-fold TimeSeriesSplit)...")
    study.optimize(
        lambda trial: objective(trial=trial, df=df_train, n_fold=n_fold, model_type=model_type,
                                lookahead_bars=lookahead_bars, epsilon=epsilon, density=density,
                                timeframe=timeframe, min_features=min_features, max_features=max_features),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=args.verbose_study_progress_bar
    )

    best_params = study.best_params
    best_params["timeframe"] = timeframe
    best_model_type = model_type

    # Extract the best selected features from the optimal trial
    best_trial = study.best_trial
    best_num_features = best_trial.params["num_features"]
    feature_weights = {k.replace("fw_", ""): v for k, v in best_trial.params.items() if k.startswith("fw_")}
    best_features = sorted(feature_weights, key=feature_weights.get, reverse=True)[:best_num_features]

    print("\n--- Optimization Finished ---")
    print(f"Best CV Mean Sharpe Ratio  : {study.best_value:.4f}")
    print(f"Best Model Type            : {best_model_type.upper()}")
    print(f"Best features selected ({len(best_features)}): {best_features}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\n--- Evaluating on Training and Hold-Out Test Set ---")

    (train_r2, train_wr, train_sharpe, train_correct_direction,
     test_r2, test_wr, test_sharpe, test_correction_direction,
     train_cons_r2, train_cons_wr, train_cons_sharpe, train_cons_correct_direction,
     test_cons_r2, test_cons_wr, test_cons_sharpe, test_cons_correction_direction) = evaluate_datasets(df_train, df_test, best_params, best_features, best_model_type)

    (alldata_r2, alldata_wr, alldata_sharpe, alldata_correct_direction,
     train_r2_2, train_wr_2, train_sharpe_2, train_correction_direction_2,
     alldata_cons_r2, alldata_cons_wr, alldata_cons_sharpe, alldata_cons_correct_direction,
     train_cons_r2_2, train_cons_wr_2, train_cons_sharpe_2, train_cons_correction_direction_2) = evaluate_datasets(df, df_train, best_params, best_features, best_model_type)

    recomputed_test_win_rate = alldata_correct_direction.reindex(df_test.index).mean()
    recomputed_test_cons_win_rate = alldata_cons_correct_direction.reindex(df_test.index).mean()

    if train_wr in [0.] and test_wr in [0.] and train_r2 in [-1e9] and train_sharpe in [-1e9] and test_sharpe in [-1e9]:
        train_r2, train_wr, train_sharpe, train_correct_direction = train_r2_2, train_wr_2, train_sharpe_2, train_correction_direction_2
        test_wr = recomputed_test_win_rate
        train_cons_wr = train_cons_wr_2
        train_cons_sharpe = train_cons_sharpe_2
        test_cons_wr = recomputed_test_cons_win_rate

    _test_str_keeped = f"*** ML Win Rate on the Test Dataset : {recomputed_test_win_rate:.2%}  (recomputed) ({df_test.index[0].strftime('%Y-%m-%d')}::{df_test.index[-1].strftime('%Y-%m-%d')})"
    _test_str_keeped_cons = f"*** Consensus Win Rate on the Test Dataset : {recomputed_test_cons_win_rate:.2%}  (recomputed) ({df_test.index[0].strftime('%Y-%m-%d')}::{df_test.index[-1].strftime('%Y-%m-%d')})"

    print(f"* All DF Set R-squared           : {alldata_r2:.4f}")
    print(f"* All DF Set ML Win Rate         : {alldata_wr:.2%}")
    print(f"* All DF Set ML Sharpe           : {alldata_sharpe:.4f}")
    print(f"* All DF Set Consensus Win Rate  : {alldata_cons_wr:.2%}")
    print(f"* All DF Set Consensus Sharpe    : {alldata_cons_sharpe:.4f}")
    print(f"* Train Set R-squared            : {train_r2:.4f}")
    print(f"* Train Set ML Win Rate          : {train_wr:.2%}")
    print(f"* Train Set ML Sharpe            : {train_sharpe:.4f}")
    print(f"* Train Set Consensus Win Rate   : {train_cons_wr:.2%}")
    print(f"* Train Set Consensus Sharpe     : {train_cons_sharpe:.4f}")
    print(f"* Test Set R-squared             : {test_r2:.4f}")
    print(f"* Test Set ML Win Rate           : {test_wr:.2%}")
    print(f"* Test Set ML Sharpe             : {test_sharpe:.4f}")
    print(f"* Test Set Consensus Win Rate    : {test_cons_wr:.2%}")
    print(f"* Test Set Consensus Sharpe      : {test_cons_sharpe:.4f}")
    print(_test_str_keeped)
    print(_test_str_keeped_cons)

    print("\n" + "=" * 60)
    print("Saving the optimized model...")
    print("=" * 60)

    df_feat = compute_features(df, best_params)
    df_feat['forward_return'] = (df_feat["ground_truth_spx"] / df_feat["spx"]) - 1.0
    valid_df = df_feat.dropna(subset=best_features + ['forward_return'])

    X_train = valid_df[best_features].to_numpy(dtype=np.float32)
    y_train = valid_df['forward_return'].to_numpy(dtype=np.float32)

    final_model, final_scaler = get_model_and_scaler(best_model_type, best_params)
    X_train_scaled = final_scaler.fit_transform(X_train) if final_scaler else X_train
    final_model.fit(X_train_scaled, y_train)

    p95 = np.percentile(y_train, 95)
    p05 = np.percentile(y_train, 5)
    max_abs_ret = max(abs(p95), abs(p05))

    model_filename = f"market_model_{best_model_type}_{timeframe}_la{lookahead_bars}_eps{epsilon}_d{density}_{train_wr:.6f}_{test_wr:.6f}__{datetime.now().strftime('%Y%m%d')}.joblib"
    model_data = {
        "model": final_model, "scaler": final_scaler, "params": best_params, "model_type": best_model_type,
        "features": best_features, "max_abs_ret": max_abs_ret,
        "all_r2": alldata_r2, "all_wr": alldata_wr, "all_sharpe": alldata_sharpe,
        "train_r2": train_r2, "train_wr": train_wr, "train_sharpe": train_sharpe,
        "test_r2": test_r2, "test_wr": test_wr, "test_sharpe": test_sharpe,
        "recomputed_test_win_rate": recomputed_test_win_rate, "test_str_keeped": _test_str_keeped,
        "all_cons_wr": alldata_cons_wr, "all_cons_sharpe": alldata_cons_sharpe,
        "train_cons_wr": train_cons_wr, "train_cons_sharpe": train_cons_sharpe,
        "test_cons_wr": test_cons_wr, "test_cons_sharpe": test_cons_sharpe,
        "recomputed_test_cons_win_rate": recomputed_test_cons_win_rate, "test_str_keeped_cons": _test_str_keeped_cons,
    }
    joblib.dump(model_data, model_filename)
    print(f"Model successfully saved to '{model_filename}'")

    print("\n" + "=" * 60)
    print(f"LATEST MARKET PREDICTION (Retrained on Full Dataset)")
    print("=" * 60)
    latest_date, heuristic_score, ml_score = predict_latest_score(df, best_params, best_model_type, best_features, predict_next_period=False)
    print(f"Date of Prediction : {latest_date}")
    print(f"Heuristic Score    : {heuristic_score:+.1f} / 100")
    print(f"ML Predicted Score : {ml_score:+.1f} / 100  (Scaled {best_model_type.upper()} prediction)")
    consensus_score = (heuristic_score + ml_score) / 2
    print(f"Consensus Score    : {consensus_score:+.1f} / 100")
    if consensus_score >= 30:
        print("Market Bias      : 🟢 BULLISH")
    elif consensus_score <= -30:
        print("Market Bias      : 🔴 BEARISH")
    else:
        print("Market Bias      : 🟡 NEUTRAL / CHOPPY")

    if not args.skip_tomorrow:
        if timeframe == "week":
            period_name = "Next Week's"
        elif timeframe == "month":
            period_name = "Next Month's"
        else:
            period_name = "Tomorrow's"

        print("\n" + "=" * 60)
        print(f"{period_name.upper()} MARKET ANTICIPATION")
        print("=" * 60)
        tomorrow_date, tomorrow_heuristic, tomorrow_ml = predict_latest_score(df, best_params, best_model_type, best_features, predict_next_period=True)
        print(f"Date of Prediction : {tomorrow_date}")
        print(f"Heuristic Score    : {tomorrow_heuristic:+.1f} / 100")
        print(f"ML Predicted Score : {tomorrow_ml:+.1f} / 100  (Scaled {best_model_type.upper()} prediction)")
        tomorrow_consensus = (tomorrow_heuristic + tomorrow_ml) / 2
        print(f"Consensus Score    : {tomorrow_consensus:+.1f} / 100")
        if tomorrow_consensus >= 30:
            print("Market Bias      : 🟢 BULLISH")
        elif tomorrow_consensus <= -30:
            print("Market Bias      : 🔴 BEARISH")
        else:
            print("Market Bias      : 🟡 NEUTRAL / CHOPPY")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Next period market anticipation skipped (--skip-tomorrow).")
        print("=" * 60)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)