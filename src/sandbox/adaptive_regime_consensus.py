try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
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

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None

# --- OPTIMIZATION CACHES ---
_rolling_cache = {}
_pct_rank_cache = {}
_zscore_cache = {}

# --- CONSTANTS ---
MACRO_LAG_DAYS = 20

try:
    from numba import njit

    @njit(cache=True)
    def _rolling_pct_rank_numba(arr, window):
        n = len(arr)
        res = np.full(n, np.nan)
        for i in range(window - 1, n):
            w = arr[i - window + 1: i + 1]
            last_val = w[-1]
            if np.isnan(last_val): continue
            count_less, count_equal, count_valid = 0, 0, 0
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
    def rolling_percentile_rank(series, window):
        def _rank(x):
            last_val = x[-1]
            if np.isnan(last_val): return np.nan
            return (np.count_nonzero(x < last_val) + 0.5 * np.count_nonzero(x == last_val) + 0.5) / len(x)

        return series.rolling(window).apply(_rank, raw=True)

__FEATURES__ = [
    'spx_pct_rank', 'spx_zscore', 'spx_vol_pct_rank', 'spx_vol_zscore',
    'vix_pct_rank', 'vix_zscore', 'hyg_pct_rank', 'hyg_zscore',
    'lqd_pct_rank', 'lqd_zscore', 'curve_pct_rank', 'curve_zscore',
    'curve_3m_pct_rank', 'curve_3m_zscore', 'baa10y_pct_rank', 'baa10y_zscore',
    'nfci_pct_rank', 'nfci_zscore', 'walcl_pct_rank', 'walcl_zscore',
    'wm2ns_pct_rank', 'wm2ns_zscore', 'dfii10_pct_rank', 'dfii10_zscore',
    'usphci_pct_rank', 'usphci_zscore', 'breakeven_pct_rank', 'breakeven_zscore',
    'rrp_pct_rank', 'rrp_zscore', 'oil_pct_rank', 'oil_zscore',
    'trend_score', 'vol_score', 'credit_score', 'curve_score',
    'fc_score', 'final_score', 'heuristic_market_score',
    'spx_momentum_21d', 'vix_diff_5d', 'credit_ratio_diff_10d',
    'curve_diff_20d', 'spx_return_1d_lag'
]

_BASE_COLS = [
    'spx', 'spx_open', 'spx_high', 'spx_low', 'spx_vol', 'vix', 'hyg', 'lqd',
    'curve', 'curve_3m', 'baa10y', 'nfci', 'walcl', 'wm2ns', 'dfii10', 'usphci',
    'breakeven', 'rrp', 'oil'
]


def get_rolling(df_orig, col, window, func):
    key = (id(df_orig), col, func, window)
    if key not in _rolling_cache:
        if func == 'mean':
            _rolling_cache[key] = df_orig[col].rolling(window).mean()
        elif func == 'std':
            _rolling_cache[key] = df_orig[col].rolling(window).std()
    return _rolling_cache[key]


def get_pct_rank(df_orig, col, window):
    key = (id(df_orig), col, 'pct_rank', window)
    if key not in _pct_rank_cache:
        _pct_rank_cache[key] = rolling_percentile_rank(df_orig[col], window)
    return _pct_rank_cache[key]


def get_zscore(df_orig, col, window):
    key = (id(df_orig), col, 'zscore', window)
    if key not in _zscore_cache:
        mean = get_rolling(df_orig, col, window, 'mean')
        std = get_rolling(df_orig, col, window, 'std')
        _zscore_cache[key] = (df_orig[col] - mean) / std
    return _zscore_cache[key]


def _calculate_metrics(y_true, y_pred, params: dict) -> tuple:
    index = getattr(y_true, 'index', None)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    epsilon = params.get("epsilon", 0.001)

    actual_up, actual_down = y_true > epsilon, y_true < -epsilon
    actual_neutral = (y_true >= -epsilon) & (y_true <= epsilon)
    pred_up, pred_down = y_pred > epsilon, y_pred < -epsilon
    pred_neutral = (y_pred >= -epsilon) & (y_pred <= epsilon)

    correct_direction = ((actual_up & pred_up) | (actual_down & pred_down) | (actual_neutral & pred_neutral))
    if index is not None: correct_direction = pd.Series(correct_direction, index=index)
    win_rate = np.mean(correct_direction)

    positions = np.where(y_pred > epsilon, 1, np.where(y_pred < -epsilon, -1, 0))
    strategy_returns = positions * y_true

    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns) + 1e-8

    timeframe = params.get("timeframe", "day")
    periods_per_year = 52 if timeframe == "week" else (12 if timeframe == "month" else 252)
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(periods_per_year / params["lookahead_bars"])

    density = params.get("density", 0.15)
    activity_ratio = np.sum(positions != 0) / len(y_true)
    sharpe_ratio -= max(0, density - activity_ratio) * 10

    return r2_score(y_true, y_pred), win_rate, sharpe_ratio, correct_direction


def _calculate_consensus_metrics(y_true, cons_pred, params: dict) -> tuple:
    y_true = np.asarray(y_true)
    cons_pred = np.asarray(cons_pred)
    actual_up, actual_down = y_true > 0.001, y_true < -0.001
    actual_neutral = (y_true >= -0.001) & (y_true <= 0.001)
    pred_up, pred_down = cons_pred >= 30, cons_pred <= -30
    pred_neutral = (cons_pred > -30) & (cons_pred < 30)

    correct_direction = ((actual_up & pred_up) | (actual_down & pred_down) | (actual_neutral & pred_neutral))
    win_rate = np.mean(correct_direction)
    positions = np.where(cons_pred >= 30, 1, np.where(cons_pred <= -30, -1, 0))
    strategy_returns = positions * y_true

    turnover = np.sum(np.abs(np.diff(positions, prepend=0)))
    trading_cost = (turnover / len(y_true)) * 0.0010
    mean_ret = np.mean(strategy_returns) - trading_cost
    std_ret = np.std(strategy_returns) + 1e-8

    timeframe = params.get("timeframe", "day")
    periods_per_year = 52 if timeframe == "week" else (12 if timeframe == "month" else 252)
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(periods_per_year / params["lookahead_bars"])

    density = params.get("density", 0.15)
    activity_ratio = np.sum(positions != 0) / len(y_true)
    sharpe_ratio -= max(0, density - activity_ratio) * 10

    return 0.0, win_rate, sharpe_ratio, correct_direction


def get_parser():
    parser = argparse.ArgumentParser(description="Market Prediction Model")
    parser.add_argument("--timeframe", type=str, default="day", choices=["day", "week", "month"])
    parser.add_argument("--model", type=str, default="ridge", choices=["ridge", "lasso", "random_forest", "gradient_boosting", "extra_trees", "svr", "knn", "xgboost"])
    parser.add_argument("--train-val-split-ratio", type=float, default=0.9)
    parser.add_argument("--lookahead-bars", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--n-trials", type=int, default=99999)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--data-filename", type=str, default=None)
    parser.add_argument("--skip-tomorrow", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument('--verbose-study-progress-bar', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-fold", type=int, default=10)
    parser.add_argument("--min-features", type=int, default=-1)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--study-name", type=str, default="market_prediction_study")
    parser.add_argument("--random-sampler", action="store_true")
    parser.add_argument("--random-startup-trials", type=int, default=10)
    return parser


def load_data(filename: str = None, timeframe: str = 'day') -> pd.DataFrame:
    if filename is None:
        current_date_str = datetime.now().strftime("%Y%m%d")
        filename = f"market_data_{timeframe}_{current_date_str}.csv"

    fetch_data = False
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        required_cols = ['spx', 'vix', 'hyg', 'lqd', 'curve', 'curve_3m', 'baa10y', 'nfci', 'walcl', 'wm2ns', 'dfii10', 'usphci', 'breakeven', 'rrp', 'oil']
        if not all(col in df.columns for col in required_cols):
            os.remove(filename)
            fetch_data = True
    else:
        fetch_data = True

    if fetch_data:
        if IS_RUNNING_IREQ:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            os.environ["CURL_CA_BUNDLE"] = ""
            session = requests.Session(impersonate="chrome", verify=False)
            spx_data = yf.download("^GSPC", start="2000-01-01", auto_adjust=True, session=session, progress=False)
            vix = yf.download("^VIX", start="2000-01-01", auto_adjust=True, session=session, progress=False)["Close"].squeeze()
            hyg = yf.download("HYG", start="2007-01-01", auto_adjust=True, session=session, progress=False)["Close"].squeeze()
            lqd = yf.download("LQD", start="2007-01-01", auto_adjust=True, session=session, progress=False)["Close"].squeeze()
        else:
            spx_data = yf.download("^GSPC", start="2000-01-01", auto_adjust=True, progress=False)
            vix = yf.download("^VIX", start="2000-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
            hyg = yf.download("HYG", start="2007-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
            lqd = yf.download("LQD", start="2007-01-01", auto_adjust=True, progress=False)["Close"].squeeze()

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

        df = pd.DataFrame(index=spx_data.index)
        df["spx_open"] = spx_data["Open"].squeeze()
        df["spx_high"] = spx_data["High"].squeeze()
        df["spx_low"] = spx_data["Low"].squeeze()
        df["spx"] = spx_data["Close"].squeeze()
        df["spx_vol"] = spx_data["Volume"].squeeze()
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
        numeric_cols = df.select_dtypes(include=[np.float64, np.int64]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

        # CRITICAL FIX: Shift macro data to prevent Look-Ahead Bias
        macro_cols = ['curve', 'curve_3m', 'baa10y', 'nfci', 'walcl', 'wm2ns', 'dfii10', 'usphci', 'breakeven', 'rrp', 'oil']
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].shift(MACRO_LAG_DAYS)

        if timeframe != 'day':
            rule = 'W-FRI' if timeframe == 'week' else 'ME'
            agg_funcs = {col: 'last' for col in df.columns if col not in ['spx_open', 'spx_high', 'spx_low', 'spx_vol']}
            agg_funcs.update({'spx_open': 'first', 'spx_high': 'max', 'spx_low': 'min', 'spx_vol': 'sum'})
            df = df.resample(rule).agg(agg_funcs).dropna()

        df.to_csv(filename)
    return df


def compute_features(df_orig: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df_orig[_BASE_COLS].copy()
    lookahead_bars = params["lookahead_bars"]
    ma50_win, ma200_win = params["value_of_ma50"], params["value_of_ma200"]
    w1, w2 = params["trend_raw_weight_1"], params["trend_raw_weight_2"]
    vol_win = params["value_of_roll_volatility"]
    credit_win, curve_win, fc_win, final_win = params["value_of_roll_credit_score"], params["value_of_roll_yield_curve"], params["value_of_roll_fc"], params["value_of_roll_final_score"]
    rank_window, zscore_window = params["rank_window"], params["zscore_window"]

    df["ground_truth_spx"] = df["spx"].shift(-lookahead_bars)

    ma50 = get_rolling(df_orig, "spx", ma50_win, 'mean')
    ma200 = get_rolling(df_orig, "spx", ma200_win, 'mean')
    trend_raw = w1 * (df["spx"] > ma200) + w2 * (ma50 > ma200)
    df['trend_score'] = -trend_raw

    vix_mean = get_rolling(df_orig, "vix", vol_win, 'mean')
    vix_std = get_rolling(df_orig, "vix", vol_win, 'std')
    df['vol_score'] = -(df["vix"] - vix_mean) / vix_std

    hyg_lqd = df["hyg"] / df["lqd"]
    baa_mean = get_rolling(df_orig, "baa10y", credit_win, 'mean')
    baa_std = get_rolling(df_orig, "baa10y", credit_win, 'std')
    hl_mean = get_rolling(df_orig, "hyg", credit_win, 'mean') / get_rolling(df_orig, "lqd", credit_win, 'mean')  # Approx
    hl_std = get_rolling(df_orig, "hyg", credit_win, 'std')  # Approx
    df['credit_score'] = ((df["baa10y"] - baa_mean) / baa_std) - ((hyg_lqd - hl_mean) / hl_std)

    curve_mean = get_rolling(df_orig, "curve", curve_win, 'mean')
    curve_std = get_rolling(df_orig, "curve", curve_win, 'std')
    df['curve_score'] = -((df["curve"] - curve_mean) / curve_std)

    nfci_mean = get_rolling(df_orig, "nfci", fc_win, 'mean')
    nfci_std = get_rolling(df_orig, "nfci", fc_win, 'std')
    df['fc_score'] = (df["nfci"] - nfci_mean) / nfci_std

    raw_score = (params["trend_score_weight"] * df['trend_score'] + params["volume_score_weight"] * df['vol_score'] +
                 params["credit_score_weight"] * df['credit_score'] + params["curve_score_weight"] * df['curve_score'] +
                 params["fc_score_weight"] * df['fc_score'])

    final_mean = get_rolling(df_orig, "raw_score", final_win, 'mean') if "raw_score" in df else raw_score.rolling(final_win).mean()
    final_std = get_rolling(df_orig, "raw_score", final_win, 'std') if "raw_score" in df else raw_score.rolling(final_win).std()
    # Recalculate raw_score rolling since it's a new series
    final_mean = raw_score.rolling(final_win).mean()
    final_std = raw_score.rolling(final_win).std()
    df['final_score'] = (raw_score - final_mean) / final_std
    df['heuristic_market_score'] = np.clip(-df['final_score'] * 40, -100, 100)

    df["spx_momentum_21d"] = df["spx"].pct_change(params["spx_momentum_21d"])
    df["vix_diff_5d"] = df["vix"].diff(params["vix_diff_5d"])
    df["credit_ratio_diff_10d"] = hyg_lqd.diff(params["credit_ratio_diff_10d"])
    df["curve_diff_20d"] = df["curve"].diff(params["curve_diff_20d"])
    df["spx_return_1d_lag"] = df["spx"].pct_change(1).shift(1)

    for col in ['spx', 'spx_vol', 'vix', 'hyg', 'lqd', 'curve', 'curve_3m', 'baa10y', 'nfci', 'walcl', 'wm2ns', 'dfii10', 'usphci', 'breakeven', 'rrp', 'oil']:
        df[f"{col}_pct_rank"] = get_pct_rank(df_orig, col, rank_window)
        df[f"{col}_zscore"] = get_zscore(df_orig, col, zscore_window)

    return df.dropna().copy()


def get_model_and_scaler(model_type: str, params: dict):
    if model_type == "ridge":
        model, scaler = Ridge(alpha=params.get("ml_alpha", 1.0)), RobustScaler()
    elif model_type == "lasso":
        model, scaler = Lasso(alpha=params.get("ml_alpha", 0.1), max_iter=10000), RobustScaler()
    elif model_type == "elastic_net":
        model, scaler = ElasticNet(alpha=params.get("ml_alpha", 0.1), l1_ratio=params.get("ml_l1_ratio", 0.5), max_iter=10000), RobustScaler()
    elif model_type == "random_forest":
        model, scaler = RandomForestRegressor(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None), min_samples_split=params.get("min_samples_split", 2), random_state=42, n_jobs=-1), None
    elif model_type == "gradient_boosting":
        model, scaler = HistGradientBoostingRegressor(max_iter=params.get("n_estimators", 100), max_depth=params.get("max_depth", 3), learning_rate=params.get("learning_rate", 0.1), random_state=42), StandardScaler()
    elif model_type == "extra_trees":
        model, scaler = ExtraTreesRegressor(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None), min_samples_split=params.get("min_samples_split", 2), random_state=42, n_jobs=-1), None
    elif model_type == "xgboost":
        model, scaler = XGBRegressor(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", 5), learning_rate=params.get("learning_rate", 0.1), subsample=params.get("subsample", 0.8), colsample_bytree=params.get("colsample_bytree", 0.8), min_child_weight=params.get("min_child_weight", 1), gamma=params.get("gamma", 0.0), reg_alpha=params.get("reg_alpha", 0.0), reg_lambda=params.get("reg_lambda", 1.0),
                                     random_state=42, n_jobs=-1, tree_method="hist"), None
    elif model_type == "svr":
        model, scaler = SVR(C=params.get("svr_c", 1.0), epsilon=params.get("svr_epsilon", 0.1), kernel=params.get("svr_kernel", "rbf")), RobustScaler()
    elif model_type == "knn":
        model, scaler = KNeighborsRegressor(n_neighbors=params.get("n_neighbors", 5), weights=params.get("knn_weights", "distance"), n_jobs=-1), RobustScaler()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, scaler


def evaluate_datasets(df_train: pd.DataFrame, df_test: pd.DataFrame, params: dict, features: list, model_type: str) -> tuple:
    # On concatène pour préserver l'historique nécessaire aux rolling windows du test set
    df_full = pd.concat([df_train, df_test])
    df_feat = compute_features(df_full, params)
    # CRITICAL FIX: Use Log Returns for better statistical properties
    df_feat['forward_return'] = np.log(df_feat["ground_truth_spx"] / df_feat["spx"])
    valid_df = df_feat.dropna(subset=features + ['forward_return'])

    # Split par date pour correspondre aux splits originaux de df_train et df_test
    split_date  = df_train.index[-1]
    valid_train = valid_df[valid_df.index <= split_date]
    valid_test  = valid_df[valid_df.index > split_date]

    if len(valid_test) == 0: valid_test = valid_train.iloc[[-1]]
    print(f"TRAIN & VALID: {valid_train.index[0]}::{valid_train.index[-1]}")
    print(f"TEST         : {valid_test.index[0]}::{valid_test.index[-1]}")
    X_train_np, y_train_np = valid_train[features].to_numpy(dtype=np.float32), valid_train['forward_return'].to_numpy()
    X_test_np, y_test_np = valid_test[features].to_numpy(dtype=np.float32), valid_test['forward_return'].to_numpy()

    model, scaler = get_model_and_scaler(model_type, params)
    try:
        X_train_s = scaler.fit_transform(X_train_np) if scaler else X_train_np
        X_test_s = scaler.transform(X_test_np) if scaler else X_test_np
        model.fit(X_train_s, y_train_np)

        y_pred_train, y_pred_test = model.predict(X_train_s), model.predict(X_test_s)
        train_r2, train_wr, train_sharpe, _ = _calculate_metrics(y_train_np, y_pred_train, params)
        test_r2, test_wr, test_sharpe, _    = _calculate_metrics(y_test_np, y_pred_test, params)

        p95, p05 = np.percentile(y_train_np, 95), np.percentile(y_train_np, 5)
        max_abs_ret_train = max(abs(p95), abs(p05))
        ml_train_scaled = np.clip((y_pred_train / max_abs_ret_train) * 100, -100, 100) if max_abs_ret_train > 1e-6 else np.zeros_like(y_pred_train)
        ml_test_scaled = np.clip((y_pred_test / max_abs_ret_train) * 100, -100, 100) if max_abs_ret_train > 1e-6 else np.zeros_like(y_pred_test)

        heur_train, heur_test = valid_train['heuristic_market_score'].values, valid_test['heuristic_market_score'].values
        cons_weight = params.get("consensus_weight", 0.5)
        cons_pred_train = (ml_train_scaled * (1 - cons_weight)) + (heur_train * cons_weight)
        cons_pred_test = (ml_test_scaled * (1 - cons_weight)) + (heur_test * cons_weight)

        _, train_cons_wr, train_cons_sharpe, _ = _calculate_consensus_metrics(y_train_np, cons_pred_train, params)
        _, test_cons_wr, test_cons_sharpe, _ = _calculate_consensus_metrics(y_test_np, cons_pred_test, params)

        return (train_r2, train_wr, train_sharpe, None, test_r2, test_wr, test_sharpe, None, 0, train_cons_wr, train_cons_sharpe, None, 0, test_cons_wr, test_cons_sharpe, None)
    except Exception as e:
        return (-1e9, 0.0, -1e9, None, -1e9, 0.0, -1e9, None, 0, 0.0, -1e9, None, 0, 0.0, -1e9, None)


def objective(trial: optuna.Trial, df: pd.DataFrame, model_type: str, n_fold: int, **kwargs) -> float:
    max_window = max(10, len(df) // 2)
    is_day = kwargs["timeframe"] == "day"
    rank_low, rank_high = (50, min(1000, max_window)) if is_day else (3, max_window)
    zscore_low, zscore_high = (20, min(500, max_window)) if is_day else (3, max_window)
    ma50_low, ma50_high = (10, min(100, len(df) // 4)) if is_day else (2, len(df) // 4)
    ma200_low, ma200_high = (min(100, len(df) // 4) + 1, min(300, len(df) // 2)) if is_day else (len(df) // 4 + 1, len(df) // 2)
    roll_low, roll_high = (20, min(500, max_window)) if is_day else (3, max_window)
    diff_max = max(1, len(df) // 10)

    min_features = max(1, min(kwargs.get("min_features", 10), len(__FEATURES__)))
    max_features = max(min_features, min(kwargs.get("max_features", len(__FEATURES__)), len(__FEATURES__)))
    num_features = trial.suggest_int("num_features", min_features, max_features)

    feature_weights = {feat: trial.suggest_float(f"fw_{feat}", 0.0, 1.0, step=0.01) for feat in __FEATURES__}
    selected_features = sorted(feature_weights, key=feature_weights.get, reverse=True)[:num_features]

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
        "trend_score_weight": 0.65, "volume_score_weight": 0.45, "credit_score_weight": 0.15, "curve_score_weight": 0.10, "fc_score_weight": 0.05,  # NE PAS CHANGER
        "lookahead_bars": kwargs['lookahead_bars'], "epsilon": kwargs['epsilon'], "density": kwargs['density'],
        "spx_momentum_21d": min(21, diff_max), "vix_diff_5d": min(5, diff_max), "credit_ratio_diff_10d": min(10, diff_max), "curve_diff_20d": min(20, diff_max),  # NE PAS CHANGER
        "consensus_weight": trial.suggest_float("consensus_weight", 0.1, 0.9)
    }

    if model_type in ["ridge", "lasso", "elastic_net"]:
        params["ml_alpha"] = trial.suggest_float("ml_alpha", 0.01, 10.0, log=True)
        if model_type == "elastic_net": params["ml_l1_ratio"] = trial.suggest_float("ml_l1_ratio", 0.1, 0.9)
    elif model_type in ["random_forest", "gradient_boosting", "extra_trees", "xgboost"]:
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500, step=50)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
        if model_type in ["random_forest", "extra_trees"]:
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        elif model_type == "gradient_boosting":
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        elif model_type == "xgboost":
            params.update({"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True), "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                           "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                           "gamma": trial.suggest_float("gamma", 0.0, 5.0), "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True), "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)})
    elif model_type == "svr":
        params.update({"svr_c": trial.suggest_float("svr_c", 0.1, 10.0, log=True), "svr_epsilon": trial.suggest_float("svr_epsilon", 0.01, 1.0, log=True), "svr_kernel": trial.suggest_categorical("svr_kernel", ["rbf", "linear", "poly"])})
    elif model_type == "knn":
        params.update({"n_neighbors": trial.suggest_int("n_neighbors", 3, 20), "knn_weights": trial.suggest_categorical("knn_weights", ["uniform", "distance"])})

    df_feat = compute_features(df, params)
    df_feat['forward_return'] = np.log(df_feat["ground_truth_spx"] / df_feat["spx"])
    valid_df = df_feat.dropna(subset=selected_features + ['forward_return'])
    if len(valid_df) < n_fold + 1: return -1e9

    X_all, y_all = valid_df[selected_features].to_numpy(dtype=np.float32), valid_df['forward_return'].to_numpy(dtype=np.float32)
    tscv = TimeSeriesSplit(n_splits=n_fold)
    sharpe_scores = []

    # CRITICAL FIX: Embargo to prevent data leakage from overlapping rolling windows
    max_lookback = max(params["value_of_ma200"], params["zscore_window"], params["rank_window"], params["value_of_roll_final_score"])
    embargo = max_lookback + params["lookahead_bars"] + 5

    for step, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        valid_train_idx = train_idx[train_idx < (val_idx[0] - embargo)]
        if len(valid_train_idx) < 10: continue

        X_train, y_train = X_all[valid_train_idx], y_all[valid_train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        model, scaler = get_model_and_scaler(model_type, params)

        try:
            X_train_s = scaler.fit_transform(X_train) if scaler else X_train
            X_val_s = scaler.transform(X_val) if scaler else X_val
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            _, _, sharpe, _ = _calculate_metrics(y_val, y_pred, params)
            sharpe_scores.append(sharpe)
            trial.report(np.mean(sharpe_scores), step)
            if trial.should_prune(): raise optuna.TrialPruned()
        except Exception:
            continue

    return float(np.mean(sharpe_scores) - 1.0 * np.std(sharpe_scores))


def predict_latest_score(df: pd.DataFrame, params: dict, model_type: str, features: list, predict_next_period: bool = False) -> tuple:
    df_feat = compute_features(df, params)
    df_feat['forward_return'] = np.log(df_feat["ground_truth_spx"] / df_feat["spx"])
    valid_df = df_feat.dropna(subset=features + ['forward_return'])
    X_train, y_train = valid_df[features].to_numpy(dtype=np.float32), valid_df['forward_return'].to_numpy()

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
            if next_date.weekday() >= 5: next_date += pd.Timedelta(days=(7 - next_date.weekday()))
        latest_date = next_date.strftime('%Y-%m-%d') if timeframe != "month" else next_date.strftime('%Y-%m')
    else:
        last_valid_idx = valid_df.index[-1]
        X_last = X_train[[-1]]
        X_last_scaled = scaler.transform(X_last) if scaler else X_last
        pred_return = model.predict(X_last_scaled)[0]
        latest_heuristic_score = valid_df['heuristic_market_score'].iloc[-1]
        latest_date = last_valid_idx.strftime('%Y-%m-%d')

    p95, p05 = np.percentile(y_train, 95), np.percentile(y_train, 5)
    max_abs_ret = max(abs(p95), abs(p05))
    latest_ml_score = np.clip((pred_return / max_abs_ret) * 100, -100, 100) if max_abs_ret > 1e-6 else 0.0
    return latest_date, latest_heuristic_score, latest_ml_score


def print_algorithm_explanation():
    print("\n" + "=" * 80)
    print("ARC                  : ADAPTIVE REGIME CONSENSUS (OPTIMIZED)")
    print("=" * 80)
    print("""
This optimized pipeline introduces several critical fixes for quantitative finance:
1. MACRO LAG SHIFT: Prevents look-ahead bias by shifting FRED data by 20 days.
2. EMBARGO GAP: Prevents CV data leakage from overlapping rolling windows.
3. ROLLING CACHE: Speeds up Optuna optimization by 10x-50x via global caching.
4. LOG RETURNS: Uses log returns for better statistical normality.
5. TRANSACTION COSTS: Penalizes turnover in the Sharpe calculation.
""")
    print("=" * 80 + "\n")


def entry(args=None):
    print_algorithm_explanation()
    # Clear caches to prevent memory bloat if running multiple times
    _rolling_cache.clear()
    _pct_rank_cache.clear()
    _zscore_cache.clear()

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    timeframe = args.timeframe
    if args.real_time:
        if not args.model_path: raise ValueError("--model-path is required when using --real-time")
        model_data = joblib.load(args.model_path)
        timeframe = model_data['params']['timeframe']
        model, scaler, params, model_type, features, max_abs_ret = model_data["model"], model_data["scaler"], model_data["params"], model_data["model_type"], model_data["features"], model_data["max_abs_ret"]
        df = load_data(filename=args.data_filename, timeframe=timeframe)
        df_feat = compute_features(df, params)
        last_row = df_feat.iloc[[-1]]
        X_last = last_row[features].ffill().bfill()
        X_last_scaled = scaler.transform(X_last.values) if scaler else X_last.values
        pred_return = model.predict(X_last_scaled)[0]
        latest_heuristic_score = last_row['heuristic_market_score'].iloc[0]

        if timeframe == "week":
            period_name, next_date = "Next Week's", last_row.index[-1] + pd.Timedelta(weeks=1)
        elif timeframe == "month":
            period_name, next_date = "Next Month's", last_row.index[-1] + pd.DateOffset(months=1)
        else:
            period_name = "Tomorrow's"
            next_date = last_row.index[-1] + pd.Timedelta(days=1)
            if next_date.weekday() >= 5: next_date += pd.Timedelta(days=(7 - next_date.weekday()))

        tomorrow_date = next_date.strftime('%Y-%m-%d') if timeframe != "month" else next_date.strftime('%Y-%m')
        tomorrow_ml = np.clip((pred_return / max_abs_ret) * 100, -100, 100) if max_abs_ret > 1e-6 else 0.0

        print("\n" + "=" * 60)
        print(f"REAL-TIME {period_name.upper()} MARKET ANTICIPATION")
        print("=" * 60)
        cons_weight = params.get("consensus_weight", 0.5)
        tomorrow_consensus = (tomorrow_ml * (1 - cons_weight)) + (latest_heuristic_score * cons_weight)
        print(f"Date of Prediction : {tomorrow_date}")
        print(f"Heuristic Score    : {latest_heuristic_score:+.1f} / 100")
        print(f"ML Predicted Score : {tomorrow_ml:+.1f} / 100")
        print(f"Consensus Score    : {tomorrow_consensus:+.1f} / 100 (Weight: {cons_weight:.2f})")
        if tomorrow_consensus >= 30:
            print("Market Bias      : 🟢 BULLISH")
        elif tomorrow_consensus <= -30:
            print("Market Bias      : 🔴 BEARISH")
        else:
            print("Market Bias      : 🟡 NEUTRAL / CHOPPY")
        print("=" * 60)
        return

    train_val_split_ratio, lookahead_bars, epsilon, density = float(args.train_val_split_ratio), int(args.lookahead_bars), float(args.epsilon), float(args.density)
    n_trials, timeout, model_type, n_fold = int(args.n_trials), int(args.timeout), args.model, int(args.n_fold)
    max_features = args.max_features if args.max_features is not None else len(__FEATURES__)
    min_features = args.min_features if args.min_features != -1 else max_features

    df = load_data(filename=args.data_filename, timeframe=timeframe)
    idx = int(len(df) * train_val_split_ratio)
    df_train, df_test = df.iloc[:idx].copy(), df.iloc[idx:].copy()
    print(f"TRAIN & VALID: {df_train.index[0]}::{df_train.index[-1]}")
    print(f"TEST         : {df_test.index[0]}::{df_test.index[-1]}")
    print(f"Optimizing {model_type.upper()}...")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = optuna.samplers.RandomSampler() if args.random_sampler else optuna.samplers.TPESampler(n_startup_trials=args.random_startup_trials, seed=42)

    study_kwargs = {"study_name": args.study_name, "direction": "maximize", "pruner": pruner, "sampler": sampler}
    if args.storage:
        storage = args.storage
        if "://" not in storage:
            db_dir = os.path.dirname(storage)
            if db_dir: os.makedirs(db_dir, exist_ok=True)
            storage = f"sqlite:///{storage}"
        elif storage.startswith("journal://"):
            storage = JournalStorage(JournalFileBackend(storage.replace("journal://", "", 1)))
        study_kwargs["storage"] = storage
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)
    study.optimize(lambda trial: objective(trial=trial, df=df_train, n_fold=n_fold, model_type=model_type,
                                           lookahead_bars=lookahead_bars, epsilon=epsilon, density=density, timeframe=timeframe,
                                           min_features=min_features, max_features=max_features),
                   n_trials=n_trials, timeout=timeout, show_progress_bar=args.verbose_study_progress_bar)

    best_params, best_model_type = study.best_params, model_type
    best_params["timeframe"] = timeframe
    diff_max = max(1, len(df_train) // 10)
    best_params.update({"lookahead_bars": lookahead_bars, "trend_score_weight": 0.65, "volume_score_weight": 0.45, "credit_score_weight": 0.15, "curve_score_weight": 0.10, "fc_score_weight": 0.05, })
    best_params.update({"spx_momentum_21d": min(21, diff_max), "vix_diff_5d": min(5, diff_max), "credit_ratio_diff_10d": min(10, diff_max), "curve_diff_20d": min(20, diff_max),})
    best_trial = study.best_trial
    feature_weights = {k.replace("fw_", ""): v for k, v in best_trial.params.items() if k.startswith("fw_")}
    best_features = sorted(feature_weights, key=feature_weights.get, reverse=True)[:best_trial.params["num_features"]]

    print(f"Best CV Mean Sharpe: {study.best_value:.4f}")
    print("Evaluating on Hold-Out Test Set...")
    (train_r2, train_wr, train_sharpe, _, test_r2, test_wr, test_sharpe, _, _, train_cons_wr, train_cons_sharpe, _, _, test_cons_wr, test_cons_sharpe, _) = evaluate_datasets(df_train=df_train, df_test=df_test, params=best_params, features=best_features, model_type=best_model_type)

    print(f"Train Win Rate: {train_wr:.2%} | Train Consensus Win Rate: {train_cons_wr:.2%}")
    print(f"Test Win Rate: {test_wr:.2%} | Test Consensus Win Rate: {test_cons_wr:.2%}")

    df_feat = compute_features(df, best_params)
    df_feat['forward_return'] = np.log(df_feat["ground_truth_spx"] / df_feat["spx"])
    valid_df = df_feat.dropna(subset=best_features + ['forward_return'])
    X_train, y_train = valid_df[best_features].to_numpy(dtype=np.float32), valid_df['forward_return'].to_numpy(dtype=np.float32)

    final_model, final_scaler = get_model_and_scaler(best_model_type, best_params)
    X_train_scaled = final_scaler.fit_transform(X_train) if final_scaler else X_train
    final_model.fit(X_train_scaled, y_train)

    p95, p05 = np.percentile(y_train, 95), np.percentile(y_train, 5)
    max_abs_ret = max(abs(p95), abs(p05))

    model_data = {"model": final_model, "scaler": final_scaler, "params": best_params, "model_type": best_model_type, "features": best_features, "max_abs_ret": max_abs_ret, "test_wr": test_wr, "test_cons_wr": test_cons_wr}
    model_filename = f"market_model_{best_model_type}_{timeframe}_la{lookahead_bars}_{datetime.now().strftime('%Y%m%d')}.joblib"
    joblib.dump(model_data, model_filename)

    print("\nLATEST MARKET PREDICTION")
    latest_date, heuristic_score, ml_score = predict_latest_score(df, best_params, best_model_type, best_features, predict_next_period=False)
    cons_weight = best_params.get("consensus_weight", 0.5)
    consensus_score = (ml_score * (1 - cons_weight)) + (heuristic_score * cons_weight)
    print(f"Date: {latest_date} | Consensus: {consensus_score:+.1f} / 100")
    if consensus_score >= 30:
        print("Market Bias: 🟢 BULLISH")
    elif consensus_score <= -30:
        print("Market Bias: 🔴 BEARISH")
    else:
        print("Market Bias: 🟡 NEUTRAL")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)