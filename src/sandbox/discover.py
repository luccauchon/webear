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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from curl_cffi import requests
import urllib3
from constants import FRED_API_KEY, IS_RUNNING_IREQ

# Suppress Optuna & pandas debug logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None

# OPTIMIZATION 3: Removed duplicate features to save memory and computation
__FEATURES__ = [
    'spx_pct_rank', 'spx_zscore',
    'spx_vol_pct_rank', 'spx_vol_zscore',  # <-- ADDED: SPX Volume Features
    'vix_pct_rank', 'vix_zscore',
    'hyg_pct_rank', 'hyg_zscore',
    'lqd_pct_rank', 'lqd_zscore',
    'curve_pct_rank', 'curve_zscore',
    'baa10y_pct_rank', 'baa10y_zscore',
    'nfci_pct_rank', 'nfci_zscore',
    'trend_score', 'vol_score', 'credit_score', 'curve_score',
    'fc_score', 'final_score', 'heuristic_market_score',
    'spx_momentum_21d', 'vix_diff_5d', 'credit_ratio_diff_10d',
    'curve_diff_20d', 'spx_return_1d_lag'
]


def _calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, params: dict) -> tuple:
    """Helper to calculate R2, Win Rate, and Sharpe for a given true/pred pair."""
    epsilon = params.get("epsilon", 0.001)

    actual_up = y_true > epsilon
    actual_down = y_true < -epsilon
    actual_neutral = (y_true >= -epsilon) & (y_true <= epsilon)

    pred_up = y_pred > epsilon
    pred_down = y_pred < -epsilon
    pred_neutral = (y_pred >= -epsilon) & (y_pred <= epsilon)

    correct_direction = ((actual_up & pred_up) | (actual_down & pred_down) | (actual_neutral & pred_neutral))
    win_rate = np.mean(correct_direction)

    positions = np.where(y_pred > epsilon, 1, np.where(y_pred < -epsilon, -1, 0))
    strategy_returns = positions * y_true.values

    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns) + 1e-8
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252 / params["lookahead_bars"])

    density = params.get("density", 0.15)
    active_days = np.sum(positions != 0)
    activity_ratio = active_days / len(y_true)
    penalty = max(0, density - activity_ratio)
    sharpe_ratio -= penalty * 10

    return r2_score(y_true, y_pred), win_rate, sharpe_ratio


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Market Prediction Model with Optuna Optimization and Walk-Forward Validation"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["ridge", "lasso", "random_forest", "gradient_boosting", "extra_trees", "svr", "knn", "xgboost"],
        help="Model to use for prediction. Options: ridge, lasso, random_forest, gradient_boosting, extra_trees, svr, knn, xgboost (default: random_forest)"
    )
    parser.add_argument(
        "--train-val-split-ratio",
        type=float,
        default=0.7,
        help="Ratio of data to use for training and validation (default: 0.7)"
    )
    parser.add_argument(
        "--lookahead-bars",
        type=int,
        default=1,
        help="Number of lookahead bars for the target variable (default: 1)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Threshold for directional win rate calculation (default: 0.001)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.15,
        help="Target density for active trading days to penalize low activity (default: 0.15)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=99999,
        help="Maximum number of trials for Optuna optimization (default: 99)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for Optuna optimization (default: 300)"
    )
    parser.add_argument(
        "--data-filename",
        type=str,
        default=None,
        help="Custom filename for the market data CSV. Defaults to 'market_data_YYYYMMDD.csv' if not provided."
    )
    parser.add_argument(
        "--skip-tomorrow",
        action="store_true",
        help="Skip the 'Tomorrow's Market Anticipation' section in the final output."
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Run in real-time prediction mode using a saved model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the saved model file (required if --real-time is used)."
    )

    return parser


def load_data(filename: str = None) -> pd.DataFrame:
    """Loads market data from disk or fetches it from Yahoo Finance and FRED."""
    if filename is None:
        current_date_str = datetime.now().strftime("%Y%m%d")
        filename = f"market_data_{current_date_str}.csv"

    if os.path.exists(filename):
        print(f"File '{filename}' exists. Reloading data from disk...")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        if IS_RUNNING_IREQ:
            # Globally disable SSL warnings in the console
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            # Patch FRED's underlying environment variable before initializing
            os.environ["CURL_CA_BUNDLE"] = ""

            # Create a curl_cffi Session that bypasses SSL and impersonates Chrome
            session = requests.Session(impersonate="chrome", verify=False)

            # Yahoo Finance Data (Passing the non-verified curl_cffi session)
            spx_data = yf.download("^GSPC", start="2000-01-01", auto_adjust=True, session=session)
            spx, spx_vol = spx_data["Close"].squeeze(), spx_data["Volume"].squeeze()
            vix = yf.download("^VIX", start="2000-01-01", auto_adjust=True, session=session)["Close"].squeeze()
            hyg = yf.download("HYG", start="2007-01-01", auto_adjust=True, session=session)["Close"].squeeze()
            lqd = yf.download("LQD", start="2007-01-01", auto_adjust=True, session=session)["Close"].squeeze()
        else:
            # Yahoo Finance Data
            spx_data = yf.download("^GSPC", start="2000-01-01", auto_adjust=True)
            spx, spx_vol = spx_data["Close"].squeeze(), spx_data["Volume"].squeeze()
            vix = yf.download("^VIX", start="2000-01-01", auto_adjust=True)["Close"].squeeze()
            hyg = yf.download("HYG", start="2007-01-01", auto_adjust=True)["Close"].squeeze()
            lqd = yf.download("LQD", start="2007-01-01", auto_adjust=True)["Close"].squeeze()

        print(f"File '{filename}' not found. Fetching new data...")
        fred = Fred(api_key=FRED_API_KEY)

        # FRED Macroeconomic Data
        curve = fred.get_series("T10Y2Y")
        baa10y = fred.get_series("BAA10Y")
        nfci = fred.get_series("NFCI")

        # Align and merge data
        df = pd.DataFrame(index=spx.index)
        df["spx"] = spx
        df["spx_vol"] = spx_vol
        df["vix"] = vix
        df["hyg"] = hyg
        df["lqd"] = lqd
        df["curve"] = curve
        df["baa10y"] = baa10y
        df["nfci"] = nfci

        # Filter to post-2008 financial crisis data and forward-fill missing values
        df = df.loc["2008-01-01":].ffill()
        df.to_csv(filename)
        print(f"Data successfully saved to '{filename}'")

    return df


# OPTIMIZATION 1: Pure NumPy rolling percentile rank (10x-50x faster)
def rolling_percentile_rank(series, window):
    def _rank(x):
        last_val = x[-1]
        if np.isnan(last_val):
            return np.nan

        # Filter out NaNs to exactly match pandas rank behavior
        valid_x = x[~np.isnan(x)]
        n_valid = len(valid_x)
        if n_valid == 0:
            return np.nan

        last_val = valid_x[-1]
        # Calculate average rank for ties: 1 + sum(<) + 0.5 * (sum(==) - 1)
        rank = 1.0 + np.sum(valid_x < last_val) + 0.5 * (np.sum(valid_x == last_val) - 1.0)
        return rank / n_valid

    # raw=True passes numpy arrays directly, avoiding Pandas Series instantiation overhead
    return series.rolling(window).apply(_rank, raw=True)


def rolling_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def compute_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Computes all feature scores and new momentum/diff features for a given dataset."""
    df = df.copy()

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

    # Target variable
    df["ground_truth_spx"] = df["spx"].shift(-lookahead_bars)

    # 1. Trend Score
    ma50 = df["spx"].rolling(ma50_win).mean()
    ma200 = df["spx"].rolling(ma200_win).mean()
    # OPTIMIZATION 4: Removed .astype(int). NumPy handles bool * float natively and faster.
    trend_raw = w1 * (df["spx"] > ma200) + w2 * (ma50 > ma200)
    trend_score = -trend_raw  # Inverted so higher is better (bullish)

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

    # Composite Score
    raw_score = (trend_score_weight * trend_score +
                 volume_score_weight * vol_score +
                 credit_score_weight * credit_score +
                 curve_score_weight * curve_score +
                 fc_score_weight * fc_score)

    final_score = (raw_score - raw_score.rolling(final_win).mean()) / raw_score.rolling(final_win).std()

    # --- NEW PERTINENT FEATURES (Momentum, Diff, Shift) ---
    df["spx_momentum_21d"] = df["spx"].pct_change(21)
    df["vix_diff_5d"] = df["vix"].diff(5)
    df["credit_ratio_diff_10d"] = hyg_lqd.diff(10)
    df["curve_diff_20d"] = df["curve"].diff(20)
    df["spx_return_1d_lag"] = df["spx"].pct_change(1).shift(1)
    # --------------------------------------------------------

    # Assign base scores to dataframe
    df['trend_score'] = trend_score
    df['vol_score'] = vol_score
    df['credit_score'] = credit_score
    df['curve_score'] = curve_score
    df['fc_score'] = fc_score
    df['final_score'] = final_score
    df['heuristic_market_score'] = np.clip(-final_score * 40, -100, 100)

    # ======================================================
    # Regime-normalized market and macro features
    # ======================================================
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

    return df


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
        scaler = None  # Tree-based models are scale-invariant
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42
        )
        scaler = StandardScaler()  # Benefits from scaling
        # scaler = RobustScaler()  # to be tested
    elif model_type == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1
        )
        scaler = None  # Tree-based models are scale-invariant
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
        scaler = None  # Tree-based models are scale-invariant
    elif model_type == "svr":
        model = SVR(
            C=params.get("svr_c", 1.0),
            epsilon=params.get("svr_epsilon", 0.1),
            kernel=params.get("svr_kernel", "rbf")
        )
        scaler = RobustScaler()  # Required for SVR
    elif model_type == "knn":
        model = KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("knn_weights", "distance"),
            n_jobs=-1
        )
        scaler = RobustScaler()  # Required for KNN
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, scaler


def evaluate_datasets(df_train: pd.DataFrame, df_test: pd.DataFrame, params: dict, features: list, model_type: str) -> tuple:
    """Fits model on TRAIN set and evaluates on BOTH train and TEST sets properly."""
    # Process Train
    df_train_feat = compute_features(df_train, params)
    df_train_feat['forward_return'] = (df_train_feat["ground_truth_spx"] / df_train_feat["spx"]) - 1.0
    valid_train = df_train_feat.dropna(subset=features + ['forward_return'])

    # Process Test
    df_test_feat = compute_features(df_test, params)
    df_test_feat['forward_return'] = (df_test_feat["ground_truth_spx"] / df_test_feat["spx"]) - 1.0
    valid_test = df_test_feat.dropna(subset=features + ['forward_return'])

    if len(valid_train) < 100 or len(valid_test) < 100:
        return -1e9, 0.0, -1e9, -1e9, 0.0, -1e9

    X_train, y_train = valid_train[features], valid_train['forward_return']
    X_test, y_test = valid_test[features], valid_test['forward_return']

    model, scaler = get_model_and_scaler(model_type, params)
    try:
        X_train_s = scaler.fit_transform(X_train) if scaler else X_train.values
        X_test_s = scaler.transform(X_test) if scaler else X_test.values

        # FIT ONLY ON TRAIN DATA
        model.fit(X_train_s, y_train)

        # Evaluate on Train
        y_pred_train = model.predict(X_train_s)
        train_r2, train_wr, train_sharpe = _calculate_metrics(y_train, y_pred_train, params)

        # Evaluate on TEST (Out-of-sample)
        y_pred_test = model.predict(X_test_s)
        test_r2, test_wr, test_sharpe = _calculate_metrics(y_test, y_pred_test, params)

        return train_r2, train_wr, train_sharpe, test_r2, test_wr, test_sharpe
    except Exception:
        return -1e9, 0.0, -1e9, -1e9, 0.0, -1e9


def objective(trial: optuna.Trial, df: pd.DataFrame, model_type: str, **kwargs) -> float:
    """Optuna objective with conditional hyperparameters for the specified model."""
    mow, pow_val = -0.25, 0.25

    params = {
        "rank_window": trial.suggest_int("rank_window", 50, 1000),
        "zscore_window": trial.suggest_int("zscore_window", 20, 500),
        "value_of_ma50": trial.suggest_int("value_of_ma50", 10, 100),
        "value_of_ma200": trial.suggest_int("value_of_ma200", 100, 300),
        "trend_raw_weight_1": trial.suggest_float("trend_raw_weight_1", 0.0, 1.0, step=0.01),
        "trend_raw_weight_2": trial.suggest_float("trend_raw_weight_2", 0.0, 1.0, step=0.01),
        "value_of_roll_volatility": trial.suggest_int("value_of_roll_volatility", 20, 500),
        "value_of_roll_credit_score": trial.suggest_int("value_of_roll_credit_score", 20, 500),
        "value_of_roll_yield_curve": trial.suggest_int("value_of_roll_yield_curve", 20, 500),
        "value_of_roll_fc": trial.suggest_int("value_of_roll_fc", 20, 500),
        "value_of_roll_final_score": trial.suggest_int("value_of_roll_final_score", 20, 500),
        "trend_score_weight": trial.suggest_float("trend_score_weight", 0.25 + mow, 0.35 + pow_val),
        "volume_score_weight": trial.suggest_float("volume_score_weight", 0.20 + mow, 0.30 + pow_val),
        "credit_score_weight": trial.suggest_float("credit_score_weight", 0.15 + mow, 0.25 + pow_val),
        "curve_score_weight": trial.suggest_float("curve_score_weight", 0.10 + mow, 0.20 + pow_val),
        "fc_score_weight": trial.suggest_float("fc_score_weight", 0.05 + mow, 0.15 + pow_val),
        "lookahead_bars": trial.suggest_int("lookahead_bars", kwargs['lookahead_bars'], kwargs['lookahead_bars']),
        "epsilon": trial.suggest_float("epsilon", kwargs['epsilon'], kwargs['epsilon']),
        "density": trial.suggest_float("density", kwargs['density'], kwargs['density']),
    }

    params["model_type"] = model_type

    # Conditional Hyperparameters based on the user-specified model
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

    features = __FEATURES__
    df_feat = compute_features(df, params)
    df_feat['forward_return'] = (df_feat["ground_truth_spx"] / df_feat["spx"]) - 1.0
    valid_df = df_feat.dropna(subset=features + ['forward_return'])

    if len(valid_df) < 100:
        return -1e9

    tscv = TimeSeriesSplit(n_splits=5)
    sharpe_scores = []

    # FIXED: Split directly on valid_df to guarantee clean, contiguous time-series folds
    for train_idx, val_idx in tscv.split(valid_df):
        df_train = valid_df.iloc[train_idx]
        df_val = valid_df.iloc[val_idx]

        if len(df_train) < 50 or len(df_val) < 50:
            continue

        X_train, y_train = df_train[features], df_train['forward_return']
        X_val, y_val = df_val[features], df_val['forward_return']

        model, scaler = get_model_and_scaler(model_type, params)
        try:
            X_train_s = scaler.fit_transform(X_train) if scaler else X_train.values
            X_val_s = scaler.transform(X_val) if scaler else X_val.values

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)

            # Use the centralized metric calculator (removed the dead weighted_sharpe code)
            _, _, sharpe = _calculate_metrics(y_val, y_pred, params)
            sharpe_scores.append(sharpe)
        except Exception:
            continue

    return np.mean(sharpe_scores) if sharpe_scores else -1e9


def predict_latest_score(df: pd.DataFrame, params: dict, model_type: str, predict_tomorrow: bool = False) -> tuple:
    """Trains the final ML model on the ENTIRE dataset and outputs a scaled market score."""
    features = __FEATURES__

    df_feat = compute_features(df, params)
    df_feat['forward_return'] = (df_feat["ground_truth_spx"] / df_feat["spx"]) - 1.0

    valid_df = df_feat.dropna(subset=features + ['forward_return'])
    if len(valid_df) < 50:
        return "Unknown", 0.0, 0.0

    X_train = valid_df[features]
    y_train = valid_df['forward_return']

    model, scaler = get_model_and_scaler(model_type, params)
    X_train_scaled = scaler.fit_transform(X_train) if scaler else X_train.values
    model.fit(X_train_scaled, y_train)

    if predict_tomorrow:
        last_row = df_feat.iloc[[-1]]
        X_last = last_row[features].ffill().bfill()
        X_last_scaled = scaler.transform(X_last) if scaler else X_last.values
        pred_return = model.predict(X_last_scaled)[0]
        latest_heuristic_score = last_row['heuristic_market_score'].iloc[0]
        next_date = last_row.index[-1] + pd.Timedelta(days=1)
        latest_date = next_date.strftime('%Y-%m-%d')
    else:
        last_valid_idx = valid_df.index[-1]
        X_last = X_train.loc[[last_valid_idx]]
        X_last_scaled = scaler.transform(X_last) if scaler else X_last.values
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

    # --- REAL-TIME PREDICTION MODE ---
    if args.real_time:
        if not args.model_path:
            raise ValueError("--model-path is required when using --real-time")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}")

        print(f"Loading model from {args.model_path}...")
        model_data = joblib.load(args.model_path)

        model = model_data["model"]
        scaler = model_data["scaler"]
        params = model_data["params"]
        model_type = model_data["model_type"]
        features = model_data["features"]
        max_abs_ret = model_data["max_abs_ret"]

        # Display Train vs Test results
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE SUMMARY (Train vs Test)")
        print("=" * 70)
        print(f"{'Metric':<22} | {'Train':<18} | {'Test':<18}")
        print("-" * 70)

        train_r2 = model_data.get("train_r2")
        test_r2 = model_data.get("test_r2")
        train_wr = model_data.get("train_wr")
        test_wr = model_data.get("test_wr")
        train_sharpe = model_data.get("train_sharpe")
        test_sharpe = model_data.get("test_sharpe")

        def fmt(val, is_pct=False):
            if val is None:
                return "N/A"
            return f"{val:.2%}" if is_pct else f"{val:.4f}"

        print(f"{'R-squared':<22} | {fmt(train_r2):<18} | {fmt(test_r2):<18}")
        print(f"{'3-Class Win Rate':<22} | {fmt(train_wr, is_pct=True):<18} | {fmt(test_wr, is_pct=True):<18}")
        print(f"{'Simulated Sharpe':<22} | {fmt(train_sharpe):<18} | {fmt(test_sharpe):<18}")
        print("=" * 70 + "\n")

        print("Fetching latest data...")
        df = load_data(filename=args.data_filename)

        print("Computing features for the latest datapoint...")
        df_feat = compute_features(df, params)

        last_row = df_feat.iloc[[-1]]
        X_last = last_row[features].ffill().bfill()
        X_last_scaled = scaler.transform(X_last) if scaler else X_last.values

        pred_return = model.predict(X_last_scaled)[0]
        latest_heuristic_score = last_row['heuristic_market_score'].iloc[0]

        # Calculate tomorrow's date, skipping weekends
        next_date = last_row.index[-1] + pd.Timedelta(days=1)
        if next_date.weekday() == 5:  # Saturday
            next_date += pd.Timedelta(days=2)
        elif next_date.weekday() == 6:  # Sunday
            next_date += pd.Timedelta(days=1)

        tomorrow_date = next_date.strftime('%Y-%m-%d')
        tomorrow_ml = np.clip((pred_return / max_abs_ret) * 100, -100, 100) if max_abs_ret > 1e-6 else 0.0

        print("\n" + "=" * 60)
        print("REAL-TIME TOMORROW'S MARKET ANTICIPATION")
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
    train_val_split_ratio = args.train_val_split_ratio
    lookahead_bars = args.lookahead_bars
    epsilon = args.epsilon
    density = args.density
    n_trials = args.n_trials
    timeout = args.timeout
    model_type = args.model  # User-specified model

    df = load_data(filename=args.data_filename)
    idx = int(len(df) * train_val_split_ratio)
    df_train = df.iloc[:idx].copy()
    df_test = df.iloc[idx:].copy()

    print(f"Total Data    : {df.index[0].strftime('%Y-%m-%d')} :: {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} rows)")
    print(f"Train&Val Set : {df_train.index[0].strftime('%Y-%m-%d')} :: {df_train.index[-1].strftime('%Y-%m-%d')} ({len(df_train)} rows)")
    print(f"Test Set      : {df_test.index[0].strftime('%Y-%m-%d')} :: {df_test.index[-1].strftime('%Y-%m-%d')} ({len(df_test)} rows)")

    print(f"\n>>> Optimizing {model_type.upper()} Model via Walk-Forward Validation (Maximizing Sharpe Ratio) <<<\n")

    study = optuna.create_study(direction="maximize")
    print("Starting Optuna optimization on Training Set (5-fold TimeSeriesSplit)...")

    study.optimize(
        lambda trial: objective(trial, df_train, model_type=model_type, lookahead_bars=lookahead_bars, epsilon=epsilon, density=density),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_model_type = model_type  # Use the user-specified model

    print("\n--- Optimization Finished ---")
    print(f"Best CV Mean Sharpe Ratio  : {study.best_value:.4f}")
    print(f"Best Model Type            : {best_model_type.upper()}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    features = __FEATURES__

    print("\n--- Evaluating on Training and Hold-Out Test Set ---")
    train_r2, train_wr, train_sharpe, test_r2, test_wr, test_sharpe = evaluate_datasets(
        df_train, df_test, best_params, features, best_model_type
    )

    print(f"Train Set R-squared        : {train_r2:.4f}")
    print(f"Train Set 3-Class Win Rate : {train_wr:.2%}")
    print(f"Train Set Simulated Sharpe : {train_sharpe:.4f}")

    print(f"Test Set R-squared         : {test_r2:.4f}")
    print(f"Test Set 3-Class Win Rate  : {test_wr:.2%}")
    print(f"Test Set Simulated Sharpe  : {test_sharpe:.4f}")

    # --- SAVE THE OPTIMIZED MODEL ---
    print("\n" + "=" * 60)
    print("Saving the optimized model...")
    print("=" * 60)

    df_feat = compute_features(df, best_params)
    df_feat['forward_return'] = (df_feat["ground_truth_spx"] / df_feat["spx"]) - 1.0
    valid_df = df_feat.dropna(subset=__FEATURES__ + ['forward_return'])

    X_train = valid_df[__FEATURES__]
    y_train = valid_df['forward_return']

    final_model, final_scaler = get_model_and_scaler(best_model_type, best_params)
    X_train_scaled = final_scaler.fit_transform(X_train) if final_scaler else X_train.values
    final_model.fit(X_train_scaled, y_train)

    p95 = np.percentile(y_train, 95)
    p05 = np.percentile(y_train, 5)
    max_abs_ret = max(abs(p95), abs(p05))

    model_filename = f"market_model_{best_model_type}_la{lookahead_bars}_eps{epsilon}_d{density}__{datetime.now().strftime('%Y%m%d')}.joblib"
    model_data = {
        "model": final_model,
        "scaler": final_scaler,
        "params": best_params,
        "model_type": best_model_type,
        "features": __FEATURES__,
        "max_abs_ret": max_abs_ret,
        "train_r2": train_r2,
        "train_wr": train_wr,
        "train_sharpe": train_sharpe,
        "test_r2": test_r2,
        "test_wr": test_wr,
        "test_sharpe": test_sharpe
    }
    joblib.dump(model_data, model_filename)
    print(f"Model successfully saved to '{model_filename}'")

    # --- LATEST PREDICTIONS ---
    print("\n" + "=" * 60)
    print(f"LATEST MARKET PREDICTION (Retrained on Full Dataset)")
    print("=" * 60)

    latest_date, heuristic_score, ml_score = predict_latest_score(df, best_params, best_model_type, predict_tomorrow=False)

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
        print("\n" + "=" * 60)
        print("TOMORROW'S MARKET ANTICIPATION")
        print("=" * 60)

        tomorrow_date, tomorrow_heuristic, tomorrow_ml = predict_latest_score(df, best_params, best_model_type, predict_tomorrow=True)

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
        print("Tomorrow's market anticipation skipped (--skip-tomorrow).")
        print("=" * 60)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)