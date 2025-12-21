#!/usr/bin/env python3
"""
SPX drop anticipation scanner + auto-backtester
- Save as spx_scanner_with_backtest.py
- Backtester computes signals historically and reports empirical probabilities.
"""
try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
import os
import time
from datetime import datetime, timedelta
import copy
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import pandas_datareader.data as web
import logging
from constants import FYAHOO_SPX500__OUTPUTFILENAME as LOCAL_SP500_PARQUET, FYAHOO__OUTPUTFILENAME_DAY
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool
import pickle


# ---------------------------
# CONFIG
# ---------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "213742dc08592772cb9502214cdc4397")
VERBOSE = True

config = {
    "ma_windows": {"short": 50, "long": 200},
    "category_a_threshold_pct": 50.0,
    "rsp_divergence_days": 20,
    "rsp_divergence_threshold": 0.03,
    "weekly_rsi_threshold": 68,
    "bb_length": 20,
    "bb_std": 2,
    "monthly_ema_months": 20,
    "real_yield_bps_move": 20,
    "score_max": 10
}

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("spx_scanner_bt")
logger.setLevel(logging.INFO if not VERBOSE else logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)


# ---------------------------
# Utilities
# ---------------------------
def extract_close(df, ticker):
    if isinstance(df, pd.Series):
        return df.dropna()
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if ('Close', ticker) in df.columns:
            return df[('Close', ticker)].dropna()
        # fallback: find second level ticker
        for c0, c1 in df.columns:
            if c1 == ticker and 'Close' in str(c0):
                return df[(c0, c1)].dropna()
    else:
        # if single-ticker single-col with 'Close'
        if 'Close' in df.columns and len(df.columns) == 1:
            return df['Close'].dropna()
        if ticker in df.columns:
            return df[ticker].dropna()
    return pd.Series(dtype=float)


def compute_percent_above_mas_from_parquet(df_parquet_path, window=50, ma_type='SMA', asof=None):
    """
    Read parquet once; compute percent of tickers whose latest close (asof) > MA(window) using only data <= asof.
    """
    df = df_parquet_path

    if not isinstance(df.columns, pd.MultiIndex):
        logger.warning("Parquet columns not MultiIndex - skipping")
        return None, pd.DataFrame()

    tickers = df.columns.get_level_values(1).unique()
    rows = []
    for t in tickers:
        col = ('Close', t)
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if asof is not None:
            dates = pd.to_datetime(s.index.get_level_values(1))
            mask = dates <= asof
            s = s[mask]
        if len(s) < window:
            continue
        if ma_type.upper() == 'SMA':
            ma = s.rolling(window=window).mean()
        else:
            ma = s.ewm(span=window, adjust=False).mean()
        last_close = s.iloc[-1]
        last_ma = ma.iloc[-1]
        rows.append({'Ticker': t, 'Last_Close': last_close, 'Last_MA': last_ma, 'Above_MA': last_close > last_ma})
    details = pd.DataFrame(rows)
    if details.empty:
        return None, details
    pct = details['Above_MA'].mean() * 100.0
    return pct, details


# ---------------------------
# Single-day signalcalculator (used by backtester)
# ---------------------------
def compute_signals_for_date(my_df, asof, spx_series, rsp_series, dfii10_series, nfci_series):
    """
    Compute raw A,B,C points for a single asof date using series that contain full history.
    Returns (A_points, B_points, C_points)
    """
    # ---- Category A ----
    A_pts = 0
    # 50d breadth
    pct50, _ = compute_percent_above_mas_from_parquet(copy.deepcopy(my_df), window=config['ma_windows']['short'], asof=asof)
    if pct50 is not None and pct50 < config['category_a_threshold_pct']:
        A_pts += 1

    # 200d breadth
    pct200, _ = compute_percent_above_mas_from_parquet(copy.deepcopy(my_df), window=config['ma_windows']['long'], asof=asof)
    if pct200 is not None and pct200 < 60.0:
        A_pts += 1

    # RSP divergence (use data up to asof)
    spx_slice = spx_series.loc[:pd.to_datetime(asof)].dropna()
    rsp_slice = rsp_series.loc[:pd.to_datetime(asof)].dropna()
    combined = pd.concat([spx_slice, rsp_slice], axis=1, join='inner')
    combined.columns = ['SPX', 'RSP']
    if len(combined) >= config['rsp_divergence_days'] + 1:
        spx_ret = combined['SPX'].iloc[-1] / combined['SPX'].iloc[-(config['rsp_divergence_days'] + 1)] - 1
        rsp_ret = combined['RSP'].iloc[-1] / combined['RSP'].iloc[-(config['rsp_divergence_days'] + 1)] - 1
        underperf = spx_ret - rsp_ret
        if underperf > config['rsp_divergence_threshold']:
            A_pts += 1

    # ---- Category B ----
    B_pts = 0
    spx_slice = spx_series.loc[:pd.to_datetime(asof)].dropna()
    if len(spx_slice) > 50:
        # weekly RSI
        weekly = spx_slice.resample('W-FRI').last().dropna()
        if len(weekly) > 14:
            weekly_rsi = ta.rsi(weekly, length=14)
            if not weekly_rsi.empty and weekly_rsi.iloc[-1] > config['weekly_rsi_threshold']:
                B_pts += 1

        # bollinger: price > upper
        bb = ta.bbands(spx_slice, length=config['bb_length'], std=config['bb_std'])
        if isinstance(bb, pd.DataFrame):
            bbu_cols = [c for c in bb.columns if c.startswith("BBU_")]
            if bbu_cols:
                bbu = bb[bbu_cols[0]]
                if spx_slice.iloc[-1] > bbu.iloc[-1]:
                    B_pts += 1

        # monthly vs 20-month ema
        monthly = spx_slice.resample('ME').last().dropna()
        if len(monthly) >= config['monthly_ema_months']:
            ema20m = ta.ema(monthly, length=config['monthly_ema_months'])
            if not ema20m.empty and monthly.iloc[-1] > 1.02 * ema20m.iloc[-1]:
                B_pts += 1

    # ---- Category C ----
    C_pts = 0
    # DFII10
    try:
        dfii = dfii10_series.loc[:pd.to_datetime(asof)].dropna()
        if len(dfii) >= 5:
            recent = float(dfii.iloc[-1])
            two_weeks_ago = pd.to_datetime(asof) - pd.Timedelta(days=14)
            past_val = float(dfii.asof(two_weeks_ago))
            change_bps = (recent - past_val) * 100.0
            if change_bps >= config['real_yield_bps_move']:
                C_pts += 1
    except Exception:
        pass

    # NFCI
    try:
        nfci = nfci_series.loc[:pd.to_datetime(asof)].dropna()
        if len(nfci) >= 5:
            recent = float(nfci.iloc[-1])
            past_val = float(nfci.asof(pd.to_datetime(asof) - pd.Timedelta(days=14)))
            if recent > past_val:
                C_pts += 1
    except Exception:
        pass

    return int(A_pts), int(B_pts), int(C_pts)


# ---------------------------
# Backtester
# ---------------------------
def backtest(start_date, end_date, parquet_path=LOCAL_SP500_PARQUET, resample_freq='B', save_csv=True, out_csv="backtest_results.csv"):
    """
    Run the backtest between start_date and end_date (inclusive).
    - Downloads SPX, RSP.
    - Downloads FRED series DFII10 and NFCI once.
    - For each business day in the SPX index between start and end, compute signals as-of that day.
    - Compute realized future min returns over next 30 and 45 trading days.
    - Save results CSV and print calibration table by normalized score buckets.
    """
    logger.info("Starting backtest from %s to %s", start_date, end_date)

    # 1) Download price series once
    tickers = ["^GSPC", "RSP"]
    one_dataset_filename = get_filename_for_dataset("day", older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)

    spx = extract_close(copy.deepcopy(master_data_cache["^GSPC"]), "^GSPC").rename("SPX")
    rsp = extract_close(copy.deepcopy(master_data_cache["RSP"]), "RSP").rename("RSP")

    # unify index to business days for forward-looking window selection
    spx = spx.sort_index()
    trading_days = spx.loc[start_date:end_date].index

    # 2) Download FRED once
    try:
        dfii = web.DataReader('DFII10', 'fred', start_date, end_date, api_key=FRED_API_KEY).squeeze()
        nfci = web.DataReader('NFCI', 'fred', start_date, end_date, api_key=FRED_API_KEY).squeeze()
    except Exception as e:
        logger.warning("FRED download failed: %s", e)
        dfii = pd.Series(dtype=float)
        nfci = pd.Series(dtype=float)

    results = []
    try:
        my_df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error("Failed to read parquet: %s", e)
        return None, pd.DataFrame()

    from tqdm import tqdm
    logger.info("Looping through %d trading days...", len(trading_days))
    for current_date in tqdm(trading_days):
        asof = current_date
        # print(f"Processing {asof}")
        # compute signals using slices
        try:
            A_pts, B_pts, C_pts = compute_signals_for_date(my_df, asof, copy.deepcopy(spx), copy.deepcopy(rsp), copy.deepcopy(dfii), copy.deepcopy(nfci))
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.debug("Signal compute error on %s: %s", asof, e)
            continue

        raw_total = A_pts + B_pts + C_pts
        max_raw = 8.0
        norm_score = float(np.clip((raw_total / max_raw) * config['score_max'], 0, config['score_max']))

        # compute future returns windows (work on spx series)
        try:
            idx_pos = spx.index.get_loc(asof)
        except KeyError:
            # if asof not in spx index exactly, find nearest prior
            idx_pos = spx.index.get_indexer([asof], method='ffill')[0]
        # windows: next 30 and next 45 trading days
        future_30_idx_end = idx_pos + 30
        future_45_idx_end = idx_pos + 45
        # ensure within bounds
        future_prices = spx.values
        today_price = float(spx.iloc[idx_pos])

        min_ret_30 = np.nan
        min_ret_45 = np.nan
        if future_30_idx_end < len(future_prices):
            fut30 = future_prices[idx_pos + 1: future_30_idx_end + 1]
            if len(fut30) > 0:
                min_ret_30 = (fut30.min() / today_price) - 1.0
        else:
            # partial window available
            fut30 = future_prices[idx_pos + 1: len(future_prices)]
            if len(fut30) > 0:
                min_ret_30 = (fut30.min() / today_price) - 1.0

        if future_45_idx_end < len(future_prices):
            fut45 = future_prices[idx_pos + 1: future_45_idx_end + 1]
            if len(fut45) > 0:
                min_ret_45 = (fut45.min() / today_price) - 1.0
        else:
            fut45 = future_prices[idx_pos + 1: len(future_prices)]
            if len(fut45) > 0:
                min_ret_45 = (fut45.min() / today_price) - 1.0

        drop_30 = bool(min_ret_30 is not None and not np.isnan(min_ret_30) and min_ret_30 <= -0.05)
        # drop between 5% and 8% in 30-45: check min in 30..45 window specifically
        # We'll compute min over days 31..45 relative to today (if available)
        min_ret_30to45 = np.nan
        start_30to45 = idx_pos + 30
        end_30to45 = idx_pos + 45
        if start_30to45 < len(future_prices):
            fut30to45 = future_prices[start_30to45: min(end_30to45 + 1, len(future_prices))]
            if len(fut30to45) > 0:
                min_ret_30to45 = (fut30to45.min() / today_price) - 1.0
        drop_30to45_5_8 = bool(min_ret_30to45 is not None and not np.isnan(min_ret_30to45) and (min_ret_30to45 <= -0.05) and (min_ret_30to45 >= -0.08))

        results.append({
            "asof": asof,
            "A": A_pts, "B": B_pts, "C": C_pts,
            "raw_total": raw_total,
            "norm_score": norm_score,
            "min_ret_30": min_ret_30,
            "min_ret_45": min_ret_45,
            "drop_30": int(drop_30),
            "min_ret_30to45": min_ret_30to45,
            "drop_30to45_5_8": int(drop_30to45_5_8)
        })

    res_df = pd.DataFrame(results).set_index("asof").sort_index()
    if save_csv:
        res_df.to_csv(out_csv)
        logger.info("Saved backtest results to %s (%d rows)", out_csv, len(res_df))

    # calibration: bucket norm_score into bins and compute empirical probabilities
    bins = [0, 2, 4, 6, 8, 10]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]
    res_df['score_bucket'] = pd.cut(res_df['norm_score'], bins=bins, labels=labels, include_lowest=True, right=True)
    summary = res_df.groupby('score_bucket',observed=False).agg(
        count=('raw_total', 'count'),
        prob_drop_30=('drop_30', 'mean'),
        prob_drop_30to45_5_8=('drop_30to45_5_8', 'mean'),
        avg_norm_score=('norm_score', 'mean')
    ).sort_index()

    # convert probs to %
    summary['prob_drop_30'] = (summary['prob_drop_30'] * 100).round(2)
    summary['prob_drop_30to45_5_8'] = (summary['prob_drop_30to45_5_8'] * 100).round(2)

    logger.info("Calibration summary (by normalized score bucket):")
    logger.info("\n%s", summary.to_string())

    return {"results_df": res_df, "summary": summary}


# ---------------------------
# Live Scan Function
# ---------------------------
def run_live_scan(parquet_path=LOCAL_SP500_PARQUET):
    """
    Perform a live scan using the most recent available data (as of today).
    """
    logger.info("Running live SPX drop anticipation scan...")

    # 1. Determine asof date: last business day with market data (use today or previous close)
    today = pd.Timestamp.today().normalize()
    # Use yesterday if market hasn't closed yet (rough heuristic)
    asof = today - pd.Timedelta(days=1) if datetime.now().hour < 16 else today
    asof = asof.tz_localize(None)  # Ensure naive datetime

    # 2. Fetch fresh SPX and RSP (don't rely on potentially stale pickle)
    try:
        tickers_data = yf.download(["^GSPC", "RSP"], period="2y", interval="1d", progress=False, auto_adjust=True)
        spx = extract_close(tickers_data, "^GSPC").rename("SPX")
        rsp = extract_close(tickers_data, "RSP").rename("RSP")
    except Exception as e:
        logger.error("Failed to download live SPX/RSP: %s", e)
        return None

    # 3. Fetch FRED series (DFII10, NFCI) up to today
    try:
        dfii = web.DataReader('DFII10', 'fred', "2020-01-01", today, api_key=FRED_API_KEY).squeeze()
    except Exception as e:
        logger.warning("Failed to fetch DFII10: %s", e)
        dfii = pd.Series(dtype=float)
    try:
        nfci = web.DataReader('NFCI', 'fred', "2020-01-01", today, api_key=FRED_API_KEY).squeeze()
    except Exception as e:
        logger.warning("Failed to fetch NFCI: %s", e)
        nfci = pd.Series(dtype=float)

    # 4. Load breadth parquet
    try:
        my_df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error("Failed to read parquet for live scan: %s", e)
        return None

    # 5. Compute signals
    try:
        A_pts, B_pts, C_pts = compute_signals_for_date(
            my_df, asof, spx, rsp, dfii, nfci
        )
    except Exception as e:
        logger.error("Error during live signal computation: %s", e)
        import traceback
        traceback.print_exc()
        return None

    raw_total = A_pts + B_pts + C_pts
    max_raw = 8.0  # Max possible: A=3, B=3, C=2
    norm_score = float(np.clip((raw_total / max_raw) * config['score_max'], 0, config['score_max']))

    # 6. Output
    logger.info("=" * 50)
    logger.info("LIVE SCAN RESULT (as of %s)", asof.strftime("%Y-%m-%d"))
    logger.info("Category A Points: %d/3", A_pts)
    logger.info("Category B Points: %d/3", B_pts)
    logger.info("Category C Points: %d/2", C_pts)
    logger.info("Raw Score: %d / 8", raw_total)
    logger.info("Normalized Score: %.2f / 10", norm_score)
    logger.info("=" * 50)

    if norm_score >= 7.0:
        logger.warning("⚠️  HIGH DROP RISK SIGNAL DETECTED!")
    elif norm_score >= 5.0:
        logger.info("🟡 Moderate risk signal.")
    else:
        logger.info("🟢 Low risk environment.")

    return {
        "asof": asof,
        "A": A_pts,
        "B": B_pts,
        "C": C_pts,
        "raw_total": raw_total,
        "norm_score": norm_score
    }


# ---------------------------
# CLI Update
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="SPX scanner + backtest")
    parser.add_argument("--backtest", action="store_true", help="Run the historical backtester")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--out", type=str, default="backtest_results.csv", help="CSV output for backtest")
    args = parser.parse_args()

    tmpstr = (r"""
    "(PY312_HT) D:\PyCharmProjects\webear\src\sandbox>python spx_scanner_with_backtest.py --backtest --start=2020-01-01
2025-12-20 11:34:41 INFO: Starting backtest from 2020-01-01 to 2025-12-20
2025-12-20 11:34:45 INFO: Looping through 1501 trading days...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1501/1501 [4:24:16<00:00, 10.56s/it]
2025-12-20 15:59:01 INFO: Saved backtest results to backtest_results.csv (1501 rows)
2025-12-20 15:59:01 INFO: Calibration summary (by normalized score bucket):
2025-12-20 15:59:01 INFO:
              count  prob_drop_30  prob_drop_30to45_5_8  avg_norm_score
score_bucket
0-2             435         17.01                  3.68        1.146552
2-4             793         20.30                  7.82        3.051702
4-6             225         43.56                 13.78        5.000000
6-8              48         35.42                  6.25        6.432292
8-10              0           NaN                   NaN             NaN
2025-12-20 15:59:02 INFO: Backtest completed. Summary:
              count  prob_drop_30  prob_drop_30to45_5_8  avg_norm_score
score_bucket
0-2             435         17.01                  3.68        1.146552
2-4             793         20.30                  7.82        3.051702
4-6             225         43.56                 13.78        5.000000
6-8              48         35.42                  6.25        6.432292
8-10              0           NaN                   NaN             NaN
    """)
    print(tmpstr)
    if args.backtest:
        out = backtest(args.start, args.end, parquet_path=LOCAL_SP500_PARQUET, out_csv=args.out)
        if out is None:
            logger.error("Backtest failed or returned no output.")
        else:
            logger.info("Backtest completed. Summary:\n%s", out['summary'])
    else:
        # Run live scan
        run_live_scan()


if __name__ == "__main__":
    main()
