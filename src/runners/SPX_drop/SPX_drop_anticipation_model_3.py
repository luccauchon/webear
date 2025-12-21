#!/usr/bin/env python3
"""
Enhanced SPX drop anticipation scanner.
- More robust yfinance handling (flat or MultiIndex columns)
- Better Category A (market internals): percent above 50d & 200d MA from your local parquet
- Category B (technical): weekly RSI, 20-day BB upper, monthly vs 20-month EMA
- Category C (macro): real yields (DFII10) and NFCI from FRED
- Configurable thresholds and normalized final score (0-10)
"""
try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
import logging
import os
from datetime import datetime, timedelta
import time
from constants import FYAHOO_SPX500__OUTPUTFILENAME as LOCAL_SP500_PARQUET
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import pandas_datareader.data as web

# ---------------------------
# CONFIG / TUNABLES
# ---------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "213742dc08592772cb9502214cdc4397")
YF_RETRY = 2
YF_SLEEP_BETWEEN = 1.0
VERBOSE = True

config = {
    "ma_windows": {"short": 50, "long": 200},
    "category_a_threshold_pct": 50.0,  # percent above 50d below this triggers bearish point
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
logger = logging.getLogger("spx_scanner")
logger.setLevel(logging.INFO if not VERBOSE else logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)


# ---------------------------
# Utilities
# ---------------------------
def safe_yf_download(tickers, period="6mo", interval="1d", auto_adjust=True):
    """Robust yfinance downloader with minimal retry/backoff."""
    last_exc = None
    for attempt in range(1, YF_RETRY + 1):
        try:
            df = yf.download(tickers, period=period, interval=interval, auto_adjust=auto_adjust, threads=True)
            # normalize: yfinance sometimes returns a MultiIndex with ('Close','^GSPC') or a flat frame
            if df is None:
                raise ValueError("yfinance returned None")
            return df
        except Exception as e:
            last_exc = e
            logger.warning("yfinance attempt %d failed: %s", attempt, e)
            time.sleep(YF_SLEEP_BETWEEN * attempt)
    logger.error("yfinance failed after %d attempts: %s", YF_RETRY, last_exc)
    return pd.DataFrame()


def extract_close(series_or_df, ticker):
    """
    Given yf.download output (MultiIndex or single), return the Close series for ticker.
    If series_or_df is already a Series named by ticker, returns it.
    """
    if isinstance(series_or_df, pd.Series):
        # assume it's the close series already
        return series_or_df.dropna()
    df = series_or_df
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        # prefer ('Close', ticker)
        if ('Close', ticker) in df.columns:
            return df[('Close', ticker)].dropna()
        # sometimes tickers are top-level with Close column; try pattern
        for c0, c1 in df.columns:
            if c1 == ticker and c0.lower().startswith('close'):
                return df[(c0, c1)].dropna()
    else:
        # flat frame (e.g. when single ticker passed, column may be 'Close' only)
        if 'Close' in df.columns and len(df.columns) == 1:
            return df['Close'].dropna()
        if ticker in df.columns:
            return df[ticker].dropna()
    # fallback: try index name
    return pd.Series(dtype=float)


# ---------------------------
# Category A - Market internals
# ---------------------------
def compute_percent_above_mas_from_parquet(parquet_path, window=50, ma_type='SMA'):
    """
    Load a parquet with columns like ('Close','TICK') (MultiIndex) or plain DataFrame per ticker,
    compute percent of tickers whose latest close > MA(window).
    Returns: (pct_above, details_df)
    """
    if not os.path.exists(parquet_path):
        logger.warning("Local parquet not found at %s - skipping percent-above-MA check", parquet_path)
        return None, pd.DataFrame()

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error("Failed to read parquet %s: %s", parquet_path, e)
        return None, pd.DataFrame()

    # Expect MultiIndex columns formatted e.g. ('Close','AAPL')
    if not isinstance(df.columns, pd.MultiIndex):
        logger.warning("Parquet does not have expected MultiIndex columns - skipping")
        return None, pd.DataFrame()

    tickers = df.columns.get_level_values(1).unique()
    rows = []
    for t in tickers:
        if ('Close', t) not in df.columns:
            continue
        s = df[('Close', t)].dropna()
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


def category_A_market_internals():
    """Return points (0..3) from category A signals"""
    logger.info("Running Category A: Market Internals")
    points = 0
    # 1) percent above 50d
    pct50, details50 = compute_percent_above_mas_from_parquet(LOCAL_SP500_PARQUET, window=config['ma_windows']['short'])
    if pct50 is None:
        logger.info("Percent above 50d MA: data unavailable")
    else:
        logger.info("Percent of SPX constituents above %dd MA: %.2f%%", config['ma_windows']['short'], pct50)
        if pct50 < config['category_a_threshold_pct']:
            logger.info("→ Fewer than %.1f%% above %dd: bearish → +1", config['category_a_threshold_pct'], config['ma_windows']['short'])
            points += 1

    # 2) percent above 200d
    pct200, details200 = compute_percent_above_mas_from_parquet(LOCAL_SP500_PARQUET, window=config['ma_windows']['long'])
    if pct200 is None:
        logger.info("Percent above 200d MA: data unavailable")
    else:
        logger.info("Percent above %dd MA: %.2f%%", config['ma_windows']['long'], pct200)
        # add point if very low breadth below 60% (tunable)
        if pct200 < 60.0:
            logger.info("→ Breadth below 60%% on 200d MA → +1")
            points += 1

    # 3) RSP vs SPX divergence
    d = safe_yf_download(["^GSPC", "RSP"], period="6mo", interval="1d", auto_adjust=True)
    if d.empty:
        logger.info("SPX/RSP data unavailable")
    else:
        spx = extract_close(d, "^GSPC")
        rsp = extract_close(d, "RSP")
        combined = pd.concat([spx, rsp], axis=1).dropna()
        combined.columns = ['SPX', 'RSP']
        if len(combined) >= config['rsp_divergence_days'] + 1:
            spx_ret = combined['SPX'].iloc[-1] / combined['SPX'].iloc[-(config['rsp_divergence_days'] + 1)] - 1
            rsp_ret = combined['RSP'].iloc[-1] / combined['RSP'].iloc[-(config['rsp_divergence_days'] + 1)] - 1
            underperf = spx_ret - rsp_ret
            logger.info("SPX %d-day ret: %.2f%%, RSP %d-day ret: %.2f%%, underperf: %.2f%%",
                        config['rsp_divergence_days'], spx_ret * 100,
                        config['rsp_divergence_days'], rsp_ret * 100,
                        underperf * 100)
            if underperf > config['rsp_divergence_threshold']:
                logger.info("→ RSP underperformed SPX by >%.2f%% → +1", config['rsp_divergence_threshold'] * 100)
                points += 1
        else:
            logger.info("Not enough SPX/RSP history for divergence check")

    logger.info("Category A points: %d", points)
    return points


# ---------------------------
# Category B - Technical conditions (SPX)
# ---------------------------
def Category_B_Technical_Conditions():
    logger.info("Running Category B: Technical Conditions (S&P 500)")
    points = 0
    # Fetch SPX
    d = safe_yf_download("^GSPC", period="2y", interval="1d", auto_adjust=True)
    if d.empty:
        logger.warning("SPX data missing - skipping Category B")
        return 0

    spx = extract_close(d, "^GSPC")
    if spx.empty:
        logger.warning("SPX close series empty - skipping Category B")
        return 0

    # Weekly RSI > threshold
    try:
        weekly = spx.resample('W-FRI').last().dropna()
        weekly_rsi = ta.rsi(weekly, length=14)
        weekly_rsi_val = float(weekly_rsi.iloc[-1]) if not weekly_rsi.empty else np.nan
        weekly_cond = weekly_rsi_val > config['weekly_rsi_threshold']
        logger.info("Weekly RSI: %.2f, > %d? %s", weekly_rsi_val, config['weekly_rsi_threshold'], weekly_cond)
        points += int(bool(weekly_cond))
    except Exception as e:
        logger.debug("Weekly RSI calculation failed: %s", e)

    # Price > 2σ Bollinger (20-day)
    try:
        bb = ta.bbands(spx, length=config['bb_length'], std=config['bb_std'])
        # bb returns dataframe with columns like 'BBU_20_2.0_2.0'
        if isinstance(bb, pd.DataFrame):
            bbu_col = [c for c in bb.columns if c.startswith("BBU_")]
            if bbu_col:
                bbu = bb[bbu_col[0]]
                bb_cond = spx.iloc[-1] > bbu.iloc[-1]
                logger.info("SPX last: %.2f, BB upper: %.2f, > upper? %s", spx.iloc[-1], bbu.iloc[-1], bb_cond)
                points += int(bb_cond)
    except Exception as e:
        logger.debug("Bollinger calc failed: %s", e)

    # Monthly close > 2% above 20-month EMA
    try:
        monthly = spx.resample('ME').last().dropna()  # month-end closes
        ema20m = ta.ema(monthly, length=config['monthly_ema_months'])
        if not monthly.empty and not ema20m.empty:
            parabolic_cond = monthly.iloc[-1] > 1.02 * ema20m.iloc[-1]
            logger.info("Monthly close: %.2f, 20-month EMA: %.2f, >2%%? %s", monthly.iloc[-1], ema20m.iloc[-1], parabolic_cond)
            points += int(parabolic_cond)
    except Exception as e:
        logger.debug("Monthly EMA calc failed: %s", e)

    logger.info("Category B points: %d", points)
    return points


# ---------------------------
# Category C - Macro Liquidity
# ---------------------------
def category_C_macro_liquidity():
    logger.info("Running Category C: Macro Liquidity (FRED)")
    pts = 0
    end = datetime.today()
    start = end - timedelta(weeks=6)

    # 1) Real yield DFII10 rising >= configured bps over two weeks
    try:
        df = web.DataReader('DFII10', 'fred', start, end, api_key=FRED_API_KEY)
        df = df.dropna()
        if len(df) >= 5:
            recent = float(df['DFII10'].iloc[-1])
            two_weeks_ago = (end - timedelta(weeks=2))
            past_val = float(df['DFII10'].asof(two_weeks_ago))
            change_bps = (recent - past_val) * 100.0
            logger.info("DFII10 recent: %.3f, two-weeks-ago(asof): %.3f, change: %.1f bps", recent, past_val, change_bps)
            if change_bps >= config['real_yield_bps_move']:
                logger.info("→ Real yields rose >= %d bps → +1", config['real_yield_bps_move'])
                pts += 1
        else:
            logger.info("Insufficient DFII10 data")
    except Exception as e:
        logger.warning("Error fetching DFII10: %s", e)

    # 2) NFCI (financial conditions) trending up vs two weeks ago
    try:
        df2 = web.DataReader('NFCI', 'fred', start, end, api_key=FRED_API_KEY)
        df2 = df2.dropna()
        if len(df2) >= 5:
            recent = float(df2['NFCI'].iloc[-1])
            past_val = float(df2['NFCI'].asof(end - timedelta(weeks=2)))
            logger.info("NFCI recent: %.4f, past(asof): %.4f", recent, past_val)
            if recent > past_val:
                logger.info("→ Financial conditions tightened (NFCI up) → +1")
                pts += 1
        else:
            logger.info("Insufficient NFCI data")
    except Exception as e:
        logger.warning("Error fetching NFCI: %s", e)

    logger.info("Category C points: %d", pts)
    return pts


# ---------------------------
# Final scoring / orchestration
# ---------------------------
def run_once():
    """
    Run the full scanner once and compute a normalized score 0..score_max (config['score_max'])
    Returns dictionary of raw_category_points and normalized score.
    """
    a = category_A_market_internals()         # 0..3 (approx)
    b = Category_B_Technical_Conditions()     # 0..3
    c = category_C_macro_liquidity()          # 0..2

    raw_total = a + b + c
    # For normalization, assume maximum practical raw_total is 8 (3+3+2)
    max_raw = 8
    norm_score = (raw_total / max_raw) * config['score_max']
    norm_score = float(np.clip(norm_score, 0, config['score_max']))

    logger.info("=" * 60)
    logger.info("RAW SCORE: %d (A=%d, B=%d, C=%d)", raw_total, a, b, c)
    logger.info("NORMALIZED SCORE (0-%d): %.2f", config['score_max'], norm_score)

    # simple action mapping
    action = ""
    if norm_score <= 3:
        action = "Low drop probability — normal credit put spread sizing"
    elif norm_score <= 6:
        action = "Medium risk — consider reducing size 30–50%"
    else:
        action = "Elevated risk — avoid selling credit put spreads this month"

    logger.info("RECOMMENDATION: %s", action)

    return {
        "raw_total": raw_total,
        "A": a,
        "B": b,
        "C": c,
        "normalized_score": norm_score,
        "recommendation": action
    }


def main():
    parser = argparse.ArgumentParser(description="SPX drop anticipation scanner")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    res = run_once()
    # If you want to save results:
    # pd.Series(res).to_csv("spx_scanner_last_run.csv")
    return res


if __name__ == "__main__":
    main()
