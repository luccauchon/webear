import argparse
import sys
import os
import pathlib
from colorama import init, Fore, Style
import pandas as pd
import copy
import pickle
import yfinance as yf
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# Try to import version; if not found, adjust sys.path
try:
    from version import sys__name, sys__version
except ImportError:
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

from constants import (
    FYAHOO__OUTPUTFILENAME_YEAR,
    FYAHOO__OUTPUTFILENAME,
    MY_TICKERS,
    TOP10_SP500_TICKERS,
    MY_TICKERS_SMALL_SET,
    FYAHOO_TICKER__OUTPUTFILENAME,
    FYAHOO__OUTPUTFILENAME_DAY,
    FYAHOO__OUTPUTFILENAME_MONTH,
    FYAHOO__OUTPUTFILENAME_WEEK,
    FYAHOO__OUTPUTFILENAME_QUARTER
)

def entry(
    use_all_tickers=True,
    start_date=None,
    end_date=None,
    skip_hourly=False,
    skip_daily=False,
    skip_weekly=False,
    skip_monthly=False,
    skip_quarterly=False,
    skip_yearly=False,
    skip_economic=False
):
    init(autoreset=True)

    # Default dates if not provided
    today = datetime.today()
    if end_date is None:
        end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    if start_date is None:
        # Default to 1 year ago for hourly data, but overridden per section if needed
        start_date = (today - timedelta(days=512)).strftime('%Y-%m-%d')

    tickers = sorted(list(set(MY_TICKERS if use_all_tickers else MY_TICKERS_SMALL_SET)), reverse=True)

    print(f"{len(tickers)} tickers selected. Date range: {start_date} to {end_date}", flush=True)

    ###########################################################################
    # 1 hour
    ###########################################################################
    if not skip_hourly:
        hourly_start = start_date
        hourly_end = end_date
        print(f"{len(tickers)} tickers, {hourly_start=}, {hourly_end=}, interval=1h", flush=True)
        data_cache = {}
        for ticker in tqdm(tickers, desc="Hourly"):
            if ticker == '^SKEW':
                data = yf.download(ticker, start=hourly_start, end=hourly_end, interval='1d',
                                   auto_adjust=False, ignore_tz=True, progress=False)
            else:
                data = yf.download(ticker, start=hourly_start, end=hourly_end, interval='1h',
                                   auto_adjust=False, ignore_tz=True, progress=False)
            data_cache[ticker] = data
        with open(FYAHOO__OUTPUTFILENAME, 'wb') as f:
            pickle.dump(data_cache, f)
        print(f"Hourly data saved to {FYAHOO__OUTPUTFILENAME}")

    ###########################################################################
    # 1 day (long history)
    ###########################################################################
    if not skip_daily:
        daily_start = "1960-01-01"
        daily_end = end_date
        print(f"{len(tickers)} tickers, {daily_start=}, {daily_end=}, interval=1d", flush=True)
        data_cache = {}
        for ticker in tqdm(tickers, desc="Daily"):
            data = yf.download(ticker, start=daily_start, end=daily_end, interval='1d',
                               auto_adjust=False, ignore_tz=True, progress=False)
            data_cache[ticker] = data
        with open(FYAHOO__OUTPUTFILENAME_DAY, 'wb') as f:
            pickle.dump(data_cache, f)
        print(f"Daily data saved to {FYAHOO__OUTPUTFILENAME_DAY}")

    # Reload daily data for resampling (only if needed by any of the resampled intervals)
    if not (skip_weekly and skip_monthly and skip_quarterly and skip_yearly):
        with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
            daily_data_cache = pickle.load(f)

    ###########################################################################
    # Weekly
    ###########################################################################
    if not skip_weekly:
        data_cache = {}
        for ticker in tqdm(tickers, desc="Weekly"):
            df = daily_data_cache[ticker]
            resampled = df.resample('W-FRI').agg({
                ('Open', ticker): 'first',
                ('High', ticker): 'max',
                ('Low', ticker): 'min',
                ('Close', ticker): 'last',
                ('Volume', ticker): 'sum'
            })
            data_cache[ticker] = resampled
        with open(FYAHOO__OUTPUTFILENAME_WEEK, 'wb') as f:
            pickle.dump(data_cache, f)
        print(f"Weekly data saved to {FYAHOO__OUTPUTFILENAME_WEEK}")

    ###########################################################################
    # Monthly
    ###########################################################################
    if not skip_monthly:
        data_cache = {}
        for ticker in tqdm(tickers, desc="Monthly"):
            df = daily_data_cache[ticker]
            resampled = df.resample('ME').agg({
                ('Open', ticker): 'first',
                ('High', ticker): 'max',
                ('Low', ticker): 'min',
                ('Close', ticker): 'last',
                ('Volume', ticker): 'sum'
            })
            data_cache[ticker] = resampled
        with open(FYAHOO__OUTPUTFILENAME_MONTH, 'wb') as f:
            pickle.dump(data_cache, f)
        print(f"Monthly data saved to {FYAHOO__OUTPUTFILENAME_MONTH}")

    ###########################################################################
    # Quarterly
    ###########################################################################
    if not skip_quarterly:
        data_cache = {}
        for ticker in tqdm(tickers, desc="Quarterly"):
            df = daily_data_cache[ticker]
            resampled = df.resample('QE').agg({
                ('Open', ticker): 'first',
                ('High', ticker): 'max',
                ('Low', ticker): 'min',
                ('Close', ticker): 'last',
                ('Volume', ticker): 'sum'
            })
            data_cache[ticker] = resampled
        with open(FYAHOO__OUTPUTFILENAME_QUARTER, 'wb') as f:
            pickle.dump(data_cache, f)
        print(f"Quarterly data saved to {FYAHOO__OUTPUTFILENAME_QUARTER}")

    ###########################################################################
    # Yearly
    ###########################################################################
    if not skip_yearly:
        data_cache = {}
        for ticker in tqdm(tickers, desc="Yearly"):
            df = daily_data_cache[ticker]
            resampled = df.resample('YE').agg({
                ('Open', ticker): 'first',
                ('High', ticker): 'max',
                ('Low', ticker): 'min',
                ('Close', ticker): 'last',
                ('Volume', ticker): 'sum'
            })
            data_cache[ticker] = resampled
        with open(FYAHOO__OUTPUTFILENAME_YEAR, 'wb') as f:
            pickle.dump(data_cache, f)
        print(f"Yearly data saved to {FYAHOO__OUTPUTFILENAME_YEAR}")

    ###########################################################################
    # Economic (Top 10 S&P500 metadata)
    ###########################################################################
    if not skip_economic:
        print(f"{Fore.CYAN}{Style.BRIGHT}🚀 Starting Download Tracker...{Style.RESET_ALL}\n", flush=True)
        result = {}
        for ticker in tqdm(TOP10_SP500_TICKERS, desc="Economic"):
            stock = yf.Ticker(ticker)
            result[ticker] = {
                'info': stock.info,
                'earnings_dates': stock.earnings_dates,
                'history': stock.history(period='2y')
            }
        with open(FYAHOO_TICKER__OUTPUTFILENAME, 'wb') as f:
            pickle.dump(result, f)
        print(f"{Fore.GREEN}✅ Data saved to {FYAHOO_TICKER__OUTPUTFILENAME}{Style.RESET_ALL}")

def parse_args():
    parser = argparse.ArgumentParser(description="Download and resample financial data from Yahoo Finance.")
    parser.add_argument(
        "-a", "--all-tickers",
        action="store_true",
        help="Use full ticker list (MY_TICKERS). If not set, uses MY_TICKERS_SMALL_SET."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for hourly data in YYYY-MM-DD format (default: 1 year ago)."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: tomorrow)."
    )
    parser.add_argument(
        "--skip-hourly", action="store_true", help="Skip downloading hourly data."
    )
    parser.add_argument(
        "--skip-daily", action="store_true", help="Skip downloading full daily data."
    )
    parser.add_argument(
        "--skip-weekly", action="store_true", help="Skip weekly resampling."
    )
    parser.add_argument(
        "--skip-monthly", action="store_true", help="Skip monthly resampling."
    )
    parser.add_argument(
        "--skip-quarterly", action="store_true", help="Skip quarterly resampling."
    )
    parser.add_argument(
        "--skip-yearly", action="store_true", help="Skip yearly resampling."
    )
    parser.add_argument(
        "--skip-economic", action="store_true", help="Skip downloading Top10 S&P500 economic data."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    entry(
        use_all_tickers=args.all_tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_hourly=args.skip_hourly,
        skip_daily=args.skip_daily,
        skip_weekly=args.skip_weekly,
        skip_monthly=args.skip_monthly,
        skip_quarterly=args.skip_quarterly,
        skip_yearly=args.skip_yearly,
        skip_economic=args.skip_economic
    )