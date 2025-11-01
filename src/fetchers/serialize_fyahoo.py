try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # print(parent_dir)
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from colorama import init, Fore, Style
import pandas as pd
import pickle
import yfinance as yf
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from constants import FYAHOO__OUTPUTFILENAME, MY_TICKERS, TOP_SP500_TICKERS, FYAHOO_TICKER__OUTPUTFILENAME


def entry():
    end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    tickers = list(set(MY_TICKERS))
    data_cache ={}
    print(f"{len(tickers)} tickers found , {start_date=} {end_date=} , interval=1h , [{start_date}]>>[{end_date}]", flush=True)
    for ticker in tqdm(tickers):
        data = yf.download(ticker, start=start_date, end=end_date, interval='1h', auto_adjust=False, ignore_tz=True, progress=False)
        data_cache[ticker] = data
        # time.sleep(0.1)
    # Serialize
    with open(FYAHOO__OUTPUTFILENAME, 'wb') as f:
        pickle.dump(data_cache, f)
    print(f"Data saved to {FYAHOO__OUTPUTFILENAME}")
    # Sanity check
    # Deserialize
    with open(FYAHOO__OUTPUTFILENAME, 'rb') as f:
        loaded_data = pickle.load(f)


    print(f"{Fore.CYAN}{Style.BRIGHT}🚀 Starting Download Tracker...{Style.RESET_ALL}\n", flush=True)
    result = {}
    for ticker in tqdm(TOP_SP500_TICKERS):
        # Get stock data
        stock = yf.Ticker(ticker)
        result[ticker] = {
            'info': stock.info,
            'earnings_dates': stock.earnings_dates,
            'history': stock.history(period='2y')
        }
    # Serialize
    with open(FYAHOO_TICKER__OUTPUTFILENAME, 'wb') as f:
        pickle.dump(result, f)

    print(f"{Fore.GREEN}✅ Data saved to {FYAHOO_TICKER__OUTPUTFILENAME}{Style.RESET_ALL}")


if __name__ == "__main__":
    entry()