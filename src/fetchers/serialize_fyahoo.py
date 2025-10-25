import pandas as pd
import pickle
import yfinance as yf
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from constants import FYAHOO__OUTPUTFILENAME


def get_all_tickers():
    sp500 = []
    custom = ["^GSPC", "^VIX", "SPY", "QQQ", "ADBE", "AFRM", "AMD", "AMZN", "AAPL", "BAC", "COST", "CRM", "DBRG", "GOOG", "INTC",
              "MA", "META", "MSFT", "NFLX", "NVDA", "ORCL", "PINS", "PLTR", "RDDT",
              "RKT", "TSLA", "TSM", "V", "WMT"]
    # custom = ["CRM", "^VIX", "^GSPC"]
    return list(set(sp500 + custom))

def entry():
    end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=200)).strftime('%Y-%m-%d')
    tickers = get_all_tickers()
    data_cache ={}
    print(f"{len(tickers)} tickers found , {start_date=} {end_date=}")
    for ticker in tqdm(tickers):
        data = yf.download(ticker, start=start_date, end=end_date, interval='1h', auto_adjust=False, ignore_tz=True, progress=False)
        data_cache[ticker] = data
        # time.sleep(0.1)
    # Serialize
    with open(FYAHOO__OUTPUTFILENAME, 'wb') as f:
        pickle.dump(data_cache, f)

    # Sanity check
    # Deserialize
    with open(FYAHOO__OUTPUTFILENAME, 'rb') as f:
        loaded_data = pickle.load(f)


if __name__ == "__main__":
    entry()