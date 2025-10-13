import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from algorithms import trade_prime_half_trend_strategy, trade_prime_half_trend_strategy_plus_volume_confirmation_and_atr_stop_loss
from tqdm import tqdm

# if __name__ == "__main__":
#     end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
#     start_date = (datetime.today() + timedelta(days=-31)).strftime('%Y-%m-%d')
#     # Download SPX data (^GSPC is Yahoo Finance ticker for S&P 500)
#     ticker_name = '^GSPC'
#     ticker = yf.download(ticker_name, start=start_date, end=end_date, interval='1h', auto_adjust=False, ignore_tz=True)
#     close_label, high_label, low_label = ('Close', ticker_name), ('High', ticker_name), ('Low', ticker_name)
#     if ticker.empty:
#         raise ValueError(f"Failed to download {ticker_name} data. Check your internet or yfinance.")
#     print(f"{ticker_name} data downloaded.")
#     print(f"{ticker.index[-1]=}    {ticker.index[0]=}")
#
#     trade_prime_half_trend_strategy(ticker=ticker, ticker_name=ticker_name)
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import requests
import time
import io

def get_sp500_tickers():
    """Fetch the current list of S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        tickers = []
        for row in table.find_all('tr')[1:]:  # Skip header
            ticker = row.find_all('td')[0].text.strip()
            # Fix tickers with dots or dashes (e.g., BRK.B → BRK-B for Yahoo)
            ticker = ticker.replace('.', '-')
            tickers.append(ticker)
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Fallback: hard-coded partial list (optional)
        return []

def fetch_nasdaq_nyse_tickers():
    """Fetch and return a list of common stock tickers from NASDAQ and NYSE."""
    tickers = set()

    # NASDAQ
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    # NYSE (and others like AMEX)
    other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    for url in [nasdaq_url, other_url]:
        try:
            response = requests.get(url)
            response.raise_for_status()
            lines = response.text.splitlines()
            # Skip header
            for line in lines[1:]:
                parts = line.split('|')
                if len(parts) > 1:
                    symbol = parts[0].strip()
                    # Skip test symbols, non-equity, etc.
                    if symbol and "$" not in symbol and "." not in symbol and len(symbol) <= 5:
                        # For 'otherlisted.txt', check if it's NYSE or AMEX
                        if "otherlisted" in url:
                            exchange = parts[2].strip() if len(parts) > 2 else ""
                            if exchange not in ["N", "A"]:  # N=NYSE, A=AMEX
                                continue
                        tickers.add(symbol)
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    return sorted(tickers)

def main():
    # Define date range
    end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=62)).strftime('%Y-%m-%d')
    print(f"  → Data from {start_date} to {end_date}")
    all_tickers = ["^GSPC", "SPY", "QQQ"] + ["ADBE", "AMD", "AMZN", "AAPL", "COST", "GOOG", "INTC", "MA", "META", "MSFT", "NFLX", "NVDA",
                                             "ORCL", "PINS", "PLTR", "RDDT", "RKT", "TSLA", "V", "WMT"]
    print(f"Total tickers found: {len(all_tickers)}")
    ticker__2__data, list_of_triggereds_setup = {}, {True:[], False: []}
    for buy_setup in [True, False]:
        for ticker_name in tqdm(all_tickers):
            try:
                if ticker_name not in ticker__2__data:
                    ticker = yf.download(ticker_name, start=start_date, end=end_date, interval='1h', auto_adjust=False, ignore_tz=True, progress=False)
                    ticker__2__data.update({ticker_name: ticker})
                    # Be kind to Yahoo Finance—add delay
                    time.sleep(0.5)  # 0.5 sec between requests

                ticker = ticker__2__data[ticker_name].copy()
                if ticker.empty:
                    print(f"  → No data for {ticker_name}")
                    continue

                # Optional: Skip if not enough data points (e.g., < 10 hours)
                if len(ticker) < 10:
                    print(f"  → Insufficient data for {ticker_name}")
                    continue

                df = trade_prime_half_trend_strategy(ticker=ticker, ticker_name=ticker_name, buy_setup=buy_setup)
                # df = trade_prime_half_trend_strategy_plus_volume_confirmation_and_atr_stop_loss(ticker=ticker, ticker_name=ticker_name, buy_setup=buy_setup)
                last_trigger = df[df[('setup_triggered', ticker_name)]].tail(1)
                # Check if last_trigger is today
                if not last_trigger.empty and last_trigger.index[0].date() == pd.to_datetime('today').date():
                    list_of_triggereds_setup[buy_setup].append(last_trigger)
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"  → Error processing {ticker_name}: {e}")
                time.sleep(1)  # extra delay on error

    for buy_setup in [True, False]:
        print(f"\n\n\n[{'BUY SETUP' if buy_setup else 'SELL SETUP'}]*******************************************************************")
        ztr = ''
        for setup in list_of_triggereds_setup[buy_setup]:
            ztr += f'{setup["ticker_name"].values} => {setup.index[0]}' + '\n '
        print(ztr)
if __name__ == "__main__":
    main()