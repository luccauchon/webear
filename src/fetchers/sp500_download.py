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
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime, timedelta
from constants import FYAHOO_SPX500__OUTPUTFILENAME
from tqdm import tqdm

# Step 1: Parse S&P 500 tickers from your local HTML file
def get_sp500_tickers_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Find the table with S&P 500 constituents
    table = soup.find('table', {'id': 'constituents'})
    if table is None:
        # Fallback: try the first table if ID isn't found
        table = soup.find('table')

    tickers = []
    rows = table.find_all('tr')[1:]  # Skip header row

    for row in rows:
        cells = row.find_all('td')
        if cells:  # Ensure row has data
            ticker = cells[0].text.strip()
            # Fix ticker format for Yahoo Finance (e.g., BRK.B → BRK-B)
            tickers.append(ticker.replace('.', '-'))

    return tickers


# Step 2: Download data for all tickers
def download_sp500_data(tickers, period="max"):
    all_data = {}
    failed_tickers = []

    print(f"Downloading data for {len(tickers)} tickers...")
    for i, ticker in enumerate(tqdm(tickers)):
        try:
            # data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = "2020-01-01"
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False, ignore_tz=True, progress=False)
            if not data.empty:
                all_data[ticker] = data
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            failed_tickers.append(ticker)
        time.sleep(0.5)  # Be respectful to Yahoo's servers

    print(f"\nDownload complete. Failed tickers: {failed_tickers}")
    return all_data, failed_tickers


# Step 3: Main execution
if __name__ == "__main__":
    # Save this web-page to data
    # https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    from pathlib import Path
    html_file = (Path(__file__).parent / '..' / '..' / 'data' / 'List of S&P 500 companies - Wikipedia.html').resolve()
    # html_file = r'..\..\data\List of S&P 500 companies - Wikipedia.html'

    if not os.path.exists(html_file):
        raise FileNotFoundError(f"HTML file not found: {html_file}")

    tickers = get_sp500_tickers_from_file(html_file)
    print(f"Found {len(tickers)} tickers.")

    # Download data (change period as needed: "1y", "5y", "max", etc.)
    data_dict, failed = download_sp500_data(tickers, period="max")

    # Save combined data to CSV
    if data_dict:
        combined_df = pd.concat(data_dict, names=['Ticker', 'Date'])
        # Save DataFrame to Parquet
        combined_df.to_parquet(FYAHOO_SPX500__OUTPUTFILENAME, engine='pyarrow', compression='snappy')
        print(f"\nAll data saved to {FYAHOO_SPX500__OUTPUTFILENAME}")
    else:
        print("No data downloaded.")