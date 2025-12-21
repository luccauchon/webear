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
import re

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


# Step 2: Download data for all tickers for a given date range
def download_sp500_data(tickers, start_date, end_date):
    all_data = {}
    failed_tickers = []

    print(f"Downloading data from {start_date} to {end_date} for {len(tickers)} tickers...")
    for i, ticker in enumerate(tqdm(tickers)):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False, ignore_tz=True, progress=False)
            if not data.empty:
                all_data[ticker] = data
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            failed_tickers.append(ticker)
        time.sleep(0.25)  # Be respectful to Yahoo's servers

    print(f"\nDownload complete for {start_date}–{end_date}. Failed tickers: {failed_tickers}")
    return all_data, failed_tickers


# Step 3: Generate rolling 3-year windows ending no later than today + 1 day
def generate_3year_windows(end_year, min_start_year=2000):
    """
    Returns list of (start_date_str, end_date_str) for rolling 3-year windows:
    e.g., ('2023-01-01', '2025-12-31'), ('2022-01-01', '2024-12-31'), ...
    Stops when start year < min_start_year.
    """
    windows = []
    current_end_year = end_year
    while current_end_year - 2 >= min_start_year:
        start_year = current_end_year - 2
        start_date = f"{start_year}-01-01"
        end_date = f"{current_end_year}-12-31"
        windows.append((start_date, end_date))
        current_end_year -= 1
    return windows


# Step 4: Main execution
if __name__ == "__main__":
    # Save this web-page to data
    # https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    from pathlib import Path
    html_file = (Path(__file__).parent / '..' / '..' / 'data' / 'List of S&P 500 companies - Wikipedia.html').resolve()

    if not os.path.exists(html_file):
        raise FileNotFoundError(f"HTML file not found: {html_file}")

    tickers = get_sp500_tickers_from_file(html_file)
    print(f"Found {len(tickers)} tickers.")

    # Determine current year (today is 2025-12-21, so end_year = 2025)
    today = datetime.today()
    current_year = today.year
    # Allow window to include today + 1 day (for completeness)
    end_date_default = (today + timedelta(days=1)).strftime('%Y-%m-%d')

    # Generate 3-year rolling windows: 2023–2025, 2022–2024, 2021–2023, etc.
    windows = generate_3year_windows(end_year=current_year, min_start_year=2000)

    output_dir = Path(FYAHOO_SPX500__OUTPUTFILENAME).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for start_str, end_str in windows:
        print(f"\n{'='*60}")
        print(f"Processing window: {start_str} to {end_str}")

        # Clip end_str to today+1 if beyond
        window_end = min(datetime.strptime(end_str, "%Y-%m-%d"), today + timedelta(days=1)).strftime('%Y-%m-%d')

        data_dict, failed = download_sp500_data(tickers, start_date=start_str, end_date=window_end)

        if data_dict:
            combined_df = pd.concat(data_dict, names=['Ticker', 'Date'])

            # Derive filename: e.g., spx500_2023-2025.parquet
            base_name = os.path.basename(FYAHOO_SPX500__OUTPUTFILENAME)
            name_no_ext = re.sub(r'\.parquet$', '', base_name, flags=re.IGNORECASE)
            year_suffix = f"{start_str[:4]}-{end_str[:4]}"
            output_file = output_dir / f"{name_no_ext}_{year_suffix}.parquet"

            combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
            print(f"Saved to {output_file}")
        else:
            print(f"No data downloaded for window {start_str}–{end_str}.")