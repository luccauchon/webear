import yfinance as yf
import pandas as pd
from utils import get_df_SPY_and_VIX, get_spy_and_vix_data_dir
import os
import schedule
import time


if __name__ == '__main__':
    _today = pd.Timestamp.now().date()

    print(f"Running the extraction for SPY/VIX {_today}")
    for interval in ['1d', '1mo']:
        df, _ = get_df_SPY_and_VIX(interval=interval)

        output_file = os.path.join(get_spy_and_vix_data_dir(), f"spy_and_vix__{_today}__{interval}.pkl")
        print(f"Saving to {output_file}")
        df.to_pickle(output_file)
