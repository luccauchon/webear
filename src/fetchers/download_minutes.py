import yfinance as yf
import pandas as pd
from utils import get_df_SPY_and_VIX_virgin_at_minutes
import os
import schedule
import time


def dump_dataframes_to_disk():
    output_dir = r"D:\Finance\data\daily"
    _today = pd.Timestamp.now().date()
    print(f"Running the extraction for SPY/VIX @minutes for  {_today}")
    df_spy, df_vix = get_df_SPY_and_VIX_virgin_at_minutes()
    df_spy.to_pickle(os.path.join(output_dir, f"{_today}__SPY.pkl"))
    df_vix.to_pickle(os.path.join(output_dir, f"{_today}__VIX.pkl"))


if __name__ == '__main__':
    schedule.every().day.at("17:00").do(dump_dataframes_to_disk)  # Run every day at 17:00

    while True:
        schedule.run_pending()
        time.sleep(1)