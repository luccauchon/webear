import yfinance as yf
import pandas as pd
from utils import get_df_SPY_and_VIX_virgin_at_minutes
from constants import BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR
import os
import schedule
import time


def dump_dataframes_to_disk():
    _today = pd.Timestamp.now().date()
    print(f"Running the extraction for SPY/SPX/VIX/NDX @minutes for  {_today}")
    df_spy, df_spx, df_vix, df_ndx = get_df_SPY_and_VIX_virgin_at_minutes()
    df_spy.to_pickle(os.path.join(BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR, f"{_today}__SPY.pkl"))
    df_spx.to_pickle(os.path.join(BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR, f"{_today}__SPX.pkl"))
    df_vix.to_pickle(os.path.join(BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR, f"{_today}__VIX.pkl"))
    df_ndx.to_pickle(os.path.join(BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR, f"{_today}__NDX.pkl"))

if __name__ == '__main__':
    dump_dataframes_to_disk()
    schedule.every().day.at("17:00").do(dump_dataframes_to_disk)  # Run every day at 17:00

    while True:
        schedule.run_pending()
        time.sleep(1)