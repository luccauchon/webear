import yfinance as yf
import pandas as pd
from utils import get_df_SPY_and_VIX_virgin_at_minutes
import os
import schedule
import time
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


def get_option_chain(data_dir):
    # -----------------------------
    # Settings
    # -----------------------------
    TICKER = "^SPX"

    # -----------------------------
    # Load Ticker
    # -----------------------------
    spx = yf.Ticker(TICKER)

    # Get underlying price snapshot
    spot = spx.history(period="1d")["Close"].iloc[-1]

    timestamp = datetime.utcnow()

    print(f"Downloading options snapshot at {timestamp}")
    print(f"Spot price: {spot}")

    # -----------------------------
    # Loop Through Expirations
    # -----------------------------
    for expiry in spx.options:
        try:
            opt_chain = spx.option_chain(expiry)
            calls = opt_chain.calls.copy()
            puts = opt_chain.puts.copy()

            # Add metadata
            calls["snapshot_time"] = timestamp
            puts["snapshot_time"] = timestamp
            calls["spot"] = spot
            puts["spot"] = spot
            calls["expiry"] = expiry
            puts["expiry"] = expiry

            snapshot = {
                "calls": calls,
                "puts": puts,
                "spot": spot,
                "timestamp": timestamp
            }

            file_path = os.path.join(data_dir, f"{expiry}.pkl")

            # Append if file exists
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                data.append(snapshot)
            else:
                data = [snapshot]

            with open(file_path, "wb") as f:
                pickle.dump(data, f)

            print(f"Saved snapshot for expiry {expiry}")

        except Exception as e:
            print(f"Error with expiry {expiry}: {e}")

    print("Done.")


def dump_option_chain_to_disk(output_dir=r"D:\Finance\data\option_chain"):
    data_dir = os.path.join(output_dir, "SPX")
    os.makedirs(data_dir, exist_ok=True)

    _today = pd.Timestamp.now().date()
    print(f"Running the extraction for Options Chain  {_today}")
    get_option_chain(data_dir=data_dir)


if __name__ == '__main__':
    dump_option_chain_to_disk()
    schedule.every().day.at("19:00").do(dump_option_chain_to_disk)  # Run every day at 17:00

    while True:
        schedule.run_pending()
        time.sleep(1)