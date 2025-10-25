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
from constants import FYAHOO__OUTPUTFILENAME
import os
import datetime
import argparse
from algorithms.trade_prime_half_trend import trade_prime_half_trend_strategy
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('-c', '--config', required=False, default=r'D:\Temp2\test.json', help='Path to the configuration file')
    return parser.parse_args()


def entry():
    args = parse_args()
    config_file_path = args.config

    # Check if the config file exists
    if not os.path.exists(config_file_path):
        print(f"Config file '{config_file_path}' not found.")
        return

    timestamp = os.path.getmtime(FYAHOO__OUTPUTFILENAME)
    timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Reading {FYAHOO__OUTPUTFILENAME} , last modified {timestamp}")

    with open(config_file_path, 'r') as f:
        cfg = json.load(f)


    # trade_prime_half_trend_strategy(ticker, ticker_name, buy_setup=True, **kwargs):
if __name__ == "__main__":
    entry()