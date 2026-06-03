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
import os
import argparse
import pathlib
import json
import copy
import pickle
import pprint
from tqdm import tqdm
from algorithms.trade_prime_half_trend import trade_prime_half_trend_strategy, get_entry_type, get_volume_confirmed, get_higher_timeframe_strong_trend, get_relative_strength_vs_benchmark, get_candlestick_confirmation_pattern
from algorithms.trade_prime_half_trend import set_volumed_confirmed
from datetime import datetime, timedelta, date
import os
import time
from utils import format_execution_time, get_weekdays
from copy import deepcopy
from multiprocessing import freeze_support, Lock, Process, Queue, Value
from constants import FYAHOO__OUTPUTFILENAME, NB_WORKERS
import pickle
import psutil


def entry():
    dataset_id = "day"


if __name__ == "__main__":
    freeze_support()
    entry()