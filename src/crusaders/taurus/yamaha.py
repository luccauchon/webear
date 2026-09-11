try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import numpy as np
import pandas as pd
from functools import lru_cache
from numba import njit, prange
import math
from pathlib import Path
import copy
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import pickle
import argparse
from utils import get_filename_for_dataset
from fetchers.data_factory import factory_load_data
import os
import warnings
import optuna
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import sys
from tqdm import tqdm
from argparse import Namespace
from optimizers.oerh.realtime_and_backtest_hyperparameter_search_optuna import entry as oerh_entry_point
from optimizers.autotune.realtime_and_backtest_hyperparameter_search_optuna import entry as autotune_entry_point
from utils import get_taurus_v1_models, get_next_step, round_price_for_put_credit_spread


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    return parser


def entry(args):
    verbose = False
    def dual_print(message, buffer_str):
        print(message)
        return buffer_str + message

    ###########################################################################
    msg_str = ""

    # AUTOTUNE
    ###########################################################################
    msg_str += dual_print(f"Models AutoTune", msg_str)
    autotune_models = get_taurus_v1_models()["autotune"]
    for model_name, model_info in autotune_models.items():
        model_path = model_info["filepath"]
        config = Namespace(realtime=True, model_path=model_path, use_realtime_data=True, verbose=verbose, return_values_as_dict=True, clip_n=0)
        result_autotune = autotune_entry_point(args=config)
        if 0 != result_autotune["signal"]:
            msg_str += dual_print(result_autotune["status_message"], msg_str)


    ###########################################################################
    # OERH
    ###########################################################################
    msg_str += dual_print(f"Models OERH", msg_str)
    oerh_models = get_taurus_v1_models()["oerh"]
    for model_name, model_info in oerh_models.items():
        model_path = model_info["filepath"]
        config = Namespace(realtime=True, model_path=model_path, use_realtime_data=True, verbose=verbose, return_values_as_dict=True, clip_n=0)
        result_oerh = oerh_entry_point(args=config)
        if 1 == result_oerh["signal"]:
            assert 'long_accuracy::any_half_B' == result_oerh['metric_target_type']
            lookahead_bars = result_oerh['lookahead_bars']
            entry_price = result_oerh['current_price'] * (1 + result_oerh['threshold_pct'])
            assert math.isclose(entry_price, result_oerh['target_price'], abs_tol=0.1)
            date_future_t_half_lookahead = get_next_step(the_date=result_oerh['current_date'], dataset_id=result_oerh['dataset_id'], nn=lookahead_bars//2)
            now = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
            target_price = (1+result_oerh["threshold_pct"]) * entry_price
            test_win_rate = result_oerh["val_win_rate"]
            msg_str += dual_print(f"Today @{now} , buy a Put Credit Spread located at {round_price_for_put_credit_spread(entry_price):.0f} , and get profit starting on {date_future_t_half_lookahead.strftime("%Y%m%d")} , "
                  f"targetting that price shall be above {target_price:.0f} at that point on, grabbing time decay.\n\t"
                  f"Model has a {test_win_rate:.1%} Test Win Rate", msg_str)
        else:
            if verbose: msg_str += dual_print(f"\t{model_name} ({Path(model_path).stem}) has not triggered a signal", msg_str)


    ###########################################################################
    # DGDR
    ###########################################################################

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)