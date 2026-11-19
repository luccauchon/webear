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
from utils import get_filename_for_dataset, send_html_email
from fetchers.data_factory import factory_load_data
import os
import warnings
import optuna
from sklearn.model_selection import TimeSeriesSplit
from constants import GET_EMAILS, TITLE_WEBEAR, CLAUSE_NON_RESPONSABILITE, GET_SUBJECT
from datetime import datetime
import sys
from tqdm import tqdm
from argparse import Namespace
from optimizers.oerh.realtime_and_backtest_hyperparameter_search_optuna import entry as oerh_entry_point
from optimizers.apcs.realtime_and_backtest_hyperparameter_search_optuna import entry as apcs_entry_point
from optimizers.autotune.realtime_and_backtest_hyperparameter_search_optuna import entry as autotune_entry_point
from optimizers.dgdr.realtime_and_backtest_hyperparameter_search_optuna import entry as dgdr_entry_point
from utils import get_taurus_v1_models, get_taurus_v2_models, get_next_step, round_price_for_put_credit_spread, compute_break_even_credit
import textwrap
# Codes couleur ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument('--retro-bars-n', type=int, default=90, help="")
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Verbose output')
    return parser


def entry(args):
    func_models = get_taurus_v1_models
    if getattr(args, "version", "v1") == "v2": func_models = get_taurus_v2_models
    ticker = "^GSPC"
    close_col = ("Close", ticker)
    df_allin = factory_load_data(_dataset_id="day", _ticker=ticker, _args={})
    n_bars = args.retro_bars_n
    dict_results = {"txt": {"oerh": ""}}
    ###########################################################################
    # OERH
    ###########################################################################
    oerh_models = func_models()["oerh"]
    for model_name, model_info in oerh_models.items():
        model_path = model_info["filepath"]
        model__2__results = {"wins": [], "losses": [], "passes": []}
        original_test_win_rate, first_day, last_day = -1., None, None
        for clip_n in range(1, n_bars):
            config = Namespace(realtime=True, model_path=model_path, use_realtime_data=False, verbose=False, return_values_as_dict=True, clip_n=clip_n)
            result_oerh = oerh_entry_point(args=config)
            if first_day is None: first_day = result_oerh['current_date']
            last_day = result_oerh['current_date']
            if 1 == result_oerh["signal"]:
                lookahead_bars = result_oerh['lookahead_bars']
                entry_price = result_oerh['current_price'] * (1 + result_oerh['threshold_pct'])
                assert math.isclose(entry_price, result_oerh['target_price'], abs_tol=0.1)
                assert result_oerh['dataset_id'] == "day"
                if 'long_accuracy::any_half_B' == result_oerh['metric_target_type']:
                    date_future_t_lookahead = get_next_step(the_date=result_oerh['current_date'], dataset_id=result_oerh['dataset_id'], nn=lookahead_bars // 2)
                elif 'long_accuracy::exact' == result_oerh['metric_target_type']:
                    date_future_t_lookahead = get_next_step(the_date=result_oerh['current_date'], dataset_id=result_oerh['dataset_id'], nn=lookahead_bars)
                else: assert False, f"{result_oerh['metric_target_type']}  {model_path}"
                future_value = df_allin[close_col].loc[date_future_t_lookahead]
                if original_test_win_rate == -1.: original_test_win_rate = result_oerh["val_win_rate"]
                str_strike_price = round_price_for_put_credit_spread(entry_price)
                if future_value < str_strike_price: model__2__results["losses"].append(clip_n)
                else: model__2__results["wins"].append(clip_n)
            else: model__2__results["passes"].append(clip_n)
        actual_win_rate = len(model__2__results["wins"]) / (len(model__2__results["wins"]) + len(model__2__results["losses"]))
        actual_density = (len(model__2__results["wins"]) + len(model__2__results["losses"])) / (len(model__2__results["wins"]) + len(model__2__results["losses"]) + len(model__2__results["passes"]))
        _txt = (
            f"{BOLD}{CYAN}📊 For {Path(model_path).stem}:{RESET}\n"
            f"    {GREEN}📈 Win Rate{RESET}   : {BOLD}{actual_win_rate:0.2%}{RESET} ({original_test_win_rate:0.2%})"
            f"    {BLUE}🧬 Density{RESET}    : {BOLD}{actual_density:0.2%}{RESET}"
            f"    {YELLOW}📅 Tested for{RESET} : {n_bars} bars, from {last_day} to {first_day}"
        )
        if args.verbose: print(_txt)
        dict_results["txt"]["oerh"] = dict_results["txt"]["oerh"] + "\n" + _txt
    return dict_results


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)