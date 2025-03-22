import os
import pprint
from ast import literal_eval
from tqdm import tqdm
import itertools
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import TripleIndicesLookAheadBinaryClassificationDataset
from models import LSTMClassification, LSTMMetaClassification
from utils import get_latest_spy_and_vix_dataframe, find_next_sunday, find_next_saturday, find_previous_saturday, all_dicts_equal, namespace_to_dict, dict_to_namespace, get_stub_dir, get_df_SPY_and_VIX, generate_indices_with_cutoff_day, calculate_binary_classification_metrics, generate_indices_with_multiple_cutoff_day, generate_indices_basic_style, previous_weekday, next_weekday
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import torch
import numpy as np
from loguru import logger
import sys
import pandas as pd
import json
import math
from datetime import datetime, timedelta
from runners import kepler_202503_rc1 as my_runner


def generate_campaign_dates(configuration):
    training_end_date = pd.to_datetime(configuration.get("train__tav_t2", None))
    assert training_end_date is not None
    length_of_training_data = configuration.get("length_of_training_data", 365 * 2)
    length_of_mes = configuration.get("length_of_mes", 7)
    length_of_inf = configuration.get("length_of_inf", 7)
    assert 0 == length_of_mes % 7
    assert 0 == length_of_inf % 7
    train__t2 = find_next_saturday(training_end_date, move_to_next_week=False)
    train__t1 = find_next_saturday(train__t2 - pd.Timedelta(length_of_training_data, unit='days'))

    mes__t1 = train__t2 + pd.Timedelta(1, unit='days')
    assert 6 == mes__t1.weekday()  # Sunday
    mes__t2 = find_previous_saturday(mes__t1 + pd.Timedelta(length_of_mes, unit='days'))
    assert 5 == mes__t2.weekday()  # Saturday

    inf__t1 = mes__t2 + pd.Timedelta(1, unit='days')
    assert 6 == inf__t1.weekday()  # Sunday
    inf__t2 = find_previous_saturday(inf__t1 + pd.Timedelta(length_of_mes, unit='days'))
    assert 5 == inf__t2.weekday()  # Saturday

    tav_dates = [train__t1.date(), train__t2.date()]
    mes_dates = [mes__t1.date(), mes__t2.date()]
    inf_dates = [inf__t1.date(), inf__t2.date()]
    assert 5 == tav_dates[1].weekday() and 6 == mes_dates[0].weekday() and mes_dates[0] > tav_dates[1] and 1 == (mes_dates[0] - tav_dates[1]).days
    assert 6 == inf_dates[0].weekday() and 5 == inf_dates[1].weekday() and inf_dates[0] > mes_dates[1] and 1 == (inf_dates[0] - mes_dates[1]).days
    return tav_dates, mes_dates, inf_dates


def start_campaign(configuration):
    start_date = datetime.strptime(configuration.get("start_date", "2022-01-01"), "%Y-%m-%d")
    num_weeks  = configuration.get("num_weeks", 52)  # Number of weeks to iterate
    configuration.update({"length_of_training_data": 365 * 2, "length_of_mes": 7, "length_of_inf": 7})
    logger.info(f"Starting campaign @{start_date.date()} for {num_weeks} weeks.")
    os.system('pause')
    master_df_source = get_latest_spy_and_vix_dataframe()  # Use always the same dataframe through all the campaign

    results_produced = {}
    # Iterate for each week
    for i in range(num_weeks):
        current_date = start_date + timedelta(weeks=i)
        configuration.update({"train__tav_t2": f"{current_date.date()}"})
        tav_dates, mes_dates, inf_dates = generate_campaign_dates(configuration)
        logger.info(f"\n tav dates are : {tav_dates[0]} -> {tav_dates[1]}\n mes dates are : {mes_dates[0]} -> {mes_dates[1]}\n inf dates are : {inf_dates[0]} -> {inf_dates[1]}")
        configuration.update({"tav_dates": [f"{str(fxx)}" for fxx in tav_dates],
                              "mes_dates": [f"{str(fxx)}" for fxx in mes_dates],
                              "inf_dates": [f"{str(fxx)}" for fxx in inf_dates], "_today": mes_dates[1]})

        configuration.update({"stub_dir": os.path.join(get_stub_dir(), "NewYork_backtesting", f"week_{i}")})
        configuration.update({"master_df_source": master_df_source.copy()})
        _tmp_results = my_runner.start_runner(configuration)
        assert len(_tmp_results) == len([a_date for a_date, _ in _tmp_results.items() if a_date not in results_produced])
        results_produced.update(_tmp_results)
    pprint.PrettyPrinter(indent=4).pprint(results_produced)
    compilation_for_positive_confidence = []
    for a_date, results in results_produced.items():
        if results['confidence'] > 0.5:
            success = 1 if results['ground_truth'] == results['prediction'] else 0
            compilation_for_positive_confidence.append(success)
    logger.info(f"When confidence was >0.5, succes rate is {np.count_nonzero(compilation_for_positive_confidence)/len(compilation_for_positive_confidence)*100}%")


if __name__ == '__main__':
    freeze_support()
    from base_configuration import *
    logger.info(f"\n{'*' * 80}\nPerform a campaign with |{my_runner.__name__}|\n{'*' * 80}")

    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]
    namespace = {}
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read())
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            if key in globals():
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(globals()[key]), f"{type(attempt)} != {type(globals()[key])}"
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                globals()[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
    config = {k: globals()[k] for k in config_keys}
    tmp = {k: namespace[k] for k in [k for k, v in namespace.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]}
    config.update({k: tmp[k] for k, v in config.items() if k in tmp})
    configuration = dict_to_namespace(config)
    # -----------------------------------------------------------------------------
    # pprint.PrettyPrinter(indent=4).pprint(configuration)
    configuration = namespace_to_dict(configuration)
    #configuration.update({"fast_execution_for_debugging": True})
    start_campaign(configuration)