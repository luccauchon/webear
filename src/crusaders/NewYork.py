import os
import pprint
import time
from ast import literal_eval
from utils import get_latest_spy_and_vix_dataframe, find_next_saturday, find_previous_saturday, namespace_to_dict, dict_to_namespace, get_stub_dir
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import numpy as np
from loguru import logger
import sys
import pandas as pd
from datetime import datetime, timedelta
from runners import kepler_202503_rc1 as my_runner


def generate_campaign_dates(configuration):
    training_end_date = pd.to_datetime(configuration.get("crusader__tav_t2", None))
    assert training_end_date is not None
    length_of_training_data = configuration.get("crusader__length_of_training_data", 365 * 2)
    length_of_mes = configuration.get("crusader__length_of_mes", 7)
    length_of_inf = configuration.get("crusader__length_of_inf", 7)
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
    _start_date = datetime.strptime(configuration["crusader__start_date"], "%Y-%m-%d")
    _num_weeks  = int(configuration["crusader__num_weeks"])
    assert int(_num_weeks) > 0
    configuration.update({"crusader__length_of_training_data": 365 * 2, "crusader__length_of_mes": 7, "crusader__length_of_inf": 7})
    logger.info(f"Starting campaign @{_start_date.date()} for {_num_weeks} week{'' if 1==_num_weeks else 's'}.")
    master_df_source = get_latest_spy_and_vix_dataframe()  # Use always the same dataframe through all the campaign
    output_dir = os.path.join(get_stub_dir(), f"NewYork_{_start_date.date().strftime('%Y')}_{pd.Timestamp.now().strftime('%d_%Hh%Mm')}")
    os.makedirs(output_dir)
    results_produced = {}
    # Iterate for each week
    for i in range(_num_weeks):
        current_date = _start_date + timedelta(weeks=i)
        configuration.update({"crusader__tav_t2": f"{current_date.date()}"})
        tav_dates, mes_dates, inf_dates = generate_campaign_dates(configuration)
        logger.debug(f"\n tav dates are : {tav_dates[0]} -> {tav_dates[1]}\n mes dates are : {mes_dates[0]} -> {mes_dates[1]}\n inf dates are : {inf_dates[0]} -> {inf_dates[1]}")
        configuration.update({"runner__tav_dates": [f"{str(fxx)}" for fxx in tav_dates],
                              "runner__mes_dates": [f"{str(fxx)}" for fxx in mes_dates],
                              "runner__inf_dates": [f"{str(fxx)}" for fxx in inf_dates], "_today": mes_dates[1]})
        configuration.update({"stub_dir": os.path.join(output_dir, f"week_{i}")})
        configuration.update({"runner__master_df_source": master_df_source.copy()})
        _tmp_results, _accuracy = my_runner.start_runner(configuration)
        assert len(_tmp_results) == len([a_date for a_date, _ in _tmp_results.items() if a_date not in results_produced])
        results_produced.update(_tmp_results)
    #pprint.PrettyPrinter(indent=4).pprint(results_produced)
    for confidence_threshold in [0.5]:
        compilation_for_positive_confidence = []
        for a_date, results in results_produced.items():
            if results['prediction_confidence'] > confidence_threshold:
                success = 1 if results['ground_truth'] == results['prediction_value'] else 0
                compilation_for_positive_confidence.append(success)
        rate_of_positive_confidence = len(compilation_for_positive_confidence) / len(results_produced) if 0 != len(results_produced) else 0.
        rate_of_success_of_pos_conf = np.count_nonzero(compilation_for_positive_confidence)/len(compilation_for_positive_confidence) if 0 != len(compilation_for_positive_confidence) else 0
        logger.info(f"When confidence was gt {confidence_threshold*100:.1f}% (this happens {rate_of_positive_confidence*100:.1f}%), success rate is {rate_of_success_of_pos_conf*100:.1f}%")


if __name__ == '__main__':
    freeze_support()
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
            globals()[key] = val
    config = {k: globals()[k] for k in config_keys}
    config.update({k: globals()[k] for k in globals() if k.startswith("crusader__") or k.startswith("runner__") or k.startswith("trainer__") or k in ['device', 'seed_offset', 'stub_dir']})
    tmp = {k: namespace[k] for k in [k for k, v in namespace.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]}
    config.update({k: tmp[k] for k, v in config.items() if k in tmp})
    configuration = dict_to_namespace(config)
    # -----------------------------------------------------------------------------
    # pprint.PrettyPrinter(indent=4).pprint(configuration)
    configuration = namespace_to_dict(configuration)
    logger.remove()
    logger.add(sys.stdout, level=configuration.get("crusader__debug_level", "INFO"))
    start_campaign(configuration)