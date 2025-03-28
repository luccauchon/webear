from multiprocessing import Lock, Process, Queue, Value, freeze_support
from utils import namespace_to_dict, dict_to_namespace, get_stub_dir
from datetime import datetime
import pandas as pd
import sys
import os
from ast import literal_eval
from loguru import logger
from trainers.day_ahead_binary_classification_rc1 import train as train_model


def optimize(configuration):
    _a_direction = 'up'
    _tav_dates = [["2024-09-01", "2025-03-15"],
                  ["2024-06-01", "2025-03-15"],
                  ["2024-01-01", "2025-03-15"],
                  ["2023-01-01", "2025-03-15"],
                  ["2020-01-01", "2025-03-15"]
                  ]
    _mes_dates = configuration.get("mes_dates", ["2025-03-16", "2025-03-22"])
    _available_x_seq_length = configuration.get("available_x_seq_length", [5, 10, 14])
    _available_margin = configuration.get("available_margin", [0, 0.2, 0.4])  # In order of preference
    _type_margin = 'relative'
    _x_cols = [['Close'] + ['day_of_week'],
               ['Close', 'Open', 'Low', 'High'] + ['day_of_week'],
               ['Close', 'Close_MA3'] + ['day_of_week'],
               ['Close', 'Close_MA3', 'Open'] + ['day_of_week'],
               ['Close', 'Close_MA3', 'Open', 'Open_MA3'] + ['day_of_week'],
               ['Close', 'Close_MA3', 'Open_MA3', 'Open', 'Low', 'High'] + ['day_of_week'],
               ]
    device="cuda"
    _master_df_source = configuration.get("master_df_source", None)  # Use the specified dataframe instead of using a dataframe from an experience
    assert _master_df_source is not None
    logger.debug(f"Reading {_master_df_source}...")
    df_source = pd.read_pickle(_master_df_source)
    _maxwell_root_dir = configuration.get("stub_dir", os.path.join(get_stub_dir(), f"maxwell__{pd.Timestamp.now().strftime('%m%d_%Hh%Mm')}"))

    results_of_experiences = {}
    base_output_dir_for_all_experiences = os.path.join(_maxwell_root_dir, f"__optimizers__")
    logger.debug(f"Number of runs: {len(_available_x_seq_length)*len(_available_margin)*len(_x_cols)*len(_tav_dates)}")
    for x_seq_length in _available_x_seq_length:
        for a_margin in _available_margin:
            for a_x_cols in _x_cols:
                for a_tav_dates in _tav_dates:
                    a_version = f"LX{x_seq_length}_M{a_margin}_CX{'_'.join(a_x_cols)}_T{''.join(a_tav_dates)}"
                    a_stub_dir = os.path.join(base_output_dir_for_all_experiences, f"__{_a_direction}__{a_margin}")
                    os.makedirs(a_stub_dir, exist_ok=True)
                    run_id = f'MXW_{datetime.now().strftime("%d__%H_%M")}'  # root directory used by runner
                    configuration_for_experience = {"margin": a_margin, "type_margin": _type_margin,
                                                    "direction": _a_direction,
                                                    "lr_decay_iters": 4001,
                                                    "max_iters": 4001,
                                                    "log_interval": 1000,
                                                    "learning_rate": 1e-3,
                                                    "min_lr": 1e-4,
                                                    "eval_interval": 1,
                                                    "weight_decay": 0.1,
                                                    "run_id": run_id, "version": a_version,
                                                    "tav_dates": a_tav_dates, "mes_dates": _mes_dates,
                                                    "stub_dir": a_stub_dir,
                                                    "batch_size": 1024,
                                                    "x_cols": a_x_cols,
                                                    "x_cols_to_norm": [],
                                                    "y_cols": [('Close_MA3', 'SPY')],
                                                    "x_seq_length": x_seq_length,
                                                    "y_seq_length": 1,
                                                    "device": device,
                                                    'df_source': df_source.copy()  # Use the same data for all the experiences
                                                    }
                    results = train_model(configuration_for_experience)
                    results_of_experiences.update({a_version: results})
    logger.debug(results_of_experiences)
    best_exp_based_on_val_loss, best_exp_based_on_val_acc = (None, 999999), (None, 0.)
    for key, data in results_of_experiences.items():
        val_acc, val_loss = data['best_val_accuracy'][0], data['best_val_loss'][0]
        if val_acc > best_exp_based_on_val_acc[1]:
            best_exp_based_on_val_acc = key, val_acc
        if val_loss < best_exp_based_on_val_loss[1]:
            best_exp_based_on_val_loss = key, val_loss
    best_exp_based_on_val_acc  = results_of_experiences[best_exp_based_on_val_acc[0]]
    best_exp_based_on_val_loss = results_of_experiences[best_exp_based_on_val_loss[0]]
    logger.info(f"Best experience based on validation accuracy\n: {best_exp_based_on_val_acc}")
    logger.info(f"Best experience based on validation loss\n: {best_exp_based_on_val_loss}")


if __name__ == '__main__':
    freeze_support()
    from base_configuration import *
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
    #pprint.PrettyPrinter(indent=4).pprint(namespace_to_dict(configuration))

    logger.remove()
    logger.add(sys.stdout, level=configuration.debug_level)

    ###########################################################################
    # Description
    ###########################################################################
    logger.info("The goal is to make a prediction about the value (higher or lower) based on the preceding P days")

    optimize(configuration = namespace_to_dict(configuration))