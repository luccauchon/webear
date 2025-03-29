from multiprocessing import Lock, Process, Queue, Value, freeze_support
import pprint
import pandas as pd
from trainers.day_ahead_binary_classification_rc1 import train as train_model
from trainers.day_ahead_binary_classification_rc1 import create_meta_model as create_meta_model
from trainers.day_ahead_binary_classification_rc1 import generate_dataloader_for_inference as get_dataloader
from datetime import datetime
from loguru import logger
from pathlib import Path
import pathlib
import sys
from ast import literal_eval
import os
import json
from utils import extract_info_from_filename, previous_weekday_with_check, namespace_to_dict, dict_to_namespace, next_weekday_with_check, is_weekday, get_stub_dir
import torch


def scan_results(_all_candidats):
    _best_accuracy, _best_loss = 0., 999999.
    _best_accuracy__candidat, _best_accuracy__candidats = None, []
    _best_lost__candidat, _best_lost__candidats = None, []
    for _one_candidat in _all_candidats:
            if 'val_accuracy' in _one_candidat:
                if float(_one_candidat['val_accuracy']) > _best_accuracy:
                    _best_accuracy = float(_one_candidat['val_accuracy'])
                    _best_accuracy__candidat = _one_candidat
                    _best_accuracy__candidats = []
                elif float(_one_candidat['val_accuracy']) == _best_accuracy:
                    _best_accuracy__candidats.append(_one_candidat)
            if 'val_loss' in _one_candidat:
                if float(_one_candidat['val_loss']) < _best_loss:
                    _best_loss = float(_one_candidat['val_loss'])
                    _best_lost__candidat = _one_candidat
                    _best_lost__candidats = []
                elif float(_one_candidat['val_loss']) == _best_loss:
                    _best_lost__candidats.append(_one_candidat)
    # Switching model, if necessary
    for _one_candidat in _best_accuracy__candidats:
        assert float(_best_accuracy__candidat['val_accuracy']) == float(_one_candidat['val_accuracy'])
        if float(_one_candidat['with_val_loss']) < float(_best_accuracy__candidat['with_val_loss']):
            _best_accuracy__candidat = _one_candidat
    for _one_candidat in _best_lost__candidats:
        assert float(_best_lost__candidat['val_loss']) == float(_one_candidat['val_loss'])
        try:
            if float(_one_candidat['with_val_accuracy']) > float(_best_accuracy__candidat['with_val_accuracy']):
                _best_lost__candidat = _one_candidat
        except Exception as eee:
            logger.warning(eee)
            logger.warning(f"\n{_one_candidat=}\n{_best_accuracy__candidat=}")
    return _best_lost__candidat, _best_accuracy__candidat


def start_runner(configuration):
    ###########################################################################
    # Parameters for Kepler
    ###########################################################################
    _skip_training_with_already_computed_results = configuration.get("skip_training_with_already_computed_results", [])
    _fast_execution_for_debugging                = configuration.get("fast_execution_for_debugging", False)

    _tav_dates                    = configuration.get("tav_dates", ["2024-01-01", "2025-03-08"])
    _mes_dates                    = configuration.get("mes_dates", ["2025-03-09", "2025-03-15"])
    _inf_dates                    = configuration.get("inf_dates", ["2025-03-16", "2025-03-29"])

    _fetch_new_dataframe         = configuration.get("fetch_new_dataframe", False)  # Use Yahoo! Finance to download data instead of using a dataframe from an experience
    _master_df_source            = configuration.get("master_df_source", None)  # Use the specified dataframe instead of using a dataframe from an experience
    _device                      = configuration.get("device", "cuda")
    _power_of_noise              = configuration.get("inf_power_of_noise", 0.1)
    _frequency_of_noise          = configuration.get("inf_frequency_of_noise", 0.25)
    nb_iter_test_in_inference    = configuration.get("nb_iter_test_in_inference", 50)
    _today                       = configuration.get("today", pd.Timestamp.now().date())
    if _fast_execution_for_debugging:
        nb_iter_test_in_inference               = 5
    _kepler_root_dir             = configuration.get("stub_dir", os.path.join(get_stub_dir(), f"kepler__{pd.Timestamp.now().strftime('%m%d_%Hh%Mm')}"))

    ###########################################################################
    #
    ###########################################################################
    output_dir_of_experiences = []
    if isinstance(_skip_training_with_already_computed_results, list) and 1==len(_skip_training_with_already_computed_results):
        root_dir = _skip_training_with_already_computed_results[0]
        directories = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
        output_dir_of_experiences = directories

    ###########################################################################
    # Perform multiple training
    ###########################################################################
    assert isinstance(output_dir_of_experiences, list)
    if 0 == len(output_dir_of_experiences):
        df_source = _master_df_source.copy() if _master_df_source is not None else None
        base_output_dir_for_all_experiences = os.path.join(_kepler_root_dir, f"__trainers__")
        for _a_direction in ['up', 'down']:
            a_version = f"LX{_a_direction}_"
            a_stub_dir = os.path.join(base_output_dir_for_all_experiences, f"__{_a_direction}__")
            os.makedirs(a_stub_dir, exist_ok=True)
            run_id = f'RKRC1_{datetime.now().strftime("%d__%H_%M")}'  # root directory used by runner
            configuration_for_experience = {"margin": configuration.get("runner__margin", 1),
                                            "type_margin": configuration.get("runner__type_margin", 'fixed'),
                                            "direction": _a_direction,
                                            "lr_decay_iters": configuration.get("runner__lr_decay_iters", 5000),
                                            "run_id": run_id, "version": a_version,
                                            "max_iters": configuration.get("runner__max_iters", 5000),
                                            "tav_dates": _tav_dates, "mes_dates": _mes_dates,
                                            "stub_dir": a_stub_dir,
                                            "batch_size": configuration.get("runner__batch_size", 4096),
                                            "eval_interval": configuration.get("runner__eval_interval", 10),
                                            "log_interval": configuration.get("runner__log_interval", 5000),
                                            "learning_rate": configuration.get("runner__learning_rate", 1e-3),
                                            "min_lr": configuration.get("runner__min_lr", 1e-5),
                                            "weight_decay": configuration.get("runner__weight_decay", 0.1),
                                            "x_cols": configuration.get("runner__x_cols", ['Close', 'High', 'Low', 'Open'] + ['day_of_week']),
                                            "x_cols_to_norm": configuration.get("runner__x_cols_to_norm", []),
                                            "y_cols": configuration.get("runner__y_cols", [('Close_MA5', 'SPY')]),
                                            "x_seq_length": configuration.get("x_seq_length", 4),
                                            "y_seq_length": configuration.get("y_seq_length", 1),
                                            "device": _device,
                                            "save_checkpoint": True,
                                            }
            if df_source is not None:  # Use the same data for all the experiences
                configuration_for_experience.update({'df_source': df_source})
            if _fast_execution_for_debugging:
                configuration_for_experience.update({'max_iters': 20, 'log_interval': 5})
            # Launch training
            results = train_model(configuration_for_experience)
            if df_source is None:
                df_source = results['df_source']
            logger.debug(results)
            output_dir_of_experiences.append(str(Path(results['output_dir']).parent))
        output_dir_of_experiences = list(set(output_dir_of_experiences))
        assert 2 == len(output_dir_of_experiences)
    else:
        logger.info(f"Skipping training , using those results: {output_dir_of_experiences}")
        assert _master_df_source is None

    ###########################################################################
    # Select best models for our Meta-Model
    ###########################################################################
    experience__2__results, candidats = {}, {'up': [], 'down': []}
    for a_direction in ['up', 'down']:
        experience__2__results.update({a_direction: []})
        for all_experiences_in_one_direction in output_dir_of_experiences:
            all_checkpoints_for_direction = [file for file in Path(all_experiences_in_one_direction).rglob('*.pt') if 'checkpoints' in str(file)]
            for filename in all_checkpoints_for_direction:
                is_up = True if "__up__" in str(filename) else False
                is_down = True if "__down__" in str(filename) else False
                assert (is_up and not is_down) or (not is_up and is_down)
                if (a_direction == 'up' and is_down) or (a_direction == 'down' and is_up):
                    continue
                experience_name = str(filename.parent.parent.name)
                info = extract_info_from_filename(Path(filename).name)
                assert info is not None
                strz = f"\nFilename: {Path(filename).stem}\n" + f"Metric 1 Name: {info['metric1_name']}\n" + f"Metric 1 Value: {info['metric1_value']}\n"
                strz += f"Metric 2 Name: {info['metric2_name']}\n" + f"Metric 2 Value: {info['metric2_value']}\n" + f"Epoch: {info['epoch']}\n" + "-" * 50
                tmp_results = {"name": f"{experience_name}__{str(filename.stem)}"}
                tmp_results.update({"model_path": filename})
                if info['metric1_name'] == 'val_accuracy':
                    test_accuracy = info['metric1_value']
                    with_val_loss = info['metric2_value']
                    tmp_results.update({'val_accuracy': test_accuracy})
                    tmp_results.update({'with_val_loss': with_val_loss})
                if info['metric1_name'] == 'val_loss':
                    test_loss = info['metric1_value']
                    with_val_accuracy = info['metric2_value']
                    tmp_results.update({'val_loss': test_loss})
                    tmp_results.update({'with_val_accuracy': with_val_accuracy})
                tmp_results.update({"directory": str(filename.parent.parent)})
                tmp_results.update({"meta_information": os.path.join(str(filename.parent.parent), "results.json")})
                # Add information in tmp_results in order to select the best model
                with open(tmp_results["meta_information"], 'r') as f:
                    tmp_results.update({'margin': json.load(f)['margin']})
                tmp_results.update({"df": os.path.join(str(filename.parent.parent), "df.pkl")})
                experience__2__results.get(a_direction).append(tmp_results)
        # Scan the results for bests models
        best_lost, best_accuracy = scan_results(_all_candidats=experience__2__results.get(a_direction))
        if best_lost is not None:
            candidats[a_direction].append(best_lost)
        if best_accuracy is not None:
            candidats[a_direction].append(best_accuracy)
        if 0 == len(candidats):
            logger.warning(f"No candidates found for direction {a_direction}! cannot evaluate from {_inf_dates[0]} to {_inf_dates[1]}")
            return {}

    ###########################################################################
    # Create Meta Model
    ###########################################################################
    meta_model, df, params = create_meta_model(up_candidats=candidats['up'],
                                               down_candidats=candidats['down'],
                                               fetch_new_dataframe=_fetch_new_dataframe, device=_device, df_source=_master_df_source)

    ###########################################################################
    # Do inferences
    ###########################################################################
    start_date, end_date = pd.to_datetime(_inf_dates[0]), pd.to_datetime(_inf_dates[1])
    logger.info(f"Created meta model with {len(candidats['up'])+len(candidats['down'])} models , inferencing [{start_date.date()}] to [{end_date.date()}]")
    results_produced = {}
    # Iterate over days
    for n in range(int((end_date - start_date).days) + 1):
        date = start_date + pd.Timedelta(n, unit='days')
        if not is_weekday(date):
            continue
        if date not in df.index and date < df.index[-1]:
            continue  # Market close
        day_of_week_full = date.strftime('%A')
        yesterday = previous_weekday_with_check(date=date, df=df)
        real_time_execution = False if date in df.index else True  # False for backtesting (we have the ground truth)

        # Do one pass on dataset Test with no data augmentation
        data_loader_without_data_augmentation = get_dataloader(df=df, device=_device, _data_augmentation=False, date_to_predict=date,
                                                               real_time_execution=real_time_execution, **params)
        if data_loader_without_data_augmentation is None:
            continue
        if not real_time_execution:
            the_ground_truth_for_date = [y for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_without_data_augmentation)]
            assert 1 == len(the_ground_truth_for_date)
            the_ground_truth_for_date = the_ground_truth_for_date[0].item()
        nb_forward_pass = 0
        nb_pred_for_down, nb_pred_for_up, nb_pred_for_sideways, nb_pred_for_unknown = 0., 0., 0., 0.
        assert 1 == len([batch_idx for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_without_data_augmentation)])
        for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_without_data_augmentation):
            if not real_time_execution:
                assert all(y == the_ground_truth_for_date)
            _logits = meta_model(x=X)
            nb_forward_pass += 1
            if 99 == _logits:
                _tmp_str = f"For {date.strftime('%Y-%m-%d')} [{day_of_week_full}],"
                logger.debug(f"  [?] :| {_tmp_str}")
            nb_pred_for_up       += 1 if 1 == _logits else 0
            nb_pred_for_sideways += 1 if 0 == _logits else 0
            nb_pred_for_down     += 1 if -1 == _logits else 0
            nb_pred_for_unknown  += 1 if 99 == _logits else 0

        # Do multiple passes on dataset Test with data augmentation
        data_loader_with_data_augmentation = get_dataloader(df=df, device=_device, _data_augmentation=True, date_to_predict=date,
                                                            real_time_execution=real_time_execution, power_of_noise=_power_of_noise, frequency_of_noise=_frequency_of_noise, **params)
        for ee in range(0, nb_iter_test_in_inference):
            for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_with_data_augmentation):
                if not real_time_execution:
                    assert all(y == the_ground_truth_for_date)
                _logits = meta_model(x=X)
                nb_forward_pass += 1
                nb_pred_for_up       += 1 if 1 == _logits else 0
                nb_pred_for_sideways += 1 if 0 == _logits else 0
                nb_pred_for_down     += 1 if -1 == _logits else 0
                nb_pred_for_unknown  += 1 if 99 == _logits else 0
        prediction_for_down, prediction_for_up = nb_pred_for_down / nb_forward_pass, nb_pred_for_up / nb_forward_pass
        prediction_for_sideways, prediction_for_unknown = nb_pred_for_sideways / nb_forward_pass, nb_pred_for_unknown / nb_forward_pass
        assert 0.99 < prediction_for_down+prediction_for_up+prediction_for_sideways+prediction_for_unknown < 1.01
        predictions = {'down': prediction_for_down,'up': prediction_for_up,'sideways': prediction_for_sideways,'unknown': prediction_for_unknown}
        prediction_label, prediction_confidence = max(predictions.items(), key=lambda x: x[1])
        #pre_str = "[*] " if _today == date.date() else ""
        print(f"{date.strftime('%Y-%m-%d')} : {predictions}")
        #results_produced.update({f"{date.strftime('%Y-%m-%d')}": {"prediction": prediction, "confidence": confidence, "ground_truth": the_ground_truth_for_date}})
        # if not real_time_execution:
        #     close_value_yesterday = df.loc[yesterday][params['y_cols']].values[0]
        #     tmp_str = f"higher than {close_value_yesterday:.2f}$" if 1 == prediction else f"lower than {close_value_yesterday:.2f}$"
        #     pre_str += f":) " if the_ground_truth_for_date == prediction else ":( "
        #     if -1 != prediction:
        #         logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], the ground truth is {the_ground_truth_for_date} , prediction is {prediction} with {confidence * 100:.2f}% confidence ({tmp_str})")
        #     else:
        #         tmp_str = f"\u2191 or \u2193 than {close_value_yesterday:.2f}$"
        #         logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], the ground truth is {the_ground_truth_for_date} , prediction is unstable because of {confidence * 100:.2f}% confidence > ({tmp_str})")
        #     results_produced.update({f"{date.strftime('%Y-%m-%d')}": {"prediction": prediction, "confidence": confidence, "ground_truth": the_ground_truth_for_date}})
        # else:
        #     assert 1 == len(params['y_cols'])
        #     try:
        #         close_value_yesterday = df.loc[yesterday][params['y_cols']].values[0]
        #     except Exception as ee:
        #         logger.warning(f"There is no data for yesterday=({yesterday.strftime('%Y-%m-%d')}) , so can't predict {date.strftime('%Y-%m-%d')}")
        #         continue
        #     if -1 != prediction:
        #         tmp_str = f"higher than {close_value_yesterday:.2f}$" if 1 == prediction else f"lower than {close_value_yesterday:.2f}$"
        #         logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], prediction is {prediction} with {confidence * 100:.2f}% confidence > ({tmp_str})")
        #     else:
        #         tmp_str = f"\u2191 or \u2193 than {close_value_yesterday:.2f}$"
        #         logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], prediction is unstable because of {confidence * 100:.2f}% confidence > ({tmp_str})")
    return results_produced


if __name__ == '__main__':
    freeze_support()
    from base_configuration import *
    logger.info(f"\n{'*' * 80}\nUsing One Day Ahead Binary Classifier RC1 , train multiple models (up & down) and evaluate a Meta-Model on inference dates\n{'*' * 80}")

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
    config.update({k: globals()[k] for k in globals() if k.startswith("runner__")})
    tmp = {k: namespace[k] for k in [k for k, v in namespace.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]}
    config.update({k: tmp[k] for k, v in config.items() if k in tmp})
    configuration = dict_to_namespace(config)
    # -----------------------------------------------------------------------------
    # pprint.PrettyPrinter(indent=4).pprint(namespace_to_dict(configuration))

    logger.remove()
    logger.add(sys.stdout, level=configuration.debug_level)
    configuration = namespace_to_dict(configuration)
    start_runner(configuration)