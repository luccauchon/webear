from multiprocessing import Lock, Process, Queue, Value, freeze_support
import pprint
import numpy as np
import pandas as pd
from trainers.ahead_binary_classification_rc1 import train as train_model
from trainers.ahead_binary_classification_rc1 import create_meta_model as create_meta_model
from trainers.ahead_binary_classification_rc1 import generate_dataloader_for_inference as get_dataloader
from datetime import datetime
from loguru import logger
from pathlib import Path
import pathlib
import sys
from ast import literal_eval
import os
import json
from utils import is_it_next_week_after_last_week_of_df, extract_info_from_filename, previous_weekday_with_check, namespace_to_dict, dict_to_namespace, get_df_SPY_and_VIX, is_weekday, get_stub_dir, get_all_checkpoints, string_to_bool
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
    logger.info(f"\n{'*' * 80}\nUsing One Day Ahead Binary Classifier RC1 , train multiple models (up & down) and evaluate a Meta-Model on inference dates\n{'*' * 80}")
    _seed_offset = configuration.get("runner__seed_offset", 1234)
    logger.debug(f"Seed is {_seed_offset}")
    np.random.seed(_seed_offset)
    torch.manual_seed(_seed_offset)
    torch.cuda.manual_seed_all(_seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    ###########################################################################
    # Parameters for Kepler
    ###########################################################################
    _skip_training_with_already_computed_results = configuration["runner__skip_training_with_already_computed_results"]
    _fast_execution_for_debugging                = string_to_bool(configuration["runner__fast_execution_for_debugging"])

    _tav_dates                    = configuration["runner__tav_dates"]
    _mes_dates                    = configuration["runner__mes_dates"]
    _inf_dates                    = configuration["runner__inf_dates"]

    _fetch_new_dataframe         = string_to_bool(configuration["runner__fetch_new_dataframe"])  # Use Yahoo! Finance to download data instead of using a dataframe from an experience
    _master_df_source            = configuration["runner__master_df_source"]  # Use the specified dataframe instead of using a dataframe from an experience
    _device                      = configuration.get("device", "cuda")
    _power_of_noise_inf          = float(configuration["runner__inf_power_of_noise"])
    _frequency_of_noise_inf      = float(configuration["runner__inf_frequency_of_noise"])
    _nb_iter_test_in_inference   = int(configuration["runner__nb_iter_test_in_inference"])
    _today                       = configuration.get("runner__today", pd.Timestamp.now().date())
    _skip_monday                 = string_to_bool(configuration["runner__skip_monday"])
    if _fast_execution_for_debugging:
        _nb_iter_test_in_inference               = 5
    _run_id                      = configuration.get("runner__run_id", "123")
    if string_to_bool(configuration.get("runner__new_root_dir", "True")):
        _kepler_root_dir = configuration.get("stub_dir", os.path.join(get_stub_dir(), f"kepler__{_run_id}__{pd.Timestamp.now().strftime('%m%d_%Hh%Mm')}"))
    else:
        _kepler_root_dir = configuration.get("stub_dir", os.path.join(get_stub_dir(), f"kepler__{_run_id}"))
    _download_data_for_inf       = string_to_bool(configuration["runner__download_data_for_inf"])
    _data_interval               = None
    ###########################################################################
    #
    ###########################################################################
    output_dir_of_experiences = []
    if isinstance(_skip_training_with_already_computed_results, str):
        if os.path.exists(_skip_training_with_already_computed_results):
            root_dir = _skip_training_with_already_computed_results
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
            logger.debug(f"Models for {_a_direction} are located in {a_stub_dir}")
            run_id = f'RKRC1_{datetime.now().strftime("%d__%H_%M")}'  # root directory used by runner
            configuration_for_experience = {}
            configuration_for_experience.update(configuration)
            assert "trainer__force_download_data" not in configuration
            assert "trainer__save_checkpoint" not in configuration
            assert "trainer__shuffle_indices" not in configuration
            assert "trainer__decay_lr" not in configuration
            configuration_for_experience.update({"trainer__direction": _a_direction,
                                                 "trainer__run_id": run_id,
                                                 "trainer__version": a_version,
                                                 "trainer__tav_dates": _tav_dates,
                                                 "trainer__mes_dates": _mes_dates,
                                                 "trainer__force_download_data": False,
                                                 "trainer__save_checkpoint": True,
                                                 "trainer__shuffle_indices": False,
                                                 "trainer__data_augmentation": False,
                                                 "trainer__frequency_of_noise": 0.,
                                                 "trainer__power_of_noise": 0.,
                                                 "trainer__decay_lr": True,
                                                 "trainer__df_source": df_source,
                                                 "stub_dir": a_stub_dir,
                                                 "device": _device,
                                                })
            if _fast_execution_for_debugging:
                configuration_for_experience.update({'trainer__max_iters': 20, 'trainer__log_interval': 5})
            # Launch training
            results = train_model(configuration_for_experience)
            if df_source is None:
                df_source = results['df_source']
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
            all_checkpoints_for_direction = get_all_checkpoints(all_experiences_in_one_direction)
            for filename in all_checkpoints_for_direction:
                is_up = True if "__up__" in str(filename) else False
                is_down = True if "__down__" in str(filename) else False
                assert (is_up and not is_down) or (not is_up and is_down)
                if (a_direction == 'up' and is_down) or (a_direction == 'down' and is_up):
                    continue
                with open(os.path.join(str(Path(filename.parent.parent)), 'results.json'), 'r') as f:
                    _data_interval = json.load(f)['configuration']['trainer__data_interval']
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
        #if best_accuracy is not None:
        #    candidats[a_direction].append(best_accuracy)
        if 0 == len(candidats):
            logger.warning(f"No candidates found for direction {a_direction}! cannot evaluate from {_inf_dates[0]} to {_inf_dates[1]}")
            return {}

    ###########################################################################
    # Create Meta Model
    ###########################################################################
    meta_model, df, params = create_meta_model(up_candidats=candidats['up'],
                                               down_candidats=candidats['down'],
                                               fetch_new_dataframe=_fetch_new_dataframe, device=_device, df_source=_master_df_source)

    # Download data , if requested
    if _download_data_for_inf:
        logger.info(f"Fetching data from Yahoo! before inferences...   using data interval of {_data_interval}")
        df, _ = get_df_SPY_and_VIX(interval=_data_interval)

    ###########################################################################
    # Do inferences
    ###########################################################################
    start_date, end_date = pd.to_datetime(_inf_dates[0]), pd.to_datetime(_inf_dates[1])
    logger.info(f"Root directory: {_kepler_root_dir}")
    logger.info(f"Created meta model with {len(candidats['up'])+len(candidats['down'])} models , inferencing [{start_date.date()}] to [{end_date.date()}]")
    for type_of, all_candidats in zip(["UP", "DOWN"], [candidats['up'], candidats['down']]):
        for one_candidat in all_candidats:
            if 'val_accuracy' in one_candidat:
                logger.info(f"[{type_of}]  val_acc: {float(one_candidat['val_accuracy']):0.4}    val_loss: {float(one_candidat['with_val_loss']):0.4}")
            if 'val_loss' in one_candidat:
                logger.info(f"[{type_of}]  val_acc: {float(one_candidat['with_val_accuracy']):.4}    val_loss: {float(one_candidat['val_loss']):.4}")
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
        if _data_interval == '1wk':
            if real_time_execution:
                # Make sure we are 'in' the next week
                if not is_it_next_week_after_last_week_of_df(df=df, date=date):
                    continue
                # Since we are 'in' the next week, just use the last date to make a prediction, whenever day it might be.
                if date.date() != end_date.date():
                    continue
        # Do one pass on dataset Test with no data augmentation
        data_loader_without_data_augmentation = get_dataloader(df=df, device=_device, _data_augmentation=False, date_to_predict=date,
                                                               real_time_execution=real_time_execution, data_interval=_data_interval, **params)
        if data_loader_without_data_augmentation is None:
            continue
        the_ground_truth_for_date, the_ground_truth_label_for_date = None, None
        if not real_time_execution:
            the_ground_truth_for_date = [y for batch_idx, (X, y, abcdef) in enumerate(data_loader_without_data_augmentation)]
            assert 1 == len(the_ground_truth_for_date)
            the_ground_truth_for_date = the_ground_truth_for_date[0].item()
            the_ground_truth_label_for_date = 'up' if the_ground_truth_for_date==1 else ('down' if the_ground_truth_for_date==-1 else 'sideways')
        nb_forward_pass = 0
        nb_pred_for_down, nb_pred_for_up, nb_pred_for_sideways, nb_pred_for_unknown = 0., 0., 0., 0.
        assert 1 == len([batch_idx for batch_idx, (X, y, abcdef) in enumerate(data_loader_without_data_augmentation)])
        for batch_idx, (X, y, abcdef) in enumerate(data_loader_without_data_augmentation):
            if not real_time_execution:
                assert all(y == the_ground_truth_for_date)
            _logits = meta_model(x=X)
            nb_forward_pass += 1
            # if 99 == _logits:
            #     _tmp_str = f"For {date.strftime('%Y-%m-%d')} [{day_of_week_full}],"
            #     logger.debug(f"  [?] :| {_tmp_str}")
            nb_pred_for_up       += 1 if 1 == _logits else 0
            nb_pred_for_sideways += 1 if 0 == _logits else 0
            nb_pred_for_down     += 1 if -1 == _logits else 0
            nb_pred_for_unknown  += 1 if 99 == _logits else 0

        # Do multiple passes on dataset Test with data augmentation
        data_loader_with_data_augmentation = get_dataloader(df=df, device=_device, _data_augmentation=True, date_to_predict=date, data_interval=_data_interval,
                                                            real_time_execution=real_time_execution, power_of_noise=_power_of_noise_inf,
                                                            frequency_of_noise=_frequency_of_noise_inf, **params)
        for ee in range(0, _nb_iter_test_in_inference):
            for batch_idx, (X, y, abcdef) in enumerate(data_loader_with_data_augmentation):
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
        prediction_value = 1 if prediction_label=='up' else (-1 if prediction_label=='down' else (99 if prediction_label=='unknown' else 0))
        pre_str = "[*] " if _today == date.date() else ""
        results_produced.update({date.strftime('%Y-%m-%d'): {"predictions": predictions, "prediction_label": prediction_label, "prediction_value": prediction_value,
                                                             "prediction_confidence": prediction_confidence, "ground_truth": the_ground_truth_for_date}})
        if not real_time_execution:
            close_value_yesterday = df.loc[yesterday][('Close',  'SPY')]
            tmp_str = ""#f"higher than {close_value_yesterday:.2f}$" if 1 == prediction_value else f"lower than {close_value_yesterday:.2f}$"
            pre_str += f":)  " if the_ground_truth_for_date == prediction_value else (":|  " if prediction_value == 99 else ":(  ")
            if _data_interval == '1d':
                if 99 != prediction_value:
                    logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], the ground truth is {the_ground_truth_for_date} , prediction is {prediction_value}  {tmp_str}")
                else:
                    tmp_str = f"\u2191 or \u2193 than {close_value_yesterday:.2f}$"
                    logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], the ground truth is {the_ground_truth_for_date} , prediction is unstable ({prediction_value}) > {tmp_str}")
            if _data_interval == '1wk':
                assert 0 == (date+pd.Timedelta(days=-4)).day_of_week and 4 == date.day_of_week
                desc_week = f"{(date+pd.Timedelta(days=-4)).strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}"
                if 99 != prediction_value:
                    logger.info(f"{pre_str}For {desc_week} , the ground truth is {the_ground_truth_for_date} , prediction is {prediction_value}  {tmp_str}")
                else:
                    tmp_str = f"\u2191 or \u2193 than {close_value_yesterday:.2f}$"
                    logger.info(f"{pre_str}For {desc_week} , the ground truth is {the_ground_truth_for_date} , prediction is unstable ({prediction_value}) > {tmp_str}")
        else:
            pre_str = "[R] "
            assert 1 == len(params['y_cols'])
            try:
                close_value_yesterday = df.loc[yesterday][('Close',  'SPY')]
            except Exception as ee:
                logger.warning(f"There is no data for yesterday=({yesterday.strftime('%Y-%m-%d')}) , so can't predict {date.strftime('%Y-%m-%d')}")
                continue
            if _data_interval == '1d':
                logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], prediction {prediction_value} ({prediction_label})")
            if _data_interval == '1wk':
                desc_week = f"{(date + pd.Timedelta(days=-4)).strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}"
                logger.info(f"{pre_str}For {desc_week} , prediction is {prediction_value} ({prediction_label})")
    hit, miss = 0, 0
    for a_date, values in results_produced.items():
        if 0==pd.to_datetime(a_date).day_of_week and _skip_monday:
            continue
        if values['ground_truth'] is None:
            continue
        if values["prediction_value"] == values['ground_truth']:
            hit += 1
        else:
            miss +=1
    if hit+miss>0:
        logger.info(f"Accuracy: {hit/(hit+miss)*100:.4}% {'(skipping mondays)' if _skip_monday else ''} (N={hit+miss}, from {start_date.date()} to {end_date.date()})")
    else:
        logger.info(f"Accuracy: not available {'(skipping mondays)' if _skip_monday else ''}")
    return results_produced, hit/(hit+miss) if hit+miss>0 else -1

if __name__ == '__main__':
    freeze_support()

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
            print(f"Overriding config with {key} = {val}")
    config = {k: globals()[k] for k in config_keys}
    config.update({k: globals()[k] for k in globals() if k.startswith("runner__") or k.startswith("trainer__") or k in ['device','seed_offset','stub_dir']})
    tmp = {k: namespace[k] for k in [k for k, v in namespace.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]}
    config.update({k: tmp[k] for k, v in config.items() if k in tmp})
    configuration = dict_to_namespace(config)
    # -----------------------------------------------------------------------------
    # pprint.PrettyPrinter(indent=4).pprint(namespace_to_dict(configuration))

    logger.remove()
    configuration = namespace_to_dict(configuration)
    logger.add(sys.stdout, level=configuration.get("runner__debug_level","INFO"))

    start_runner(configuration)