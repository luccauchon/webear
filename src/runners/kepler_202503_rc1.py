from multiprocessing import Lock, Process, Queue, Value, freeze_support
import pprint
import pandas as pd
from trainers.one_day_ahead_binary_classification_rc1 import train as train_model
from trainers.one_day_ahead_binary_classification_rc1 import create_meta_model as create_meta_model
from trainers.one_day_ahead_binary_classification_rc1 import generate_dataloader_to_predict as get_dataloader
from datetime import datetime
from loguru import logger
from pathlib import Path
import os
import json
from utils import extract_info_from_filename, previous_weekday_with_check, namespace_to_dict, dict_to_namespace, next_weekday_with_check, is_weekday, get_stub_dir
import torch


def scan_results(_selected_margin, _experience__2__results):
    _best_accuracy, _best_loss = 0., 999999.
    _best_accuracy__candidat, _best_accuracy__candidats = None, []
    _best_lost__candidat, _best_lost__candidats = None, []
    for _one_experience, _v in _experience__2__results.items():
        for _one_candidat in _v:
            if _selected_margin != _one_candidat['test_margin']:
                continue
            if 'test_accuracy' in _one_candidat:
                if float(_one_candidat['test_accuracy']) > _best_accuracy:
                    _best_accuracy = float(_one_candidat['test_accuracy'])
                    _best_accuracy__candidat = _one_candidat
                    _best_accuracy__candidats = []
                elif float(_one_candidat['test_accuracy']) == _best_accuracy:
                    _best_accuracy__candidats.append(_one_candidat)
            if 'test_loss' in _one_candidat:
                if float(_one_candidat['test_loss']) < _best_loss:
                    _best_loss = float(_one_candidat['test_loss'])
                    _best_lost__candidat = _one_candidat
                    _best_lost__candidats = []
                elif float(_one_candidat['test_loss']) == _best_loss:
                    _best_lost__candidats.append(_one_candidat)
    # Switching model, if necessary
    for _one_candidat in _best_accuracy__candidats:
        assert float(_best_accuracy__candidat['test_accuracy']) == float(_one_candidat['test_accuracy'])
        if float(_one_candidat['with_test_loss']) < float(_best_accuracy__candidat['with_test_loss']):
            _best_accuracy__candidat = _one_candidat
    for _one_candidat in _best_lost__candidats:
        assert float(_best_lost__candidat['test_loss']) == float(_one_candidat['test_loss'])
        try:
            if float(_one_candidat['with_test_accuracy']) > float(_best_accuracy__candidat['with_test_accuracy']):
                _best_lost__candidat = _one_candidat
        except Exception as eee:
            logger.warning(eee)
            logger.warning(f"\n{_one_candidat=}\n{_best_accuracy__candidat=}")
    return _best_lost__candidat, _best_accuracy__candidat


def start_runner(configuration):
    ###########################################################################
    # Parameters for campaign
    ###########################################################################
    skip_training_with_already_computed_results = configuration.get("skip_training_with_already_computed_results", [])
    fast_execution_for_debugging                = configuration.get("fast_execution_for_debugging", False)

    tav_dates                    = configuration.get("tav_dates", ["2024-01-01", "2025-03-08"])
    mes_dates                    = configuration.get("mes_dates", ["2025-03-09", "2025-03-15"])
    inf_dates                    = configuration.get("inf_dates", ["2025-03-16", "2025-03-29"])

    fetch_new_dataframe          = configuration.get("fetch_new_dataframe", False)  # Use Yahoo! Finance to download data instead of using a dataframe from an experience
    master_df_source             = configuration.get("master_df_source", None)  # Use the specified dataframe instead of using a dataframe from an experience
    device                       = "cuda"
    power_of_noise               = 0.1
    nb_iter_test_in_inference    = configuration.get("nb_iter_test_in_inference", 50)
    _today                       = configuration.get("_today", pd.Timestamp.now().date())
    apply_constrain_on_best_model_selection = configuration.get("apply_constrain_on_best_model_selection", True)
    if fast_execution_for_debugging:
        apply_constrain_on_best_model_selection = False
        nb_iter_test_in_inference               = 5

    ###########################################################################
    # Select a base directory with "run_id"
    ###########################################################################
    run_id = f'RKRC1_{datetime.now().strftime("%d__%H_%M")}'
    output_dir = []
    if isinstance(skip_training_with_already_computed_results, list) and 1==len(skip_training_with_already_computed_results) and os.path.exists(skip_training_with_already_computed_results[0]):
        output_dir = skip_training_with_already_computed_results


    ###########################################################################
    # Perform multiple training
    ###########################################################################
    _available_margin = configuration.get("available_margin", [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2.5, -2.5])  # In order of preference
    assert isinstance(output_dir, list)
    if 0 == len(output_dir):
        df_source = master_df_source.copy() if master_df_source is not None else None
        if fast_execution_for_debugging:
            available_margin = [-3, 0]
        for a_margin in _available_margin:
            version = f"M{a_margin}_"
            configuration_for_experience = {"train_margin": a_margin, "test_margin": a_margin,
                                            "run_id": run_id, "version": version,
                                            "tav_dates": tav_dates, "mes_dates": mes_dates,
                                            "stub_dir": configuration.get("stub_dir", get_stub_dir())}
            if df_source is not None:  # Use the same data for all the experiences
                configuration_for_experience.update({'df_source': df_source})
            if fast_execution_for_debugging:
                configuration_for_experience.update({'max_iters': 50, 'log_interval': 10})
            # Launch training
            results = train_model(configuration_for_experience)
            if df_source is None:
                df_source = results['df_source']
            logger.debug(results)
            output_dir.append(str(Path(results['output_dir']).parent))
        output_dir = list(set(output_dir))
        assert 1 == len(output_dir)
    else:
        logger.info(f"Skipping training , using those results: {output_dir[0]}")
        assert master_df_source is None
    assert os.path.exists(output_dir[0]),f"Missing {output_dir[0]}"
    output_dir = output_dir[0]


    ###########################################################################
    # Select best models
    ###########################################################################
    experience__2__results = {}
    # Populate the results
    for one_experience_directory in os.listdir(output_dir):
        experience__2__results.update({one_experience_directory:[]})
        for filename in os.listdir(os.path.join(output_dir, one_experience_directory, "checkpoints")):
            if filename.endswith(".pt"):
                model_path = os.path.join(output_dir, one_experience_directory, "checkpoints", filename)
                info = extract_info_from_filename(filename)
                assert info is not None
                strz =  f"\nFilename: {filename}\n" + f"Metric 1 Name: {info['metric1_name']}\n" + f"Metric 1 Value: {info['metric1_value']}\n"
                strz += f"Metric 2 Name: {info['metric2_name']}\n"+f"Metric 2 Value: {info['metric2_value']}\n"+f"Epoch: {info['epoch']}\n"+ "-" * 50
                #logger.debug(strz)
                tmp_results = {}
                tmp_results.update({"model_path": model_path})
                if info['metric1_name'] == 'test_accuracy':
                    test_accuracy  = info['metric1_value']
                    with_test_loss = info['metric2_value']
                    tmp_results.update({'test_accuracy': test_accuracy})
                    tmp_results.update({'with_test_loss': with_test_loss})
                if info['metric1_name'] == 'test_loss':
                    test_loss = info['metric1_value']
                    with_test_test_accuracy = info['metric2_value']
                    tmp_results.update({'test_loss': test_loss})
                    tmp_results.update({'with_test_accuracy': with_test_test_accuracy})
                tmp_results.update({"directory": os.path.join(output_dir, one_experience_directory)})
                tmp_results.update({"meta_information": os.path.join(output_dir, one_experience_directory, "results.json")})

                # Add information in tmp_results in order to select the best model
                with open(tmp_results["meta_information"], 'r') as f:
                    _tmp_toto = json.load(f)
                tmp_results.update({'test_margin': _tmp_toto['test_margin']})

                tmp_results.update({"df": os.path.join(output_dir, one_experience_directory, "df.pkl")})
                experience__2__results.get(one_experience_directory).append(tmp_results)

    # Scan the results for bests models
    candidats = {}
    for _sm in _available_margin:
        best_lost, best_accuracy = scan_results(_selected_margin=_sm, _experience__2__results=experience__2__results)
        if apply_constrain_on_best_model_selection:
            if 1 == float(best_lost['with_test_accuracy']) and 1 == float(best_accuracy['test_accuracy']):
                candidats.update({f'best_loss_at_margin_{_sm}': best_lost,  f'best_accuracy_at_margin_{_sm}': best_accuracy})
        else:
            if best_lost is not None:
                candidats.update({f'best_loss_at_margin_{_sm}': best_lost})
            if best_accuracy is not None:
                candidats.update({f'best_accuracy_at_margin_{_sm}': best_accuracy})

    ###########################################################################
    # Do inferences
    ###########################################################################
    meta_model, df, params = create_meta_model(candidats=candidats, fetch_new_dataframe=fetch_new_dataframe, device=device, df_source=master_df_source)
    logger.debug(f"Created meta model with {len(candidats)} models")
    start_date, end_date = pd.to_datetime(inf_dates[0]), pd.to_datetime(inf_dates[1])
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
        data_loader_without_data_augmentation = get_dataloader(df=df, device=device, data_augmentation=False, mode='inference', date_to_predict=date,
                                                               real_time_execution=real_time_execution, **params)
        if data_loader_without_data_augmentation is None:
            continue
        if not real_time_execution:
            the_ground_truth_for_date = [y for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_without_data_augmentation)]
            assert 1 == len(the_ground_truth_for_date)
            the_ground_truth_for_date = the_ground_truth_for_date[0].item()
        nb_forward_pass, unstable_prediction__2__models = 0, {}
        nb_pred_for_0, nb_pred_for_1 = 0., 0.
        assert 1 == len([batch_idx for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_without_data_augmentation)])
        for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_without_data_augmentation):
            if not real_time_execution:
                assert all(y == the_ground_truth_for_date)
            _logits, _ = meta_model(x=X)
            assert _logits.shape[0] == meta_model.get_number_models(), f"We should have one prediction per model"
            nb_pred_for_0 += torch.count_nonzero(_logits[_logits < 0.5]).item()
            nb_pred_for_1 += torch.count_nonzero(_logits[_logits >= 0.5]).item()
            nb_forward_pass += 1
            if nb_pred_for_1 == nb_pred_for_0:
                _tmp_str = f"For {date.strftime('%Y-%m-%d')} [{day_of_week_full}],"
                for yy in range(0, _logits.shape[0]):
                    ppr = 0 if _logits[yy]<0.5 else 1
                    _tmp_str += f" [model <<{meta_model.get_corresponding_model_names()[yy]}>> predicted {ppr}] "
                    unstable_prediction__2__models.update({ppr: meta_model.get_corresponding_model_names()[yy]})
                logger.debug(f"  [?] :| {_tmp_str}")

        # Do multiple passes on dataset Test with data augmentation
        data_loader_with_data_augmentation = get_dataloader(df=df, device=device, data_augmentation=True, mode='inference', date_to_predict=date,
                                                            real_time_execution=real_time_execution, power_of_noise=power_of_noise, **params)
        for ee in range(0, nb_iter_test_in_inference):
            for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_with_data_augmentation):
                if not real_time_execution:
                    assert all(y == the_ground_truth_for_date)
                _logits, _ = meta_model(x=X)
                nb_pred_for_0 += torch.count_nonzero(_logits[_logits < 0.5]).item()
                nb_pred_for_1 += torch.count_nonzero(_logits[_logits >= 0.5]).item()
                nb_forward_pass += 1
        prediction = 1 if nb_pred_for_1 > nb_pred_for_0 else 0
        prediction = prediction if nb_pred_for_1 != nb_pred_for_0 else -1
        pre_str = "[*] " if _today == date.date() else ""
        if not real_time_execution:
            confidence  = nb_pred_for_1 / (nb_pred_for_0 + nb_pred_for_1) if 1 == the_ground_truth_for_date else nb_pred_for_0 / (nb_pred_for_0 + nb_pred_for_1)
            close_value_yesterday = df.loc[yesterday][params['y_cols']].values[0]
            tmp_str = f"higher than {close_value_yesterday:.2f}$" if 1 == prediction else f"lower than {close_value_yesterday:.2f}$"
            pre_str += f":) " if the_ground_truth_for_date == prediction else ":( "
            if -1 != prediction:
                logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], the ground truth is {the_ground_truth_for_date} , prediction is {prediction} with {confidence * 100:.2f}% confidence ({tmp_str})")
            else:
                tmp_str = f"\u2191 or \u2193 than {close_value_yesterday:.2f}$"
                logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], the ground truth is {the_ground_truth_for_date} , prediction is unstable because of {confidence * 100:.2f}% confidence > ({tmp_str})")
                logger.debug(f"  {''.join([' ' for jj in range(len(pre_str))])} The model <<{unstable_prediction__2__models[the_ground_truth_for_date]}>> made the good prediction")
            results_produced.update({f"{date.strftime('%Y-%m-%d')}": {"prediction": prediction, "confidence": confidence, "ground_truth": the_ground_truth_for_date}})
        else:
            assert 1 == len(params['y_cols'])
            try:
                close_value_yesterday = df.loc[yesterday][params['y_cols']].values[0]
            except Exception as ee:
                logger.warning(f"There is no data for yesterday=({yesterday.strftime('%Y-%m-%d')}) , so can't predict {date.strftime('%Y-%m-%d')}")
                continue
            confidence = nb_pred_for_1 / (nb_pred_for_0 + nb_pred_for_1) if 1 == prediction else nb_pred_for_0 / (nb_pred_for_0 + nb_pred_for_1)
            if -1 != prediction:
                tmp_str = f"higher than {close_value_yesterday:.2f}$" if 1 == prediction else f"lower than {close_value_yesterday:.2f}$"
                logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], prediction is {prediction} with {confidence * 100:.2f}% confidence > ({tmp_str})")
            else:
                tmp_str = f"\u2191 or \u2193 than {close_value_yesterday:.2f}$"
                logger.info(f"{pre_str}For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], prediction is unstable because of {confidence * 100:.2f}% confidence > ({tmp_str})")
    return results_produced


if __name__ == '__main__':
    freeze_support()

    logger.info(f"\n{'*' * 80}\nUsing One Day Ahead Binary Classifier RC1 , train multiple models and evaluate them on inf dates\n{'*' * 80}")

    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]
    namespace = {}
    exec(open('configurator.py').read(), namespace)  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}
    tmp = {k: namespace[k] for k in [k for k, v in namespace.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]}
    config.update({k: tmp[k] for k, v in config.items() if k in tmp})
    configuration = dict_to_namespace(config)
    # -----------------------------------------------------------------------------
    pprint.PrettyPrinter(indent=4).pprint(namespace_to_dict(configuration))

    configuration = namespace_to_dict(configuration)
    configuration.update({"fetch_new_dataframe": True,
                          "skip_training_with_already_computed_results": [rf"D:\PyCharmProjects\webear\stubs\2025_03_21__14_06_39"],
                          "nb_iter_test_in_inference":1})
    start_runner(configuration)