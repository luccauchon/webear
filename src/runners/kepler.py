import pandas as pd
from trainers.one_day_ahead_binary_classification_rc1 import train as train_model
from trainers.one_day_ahead_binary_classification_rc1 import load_model as load_model
from trainers.one_day_ahead_binary_classification_rc1 import create_meta_model as create_meta_model
from trainers.one_day_ahead_binary_classification_rc1 import generate_dataloader_to_predict as get_dataloader
from datetime import datetime
from loguru import logger
from pathlib import Path
import os
import json
from utils import extract_info_from_filename, previous_weekday
import torch


if __name__ == '__main__':
    ###########################################################################
    # Parameters for campaign
    ###########################################################################
    skip_training_with_already_computed_results = [rf"D:\PyCharmProjects\webear\stubs\2025_03_17__21_27_50"]
    fast_execution_for_debugging                = False

    tav_dates                    = ["2024-01-01", "2025-03-08"]
    mes_dates                    = ["2025-03-09", "2025-03-15"]
    inf_dates                    = ["2025-03-09", "2025-03-21"]
    fetch_new_dataframe          = True  # Use Yahoo! Finance to download data instead of using a dataframe from an experience
    device                       = "cuda"
    test_margin                  = 0
    power_of_noise               = 0.01
    nb_iter_test                 = 500


    ###########################################################################
    # Select a base directory with "run_id"
    ###########################################################################
    run_id = f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
    output_dir = []
    if isinstance(skip_training_with_already_computed_results, list) and 1==len(skip_training_with_already_computed_results) and os.path.exists(skip_training_with_already_computed_results[0]):
        output_dir = skip_training_with_already_computed_results


    ###########################################################################
    # Perform multiple training
    ###########################################################################
    assert isinstance(output_dir, list)
    if 0 == len(output_dir):
        margin = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
        if fast_execution_for_debugging:
            margin = [-3,0]
        for a_margin in margin:
            version = f"M{a_margin}_"
            configuration_for_experience = {"train_margin": a_margin, "test_margin": a_margin, "run_id": run_id, "version": version, "tav_dates": tav_dates, "mes_dates": mes_dates}
            if fast_execution_for_debugging:
                configuration_for_experience.update({'max_iters': 50, 'log_interval': 10})

            # Launch training
            results = train_model(configuration_for_experience)
            logger.debug(results)
            output_dir.append(Path(results['output_dir']).parent)
        output_dir = list(set(output_dir))
        assert 1 == len(output_dir)
    else:
        logger.info(f"Skipping training , using those results: {output_dir[0]}")
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
                tmp_results.update({"df": os.path.join(output_dir, one_experience_directory, "df.pkl")})
                experience__2__results.get(one_experience_directory).append(tmp_results)
    # Scan the results for bests models
    best_accuracy, best_loss = 0., 999999.
    best_accuracy__candidat, best_accuracy__candidats = None, []
    best_lost__candidat, best_lost__candidats = None, []
    for one_experience, v in experience__2__results.items():
        for one_candidat in v:
            if 'test_accuracy' in one_candidat:
                if float(one_candidat['test_accuracy']) > best_accuracy:
                    best_accuracy = float(one_candidat['test_accuracy'])
                    best_accuracy__candidat = one_candidat
                    best_accuracy__candidats = []
                elif float(one_candidat['test_accuracy']) == best_accuracy:
                    best_accuracy__candidats.append(one_candidat)

            if 'test_loss' in one_candidat:
                if float(one_candidat['test_loss']) < best_loss:
                    best_loss = float(one_candidat['test_loss'])
                    best_lost__candidat = one_candidat
                    best_lost__candidats = []
                elif float(one_candidat['test_loss']) == best_loss:
                    best_lost__candidats.append(one_candidat)
    # Switching model, if necessary
    for one_candidat in best_accuracy__candidats:
        assert float(best_accuracy__candidat['test_accuracy']) == float(one_candidat['test_accuracy'])
        if float(one_candidat['with_test_loss']) < float(best_accuracy__candidat['with_test_loss']):
            best_accuracy__candidat = one_candidat
    for one_candidat in best_lost__candidats:
        assert float(best_lost__candidat['test_loss']) == float(one_candidat['test_loss'])
        if float(one_candidat['with_test_accuracy']) > float(best_accuracy__candidat['with_test_accuracy']):
            best_lost__candidat = one_candidat

    ###########################################################################
    # Do inferences
    ###########################################################################
    meta_model, df, params = create_meta_model(candidats=[best_lost__candidat] + [best_accuracy__candidat], fetch_new_dataframe=fetch_new_dataframe, device=device)
    start_date, end_date = pd.to_datetime(inf_dates[0]), pd.to_datetime(inf_dates[1])
    # Iterate over days
    for n in range(int((end_date - start_date).days) + 1):
        date = start_date + pd.Timedelta(n, unit='days')
        day_of_week_full = date.strftime('%A')
        yesterday = previous_weekday(date)
        data_loader_without_data_augmentation, just_x_no_y = get_dataloader(df=df, device=device, data_augmentation=False, mode='inference', date_to_predict=date, test_margin=test_margin, **params)
        if data_loader_without_data_augmentation is None:
            continue
        if not just_x_no_y:
            the_ground_truth_for_date = [y for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_without_data_augmentation)]
            assert 1 == len(the_ground_truth_for_date)
            the_ground_truth_for_date = the_ground_truth_for_date[0].item()
        nb_forward_pass = 0
        nb_pred_for_0, nb_pred_for_1 = 0., 0.
        for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_without_data_augmentation):
            if not just_x_no_y:
                assert just_x_no_y or all(y == the_ground_truth_for_date)
            _logits, _ = meta_model(x=X)
            nb_pred_for_0 += torch.count_nonzero(_logits[_logits < 0.5]).item()
            nb_pred_for_1 += torch.count_nonzero(_logits[_logits >= 0.5]).item()
            nb_forward_pass += 1
        data_loader_with_data_augmentation, just_x_no_y = get_dataloader(df=df, device=device, data_augmentation=True, mode='inference', date_to_predict=date, test_margin=test_margin, power_of_noise=power_of_noise, **params)
        for ee in range(0, nb_iter_test):
            for batch_idx, (X, y, x_data_norm) in enumerate(data_loader_with_data_augmentation):
                if not just_x_no_y:
                    assert all(y == the_ground_truth_for_date)
                _logits, _ = meta_model(x=X)
                nb_pred_for_0 += torch.count_nonzero(_logits[_logits < 0.5]).item()
                nb_pred_for_1 += torch.count_nonzero(_logits[_logits >= 0.5]).item()
                nb_forward_pass += 1
        prediction = 1 if nb_pred_for_1 > nb_pred_for_0 else 0
        if not just_x_no_y:
            confidence  = nb_pred_for_1 / (nb_pred_for_0 + nb_pred_for_1) if 1 == the_ground_truth_for_date else nb_pred_for_0 / (nb_pred_for_0 + nb_pred_for_1)
            close_value_yesterday = df.loc[yesterday][params['y_cols']].values[0]
            tmp_str = f"higher than {close_value_yesterday:.1f}$" if 1 == prediction else f"lower than {close_value_yesterday:.1f}$"
            logger.info(f"For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], the ground truth is {the_ground_truth_for_date} , prediction is {prediction} with {confidence * 100:.2f}% confidence ({tmp_str})")
        else:
            assert 1 == len(params['y_cols'])
            try:
                close_value_yesterday = df.loc[yesterday][params['y_cols']].values[0]
            except Exception as ee:
                logger.warning(f"There is no data for yesterday=({yesterday.strftime('%Y-%m-%d')}) , so can't predict {date.strftime('%Y-%m-%d')}")
                continue
            confidence = nb_pred_for_1 / (nb_pred_for_0 + nb_pred_for_1) if 1 == prediction else nb_pred_for_0 / (nb_pred_for_0 + nb_pred_for_1)
            tmp_str = f"higher than {close_value_yesterday:.1f}$" if 1==prediction else f"lower than {close_value_yesterday:.1f}$"
            logger.info(f"For {date.strftime('%Y-%m-%d')} [{day_of_week_full}], prediction is {prediction} with {confidence * 100:.2f}% confidence > ({tmp_str})")