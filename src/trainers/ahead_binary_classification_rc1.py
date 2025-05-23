import os
import pprint
import random
import itertools
from pathlib import Path
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader
from datasets import TripleIndicesLookAheadBinaryClassificationDataset
from models import LSTMClassification, LSTMMetaUpDownClassification
from utils import string_to_bool, namespace_to_dict, dict_to_namespace, get_stub_dir, get_df_SPY_and_VIX, get_all_checkpoints, calculate_binary_classification_metrics, extract_info_from_filename, generate_indices_basic_style, previous_weekday, next_weekday
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import torch
import numpy as np
from loguru import logger
import sys
import pandas as pd
import json
import math
import datetime


def _get_lr(_it, _warmup_iters, _learning_rate, _min_lr, _lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if _it < _warmup_iters:
        return _learning_rate * _it / _warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if _it > _lr_decay_iters:
        return _min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (_it - _warmup_iters) / (_lr_decay_iters - _warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return _min_lr + coeff * (_learning_rate - _min_lr)


def _get_batch(_batch_size, iterator):
    the_x, the_y, (x_data_norm, _) = next(iterator)
    while len(the_x) < _batch_size:
        x, y, (xn, _) = next(iterator)
        the_x = torch.concatenate([the_x, x])
        the_y = torch.concatenate([the_y, y])
        x_data_norm = torch.concatenate([x_data_norm, xn])
    the_x, the_y, x_data_norm = the_x[:_batch_size, ...], the_y[:_batch_size, ...], x_data_norm[:_batch_size, ...]
    return the_x, the_y, x_data_norm


def load_model(ckpt_path, device):
    logger.debug(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    model_args.update({'device': device})
    model = LSTMClassification(num_input_features=model_args['num_input_features'],
                               hidden_size=model_args['hidden_size'], num_layers=model_args['num_layers'], seq_length=model_args['seq_length'],
                               bidirectional=model_args['bidirectional'], device=model_args['device'], dropout=model_args['dropout'])
    model = model.cuda() if device != 'cpu' else model
    state_dict = checkpoint['model']

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_test_loss']
    logger.debug(f" --> {iter_num=}   {best_val_loss[0]=}")
    model.eval()

    return model


def create_meta_model(up_candidats, down_candidats, device, fetch_new_dataframe=False, df_source=None):
    up__list_of_models, down__list_of_models, df_list, params_list = {}, {}, [], []
    for one_candidats, one_bag in zip([up_candidats, down_candidats], [up__list_of_models, down__list_of_models]):
        for one_candidat in one_candidats:
            train_df = pd.read_pickle(one_candidat["df"])
            df_list.append(train_df)
            with open(one_candidat["meta_information"], 'r') as f:
                results = json.load(f)
            params_list.append(results)
            model = load_model(ckpt_path=one_candidat['model_path'], device=device)
            name = one_candidat['name']
            one_bag.update({name: model})
    meta_model = LSTMMetaUpDownClassification(up_models=up__list_of_models, down_models=down__list_of_models)
    meta_model = meta_model.cuda() if device != 'cpu' else meta_model
    if not all(dfk.index.equals(df_list[0].index) for dfk in df_list[1:]):
        logger.warning(f"{df_list[0].index=}")
        for dfk in df_list[1:]:
            logger.warning(f"{dfk.index=}")
    assert all(ppp['x_cols']==params_list[0]['x_cols'] for ppp in params_list[1:])
    assert all(ppp['x_cols_to_norm'] == params_list[0]['x_cols_to_norm'] for ppp in params_list[1:])
    if not all(ppp['y_cols'] == params_list[0]['y_cols'] for ppp in params_list[1:]):
        logger.warning(f"{df_list[0]['y_cols']=}")
        for dfk in df_list[1:]:
            logger.warning(f"{dfk['y_cols']=}")
    assert all(ppp['x_seq_length'] == params_list[0]['x_seq_length'] for ppp in params_list[1:])
    assert all(ppp['y_seq_length'] == params_list[0]['y_seq_length'] for ppp in params_list[1:])
    assert all(ppp['margin'] == params_list[0]['margin'] for ppp in params_list[1:])
    assert all(ppp['type_margin'] == params_list[0]['type_margin'] for ppp in params_list[1:])
    if fetch_new_dataframe:
        assert df_source is None
        df, df_name = _fetch_dataframe()
        df_list[0] = df
    elif df_source is not None:
        assert not fetch_new_dataframe
        df = df_source.copy()
        df_list[0] = df
    return meta_model, df_list[0], {'x_cols': params_list[0]['x_cols'], 'y_cols': params_list[0]['y_cols'], 'type_margin': params_list[0]['type_margin'],
                                    'x_cols_to_norm': params_list[0]['x_cols_to_norm'], 'margin': params_list[0]['margin'],
                                    'x_seq_length': params_list[0]['x_seq_length'], 'y_seq_length': params_list[0]['y_seq_length']}


def generate_dataloader_for_inference(df, device, _data_augmentation, date_to_predict, real_time_execution, data_interval='1d', **kwargs):
    _x_seq_length = kwargs['x_seq_length']
    _y_seq_length = kwargs['y_seq_length']
    assert 1 == _y_seq_length
    _x_cols = kwargs['x_cols']
    _y_cols = kwargs['y_cols']
    _x_cols_to_norm = kwargs['x_cols_to_norm']
    _power_of_noise = kwargs.get("power_of_noise", 0.01)
    _frequency_of_noise = kwargs.get("frequency_of_noise", 0.25)
    _indices, test_df = generate_indices_basic_style(df=df.copy(), dates=[date_to_predict, date_to_predict], x_seq_length=_x_seq_length,
                                                     y_seq_length=_y_seq_length, just_x_no_y=real_time_execution, data_interval=data_interval)
    assert len(_indices) in [0, 1]
    if 0 == len(_indices):
        return None
    _margin = kwargs['margin']
    _type_of_margin = kwargs['type_margin']
    _dataset = TripleIndicesLookAheadBinaryClassificationDataset(_df=test_df, _feature_cols=_x_cols, _target_col=_y_cols, _device=device, _x_cols_to_norm=_x_cols_to_norm,
                                                                 _indices=_indices, _mode='inference', _data_augmentation=_data_augmentation, _margin=_margin,
                                                                 _power_of_noise=_power_of_noise, _type_of_margin=_type_of_margin,
                                                                 _just_x_no_y=real_time_execution, _frequency_of_noise=_frequency_of_noise, _direction=None)
    _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False)
    return _dataloader


def _fetch_dataframe(interval):
    df, df_name = get_df_SPY_and_VIX(interval=interval)
    return df, df_name


def evaluation(_model, _dataloader, _loss_function):
    _losses1, _accuracy1 = [], []
    for _batch_idx, (_x1, _y1, (_, prediction_date)) in enumerate(_dataloader):
        _y1 = _y1.unsqueeze(1)
        _test_logits1, _ = _model(x=_x1)  # forward pass
        assert _test_logits1.shape == _y1.shape
        _loss1 = _loss_function(_test_logits1, _y1.float())
        _losses1.append(0 if torch.isnan(_loss1) else _loss1.item())
        _accuracy1.append(calculate_binary_classification_metrics(y_true=_y1, y_pred=torch.nn.Sigmoid()(_test_logits1) >= 0.5)['accuracy'])
    return np.mean(_losses1), np.mean([ff.item() for ff in _accuracy1])


def update_running_var(_running__value, _value):
    _running__value = _value if _running__value == -1.0 else 0.9 * _running__value + 0.1 * _value
    return _running__value


def train(configuration):
    ###########################################################################
    # Description
    ###########################################################################
    logger.info("The goal is to make a prediction about the value (higher or lower) based on the preceding P time-unit")

    _seed_offset = configuration.get("seed_offset", 123)
    logger.debug(f"Seed: {_seed_offset}")
    np.random.seed(_seed_offset)
    torch.manual_seed(_seed_offset)
    torch.cuda.manual_seed_all(_seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    device = configuration.get("device", 'cuda')

    _data_augmentation   = string_to_bool(configuration["trainer__data_augmentation"])
    _force_download_data = string_to_bool(configuration["trainer__force_download_data"])
    _direction = configuration["trainer__direction"]
    _tav_dates = configuration["trainer__tav_dates"]
    _mes_dates = configuration["trainer__mes_dates"]
    _skip_training = string_to_bool(configuration.get("trainer__skip_training", False))
    _skip_eval_on_test_set_at_the_end = string_to_bool(configuration.get("trainer__skip_eval_on_test_set_at_the_end", False))
    _x_cols = configuration["trainer__x_cols"]
    _x_cols_to_norm = []
    assert configuration.get("trainer__x_cols_to_norm", None) is None
    _y_cols = configuration["trainer__y_cols"]
    _x_seq_length = int(configuration["trainer__x_seq_length"])
    _y_seq_length = int(configuration["trainer__y_seq_length"])
    _margin = float(configuration["trainer__margin"])
    _shuffle_indices = string_to_bool(configuration["trainer__shuffle_indices"])
    _save_checkpoint = string_to_bool(configuration["trainer__save_checkpoint"])
    _type_margin = configuration["trainer__type_margin"]
    assert _type_margin in ['fixed', 'relative']
    assert _y_seq_length == 1
    _jump_ahead = int(configuration['trainer__jump_ahead'])
    _number_of_timestep_for_validation = int(configuration['trainer__number_of_timestep_for_validation'])
    logger.debug(f"Using {_number_of_timestep_for_validation} time steps for validation")
    run_id = configuration["trainer__run_id"]
    version = configuration["trainer__version"]
    output_dir = os.path.join(configuration.get("stub_dir", get_stub_dir()), f"{run_id}", f"ODABC__{version}")
    os.makedirs(output_dir, exist_ok=True)
    _data_interval = configuration["trainer__data_interval"]
    assert _data_interval in ['1d', '1wk', '1mo']
    logger.add(os.path.join(output_dir, "train.txt"), level='DEBUG')

    _frequency_of_noise  = float(configuration["trainer__frequency_of_noise"])
    _power_of_noise      = float(configuration["trainer__power_of_noise"])

    ###########################################################################
    # Load source data
    ###########################################################################
    df_filename, _df_source = os.path.join(output_dir, f"df.pkl"), configuration["trainer__df_source"]
    if _df_source is None:
        if not os.path.exists(df_filename) or _force_download_data:
            df, df_name = _fetch_dataframe(interval=_data_interval)
            logger.debug(f"Writing {df_filename}...")
            df.to_pickle(df_filename)
        else:
            logger.debug(f"Reading {df_filename}...")
            df = pd.read_pickle(df_filename)
        _df_source = df.copy()  # Keep a fresh copy of the data to be returned later.
    else: # Use the data provided and dump it to disk
        if isinstance(_df_source, str):
            logger.debug(f"Reading {_df_source}...")
            df = pd.read_pickle(_df_source)
        else:
            df = _df_source.copy()
            df.to_pickle(df_filename)
    assert df is not None
    logger.debug(f"Data is ranging from {df.index[0]} to {df.index[-1]} , with interval of {_data_interval}")
    num_input_features = len(df[_x_cols].columns)
    num_output_features = len(df[_y_cols].columns)
    _tmp3 = ', '.join([f"({', '.join(map(str, col))})" for col in df[_x_cols_to_norm].columns])
    _tmp, _tmp2 = ', '.join([f"({', '.join(map(str, col))})" for col in df[_x_cols].columns]), "no normalization" if 0 == len(_x_cols_to_norm) else f"normalization applied to  {_tmp3} "
    logger.debug(f"Xs: {_tmp} , {_tmp2} ")
    logger.debug(f"Ys: {_y_cols}  ")
    ###########################################################################
    # Configuration
    ###########################################################################
    __batch_size   = int(configuration["trainer__batch_size"])
    betas          = (0.9, 0.95)
    _decay_lr       = string_to_bool(configuration["trainer__decay_lr"])
    _eval_interval  = int(configuration["trainer__eval_interval"])
    iter_num       = 0
    __learning_rate  = float(configuration["trainer__learning_rate"])
    _log_interval    = int(configuration["trainer__log_interval"])
    __lr_decay_iters = float(configuration["trainer__lr_decay_iters"])
    _max_iters       = int(configuration["trainer__max_iters"])
    __min_lr         = float(configuration["trainer__min_lr"])
    warmup_iters     = 0
    _weight_decay    = float(configuration["trainer__weight_decay"])

    model_args = {'bidirectional': False,
                  'dropout': 0.5,
                  'hidden_size': 256,
                  'num_layers': 1,
                  'num_output_vars': num_output_features,
                  'num_input_features': num_input_features,
                  'device': device,
                  'seq_length': _x_seq_length,
                  }


    ###########################################################################
    # Data preparation
    ###########################################################################
    assert 1 == _y_seq_length
    logger.debug(f"Using a step ahead of {_jump_ahead}  (0 mean tomorrow, next week, next month, depending on the data interval used)")
    train_indices, train_df = generate_indices_basic_style(df=df.copy(), dates=_tav_dates, x_seq_length=_x_seq_length, y_seq_length=_y_seq_length, jump_ahead=_jump_ahead)
    test_indices, test_df   = generate_indices_basic_style(df=df.copy(), dates=_mes_dates, x_seq_length=_x_seq_length, y_seq_length=_y_seq_length, jump_ahead=_jump_ahead)
    if _shuffle_indices:
        assert _number_of_timestep_for_validation is None
        logger.debug("Shuffling indices...")
        random.shuffle(train_indices)
    if _number_of_timestep_for_validation is not None:
        assert not _shuffle_indices
        split_nm_idx = _number_of_timestep_for_validation
        assert len(train_indices) > split_nm_idx
    else:
        assert False
    train_indices, val_indices, val_df = train_indices[split_nm_idx:], train_indices[:split_nm_idx], train_df.copy()
    assert 0 != len(train_indices) and 0 != len(test_indices), f"{len(train_indices)} and {len(test_indices)}  > take care of trainer__x_seq_length ?"

    logger.debug(f"Training+Validation are ranging from {_tav_dates[0]} to {_tav_dates[1]}  ,  {num_input_features} input features ({_x_seq_length} steps) and {num_output_features} output features ({_y_seq_length} steps)")
    if not _shuffle_indices:
        logger.debug(f"Training is ranging from {train_df.iloc[train_indices[-1][2]:train_indices[-1][3]].index[0].date()} to {train_df.iloc[train_indices[0][2]:train_indices[0][3]].index[0].date()}  ")
        logger.debug(f"Validation is ranging from {val_df.iloc[val_indices[-1][2]:val_indices[-1][3]].index[0].date()} to {val_df.iloc[val_indices[0][2]:val_indices[0][3]].index[0].date()}  ")
    logger.debug(f"MES is ranging from {_mes_dates[0]} to {_mes_dates[1]}")
    for a_bag_of_indices in test_indices:
        dd1 = test_df.iloc[a_bag_of_indices[0]:a_bag_of_indices[1]].index[0].date()
        dd2 = test_df.iloc[a_bag_of_indices[0]:a_bag_of_indices[1]].index[-1].date()
        dd3 = test_df.iloc[a_bag_of_indices[2]:a_bag_of_indices[3]].index[0].date()
        if _data_interval == '1d':
            logger.debug(f"Predicting  {dd3} ({dd3.strftime('%A')})  using [{dd2} ({dd2.strftime('%A')})  >>  {dd1} ({dd1.strftime('%A')})]")
        elif _data_interval == '1w':
            logger.debug(f"Predicting the week of {dd3} using weeks [{dd1}  >>  {dd2}]")

    if _data_augmentation:
        logger.debug(f"Using data augmentation @F={_frequency_of_noise} and @A={_power_of_noise}")
    train_dataset    = TripleIndicesLookAheadBinaryClassificationDataset(_df=train_df, _feature_cols=_x_cols, _target_col=_y_cols, _device=device, _x_cols_to_norm=_x_cols_to_norm,
                                                                         _indices=train_indices, _mode='train', _data_augmentation=_data_augmentation, _margin=_margin,
                                                                         _power_of_noise=_power_of_noise, _frequency_of_noise=_frequency_of_noise, _direction=_direction,
                                                                         _type_of_margin=_type_margin)
    val_dataset = TripleIndicesLookAheadBinaryClassificationDataset(_df=val_df, _feature_cols=_x_cols, _target_col=_y_cols, _device=device, _x_cols_to_norm=_x_cols_to_norm,
                                                                    _indices=val_indices, _mode='val', _data_augmentation=False, _margin=_margin, _direction=_direction,
                                                                    _type_of_margin=_type_margin)
    test_dataset     = TripleIndicesLookAheadBinaryClassificationDataset(_df=test_df, _feature_cols=_x_cols, _target_col=_y_cols, _device=device, _x_cols_to_norm=_x_cols_to_norm,
                                                                         _indices=test_indices, _mode='test', _data_augmentation=False, _margin=_margin, _direction=_direction,
                                                                         _type_of_margin=_type_margin)

    train_dataloader = DataLoader(train_dataset, batch_size=__batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=__batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_dataset,  batch_size=__batch_size, shuffle=False)
    logger.debug(f"{__batch_size=}   TRAIN SIZE:{len(train_indices)}    VAL SIZE:{len(val_indices)}    TEST SIZE:{len(test_indices)}")
    ground_truth_sequence, train__ones_and_zeros, val__ones_and_zeros = [], {}, {}
    for desc, a_dataloader in zip(["train", "val", "test"], [train_dataloader, val_dataloader, test_dataloader]):
        zeros, ones = 0, 0
        for _x, _y, abcdef in a_dataloader:
            assert torch.count_nonzero(torch.where(_y == 0, 1, 0)) == torch.count_nonzero(_y == 0)
            zeros += torch.count_nonzero(torch.where(_y == 0, 1, 0))
            ones  += torch.count_nonzero(_y == 1)
            if desc == 'test':
                ground_truth_sequence.extend(_y.cpu())
        ground_truth_sequence = [uu.item() for uu in ground_truth_sequence]
        logger.debug(f"In {desc} dataset (N={zeros+ones}) --> {(zeros/(zeros+ones))*100:.2f}% 0s  ,  {(ones/(zeros+ones))*100:.2f}% 1s")
        if desc == 'test':
            logger.debug(f"Test sequence: {ground_truth_sequence}")
        if desc == 'train':
            train__ones_and_zeros.update({'ones': (ones/(zeros+ones)).item(), 'zeros': (zeros/(zeros+ones)).item()})
        if desc == 'val':
            val__ones_and_zeros.update({'ones': (ones / (zeros + ones)).item(), 'zeros': (zeros / (zeros + ones)).item()})

    ###########################################################################
    # Model preparation
    ###########################################################################
    pos_weight = None
    if train__ones_and_zeros['ones'] < 0.45:
        pos_weight=configuration["trainer__wanted_pos_weight"]
    model = LSTMClassification(num_input_features=model_args['num_input_features'], pos_weight=pos_weight,
                               hidden_size=model_args['hidden_size'], num_layers=model_args['num_layers'], seq_length=model_args['seq_length'],
                               bidirectional=model_args['bidirectional'], device=model_args['device'], dropout=model_args['dropout'])

    ###########################################################################
    # Train
    ###########################################################################
    logger.debug(f"{_max_iters=}   {_log_interval=}   {_direction=}   {__lr_decay_iters=}   {_eval_interval=}")
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)
    optimizer = model.configure_optimizers(weight_decay=_weight_decay, learning_rate=__learning_rate, betas=betas, device_type=device)
    train_loss, best_val_loss, best_val_accuracy = torch.tensor(999999999), (999999999, 999999999), (0., 0)
    best_test_loss, best_test_accuracy = (999999999, 999999999), (0., 0)
    running__train_losses, running__train_accuracy, running__val_losses, running__val_accuracy, running__test_losses, running__test_accuracy = -1, -1, -1, -1, -1, -1
    train_iterator, old__is_best_val_loss_achieved_file, old__is_best_val_accuracy_achieved_file = itertools.cycle(train_dataloader), None, None
    _x, _y, _x_data_norm = _get_batch(_batch_size=__batch_size, iterator=train_iterator)
    while iter_num < _max_iters and not _skip_training:
        lr = _get_lr(_it=iter_num, _warmup_iters=warmup_iters, _learning_rate=__learning_rate, _min_lr=__min_lr, _lr_decay_iters=__lr_decay_iters) if _decay_lr else __learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % _eval_interval == 0 or 1 == iter_num or iter_num == _max_iters - 1:
            model.eval()
            val_loss,  val_accuracy  = evaluation(_model=model, _dataloader=val_dataloader, _loss_function=loss_function)
            test_loss, test_accuracy = evaluation(_model=model, _dataloader=test_dataloader, _loss_function=loss_function)
            test_accuracy, test_loss = np.mean(test_accuracy), np.mean(test_loss)

            running__val_losses    = update_running_var(_running__value=running__val_losses, _value=val_loss)
            running__val_accuracy  = update_running_var(_running__value=running__val_accuracy, _value=val_accuracy)
            running__test_losses   = update_running_var(_running__value=running__test_losses, _value=test_loss)
            running__test_accuracy = update_running_var(_running__value=running__test_accuracy, _value=test_accuracy)

            is_best_val_loss_achieved      = val_loss     <= best_val_loss[0]
            is_best_val_accuracy_achieved  = val_accuracy >= best_val_accuracy[0]
            best_val_loss      = (val_loss, iter_num, test_accuracy)     if is_best_val_loss_achieved or 0 == iter_num else best_val_loss
            best_val_accuracy  = (val_accuracy, iter_num, test_accuracy) if is_best_val_accuracy_achieved or 0 == iter_num else best_val_accuracy

            is_best_test_loss_achieved     = test_loss     <= best_test_loss[0] and iter_num > 0
            is_best_test_accuracy_achieved = test_accuracy >= best_test_accuracy[0] and iter_num > 0
            best_test_loss     = (test_loss, iter_num) if is_best_test_loss_achieved or 0 == iter_num else best_test_loss
            best_test_accuracy = (test_accuracy, iter_num) if is_best_test_accuracy_achieved or 0 == iter_num else best_test_accuracy

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
                'best_test_loss': best_test_loss,
                'best_test_accuracy': best_test_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'train_loss': train_loss.item(),
            }
            if is_best_val_loss_achieved or is_best_val_accuracy_achieved and 0 != iter_num and _save_checkpoint:
                os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
                if is_best_val_loss_achieved:
                    checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'best__val_loss_{best_val_loss[0]:.8f}__with__val_accuracy_{val_accuracy:.8f}__at_{iter_num}.pt')
                    torch.save(checkpoint, checkpoint_filename)
                    try:
                        os.remove(old__is_best_val_loss_achieved_file)
                    except:
                        pass
                    old__is_best_val_loss_achieved_file = checkpoint_filename

                if is_best_val_accuracy_achieved:
                    checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'best__val_accuracy_{best_val_accuracy[0]:.4f}__with__val_loss_{val_loss:.8f}_at_{iter_num}.pt')
                    torch.save(checkpoint, checkpoint_filename)
                    try:
                        os.remove(old__is_best_val_accuracy_achieved_file)
                    except:
                        pass
                    old__is_best_val_accuracy_achieved_file = checkpoint_filename

            if 0 == iter_num % _log_interval or 1 == iter_num:
                logger.debug(f"i: {iter_num:6d}  lr: {lr:0.3E}  train_acc:({running__train_accuracy:.4f})  "
                             f"val_acc:({running__val_accuracy:.4f}) >> {best_val_accuracy[0]:.2f}@{best_val_accuracy[1]} (wT:{best_val_accuracy[2]:.2f})    "
                             f"test_acc:({running__test_accuracy:.4f}) >> {best_test_accuracy[0]:.2f}@{best_test_accuracy[1]}  "
                             f"")
            model.train()
        train_logits, train_loss = model(x=_x, y=_y)  # forward pass
        train_loss = train_loss * 1000.0  # scale loss to avoid gradient vanishing

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        _x, _y, _x_data_norm = _get_batch(_batch_size=__batch_size, iterator=train_iterator)

        train_loss.backward()  # calculates the loss of the loss function
        optimizer.step()  # improve from loss, i.e. backprop
        optimizer.zero_grad()

        # Compute accuracy
        training_accuracy = calculate_binary_classification_metrics(y_true=_y, y_pred=torch.nn.Sigmoid()(train_logits) >= 0.5)['accuracy']

        running__train_accuracy = training_accuracy if running__train_accuracy == -1.0 else 0.9 * running__train_accuracy + 0.1 * training_accuracy
        running__train_losses = train_loss.item() if running__train_losses == -1.0 else 0.9 * running__train_losses + 0.1 * train_loss.item()

        iter_num += 1
    results = {}
    if not _skip_training:
        if 'trainer__df_source' in configuration:
            configuration['trainer__df_source'] = None
        if 'runner__master_df_source' in configuration:
            configuration['runner__master_df_source'] = None
        if '_today' in configuration:
            configuration['_today'] = None
        results = {'running__train_losses': running__train_losses, 'running__train_accuracy': running__train_accuracy.item(),
                   'running__test_losses': running__test_losses, 'ground_truth_sequence': ground_truth_sequence,
                   'best_val_loss': best_val_loss, 'best_val_accuracy': best_val_accuracy,
                   'best_test_loss': best_test_loss, 'best_test_accuracy': best_test_accuracy, 'type_margin': _type_margin,
                   'output_dir': output_dir,'data_augmentation': _data_augmentation, 'margin': _margin,
                   'configuration': configuration, 'x_cols': _x_cols, 'y_cols': _y_cols, 'x_cols_to_norm': _x_cols_to_norm,
                   'tav_dates': _tav_dates if isinstance(_tav_dates, str) else [f"{str(fxx)}" for fxx in _tav_dates],
                   'mes_dates': _mes_dates if isinstance(_mes_dates, str) else [f"{str(fxx)}" for fxx in _mes_dates],
                   'x_seq_length': _x_seq_length, 'y_seq_length': _y_seq_length}
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        results.update({'df_source': _df_source})  # Dataframe is not serializable
    if 0 == len(get_all_checkpoints(output_dir)):
        logger.warning(f"No checkpoints found under {output_dir}")
    if not _skip_eval_on_test_set_at_the_end:
        # Just dump some results at the screen
        for ckpt_path_filename in get_all_checkpoints(output_dir):
            info = extract_info_from_filename(Path(ckpt_path_filename).name)
            model_for_eval = load_model(ckpt_path_filename, device)
            for _batch_idx, (_x1, _y1, (_, _prediction_date_milliseconds)) in enumerate(test_dataloader):
                _y1 = _y1.unsqueeze(1)
                _test_logits1, _ = model_for_eval(x=_x1)  # forward pass
                assert _test_logits1.shape == _y1.shape
                y_prediction = torch.where(torch.nn.Sigmoid()(_test_logits1) >= 0.5, 1, 0)
                date_objs = [datetime.datetime.fromtimestamp(ppm / 1000).date() for ppm in _prediction_date_milliseconds.squeeze().cpu().numpy()]
                df = pd.DataFrame({'Prediction': y_prediction.squeeze().cpu().numpy(), 'Actual': _y1.squeeze().cpu().numpy(), 'Date': date_objs})
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                logger.debug(f"{ckpt_path_filename}\n{df}")

    return results


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
    config = {k: globals()[k] for k in config_keys}
    tmp = {k: namespace[k] for k in [k for k, v in namespace.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]}
    config.update({k: tmp[k] for k, v in config.items() if k in tmp})
    config.update({k: globals()[k] for k in globals() if k.startswith("trainer__") or k in ['device','seed_offset','stub_dir']})
    configuration = dict_to_namespace(config)
    # -----------------------------------------------------------------------------
    #pprint.PrettyPrinter(indent=4).pprint(namespace_to_dict(configuration))

    logger.remove()
    logger.add(sys.stdout, level=namespace_to_dict(configuration).get("trainer__debug_level", "INFO"))
    train(configuration = namespace_to_dict(configuration))