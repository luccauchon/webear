import os
import pprint
from tqdm import tqdm
import itertools
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import TripleIndicesLookAheadClassificationDataset
from models import LSTMClassification, LSTMMetaClassification
from utils import all_dicts_equal, namespace_to_dict, dict_to_namespace, get_stub_dir, get_df_SPY_and_VIX, generate_indices_with_cutoff_day, calculate_classification_metrics, generate_indices_with_multiple_cutoff_day, generate_indices_basic_style
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import torch
import numpy as np
from loguru import logger
import sys
import pandas as pd
import json
import math


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
    the_x, the_y, x_data_norm = next(iterator)
    while len(the_x) < _batch_size:
        x, y, xn = next(iterator)
        the_x = torch.concatenate([the_x, x])
        the_y = torch.concatenate([the_y, y])
        x_data_norm = torch.concatenate([x_data_norm, xn])
    the_x, the_y, x_data_norm = the_x[:_batch_size, ...], the_y[:_batch_size, ...], x_data_norm[:_batch_size, ...]
    return the_x, the_y, x_data_norm


def load_model(ckpt_path, device, df, **kwargs):
    logger.info(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    model = LSTMClassification(num_input_features=model_args['num_input_features'],
                               hidden_size=model_args['hidden_size'], num_layers=model_args['num_layers'], seq_length=model_args['seq_length'],
                               bidirectional=model_args['bidirectional'], device=model_args['device'], dropout=model_args['dropout'])
    model = model.cuda() if device != 'cpu' else model
    state_dict = checkpoint['model']

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_test_loss']
    logger.info(f" --> {iter_num=}   {best_val_loss[0]=}")
    model.eval()

    return model


def create_meta_model(candidats, device, fetch_new_dataframe=False):
    list_of_models, df_list, params_list = [], [], []
    for one_candidat in candidats:
        train_df = pd.read_pickle(one_candidat["df"])
        df_list.append(train_df)
        with open(one_candidat["meta_information"], 'r') as f:
            results = json.load(f)
        params_list.append(results)
        model = load_model(ckpt_path=one_candidat['model_path'], device=device, df=train_df, **results)
        list_of_models.append(model)
    meta_model = LSTMMetaClassification(models=list_of_models)
    meta_model = meta_model.cuda() if device != 'cpu' else meta_model
    assert all(df.equals(df_list[0]) for df in df_list[1:])
    assert all(ppp['x_cols']==params_list[0]['x_cols'] for ppp in params_list[1:])
    assert all(ppp['x_cols_to_norm'] == params_list[0]['x_cols_to_norm'] for ppp in params_list[1:])
    assert all(ppp['y_cols'] == params_list[0]['y_cols'] for ppp in params_list[1:])
    assert all(ppp['x_seq_length'] == params_list[0]['x_seq_length'] for ppp in params_list[1:])
    assert all(ppp['y_seq_length'] == params_list[0]['y_seq_length'] for ppp in params_list[1:])
    if fetch_new_dataframe:
        df, df_name = _fetch_dataframe()
        df_list[0] = df
    return meta_model, df_list[0], {'x_cols': params_list[0]['x_cols'], 'y_cols': params_list[0]['y_cols'],
                                    'x_cols_to_norm': params_list[0]['x_cols_to_norm'],
                                    'x_seq_length': params_list[0]['x_seq_length'], 'y_seq_length': params_list[0]['y_seq_length']}


def generate_dataloader_to_predict(df, device, data_augmentation, date_to_predict, test_margin, mode, **kwargs):
    x_seq_length = kwargs['x_seq_length']
    y_seq_length = kwargs['y_seq_length']
    assert 1 == y_seq_length
    x_cols = kwargs['x_cols']
    y_cols = kwargs['y_cols']
    x_cols_to_norm = kwargs['x_cols_to_norm']

    _indices, test_df = generate_indices_basic_style(df=df.copy(), dates=[date_to_predict, date_to_predict], x_seq_length=x_seq_length, y_seq_length=y_seq_length)
    assert len(_indices) in [0, 1]
    if 0 == len(_indices):
        return None
    _dataset = TripleIndicesLookAheadClassificationDataset(_df=test_df, _feature_cols=x_cols, _target_col=y_cols, _device=device, _x_cols_to_norm=x_cols_to_norm,
                                                           _indices=_indices, _mode=mode, _data_augmentation=data_augmentation, _margin=test_margin)
    _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False)
    return _dataloader


def _fetch_dataframe():
    df, df_name = get_df_SPY_and_VIX()
    return df, df_name


def train(configuration):
    seed_offset = configuration.get("seed_offset", 123)
    np.random.seed(seed_offset)
    torch.manual_seed(seed_offset)
    torch.cuda.manual_seed_all(seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    debug_level = "DEBUG"
    device = 'cuda'

    data_augmentation   = configuration.get("data_augmentation", True)
    force_download_data = False

    tav_dates = configuration.get("tav_dates", ["2018-01-01", "2025-03-08"])
    mes_dates = configuration.get("mes_dates", ["2025-03-09", "2025-03-15"])

    x_cols = ['Close', 'High', 'Low', 'Open'] + ['Volume'] + ['day_of_week']  # For SPY and VIX
    x_cols_to_norm = ['Close', 'Volume']
    y_cols = [('Close', 'SPY')]
    x_seq_length = configuration.get("x_seq_length", 10)
    y_seq_length = 1
    train_margin = configuration.get("train_margin", 2.)
    test_margin  = configuration.get("test_margin", 2.)
    assert y_seq_length == 1
    # cutoff_days = [1,2,3]

    run_id = configuration.get("run_id", 123)
    version = configuration.get("version", "rc1")
    output_dir = os.path.join(get_stub_dir(), f"{run_id}", f"trainer__one_day_ahead_binary_classification__{version}")
    os.makedirs(output_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=debug_level)
    logger.add(os.path.join(output_dir, "train.txt"), level='DEBUG')

    ###########################################################################
    # Description
    ###########################################################################
    logger.info("The goal is to predict the close value of next day (higher or lower) based on the preceding P days")


    ###########################################################################
    # Load source data
    ###########################################################################
    df_filename = os.path.join(output_dir, "df.pkl")
    if not os.path.exists(df_filename) or force_download_data:
        df, df_name = _fetch_dataframe()
        logger.info(f"Writing {df_filename}...")
        df.to_pickle(df_filename)
    else:
        logger.debug(f"Reading {df_filename}...")
        df = pd.read_pickle(df_filename)
    assert df is not None
    num_input_features = len(df[x_cols].columns)
    num_output_features = len(df[y_cols].columns)
    logger.info(f"Data is ranging from {df.index[0]} to {df.index[-1]}")
    logger.info(f"Training is ranging from {tav_dates[0]} to {tav_dates[1]}  ,  {num_input_features} input features ({x_seq_length} steps) and {num_output_features} output features ({y_seq_length} steps)")
    logger.info(f"MES is ranging from {mes_dates[0]} to {mes_dates[1]}")

    ###########################################################################
    # Configuration
    ###########################################################################
    batch_size     = 1024
    betas          = (0.9, 0.95)
    decay_lr       = True
    eval_interval  = 10
    iter_num       = 0
    learning_rate  = 1e-3
    log_interval   = configuration.get("log_interval", 5000)
    lr_decay_iters = configuration.get("lr_decay_iters", 50000)
    max_iters      = configuration.get("max_iters", 50000)
    min_lr         = 1e-6
    stats_check    = True
    warmup_iters   = 0
    weight_decay   = 0.1

    model_args = {'bidirectional': False,
                  'dropout': 0.5,
                  'hidden_size': 256,
                  'num_layers': 1,
                  'num_output_vars': num_output_features,
                  'num_input_features': num_input_features,
                  'device': device,
                  'activation_minmax': (-2, 2),
                  'seq_length': x_seq_length,
                  }


    ###########################################################################
    # Data preparation
    ###########################################################################
    assert 1 == y_seq_length
    logger.debug(f"Generating indices...")
    train_indices, train_df = generate_indices_basic_style(df=df.copy(), dates=tav_dates, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
    test_indices, test_df   = generate_indices_basic_style(df=df.copy(), dates=mes_dates, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
    #train_indices, train_df = generate_indices_with_multiple_cutoff_day(cutoff_days=cutoff_days, _df=df.copy(), _dates=tav_dates, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
    #test_indices,  test_df  = generate_indices_with_multiple_cutoff_day(cutoff_days=cutoff_days, _df=df.copy(), _dates=mes_dates, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
    assert 0 != len(train_indices) and 0 != len(test_indices)
    train_dataset    = TripleIndicesLookAheadClassificationDataset(_df=train_df, _feature_cols=x_cols, _target_col=y_cols, _device=device, _x_cols_to_norm=x_cols_to_norm,
                                                                   _indices=train_indices, _mode='train', _data_augmentation=data_augmentation, _margin=train_margin)
    test_dataset     = TripleIndicesLookAheadClassificationDataset(_df=test_df, _feature_cols=x_cols, _target_col=y_cols, _device=device, _x_cols_to_norm=x_cols_to_norm,
                                                                   _indices=test_indices, _mode='test', _data_augmentation=data_augmentation, _margin=test_margin)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    ground_truth_sequence = []
    if stats_check:
        for desc, a_dataloader in zip(["train", "test"], [train_dataloader, test_dataloader]):
            zeros, ones = 0, 0
            for _x, _y, _x_data_norm in a_dataloader:
                zeros += torch.count_nonzero(_y == 0)
                ones  += torch.count_nonzero(_y == 1)
                if desc == 'test':
                    ground_truth_sequence.extend(_y.cpu())
            ground_truth_sequence = [uu.item() for uu in ground_truth_sequence]
            logger.info(f"In {desc} dataset --> {(zeros/(zeros+ones))*100:.2f}% 0s  ,  {(ones/(zeros+ones))*100:.2f}% 1s")
            if desc == 'test':
                logger.info(f"Test sequence: {ground_truth_sequence}")

    ###########################################################################
    # Model preparation
    ###########################################################################
    model = LSTMClassification(num_input_features=model_args['num_input_features'],
                               hidden_size=model_args['hidden_size'], num_layers=model_args['num_layers'], seq_length=model_args['seq_length'],
                               bidirectional=model_args['bidirectional'], device=model_args['device'], dropout=model_args['dropout'])

    ###########################################################################
    # Train
    ###########################################################################
    logger.info(f"{batch_size=}   TRAIN SIZE:{len(train_indices)}    TEST SIZE:{len(test_indices)}")
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)  # mean-squared error for regression
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas, device_type=device)
    train_loss, best_test_loss, best_test_accuracy = torch.tensor(999999999), (999999999, 999999999), (0., 0)
    running__train_losses, running__test_losses, running__train_accuracy = -1, -1, -1
    train_iterator = itertools.cycle(train_dataloader)
    _x, _y, _x_data_norm = _get_batch(_batch_size=batch_size, iterator=train_iterator)
    while iter_num < max_iters:
        lr = _get_lr(_it=iter_num, _warmup_iters=warmup_iters, _learning_rate=learning_rate, _min_lr=min_lr, _lr_decay_iters=lr_decay_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 or 1 == iter_num or iter_num == max_iters - 1:
            model.eval()
            test_loss, test_accuracy = [], []
            for batch_idx, (x_test, y_test, _) in enumerate(test_dataloader):
                y_test = y_test.unsqueeze(1)
                test_logits, _ = model(x=x_test)  # forward pass
                assert test_logits.shape == y_test.shape
                _loss = loss_function(test_logits, y_test.float())
                test_loss.append(0 if torch.isnan(_loss) else _loss.item())
                test_accuracy.append(calculate_classification_metrics(y_true=y_test, y_pred=test_logits > 0.5)['accuracy'])
            test_accuracy, test_loss = np.mean(test_accuracy), np.mean(test_loss)
            running__test_losses = test_loss if running__test_losses == -1.0 else 0.9 * running__test_losses + 0.1 * test_loss
            is_best_test_loss_achieved     = test_loss < best_test_loss[0]
            is_best_test_accuracy_achieved = test_accuracy > best_test_accuracy[0]
            best_test_loss     = (test_loss, iter_num)     if is_best_test_loss_achieved     or 0 == iter_num else best_test_loss
            best_test_accuracy = (test_accuracy, iter_num) if is_best_test_accuracy_achieved or 0 == iter_num else best_test_accuracy

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_test_loss': best_test_loss,
                'best_test_accuracy': best_test_accuracy,
                'train_loss': train_loss.item(),
            }
            if is_best_test_loss_achieved or is_best_test_accuracy_achieved and 0 != iter_num:
                os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
                if is_best_test_loss_achieved:
                    checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'best__test_loss_{best_test_loss[0]:.8f}__with__test_accuracy_{test_accuracy:.8f}__at_{iter_num}.pt')
                    torch.save(checkpoint, checkpoint_filename)
                if is_best_test_accuracy_achieved:
                    checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'best__test_accuracy_{best_test_accuracy[0]:.4f}__with__test_loss_{test_loss:.8f}_at_{iter_num}.pt')
                    torch.save(checkpoint, checkpoint_filename)


            if 0 == iter_num % log_interval or 1 == iter_num:
                logger.info(f"iter: {iter_num:6d}   lr: {lr:0.3E}   train:({running__train_losses:.4f})/({running__train_accuracy:.4f})   "
                            f"test:({running__test_losses:.4f})/{test_accuracy:.4f} "
                            f">> {best_test_loss[0]:.4f}@{best_test_loss[1]} , {best_test_accuracy[0]:.2f}@{best_test_accuracy[1]}")
            model.train()

        train_logits, train_loss = model(x=_x, y=_y)  # forward pass
        train_loss = train_loss * 100.0  # scale loss to avoid gradient vanishing

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        _x, _y, _x_data_norm = _get_batch(_batch_size=batch_size, iterator=train_iterator)

        train_loss.backward()  # calculates the loss of the loss function
        optimizer.step()  # improve from loss, i.e backprop
        optimizer.zero_grad()

        # Compute accuracy
        training_accuracy = calculate_classification_metrics(y_true=_y, y_pred=train_logits>0.5)['accuracy']

        running__train_accuracy = training_accuracy if running__train_accuracy == -1.0 else 0.9 * running__train_accuracy + 0.1 * training_accuracy
        running__train_losses = train_loss.item() if running__train_losses == -1.0 else 0.9 * running__train_losses + 0.1 * train_loss.item()

        iter_num += 1
    results = {'running__train_losses': running__train_losses, 'running__train_accuracy': running__train_accuracy.item(),
               'running__test_losses': running__test_losses, 'ground_truth_sequence': ground_truth_sequence,
               'output_dir': output_dir,'data_augmentation': data_augmentation, 'test_margin': test_margin,
               'configuration': configuration, 'x_cols': x_cols, 'y_cols': y_cols, 'x_cols_to_norm': x_cols_to_norm,
               'tav_dates': tav_dates, 'mes_dates': mes_dates, 'x_seq_length': x_seq_length, 'y_seq_length': y_seq_length}
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    return results


if __name__ == '__main__':
    freeze_support()

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
    configuration.update({'max_iters': 500, 'log_interval': 10})
    train(configuration)
