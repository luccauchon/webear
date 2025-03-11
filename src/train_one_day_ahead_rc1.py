import os
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import itertools
import pprint
import math
from models import LSTM6v2
from utils import all_dicts_equal, namespace_to_dict, dict_to_namespace, get_df_SPY_and_VIX, get_stub_dir, generate_indices_naked_monday_style
from datasets import TripleIndicesDataset
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import torch
import numpy as np
from loguru import logger
import sys

###############################################################################
# Load default configuration
from config.default.train_one_day_ahead_rc1 import *
###############################################################################


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


def main(cc):
    np.random.seed(cc.toda__seed_offset)
    torch.manual_seed(cc.toda__seed_offset)
    torch.cuda.manual_seed_all(cc.toda__seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    output_dir = os.path.join(get_stub_dir(), f"trainer_one_day_ahead__{toda__version}")
    os.makedirs(output_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=cc.toda__debug_level)
    logger.add(os.path.join(output_dir, "train.txt"), level='DEBUG')

    ###########################################################################
    # Load source data
    ###########################################################################
    df_filename = os.path.join(output_dir, "df.pkl")
    if not os.path.exists(df_filename):
        df, df_name = get_df_SPY_and_VIX()
        df.to_pickle(df_filename)
    else:
        logger.debug(f"Reading {df_filename}...")
        df = pd.read_pickle(df_filename)
    assert df is not None

    ###########################################################################
    # Configuration
    ###########################################################################
    feature_cols         = ['Close', 'High', 'Low', 'Open', 'Volume', 'Close_direction'] + ['day_of_week']  # For SPY and VIX
    target_col           = [('Close', 'SPY')]
    x_cols_to_norm       = ['Close', 'High', 'Low', 'Open', 'Volume']
    y_cols_to_norm       = [('Close', 'SPY')]

    num_input_features   = 2 * len(feature_cols) + -1 + -1  # For SPY and VIX + -1 for day_of_week and -1 No volume for VIX
    num_output_vars      = len(target_col)
    tav_dates            = ["2019-01-01", "2024-08-31"]
    mes_dates            = ["2024-08-01", "2099-12-31"]
    x_seq_length         = 15
    y_seq_length         = 3

    batch_size           = 256
    betas                = (0.9, 0.95)
    decay_lr             = True
    device               = "cuda"
    eval_interval        = 1000
    iter_num             = 0
    learning_rate        = 1e-3
    log_interval         = 10000
    lr_decay_iters       = 999999
    max_iters            = 999999
    min_lr               = 1e-5
    warmup_iters         = 0
    weight_decay         = 0.1
    model_args = {'bidirectional': False,
                  'dropout': 0.5,
                  'hidden_size': 256,
                  'num_layers': 1,
                  'num_output_vars': num_output_vars,
                  'num_input_features': num_input_features,
                  't_length_output_vars': -1,
                  'device': device,
                  'activation_minmax': (-2, 2),
                  'seq_length': x_seq_length,
                  }

    ###########################################################################
    # Data preparation
    ###########################################################################
    train_indices, train_df = generate_indices_naked_monday_style(df=df.loc[tav_dates[0]:tav_dates[1]].copy(), seq_length=x_seq_length)
    test_indices,  test_df  = generate_indices_naked_monday_style(df=df.loc[mes_dates[0]:mes_dates[1]].copy(), seq_length=x_seq_length)
    assert y_seq_length == 3
    model_args.update({'t_length_output_vars': y_seq_length})
    train_dataset   = TripleIndicesDataset(_df=train_df, _indices=train_indices, _feature_cols=feature_cols, _target_col=target_col, _device=device,
                                           _x_seq_length=x_seq_length, _y_seq_length=y_seq_length, _x_cols_to_norm=x_cols_to_norm, _y_cols_to_norm=y_cols_to_norm)
    test_dataset = TripleIndicesDataset(_df=test_df, _indices=test_indices, _feature_cols=feature_cols, _target_col=target_col, _device=device,
                                         _x_seq_length=x_seq_length, _y_seq_length=y_seq_length, _x_cols_to_norm=x_cols_to_norm, _y_cols_to_norm=y_cols_to_norm)
    train_dataloader        = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader         = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ###########################################################################
    # Model preparation
    ###########################################################################
    model = LSTM6v2(num_output_vars=model_args['num_output_vars'], num_input_features=model_args['num_input_features'], t_length_output_vars=model_args['t_length_output_vars'],
                    hidden_size=model_args['hidden_size'], num_layers=model_args['num_layers'], seq_length=model_args['seq_length'],
                    bidirectional=model_args['bidirectional'], device=model_args['device'], activation_minmax=model_args['activation_minmax'], dropout=model_args['dropout'])

    ###########################################################################
    # Train
    ###########################################################################
    logger.info(f"{len(train_indices)=}   {len(test_indices)=}   {len(train_df.columns)=}")
    loss_function = torch.nn.MSELoss(reduction='mean').to(device)  # mean-squared error for regression
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas, device_type=device)
    train_loss, best_test_loss = torch.tensor(999999999), (999999999, 999999999, '0/0/0/0', None)
    running_train_losses, running_test_losses, running_train_precision = -1, -1, -1
    iterator = itertools.cycle(train_dataloader)
    the_X, the_y, x_data_norm, y_data_norm = next(iterator)
    checkpoint_filename = os.path.join('stubs', 'models', f'best_test_loss_{best_test_loss[0]:.8f}_at_{iter_num}.pt')
    os.makedirs(Path(checkpoint_filename).parent, exist_ok=True)
    while iter_num < max_iters:
        lr = _get_lr(_it=iter_num, _warmup_iters=warmup_iters, _learning_rate=learning_rate, _min_lr=min_lr, _lr_decay_iters=lr_decay_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_logits, train_loss = model(x=the_X, y=the_y)  # forward pass
        train_loss = train_loss * 100.0  # scale loss to avoid gradient vanishing
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        the_X, the_y, x_data_norm, y_data_norm = next(iterator)

        train_loss.backward()  # calculates the loss of the loss function
        optimizer.step()  # improve from loss, i.e backprop
        optimizer.zero_grad()

        # Compute the precision @0.5
        assert train_logits.shape == the_y.shape
        prediction   = train_logits * y_data_norm.unsqueeze(-2)
        ground_truth = the_y        * y_data_norm.unsqueeze(-2)
        numerator = torch.count_nonzero(torch.abs(prediction - ground_truth) < 0.5)
        denominator = np.prod(train_logits.shape)
        training_precision = numerator / denominator
        running_train_precision = training_precision if running_train_precision == -1.0 else 0.99 * running_train_precision + 0.01 * training_precision

        running_train_losses = train_loss.item() if running_train_losses == -1.0 else 0.99 * running_train_losses + 0.01 * train_loss.item()
        if 0 == iter_num % log_interval:
            logger.info(f"iter: {iter_num:6d}   lr: {lr:0.3E}   train loss: ({running_train_losses:.4f})   precision: ({running_train_precision:.4f})")

        iter_num += 1


if __name__ == '__main__':
    freeze_support()

    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]
    namespace = {}
    exec(open('configurator.py').read(), namespace)  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}
    tmp = {k: namespace[k] for k in [k for k, v in namespace.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]}
    config.update({k: tmp[k] for k, v in config.items() if k in tmp})
    cc = dict_to_namespace(config)
    # -----------------------------------------------------------------------------
    pprint.PrettyPrinter(indent=4).pprint(namespace_to_dict(cc))

    main(cc)