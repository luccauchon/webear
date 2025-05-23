import os
import pprint
from tqdm import tqdm
import itertools
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import TripleIndicesRegressionDataset
from models import LSTMRegression
from utils import all_dicts_equal, namespace_to_dict, dict_to_namespace, get_stub_dir, get_df_SPY_and_VIX, generate_indices_basic_style
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import torch
import numpy as np
from loguru import logger
import sys
import pandas as pd
import math

###############################################################################
# Load default configuration
from config.default.train_basic_regression_rc1 import *
###############################################################################


def _get_batch(_batch_size, iterator):
    the_x, the_y, x_data_norm, y_data_norm = next(iterator)
    while len(the_x) < _batch_size:
        x, y, xn, yn = next(iterator)
        the_x = torch.concatenate([the_x, x])
        the_y = torch.concatenate([the_y, y])
        x_data_norm = torch.concatenate([x_data_norm, xn])
        y_data_norm = torch.concatenate([y_data_norm, yn])
    the_x, the_y, x_data_norm, y_data_norm = the_x[:_batch_size, ...], the_y[:_batch_size, ...], x_data_norm[:_batch_size, ...], y_data_norm[:_batch_size, ...]
    return the_x, the_y, x_data_norm, y_data_norm


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

    output_dir = os.path.join(get_stub_dir(), f"trainer_basic_regression__{cc.toda__version}__run_{cc.toda__run_id}")
    os.makedirs(output_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=cc.toda__debug_level)
    logger.add(os.path.join(output_dir, "train.txt"), level='DEBUG')

    ###########################################################################
    # Description
    ###########################################################################
    logger.info("The goal is to predict the SPY close value of the next N days based on the preceding P days")


    ###########################################################################
    # Load source data
    ###########################################################################
    df_filename = os.path.join(output_dir, "df.pkl")
    if not os.path.exists(df_filename) or cc.toda__force_download_data:
        df, df_name = get_df_SPY_and_VIX()
        logger.info(f"Writing {df_filename}...")
        df.to_pickle(df_filename)
    else:
        logger.debug(f"Reading {df_filename}...")
        df = pd.read_pickle(df_filename)
    assert df is not None
    num_input_features = len(df[cc.toda__x_cols].columns)
    num_output_features = len(df[cc.toda__y_cols].columns)
    logger.info(f"Data is ranging from {df.index[0]} to {df.index[-1]}  ,  {num_input_features} input features ({cc.toda__x_seq_length} steps) and {num_output_features} output features ({cc.toda__y_seq_length} steps)")

    ###########################################################################
    # Configuration
    ###########################################################################
    batch_size = 256
    betas = (0.9, 0.95)
    decay_lr = True
    eval_interval  = 10
    iter_num = 0
    learning_rate  = 1e-3
    log_interval   = 10000
    assert log_interval >= eval_interval
    lr_decay_iters = 999999
    max_iters      = 999999
    min_lr         = 1e-6
    sanity_check = False
    warmup_iters = 0
    weight_decay = 0.2
    model_args = {'bidirectional': False,
                  'dropout': 0.2,
                  'hidden_size': 256,
                  'num_layers': 1,
                  'num_output_vars': num_output_features,
                  'num_input_features': num_input_features,
                  't_length_output_vars': cc.toda__y_seq_length,
                  'device': cc.toda__device,
                  'activation_minmax': (-2, 2),
                  'seq_length': cc.toda__x_seq_length,
                  }


    ###########################################################################
    # Data preparation
    ###########################################################################
    train_indices, train_df = generate_indices_basic_style(df=df.loc[cc.toda__tav_dates[0]:cc.toda__tav_dates[1]].copy(), x_seq_length=cc.toda__x_seq_length, y_seq_length=cc.toda__y_seq_length)
    test_indices, test_df   = generate_indices_basic_style(df=df.loc[cc.toda__mes_dates[0]:cc.toda__mes_dates[1]].copy(), x_seq_length=cc.toda__x_seq_length, y_seq_length=cc.toda__y_seq_length)
    train_dataset = TripleIndicesRegressionDataset(_df=train_df, _indices=train_indices, _feature_cols=cc.toda__x_cols, _target_col=cc.toda__y_cols, _device=cc.toda__device, _add_noise=cc.toda__data_augmentation,
                                                   _x_seq_length=cc.toda__x_seq_length, _y_seq_length=cc.toda__y_seq_length, _x_cols_to_norm=cc.toda__x_cols_to_norm, _y_cols_to_norm=cc.toda__y_cols_to_norm)
    test_dataset = TripleIndicesRegressionDataset(_df=test_df, _indices=test_indices, _feature_cols=cc.toda__x_cols, _target_col=cc.toda__y_cols, _device=cc.toda__device,
                                                  _x_seq_length=cc.toda__x_seq_length, _y_seq_length=cc.toda__y_seq_length, _x_cols_to_norm=cc.toda__x_cols_to_norm, _y_cols_to_norm=cc.toda__y_cols_to_norm)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if sanity_check:
        for a_dataloader in [train_dataloader, test_dataloader]:
            for the_x, the_y, x_data_norm, y_data_norm in tqdm(a_dataloader):
                assert torch.min(the_y) > model_args['activation_minmax'][0] and torch.max(the_y) < model_args['activation_minmax'][1]


    ###########################################################################
    # Model preparation
    ###########################################################################
    model = LSTMRegression(num_output_vars=model_args['num_output_vars'], num_input_features=model_args['num_input_features'], t_length_output_vars=model_args['t_length_output_vars'],
                           hidden_size=model_args['hidden_size'], num_layers=model_args['num_layers'], seq_length=model_args['seq_length'],
                           bidirectional=model_args['bidirectional'], device=model_args['device'], activation_minmax=model_args['activation_minmax'], dropout=model_args['dropout'])


    ###########################################################################
    # Train
    ###########################################################################
    logger.info(f"{batch_size=}   {len(train_indices)=}   {len(test_indices)=}   {len(train_df.columns)=}   {cc.toda__precision_spread=}")
    loss_function = torch.nn.MSELoss(reduction='mean').to(cc.toda__device)
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas, device_type=cc.toda__device)
    train_loss, best_test_loss, best_test_precision = torch.tensor(999999999), (999999999, 999999999), (0., 0)
    running_train_losses, running_test_losses, running_train_precision = -1, -1, -1
    train_iterator = itertools.cycle(train_dataloader)
    the_x, the_y, x_data_norm, y_data_norm = _get_batch(_batch_size=batch_size, iterator=train_iterator)
    checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'123.pt')
    os.makedirs(Path(checkpoint_filename).parent, exist_ok=True)
    while iter_num < max_iters:
        lr = _get_lr(_it=iter_num, _warmup_iters=warmup_iters, _learning_rate=learning_rate, _min_lr=min_lr, _lr_decay_iters=lr_decay_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 or 1 == iter_num or iter_num == max_iters - 1:
            model.eval()
            test_losses, test_precision = [], []
            for batch_idx, (x_test, y_test, _, y_data_norm_test) in enumerate(test_dataloader):
                test_logits, _ = model(x=x_test)  # forward pass
                assert test_logits.shape == y_test.shape
                prediction   = (test_logits + 1) * y_data_norm_test.unsqueeze(-2)
                ground_truth = (y_test + 1)      * y_data_norm_test.unsqueeze(-2)
                _loss = loss_function(prediction, ground_truth)
                test_losses.append(0 if torch.isnan(_loss) else _loss.item())
                numerator    = torch.count_nonzero(torch.abs(prediction - ground_truth) < cc.toda__precision_spread)
                denominator  = np.prod(prediction.shape)
                test_precision.append((numerator / denominator).cpu().numpy())
            test_precision = np.mean(test_precision)
            running_test_losses = np.mean(test_losses) if running_test_losses == -1.0 else 0.9 * running_test_losses + 0.1 * np.mean(test_losses)
            is_best_test_loss_achieved = np.mean(test_losses) < best_test_loss[0]
            if is_best_test_loss_achieved or 0 == iter_num:
                best_test_loss = (np.mean(test_losses), iter_num)
            if test_precision > best_test_precision[0]:
                best_test_precision = (test_precision, iter_num)
            if is_best_test_loss_achieved and 0 != iter_num:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_test_loss': best_test_loss,
                    'train_loss': train_loss.item(),
                }
                try:
                    os.remove(checkpoint_filename)
                except:
                    pass
                checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'best_test__loss_{best_test_loss[0]:.8f}__precision_{test_precision:.8f}_at_{iter_num}.pt')
                torch.save(checkpoint, checkpoint_filename)
            if 0 == iter_num % log_interval or 1 == iter_num:
                logger.info(f"iter: {iter_num:6d}   lr: {lr:0.3E}   train:({running_train_losses:.4f})/({running_train_precision:.4f})   test:({running_test_losses:.4f})/({test_precision:.4f}) "
                            f">> best: {best_test_loss[0]:.4f}/{best_test_precision[0]:.4f} @({best_test_loss[1]}/{best_test_precision[1]})")
            model.train()

        train_logits, train_loss = model(x=the_x, y=the_y)  # forward pass
        train_loss = train_loss * 100.0  # scale loss to avoid gradient vanishing
        copy_of__the_y, copy_of__train_logits, copy_of__y_data_norm = train_logits.clone().detach().cpu(), the_y.clone().detach().cpu(), y_data_norm.clone().cpu()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        the_x, the_y, x_data_norm, y_data_norm = _get_batch(_batch_size=batch_size, iterator=train_iterator)

        train_loss.backward()  # calculates the loss of the loss function
        optimizer.step()  # improve from loss, i.e backprop
        optimizer.zero_grad()

        # Compute the precision
        assert copy_of__train_logits.shape == copy_of__the_y.shape
        prediction   = (copy_of__train_logits + 1) * copy_of__y_data_norm.unsqueeze(-2)
        ground_truth = (copy_of__the_y + 1)        * copy_of__y_data_norm.unsqueeze(-2)
        numerator    = torch.count_nonzero(torch.abs(prediction - ground_truth) < cc.toda__precision_spread)
        denominator  = np.prod(prediction.shape)
        training_precision = numerator / denominator
        running_train_precision = training_precision if running_train_precision == -1.0 else 0.9 * running_train_precision + 0.1 * training_precision

        running_train_losses = train_loss.item() if running_train_losses == -1.0 else 0.9 * running_train_losses + 0.1 * train_loss.item()
        if 0 == iter_num % log_interval and False:
            logger.info(f"iter: {iter_num:6d}   lr: {lr:0.3E} [TRAIN]  loss: ({running_train_losses:.4f})   precision: ({running_train_precision:.4f})")

        iter_num += 1

if __name__ == '__main__':
    freeze_support()
    os.makedirs('stubs', exist_ok=True)

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