import os
import pprint
from tqdm import tqdm
import itertools
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import TripleIndicesLookAheadClassificationDataset
from models import LSTMClassification
from utils import all_dicts_equal, namespace_to_dict, dict_to_namespace, get_stub_dir, get_df_SPY_and_VIX, generate_indices_with_cutoff_day, calculate_classification_metrics, generate_indices_with_multiple_cutoff_day
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import torch
import numpy as np
from loguru import logger
import sys
import pandas as pd
import math

###############################################################################
# Load default configuration
from config.default.train_basic_binary_classification_rc1 import *
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


def _get_batch(_batch_size, iterator):
    the_x, the_y, x_data_norm = next(iterator)
    while len(the_x) < _batch_size:
        x, y, xn = next(iterator)
        the_x = torch.concatenate([the_x, x])
        the_y = torch.concatenate([the_y, y])
        x_data_norm = torch.concatenate([x_data_norm, xn])
    the_x, the_y, x_data_norm = the_x[:_batch_size, ...], the_y[:_batch_size, ...], x_data_norm[:_batch_size, ...]
    return the_x, the_y, x_data_norm


def main(cc):
    np.random.seed(cc.toda__seed_offset)
    torch.manual_seed(cc.toda__seed_offset)
    torch.cuda.manual_seed_all(cc.toda__seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    output_dir = os.path.join(get_stub_dir(), f"trainer_basic_classification__{cc.toda__version}__run_{cc.toda__run_id}")
    os.makedirs(output_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=cc.toda__debug_level)
    logger.add(os.path.join(output_dir, "train.txt"), level='DEBUG')

    ###########################################################################
    # Description
    ###########################################################################
    logger.info("The goal is to predict if the value of thursday SPY.close will be higher or lower then the monday close value, based on the preceding P days")


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
    logger.info(f"Data is ranging from {df.index[0]} to {df.index[-1]}")
    logger.info(f"Training is ranging from {cc.toda__tav_dates[0]} to {cc.toda__tav_dates[1]}  ,  {num_input_features} input features ({cc.toda__x_seq_length} steps) and {num_output_features} output features ({cc.toda__y_seq_length} steps)")
    logger.info(f"MES is ranging from {cc.toda__mes_dates[0]} to {cc.toda__mes_dates[1]}")

    ###########################################################################
    # Configuration
    ###########################################################################
    batch_size     = 512
    betas          = (0.9, 0.95)
    decay_lr       = True
    eval_interval  = 10
    iter_num       = 0
    learning_rate  = 1e-4
    log_interval   = 100
    lr_decay_iters = 50000
    max_iters      = 50000
    min_lr         = 1e-6
    sanity_check   = True
    warmup_iters   = 0
    weight_decay   = 0.5

    model_args = {'bidirectional': False,
                  'dropout': 0.5,
                  'hidden_size': 256,
                  'num_layers': 1,
                  'num_output_vars': num_output_features,
                  'num_input_features': num_input_features,
                  'device': cc.toda__device,
                  'activation_minmax': (-2, 2),
                  'seq_length': cc.toda__x_seq_length,
                  }


    ###########################################################################
    # Data preparation
    ###########################################################################
    assert 1 == cc.toda__y_seq_length
    train_indices, train_df = generate_indices_with_multiple_cutoff_day(cutoff_days=cc.toda__cutoff_days, _df=df.copy(), _dates=cc.toda__tav_dates, x_seq_length=cc.toda__x_seq_length, y_seq_length=cc.toda__y_seq_length)
    test_indices,  test_df  = generate_indices_with_multiple_cutoff_day(cutoff_days=cc.toda__cutoff_days, _df=df.copy(), _dates=cc.toda__mes_dates, x_seq_length=cc.toda__x_seq_length, y_seq_length=cc.toda__y_seq_length)

    train_dataset    = TripleIndicesLookAheadClassificationDataset(_df=train_df, _feature_cols=cc.toda__x_cols, _target_col=cc.toda__y_cols, _device=cc.toda__device, _x_cols_to_norm=cc.toda__x_cols_to_norm, _indices=train_indices, _mode='train')
    test_dataset     = TripleIndicesLookAheadClassificationDataset(_df=test_df, _feature_cols=cc.toda__x_cols, _target_col=cc.toda__y_cols, _device=cc.toda__device, _x_cols_to_norm=cc.toda__x_cols_to_norm, _indices=test_indices, _mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    if sanity_check:
        for a_dataloader in [train_dataloader, test_dataloader]:
            for train_x, train_y, train_x_data_norm in tqdm(a_dataloader):
                pass

    ###########################################################################
    # Model preparation
    ###########################################################################
    model = LSTMClassification(num_input_features=model_args['num_input_features'],
                               hidden_size=model_args['hidden_size'], num_layers=model_args['num_layers'], seq_length=model_args['seq_length'],
                               bidirectional=model_args['bidirectional'], device=model_args['device'], dropout=model_args['dropout'])

    ###########################################################################
    # Train
    ###########################################################################
    logger.info(f"{batch_size=}   {len(train_indices)=}   {len(test_indices)=}   {len(df.columns)=}")
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='sum').to(cc.toda__device)  # mean-squared error for regression
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas, device_type=cc.toda__device)
    train_loss, best_test_loss, best_test_accuracy = torch.tensor(999999999), (999999999, 999999999), (0., 0)
    running_train_losses, running_test_losses, running_train_accuracy = -1, -1, -1
    train_iterator = itertools.cycle(train_dataloader)
    train_x, train_y, train_x_data_norm = _get_batch(_batch_size=batch_size, iterator=train_iterator)
    checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'123.pt')
    os.makedirs(Path(checkpoint_filename).parent, exist_ok=True)
    while iter_num < max_iters:
        lr = _get_lr(_it=iter_num, _warmup_iters=warmup_iters, _learning_rate=learning_rate, _min_lr=min_lr, _lr_decay_iters=lr_decay_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 or 1 == iter_num or iter_num == max_iters - 1:
            model.eval()
            test_losses, test_accuracy = [], []
            for batch_idx, (x_test, y_test, _) in enumerate(test_dataloader):
                y_test = y_test.unsqueeze(1)
                test_logits, _ = model(x=x_test)  # forward pass
                assert test_logits.shape == y_test.shape
                _loss = loss_function(test_logits, y_test.float())
                test_losses.append(0 if torch.isnan(_loss) else _loss.item())
                test_accuracy.append(calculate_classification_metrics(y_true=y_test, y_pred=test_logits > 0.5)['accuracy'])
            test_accuracy = np.mean(test_accuracy)
            running_test_losses = np.mean(test_losses) if running_test_losses == -1.0 else 0.9 * running_test_losses + 0.1 * np.mean(test_losses)
            is_best_test_loss_achieved = np.mean(test_losses) < best_test_loss[0]
            if is_best_test_loss_achieved or 0 == iter_num:
                best_test_loss = (np.mean(test_losses), iter_num)
            if test_accuracy > best_test_accuracy[0]:
                best_test_accuracy = (test_accuracy, iter_num)
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
                checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'best_test__loss_{best_test_loss[0]:.8f}__precision_{test_accuracy:.8f}_at_{iter_num}.pt')
                torch.save(checkpoint, checkpoint_filename)
            if 0 == iter_num % log_interval or 1 == iter_num:
                logger.info(f"iter: {iter_num:6d}   lr: {lr:0.3E}   train:({running_train_losses:.4f})/({running_train_accuracy:.4f})   "
                            f"test:({running_test_losses:.4f})/{test_accuracy:.4f} "
                            f">> best accuracy test: {best_test_loss[0]:.4f}/{best_test_accuracy[0]:.4f} @({best_test_loss[1]}/{best_test_accuracy[1]})")
            model.train()

        train_logits, train_loss = model(x=train_x, y=train_y)  # forward pass
        train_loss = train_loss * 100.0  # scale loss to avoid gradient vanishing

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        train_x, train_y, train_x_data_norm = _get_batch(_batch_size=batch_size, iterator=train_iterator)

        train_loss.backward()  # calculates the loss of the loss function
        optimizer.step()  # improve from loss, i.e backprop
        optimizer.zero_grad()

        # Compute accuracy
        training_accuracy = calculate_classification_metrics(y_true=train_y, y_pred=train_logits>0.5)['accuracy']

        running_train_accuracy = training_accuracy if running_train_accuracy == -1.0 else 0.9 * running_train_accuracy + 0.1 * training_accuracy
        running_train_losses = train_loss.item() if running_train_losses == -1.0 else 0.9 * running_train_losses + 0.1 * train_loss.item()

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