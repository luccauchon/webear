import os
import pandas as pd
import pprint
import models
from models import LSTMRegression
from utils import all_dicts_equal, namespace_to_dict, dict_to_namespace
from version import sys__version, sys__name
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import torch
import numpy as np
from loguru import logger
import sys
import shutil
from datetime import datetime
import yfinance as yf


###############################################################################
# Load default configuration
from config.default.data_prepare import *
###############################################################################


def main(cc):
    np.random.seed(cc.dp__seed_offset)
    torch.manual_seed(cc.dp__seed_offset)
    torch.cuda.manual_seed_all(cc.dp__seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    os.makedirs(cc.dp__output_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level=cc.dp__debug_level)
    logger.add(os.path.join(cc.dp__output_dir, "data_prepare.txt"), level='DEBUG')
    logger.info(f"{cc.sys__name} {cc.sys__version}")

    # Download the data
    dfs = []
    for a_dict in cc.dp__list_of_symbols:
        symbol, period, interval = a_dict['symbol'], a_dict['period'], a_dict['interval']
        _df = yf.download(symbol, period=period, interval=interval)
        if "drop" in a_dict:
            for a_drop in a_dict["drop"]:
                _df = _df.drop(a_drop, axis=1)
        dfs.append(_df)

    # Merge the data
    merged_df = pd.merge(*dfs, on='Date', how='left')

    # Add features
    if cc.dp__day_of_week:
        merged_df['day_of_week'] = merged_df.index.dayofweek + 1

    # Set it from oldest to newest
    merged_df = merged_df.sort_index(ascending=True)
    logger.debug(merged_df.tail(5))

    # Save data
    merged_df.to_pickle(cc.dp__output_filename)

    # Copy data elsewhere, if required
    if cc.dp__final_destination is not None:
        shutil.copy(src=cc.dp__output_filename, dst=cc.dp__final_destination)


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