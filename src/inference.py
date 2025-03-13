import os
import pprint
import models
from models import LSTMRegression
from utils import all_dicts_equal, namespace_to_dict, dict_to_namespace
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import torch
import numpy as np
from loguru import logger
import sys

###############################################################################
# Load default configuration
from config.default.inference import *
###############################################################################


def main(cc):
    np.random.seed(cc.inf__seed_offset)
    torch.manual_seed(cc.inf__seed_offset)
    torch.cuda.manual_seed_all(cc.inf__seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    os.makedirs(cc.inf__output_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level=cc.inf__debug_level)
    logger.add(os.path.join(cc.inf__output_dir, "train.txt"), level='DEBUG')
    logger.info(f"{cc.sys__name} {cc.sys__version}")




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