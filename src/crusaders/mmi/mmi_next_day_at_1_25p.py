try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
from argparse import Namespace
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool
import copy
import numpy as np
from datetime import datetime, timedelta
from crusaders.mmi.mmi_next import main as MMI_next


# (PY311) C:\PYCHARMPROJECTS\webear\src\runners>python .\MMI_hyperparameter_search_optuna.py --dataset_id=day --step_back_range=8000 --n_trials=33333 --return_threshold_min=0.0125 --return_threshold_max=0.0125 --lookahead_min=1 --lookahead_max=1
# 🔧 Arguments:
#     ticker.................................. ^GSPC
#     col..................................... Close
#     dataset_id.............................. day
#     step_back_range......................... 8000
#     n_trials................................ 33333
#     return_threshold_min.................... 0.0125
#     return_threshold_max.................... 0.0125
#     mmi_trend_max_min....................... 1
#     mmi_trend_max_max....................... 500
#     mmi_period_min.......................... 1
#     mmi_period_max.......................... 500
#     sma_period_min.......................... 1
#     sma_period_max.......................... 500
#     lookahead_min........................... 1
#     lookahead_max........................... 1
# --------------------------------------------------------------------------------
#
# Best trial: 11210. Best value: 0.809875: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33333/33333 [261:33:29<00:00, 28.25s/it]

# ===== BEST PARAMETERS =====
# {'LOOKAHEAD': 1, 'RETURN_THRESHOLD': 0.0125, 'MMI_TREND_MAX': 43, 'MMI_PERIOD': 156, 'SMA_PERIOD': 188}
# Best Score: 0.80987500
CONFIGURATION_FOR_MMI_NEXT_DAY = Namespace(
        dataset_id="day", older_dataset=None,
        mmi_period=156,
        mmi_trend_max=43,
        sma_period=188,
        return_threshold=0.0125,
        use_ema=False,
        verbose=False,
    )


def main(args):
    if args.verbose:
        print("\n" + "=" * 80)
        print(f"Historical performance of 80.9875% (overall accuracy)")
        print("=" * 80)
    config_dict = vars(CONFIGURATION_FOR_MMI_NEXT_DAY)
    config_dict.update({'ticker': args.ticker,'col': args.col,'verbose': args.verbose, 'keep_last_step': args.keep_last_step,})
    configuration = Namespace(**config_dict)
    return MMI_next(configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument('--keep_last_step', type=str2bool, default=True)
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)
