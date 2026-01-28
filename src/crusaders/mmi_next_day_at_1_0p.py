try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
from argparse import Namespace
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool
import copy
import numpy as np
from datetime import datetime, timedelta
from crusaders.mmi_next import main as MMI_next


# PY312_HT) D:\PyCharmProjects\webear\src\runners>python MMI_hyperparameter_search_optuna.py --n_trials=1000 --step_back_range=10000 --return_threshold_min=0.01 --return_threshold_max=0.01 --lookahead_min=1 --lookahead_max=1
# 🔧 Arguments:
#     ticker.................................. ^GSPC
#     col..................................... Close
#     dataset_id.............................. day
#     step_back_range......................... 10000
#     n_trials................................ 1000
#     return_threshold_min.................... 0.01
#     return_threshold_max.................... 0.01
#     mmi_trend_max_min....................... 1
#     mmi_trend_max_max....................... 500
#     mmi_period_min.......................... 1
#     mmi_period_max.......................... 500
#     sma_period_min.......................... 1
#     sma_period_max.......................... 500
#     lookahead_min........................... 1
#     lookahead_max........................... 1
# --------------------------------------------------------------------------------
# Best trial: 14. Best value: 0.7442: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [22:02:37<00:00, 79.36s/it]
#
# ===== BEST PARAMETERS =====
# {'LOOKAHEAD': 1, 'RETURN_THRESHOLD': 0.01, 'MMI_TREND_MAX': 28, 'MMI_PERIOD': 346, 'SMA_PERIOD': 4}
# Best Score: 0.74420000
CONFIGURATION_FOR_MMI_NEXT_DAY = Namespace(
        dataset_id="day", older_dataset=None,
        mmi_period=346,
        mmi_trend_max=28,
        sma_period=4,
        return_threshold=0.010,
        use_ema=False,
        verbose=False,
    )


def main(args):
    if args.verbose:
        print("\n" + "=" * 80)
        print(f"Historical performance of 74.42% (overall accuracy)")
        print("=" * 80)
    config_dict = vars(CONFIGURATION_FOR_MMI_NEXT_DAY)
    config_dict.update({'ticker': args.ticker,'col': args.col,'verbose': args.verbose,})
    configuration = Namespace(**config_dict)
    return MMI_next(configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)
