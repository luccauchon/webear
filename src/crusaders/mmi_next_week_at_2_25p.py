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


# PY312_HT) D:\PyCharmProjects\webear\src\runners>python MMI_hyperparameter_search_optuna.py --dataset_id=week --step_back_range=10000 --n_trials=30000 --return_threshold_min=0.02 --return_threshold_max=0.02 --lookahead_min=1 --lookahead_max=1
# 🔧 Arguments:
#     ticker.................................. ^GSPC
#     col..................................... Close
#     dataset_id.............................. week
#     step_back_range......................... 10000
#     n_trials................................ 30000
#     return_threshold_min.................... 0.02
#     return_threshold_max.................... 0.02
#     mmi_trend_max_min....................... 1
#     mmi_trend_max_max....................... 500
#     mmi_period_min.......................... 1
#     mmi_period_max.......................... 500
#     sma_period_min.......................... 1
#     sma_period_max.......................... 500
#     metric.................................. overall_accuracy
#     lookahead_min........................... 1
#     lookahead_max........................... 1
# --------------------------------------------------------------------------------
# Best trial: 136. Best value: 0.724608: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [92:36:07<00:00, 11.11s/it]
#
# ===== BEST PARAMETERS =====
# {'LOOKAHEAD': 1, 'RETURN_THRESHOLD': 0.02, 'MMI_TREND_MAX': 26, 'MMI_PERIOD': 2, 'SMA_PERIOD': 1}
# Best Score: 0.72460824
CONFIGURATION_FOR_MMI_NEXT_WEEEK = Namespace(
        dataset_id="week", older_dataset=None,
        mmi_period=2,
        mmi_trend_max=26,
        sma_period=1,
        return_threshold=0.02,
        use_ema=False,
        verbose=False,
    )


def main(args):
    if args.verbose:
        print("\n" + "=" * 80)
        print(f"Historical performance of 72.4608% (overall accuracy)")
        print("=" * 80)
    config_dict = vars(CONFIGURATION_FOR_MMI_NEXT_WEEEK)
    config_dict.update({'ticker': args.ticker, 'col': args.col, 'verbose': args.verbose, })
    configuration = Namespace(**config_dict)
    return MMI_next(configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)
