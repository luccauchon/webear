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


# ===== BEST PARAMETERS =====
BEST_PARAMETERS = {'LOOKAHEAD': 2, 'RETURN_THRESHOLD': 0.02, 'MMI_TREND_MAX': 3, 'MMI_PERIOD': 2, 'SMA_PERIOD': 1}
BEST_SCORE = 0.87740774
CONFIGURATION_FOR_MMI_NEXT_DAY = Namespace(
        dataset_id="day", older_dataset=None,
        mmi_period=BEST_PARAMETERS['MMI_PERIOD'],
        mmi_trend_max=BEST_PARAMETERS['MMI_TREND_MAX'],
        sma_period=BEST_PARAMETERS['SMA_PERIOD'],
        return_threshold=BEST_PARAMETERS['RETURN_THRESHOLD'],
        use_ema=True,
        verbose=False,
        filter_open_gaps=False,
    )


def main(args):
    if args.verbose:
        print("\n" + "=" * 80)
        print(f"Historical performance of {BEST_SCORE*100:.1f}% (overall accuracy)")
        print("=" * 80)
    config_dict = vars(CONFIGURATION_FOR_MMI_NEXT_DAY)
    config_dict.update({'ticker': args.ticker,'col': args.col,'verbose': args.verbose, 'keep_last_step': args.keep_last_step,})
    configuration = Namespace(**config_dict)
    return MMI_next(configuration, lookahead=BEST_PARAMETERS['LOOKAHEAD'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument('--keep_last_step', type=str2bool, default=True)
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)
