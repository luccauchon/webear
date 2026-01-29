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
# {'LOOKAHEAD': 1, 'RETURN_THRESHOLD': 0.025, 'MMI_TREND_MAX': 7, 'MMI_PERIOD': 2, 'SMA_PERIOD': 1}
# Best Score: 0.81514800
CONFIGURATION_FOR_MMI_NEXT_WEEEK = Namespace(
        dataset_id="week", older_dataset=None,
        mmi_period=2,
        mmi_trend_max=7,
        sma_period=1,
        return_threshold=0.025,
        use_ema=False,
        verbose=False,
    )


def main(args):
    if args.verbose:
        print("\n" + "=" * 80)
        print(f"Historical performance of 81.5148% (overall accuracy)")
        print("=" * 80)
    config_dict = vars(CONFIGURATION_FOR_MMI_NEXT_WEEEK)
    config_dict.update({'ticker': args.ticker, 'col': args.col, 'verbose': args.verbose, 'older_dataset': args.older_dataset})
    configuration = Namespace(**config_dict)
    return MMI_next(configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)
