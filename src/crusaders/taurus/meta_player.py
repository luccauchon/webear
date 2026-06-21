try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
import pathlib
from argparse import Namespace
from datetime import datetime
from crusaders.taurus.player import entry as player_entry
import os
from pprint import pprint
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )
    return parser.parse_args()


def entry(args):
    verbose = True
    # Get the current date and time
    current_date = datetime.now()

    # Format the date as YYYY_MM_DD
    date_string = current_date.strftime("%Y_%m_%d")

    # Combine into your final filename
    filename = f"report_{date_string}.pkl"
    save_to  = f"{filename}"
    if not os.path.exists(save_to):
        configuration = Namespace(prime_rsi_target_dir=r"C:\Temp2\prime_rsi_1B_to_20B_V2",
                                  autotune_target_dir=".", dgdr_target_dir=".", oerh_target_dir=".", load_from=None,
                                  nb_workers=20, save_to=save_to, info=None, threshold=None, verbose=False,
                                  min_val_rate=None, min_train_rate=None, hide_zero_signal=False, signal=None, indicator=None, method=None, optimize_target=None)
        player_entry(args=configuration)

    result = {}
    for lookahead in tqdm(range(1, 21), desc="Progression lookahead"):
        lookahead_str = f"{lookahead}"
        result[lookahead_str] = {}

        for optimize_target in ["buy_wr", "sell_wr"]:
            result[lookahead_str][optimize_target] = {}
            _ranges = np.arange(0.999, 0.94, -0.001)
            if optimize_target in ["sell_wr"]:
                _ranges = np.arange(1.001, 1.004, 0.001)
            for threshold in _ranges:
                thresh_str = f"{threshold:.3f}::"
                if optimize_target in ["sell_wr"]:
                    thresh_str = f"::{threshold:.3f}"
                configuration = Namespace(
                    prime_rsi_target_dir=None, nb_workers=None, verbose=False,
                    autotune_target_dir=None, dgdr_target_dir=None, oerh_target_dir=None,
                    load_from=save_to, save_to=None,
                    info=["^GSPC", "::day", f"::{lookahead} "],
                    threshold=[thresh_str], min_val_rate=None, min_train_rate=None,
                    hide_zero_signal=True, signal=None, indicator=None, method=None,
                    optimize_target=optimize_target
                )

                result[lookahead_str][optimize_target][thresh_str] = player_entry(args=configuration)


    if verbose:
        pprint(result)


if __name__ == "__main__":
    args = parse_args()
    entry(args)