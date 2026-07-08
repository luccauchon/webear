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
import json
import argparse
import pickle
import pathlib
from argparse import Namespace
from datetime import datetime
from crusaders.taurus.player import entry as player_entry
import os
from pprint import pprint
import numpy as np
from tqdm import tqdm
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )
    parser.add_argument(
        "--autotune-target-dir",
        required=False,
        default=".",
        help="Target directory for autotune models"
    )
    parser.add_argument(
        "--dgdr-target-dir",
        required=False,
        default=".",
        help="Target directory for dgdr models"
    )
    parser.add_argument(
        "--oerh-target-dir",
        required=False,
        default=".",
        help="Target directory for oerh models"
    )
    parser.add_argument(
        "--prime-rsi-target-dir",
        required=False,
        default=".",
        help="Target directory for prime_rsi models"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output during processing"
    )
    parser.add_argument('--dataset-id', type=str, default='day')
    parser.add_argument(
        "--json-file",
        required=False,
        default=None,
        help="Specify the output JSON file name. If not provided, defaults to 'taurus_visualization_{dataset_id}_{date}.json'"
    )
    parser.add_argument(
        "--pkl-file",
        required=False,
        default=None,
        help="Specify the input .pkl file to load extracted information from. If not provided, defaults to a generated filename based on the date and clip flag."
    )
    parser.add_argument("--clip", action="store_true", help="Exclude incomplete current bar in real-time")
    parser.add_argument(
        "-r", "--remove-results",
        action="store_true",
        default=False,
        help="Remove the file that contains results computed by workers"
    )
    return parser.parse_args()


def entry(args):
    # Get the current date and time and format the date as YYYY_MM_DD
    date_string = datetime.now().strftime("%Y_%m_%d")

    # Determine the output JSON file name
    if args.json_file:
        json_file = args.json_file
    else:
        json_file = f"taurus_visualization_{args.dataset_id}_{date_string}.json"

    if os.path.exists(json_file):
        print(f"Found {json_file}. Exiting.")
        sys.exit(0)

    ###########################################################################
    # Process all use cases and save the result
    if args.pkl_file:
        filename_extracted_information = args.pkl_file
    else:
        filename_extracted_information = f"taurus_played__{date_string}.pkl"
        if args.clip:
            filename_extracted_information = f"taurus_played_clipped__{date_string}.pkl"

    if not os.path.exists(filename_extracted_information):
        configuration = Namespace(prime_rsi_target_dir=args.prime_rsi_target_dir,
                                  autotune_target_dir=args.autotune_target_dir,
                                  dgdr_target_dir=args.dgdr_target_dir,
                                  oerh_target_dir=args.oerh_target_dir,
                                  load_from=None, nb_workers=20, save_to=filename_extracted_information, info=None, threshold=None, verbose=False, clip=args.clip,
                                  min_val_rate=None, min_train_rate=None, hide_zero_signal=False, signal=None, indicator=None, method=None, optimize_target=None)
        player_entry(args=configuration)

    ###########################################################################
    # Extract boundaries information from all the use cases processed
    with open(filename_extracted_information, 'rb') as f:
        data_from_workers = pickle.load(f)

    if args.remove_results and not args.pkl_file:
        os.remove(filename_extracted_information)
    else:
        if not args.pkl_file:
            print(f"Computed results are in {filename_extracted_information}")

    ranges_for__optimize_target, ranges_for__put_threshold, ranges_for__call_threshold = [], [], []
    ranges_for__lookahead, current_price = [], -1

    for one_use_case in data_from_workers:
        assert 3 == len(one_use_case["info"].split("::"))
        if str(one_use_case["info"].split("::")[1]).strip() not in [args.dataset_id]:
            continue
        ranges_for__lookahead.append(int(str(one_use_case["info"].split("::")[2]).strip()))
        assert one_use_case["optimize_target"] in ["sell_wr", "buy_wr"]
        ranges_for__optimize_target.append(one_use_case["optimize_target"])
        if one_use_case["optimize_target"] == "buy_wr":
            ranges_for__put_threshold.append(float(one_use_case["threshold"]))
        if one_use_case["optimize_target"] == "sell_wr":
            ranges_for__call_threshold.append(float(one_use_case["threshold"]))
        if current_price == -1:
            current_price = one_use_case["current_price"]
        assert current_price == one_use_case["current_price"]
    ranges_for__optimize_target = list(set(ranges_for__optimize_target))
    ranges_for__put_threshold = list(sorted(set(ranges_for__put_threshold)))
    ranges_for__call_threshold = list(sorted(set(ranges_for__call_threshold)))
    ranges_for__lookahead = list(sorted(set(ranges_for__lookahead)))
    ranges_for__lookahead = list(range(1, max(ranges_for__lookahead) + 1))

    ###########################################################################
    # Generate the json to be used for visualization
    result = {"now": f"{date_string}", "dataset_id": args.dataset_id, "current_price": current_price}
    outer_loop = tqdm(ranges_for__lookahead, desc="Lookahead") if args.verbose else ranges_for__lookahead
    for lookahead in outer_loop:
        lookahead_str = f"{lookahead}"
        result[lookahead_str] = {}
        for optimize_target in ranges_for__optimize_target:
            result[lookahead_str][optimize_target] = {}
            _ranges = ranges_for__put_threshold
            if optimize_target in ["sell_wr"]:
                _ranges = ranges_for__call_threshold
            inner_loop = tqdm(_ranges, desc=f"Target (LH {lookahead} / OT {optimize_target})", leave=False) if args.verbose else _ranges
            for threshold in inner_loop:
                thresh_str = f"{threshold:.3f}"
                configuration = Namespace(
                    prime_rsi_target_dir=None, nb_workers=None, verbose=False,
                    autotune_target_dir=None, dgdr_target_dir=None, oerh_target_dir=None,
                    load_from=filename_extracted_information, save_to=None, dataset_id=args.dataset_id,
                    info=["^GSPC", f"::{args.dataset_id}", f"::{lookahead} "], min_threshold=None, max_threshold=None,
                    threshold=[thresh_str], min_val_rate=None, min_train_rate=None,
                    hide_zero_signal=True, signal=None, indicator=None, method=None,
                    optimize_target=optimize_target
                )
                result[lookahead_str][optimize_target][thresh_str] = player_entry(args=configuration)

    # Write the result
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    entry(args)