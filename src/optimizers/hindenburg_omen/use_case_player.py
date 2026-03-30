#!/usr/bin/env python3
"""
Hindenburg Omen Realtime Backtest Runner

This script iterates through files in a specified experience directory,
runs a realtime backtest optimization on each file, and reports active signals
based on the Hindenburg Omen indicator logic.

All configuration parameters are exposed via command-line arguments.
"""
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
import os
import sys
from pathlib import Path
from tqdm import tqdm
from constants import IS_RUNNING_ON_CASIR
# Import the core logic function from the optimizer module
from optimizers.hindenburg_omen.realtime_backtest_and_hyperparameter_search_optuna import run_realtime_only


def parse_arguments():
    """
    Configures and parses command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="Run realtime backtest and hyperparameter search on Hindenburg Omen data files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example:\n  python script.py --base-dir 'D:\\Temp2\\use_case' --experience-id 'alpha_3' --verbose"
    )

    # Directory Configuration
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="The root directory containing the experience folders. (Default: D:\\Temp2\\use_case)"
    )

    parser.add_argument(
        "--experience-id",
        type=str,
        required=True,
        help="The specific experience ID subfolder to process within the base directory. (Default: alpha_3)"
    )

    # Execution Flags
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable detailed logging for every file processed (Win Rate, Baseline, etc.). (Default: False)"
    )

    parser.add_argument(
        "--run-flag",
        action="store_true",
        default=False,
        help="Secondary boolean flag passed to run_realtime_only(). Use this if the underlying function requires a specific mode. (Default: False)"
    )

    parser.add_argument(
        "--disable-progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars. Useful when logging output to a file. (Default: False)"
    )

    return parser.parse_args()


def main():
    """
    Main execution flow.
    1. Parses arguments.
    2. Constructs the target directory path.
    3. Iterates through files and runs the backtest.
    4. Prints signal information if active.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Construct the full path to the experience directory
    # Using pathlib for robust path handling across OS
    target_directory = os.path.join(args.base_dir, args.experience_id)

    # Validate directory existence before proceeding
    if not os.path.exists(target_directory):
        print(f"Error: Directory not found: {target_directory}")
        sys.exit(1)

    if not os.path.isdir(target_directory):
        print(f"Error: Path is not a directory: {target_directory}")
        sys.exit(1)

    print(f"Processing files in: {target_directory}")
    print("-" * 50)

    # Collect all files to process
    # We filter to ensure we only process actual files, not subdirectories
    all_files_to_process = [file for file in Path(target_directory).iterdir() if file.is_file()]

    if not all_files_to_process:
        print("No files found to process.")
        return

    # Configure tqdm (progress bar)
    # If --disable-progress is set, we use a simple iterator instead of tqdm wrapper
    iterator = all_files_to_process
    if not args.disable_progress:
        iterator = tqdm(all_files_to_process, desc="Processing Files")
    _tmp_str_result = ''

    results = []
    for file in iterator:
        info = run_realtime_only(params_file=file, verbose=args.run_flag)

        is_active_now = info["is_active_now"]

        if is_active_now:
            event_direction = info["event_direction"]
            current_count = info["current_count"]
            cluster_threshold = info["cluster_threshold"]

            direction_word = "DROP" if event_direction == "drop" else "SPIKE"

            # ✅ Correct colors
            RED = "\033[91m"
            GREEN = "\033[92m"
            BOLD = "\033[1m"
            RESET = "\033[0m"

            edge = info['win_rate'] - info['baseline']
            edge_color = GREEN if edge > 0 else RED
            _tmp_str = f"SIGNAL ACTIVE: YES - PREDICTING {direction_word}"
            _tmp_str2 = f"{current_count} / {cluster_threshold}"

            _tmp_str3 = (
                f"[{event_direction}]    "
                f"{BOLD}{edge_color}Edge: {edge:.2f}%{RESET}   "
                f"{BOLD}{GREEN}Win Rate: {info['win_rate']:.2f}%{RESET}   "
                f"Baseline: {info['baseline']:.2f}%   ({info['total_events']} / {info['total_days']})"
            )

            result_str = (
                f"{file.name} {info['last_date'].strftime('%Y-%m-%d')}  {_tmp_str} ({_tmp_str2})\n"
                f"\t{info['is_active_str']}  {_tmp_str3}\n\n"
            )

            # ✅ store (edge, string)
            results.append((edge, result_str))

    # ✅ Sort by edge
    results.sort(key=lambda x: x[0], reverse=False)

    # ✅ Print
    for _, res in results:
        print(res)

    print("-" * 50)
    print("Processing complete.")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)