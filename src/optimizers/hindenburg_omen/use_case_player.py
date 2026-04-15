#!/usr/bin/env python3
"""
Hindenburg Omen Realtime Backtest Runner with Visualization

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
from utils import next_weekday


def parse_arguments():
    """
    Configures and parses command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="Run realtime backtest and hyperparameter search on Hindenburg Omen data files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example:\n  python script.py --base-dir 'D:\\Temp2\\use_case' --experience-id 'alpha_3' --visualize"
    )

    # Directory Configuration
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="The root directory containing the experience folders."
    )

    parser.add_argument(
        "--experience-id",
        type=str,
        required=True,
        help="The specific experience ID subfolder to process within the base directory."
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
        help="Secondary boolean flag passed to run_realtime_only(). (Default: False)"
    )

    parser.add_argument(
        "--disable-progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars. (Default: False)"
    )

    # Visualization Options
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate visualization plots instead of text output. (Default: False)"
    )

    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save the plot to this file path (e.g., 'results.png'). If not specified, shows interactively."
    )

    return parser.parse_args()


def create_visualization(results, save_path=None):
    """
    Create matplotlib visualization of the backtest results.

    Args:
        results: List of tuples (edge, result_dict)
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    # Extract data from results
    # Sort by days (extract from filename or use the prediction horizon)
    sorted_results = sorted(results, key=lambda x: x[1]['prediction_days'])

    days = [r[1]['prediction_days'] for r in sorted_results]
    win_rates = [r[1]['win_rate'] for r in sorted_results]
    baselines = [r[1]['baseline'] for r in sorted_results]
    edges = [r[1]['win_rate'] - r[1]['baseline'] for r in sorted_results]
    dates = [r[1]['last_date'] for r in sorted_results]
    directions = [r[1]['event_direction'] for r in sorted_results]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Hindenburg Omen Backtest Results', fontsize=16, fontweight='bold')

    # Colors based on direction
    colors = ['red' if d == 'drop' else 'green' for d in directions]

    # Plot 1: Win Rate vs Time Horizon (Days)
    ax1 = axes[0]
    scatter1 = ax1.scatter(days, win_rates, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax1.plot(days, win_rates, 'o-', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Prediction Horizon (Days)', fontsize=11)
    ax1.set_ylabel('Win Rate (%)', fontsize=11)
    ax1.set_title('Win Rate vs Time Horizon', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Baseline vs Time Horizon
    ax2 = axes[1]
    scatter2 = ax2.scatter(days, baselines, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax2.plot(days, baselines, 's-', linewidth=2, alpha=0.5, color='orange')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax2.set_xlabel('Prediction Horizon (Days)', fontsize=11)
    ax2.set_ylabel('Baseline (%)', fontsize=11)
    ax2.set_title('Baseline vs Time Horizon', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def main():
    """
    Main execution flow.
    1. Parses arguments.
    2. Constructs the target directory path.
    3. Iterates through files and runs the backtest.
    4. Prints signal information if active OR creates visualization.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Construct the full path to the experience directory
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
    all_files_to_process = [file for file in Path(target_directory).iterdir() if file.is_file()]

    if not all_files_to_process:
        print("No files found to process.")
        return

    # Configure tqdm (progress bar)
    iterator = all_files_to_process
    if not args.disable_progress:
        iterator = tqdm(all_files_to_process, desc="Processing Files")

    results = []

    for file in iterator:
        info = run_realtime_only(params_file=file, verbose=args.run_flag)

        is_active_now = info["is_active_now"]

        if is_active_now:
            event_direction = info["event_direction"]
            current_count = info["current_count"]
            cluster_threshold = info["cluster_threshold"]

            prediction_days = info["forward_days"]

            direction_word = "DROP" if event_direction == "drop" else "SPIKE"

            # Colors
            RED = "\033[91m"
            GREEN = "\033[92m"
            BOLD = "\033[1m"
            RESET = "\033[0m"

            edge = info['win_rate'] - info['baseline']
            edge_color = GREEN if edge > 0 else RED
            win_rate = info['win_rate']
            if not args.visualize:
                # Text output mode (original behavior)
                _tmp_str = f"SIGNAL ACTIVE: YES - PREDICTING {direction_word}"
                _tmp_str2 = f"{current_count} / {cluster_threshold}"

                _tmp_str3 = (
                    f"[{event_direction}]    "
                    f"{BOLD}{edge_color}Edge: {edge:.2f}%{RESET}   "
                    f"{BOLD}{GREEN}Win Rate: {info['win_rate']:.2f}%{RESET}   "
                    f"Baseline: {info['baseline']:.2f}%   ({info['total_events']} / {info['total_days']})"
                )

                result_str = (
                    f"{file.name} {info['last_date'].strftime('%Y-%m-%d')}  {_tmp_str} ({_tmp_str2})  {next_weekday(info['last_date'], int(prediction_days))}\n"
                    f"\t{info['is_active_str']}  {_tmp_str3}\n\n"
                )

                results.append((win_rate, result_str))
            else:
                # Visualization mode - store structured data
                results.append((win_rate, {
                    'filename': file.name,
                    'last_date': info['last_date'],
                    'prediction_days': prediction_days,
                    'win_rate': win_rate,
                    'baseline': info['baseline'],
                    'edge': edge,
                    'event_direction': event_direction,
                    'total_events': info['total_events'],
                    'total_days': info['total_days']
                }))

    if args.visualize:
        # Create visualization
        if results:
            create_visualization(results, save_path=args.save_plot)
        else:
            print("No active signals found to visualize.")
    else:
        # Sort and print text results (original behavior)
        results.sort(key=lambda x: x[0], reverse=False)

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
        import traceback

        traceback.print_exc()
        sys.exit(1)