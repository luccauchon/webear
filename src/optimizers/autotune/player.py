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

from optimizers.autotune.realtime_and_backtest_hyperparameter_search_optuna import entry as autotune
from datetime import datetime


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Creates and configures the argument parser for the batch autotune script.
    """
    parser = argparse.ArgumentParser(
        prog="autotune_batch",
        description="Sequentially runs the autotune optimizer on files in a specified directory.",
        epilog="Example: python script.py -d ./configs -e .pkl --dataset-id day --ticker ^GSPC -v"
    )
    parser.add_argument(
        "-d", "--target-dir",
        type=str,
        default="./models",
        help="Directory containing .pkl model files (default: ./models)"
    )
    parser.add_argument(
        "-f", "--target-files",
        type=str,
        nargs='+',
        help="List of specific .pkl model files to process"
    )
    parser.add_argument(
        "--hide-zero-signal",
        action="store_true",
        default=False,
        help="Hide rows where signal is 0 (default: False)"
    )
    parser.add_argument('--dataset-id', type=str, default='day')
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output during processing"
    )
    return parser


def entry(args: argparse.Namespace | dict | None = None) -> None:
    """
    Main entry point for batch autotune processing.

    Args:
        args (argparse.Namespace or dict, optional): Parsed CLI arguments or config dict.
            If None, arguments are automatically parsed from sys.argv.
    """
    # Parse from CLI if no args provided
    if args is None:
        parser = create_argument_parser()
        args = parser.parse_args()

    files, extension = [], ".pkl"
    if args.target_files:
        # Process specific files provided by the user
        files = [pathlib.Path(f).resolve() for f in args.target_files]
        files = sorted([f for f in files if f.is_file() and f.suffix.lower() == ".pkl"])
    else:
        # Fallback to directory scanning
        target_dir = pathlib.Path(args.target_dir or "./models").resolve()
        try:
            files = sorted([
                f for f in target_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".pkl"
            ])
        except Exception:
            files = []

    results = []

    if not files:
        if args.verbose: print(f"⚠️ No valid {extension} files found.")
        return results

    if args.verbose: print(f"Found {len(files)} {extension} file(s). Starting sequential autotune...\n")
    dir_path = pathlib.Path(".").resolve()
    # Loop through files one by one
    for file_path in files:
        if args.verbose:
            print(f"\n{'=' * 60}")
            print(f"STARTING AUTOTUNE FOR: {file_path.name}")
            print(f"{'=' * 60}")

        # Build configuration for the current file using CLI arguments
        configuration = Namespace(
            real_time=True,
            model_path=str(file_path),
            verbose_short=False,
            dataset_id=None,
            ticker="^GSPC",
            verbose=False,
            output_dir=str(dir_path),
            length_dataset=999999,
            optimize=False,
            clip=False,
        )
        try:
            result = autotune(configuration)
            result["file"]: file_path.name
            results.append(result)
        except Exception as e:
            print(f"❌ ERROR processing {file_path.name}: {e}")

        # 1. Filter results if hide_zero_signal is enabled
        if args.hide_zero_signal:
            results = [r for r in results if r["signal"] != 0]

        # 2. Sort by target_date descending (most recent first)
        # Normalize dates to YYYY-MM-DD strings to safely compare pandas.Timestamp vs datetime.date
        results.sort(
            key=lambda r: (
                r["target_date"] is not None,
                r["target_date"].strftime('%Y-%m-%d') if hasattr(r["target_date"], 'strftime') else str(r["target_date"])
            ),
            reverse=False
        )

        # Print results
        headers = ["Info", "Signal", "Current Price", "Current Date", "Target Price", "Target Date", "Train Win Rate", "Val Win Rate", "Optimize Target", "Method", "Threshold"]
        table_rows = []
        for res in results:
            sig = str(res["signal"]) if res["signal"] is not None else "N/A"
            current_price = f"{res['current_price']:.2f}" if isinstance(res['current_price'], (int, float)) else "N/A"
            current_date = res["current_date"]
            target_price = f"{res['target_price']:.2f}" if isinstance(res['target_price'], (int, float)) else "N/A"
            target_date = res["target_date"].strftime('%Y-%m-%d') if hasattr(res["target_date"], 'strftime') else str(res["target_date"] or "N/A")
            train_score = f"{res['train_score']:.4%}"
            val_score = f"{res['val_score']:.4%}"
            train_win_rate = f"{res['train_win_rate']:.4%}"
            val_win_rate = f"{res['val_win_rate']:.4%}"
            optimize = str(res["optimization_metric"])
            method = f'{res["method"]}'
            threshold = f'{res["threshold"]}'
            info = f"{res['ticker']:<8}::{res['dataset_id']:<8}::{res['lookahead']:<3}"
            table_rows.append([info, sig, current_price, current_date, target_price, target_date, train_win_rate, val_win_rate, optimize, method, threshold])

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in table_rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        col_widths = [w + 2 for w in col_widths]
        total_width = sum(col_widths)

        # Formatting helper
        def format_row(cells):
            return "".join(f"{str(cell):<{w}}" for cell, w in zip(cells, col_widths)).rstrip()

        if args.verbose:
            print("\n" + "=" * total_width)
            print(f"{'AUTOTUNE RESULTS SUMMARY':^{total_width}}")
            print("=" * total_width)
            print(format_row(headers))
            print("-" * total_width)
        for row in table_rows:
            if args.verbose: print(format_row(row))
        if args.verbose: print("=" * total_width)
    results = []
    for row in table_rows:
        info, signal, current_price, current_date, target_price, target_date, train_win_rate, val_win_rate, optimization_target, method, threshold = row
        format_date = "%Y-%m-%d"
        results.append({"info": info, "signal": float(signal), "current_price": float(current_price), "current_date": datetime.strptime(current_date, format_date),
                        "target_price": float(target_price), "target_date": datetime.strptime(target_date, format_date), "method": method,
                        "train_win_rate": float(train_win_rate.strip('%')) /100., "val_win_rate": float(val_win_rate.strip('%')) /100., "optimize_target": optimization_target,
                        "threshold": threshold, "method": optimization_target, "app": "AutoTune"})
    return results


# =============================================================================
# 1. MAIN
# =============================================================================
if __name__ == "__main__":
    # When run directly, argparse will automatically handle sys.argv
    entry()