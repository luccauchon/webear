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
import traceback
import numpy as np
import argparse
import pathlib
from argparse import Namespace
from datetime import datetime
from optimizers.prime_rsi.realtime_and_backtest_hyperparameter_search_optuna import entry as prime_rsi


def parse_args():
    parser = argparse.ArgumentParser(
        prog="prime_rsi_runner",
        description="Run Prime RSI on multiple .pkl model files and display a results summary."
    )

    # Create a mutually exclusive group for target directory vs target files
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "-d", "--target-dir",
        type=str,
        help="Directory containing .pkl model files (default: ./models)"
    )
    target_group.add_argument(
        "-f", "--target-files",
        type=str,
        nargs='+',
        help="List of specific .pkl model files to process"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output during processing"
    )
    parser.add_argument(
        "--hide-zero-signal",
        action="store_true",
        default=False,
        help="Hide rows where signal is 0 (default: False)"
    )
    parser.add_argument("--clip", action="store_true", help="Exclude incomplete current bar in real-time")
    return parser.parse_args()


def entry(args):
    verbose = args.verbose

    files = []
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
        if args.target_files:
            if verbose: print("⚠️ No valid .pkl files found among the specified target files.")
        else:
            target_dir = pathlib.Path(args.target_dir or "./models").resolve()
            if verbose: print(f"⚠️ No .pkl files found in {target_dir}")
        return results

    # Parse all files
    for file_path in files:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"STARTING PRIME RSI FOR: {file_path.name}")
            print(f"{'=' * 60}")

        configuration = Namespace(
            real_time=True,
            seed=123,
            model_path=str(file_path),
            verbose=verbose,
            verbose_short=verbose,
            clip=args.clip,
            dataset_id=None,
            ticker="^GSPC",
            length_dataset=999999,
        )
        try:
            result = prime_rsi(configuration)
            train_score, val_score = result['train_score'], result['val_score']
            buy_signal_detected = result['buy_signal_detected']
            sell_signal_detected = result['sell_signal_detected']
            optimize_target = result['optimize_target']
            signal = 1 if buy_signal_detected else (-1 if sell_signal_detected else 0)
            current_price = result['current_price']
            target_price = result['target_price']
            target_date = result['target_date']
            put_strike, call_strike = result['put_strike_pct'], result['call_strike_pct']
            results.append({
                "file": file_path.name,
                "signal": signal,
                "current_price": current_price,
                "current_date": result['current_date'],
                "target_price": target_price,
                "target_date": target_date,
                "train_score": train_score,
                "val_score": val_score,
                "train_win_rate": result['train_win_rate'],
                "val_win_rate": result['val_win_rate'],
                "optimize_target": optimize_target,
                "dataset_id": result['dataset_id'],
                "ticker": result['ticker'],
                "lookahead": result['lookahead'],
                "method": result['method'],
                "put_threshold": put_strike,
                "call_threshold": call_strike,
            })
        except Exception as e:
            if verbose:
                print(f"❌ ERROR processing {file_path.name}: {e}")
                traceback.print_exc()

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
        train_win_rate = f"{res['train_win_rate']:.4%}"
        val_win_rate = f"{res['val_win_rate']:.4%}"
        optimize = str(res["optimize_target"])
        assert optimize in ["buy_wr", "sell_wr"]
        method = str(res["method"])
        info = f"{res['ticker']:<8}::{res['dataset_id']:<8}::{res['lookahead']:<3}"
        threshold = f"{res['put_threshold']}"
        if optimize == "sell_wr":
            threshold = f"{res['call_threshold']}"
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

    if verbose:
        print("\n" + "=" * total_width)
        print(f"{'PRIME RSI RESULTS SUMMARY':^{total_width}}")
        print("=" * total_width)
        print(format_row(headers))
    if verbose: print("-" * total_width)
    for row in table_rows:
        if verbose: print(format_row(row))
    if verbose:
        print("=" * total_width)
    results = []
    for row in table_rows:
        info, signal, current_price, current_date, target_price, target_date, train_win_rate, val_win_rate, optimization_target, method, thresholds = row
        format_date = "%Y-%m-%d"
        put_threshold, call_threshold = thresholds, thresholds
        results.append({"info": info, "signal": float(signal), "current_price": float(current_price), "current_date": datetime.strptime(current_date, format_date),
                        "target_price": float(0) if target_price == "N/A" else float(target_price),
                        "target_date": datetime.strptime(target_date, format_date), "train_win_rate": float(train_win_rate.strip('%')) / 100.,
                        "val_win_rate": float(val_win_rate.strip('%')) / 100., "optimize_target": optimization_target,
                        "threshold": f"{put_threshold}::{call_threshold}", "method": method, "app": "Prime RSI"})
    return results


if __name__ == "__main__":
    args = parse_args()
    entry(args)