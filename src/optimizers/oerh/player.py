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
from datetime import datetime
import argparse
import pathlib
from argparse import Namespace

from optimizers.oerh.realtime_and_backtest import entry as oerh


def parse_args():
    parser = argparse.ArgumentParser(
        prog="oerh_runner",
        description="Run OERH on multiple .pkl model files and display a results summary."
    )
    parser.add_argument(
        "-d", "--target-dir",
        type=str,
        default="./models",
        help="Directory containing .pkl model files (default: ./models)"
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
    target_dir = pathlib.Path(args.target_dir).resolve()
    try:
        files = sorted([
            f for f in target_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".pkl"
        ])
    except:
        files = []

    results = []

    if not files:
        print(f"⚠️ No .pkl files found in {target_dir}")
        return results

    # Parse all files
    for file_path in files:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"STARTING OERH FOR: {file_path.name}")
            print(f"{'=' * 60}")

        configuration = Namespace(
            real_time=True,
            model_path=str(file_path),
            output_signal_only=False,
            verbose=verbose,
            validate_jit=False,
            clip=args.clip,
        )
        try:
            signal, current_price, current_date, target_price, target_date, train_acc, val_acc, threshold_pct, metric_used_and_target_type, dataset_id, ticker, lookahead_bars = oerh(configuration)
            results.append({
                "file": file_path.name,
                "signal": signal,
                "current_price": current_price,
                "current_date": current_date,
                "target_price": target_price,
                "target_date": target_date,
                'train_win_rate': train_acc,
                'val_win_rate': val_acc,
                'threshold_pct': threshold_pct,
                'metric_used_and_target_type': metric_used_and_target_type,
                'dataset_id': dataset_id,
                'ticker': ticker,
                'lookahead_bars': lookahead_bars,
            })
        except Exception as e:
            print(f"❌ ERROR processing {file_path.name}: {e}")
            results.append({
                "file": file_path.name,
                "ticker": "---", "dataset_id": "---", "lookahead_bars": "---", "train_win_rate": 0., "val_win_rate": 0.,
                "signal": None, "threshold_pct": 0., "current_date": "---", "metric_used_and_target_type": "---::---",
                "current_price": None,
                "target_price": None,
                "target_date": None,
                "status": f"ERROR: {e}"
            })

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
    headers = ["Info", "Signal", "Current Price", "Current Date", "Target Price", "Target Date", "Train Win Rate", "Val Win Rate", "Threshold Pct", "Metric Optimized::Target Type"]
    table_rows = []
    for res in results:
        info = f"{res['ticker']:<8}::{res['dataset_id']:<8}::{res['lookahead_bars']:<3}"
        sig = str(res["signal"]) if res["signal"] is not None else "N/A"
        current_price = f"{res['current_price']:.2f}" if isinstance(res['current_price'], (int, float)) else "N/A"
        target_price = f"{res['target_price']:.2f}" if isinstance(res['target_price'], (int, float)) else "N/A"
        target_date = res["target_date"].strftime('%Y-%m-%d') if hasattr(res["target_date"], 'strftime') else str(res["target_date"] or "N/A")
        train_acc = f"{res['train_win_rate']:.2%}"
        val_acc = f"{res['val_win_rate']:.2%}"
        threshold = f"{res['threshold_pct']:.2%}"
        table_rows.append([info, sig, current_price, res["current_date"], target_price, target_date, train_acc, val_acc, threshold, res["metric_used_and_target_type"]])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    col_widths = [w + 2 for w in col_widths]
    total_width = sum(col_widths)

    # Formatting helper
    def format_row(cells):
        return "".join(f"{str(cell):<{w}}" for cell, w in zip(cells, col_widths)).rstrip()
    if verbose:
        print("\n" + "=" * total_width)
        print(f"{'OERH RESULTS SUMMARY':^{total_width}}")
        print("=" * total_width)
    print(format_row(headers))
    if verbose:
        print("-" * total_width)
    for row in table_rows:
        print(format_row(row))
    if verbose:
        print("=" * total_width)
    results = []
    for row in table_rows:
        info, signal, current_price, current_date, target_price, target_date, train_accuracy, val_accuracy, threshold_pct, metric__optimize = row
        format_date = "%Y-%m-%d"
        metric_optimized = metric__optimize.split("::")[0]
        target_type = metric__optimize.split("::")[1]
        results.append({"info": info, "signal": float(signal), "current_price": float(current_price), "current_date": datetime.strptime(current_date, format_date),
                        "target_price": float(0) if target_price == "N/A" else float(target_price),
                        "target_date": datetime.strptime(target_date, format_date), "train_win_rate": float(train_accuracy.strip('%')) / 100.,
                        "val_win_rate": float(val_accuracy.strip('%')) / 100., "optimize_target": metric_optimized,
                        "threshold": f"{threshold_pct}::{threshold_pct}", "method": target_type, "app": "OERH"})
    return results


if __name__ == "__main__":
    args = parse_args()
    entry(args)