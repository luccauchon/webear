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
    return parser.parse_args()


def entry(args):
    verbose = args.verbose
    target_dir = pathlib.Path(args.target_dir).resolve()
    files = sorted([
        f for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".pkl"
    ])

    if not files:
        print(f"⚠️ No .pkl files found in {target_dir}")
        return

    results = []

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
        )
        try:
            signal, current_price, target_price, target_date = oerh(configuration)
            results.append({
                "file": file_path.name,
                "signal": signal,
                "current_price": current_price,
                "target_price": target_price,
                "target_date": target_date,
                "status": "SUCCESS"
            })
        except Exception as e:
            print(f"❌ ERROR processing {file_path.name}: {e}")
            results.append({
                "file": file_path.name,
                "signal": None,
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
    headers = ["Model File", "Signal", "Current Price", "Target Price", "Target Date"]
    table_rows = []
    for res in results:
        sig = str(res["signal"]) if res["signal"] is not None else "N/A"
        curr = f"{res['current_price']:.2f}" if isinstance(res['current_price'], (int, float)) else "N/A"
        targ = f"{res['target_price']:.2f}" if isinstance(res['target_price'], (int, float)) else "N/A"
        date = res["target_date"].strftime('%Y-%m-%d') if hasattr(res["target_date"], 'strftime') else str(res["target_date"] or "N/A")
        table_rows.append([res['file'], sig, curr, targ, date])

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
        print(f"{'OERH RESULTS SUMMARY':^{total_width}}")
        print("=" * total_width)
    print(format_row(headers))
    if verbose:
        print("-" * total_width)
    for row in table_rows:
        print(format_row(row))
    if verbose:
        print("=" * total_width)

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    error_count = len(results) - success_count
    if verbose:
        print(f"✅ Processed {len(results)} model(s) | {success_count} success | {error_count} error(s)")


if __name__ == "__main__":
    args = parse_args()
    entry(args)