from numba.core.target_extension import target_registry

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

from optimizers.prime_rsi.player import entry as prime_rsi_player
from optimizers.autotune.player import entry as autotune_player
from optimizers.oerh.player import entry as oerh_player


def parse_args():
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )
    parser.add_argument(
        "--autotune-target-dir",
        required=True,
        help="Target directory for autotune models"
    )
    parser.add_argument(
        "--prime-rsi-target-dir",
        required=True,
        help="Target directory for prime_rsi models"
    )
    parser.add_argument(
        "--oerh-target-dir",
        required=True,
        help="Target directory for oerh models"
    )
    parser.add_argument(
        "--hide-zero-signal",
        action="store_true",
        default=False,
        help="Hide rows where signal is 0 (default: False)"
    )
    return parser.parse_args()


def entry(args):
    results = []

    # Use the required parameters passed from the command line
    autotune_target_dir = args.autotune_target_dir
    prime_rsi_target_dir = args.prime_rsi_target_dir
    oerh_target_dir = args.oerh_target_dir

    # Pass hide_zero_signal dynamically from the main args
    autotune_args = Namespace(
        verbose=True,
        target_dir=autotune_target_dir,
        clip=False,
        hide_zero_signal=args.hide_zero_signal
    )
    results.extend(autotune_player(autotune_args))

    prime_rsi_args = Namespace(
        verbose=True,
        target_dir=prime_rsi_target_dir,
        clip=False,
        hide_zero_signal=False
    )
    results.extend(prime_rsi_player(prime_rsi_args))

    oerh_args = Namespace(
        verbose=True,
        target_dir=oerh_target_dir,
        clip=False,
        hide_zero_signal=False
    )
    results.extend(oerh_player(oerh_args))

    # Print results
    headers = ["Info", "Signal", "Current Price", "Current Date", "Target Price", "Target Date", "Train Win Rate", "Val Win Rate", "Optimize Target", "Method", "Threshold", "Indicator"]
    table_rows = []
    for res in results:
        info = res["info"]
        signal = res["signal"]
        current_price = f"{res['current_price']:.2f}"
        current_date = f"{res['current_date'].strftime('%Y-%m-%d')}"
        target_price = f"{res['target_price']:.2f}"
        target_date = f"{res['target_date'].strftime('%Y-%m-%d')}"
        train_win_rate = f"{res['train_win_rate']:.4%}"
        val_win_rate = f"{res['val_win_rate']:.4%}"
        optimize = str(res["optimize_target"])
        method = str(res["method"])
        threshold = str(res["threshold"])
        indicator = str(res["app"])

        if args.hide_zero_signal:
            assert signal in [-1, 0, 1]
            if signal == 0:
                continue

        table_rows.append([info, signal, current_price, current_date, target_price, target_date, train_win_rate, val_win_rate, optimize, method, threshold, indicator])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            try:
                col_widths[i] = max(col_widths[i], len(str(cell)))
            except Exception:
                col_widths[i] = max(col_widths[i], len(f"{cell}"))

    col_widths = [w + 2 for w in col_widths]
    total_width = sum(col_widths)

    # Formatting helper
    def format_row(cells):
        return "".join(f"{str(cell):<{w}}" for cell, w in zip(cells, col_widths)).rstrip()

    print("\n" + "=" * total_width)
    print(f"{'ALL INDICATORS RESULTS SUMMARY':^{total_width}}")
    print("=" * total_width)
    print(format_row(headers))
    print("-" * total_width)
    for row in table_rows:
        print(format_row(row))
    print("=" * total_width)


if __name__ == "__main__":
    args = parse_args()
    entry(args)