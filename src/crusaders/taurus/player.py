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

def parse_args():
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )
    return parser.parse_args()


def entry(args):
    results = []
    autotune_target_dir = r"D:\PyCharmProjects\webear\src\optimizers\autotune\models"
    prime_rsi_target_dir = r"D:\PyCharmProjects\webear\src\optimizers\prime_rsi\models"

    args = Namespace(verbose=True, target_dir=autotune_target_dir, clip=False, hide_zero_signal=False)
    results.extend(autotune_player(args))

    args = Namespace(verbose=True, target_dir=prime_rsi_target_dir, clip=False, hide_zero_signal=False)
    results.extend(prime_rsi_player(args))

    # Print results
    headers = ["Info", "Signal", "Current Price", "Current Date", "Target Price", "Target Date", "Train Score", "Val Score", "Optimize Target", "Method", "Threshold", "Indicator"]
    table_rows = []
    for res in results:
        info = res["info"]
        signal = res["signal"]
        current_price = f"{res['current_price']:.2f}"
        current_date = f"{res['current_date'].strftime('%Y-%m-%d')}"
        target_price = f"{res['target_price']:.2f}"
        target_date = f"{res['target_date'].strftime('%Y-%m-%d')}"
        train_score = f"{res['train_score']:.4%}"
        val_score = f"{res['val_score']:.4%}"
        optimize = str(res["optimize_target"])
        method = str(res["method"])
        threshold = str(res["threshold"])
        indicator = str(res["app"])
        table_rows.append([info, signal, current_price, current_date, target_price, target_date, train_score, val_score, optimize, method, threshold, indicator])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            try:
                col_widths[i] = max(col_widths[i], len(cell))
            except:
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