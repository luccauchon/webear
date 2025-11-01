try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # print(parent_dir)
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
from constants import TOP_SP500_TICKERS
from runners.aligned_trend import main as aligned_trend_entry_point


def main(args):
    aligned_trend_entry_point(args)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze EMA trend for a given stock.")
    parser.add_argument(
        "--stock",
        type=str,
        required=False,
        help="Ticker symbol of the stock to analyze (e.g., AMD, AAPL)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()
    for ticker in TOP_SP500_TICKERS:
        args.stock = ticker
        main(args)