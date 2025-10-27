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
from algorithms.aligned_trend import EMATrendAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze EMA trend for a given stock.")
    parser.add_argument(
        "--stock",
        type=str,
        required=True,
        help="Ticker symbol of the stock to analyze (e.g., AMD, AAPL)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    analyzer = EMATrendAnalyzer(verbose=args.verbose)
    analyzer.analyze(ticker_name=args.stock)


# Example usage
if __name__ == "__main__":
    main()