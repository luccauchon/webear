import argparse
import pathlib
from argparse import Namespace

try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

from optimizers.autotune.realtime_and_backtest_hyperparameter_search_optuna import entry as autotune


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
        "-d", "--directory",
        type=str,
        default=".",
        metavar="PATH",
        help=(
            "Path to the target directory containing files to process. "
            "Defaults to the current working directory ('.')."
        )
    )

    parser.add_argument(
        "-e", "--extension",
        type=str,
        default=".pkl",
        metavar="EXT",
        help=(
            "File extension filter (e.g., '.pkl', '.json'). "
            "Only files matching this extension will be processed. "
            "Defaults to '.pkl'. The leading dot is optional."
        )
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        default="day",
        metavar="ID",
        help=(
            "Identifier for the dataset timeframe or type (e.g., 'day', 'hour', 'min'). "
            "Passed to the autotune configuration for each file. Defaults to 'day'."
        )
    )

    parser.add_argument(
        "-t", "--ticker",
        type=str,
        default="^GSPC",
        metavar="SYMBOL",
        help=(
            "Stock/asset ticker symbol to use during autotuning (e.g., '^GSPC', 'AAPL', 'BTC-USD'). "
            "Defaults to '^GSPC'."
        )
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help=(
            "Enable verbose output mode. Displays file counts, "
            "processing status, and detailed run information."
        )
    )

    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {sys__version}",
        help="Show program's version number and exit."
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

    # Support both argparse.Namespace and dict for backward compatibility
    if isinstance(args, dict):
        target_dir   = args.get('directory', '.')
        extension    = args.get('extension', '.pkl')
        dataset_id   = args.get('dataset_id', 'day')
        ticker       = args.get('ticker', '^GSPC')
        verbose      = args.get('verbose', False)
    else:
        target_dir   = getattr(args, 'directory', '.')
        extension    = getattr(args, 'extension', '.pkl')
        dataset_id   = getattr(args, 'dataset_id', 'day')
        ticker       = getattr(args, 'ticker', '^GSPC')
        verbose      = getattr(args, 'verbose', False)

    # Normalize extension to ensure it starts with a dot
    extension = extension.lower() if extension.startswith('.') else f".{extension.lower()}"

    dir_path = pathlib.Path(target_dir).resolve()

    if not dir_path.is_dir():
        raise FileNotFoundError(f"Target directory not found: {dir_path}")

    # Filter files by the specified extension
    files = sorted([
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() == extension
    ])

    if not files:
        print(f"No {extension} files found in {dir_path}")
        return

    if verbose:
        print(f"Found {len(files)} {extension} file(s) in {dir_path}. Starting sequential autotune...\n")

    # Loop through files one by one
    for file_path in files:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"STARTING AUTOTUNE FOR: {file_path.name}")
            print(f"{'=' * 60}")

        # Build configuration for the current file using CLI arguments
        configuration = Namespace(
            real_time=True,
            model_path=str(file_path),
            verbose_short=True,
            dataset_id=dataset_id,
            ticker=ticker,
            verbose=verbose,
            output_dir=str(dir_path),
            length_dataset=999999,
        )

        try:
            autotune(configuration)
        except Exception as e:
            print(f"❌ ERROR processing {file_path.name}: {e}")


# =============================================================================
# 1. MAIN
# =============================================================================
if __name__ == "__main__":
    # When run directly, argparse will automatically handle sys.argv
    entry()