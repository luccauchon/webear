# Local custom modules
try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

from multiprocessing import freeze_support
from argparse import Namespace
import pickle
import numpy as np
import argparse
from utils import DATASET_AVAILABLE, str2bool, get_filename_for_dataset
import copy

# Third-party optimization library
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress specific numpy warnings that are non-critical for this application
import warnings

warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

from MMI_backtest import main as MMI_backtest


def main(args):
    """
    Main function to run Bayesian optimization using Optuna on stock backtesting parameters.

    Prints all arguments for transparency, then defines and runs an Optuna study
    to maximize backtest accuracy by tuning key trading strategy hyperparameters.
    """
    print("🔧 Arguments:")
    for arg, value in vars(args).items():
        print(f"    {arg:.<40} {value}")
    print("-" * 80, flush=True)

    # Extract frequently used args for clarity
    dataset_id = args.dataset_id
    step_back_range = args.step_back_range
    n_trials = args.n_trials

    # =========================
    # Optuna Objective Function
    # =========================
    def objective(trial):
        """
        Objective function for Optuna: suggests hyperparameter values and returns backtest accuracy.
        Each trial tests a unique combination of strategy parameters.
        """
        # Suggest hyperparameters within fixed ranges (you can make these dynamic via args if needed)
        LOOKAHEAD = trial.suggest_int("LOOKAHEAD", args.lookahead_min, args.lookahead_max)
        RETURN_THRESHOLD = trial.suggest_float("RETURN_THRESHOLD", args.return_threshold_min, args.return_threshold_max)
        MMI_TREND_MAX = trial.suggest_int("MMI_TREND_MAX", args.mmi_trend_max_min, args.mmi_trend_max_max)
        MMI_PERIOD = trial.suggest_int("MMI_PERIOD", args.mmi_period_min, args.mmi_period_max)
        SMA_PERIOD = trial.suggest_int("SMA_PERIOD", args.sma_period_min, args.sma_period_max)

        # Build configuration namespace for the backtest function
        configuration = Namespace(
            ticker=args.ticker,
            col=args.col,
            dataset_id=dataset_id,
            look_ahead=LOOKAHEAD,
            mmi_period=MMI_PERIOD,
            mmi_trend_max=MMI_TREND_MAX,
            return_threshold=RETURN_THRESHOLD,
            sma_period=SMA_PERIOD,
            step_back_range=step_back_range,
            use_ema=False,
            verbose=False,
        )

        # Run backtest and return accuracy (to be maximized)
        metrics, results_df = MMI_backtest(configuration)
        accuracy = metrics[args.metric]
        return accuracy

    # Create and run the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Output best results
    print("\n===== BEST PARAMETERS =====")
    print(study.best_params)
    print(f"Best Score: {study.best_value:.8f}")


if __name__ == "__main__":
    freeze_support()

    # Argument parser with detailed help messages and defaults
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization on Wavelet-based stock backtest using Optuna."
    )

    # Core dataset & symbol args
    parser.add_argument('--ticker', type=str, default='^GSPC', help="Stock/index ticker symbol (e.g., AAPL, ^GSPC)")
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                        help="Price column to use from OHLCV data")
    parser.add_argument('--dataset_id', type=str, default='day', choices=DATASET_AVAILABLE,
                        help="Identifier for dataset frequency (e.g., 'day', 'hour')")

    # Optimization control
    parser.add_argument('--step_back_range', type=int, default=15000,
                        help="Number of historical data points to consider during backtest")
    parser.add_argument('--n_trials', type=int, default=1500,
                        help="Number of Optuna trials (hyperparameter combinations to test)")

    # === New Hyperparameter Search Ranges ===
    # RETURN_THRESHOLD: min/max in decimal (e.g., 0.01 = 1%)
    parser.add_argument('--return_threshold_min', type=float, default=0.01,
                        help="Minimum return threshold (as decimal, e.g., 0.01 for 1pourcent)")
    parser.add_argument('--return_threshold_max', type=float, default=0.02,
                        help="Maximum return threshold (as decimal)")

    # MMI_TREND_MAX: integer range
    parser.add_argument('--mmi_trend_max_min', type=int, default=1,
                        help="Minimum value for MMI_TREND_MAX (trend sensitivity)")
    parser.add_argument('--mmi_trend_max_max', type=int, default=500,
                        help="Maximum value for MMI_TREND_MAX")

    # MMI_PERIOD: lookback window for market meanness index
    parser.add_argument('--mmi_period_min', type=int, default=1,
                        help="Minimum MMI period (in data points)")
    parser.add_argument('--mmi_period_max', type=int, default=500,
                        help="Maximum MMI period")

    # SMA_PERIOD: simple moving average window
    parser.add_argument('--sma_period_min', type=int, default=1,
                        help="Minimum SMA period")
    parser.add_argument('--sma_period_max', type=int, default=500,
                        help="Maximum SMA period")

    parser.add_argument('--metric', type=str, default='overall_accuracy',
                        choices=['overall_accuracy', 'bull_accuracy', 'bear_accuracy'],
                        help="Metric to optimize during Bayesian optimization")

    # LOOKAHEAD
    parser.add_argument('--lookahead_min', type=int, default=5, help="Min look-ahead steps")
    parser.add_argument('--lookahead_max', type=int, default=5, help="Max look-ahead steps")

    args = parser.parse_args()
    main(args)