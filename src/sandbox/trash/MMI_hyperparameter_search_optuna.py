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
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
from MMI_backtest import main as MMI_backtest


def main(args):
    print("🔧 Arguments:")
    for arg, value in vars(args).items():
        print(f"    {arg:.<40} {value}")
    print("-" * 80, flush=True)
    dataset_id = args.dataset_id
    step_back_range = args.step_back_range
    n_trials = args.n_trials

    # =========================
    # Optuna Objective
    # =========================
    def objective(trial):
        LOOKAHEAD = trial.suggest_int("LOOKAHEAD", 5, 5)
        RETURN_THRESHOLD = trial.suggest_float("RETURN_THRESHOLD", 1.0 / 100., 2. / 100.)
        MMI_TREND_MAX = trial.suggest_int("MMI_TREND_MAX", 1, 99)
        MMI_PERIOD = trial.suggest_int("MMI_PERIOD", 5, 300)
        SMA_PERIOD = trial.suggest_int("SMA_PERIOD", 2, 400)

        configuration = Namespace(
            ticker=args.ticker, col=args.col,
            dataset_id=dataset_id,
            look_ahead=LOOKAHEAD,
            mmi_period=MMI_PERIOD,
            mmi_trend_max=MMI_TREND_MAX,
            return_threshold=RETURN_THRESHOLD,
            sma_period=SMA_PERIOD,
            step_back_range=step_back_range,
            verbose=False,
        )

        accuracy, results_df = MMI_backtest(configuration)
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n===== BEST PARAMETERS =====")
    print(study.best_params)
    print(f"Best Score: {study.best_value:.8f}")


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="Run Bayesian optimization on Wavelet-based stock backtest using Optuna.")

    parser.add_argument('--ticker', type=str, default='^GSPC')
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    parser.add_argument('--dataset_id', type=str, default='day',
                        choices=DATASET_AVAILABLE)
    parser.add_argument('--step_back_range', type=int, default=15000)
    parser.add_argument('--n_trials', type=int, default=1500)

    args = parser.parse_args()
    main(args)