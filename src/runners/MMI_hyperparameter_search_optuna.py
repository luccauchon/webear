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

# Set verbosity to INFO to see trial progress when using storage
# (WARNING hides the optimization log table)
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
    total_n_trials = args.n_trials

    # =========================
    # Optuna Objective Function
    # =========================
    def objective(trial):
        """
        Objective function for Optuna: suggests hyperparameter values and returns backtest accuracy.
        Each trial tests a unique combination of strategy parameters.
        """
        # Suggest hyperparameters within fixed ranges
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
            use_ema=args.use_ema,
            verbose=False,
            use_vix=args.use_vix,
            filter_open_gaps=args.filter_open_gaps,
            filter_inside_open=args.filter_inside_open,
        )

        # Run backtest and return accuracy (to be maximized)
        try:
            metrics, results_df = MMI_backtest(configuration)
            accuracy = metrics[args.metric]
            return accuracy
        except Exception as e:
            # Tell Optuna this trial failed so it doesn't count as a successful value
            print(f"Trial {trial.number} failed with error: {e}")
            raise optuna.exceptions.TrialPruned()

    # =========================
    # Create/Load Study with Storage
    # =========================
    print(f"\n💾 Using Storage: {args.storage}")
    print(f"📛 Study Name: {args.study_name}")
    timeout_str = f"{args.timeout}s" if args.timeout else "None"
    print(f"🚀 Starting Optuna Optimization (Trials: {args.n_trials}, Timeout: {timeout_str})...")
    try:
        study = optuna.create_study(
            direction="maximize",
            storage=args.storage,
            study_name=args.study_name,
            load_if_exists=True,
        )
        if len(study.trials) > 0:
            print(f"⏩ Resuming existing study. Completed trials so far: {len(study.trials)}")
        else:
            print("🆕 Created new study.")
        # =========================================================
        # DB RESUME VERIFICATION
        # =========================================================
        trials = study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("\n" + "=" * 30)
        print("   OPTUNA PERSISTENCE CHECK")
        print("=" * 30)
        print(f"Database: {args.storage}")
        print(f"Total trials in DB:    {len(trials)}")
        print(f"Completed trials:      {len(completed_trials)}")

        if len(completed_trials) > 0:
            print(f"Current Best Score:    {study.best_value:.6f}")
            print(f"Last trial ID:         {trials[-1].number}")
            print(">>> Resuming from existing data...")
        else:
            print(">>> No history found. Starting a new study.")
        print("=" * 30 + "\n")
    except Exception as e:
        print(f"Error creating/loading study: {e}")
        return

    # =========================
    # Calculate Remaining Trials
    # =========================
    current_trials = len(study.trials)
    if current_trials >= total_n_trials:
        print(f"\n✅ Target number of trials ({total_n_trials}) already reached.")
        print("\n===== BEST PARAMETERS =====")
        print(study.best_params)
        print(f"Best Score: {study.best_value:.8f}")
        return

    trials_to_run = total_n_trials - current_trials
    print(f"🚀 Running {trials_to_run} additional trials to reach target of {total_n_trials}...\n")

    # Create a callback to print progress occasionally if needed,
    # but show_progress_bar handles most of it.
    study.optimize(objective, n_trials=trials_to_run, show_progress_bar=True, timeout=args.timeout)

    # Output best results
    print("\n===== BEST PARAMETERS =====")
    print(study.best_params)
    print(f"Best Score: {study.best_value:.8f}")

    # 1. Convert the study to a Pandas DataFrame
    df_trials = study.trials_dataframe()

    # 2. Clean up the columns (removes datetime/duration for a cleaner view)
    # We focus on 'value' (the score) and the 'params_' columns
    param_cols = [c for c in df_trials.columns if c.startswith('params_')]
    summary = df_trials[['value'] + param_cols].sort_values(by='value', ascending=False)

    print("\n" + "=" * 50)
    print("   TOP 10 HYPERPARAMETER COMBINATIONS")
    print("=" * 50)
    print(summary.head(10).to_string(index=False))
    print("=" * 50 + "\n")

    # 3. Quick Statistics: See the 'Spread' of your parameters
    print("Parameter Ranges in Top 20% of Trials:")
    top_20_percent = summary.head(int(len(summary) * 0.2))
    print(top_20_percent.describe().loc[['min', 'max', 'mean']])

    # Optional: Save best params to a file for easy access later
    # with open(f"best_params_{args.study_name}.pkl", "wb") as f:
    #     pickle.dump(study.best_params, f)


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
    parser.add_argument('--step-back-range', type=int, default=15000,
                        help="Number of historical data points to consider during backtest")
    parser.add_argument('--n-trials', type=int, default=1500,
                        help="Total target number of Optuna trials")
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum optimization time in seconds (None = no limit)')

    # === Persistence Args ===
    parser.add_argument('--storage', type=str, default='sqlite:///mmi_optuna_study.db',
                        help="Database URL for Optuna storage (e.g., sqlite:///db.sqlite3)")
    parser.add_argument('--study_name', type=str, default=None,
                        help="Unique name for the study. If None, defaults to ticker_dataset")

    # === Hyperparameter Search Ranges ===
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
    parser.add_argument('--use_ema', type=str2bool, default=False)
    # LOOKAHEAD
    parser.add_argument('--lookahead_min', type=int, default=5, help="Min look-ahead steps")
    parser.add_argument('--lookahead_max', type=int, default=5, help="Max look-ahead steps")

    parser.add_argument('--use_vix', type=str2bool)
    parser.add_argument('--filter-open-gaps', type=str2bool, default=False,
                        help="Remove rows where Open > Prev High or Open < Prev Low")
    parser.add_argument('--filter-inside-open', type=str2bool,
                        help="Compute accuracy only if Current Open is between Precedent Day's Low and High")
    args = parser.parse_args()

    # Auto-generate study name if not provided to avoid collisions
    if args.study_name is None:
        args.study_name = f"study_{args.ticker.replace('^', '')}_{args.dataset_id}"

    main(args)