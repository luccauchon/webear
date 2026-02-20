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
import numpy as np
import argparse
from utils import DATASET_AVAILABLE, str2bool
from runners.VIX_realtime_and_backtest import main as VVIX_realtime_and_backtest
import warnings

warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

# --- Optuna Integration ---
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Warning: optuna not installed. Run 'pip install optuna' to enable parameter search.")


def create_configuration(args, trial=None):
    """
    Creates the configuration Namespace.
    If trial is provided, uses Optuna suggestions.
    If trial is None, uses args/hardcoded defaults.
    """

    # Helper for Optuna suggestions
    if trial:
        def suggest_bool(name, default):
            return trial.suggest_categorical(name, [True, False])

        def suggest_int(name, low, high, default):
            return trial.suggest_int(name, low, high)

        def suggest_float(name, low, high, default, step=None):
            return trial.suggest_float(name, low, high, step=step)
    else:
        # Fallback to hardcoded defaults from original script if not optimizing
        def suggest_bool(name, default):
            return default

        def suggest_int(name, low, high, default):
            return default

        def suggest_float(name, low, high, default, step=None):
            return default

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False,
        put=True,
        call=True,
        iron_condor=False,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=1.,
        lower_side_scale_factor=1.,
    )
    call_low_fx, call_high_fx, call_default_fx = 1.001, 1.01, 1.001
    put_low_fx,  put_high_fx,  put_default_fx  = 0.99, 0.999, 0.999
    # --- EMA ---
    configuration.adj_call__ema = suggest_bool("adj_call__ema", False)
    configuration.adj_put__ema = suggest_bool("adj_put__ema", False)
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    if configuration.adj_call__ema or configuration.adj_put__ema:
        configuration.ema_short = suggest_int("ema_short", 5, 50, 21)
        configuration.ema_long = suggest_int("ema_long", 50, 200, 50)
        configuration.adj_call__ema_factor = suggest_float("adj_call__ema_factor", call_low_fx, call_high_fx, call_default_fx)
        configuration.adj_put__ema_factor = suggest_float("adj_put__ema_factor", put_low_fx, put_high_fx, put_default_fx)
    else:
        configuration.ema_short = 21
        configuration.ema_long = 50
        configuration.adj_call__ema_factor = 1.
        configuration.adj_put__ema_factor = 1.

    # --- SMA ---
    configuration.adj_call__sma = suggest_bool("adj_call__sma", False)
    configuration.adj_put__sma = suggest_bool("adj_put__sma", False)
    if configuration.adj_call__sma or configuration.adj_put__sma:
        configuration.sma_period = suggest_int("sma_period", 10, 100, 50)
        configuration.adj_call__sma_factor = suggest_float("adj_call__sma_factor", call_low_fx, call_high_fx, call_default_fx)
        configuration.adj_put__sma_factor = suggest_float("adj_put__sma_factor", put_low_fx, put_high_fx, put_default_fx)
    else:
        configuration.sma_period = 50
        configuration.adj_call__sma_factor = 1.
        configuration.adj_put__sma_factor = 1.

    # --- RSI ---
    configuration.adj_call__rsi = suggest_bool("adj_call__rsi", False)
    configuration.adj_put__rsi = suggest_bool("adj_put__rsi", False)
    if configuration.adj_call__rsi or configuration.adj_put__rsi:
        configuration.rsi_period = suggest_int("rsi_period", 5, 30, 14)
        configuration.adj_call__rsi_factor = suggest_float("adj_call__rsi_factor", call_low_fx, call_high_fx, call_default_fx)
        configuration.adj_put__rsi_factor = suggest_float("adj_put__rsi_factor", put_low_fx, put_high_fx, put_default_fx)
    else:
        configuration.rsi_period = 14
        configuration.adj_call__rsi_factor = 1.
        configuration.adj_put__rsi_factor = 1.

    # --- MACD ---
    configuration.adj_call__macd = suggest_bool("adj_call__macd", False)
    configuration.adj_put__macd = suggest_bool("adj_put__macd", False)
    if configuration.adj_call__macd or configuration.adj_put__macd:
        configuration.macd_fast_period = suggest_int("macd_fast_period", 5, 20, 12)
        configuration.macd_slow_period = suggest_int("macd_slow_period", 20, 50, 26)
        configuration.macd_signal_period = suggest_int("macd_signal_period", 5, 15, 9)
        configuration.adj_call__macd_factor = suggest_float("adj_call__macd_factor", call_low_fx, call_high_fx, call_default_fx)
        configuration.adj_put__macd_factor = suggest_float("adj_put__macd_factor", put_low_fx, put_high_fx, put_default_fx)
    else:
        configuration.macd_fast_period = 12
        configuration.macd_slow_period = 26
        configuration.macd_signal_period = 9
        configuration.adj_call__macd_factor = 1.
        configuration.adj_put__macd_factor = 1.

    # --- Contango ---
    configuration.adj_call_and_put__contango = suggest_bool("adj_call_and_put__contango", False)
    if configuration.adj_call_and_put__contango:
        configuration.adj_call_and_put__contango_factor = suggest_float("adj_call_and_put__contango_factor", 0.01, 0.02, 0.02)
    else:
        configuration.adj_call_and_put__contango_factor = 0.02

    return configuration


def objective(trial, args):
    """Optuna objective function."""
    # 1. Create configuration based on trial suggestions
    configuration = create_configuration(args, trial=trial)

    # 2. Run Backtest
    # We suppress verbose output during optimization to keep logs clean
    try:
        results = VVIX_realtime_and_backtest(configuration)
    except Exception as e:
        # If a specific parameter combo crashes the backtest, return a worst score
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

    # 3. Extract Scores
    put_score = results['put']['success_rate__vix999']
    call_score = results['call']['success_rate__vix999']

    # 4. Determine Target Score
    if args.optimize_target == 'put':
        score = put_score
    elif args.optimize_target == 'call':
        score = call_score
    elif args.optimize_target == 'average':
        score = (put_score + call_score) / 2.0
    else:
        score = put_score  # Fallback

    # 5. Report intermediate values if needed (optional, depends on VVIX implementation)
    # trial.report(score, step=...)

    return score


def main(args):
    if args.verbose:
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)

    if not OPTUNA_AVAILABLE:
        args.use_optuna = False
        print("⚠️  Optuna not available. Running single backtest with default parameters.")

    if args.use_optuna:
        print(f"🚀 Starting Optuna Optimization (Target: {args.optimize_target}, Trials: {args.n_trials})...")

        # Create Study
        study = optuna.create_study(direction="maximize", study_name="VIX_Strategy_Optimization")

        # Run Optimization
        # n_jobs=1 is recommended for financial backtests to avoid DB connection issues or race conditions
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, n_jobs=1, show_progress_bar=True)

        # Print Best Results
        print("\n" + "=" * 80)
        print("🏆 Optimization Finished!")
        print(f"Best Score ({args.optimize_target}): {study.best_value:.4f}")
        print("Best Parameters:")
        for key, value in study.best_params.items():
            print(f"    {key:.<40} {value}")
        print("=" * 80 + "\n")

        # Optionally run one final backtest with best params to show full verbose output
        if args.verbose:
            print("Running final backtest with best parameters...")
            best_config = create_configuration(args, trial=None)
            # We need to manually set the best params to the config or create a dummy trial
            # Easier: Just print the params above.
            # To actually run VVIX with best params, we'd need to map study.best_params back to Namespace.
            # For now, we rely on the printed parameters.

    else:
        # --- Original Single Run Logic ---
        configuration = create_configuration(args, trial=None)

        if args.verbose:
            print("⚙️  Configuration:")
            for arg, value in vars(configuration).items():
                print(f"    {arg:.<40} {value}")
            print("-" * 80, flush=True)

        results = VVIX_realtime_and_backtest(configuration)
        put_score = results['put']['success_rate__vix999']
        call_score = results['call']['success_rate__vix999']
        print(f"Put Score: {put_score:0.4f}  Call Score: {call_score:0.4f}")


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="VIX Strategy Backtest & Optimization")

    # --- Existing Args ---
    parser.add_argument('--ticker', type=str, default='^GSPC')
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    parser.add_argument('--look_ahead', type=int, default=1)
    parser.add_argument('--step_back_range', type=int, default=99999)
    parser.add_argument('--verbose', type=str2bool, default=True)

    # --- Optuna Args ---
    parser.add_argument('--use_optuna', type=str2bool, default=True,
                        help='Enable Optuna parameter search')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials for Optuna (ignored if use_optuna=False)')
    parser.add_argument('--optimize_target', type=str, default='put',
                        choices=['put', 'call', 'average'],
                        help='Which score to maximize')

    args = parser.parse_args()
    main(args)