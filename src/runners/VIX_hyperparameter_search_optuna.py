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
import sys
import numpy as np
import argparse
from utils import DATASET_AVAILABLE, str2bool
from constants import IS_RUNNING_ON_CASIR
from runners.VIX_realtime_and_backtest import main as VVIX_realtime_and_backtest
from joblib import parallel_backend
import warnings
import traceback
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


def objective(trial, configuration_specified, args):
    """Optuna objective function."""
    # 1. Create configuration based on trial suggestions
    configuration = configuration_specified(args, trial=trial)

    # 2. Run Backtest
    # We suppress verbose output during optimization to keep logs clean
    try:
        results = VVIX_realtime_and_backtest(configuration)
    except Exception as e:
        # If a specific parameter combo crashes the backtest, return a worst score
        print(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        return 0.0

    # 3. Extract Scores
    put_score         = results['put']['success_rate__vix999']
    call_score        = results['call']['success_rate__vix999']
    iron_condor_score = results['iron_condor']['success_rate__vix999']
    # 4. Determine Target Score
    if args.optimize_target == 'put':
        score = put_score
    elif args.optimize_target == 'call':
        score = call_score
    elif args.optimize_target == 'average':
        score = (put_score + call_score) / 2.0
    elif args.optimize_target == 'iron_condor':
        score = iron_condor_score
    else:
        score = put_score  # Fallback

    # 5. Report intermediate values if needed (optional, depends on VVIX implementation)
    # trial.report(score, step=...)
    if args.objective_name in ["2026_02_20__0_0pct"]:
        print(f"\nBaseline executed. Score: {score}")
        sys.exit(0)
    return score


def create_configuration___2026_02_20__1pct(args, trial):
    """
    Creates the configuration Namespace.
    If trial is provided, uses Optuna suggestions.
    If trial is None, uses args/hardcoded defaults.
    """

    def suggest_int(name, low, high, default):
        return trial.suggest_int(name, low, high)

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False, verbose_results=False, verbose_arguments=False,
        put=True,
        call=True,
        iron_condor=True,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=args.upper_side_scale_factor,
        lower_side_scale_factor=args.lower_side_scale_factor,
        sharpen_x_side_scale_factor=args.sharpen_x_side_scale_factor,
        adj_balanced=False,
    )

    # --- EMA ---
    configuration.adj_call__ema = True
    configuration.adj_put__ema = True
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    configuration.ema_short = suggest_int("ema_short", 5, 50, 21)
    configuration.ema_long = suggest_int("ema_long", 50, 200, 50)
    configuration.adj_call__ema_factor = 1.01
    configuration.adj_put__ema_factor = 0.99

    # --- SMA ---
    configuration.adj_call__sma = True
    configuration.adj_put__sma = True
    configuration.sma_period = suggest_int("sma_period", 10, 100, 50)
    configuration.adj_call__sma_factor = 1.01
    configuration.adj_put__sma_factor = 0.99

    # --- RSI ---
    configuration.adj_call__rsi = True
    configuration.adj_put__rsi = True
    configuration.rsi_period = suggest_int("rsi_period", 5, 30, 14)
    configuration.adj_call__rsi_factor = 1.01
    configuration.adj_put__rsi_factor = 0.99

    # --- MACD ---
    configuration.adj_call__macd = True
    configuration.adj_put__macd = True
    configuration.macd_fast_period = suggest_int("macd_fast_period", 5, 20, 12)
    configuration.macd_slow_period = suggest_int("macd_slow_period", 20, 50, 26)
    configuration.macd_signal_period = suggest_int("macd_signal_period", 5, 15, 9)
    configuration.adj_call__macd_factor = 1.01
    configuration.adj_put__macd_factor = 0.99

    # --- Contango ---
    configuration.adj_call_and_put__contango = True
    configuration.adj_call_and_put__contango_factor = 0.01

    return configuration


def create_configuration___2026_02_20__2pct(args, trial):
    """
    """

    def suggest_int(name, low, high, default):
        return trial.suggest_int(name, low, high)

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False, verbose_results=False, verbose_arguments=False,
        put=True,
        call=True,
        iron_condor=True,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=args.upper_side_scale_factor,
        lower_side_scale_factor=args.lower_side_scale_factor,
        sharpen_x_side_scale_factor=args.sharpen_x_side_scale_factor,
        adj_balanced=False,
    )

    # --- EMA ---
    configuration.adj_call__ema = True
    configuration.adj_put__ema = True
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    configuration.ema_short = suggest_int("ema_short", 5, 50, 21)
    configuration.ema_long = suggest_int("ema_long", 50, 200, 50)
    configuration.adj_call__ema_factor = 1.02
    configuration.adj_put__ema_factor = 0.98

    # --- SMA ---
    configuration.adj_call__sma = True
    configuration.adj_put__sma = True
    configuration.sma_period = suggest_int("sma_period", 10, 100, 50)
    configuration.adj_call__sma_factor = 1.02
    configuration.adj_put__sma_factor = 0.98

    # --- RSI ---
    configuration.adj_call__rsi = True
    configuration.adj_put__rsi = True
    configuration.rsi_period = suggest_int("rsi_period", 5, 30, 14)
    configuration.adj_call__rsi_factor = 1.02
    configuration.adj_put__rsi_factor = 0.98

    # --- MACD ---
    configuration.adj_call__macd = True
    configuration.adj_put__macd = True
    configuration.macd_fast_period = suggest_int("macd_fast_period", 5, 20, 12)
    configuration.macd_slow_period = suggest_int("macd_slow_period", 20, 50, 26)
    configuration.macd_signal_period = suggest_int("macd_signal_period", 5, 15, 9)
    configuration.adj_call__macd_factor = 1.02
    configuration.adj_put__macd_factor = 0.98

    # --- Contango ---
    configuration.adj_call_and_put__contango = True
    configuration.adj_call_and_put__contango_factor = 0.02

    return configuration


def create_configuration___2026_02_20__0_5pct(args, trial):
    """
    """

    def suggest_int(name, low, high, default):
        return trial.suggest_int(name, low, high)

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False, verbose_results=False, verbose_arguments=False,
        put=True,
        call=True,
        iron_condor=True,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=args.upper_side_scale_factor,
        lower_side_scale_factor=args.lower_side_scale_factor,
        sharpen_x_side_scale_factor=args.sharpen_x_side_scale_factor,
        adj_balanced=False,
    )

    # --- EMA ---
    configuration.adj_call__ema = True
    configuration.adj_put__ema = True
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    configuration.ema_short = suggest_int("ema_short", 5, 50, 21)
    configuration.ema_long = suggest_int("ema_long", 50, 200, 50)
    configuration.adj_call__ema_factor = 1.005
    configuration.adj_put__ema_factor = 0.995

    # --- SMA ---
    configuration.adj_call__sma = True
    configuration.adj_put__sma = True
    configuration.sma_period = suggest_int("sma_period", 10, 100, 50)
    configuration.adj_call__sma_factor = 1.005
    configuration.adj_put__sma_factor = 0.995

    # --- RSI ---
    configuration.adj_call__rsi = True
    configuration.adj_put__rsi = True
    configuration.rsi_period = suggest_int("rsi_period", 5, 30, 14)
    configuration.adj_call__rsi_factor = 1.005
    configuration.adj_put__rsi_factor = 0.995

    # --- MACD ---
    configuration.adj_call__macd = True
    configuration.adj_put__macd = True
    configuration.macd_fast_period = suggest_int("macd_fast_period", 5, 20, 12)
    configuration.macd_slow_period = suggest_int("macd_slow_period", 20, 50, 26)
    configuration.macd_signal_period = suggest_int("macd_signal_period", 5, 15, 9)
    configuration.adj_call__macd_factor = 1.005
    configuration.adj_put__macd_factor = 0.995

    # --- Contango ---
    configuration.adj_call_and_put__contango = True
    configuration.adj_call_and_put__contango_factor = 0.005

    return configuration


def create_configuration___2026_02_20__0_25pct(args, trial):
    """
    """

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False, verbose_results=False, verbose_arguments=False,
        put=True,
        call=True,
        iron_condor=True,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=args.upper_side_scale_factor,
        lower_side_scale_factor=args.lower_side_scale_factor,
        sharpen_x_side_scale_factor=args.sharpen_x_side_scale_factor,
        adj_balanced=False,
    )

    # --- EMA ---
    configuration.adj_call__ema = trial.suggest_categorical("adj_call__ema", [True])
    configuration.adj_put__ema = trial.suggest_categorical("adj_put__ema", [True])
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    configuration.ema_short = trial.suggest_int("ema_short", low=5, high=50, step=5)
    configuration.ema_long = trial.suggest_int("ema_long", low=50, high=200, step=5)
    configuration.adj_call__ema_factor = trial.suggest_float(name='adj_call__ema_factor', low=1.0025, high=1.0025, step=0.01)
    configuration.adj_put__ema_factor = trial.suggest_float(name='adj_put__ema_factor', low=0.9975, high=0.9975, step=0.01)

    # --- SMA ---
    configuration.adj_call__sma = trial.suggest_categorical("adj_call__sma", [True])
    configuration.adj_put__sma = trial.suggest_categorical("adj_put__sma", [True])
    configuration.sma_period = trial.suggest_int(name="sma_period", low=10, high=100, step=5)
    configuration.adj_call__sma_factor = trial.suggest_float(name='adj_call__sma_factor', low=1.0025, high=1.0025, step=0.01)
    configuration.adj_put__sma_factor = trial.suggest_float(name='adj_put__sma_factor', low=0.9975, high=0.9975, step=0.01)

    # --- RSI ---
    configuration.adj_call__rsi = trial.suggest_categorical("adj_call__rsi", [True])
    configuration.adj_put__rsi = trial.suggest_categorical("adj_put__rsi", [True])
    configuration.rsi_period = trial.suggest_int("rsi_period", low=5, high=30, step=1)
    configuration.adj_call__rsi_factor = trial.suggest_float(name='adj_call__rsi_factor', low=1.0025, high=1.0025, step=0.01)
    configuration.adj_put__rsi_factor = trial.suggest_float(name='adj_put__rsi_factor', low=0.9975, high=0.9975, step=0.01)

    # --- MACD ---
    configuration.adj_call__macd = trial.suggest_categorical("adj_call__macd", [True])
    configuration.adj_put__macd = trial.suggest_categorical("adj_put__macd", [True])
    configuration.macd_fast_period = trial.suggest_int("macd_fast_period", low=5, high=20, step=1)
    configuration.macd_slow_period = trial.suggest_int("macd_slow_period", low=20, high=50, step=5)
    configuration.macd_signal_period = trial.suggest_int("macd_signal_period", low=5, high=15, step=1)
    configuration.adj_call__macd_factor = trial.suggest_float(name='adj_call__macd_factor', low=1.0025, high=1.0025, step=0.01)
    configuration.adj_put__macd_factor = trial.suggest_float(name='adj_put__macd_factor', low=0.9975, high=0.9975, step=0.01)

    # --- Contango ---
    configuration.adj_call_and_put__contango = trial.suggest_categorical("adj_call_and_put__contango", [True])
    configuration.adj_call_and_put__contango_factor = trial.suggest_float(name='adj_call_and_put__contango_factor', low=0.005, high=0.005, step=0.01)

    return configuration


def create_configuration___2026_02_20__0_0pct(args, trial):
    """
    """

    def suggest_int(name, low, high, default):
        return trial.suggest_int(name, low, high)

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False, verbose_results=False, verbose_arguments=False,
        put=True,
        call=True,
        iron_condor=True,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=args.upper_side_scale_factor,
        lower_side_scale_factor=args.lower_side_scale_factor,
        sharpen_x_side_scale_factor=args.sharpen_x_side_scale_factor,
        adj_balanced=False,
    )

    # --- EMA ---
    configuration.adj_call__ema = False
    configuration.adj_put__ema = False
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    configuration.ema_short = suggest_int("ema_short", 5, 50, 21)
    configuration.ema_long = suggest_int("ema_long", 50, 200, 50)
    configuration.adj_call__ema_factor = 1.0
    configuration.adj_put__ema_factor = 1.0

    # --- SMA ---
    configuration.adj_call__sma = False
    configuration.adj_put__sma = False
    configuration.sma_period = suggest_int("sma_period", 10, 100, 50)
    configuration.adj_call__sma_factor = 1.
    configuration.adj_put__sma_factor = 1.

    # --- RSI ---
    configuration.adj_call__rsi = False
    configuration.adj_put__rsi = False
    configuration.rsi_period = suggest_int("rsi_period", 5, 30, 14)
    configuration.adj_call__rsi_factor = 1.
    configuration.adj_put__rsi_factor = 1.

    # --- MACD ---
    configuration.adj_call__macd = False
    configuration.adj_put__macd = False
    configuration.macd_fast_period = suggest_int("macd_fast_period", 5, 20, 12)
    configuration.macd_slow_period = suggest_int("macd_slow_period", 20, 50, 26)
    configuration.macd_signal_period = suggest_int("macd_signal_period", 5, 15, 9)
    configuration.adj_call__macd_factor = 1.
    configuration.adj_put__macd_factor = 1.

    # --- Contango ---
    configuration.adj_call_and_put__contango = False
    configuration.adj_call_and_put__contango_factor = 0.005

    return configuration


def create_configuration___2026_02_20__0_25pct_balanced(args, trial):
    """
    """

    def suggest_int(name, low, high, default):
        return trial.suggest_int(name, low, high)

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False, verbose_results=False, verbose_arguments=False,
        put=True,
        call=True,
        iron_condor=True,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=args.upper_side_scale_factor,
        lower_side_scale_factor=args.lower_side_scale_factor,
        sharpen_x_side_scale_factor=args.sharpen_x_side_scale_factor,
        adj_balanced=True,
    )

    # --- EMA ---
    configuration.adj_call__ema = True
    configuration.adj_put__ema = True
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    configuration.ema_short = suggest_int("ema_short", 5, 50, 21)
    configuration.ema_long = suggest_int("ema_long", 50, 200, 50)
    configuration.adj_call__ema_factor = 1.0025
    configuration.adj_put__ema_factor = 0.9975

    # --- SMA ---
    configuration.adj_call__sma = True
    configuration.adj_put__sma = True
    configuration.sma_period = suggest_int("sma_period", 10, 100, 50)
    configuration.adj_call__sma_factor = 1.0025
    configuration.adj_put__sma_factor = 0.9975

    # --- RSI ---
    configuration.adj_call__rsi = True
    configuration.adj_put__rsi = True
    configuration.rsi_period = suggest_int("rsi_period", 5, 30, 14)
    configuration.adj_call__rsi_factor = 1.0025
    configuration.adj_put__rsi_factor = 0.9975

    # --- MACD ---
    configuration.adj_call__macd = True
    configuration.adj_put__macd = True
    configuration.macd_fast_period = suggest_int("macd_fast_period", 5, 20, 12)
    configuration.macd_slow_period = suggest_int("macd_slow_period", 20, 50, 26)
    configuration.macd_signal_period = suggest_int("macd_signal_period", 5, 15, 9)
    configuration.adj_call__macd_factor = 1.0025
    configuration.adj_put__macd_factor = 0.9975

    # --- Contango ---
    configuration.adj_call_and_put__contango = True
    configuration.adj_call_and_put__contango_factor = 0.005

    return configuration


def create_configuration___2026_02_20__0_5pct_balanced(args, trial):
    """
    """

    def suggest_int(name, low, high, default):
        return trial.suggest_int(name, low, high)

    def suggest_float(name, low, high, default, step=None):
        return trial.suggest_float(name, low, high, step=step)

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False, verbose_results=False, verbose_arguments=False,
        put=True,
        call=True,
        iron_condor=True,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=args.upper_side_scale_factor,
        lower_side_scale_factor=args.lower_side_scale_factor,
        sharpen_x_side_scale_factor=args.sharpen_x_side_scale_factor,
        adj_balanced=True,
    )

    # --- EMA ---
    configuration.adj_call__ema = True
    configuration.adj_put__ema = True
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    configuration.ema_short = suggest_int("ema_short", 5, 50, 21)
    configuration.ema_long = suggest_int("ema_long", 50, 200, 50)
    configuration.adj_call__ema_factor = 1.005
    configuration.adj_put__ema_factor = 0.995

    # --- SMA ---
    configuration.adj_call__sma = True
    configuration.adj_put__sma = True
    configuration.sma_period = suggest_int("sma_period", 10, 100, 50)
    configuration.adj_call__sma_factor = 1.005
    configuration.adj_put__sma_factor = 0.995

    # --- RSI ---
    configuration.adj_call__rsi = True
    configuration.adj_put__rsi = True
    configuration.rsi_period = suggest_int("rsi_period", 5, 30, 14)
    configuration.adj_call__rsi_factor = 1.005
    configuration.adj_put__rsi_factor = 0.995

    # --- MACD ---
    configuration.adj_call__macd = True
    configuration.adj_put__macd = True
    configuration.macd_fast_period = suggest_int("macd_fast_period", 5, 20, 12)
    configuration.macd_slow_period = suggest_int("macd_slow_period", 20, 50, 26)
    configuration.macd_signal_period = suggest_int("macd_signal_period", 5, 15, 9)
    configuration.adj_call__macd_factor = 1.005
    configuration.adj_put__macd_factor = 0.995

    # --- Contango ---
    configuration.adj_call_and_put__contango = True
    configuration.adj_call_and_put__contango_factor = 0.005

    return configuration


def create_configuration___2026_02_20__1pct_balanced(args, trial):
    """
    """

    def suggest_int(name, low, high, default):
        return trial.suggest_int(name, low, high)

    # --- Base Config ---
    # Note: We always calculate both put and call to allow switching targets without re-running,
    # but the optimizer will only see the selected score.
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False, verbose_lower_vix=False, verbose_results=False, verbose_arguments=False,
        put=True,
        call=True,
        iron_condor=True,
        step_back_range=args.step_back_range,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=args.upper_side_scale_factor,
        lower_side_scale_factor=args.lower_side_scale_factor,
        sharpen_x_side_scale_factor=args.sharpen_x_side_scale_factor,
        adj_balanced=True,
    )

    # --- EMA ---
    configuration.adj_call__ema = True
    configuration.adj_put__ema = True
    # Only suggest parameters if at least one EMA is active to save search space,
    # or always suggest if the underlying function requires the values to exist.
    # Safest is to always suggest, but conditional is more efficient.
    # Assuming underlying code checks the boolean flag first.
    configuration.ema_short = suggest_int("ema_short", 5, 50, 21)
    configuration.ema_long = suggest_int("ema_long", 50, 200, 50)
    configuration.adj_call__ema_factor = 1.01
    configuration.adj_put__ema_factor = 0.99

    # --- SMA ---
    configuration.adj_call__sma = True
    configuration.adj_put__sma = True
    configuration.sma_period = suggest_int("sma_period", 10, 100, 50)
    configuration.adj_call__sma_factor = 1.01
    configuration.adj_put__sma_factor = 0.99

    # --- RSI ---
    configuration.adj_call__rsi = True
    configuration.adj_put__rsi = True
    configuration.rsi_period = suggest_int("rsi_period", 5, 30, 14)
    configuration.adj_call__rsi_factor = 1.01
    configuration.adj_put__rsi_factor = 0.99

    # --- MACD ---
    configuration.adj_call__macd = True
    configuration.adj_put__macd = True
    configuration.macd_fast_period = suggest_int("macd_fast_period", 5, 20, 12)
    configuration.macd_slow_period = suggest_int("macd_slow_period", 20, 50, 26)
    configuration.macd_signal_period = suggest_int("macd_signal_period", 5, 15, 9)
    configuration.adj_call__macd_factor = 1.01
    configuration.adj_put__macd_factor = 0.99

    # --- Contango ---
    configuration.adj_call_and_put__contango = True
    configuration.adj_call_and_put__contango_factor = 0.01

    return configuration


# --- Objective Function Registry ---
CONFIGURATION_FUNCTIONS = {
    "2026_02_20__0_0pct": create_configuration___2026_02_20__0_0pct,
    "2026_02_20__0_25pct": create_configuration___2026_02_20__0_25pct,
    "2026_02_20__0_5pct": create_configuration___2026_02_20__0_5pct,
    "2026_02_20__1pct": create_configuration___2026_02_20__1pct,
    "2026_02_20__2pct": create_configuration___2026_02_20__2pct,
    "2026_02_20__1pct_balanced": create_configuration___2026_02_20__1pct_balanced,
    "2026_02_20__0_5pct_balanced": create_configuration___2026_02_20__0_5pct_balanced,
    "2026_02_20__0_25pct_balanced": create_configuration___2026_02_20__0_25pct_balanced,
}


def main(args):
    if args.verbose:
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)

    print(f"💾 Using Storage: {args.storage}")
    print(f"📛 Study Name: {args.study_name}")

    # --- Display Existing Study Information (if present) ---
    try:
        existing_study = optuna.load_study(
            study_name=args.study_name,
            storage=f"sqlite:///{args.storage}"
        )
        n_completed = len([t for t in existing_study.trials if t.state.is_finished()])
        if n_completed > 0:
            print(f"\n📊 Existing Study Found: '{args.study_name}'")
            print(f"   Completed Trials: {n_completed}")
            print(f"   Best Value So Far: {existing_study.best_value:.8f}")
            print(f"   Best Params: {existing_study.best_params}")
            print("-" * 80)
        else:
            print(f"\n📊 Study '{args.study_name}' exists but has no completed trials yet.")
    except Exception as e:
        print(f"\n📊 Creating new study: '{args.study_name}'")
    # -------------------------------------------------------

    timeout_str = f"{args.timeout}s" if args.timeout else "None"
    print(f"🚀 Starting Optuna Optimization (Target: {args.optimize_target}, Trials: {args.n_trials}, Timeout: {timeout_str})...")

    if args.objective_name in ["2026_02_20__0_0pct"]:
        print(f" Attention! no optimization will take place! This is the baseline. Program will exit after one pass. Change the objective_name if you want to optimize.")

    selected_objective = CONFIGURATION_FUNCTIONS[args.objective_name]

    # Create Study
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=f"sqlite:///{args.storage}",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, selected_objective, args),
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=True,
        timeout=args.timeout
    )

    # Print Best Results
    print("\n" + "=" * 80)
    print("🏆 Optimization Finished!")
    print(f"Best Score ({args.optimize_target}): {study.best_value:.8f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key:.<40} {value}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="VIX Strategy Backtest & Optimization")

    # --- Existing Args ---
    parser.add_argument('--ticker', type=str, default='^GSPC')
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    parser.add_argument('--look-ahead', type=int, default=1)
    parser.add_argument('--step-back-range', type=int, default=99999)
    parser.add_argument('--verbose', type=str2bool, default=True)

    # --- Optuna Args ---
    parser.add_argument('--n-trials', type=int, default=99999,
                        help='Number of trials for Optuna (default: 99999)')
    parser.add_argument('--optimize-target', type=str, default='put',
                        choices=['put', 'call', 'average', 'iron_condor'],
                        help='Which score to maximize')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum optimization time in seconds (None = no limit)')
    parser.add_argument('--storage', type=str, default='vix_storage.db',
                        help='Database storage path')
    parser.add_argument('--study-name', type=str, default="VIX_Strategy_Optimization",
                        help='Unique name for the study within the storage. Use same name to resume.')
    parser.add_argument("--lower-side-scale-factor", type=float, default=1., help="Gets multiplied againts the current price to obtain the lower side price")
    parser.add_argument("--upper-side-scale-factor", type=float, default=1., help="Gets multiplied againts the current price to obtain the upper side price")
    parser.add_argument('--sharpen-x-side-scale-factor', type=str2bool, default=False,
                        help="Sharpen the factor wrt the VIX")
    # --- New Argument: Objective Function Selection ---
    parser.add_argument('--objective-name', type=str, default='2026_02_20__0_0pct',
                        choices=list(CONFIGURATION_FUNCTIONS.keys()),
                        help='Select the objective function logic by name (determine by its configuration). Default: 2026_02_20__0_0pct')

    args = parser.parse_args()
    main(args)