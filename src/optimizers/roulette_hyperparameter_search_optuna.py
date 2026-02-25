# Local custom modules
from matplotlib import use
import random
try:
    from version import sys__name, sys__version
except ImportError:
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
from optimizers.roulette_realtime_and_backtest import main as roulette_realtime_and_backtest
import warnings
import traceback
import itertools
import math

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

# --- Cache for Valid Combinations to avoid regenerating every trial ---
# Key: (min_val, max_val, max_slots), Value: List of Tuples
EMA_COMBINATION_CACHE = {}
EMA_SHIFT_COMBINATION_CACHE = {}
SMA_COMBINATION_CACHE = {}
SMA_SHIFT_COMBINATION_CACHE = {}
RSI_COMBINATION_CACHE = {}
RSI_SHIFT_COMBINATION_CACHE = {}
MACD_COMBINATION_CACHE = {}


def _tuple_to_str(t):
    """Convert tuple to string for Optuna storage (e.g., (2, 5) -> '2,5')"""
    return ','.join(map(str, t))


def _str_to_tuple(s):
    """Convert string back to tuple of ints (e.g., '2,5' -> (2, 5))"""
    return tuple(map(int, s.split(',')))


def select_distinct(n, a, b):
    if b - a + 1 < n:
        raise ValueError("Range too small")
    return random.sample(range(a, b + 1), n)


def get_unique_combinations(min_val, max_val, max_slots, _cache):
    """
    Generates all unique, strictly increasing combinations of numbers
    from min_val to max_val with lengths from 1 up to max_slots.

    Args:
        min_val (int): Minimum value in the range (inclusive).
        max_val (int): Maximum value in the range (inclusive).
        max_slots (int): Maximum number of items in a combination (inclusive).
        _cache (dict): A dictionary used to memoize results.

    Returns:
        list[tuple]: A list of tuples containing the combinations.
    """
    # Create a unique key for the cache based on input arguments
    cache_key = (min_val, max_val, max_slots)

    # Check if result already exists in cache
    if cache_key in _cache:
        return _cache[cache_key]

    all_combos = []
    # Create the pool of numbers to choose from
    number_pool = range(min_val, max_val + 1)
    total_available = len(number_pool)

    # Generate combinations for each length from 1 to max_slots
    # Note: We use max_slots + 1 to make the limit inclusive (fixing the original off-by-one)
    for r in range(1, max_slots + 1):
        # Optimization: If requested length > available numbers, stop
        if r > total_available:
            break

        # itertools.combinations handles the nested loop logic automatically
        # It yields tuples in sorted order (e.g., (1, 2) but not (2, 1))
        all_combos.extend(itertools.combinations(number_pool, r))

    # Store result in cache before returning
    _cache[cache_key] = all_combos

    return all_combos


def get_unique_combinations_hardcoded(min_val, max_val, max_slots, _cache):
    """

    """
    all_combos = []
    for depth in range(1, max_slots + 1):
        if 1 == depth:
            for a in range(min_val, max_val+1):
                all_combos.append((a, ))
        elif 2 == depth:
            for a in range(min_val, max_val + 1):
                for b in range(a+1, max_val + 1):
                    all_combos.append((a,b,))
        elif 3 == depth:
            for a in range(min_val, max_val + 1):
                for b in range(a+1, max_val + 1):
                    for c in range(b + 1, max_val + 1):
                        all_combos.append((a,b,c))
        elif 4 == depth:
            for a in range(min_val, max_val + 1):
                for b in range(a+1, max_val + 1):
                    for c in range(b + 1, max_val + 1):
                        for d in range(c + 1, max_val + 1):
                            all_combos.append((a,b,c,d))
        elif 5 == depth:
            for a in range(min_val, max_val + 1):
                for b in range(a+1, max_val + 1):
                    for c in range(b + 1, max_val + 1):
                        for d in range(c + 1, max_val + 1):
                            for e in range(d + 1, max_val + 1):
                                all_combos.append((a,b,c,d, e))
    return all_combos


def objective(trial, configuration_specified, args):
    """Optuna objective function."""
    configuration = configuration_specified(args, trial=trial)
    # return random.random() - random.random() /2 + random.random()/4
    try:
        f1_scores, acc_scores, avg_precision, avg_recall, avg_f1 = roulette_realtime_and_backtest(configuration)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        return 0.0

    if args.optimize_target   == 'pos_seq_0__f1':
        score = avg_f1[0]
    elif args.optimize_target == 'pos_seq_1__f1':
        score = avg_f1[1]
    elif args.optimize_target == 'pos_seq_2__f1':
        score = avg_f1[2]
    elif args.optimize_target == 'pos_seq_3__f1':
        score = avg_f1[3]
    elif args.optimize_target == 'neg_seq_0__f1':
        score = avg_f1[0]
    elif args.optimize_target == 'neg_seq_1__f1':
        score = avg_f1[1]
    elif args.optimize_target == 'neg_seq_2__f1':
        score = avg_f1[2]
    else:
        assert False, f""

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

    timeout_str = f"{args.timeout}s" if args.timeout else "None"
    print(f"🚀 Starting Optuna Optimization (Target: {args.optimize_target}, Trials: {args.n_trials}, Timeout: {timeout_str})...")

    # Create Study
    study = optuna.create_study(direction="maximize", study_name="VIX_Strategy_Optimization")

    selected_objective = CONFIGURATION_FUNCTIONS[args.objective_name]

    # Pre-validate search spaces to warn user if they are too large for unique enforcement
    if args.activate_ema_space_search:
        assert args.max_ema_slots <= 5
        combos   = get_unique_combinations_hardcoded(args.ema_min, args.ema_max, args.max_ema_slots, EMA_COMBINATION_CACHE)
        combos_2 = get_unique_combinations(args.ema_min, args.ema_max, args.max_ema_slots, EMA_COMBINATION_CACHE)
        assert len(combos) == len(combos_2)
        for a_tuple in combos:
            assert 1 == len([aaa for aaa in combos_2 if a_tuple == aaa])
        combos = get_unique_combinations(args.ema_shift_min, args.ema_shift_max, args.max_ema_shift_slots, EMA_SHIFT_COMBINATION_CACHE)
    if args.activate_sma_space_search:
        combos = get_unique_combinations(args.sma_min, args.sma_max, args.max_sma_slots, SMA_COMBINATION_CACHE)
        combos = get_unique_combinations(args.sma_shift_min, args.sma_shift_max, args.max_sma_shift_slots, SMA_SHIFT_COMBINATION_CACHE)
    if args.activate_rsi_space_search:
        combos = get_unique_combinations(args.rsi_min, args.rsi_max, args.max_rsi_slots, RSI_COMBINATION_CACHE)
        combos = get_unique_combinations(args.rsi_shift_min, args.rsi_shift_max, args.max_rsi_shift_slots, RSI_SHIFT_COMBINATION_CACHE)
    study.optimize(lambda trial: objective(trial, selected_objective, args), n_trials=args.n_trials, n_jobs=1, show_progress_bar=True, timeout=args.timeout)

    print("\n" + "=" * 80)
    print("🏆 Optimization Finished!")
    print(f"Best Score ({args.optimize_target}): {study.best_value:.8f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key:.<40} {value}")
    print("=" * 80 + "\n")


def create_configuration(args, trial):
    """
    Creates the configuration Namespace.
    Ensures unique values in windows and shifts.
    """

    # --- EMA Logic ---
    use_ema, ema_windows, shift_ema_col = False, [], []
    if args.activate_ema_space_search:
        use_ema = trial.suggest_categorical("use_ema", [True, False])
        if use_ema:
            valid_combos = get_unique_combinations(args.ema_min, args.ema_max, args.max_ema_slots, EMA_COMBINATION_CACHE)
            # Convert tuples to strings for Optuna compatibility
            combo_strings = [_tuple_to_str(c) for c in valid_combos]
            selected_str = trial.suggest_categorical("ema_windows_tuple", combo_strings)
            ema_windows = list(_str_to_tuple(selected_str))

            valid_shift_combos = get_unique_combinations(args.ema_shift_min, args.ema_shift_max, args.max_ema_shift_slots, EMA_SHIFT_COMBINATION_CACHE)
            shift_combo_strings = [_tuple_to_str(c) for c in valid_shift_combos]
            selected_shift_str = trial.suggest_categorical("ema_shifts_tuple", shift_combo_strings)
            shift_ema_col = list(_str_to_tuple(selected_shift_str))

    # --- SMA Logic ---
    use_sma, sma_windows, shift_sma_col = False, [], []
    if args.activate_sma_space_search:
        use_sma = trial.suggest_categorical("use_sma", [True, False])
        if use_sma:
            valid_combos = get_unique_combinations(args.sma_min, args.sma_max, args.max_sma_slots, SMA_COMBINATION_CACHE)
            assert valid_combos
            # if valid_combos:
            combo_strings = [_tuple_to_str(c) for c in valid_combos]
            selected_str = trial.suggest_categorical("sma_windows_tuple", combo_strings)
            sma_windows = list(_str_to_tuple(selected_str))

            valid_shift_combos = get_unique_combinations(args.sma_shift_min, args.sma_shift_max, args.max_sma_shift_slots, SMA_SHIFT_COMBINATION_CACHE)
            assert valid_shift_combos
            # if valid_shift_combos:
            shift_combo_strings = [_tuple_to_str(c) for c in valid_shift_combos]
            selected_shift_str = trial.suggest_categorical("sma_shifts_tuple", shift_combo_strings)
            shift_sma_col = list(_str_to_tuple(selected_shift_str))

    # --- RSI Logic ---
    use_rsi, rsi_windows, shift_rsi_col = False, [], []
    if args.activate_rsi_space_search:
        use_rsi = trial.suggest_categorical("use_rsi", [True, False])
        if use_rsi:
            valid_combos = get_unique_combinations(args.rsi_min, args.rsi_max, args.max_rsi_slots, RSI_COMBINATION_CACHE)
            assert valid_combos
            # if valid_combos:
            combo_strings = [_tuple_to_str(c) for c in valid_combos]
            selected_str = trial.suggest_categorical("rsi_windows_tuple", combo_strings)
            rsi_windows = list(_str_to_tuple(selected_str))

            valid_shift_combos = get_unique_combinations(args.rsi_shift_min, args.rsi_shift_max, args.max_rsi_shift_slots, RSI_SHIFT_COMBINATION_CACHE)
            assert valid_shift_combos
            # if valid_shift_combos:
            shift_combo_strings = [_tuple_to_str(c) for c in valid_shift_combos]
            selected_shift_str = trial.suggest_categorical("rsi_shifts_tuple", shift_combo_strings)
            shift_rsi_col = list(_str_to_tuple(selected_shift_str))

    # --- MACD Logic ---
    use_macd, macd_params = False, {}
    shift_macd_col = []  # MACD typically doesn't use shifts in the same way, keeping empty for compatibility
    if args.activate_macd_space_search:
        use_macd = trial.suggest_categorical("use_macd", [True, False])
        if use_macd:
            # Define search ranges (uses args if available, else defaults)
            f_min = getattr(args, 'macd_fast_min', 10)
            f_max = getattr(args, 'macd_fast_max', 20)
            s_min = getattr(args, 'macd_slow_min', 20)
            s_max = getattr(args, 'macd_slow_max', 40)
            sig_min = getattr(args, 'macd_signal_min', 5)
            sig_max = getattr(args, 'macd_signal_max', 15)

            # Create cache key
            cache_key = (f_min, f_max, s_min, s_max, sig_min, sig_max)

            # Generate valid triplets (fast, slow, signal) where fast < slow
            if cache_key not in MACD_COMBINATION_CACHE:
                valid_triplets = []
                for f in range(f_min, f_max + 1):
                    # Enforce Slow > Fast (Standard MACD definition)
                    effective_s_min = max(s_min, f + 1)
                    if effective_s_min > s_max:
                        continue
                    for s in range(effective_s_min, s_max + 1):
                        for sig in range(sig_min, sig_max + 1):
                            valid_triplets.append((f, s, sig))
                MACD_COMBINATION_CACHE[cache_key] = valid_triplets

            valid_triplets = MACD_COMBINATION_CACHE[cache_key]
            assert valid_triplets

            # Convert to strings for Optuna categorical suggestion
            triplet_strings = [_tuple_to_str(t) for t in valid_triplets]
            selected_str = trial.suggest_categorical("macd_params_tuple", triplet_strings)
            f, s, sig = _str_to_tuple(selected_str)
            macd_params = {"fast": f, "slow": s, "signal": sig}
    # print(f"")
    # print(f"{sma_windows=} {shift_sma_col=} {use_sma}   {ema_windows=} {shift_ema_col=} {use_ema}   {rsi_windows=} {shift_rsi_col=} {use_rsi}   {macd_params=} {use_macd}")
    assert 0 == len(shift_macd_col)
    configuration = Namespace(
        dataset_id=args.dataset_id, col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=False,
        target="POS_SEQ" if "pos_seq" in args.optimize_target else "NEQ_SEQ",
        convert_price_level_with_baseline='fraction',
        sma_windows=sma_windows,
        ema_windows=ema_windows,
        shift_sma_col=shift_sma_col,
        shift_ema_col=shift_ema_col,
        shift_rsi_col=shift_rsi_col,
        shift_macd_col=shift_macd_col,
        rsi_windows=rsi_windows,
        macd_params=macd_params,
        enable_macd=use_macd,
        enable_sma=use_sma,
        enable_ema=use_ema,
        enable_rsi=use_rsi,
        step_back_range=args.step_back_range,
        epsilon=args.epsilon,
    )

    return configuration


# --- Objective Function Registry ---
CONFIGURATION_FUNCTIONS = {
    "base_configuration": create_configuration,
}


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="Roulette Strategy Backtest & Optimization")

    # --- Existing Args ---
    parser.add_argument('--ticker', type=str, default='^GSPC')
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    parser.add_argument('--look_ahead', type=int, default=1)
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
    parser.add_argument('--step_back_range', type=int, default=99999)
    parser.add_argument('--verbose', type=str2bool, default=True)

    parser.add_argument('--activate_sma_space_search', type=str2bool, default=True)
    parser.add_argument('--activate_ema_space_search', type=str2bool, default=True)
    parser.add_argument('--activate_rsi_space_search', type=str2bool, default=True)
    parser.add_argument('--activate_macd_space_search', type=str2bool, default=True)

    # --- Optuna Args ---
    parser.add_argument('--n_trials', type=int, default=99999,
                        help='Number of trials for Optuna (ignored if use_optuna=False)')
    parser.add_argument('--optimize_target', type=str, default='pos_seq_1__f1',
                        choices=['pos_seq_0__f1', 'pos_seq_1__f1', 'pos_seq_2__f1', 'pos_seq_3__f1',
                                 'neg_seq_0__f1', 'neg_seq_1__f1', 'neg_seq_2__f1'],
                        help='Which score to maximize')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum optimization time in seconds (None = no limit)')
    parser.add_argument("--epsilon", type=float, default=0.,
                        help="Threshold for neutral returns. Default: 0.")

    # --- New Argument: Objective Function Selection ---
    parser.add_argument('--objective_name', type=str, default='base_configuration',
                        choices=list(CONFIGURATION_FUNCTIONS.keys()),
                        help='Select the objective function logic by name (determine by its configuration)')
    # --- Optimization Search Space Args ---
    parser.add_argument('--max_ema_slots', type=int, default=5,
                        help='Max number of EMA windows Optuna can combine (e.g., 5 allows [w1, w2, w3, w4, w5])')
    parser.add_argument('--ema_min', type=int, default=2,
                        help='Minimum EMA window size')
    parser.add_argument('--ema_max', type=int, default=10,
                        help='Maximum EMA window size')
    parser.add_argument('--max_ema_shift_slots', type=int, default=5,
                        help='')
    parser.add_argument('--ema_shift_min', type=int, default=1,
                        help='Minimum EMA shift')
    parser.add_argument('--ema_shift_max', type=int, default=5,
                        help='Maximum EMA shift')

    parser.add_argument('--max_sma_slots', type=int, default=5,
                        help='Max number of SMA windows Optuna can combine (e.g., 5 allows [w1, w2, w3, w4, w5])')
    parser.add_argument('--sma_min', type=int, default=2,
                        help='Minimum SMA window size')
    parser.add_argument('--sma_max', type=int, default=10,
                        help='Maximum SMA window size')
    parser.add_argument('--max_sma_shift_slots', type=int, default=5,
                        help='')
    parser.add_argument('--sma_shift_min', type=int, default=1,
                        help='Minimum SMA shift')
    parser.add_argument('--sma_shift_max', type=int, default=5,
                        help='Maximum SMA shift')

    parser.add_argument('--max_rsi_slots', type=int, default=3,
                        help='')
    parser.add_argument('--rsi_min', type=int, default=7,
                        help='Minimum RSI window size')
    parser.add_argument('--rsi_max', type=int, default=21,
                        help='Maximum RSI window size')
    parser.add_argument('--max_rsi_shift_slots', type=int, default=5,
                        help='')
    parser.add_argument('--rsi_shift_min', type=int, default=1,
                        help='Minimum RSI shift')
    parser.add_argument('--rsi_shift_max', type=int, default=5,
                        help='Maximum RSI shift')

    # --- MACD Optimization Search Space Args ---
    parser.add_argument('--macd_fast_min', type=int, default=10,
                        help='Minimum MACD Fast period')
    parser.add_argument('--macd_fast_max', type=int, default=20,
                        help='Maximum MACD Fast period')
    parser.add_argument('--macd_slow_min', type=int, default=20,
                        help='Minimum MACD Slow period')
    parser.add_argument('--macd_slow_max', type=int, default=40,
                        help='Maximum MACD Slow period')
    parser.add_argument('--macd_signal_min', type=int, default=5,
                        help='Minimum MACD Signal period')
    parser.add_argument('--macd_signal_max', type=int, default=15,
                        help='Maximum MACD Signal period')

    args = parser.parse_args()
    main(args)