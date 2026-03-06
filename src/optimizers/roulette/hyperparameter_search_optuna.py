try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

# Local custom modules
import multiprocessing
import random
import numpy as np

np.seterr(invalid='ignore')
import sys
import pathlib
import argparse
import numpy as np
import warnings
import traceback
import itertools

# --- Optuna Integration ---
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Warning: optuna not installed. Run 'pip install optuna' to enable parameter search.")

from utils import DATASET_AVAILABLE, str2bool
from optimizers.roulette.realtime_and_backtest import main as roulette_realtime_and_backtest

# Suppress specific numpy warnings
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

# --- Cache for Valid Combinations ---
EMA_COMBINATION_CACHE = {}
EMA_SHIFT_COMBINATION_CACHE = {}
SMA_COMBINATION_CACHE = {}
SMA_SHIFT_COMBINATION_CACHE = {}
RSI_COMBINATION_CACHE = {}
RSI_SHIFT_COMBINATION_CACHE = {}
MACD_COMBINATION_CACHE = {}

# Global variable to store fixed configuration values (Note: Unsafe with n_jobs > 1, handled in main)
ONE_CONFIGURATION_TO_ACCESS_FIXED_VALUES = None


def estimate_search_space_size(args):
    """
    Estimates the total number of possible parameter combinations
    based on the Optuna search space configuration.

    Returns:
        dict: Dictionary containing breakdown of search space by indicator type
        int: Total number of possible combinations
    """
    search_space = {}
    total_combinations = 1

    print("\n" + "=" * 80)
    print("📊 SEARCH SPACE ESTIMATION")
    print("=" * 80)

    # --- EMA Combinations ---
    if args.activate_ema_space_search:
        ema_number_pool = range(args.ema_min, args.ema_max + 1, args.ema_step)
        ema_total_available = len(ema_number_pool)
        ema_combos = 0
        for r in range(0, args.max_ema_slots + 1):
            if r <= ema_total_available:
                from math import comb
                ema_combos += comb(ema_total_available, r)

        ema_shift_number_pool = range(args.ema_shift_min, args.ema_shift_max + 1, 1)
        ema_shift_total_available = len(ema_shift_number_pool)
        ema_shift_combos = 0
        for r in range(0, args.max_ema_shift_slots + 1):
            if r <= ema_shift_total_available:
                from math import comb
                ema_shift_combos += comb(ema_shift_total_available, r)

        ema_total = ema_combos * ema_shift_combos
        search_space['EMA'] = {
            'windows': ema_combos,
            'shifts': ema_shift_combos,
            'total': ema_total
        }
        total_combinations *= ema_total
        print(f"\n📈 EMA Search Space:")
        print(f"    Windows: {args.ema_min}-{args.ema_max} (step={args.ema_step}), max_slots={args.max_ema_slots}")
        print(f"    → {ema_combos:,} window combinations")
        print(f"    Shifts: {args.ema_shift_min}-{args.ema_shift_max}, max_slots={args.max_ema_shift_slots}")
        print(f"    → {ema_shift_combos:,} shift combinations")
        print(f"    ✅ EMA Total: {ema_total:,} combinations")
    else:
        print(f"\n📈 EMA Search Space: DISABLED")

    # --- SMA Combinations ---
    if args.activate_sma_space_search:
        sma_number_pool = range(args.sma_min, args.sma_max + 1, args.sma_step)
        sma_total_available = len(sma_number_pool)
        sma_combos = 0
        for r in range(0, args.max_sma_slots + 1):
            if r <= sma_total_available:
                from math import comb
                sma_combos += comb(sma_total_available, r)

        sma_shift_number_pool = range(args.sma_shift_min, args.sma_shift_max + 1, 1)
        sma_shift_total_available = len(sma_shift_number_pool)
        sma_shift_combos = 0
        for r in range(0, args.max_sma_shift_slots + 1):
            if r <= sma_shift_total_available:
                from math import comb
                sma_shift_combos += comb(sma_shift_total_available, r)

        sma_total = sma_combos * sma_shift_combos
        search_space['SMA'] = {
            'windows': sma_combos,
            'shifts': sma_shift_combos,
            'total': sma_total
        }
        total_combinations *= sma_total
        print(f"\n📈 SMA Search Space:")
        print(f"    Windows: {args.sma_min}-{args.sma_max} (step={args.sma_step}), max_slots={args.max_sma_slots}")
        print(f"    → {sma_combos:,} window combinations")
        print(f"    Shifts: {args.sma_shift_min}-{args.sma_shift_max}, max_slots={args.max_sma_shift_slots}")
        print(f"    → {sma_shift_combos:,} shift combinations")
        print(f"    ✅ SMA Total: {sma_total:,} combinations")
    else:
        print(f"\n📈 SMA Search Space: DISABLED")

    # --- RSI Combinations ---
    if args.activate_rsi_space_search:
        rsi_number_pool = range(args.rsi_min, args.rsi_max + 1, 1)
        rsi_total_available = len(rsi_number_pool)
        rsi_combos = 0
        for r in range(0, args.max_rsi_slots + 1):
            if r <= rsi_total_available:
                from math import comb
                rsi_combos += comb(rsi_total_available, r)

        rsi_shift_number_pool = range(args.rsi_shift_min, args.rsi_shift_max + 1, 1)
        rsi_shift_total_available = len(rsi_shift_number_pool)
        rsi_shift_combos = 0
        for r in range(0, args.max_rsi_shift_slots + 1):
            if r <= rsi_shift_total_available:
                from math import comb
                rsi_shift_combos += comb(rsi_shift_total_available, r)

        rsi_total = rsi_combos * rsi_shift_combos
        search_space['RSI'] = {
            'windows': rsi_combos,
            'shifts': rsi_shift_combos,
            'total': rsi_total
        }
        total_combinations *= rsi_total
        print(f"\n📈 RSI Search Space:")
        print(f"    Windows: {args.rsi_min}-{args.rsi_max}, max_slots={args.max_rsi_slots}")
        print(f"    → {rsi_combos:,} window combinations")
        print(f"    Shifts: {args.rsi_shift_min}-{args.rsi_shift_max}, max_slots={args.max_rsi_shift_slots}")
        print(f"    → {rsi_shift_combos:,} shift combinations")
        print(f"    ✅ RSI Total: {rsi_total:,} combinations")
    else:
        print(f"\n📈 RSI Search Space: DISABLED")

    # --- MACD Combinations ---
    if args.activate_macd_space_search:
        f_min = getattr(args, 'macd_fast_min', 10)
        f_max = getattr(args, 'macd_fast_max', 20)
        s_min = getattr(args, 'macd_slow_min', 20)
        s_max = getattr(args, 'macd_slow_max', 40)
        sig_min = getattr(args, 'macd_signal_min', 5)
        sig_max = getattr(args, 'macd_signal_max', 15)

        macd_combos = 0
        for f in range(f_min, f_max + 1):
            effective_s_min = max(s_min, f + 1)
            if effective_s_min > s_max:
                continue
            for s in range(effective_s_min, s_max + 1):
                for sig in range(sig_min, sig_max + 1):
                    macd_combos += 1

        search_space['MACD'] = {
            'total': macd_combos
        }
        total_combinations *= macd_combos
        print(f"\n📈 MACD Search Space:")
        print(f"    Fast: {f_min}-{f_max}, Slow: {s_min}-{s_max}, Signal: {sig_min}-{sig_max}")
        print(f"    ✅ MACD Total: {macd_combos:,} combinations")
    else:
        print(f"\n📈 MACD Search Space: DISABLED")

    # --- VWAP Combinations ---
    if args.activate_vwap_space_search:
        vwap_combos = args.vwap_max_window - args.vwap_min_window + 1
        search_space['VWAP'] = {
            'total': vwap_combos
        }
        total_combinations *= vwap_combos
        print(f"\n📈 VWAP Search Space:")
        print(f"    Window: {args.vwap_min_window}-{args.vwap_max_window}")
        print(f"    ✅ VWAP Total: {vwap_combos:,} combinations")
    else:
        print(f"\n📈 VWAP Search Space: DISABLED")

    # --- Shift Sequence Column ---
    shift_seq_combos = args.shift_seq_col_max - args.shift_seq_col_min + 1
    search_space['SHIFT_SEQ'] = {
        'total': shift_seq_combos
    }
    total_combinations *= shift_seq_combos
    print(f"\n📈 Shift Sequence Column Search Space:")
    print(f"    Range: {args.shift_seq_col_min}-{args.shift_seq_col_max}")
    print(f"    ✅ Shift Seq Total: {shift_seq_combos:,} combinations")

    # --- Summary ---
    print("\n" + "=" * 80)
    print(f"🎯 TOTAL SEARCH SPACE SIZE: {total_combinations:,} unique combinations")
    print("=" * 80)

    # --- Trial Coverage Analysis ---
    if args.n_trials:
        coverage_percentage = min(100.0, (args.n_trials / total_combinations) * 100)
        print(f"\n📋 TRIAL COVERAGE ANALYSIS:")
        print(f"    Planned Trials: {args.n_trials:,}")
        print(f"    Search Space: {total_combinations:,}")
        print(f"    Coverage: {coverage_percentage:.2f}%")

        if coverage_percentage < 1.0:
            print(f"    ⚠️  WARNING: Very low coverage! Consider increasing n_trials or reducing search space.")
        elif coverage_percentage < 10.0:
            print(f"    ⚠️  Low coverage. Optuna's TPE sampler will help, but results may vary.")
        elif coverage_percentage < 50.0:
            print(f"    ✅ Moderate coverage. TPE sampler should find good solutions.")
        else:
            print(f"    ✅ Excellent coverage. Near-exhaustive search possible.")

    # --- Time Estimation ---
    print(f"\n⏱️  TIME ESTIMATION:")
    print(f"    Timeout: {args.timeout if args.timeout else 'None (unlimited)'} seconds")
    if args.n_trials and args.timeout:
        avg_time_per_trial = args.timeout / args.n_trials
        print(f"    Estimated avg time per trial: {avg_time_per_trial:.2f} seconds")

    print("=" * 80 + "\n")

    return search_space, total_combinations


def _tuple_to_str(t):
    """Convert tuple to string for Optuna storage (e.g., (2, 5) -> '2,5')"""
    if len(t) == 0:
        return ''
    return ','.join(map(str, t))


def _str_to_tuple(s):
    """Convert string back to tuple of ints (e.g., '2,5' -> (2, 5))"""
    if s == '':
        return ()
    return tuple(map(int, s.split(',')))


def get_unique_combinations(min_val, max_val, max_slots, _cache, _step):
    """
    Generates all unique, strictly increasing combinations of numbers
    from min_val to max_val with lengths from 0 up to max_slots.
    """
    cache_key = (min_val, max_val, max_slots, _step)

    if cache_key in _cache:
        return _cache[cache_key]

    all_combos = []
    number_pool = range(min_val, max_val + 1, _step)
    total_available = len(number_pool)

    for r in range(0, max_slots + 1):
        if r > total_available:
            break
        all_combos.extend(itertools.combinations(number_pool, r))

    _cache[cache_key] = all_combos
    return all_combos


def objective(trial, configuration_specified, args):
    """Optuna objective function with class 0 focus and penalty options."""
    # Create configuration for this trial
    configuration = configuration_specified(args, trial=trial)

    if args.skip_optimization:
        return random.random()

    try:
        f1_scores, acc_scores, avg_precision, avg_recall, avg_f1 = roulette_realtime_and_backtest(configuration)
    except ValueError:
        return 0.0
    except Exception as e:
        # Only print traceback for the first failure to avoid spam in parallel mode
        if trial.number == 0:
            print(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
        return 0.0

    # Select optimization target
    score = 0.0
    class_0_score = 0.0
    other_scores = []

    # Extract class 0 and other class scores
    if len(avg_f1) > 0:
        class_0_score = avg_f1[0]
    if len(avg_f1) > 1:
        other_scores = avg_f1[1:]

    # Calculate penalty for other classes
    penalty = 0.0
    if len(other_scores) > 0:
        penalty = np.mean(other_scores) * args.other_classes_penalty

    if args.optimize_target == 'pos_seq_0__f1':
        score = class_0_score
    elif args.optimize_target == 'pos_seq_1__f1':
        score = avg_f1[1] if len(avg_f1) > 1 else 0.0
    elif args.optimize_target == 'pos_seq_2__f1':
        score = avg_f1[2] if len(avg_f1) > 2 else 0.0
    elif args.optimize_target == 'pos_seq_3__f1':
        score = avg_f1[3] if len(avg_f1) > 3 else 0.0
    elif args.optimize_target == 'neg_seq_0__f1':
        score = class_0_score
    elif args.optimize_target == 'neg_seq_1__f1':
        score = avg_f1[1] if len(avg_f1) > 1 else 0.0
    elif args.optimize_target == 'neg_seq_2__f1':
        score = avg_f1[2] if len(avg_f1) > 2 else 0.0
    elif args.optimize_target == 'pos_seq__f1':
        score = f1_scores
    elif args.optimize_target == 'neg_seq__f1':
        score = f1_scores

    # Class 0 focus with penalty on other classes
    elif args.optimize_target == 'pos_seq_0__f1_penalty_others':
        score = class_0_score - penalty

    elif args.optimize_target == 'neg_seq_0__f1_penalty_others':
        score = class_0_score - penalty

    # Class 0 focus with weighted penalty (class 0 = 2x weight)
    elif args.optimize_target == 'pos_seq_0__f1_weighted_penalty':
        score = (2.0 * class_0_score) - penalty

    elif args.optimize_target == 'neg_seq_0__f1_weighted_penalty':
        score = (2.0 * class_0_score) - penalty

    # Custom weight optimization for class 0
    elif args.optimize_target == 'pos_seq_0__f1_custom_weight':
        score = (args.class_0_weight * class_0_score) - penalty

    elif args.optimize_target == 'neg_seq_0__f1_custom_weight':
        score = (args.class_0_weight * class_0_score) - penalty

    # Maximize class 0 only, ignore others completely
    elif args.optimize_target == 'pos_seq_0__f1_only':
        score = class_0_score

    elif args.optimize_target == 'neg_seq_0__f1_only':
        score = class_0_score

    else:
        raise ValueError(f"Unknown optimize_target: {args.optimize_target}")

    # Optional verbose logging for class-specific scores
    if args.verbose and args.verbose_info_score and trial.number % 10 == 0:
        print(f"\n📊 Trial {trial.number} Scores:")
        for i, f1 in enumerate(avg_f1):
            marker = "🎯" if i == 0 else "  "
            print(f"    {marker} Class {i} F1: {f1:.4f}")
        print(f"    Final Optimization Score: {score:.4f}")

    return score


def main(args):
    if args.verbose:
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)

        # Estimate search space before optimization ---
        if OPTUNA_AVAILABLE:
            search_space, total_combinations = estimate_search_space_size(args)
        else:
            print("\n⚠️  Search space estimation skipped (Optuna disabled)\n")

    if not OPTUNA_AVAILABLE:
        print("⚠️  Optuna not available. Running single backtest with default parameters.")

    timeout_str = f"{args.timeout}s" if args.timeout else "None"
    print(f"🚀 Starting Optuna Optimization (Target: {args.optimize_target}, Trials: {args.n_trials}, Timeout: {timeout_str}, Jobs: {args.n_jobs})...")

    # Create Study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        direction="maximize",
        study_name="Roulette_Strategy_Optimization",
        sampler=optuna.samplers.TPESampler(),
        pruner=pruner
    )

    selected_objective = CONFIGURATION_FUNCTIONS[args.objective_name]

    # Run Optimization with Parallelism
    study.optimize(
        lambda trial: objective(trial, selected_objective, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        timeout=args.timeout
    )

    print("\n" + "=" * 80)
    print("🏆 Optimization Finished!")
    if study.best_trial:
        print(f"Best Score ({args.optimize_target}): {study.best_value:.8f}")
        print("Best Parameters:")
        for key, value in study.best_params.items():
            print(f"    {key:.<40} {value}")

        # Show class-specific performance if available
        if hasattr(study.best_trial, 'user_attrs') and 'class_scores' in study.best_trial.user_attrs:
            print("\n📊 Best Trial Class Performance:")
            for i, score in enumerate(study.best_trial.user_attrs['class_scores']):
                marker = "🎯" if i == 0 else "  "
                print(f"    {marker} Class {i} F1: {score:.4f}")
    else:
        print("No successful trials completed.")
    print("=" * 80 + "\n")

    # Reconstruct command string from args and best_params
    # (Do not rely on ONE_CONFIGURATION_TO_ACCESS_FIXED_VALUES when n_jobs > 1)
    if study.best_trial:
        best_params = study.best_params

        # Derive fixed values from args (mimicking get_default_namespace logic)
        target_val = "POS_SEQ" if "pos_seq" in args.optimize_target else "NEG_SEQ"
        convert_price_val = 'fraction'
        enable_day_data_val = True

        _tmp_str = (f"python realtime_and_backtest.py "
                    f"--ticker \"{args.ticker}\" --dataset_id {args.dataset_id} --look_ahead {args.look_ahead} --step_back_range {args.step_back_range} "
                    f"--epsilon {args.epsilon} --target {target_val} --convert_price_level_with_baseline {convert_price_val} "
                    f"--verbose true --older_dataset none ")

        # EMA
        if 'ema_windows_tuple' in best_params and len(best_params['ema_windows_tuple']) > 0:
            assert args.activate_ema_space_search
            _tmp_str += f"--enable_ema true --ema_windows {best_params['ema_windows_tuple'].replace(',', ' ')} "
            if 'ema_shifts_tuple' in best_params and len(best_params['ema_shifts_tuple']) > 0:
                _tmp_str += f"--shift_ema_col {best_params['ema_shifts_tuple'].replace(',', ' ')} "
        else:
            _tmp_str += f"--enable_ema false "

        # SMA
        if 'sma_windows_tuple' in best_params and len(best_params['sma_windows_tuple']) > 0:
            assert args.activate_sma_space_search
            _tmp_str += f"--enable_sma true --sma_windows {best_params['sma_windows_tuple'].replace(',', ' ')} "
            if 'sma_shifts_tuple' in best_params and len(best_params['sma_shifts_tuple']) > 0:
                _tmp_str += f"--shift_sma_col {best_params['sma_shifts_tuple'].replace(',', ' ')} "
        else:
            _tmp_str += f"--enable_sma false "

        # RSI
        if 'rsi_windows_tuple' in best_params and len(best_params['rsi_windows_tuple']) > 0:
            assert args.activate_rsi_space_search
            _tmp_str += f"--enable_rsi true --rsi_windows {best_params['rsi_windows_tuple'].replace(',', ' ')} "
            if 'rsi_shifts_tuple' in best_params and len(best_params['rsi_shifts_tuple']) > 0:
                _tmp_str += f"--shift_rsi_col {best_params['rsi_shifts_tuple'].replace(',', ' ')} "
        else:
            _tmp_str += f"--enable_rsi false "

        # MACD
        if 'macd_params_tuple' in best_params:
            f, s, sig = _str_to_tuple(best_params['macd_params_tuple'])
            macd_params = {"fast": f, "slow": s, "signal": sig}
            _tmp2 = f"{macd_params}".replace("\'", "\\\"")
            _tmp_str += f"--enable_macd {args.activate_macd_space_search} --macd_params \"{_tmp2}\" "
        else:
            _tmp_str += f"--enable_macd false "

        # VWAP
        if 'vwap_window' in best_params:
            _tmp_str += f"--enable_vwap true --vwap_window {best_params['vwap_window']} "
        else:
            _tmp_str += f"--enable_vwap false "

        _tmp_str += f"--enable_day_data {enable_day_data_val} "

        if 'shift_seq_col' in best_params:
            _tmp_str += f"--shift_seq_col {best_params['shift_seq_col']} "

        _tmp_str += f"--min_percentage_to_keep_class {args.min_percentage_to_keep_class} "

        if len(args.specific_wanted_class) > 0:
            _tmp_str += f"--specific_wanted_class {str(args.specific_wanted_class)[1:-1].replace(',', ' ')} "

        _clean_model = str(args.base_models)[1:-1].replace(',', '').replace("'", "")
        _tmp_str += f"--base_models {_clean_model} "

        if args.add_only_vwap_z_and_vwap_triggers:
            _tmp_str += f"--add_only_vwap_z_and_vwap_triggers {args.add_only_vwap_z_and_vwap_triggers} "

        if not args.add_close_diff:
            _tmp_str += f"--add_close_diff {args.add_close_diff} "

        print(f"To run the best experiment:")
        print(_tmp_str)


def get_default_namespace(args):
    return argparse.Namespace(
        dataset_id=args.dataset_id, col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead, verbose=args.verbose_debug,
        target="POS_SEQ" if "pos_seq" in args.optimize_target else "NEG_SEQ",
        convert_price_level_with_baseline='fraction',
        sma_windows=[],
        ema_windows=[],
        shift_sma_col=[],
        shift_ema_col=[],
        shift_rsi_col=[],
        shift_macd_col=[],
        rsi_windows=[],
        macd_params=[],
        enable_macd=False,
        enable_sma=False,
        enable_ema=False,
        enable_vwap=False,
        enable_rsi=False,
        enable_day_data=True,
        shift_seq_col=1,
        step_back_range=args.step_back_range,
        epsilon=args.epsilon,
        compiled_dataset_filename=None,
        save_dataset_to_file_and_exit=None,
        min_percentage_to_keep_class=args.min_percentage_to_keep_class,
        specific_wanted_class=args.specific_wanted_class,
        base_models=args.base_models,
        save_model_path=None,
        model_overrides='{}',
        add_only_vwap_z_and_vwap_triggers=args.add_only_vwap_z_and_vwap_triggers,
        add_close_diff=args.add_close_diff,
    )


def _set_cfg(cfg):
    global ONE_CONFIGURATION_TO_ACCESS_FIXED_VALUES
    # Note: This global update is not visible to the main process when n_jobs > 1
    ONE_CONFIGURATION_TO_ACCESS_FIXED_VALUES = cfg


def create_base_configuration(args, trial):
    """
    Creates the configuration Namespace.
    Ensures unique values in windows and shifts.
    """
    # --- EMA Logic ---
    ema_windows, shift_ema_col = [], []
    if args.activate_ema_space_search:
        valid_combos = get_unique_combinations(args.ema_min, args.ema_max, args.max_ema_slots, EMA_COMBINATION_CACHE, _step=args.ema_step)
        combo_strings = [_tuple_to_str(c) for c in valid_combos]
        selected_str = trial.suggest_categorical("ema_windows_tuple", combo_strings)
        ema_windows = list(_str_to_tuple(selected_str))

        valid_shift_combos = get_unique_combinations(args.ema_shift_min, args.ema_shift_max, args.max_ema_shift_slots, EMA_SHIFT_COMBINATION_CACHE, _step=1)
        shift_combo_strings = [_tuple_to_str(c) for c in valid_shift_combos]
        selected_shift_str = trial.suggest_categorical("ema_shifts_tuple", shift_combo_strings)
        shift_ema_col = list(_str_to_tuple(selected_shift_str))

    # --- SMA Logic ---
    sma_windows, shift_sma_col = [], []
    if args.activate_sma_space_search:
        valid_combos = get_unique_combinations(args.sma_min, args.sma_max, args.max_sma_slots, SMA_COMBINATION_CACHE, _step=args.sma_step)
        if valid_combos:
            combo_strings = [_tuple_to_str(c) for c in valid_combos]
            selected_str = trial.suggest_categorical("sma_windows_tuple", combo_strings)
            sma_windows = list(_str_to_tuple(selected_str))

            valid_shift_combos = get_unique_combinations(args.sma_shift_min, args.sma_shift_max, args.max_sma_shift_slots, SMA_SHIFT_COMBINATION_CACHE, _step=1)
            if valid_shift_combos:
                shift_combo_strings = [_tuple_to_str(c) for c in valid_shift_combos]
                selected_shift_str = trial.suggest_categorical("sma_shifts_tuple", shift_combo_strings)
                shift_sma_col = list(_str_to_tuple(selected_shift_str))

    # --- RSI Logic ---
    rsi_windows, shift_rsi_col = [], []
    if args.activate_rsi_space_search:
        valid_combos = get_unique_combinations(args.rsi_min, args.rsi_max, args.max_rsi_slots, RSI_COMBINATION_CACHE, _step=1)
        if valid_combos:
            combo_strings = [_tuple_to_str(c) for c in valid_combos]
            selected_str = trial.suggest_categorical("rsi_windows_tuple", combo_strings)
            rsi_windows = list(_str_to_tuple(selected_str))

            valid_shift_combos = get_unique_combinations(args.rsi_shift_min, args.rsi_shift_max, args.max_rsi_shift_slots, RSI_SHIFT_COMBINATION_CACHE, _step=1)
            if valid_shift_combos:
                shift_combo_strings = [_tuple_to_str(c) for c in valid_shift_combos]
                selected_shift_str = trial.suggest_categorical("rsi_shifts_tuple", shift_combo_strings)
                shift_rsi_col = list(_str_to_tuple(selected_shift_str))

    # --- MACD Logic ---
    macd_params = {}
    shift_macd_col = []
    if args.activate_macd_space_search:
        f_min = getattr(args, 'macd_fast_min', 10)
        f_max = getattr(args, 'macd_fast_max', 20)
        s_min = getattr(args, 'macd_slow_min', 20)
        s_max = getattr(args, 'macd_slow_max', 40)
        sig_min = getattr(args, 'macd_signal_min', 5)
        sig_max = getattr(args, 'macd_signal_max', 15)

        cache_key = (f_min, f_max, s_min, s_max, sig_min, sig_max)

        if cache_key not in MACD_COMBINATION_CACHE:
            valid_triplets = []
            for f in range(f_min, f_max + 1):
                effective_s_min = max(s_min, f + 1)
                if effective_s_min > s_max:
                    continue
                for s in range(effective_s_min, s_max + 1):
                    for sig in range(sig_min, sig_max + 1):
                        valid_triplets.append((f, s, sig))
            MACD_COMBINATION_CACHE[cache_key] = valid_triplets

        valid_triplets = MACD_COMBINATION_CACHE[cache_key]
        if valid_triplets:
            triplet_strings = [_tuple_to_str(t) for t in valid_triplets]
            selected_str = trial.suggest_categorical("macd_params_tuple", triplet_strings)
            f, s, sig = _str_to_tuple(selected_str)
            macd_params = {"fast": f, "slow": s, "signal": sig}

    # --- VWAP Logic ---
    vwap_window = None
    if args.activate_vwap_space_search:
        vwap_window = trial.suggest_int(name="vwap_window", low=args.vwap_min_window, high=args.vwap_max_window)

    configuration = get_default_namespace(args)
    configuration.enable_sma = args.activate_sma_space_search
    configuration.sma_windows = sma_windows
    configuration.shift_sma_col = shift_sma_col

    configuration.enable_ema = args.activate_ema_space_search
    configuration.ema_windows = ema_windows
    configuration.shift_ema_col = shift_ema_col

    configuration.enable_rsi = args.activate_rsi_space_search
    configuration.shift_rsi_col = shift_rsi_col
    configuration.rsi_windows = rsi_windows

    configuration.enable_macd = args.activate_macd_space_search
    configuration.macd_params = macd_params
    configuration.shift_macd_col = shift_macd_col

    configuration.enable_vwap = args.activate_vwap_space_search
    configuration.vwap_window = vwap_window

    configuration.shift_seq_col = trial.suggest_int(name="shift_seq_col", low=args.shift_seq_col_min, high=args.shift_seq_col_max)

    _set_cfg(configuration)
    return configuration


# --- Objective Function Registry ---
CONFIGURATION_FUNCTIONS = {
    "base_configuration": create_base_configuration,
}


if __name__ == "__main__":
    # Essential for multiprocessing on Windows, harmless on Linux
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Roulette Strategy Backtest & Optimization")

    # --- Existing Args ---
    parser.add_argument('--ticker', type=str, default='^GSPC')
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    parser.add_argument('--look_ahead', type=int, default=1)
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
    parser.add_argument('--step_back_range', type=int, default=99999)
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--verbose_info_score', type=str2bool, default=False,
                        help='Print info about score while doing optimization')
    parser.add_argument('--verbose_debug', type=str2bool, default=False,
                        help='Whether to enable verbose debugging or not in the realtime-backtest module')

    parser.add_argument('--activate_sma_space_search', type=str2bool, default=False)
    parser.add_argument('--activate_ema_space_search', type=str2bool, default=False)
    parser.add_argument('--activate_rsi_space_search', type=str2bool, default=False)
    parser.add_argument('--activate_macd_space_search', type=str2bool, default=False)
    parser.add_argument('--activate_vwap_space_search', type=str2bool, default=False)
    parser.add_argument('--add_only_vwap_z_and_vwap_triggers', type=str2bool, default=False)

    # --- Optuna Args ---
    parser.add_argument('--n_trials', type=int, default=99999,
                        help='Number of trials for Optuna')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs. -1 means all CPUs. (Critical for speed)')
    parser.add_argument('--optimize_target', type=str, default='pos_seq__f1',
                        choices=[
                            'pos_seq_0__f1', 'pos_seq_1__f1', 'pos_seq_2__f1', 'pos_seq_3__f1',
                            'pos_seq__f1',
                            'neg_seq_0__f1', 'neg_seq_1__f1', 'neg_seq_2__f1',
                            'neg_seq__f1',
                            # Class 0 focus with penalty
                            'pos_seq_0__f1_penalty_others',
                            'neg_seq_0__f1_penalty_others',
                            # Class 0 focus with weighted penalty
                            'pos_seq_0__f1_weighted_penalty',
                            'neg_seq_0__f1_weighted_penalty',
                            # Custom weight optimization
                            'pos_seq_0__f1_custom_weight',
                            'neg_seq_0__f1_custom_weight',
                            # Class 0 only (ignore others)
                            'pos_seq_0__f1_only',
                            'neg_seq_0__f1_only',
                        ],
                        help='Which score to maximize')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum optimization time in seconds')
    parser.add_argument("--epsilon", type=float, default=0.,
                        help="Threshold for neutral returns.")

    parser.add_argument('--objective_name', type=str, default='base_configuration',
                        choices=list(CONFIGURATION_FUNCTIONS.keys()),
                        help='Select the objective function logic by name')

    # --- NEW: Class 0 Optimization Weights ---
    parser.add_argument('--class_0_weight', type=float, default=1.0,
                        help='Weight for class 0 in optimization (higher = more focus on class 0)')
    parser.add_argument('--other_classes_penalty', type=float, default=0.0,
                        help='Penalty multiplier for other classes (0 = no penalty, 1 = full penalty)')

    # --- EMA Optimization Search Space ---
    parser.add_argument('--max_ema_slots', type=int, default=2)
    parser.add_argument('--ema_min', type=int, default=5)
    parser.add_argument('--ema_max', type=int, default=20)
    parser.add_argument('--ema_step', type=int, default=5)
    parser.add_argument('--max_ema_shift_slots', type=int, default=1)
    parser.add_argument('--ema_shift_min', type=int, default=1)
    parser.add_argument('--ema_shift_max', type=int, default=3)

    # --- SMA Optimization Search Space ---
    parser.add_argument('--max_sma_slots', type=int, default=2)
    parser.add_argument('--sma_min', type=int, default=50)
    parser.add_argument('--sma_max', type=int, default=200)
    parser.add_argument('--sma_step', type=int, default=20)
    parser.add_argument('--max_sma_shift_slots', type=int, default=1)
    parser.add_argument('--sma_shift_min', type=int, default=1)
    parser.add_argument('--sma_shift_max', type=int, default=3)

    # --- RSI Optimization Search Space ---
    parser.add_argument('--max_rsi_slots', type=int, default=1)
    parser.add_argument('--rsi_min', type=int, default=2)
    parser.add_argument('--rsi_max', type=int, default=14)
    parser.add_argument('--max_rsi_shift_slots', type=int, default=1)
    parser.add_argument('--rsi_shift_min', type=int, default=1)
    parser.add_argument('--rsi_shift_max', type=int, default=3)

    # --- MACD Optimization Search Space ---
    parser.add_argument('--macd_fast_min', type=int, default=8)
    parser.add_argument('--macd_fast_max', type=int, default=16)
    parser.add_argument('--macd_slow_min', type=int, default=21)
    parser.add_argument('--macd_slow_max', type=int, default=34)
    parser.add_argument('--macd_signal_min', type=int, default=7)
    parser.add_argument('--macd_signal_max', type=int, default=12)

    # --- VWAP & Sequence ---
    parser.add_argument('--vwap_min_window', type=int, default=5)
    parser.add_argument('--vwap_max_window', type=int, default=30)
    parser.add_argument('--shift_seq_col_min', type=int, default=1)
    parser.add_argument('--shift_seq_col_max', type=int, default=3)

    parser.add_argument('--min_percentage_to_keep_class', type=float, default=-1)
    parser.add_argument("--specific_wanted_class", type=int, nargs='+', default=[])

    parser.add_argument('--add_close_diff', type=str2bool, default=True)
    parser.add_argument('--skip_optimization', type=str2bool, default=False)
    parser.add_argument(
        "--base_models",
        type=str,
        nargs="+",
        default=["xgb"],
        choices=["xgb", "lgb", "cat", "hgb", "rf", "et", "svm", "knn", "mlp", "lr", "dt"],
        help="Base model(s) to use for training."
    )

    args = parser.parse_args()
    main(args)