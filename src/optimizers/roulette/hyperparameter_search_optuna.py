import json

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
import json
import numpy as np
import os
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


def objective_fct(trial, configuration_specified, args):
    """Optuna objective function with class 0 focus and penalty options."""
    # Create configuration for this trial
    configuration = configuration_specified(args, trial=trial)
    try:
        f1_scores, acc_scores = roulette_realtime_and_backtest(configuration)
    except ValueError:
        return 0.0
    except Exception as e:
        # Only print traceback for the first failure to avoid spam in parallel mode
        if trial.number == 0:
            print(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
        return 0.0
    score = 0.
    # Select optimization target
    if args.optimize_target == 'seq__f1':
        score = f1_scores
    elif args.optimize_target == 'seq__precision':
        score = acc_scores

    return score


def main(args):
    if args.verbose:
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)

    if not OPTUNA_AVAILABLE:
        print("⚠️  Optuna not available. Running single backtest with default parameters.")
        return

    if args.compiled_dataset_filename is not None and os.path.exists(args.compiled_dataset_filename):
        print(f"🚀 Optimizing only the model, using pre compiled dataset {args.compiled_dataset_filename}")
        print(f"    Please ignore the parameters since the dataset is already built.")

    # --- Storage Configuration ---
    storage_url = None
    load_if_exists = False

    if args.storage:
        # Ensure SQLAlchemy is available for RDB Storage
        try:
            import sqlalchemy
        except ImportError:
            print("❌ Error: --storage requires 'sqlalchemy'. Please run: pip install 'optuna[sql]'")
            return

        # Format storage URL if not fully provided (e.g., 'study.db' -> 'sqlite:///study.db')
        if not args.storage.startswith("sqlite:///"):
            storage_url = f"sqlite:///{args.storage}"
        else:
            storage_url = args.storage

        load_if_exists = True
        print(f"💾 Using Storage: {storage_url}")
        print(f"📛 Study Name: {args.study_name}")

    timeout_str = f"{args.timeout}s" if args.timeout else "None"
    print(f"🚀 Starting Optuna Optimization (Target: {args.optimize_target}, Trials: {args.n_trials}, Timeout: {timeout_str}, Jobs: {args.n_jobs})...")

    # Create Study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    try:
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=storage_url,
            load_if_exists=load_if_exists,
            sampler=optuna.samplers.TPESampler(),
            pruner=pruner,
        )

        # Inform user about resume status
        if len(study.trials) > 0:
            print(f"✅ Loaded existing study with {len(study.trials)} completed trials. Running {args.n_trials} additional trials.")
        else:
            print("✅ Created new study.")

    except Exception as e:
        print(f"❌ Failed to initialize Optuna Study: {e}")
        return

    selected_objective = CONFIGURATION_FUNCTIONS[args.objective_name]

    # Run Optimization with Parallelism
    study.optimize(
        lambda trial: objective_fct(trial, selected_objective, args),
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
        convert_price_val = 'fraction'
        enable_day_data_val = True

        _tmp_str = (f"python realtime_and_backtest.py "
                    f"--ticker \"{args.ticker}\" --dataset_id {args.dataset_id} --look_ahead {args.look_ahead} --step_back_range {args.step_back_range} "
                    f"--epsilon {args.epsilon} --convert_price_level_with_baseline {convert_price_val} "
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
            _tmp_str += f"--enable_macd {args.activate_macd_space_search} --macd_params '{_tmp2}' "
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

        _tmp_str += f"--add_only_vwap_z_and_vwap_triggers {args.add_only_vwap_z_and_vwap_triggers} "

        _tmp_str += f"--add_close_diff {args.add_close_diff} "
        # Add model_overrides
        # Reconstruct model_overrides from best_params (for optimize_only_models_configuration)
        VALID_MODEL_NAMES = ["xgb", "lgb", "cat", "hgb", "rf", "et", "svm", "knn", "mlp", "lr", "dt"]
        reconstructed_model_overrides = {}
        for key, value in best_params.items():
            if "__" in key:
                parts = key.split("__", 1)
                if len(parts) == 2:
                    model_name, param_name = parts
                    if model_name in VALID_MODEL_NAMES:
                        if model_name not in reconstructed_model_overrides:
                            reconstructed_model_overrides[model_name] = {}
                        reconstructed_model_overrides[model_name][param_name] = value

        if len(reconstructed_model_overrides) > 0:
            model_overrides_json = json.dumps(reconstructed_model_overrides)
            _tmp_str += f"--model-overrides '{model_overrides_json}' "
        print(f"To run the best experiment (roulette):")
        print(_tmp_str)


def get_default_namespace(args, dataset_provided=False):
    if not dataset_provided:
        assert args.compiled_dataset_filename is None
        return argparse.Namespace(
            dataset_id=args.dataset_id, col=args.col, ticker=args.ticker,
            look_ahead=args.look_ahead, verbose=args.verbose_debug,
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
            model_overrides=args.model_overrides,
            add_only_vwap_z_and_vwap_triggers=args.add_only_vwap_z_and_vwap_triggers,
            add_close_diff=args.add_close_diff,
            drop_when_out_of_range=args.drop_when_out_of_range,
            real_time_only=False,
            load_model_path=None,
        )
    else:
        assert os.path.exists(args.compiled_dataset_filename)
        return argparse.Namespace(
            compiled_dataset_filename=args.compiled_dataset_filename,
            model_overrides=args.model_overrides,
            base_models=args.base_models,
            load_model_path=None,
            save_model_path=None,
            verbose=args.verbose_debug,
        )


def _set_cfg(cfg):
    global ONE_CONFIGURATION_TO_ACCESS_FIXED_VALUES
    # Note: This global update is not visible to the main process when n_jobs > 1
    ONE_CONFIGURATION_TO_ACCESS_FIXED_VALUES = cfg


def create__base_configuration(args, trial):
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


def create__optimize_only_models_configuration(args, trial):
    configuration = get_default_namespace(args, dataset_provided=True)

    models_choices = ["xgb", "lgb", "cat", "hgb", "rf", "et", "svm", "knn", "mlp", "lr", "dt"]
    # Create an empty list to store selected models
    selected_models = []
    for model_name in models_choices:
        # Ask Optuna: "Should we include this model?"
        if trial.suggest_categorical(f"use_{model_name}", [True, False]):
            selected_models.append(model_name)
    # Ensure at least one model is selected (to prevent errors)
    if not selected_models:
        # Default to a safe choice or force one selection
        selected_models = ["knn"]
    configuration.base_models = selected_models

    model_overrides = {}
    if "xgb" in configuration.base_models:
        model_overrides.update({"xgb": {}})
        model_overrides["xgb"].update({"learning_rate": trial.suggest_float("xgb__learning_rate", 0.001, 0.5, log=True)})
        model_overrides["xgb"].update({"n_estimators": trial.suggest_int("xgb__n_estimators", 100, 500)})
        model_overrides["xgb"].update({"max_depth": trial.suggest_int("xgb__max_depth", 3, 8)})
        model_overrides["xgb"].update({"subsample": trial.suggest_float("xgb__subsample", 0.7, 1.0)})
        model_overrides["xgb"].update({"colsample_bytree": trial.suggest_float("xgb__colsample_bytree", 0.7, 1.0)})
    if "rf" in configuration.base_models:
        model_overrides.update({"rf": {}})
        model_overrides["rf"].update({"n_estimators": trial.suggest_int("rf__n_estimators", 100, 2000)})
        model_overrides["rf"].update({"max_depth": trial.suggest_int("rf__max_depth", 1, 9)})
    if "lgb" in configuration.base_models:
        model_overrides.update({"lgb": {}})
        model_overrides["lgb"].update({"n_estimators": trial.suggest_int("lgb__n_estimators", 200, 400)})
        model_overrides["lgb"].update({"max_depth":  trial.suggest_int("lgb__max_depth", 1, 9)})
        model_overrides["lgb"].update({"learning_rate": trial.suggest_float("lgb__learning_rate", 0.001, 0.5, log=True)})
        model_overrides["lgb"].update({"subsample": trial.suggest_float("lgb__subsample", 0.1, 0.9)})
        model_overrides["lgb"].update({"colsample_bytree": trial.suggest_float("lgb__colsample_bytree", 0.1, 0.9)})
    if "cat" in configuration.base_models:
        model_overrides.update({"cat": {}})
        model_overrides["cat"].update({"iterations": trial.suggest_int("cat__iterations", 300, 500)})
        model_overrides["cat"].update({"depth": trial.suggest_int("cat__depth", 1, 9)})
        model_overrides["cat"].update({"learning_rate": trial.suggest_float("cat__learning_rate", 0.001, 0.5, log=True)})
    if "hgb" in configuration.base_models:
        model_overrides.update({"hgb": {}})
        model_overrides["hgb"].update({"max_iter": trial.suggest_int("hgb__max_iter", 10, 1000)})
        model_overrides["hgb"].update({"max_depth": trial.suggest_int("hgb__max_depth", 1, 9)})
        model_overrides["hgb"].update({"learning_rate": trial.suggest_float("hgb__learning_rate", 0.001, 0.5, log=True)})
    if "et" in configuration.base_models:
        model_overrides.update({"et": {}})
        model_overrides["et"].update({"n_estimators": trial.suggest_int("et__n_estimators", 200, 400)})
        model_overrides["et"].update({"max_depth": trial.suggest_int("et__max_depth", 1, 9)})
    if "svm" in configuration.base_models:
        model_overrides.update({"svm": {}})
        model_overrides["svm"].update({"C": trial.suggest_float("svm__C", 1e-2, 1e3, log=True)})
    if "knn" in configuration.base_models:
        model_overrides.update({"knn": {}})
        model_overrides["knn"].update({"n_neighbors": trial.suggest_int("knn__n_neighbors", 3, 15)})
    #if "mlp" in configuration.base_models:
        # model_overrides.update({"mlp": {"n_neighbors": trial.suggest_int("knn__n_neighbors", 1, 20)}})
    #     "": get_base_config("mlp", MLPClassifier, {
    #         'hidden_layer_sizes': (100, 50), 'activation': 'relu',
    #         'solver': 'adam', 'alpha': 0.0001, 'batch_size': 'auto',
    #         'learning_rate': 'adaptive', 'learning_rate_init': 0.05,
    #         'max_iter': 500, 'early_stopping': False,
    #         'validation_fraction': 0.1, 'n_iter_no_change': 20,
    #         'random_state': 42,
    #     }),
    if "lr" in configuration.base_models:
        model_overrides.update({"lr": {}})
        model_overrides["lr"].update({"C": trial.suggest_float("lr__C", 1e-2, 1e3, log=True)})
    if "dt" in configuration.base_models:
        model_overrides.update({"dt": {}})
        model_overrides["dt"].update({"max_depth": trial.suggest_int("dt__max_depth", 1, 9)})

    configuration.model_overrides = json.dumps(model_overrides)
    _set_cfg(configuration)
    return configuration


# --- Objective Function Registry ---
CONFIGURATION_FUNCTIONS = {
    "base_configuration": create__base_configuration,
    "optimize_only_models_configuration": create__optimize_only_models_configuration,
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
    parser.add_argument('--verbose-debug', type=str2bool, default=False,
                        help='Whether to enable verbose debugging or not in the realtime-backtest module')

    parser.add_argument('--drop_when_out_of_range', type=str2bool, default=False)
    parser.add_argument('--activate_sma_space_search', type=str2bool, default=False)
    parser.add_argument('--activate_ema_space_search', type=str2bool, default=False)
    parser.add_argument('--activate_rsi_space_search', type=str2bool, default=False)
    parser.add_argument('--activate_macd_space_search', type=str2bool, default=False)
    parser.add_argument('--activate_vwap_space_search', type=str2bool, default=False)
    parser.add_argument('--add_only_vwap_z_and_vwap_triggers', type=str2bool, default=False)

    # --- Optuna Args ---
    parser.add_argument('--n_trials', type=int, default=999999,
                        help='Number of trials for Optuna')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs. -1 means all CPUs. (Critical for speed)')
    parser.add_argument('--optimize-target', type=str, default='seq__f1', choices=['seq__f1', 'seq__precision'],
                        help='Which score to maximize')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum optimization time in seconds')
    parser.add_argument("--epsilon", type=float, default=0.,
                        help="Threshold for neutral returns.")

    parser.add_argument('--objective-name', type=str, default='base_configuration',
                        choices=list(CONFIGURATION_FUNCTIONS.keys()),
                        help='Select the objective function logic by name')

    # --- EMA Optimization Search Space ---
    parser.add_argument('--max_ema_slots', type=int, default=2)
    parser.add_argument('--ema_min', type=int, default=5)
    parser.add_argument('--ema_max', type=int, default=20)
    parser.add_argument('--ema_step', type=int, default=2)
    parser.add_argument('--max_ema_shift_slots', type=int, default=1)
    parser.add_argument('--ema_shift_min', type=int, default=1)
    parser.add_argument('--ema_shift_max', type=int, default=3)

    # --- SMA Optimization Search Space ---
    parser.add_argument('--max_sma_slots', type=int, default=2)
    parser.add_argument('--sma_min', type=int, default=50)
    parser.add_argument('--sma_max', type=int, default=200)
    parser.add_argument('--sma_step', type=int, default=5)
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
        "--base-models",
        type=str,
        nargs="+",
        default=["xgb"],
        choices=["xgb", "lgb", "cat", "hgb", "rf", "et", "svm", "knn", "mlp", "lr", "dt"],
        help="Base model(s) to use for training."
    )
    parser.add_argument('--model-overrides', type=str, default='{}',
                        help='JSON overrides for model parameters')
    parser.add_argument('--compiled-dataset-filename', type=str, default=None,
                        help="Load pre-compiled dataset")
    # --- Storage & Resume Args ---
    parser.add_argument('--storage', type=str, default=None,
                        help='Path to SQLite DB for persistence (e.g., "optuna_study.db"). Enables resume capability.')
    parser.add_argument('--study_name', type=str, default='Roulette_Strategy_Optimization',
                        help='Unique name for the study within the storage. Use same name to resume.')

    args = parser.parse_args()
    main(args)