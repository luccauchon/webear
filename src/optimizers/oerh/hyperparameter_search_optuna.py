# optuna_optimizer.py
import optuna
import argparse
import sys
from optimizers.oerh.realtime_and_backtest import entry, setup_argparse
import pickle
# ==========================================================
# CONFIGURATION
# ==========================================================
OPTUNA_VERBOSITY = optuna.logging.WARNING

# Available metric keys from entry() output
AVAILABLE_METRICS = {
    "accuracy": "Overall accuracy (all signals combined)",
    "long_accuracy": "Accuracy on long/bullish signals only (default)",
    "short_accuracy": "Accuracy on short/bearish signals only"
}


# ==========================================================
# OPTUNA OBJECTIVE FUNCTION
# ==========================================================
def objective(trial, base_args, min_signal_ratio, penalty_weight, metric_key):
    # 1️⃣ Sample Hyperparameters
    rsi_period = trial.suggest_int("rsi_period", 5, 30)
    rsi_oversold = trial.suggest_float("rsi_oversold", 10.0, 45.0)
    rsi_overbought = trial.suggest_float("rsi_overbought", 55.0, 90.0)

    if rsi_oversold >= rsi_overbought:
        raise optuna.exceptions.TrialPruned()

    rsi_mode = trial.suggest_categorical("rsi_mode", ["momentum", "reversion"])

    macd_fast = trial.suggest_int("macd_fast", 6, 16)
    macd_slow = trial.suggest_int("macd_slow", 20, 40)
    macd_signal = trial.suggest_int("macd_signal", 5, 15)

    if macd_fast >= macd_slow:
        raise optuna.exceptions.TrialPruned()

    one_euro_min = trial.suggest_float("one_euro_min", 1.0, 50.0)
    one_euro_factor = trial.suggest_float("one_euro_factor", 0.01, 1.0)

    # 🔧 lookahead_bars: fixed user value, but tracked via Optuna for logging/reproducibility
    lookahead_bars = trial.suggest_int("lookahead_bars", base_args.lookahead_bars, base_args.lookahead_bars)

    threshold_pct = trial.suggest_float("threshold_pct", base_args.threshold_pct, base_args.threshold_pct)  # ✅ Fixed + logged

    # 2️⃣ Build argparse.Namespace with sampled values + defaults
    args = argparse.Namespace(**vars(base_args))
    args.rsi_period = rsi_period
    args.rsi_oversold = rsi_oversold
    args.rsi_overbought = rsi_overbought
    args.rsi_mode = rsi_mode
    args.macd_fast = macd_fast
    args.macd_slow = macd_slow
    args.macd_signal = macd_signal
    args.one_euro_min = one_euro_min
    args.one_euro_factor = one_euro_factor
    args.lookahead_bars = lookahead_bars  # Will always equal base_args.lookahead_bars
    args.threshold_pct = threshold_pct

    # 3️⃣ Enforce Silent Mode
    args.disable_print = True
    args.disable_plot_sample = True

    # 4️⃣ Execute entry function & apply penalty
    try:
        metrics = entry(args)
        assert "total_bars" in metrics and "total_signals" in metrics and metric_key in metrics
        # ✅ Use configurable metric key
        accuracy = metrics.get(metric_key, 0.0)
        total_signals = metrics.get("total_signals", 0)
        total_bars = metrics.get("total_bars", 1000)

        signal_ratio = total_signals / total_bars if total_bars > 0 else 0.0

        # 🔽 SIGNAL FREQUENCY PENALTY
        penalty = max(0.0, min_signal_ratio - signal_ratio) * penalty_weight
        adjusted_score = accuracy - penalty

        trial.set_user_attr("total_signals", total_signals)
        trial.set_user_attr("total_bars", total_bars)
        trial.set_user_attr("signal_ratio", signal_ratio)
        trial.set_user_attr("raw_accuracy", accuracy)
        trial.set_user_attr("metric_used", metric_key)
        trial.set_user_attr("target_type", args.target_type)
        return max(0.0, min(1.0, adjusted_score))

    except Exception as e:
        return 0.0


# ==========================================================
# STUDY STATE DISPLAY
# ==========================================================
def display_study_state(study):
    """Display current state of an existing study before optimization."""
    print(f"\n📊 Existing Study Found: '{study.study_name}'")
    print(f"   Trials completed: {len(study.trials)}")
    print(f"   Best value so far: {study.best_value:.4f}")
    print(f"   Direction: {study.direction}")

    if study.best_params:
        print("   Best params so far:")
        for k, v in study.best_params.items():
            print(f"      {k}: {v}")

    # Show metric used if stored
    if study.best_trial and study.best_trial.user_attrs.get("metric_used"):
        print(f"   Optimization metric: {study.best_trial.user_attrs['metric_used']}")

    # Show recent trial stats
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        recent = completed_trials[-5:]
        print(f"\n   Last 5 completed trials:")
        for i, t in enumerate(recent, start=len(completed_trials) - 4):
            metric_val = t.user_attrs.get("raw_accuracy", "N/A")
            print(f"      #{t.number}: value={t.value:.4f}, {t.params.get('metric_used', 'metric')}: {metric_val}")
    print("-" * 50)


# ==========================================================
# STUDY RUNNER
# ==========================================================
def run_optimization(sampler, study_name, n_trials, base_args, min_signal_ratio, penalty_weight, metric_key, storage):
    optuna.logging.set_verbosity(OPTUNA_VERBOSITY)

    # ✅ Handle None storage: study_name and load_if_exists only valid with storage backend
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage,
        study_name=study_name if storage else None,
        load_if_exists=True if storage else False
    )

    # ✅ REQ #1: Display study state if it already has trials (only possible with storage)
    if storage and len(study.trials) > 0:
        display_study_state(study)
        print(f"➕ Adding {n_trials} more trials... (optimizing for '{metric_key}')\n")
    elif storage:
        print(f"\n🔍 Starting New Study: '{study_name}' ({type(sampler).__name__})")
        print(f"   🎯 Optimization metric: {metric_key} - {AVAILABLE_METRICS.get(metric_key, 'Custom metric')}\n")
    else:
        # In-memory mode
        print(f"\n🔍 Starting In-Memory Study ({type(sampler).__name__})")
        print(f"   🎯 Optimization metric: {metric_key} - {AVAILABLE_METRICS.get(metric_key, 'Custom metric')}")
        print(f"   ⚠️  Results will NOT be persisted after this run\n")

    # Run optimization loop
    study.optimize(
        lambda trial: objective(trial, base_args, min_signal_ratio, penalty_weight, metric_key),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Report results
    print(f"\n✅ {study_name or 'In-Memory Study'} Finished!")
    print(f"🏆 Best {metric_key}: {study.best_value:.4f}")
    print("🔧 Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    return study


# ==========================================================
# MODEL PERSISTENCE
# ==========================================================
def save_best_model(study, output_dir: str = "models", custom_name: str = None, **metadata):
    """Save the best trial parameters as a loadable model file."""
    import os
    from datetime import datetime
    print(metadata)
    os.makedirs(output_dir, exist_ok=True)

    # Build model name with user-specified parameters for easy identification
    if custom_name:
        model_name = custom_name
    else:
        metric = metadata.get('metric', 'accuracy')
        lookahead = metadata.get('lookahead_bars', 5)
        threshold = metadata.get('threshold_pct', 0.01)
        ticker = metadata.get('ticker', 'SPX').replace("^", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_type = metadata.get('target_type', 'any')
        model_name = f"model_{metric}_lb{lookahead}_th{threshold:.3f}_tt{target_type}_{ticker}_{timestamp}"

    model_data = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'metadata': {
            'study_name': study.study_name,
            'direction': study.direction.name,
            'created_at': datetime.now().isoformat(),
            'optimizer_version': '1.0',
            **metadata  # Include user-specified params like metric, lookahead_bars, etc.
        },
        'user_attrs': study.best_trial.user_attrs if study.best_trial else {}
    }

    filepath = os.path.join(output_dir, f"{model_name}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n💾 Best model saved to: {filepath}")
    print(f"🔑 Key params: metric={metadata.get('metric')}, lookahead={metadata.get('lookahead_bars')}, threshold={metadata.get('threshold_pct'):.3f}, "
          f"target={metadata.get('target_type')}")
    return filepath


# ==========================================================
# ARGPARSE SETUP FOR OPTIMIZER
# ==========================================================
def setup_optimizer_argparse():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimizer for Trading Strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Metrics:
  accuracy        Overall accuracy (all signals combined)
  long_accuracy   Accuracy on long/bullish signals only [DEFAULT]
  short_accuracy  Accuracy on short/bearish signals only

Examples:
  %(prog)s --sampler tpe --metric long_accuracy --n-trials 100
  %(prog)s --sampler random --metric short_accuracy --min-signal-ratio 0.03
  %(prog)s --lookahead-bars 30 --n-trials 75  
  %(prog)s --storage sqlite:///my_custom.db --study-name my_study
        """
    )

    # Sampler choice (default: TPE)
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "random"],
        default="tpe",
        help="Optimization algorithm: 'tpe' (Tree-structured Parzen Estimator, default) or 'random' (baseline)"
    )

    # ✅ NEW: Metric choice for optimization target
    parser.add_argument(
        "--metric",
        type=str,
        choices=["accuracy", "long_accuracy", "short_accuracy"],
        default="long_accuracy",
        help=f"Metric to optimize. Options: "
             f"'accuracy' ({AVAILABLE_METRICS['accuracy']}), "
             f"'long_accuracy' ({AVAILABLE_METRICS['long_accuracy']}), "
             f"'short_accuracy' ({AVAILABLE_METRICS['short_accuracy']}). Default: 'long_accuracy'"
    )

    parser.add_argument(
        "--target-type",
        type=str,
        choices=["exact", "any"],
        default="any",
        help="Target labeling method: 'exact' (price at t+lookahead) or 'any' (price > threshold anywhere in window)"
    )

    # ✅ NEW: User-controlled lookahead_bars parameter (1-40, default: 20)
    parser.add_argument(
        "--lookahead-bars",
        type=int,
        choices=range(1, 41),
        default=20,
        help="Number of lookahead bars for signal validation. Range: 1-40 (default: 20). "
             "This value is fixed during optimization but tracked in Optuna for reproducibility."
    )

    # Configurable penalty parameters
    parser.add_argument(
        "--min-signal-ratio",
        type=float,
        default=0.1,
        help="Minimum signal ratio (signals/total_bars) before penalty applies. "
             "Trials with fewer signals are penalized. Default: 0.1 (10%%)"
    )
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=10.0,
        help="Multiplier for the signal ratio penalty. Higher = stronger penalty for low signal frequency. Default: 10.0"
    )

    # Other common args
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials to run (default: 50)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Custom study name for persistence. Auto-generated if not provided"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    # Configurable Optuna storage URL
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_forecast_optimization.db",
        help="Optuna storage URL for study persistence. "
             "Examples: 'sqlite:///results.db', 'postgresql://user:pass@host/db'. "
             "Use 'none' to disable storage (in-memory only, not persisted). "
             "Default: 'sqlite:///optuna_forecast_optimization.db'"
    )
    # User-controlled threshold_pct parameter (0.0-0.05, step 0.005)
    parser.add_argument(
        "--threshold-pct",
        type=float,
        choices=[round(x * 0.005, 3) for x in range(0, 11)],  # 0.0, 0.005, 0.01, ... 0.05
        default=0.03,
        help="Threshold percentage for signal validation. Range: 0.0 to 0.05 in steps of 0.005. "
             "This value is fixed during optimization but tracked in Optuna for reproducibility. Default: 0.03"
    )

    return parser


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    # Parse optimizer-specific arguments first
    optimizer_parser = setup_optimizer_argparse()
    opt_args, remaining_args = optimizer_parser.parse_known_args()
    # 🔧 Normalize storage: convert "none" to None for in-memory studies
    storage_url = None if opt_args.storage.lower() == "none" else opt_args.storage
    print("🚀 Starting Optuna Hyperparameter Optimization...")
    print(f"💡 Sampler: {opt_args.sampler.upper()} | Trials: {opt_args.n_trials}")
    print(f"💡 Metric: {opt_args.metric} - {AVAILABLE_METRICS[opt_args.metric]}")
    print(f"💡 Penalty: min_ratio={opt_args.min_signal_ratio}, weight={opt_args.penalty_weight}")
    print(f"💡 Lookahead bars: {opt_args.lookahead_bars} (fixed during optimization)")
    print(f"💡 Threshold pct: {opt_args.threshold_pct} (fixed during optimization)")  # ✅ Added
    print(f"💡 Target Type: {opt_args.target_type}")  # ✅ Added
    print(f"💡 Studies persisted in: {opt_args.storage}\n")  # ✅ Updated print
    # ✅ Updated storage print to handle None
    if storage_url:
        print(f"💡 Studies persisted in: {storage_url}\n")
    else:
        print(f"💡 Storage disabled - using in-memory study (not persisted)\n")
    # Load default arguments from your existing parser
    base_parser = setup_argparse()
    base_args = base_parser.parse_args([])  # Parse empty to get defaults
    base_args.dataset_id = "day"
    base_args.ticker = "^GSPC"
    base_args.older_dataset = None
    base_args.seed = opt_args.seed
    base_args.lookahead_bars = opt_args.lookahead_bars  # ✅ Apply user-specified lookahead_bars to base_args
    base_args.threshold_pct = opt_args.threshold_pct  # ✅ Apply user-specified threshold
    base_args.target_type = opt_args.target_type
    # Select sampler based on user choice
    if opt_args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=opt_args.seed)
        default_study_name = "forecast_random_study"
    else:  # default to TPE
        sampler = optuna.samplers.TPESampler(seed=opt_args.seed)
        default_study_name = "forecast_tpe_study"

    study_name = opt_args.study_name or f"{default_study_name}_{opt_args.metric}"

    # Run single optimization
    study = run_optimization(
        sampler=sampler,
        study_name=study_name,
        n_trials=opt_args.n_trials,
        base_args=base_args,
        min_signal_ratio=opt_args.min_signal_ratio,
        penalty_weight=opt_args.penalty_weight,
        metric_key=opt_args.metric,
        storage=storage_url  # ✅ Pass normalized value
    )

    # ✅ SAVE BEST MODEL AFTER OPTIMIZATION (still works with in-memory studies)
    save_best_model(
        study,
        output_dir="models",
        metric=opt_args.metric,
        lookahead_bars=opt_args.lookahead_bars,
        threshold_pct=opt_args.threshold_pct,
        ticker=base_args.ticker,
        dataset_id=base_args.dataset_id,
        target_type=base_args.target_type,
    )

    print("\n" + "=" * 60)
    if storage_url:
        print(f"📦 Results saved in: {opt_args.storage}")
        print(f"🔍 View study with: optuna-dashboard {opt_args.storage}")
    else:
        print(f"📦 Storage disabled - study not persisted to disk")
        print(f"💡 Tip: Use --storage sqlite:///my.db to save results for later analysis")
    print(f"🎯 Optimized for: {opt_args.metric} ({AVAILABLE_METRICS[opt_args.metric]})")
    print(f"🔭 Lookahead bars (fixed): {opt_args.lookahead_bars}")
    print(f"🎚️  Threshold pct (fixed): {opt_args.threshold_pct}")
    print("=" * 60)