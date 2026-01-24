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
import copy
from runners.wavelet_backtest import main as wavelet_backtest
# Third-party optimization library
import optuna
from optuna.trial import TrialState
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")


def main(args):
    # --- Nicely print the arguments ---
    print("🔧 Arguments:")
    for arg, value in vars(args).items():
        print(f"    {arg:.<40} {value}")
    print("-" * 80, flush=True)

    ticker = args.ticker
    col = args.col
    backtest_strategy = args.backtest_strategy
    dataset_id = args.dataset_id
    n_forecast_length = args.n_forecast_length
    number_of_step_back = args.step_back_range
    show_n_top_configurations = 5
    verbose = args.verbose
    scale_factor_for_ground_truth = args.scale_factor_for_ground_truth
    assert  0 <= scale_factor_for_ground_truth <= 0.2
    sell_call_credit_spread = args.sell_call_credit_spread
    sell_put_credit_spread = args.sell_put_credit_spread
    assert (sell_call_credit_spread and not sell_put_credit_spread) or \
           (not sell_call_credit_spread and sell_put_credit_spread), \
        "Exactly one of sell_call_credit_spread or sell_put_credit_spread must be True."
    if dataset_id == 'day':
        assert number_of_step_back <= 4000
    warrior_pred_scale_factor = args.prediction_scale_factor

    # Storage for all evaluated configurations and results
    results = []

    def objective(trial):
        """Optuna objective: returns -success_rate (to be minimized)."""
        # Sample hyperparameters
        n_forecast_length_in_training = trial.suggest_int('n_forecast_length_in_training', 1, 99)
        n_forecasts = trial.suggest_int('n_forecasts', 19, 99)  # To avoid "D:\PyCharmProjects\webear\src\optimizers\wavelet_opt.py:453: RuntimeWarning: Mean of empty slice"
        if sell_put_credit_spread:
            warrior_spread = 'put'
        else:
            warrior_spread = 'call'

        # Build configuration namespace
        configuration = Namespace(
            backtest_strategy=backtest_strategy,
            col=col,
            dataset_id=dataset_id,
            exit_strategy=None,
            n_forecast_length=n_forecast_length,
            n_forecast_length_in_training=n_forecast_length_in_training,
            n_models_to_keep=n_forecasts,
            step_back_range=number_of_step_back,
            save_to_disk=False,
            strategy_for_exit='hold_until_the_end_with_roll',
            thresholds_ep='(0.0125, 0.0125)',
            threshold_for_shape_similarity=0,
            ticker=ticker,
            use_last_week_only=False,
            verbose=verbose,
            warrior_gt_range_for_success=scale_factor_for_ground_truth,
            warrior_pred_scale_factor=warrior_pred_scale_factor,
            warrior_spread=warrior_spread,
        )

        # Run backtest
        result = wavelet_backtest(configuration)
        success_rate = result['success_rate']

        # Store full result for post-analysis
        results.append({
            'config': copy.deepcopy(configuration),
            'result': result,
            'success_rate': success_rate,
            'trial_number': trial.number
        })

        # Optuna minimizes → return negative success rate
        return -success_rate

    # Create Optuna study
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

    # Set timeout (None if no limit)
    timeout = args.time_limit_seconds if args.time_limit_seconds != -1 else None
    print(f"🚀 Starting Bayesian optimization with Optuna (timeout: {timeout}s)...")
    try:
        study.optimize(objective, timeout=timeout)
    except KeyboardInterrupt:
        print("\n⏹️ Optimization interrupted by user.")

    # Map trial number to trial state
    trial_state_map = {t.number: t.state for t in study.trials}

    # Keep only results from completed trials
    filtered_results = [
        r for r in results if trial_state_map.get(r['trial_number']) == TrialState.COMPLETE
    ]

    top_results = sorted(filtered_results, key=lambda x: x['success_rate'], reverse=True)[:show_n_top_configurations]

    print("\n" + "=" * 60)
    print(f"🏆 TOP {show_n_top_configurations} CONFIGURATIONS BY SUCCESS RATE")
    print("=" * 60)
    for i, res in enumerate(top_results, 1):
        cfg = res['config']
        sr = res['success_rate'] / 100.0  # convert from percentage
        print(f"{i}. Success Rate: {sr:.2%}")
        print(f"   • Forecast Length (training): {cfg.n_forecast_length_in_training}")
        print(f"   • Number of Forecasts: {cfg.n_models_to_keep}")
        print(f"   • Scale Factor: {cfg.warrior_pred_scale_factor:.4f}")
        print("-" * 60)

    sys.exit(0)


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="Run Bayesian optimization on Wavelet-based stock backtest using Optuna.")

    parser.add_argument('--ticker', type=str, default='^GSPC')
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    parser.add_argument('--dataset_id', type=str, default='month',
                        choices=DATASET_AVAILABLE)
    parser.add_argument('--n_forecast_length', type=int, default=1)
    parser.add_argument("--prediction_scale_factor", type=float, default=0.)
    parser.add_argument('--step-back-range', type=int, default=300)
    parser.add_argument("--scale_factor_for_ground_truth", type=float, default=0.1)
    parser.add_argument("--backtest_strategy", type=str, default="warrior")
    parser.add_argument("--sell_call_credit_spread", type=str2bool, default=True)
    parser.add_argument("--sell_put_credit_spread", type=str2bool, default=False)
    parser.add_argument('--time_limit_seconds', type=int, default=-1,
                        help="Max time in seconds; -1 = no limit")
    parser.add_argument('--verbose', type=str2bool, default=False)

    args = parser.parse_args()
    main(args)