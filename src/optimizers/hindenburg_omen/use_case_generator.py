import argparse
import os
import sys
import time
import pathlib
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import traceback
# Attempt to import psutil, handle if missing gracefully if needed,
# but assuming it's required based on original script.
try:
    import psutil
except ImportError:
    psutil = None

# Import project specific modules
try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

from constants import IS_RUNNING_ON_CASIR
from utils import format_execution_time

# Default Grid Parameters (used if CLI args are not provided)
DEFAULT_FORWARD_DAYS = [1, 5, 10, 20]
DEFAULT_THRESHOLDS_NEG = [-0.01, -0.0125, -0.020, -0.025, -0.03]
DEFAULT_THRESHOLDS_POS = [0.01, 0.0125, 0.020, 0.025, 0.03]
DEFAULT_PENALTIES = [0.5, 0.8, 0.99]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f"{sys__name} (v{sys__version}) - Parallel Hyperparameter Optimization Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (auto-detects CASIR environment)
  python script.py

  # Specify custom workers and experience ID
  python script.py --nb-workers 8 --experience-id my_exp_v1

  # Custom hyperparameter grid
  python script.py --forward-days 1 5 10 --thresholds -0.05 -0.01 --penalties 0.9 0.99
        """
    )

    # --- Resource Management ---
    res_group = parser.add_argument_group("Resource Management")
    res_group.add_argument(
        "--nb-workers",
        type=int,
        required=True,
        help="Number of parallel worker processes."
    )
    res_group.add_argument(
        "--timeout",
        type=int,
        required=True,
        help="Timeout per trial in seconds."
    )
    res_group.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Juste compute the use cases and display the time required for execution"
    )

    # --- Experiment Configuration ---
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument(
        "--experience-id",
        type=str,
        required=True,
        help="Unique identifier for this experiment run. Used for naming output directories."
    )
    exp_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save results."
    )

    # --- Hyperparameter Grid ---
    grid_group = parser.add_argument_group("Hyperparameter Grid")
    grid_group.add_argument(
        "--forward-days",
        type=int,
        nargs="+",
        default=None,
        help="List of forward days to test (space separated). e.g., 1 2 5 10"
    )
    grid_group.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="List of threshold values to test (space separated). Negative for 'drop', Positive for 'rise'."
    )
    grid_group.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=None,
        help="List of penalty factor values for low events (space separated). e.g., 0.5 0.75 0.98"
    )

    return parser.parse_args()


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared):
    # Import inside worker to avoid multiprocessing pickle issues on some platforms
    from optimizers.hindenburg_omen.realtime_backtest_and_hyperparameter_search_optuna import run_professional_optimization

    # Wait for start signal
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)

    # Process queue
    while True:
        try:
            use_case = use_cases__shared.get(timeout=1)
        except Exception:
            break  # Queue empty or closed

        # print(f"[{os.getpid()}] Processing use_case", flush=True)
        try:
            run_professional_optimization(args=use_case)
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[{os.getpid()}] Error processing use_case: {e}", flush=True)
            print(f"[{os.getpid()}] Error processing use_case: {error_msg}", flush=True)

    # Signal completion
    out__shared.put((1, 2))


def main(args):
    # 1. Resolve Defaults based on Environment if not provided via CLI
    nb_workers = args.nb_workers
    timeout = args.timeout

    if IS_RUNNING_ON_CASIR:
        if nb_workers is None:
            nb_workers = 35
        if timeout is None:
            timeout = 7200
    else:
        if nb_workers is None:
            nb_workers = 12
        if timeout is None:
            timeout = 120

    # 2. Resolve Output Directory
    output_dir = os.path.join(args.output_dir, args.experience_id)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Resolve Hyperparameter Grids
    forward_days_list = args.forward_days if args.forward_days else DEFAULT_FORWARD_DAYS

    # If thresholds are provided, use them. Otherwise, use both neg and pos defaults.
    if args.thresholds:
        thresholds_list = args.thresholds
    else:
        thresholds_list = DEFAULT_THRESHOLDS_NEG + DEFAULT_THRESHOLDS_POS

    penalties_list = args.penalties if args.penalties else DEFAULT_PENALTIES

    # 4. Generate Use Cases
    use_cases = []

    # We iterate through all combinations provided in the arguments
    for a_forward_days in forward_days_list:
        for a_threshold in thresholds_list:
            for a_threshold_penalty_for_low_events in penalties_list:
                # Determine mode based on threshold sign (mimicking original logic)
                # Original: Negative -> 'drop', Positive -> 'upper' (though 2nd loop was commented out)
                mode = 'drop' if a_threshold < 0 else 'upper'

                output_filename = os.path.join(output_dir, f"use_case__{mode}_{a_forward_days}_{a_threshold}_{a_threshold_penalty_for_low_events}.json")
                configuration_experimentation = argparse.Namespace(
                    dataset_id="day",
                    softer_penalty_for_low_events=None,
                    forward_days=a_forward_days,
                    threshold=a_threshold,
                    cluster_mode='every_day',
                    ticker='^GSPC',
                    seed=42,
                    min_signals_required=None,
                    fixed_cluster_threshold=None,
                    fixed_cluster_window=None,
                    verbose=False,
                    mode=mode,
                    use_z_score_boost=False,
                    sampler='tpe',
                    n_startup_trials=10,
                    trials=999999,
                    timeout=timeout,
                    threshold_penalty_for_low_events=a_threshold_penalty_for_low_events,
                    no_plot=True,
                    save_params_to=output_filename,
                    disable_ema_stretch=False,
                    disable_volatility_compression=False,
                    disable_market_breadth_proxy=False,
                    disable_trend_regime_filter=False,
                    disable_rsi=False,
                    disable_macd=False,
                    disable_stochastic=False,
                    sma_len_params="2,100,false,1",
                    ema_stretch_params_ema_len="2,200,false,1",
                    ema_stretch_params_stretch_treshold="0.01,0.08,false,0.01",
                    cluster_window_params="2,120,false,1",
                    cluster_threshold_params="2,120,false,1",
                    storage=None,
                )
                use_cases.append(configuration_experimentation)
    total_estimated_time = len(use_cases) * timeout / nb_workers
    print(f"Generated {len(use_cases)} use cases.")
    print(f"Output directory: {output_dir}")
    print(f"Estimated total time: {format_execution_time(total_estimated_time)} ({nb_workers} workers). Be patient.")
    if args.dry_run:
        sys.exit()
    # 5. Setup Multiprocessing
    # Note: Queue(maxsize) is valid, but standard Queue() is often safer for infinite buffering
    use_cases__shared = Queue(9999)
    master_cmd__shared = Value('i', 0)  # 'i' for signed int

    # Create a specific output queue for each worker to collect completion signals
    out__shared = [Queue(1) for _ in range(nb_workers)]

    processes = []
    for k in range(nb_workers):
        p = Process(
            target=_worker_processor,
            args=(use_cases__shared, master_cmd__shared, out__shared[k])
        )
        p.start()
        processes.append(p)

        # Optional: Adjust priority if psutil is available
        if psutil:
            try:
                p_obj = psutil.Process(p.pid)
                # p_obj.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS) # Windows specific
            except Exception:
                pass

    # 6. Feed Work
    for use_case in use_cases:
        use_cases__shared.put(use_case)

    # 7. Start Signal
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1

    # 8. Collect Results & Join
    data_from_workers = []
    for k in range(nb_workers):
        try:
            data_from_workers.append(out__shared[k].get(timeout=timeout * 2))
        except Exception:
            data_from_workers.append(None)

    for p in processes:
        p.join()

    print(f"Workers finished.")


if __name__ == "__main__":
    freeze_support()  # Required for multiprocessing on Windows

    # Parse arguments before running main
    args = parse_arguments()
    main(args)