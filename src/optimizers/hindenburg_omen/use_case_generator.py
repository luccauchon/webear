try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import psutil
import time
import os
import argparse
from constants import IS_RUNNING_ON_CASIR
from utils import format_execution_time


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared):
    from optimizers.hindenburg_omen.realtime_backtest_and_hyperparameter_search_optuna import run_professional_optimization
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)
    # print(f"[{os.getpid()}] Hello world!", flush=True)
    while True:
        try:
            use_case = use_cases__shared.get(timeout=1)
        except:
            break  # Terminé
        # print(f"[{os.getpid()}] Processing {use_case}", flush=True)
        run_professional_optimization(use_case)

    # print(f"[{os.getpid()}] Bye bye!", flush=True)
    out__shared.put((1, 2))

def main():
    _nb_workers = 12
    experience_id = "alpha_2"
    output_dir = os.path.join(fr"D:\Temp2\use_case", experience_id)
    timeout = 120
    if IS_RUNNING_ON_CASIR:
        timeout = 7200
        _nb_workers = 35
        output_dir = os.path.join("/gpfs/home/cj3272/14b/cj3272/experiences/", experience_id)

    os.makedirs(output_dir, exist_ok=True)
    use_cases = []
    for a_forward_days in [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        for a_threshold in [-0.01, -0.0125, -0.020, -0.025, -0.03]:
            for a_threshold_penalty_for_low_events in [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
                output_filename = os.path.join(output_dir, f"use_case__drop_{a_forward_days}_{a_threshold}_{a_threshold_penalty_for_low_events}.json")
                configuration_experimentation = argparse.Namespace(forward_days=a_forward_days, threshold=a_threshold, cluster_mode='every_day', ticker='^GSPC', seed=42,
                                                                   min_signals_required=None, fixed_cluster_threshold=None, fixed_cluster_window=None, verbose=False,
                                                                   mode='drop', use_z_score_boost=False, sampler='tpe', n_startup_trials=10, trials=999999, timeout=timeout,
                                                                   threshold_penalty_for_low_events=a_threshold_penalty_for_low_events, no_plot=True, save_params_to=output_filename)
                use_cases.append(configuration_experimentation)
    for a_forward_days in [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        for a_threshold in [0.01, 0.0125, 0.020, 0.025, 0.03]:
            for a_threshold_penalty_for_low_events in [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
                output_filename = os.path.join(output_dir, f"use_case__rise_{a_forward_days}_{a_threshold}_{a_threshold_penalty_for_low_events}.json")
                configuration_experimentation = argparse.Namespace(forward_days=a_forward_days, threshold=a_threshold, cluster_mode='every_day', ticker='^GSPC', seed=42,
                                                                   min_signals_required=None, fixed_cluster_threshold=None, fixed_cluster_window=None, verbose=False,
                                                                   mode='upper', use_z_score_boost=False, sampler='tpe', n_startup_trials=10, trials=999999, timeout=timeout,
                                                                   threshold_penalty_for_low_events=a_threshold_penalty_for_low_events, no_plot=True, save_params_to=output_filename)
                # use_cases.append(configuration_experimentation)
    print(f"Generated {len(use_cases)} use cases, total time of {format_execution_time(len(use_cases) * timeout / _nb_workers)}. Be patient.")
    use_cases__shared, master_cmd__shared = Queue(len(use_cases)), Value("i", 0)
    out__shared = [Queue(1) for k in range(0, _nb_workers)]
    # Lancement des workers
    for k in range(0, _nb_workers):
        p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k],))
        p.start()
        pid = p.pid
        p_obj = psutil.Process(pid)
        # p_obj.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    # Envoie les informations aux workers pour traitement
    for use_case in use_cases:
        use_cases__shared.put(use_case)
    # Autoriser les workers à traiter
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1
    # Récupération des résultats
    data_from_workers = []
    for k in range(0, _nb_workers):
        data_from_workers.append(out__shared[k].get())
    print(f"{data_from_workers}")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    freeze_support()  # Required for multiprocessing on Windows

    main()