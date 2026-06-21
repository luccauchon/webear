import pickle
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
import argparse
import pathlib
import psutil
from argparse import Namespace
import os
from optimizers.autotune.player import entry as autotune_player
from optimizers.dgdr.player import entry as dgdr_player
from optimizers.oerh.player import entry as oerh_player
from optimizers.prime_rsi.player import entry as prime_rsi_player
import time


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared):
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)

    # Traitement des requêtes
    all_results_computed = []
    while True:
        use_case_batch = []
        while len(use_case_batch) < 10:
            try:
                item = use_cases__shared.get(timeout=0.1)
                use_case_batch.append(item)
            except:
                break  # Queue is empty or no more items within timeout
        if 0 == len(use_case_batch):
            break
        for use_case in use_case_batch:
            _indicator, _args = use_case['indicator'], use_case['args']
            if _indicator == "prime_rsi":
                prime_rsi_args = _args
                all_results_computed.extend(prime_rsi_player(prime_rsi_args))
    out__shared.put(all_results_computed)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )
    parser.add_argument(
        "--autotune-target-dir",
        required=False,
        default=".",
        help="Target directory for autotune models"
    )
    parser.add_argument(
        "--dgdr-target-dir",
        required=False,
        default=".",
        help="Target directory for dgdr models"
    )
    parser.add_argument(
        "--oerh-target-dir",
        required=False,
        default=".",
        help="Target directory for oerh models"
    )
    parser.add_argument(
        "--prime-rsi-target-dir",
        required=False,
        default=".",
        help="Target directory for prime_rsi models"
    )
    parser.add_argument(
        "--hide-zero-signal",
        action="store_true",
        default=False,
        help="Hide rows where signal is 0 (default: False)"
    )
    parser.add_argument(
        "--nb-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes."
    )
    # ==========================================
    # NEW FILTER ARGUMENTS
    # ==========================================
    parser.add_argument(
        "--min-val-rate",
        type=float,
        default=None,
        help="Minimum validation win rate (e.g., 80 for 80%% or 0.8)"
    )
    parser.add_argument(
        "--min-train-rate",
        type=float,
        default=None,
        help="Minimum training win rate (e.g., 80 for 80%% or 0.8)"
    )
    parser.add_argument(
        "--signal",
        type=int,
        nargs='+',
        default=None,
        help="Filter by signal values (e.g., -1 1)"
    )
    parser.add_argument(
        "--indicator",
        type=str,
        nargs='+',
        default=None,
        help="Filter by indicator names"
    )
    parser.add_argument(
        "--method",
        type=str,
        nargs='+',
        default=None,
        help="Filter by method names"
    )
    parser.add_argument(
        "--optimize-target",
        type=str,
        nargs='+',
        default=None,
        help="Filter by optimize target names"
    )
    parser.add_argument(
        "--info",
        type=str,
        nargs='+',
        default=None,
        help="Filter by info/ticker substring (e.g., ^GSPC)"
    )
    parser.add_argument(
        "--threshold",
        type=str,
        nargs='+',
        default=None,
        help="Filter by threshold substring (e.g., 0.987)"
    )

    # ==========================================
    # SAVE / LOAD ARGUMENTS
    # ==========================================
    parser.add_argument(
        "--save-to",
        type=str,
        default=None,
        help="Save data_from_workers to the specified file after computation."
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="Load data_from_workers from the specified file and bypass computation."
    )

    return parser.parse_args()


def entry(args):
    results = []

    # Use the required parameters passed from the command line
    autotune_target_dir = args.autotune_target_dir
    dgdr_target_dir = args.dgdr_target_dir
    oerh_target_dir = args.oerh_target_dir
    prime_rsi_target_dir = args.prime_rsi_target_dir
    nb_worker = args.nb_workers

    data_from_workers = []

    # ==========================================
    # LOAD FROM FILE OR COMPUTE
    # ==========================================
    if args.load_from:
        print(f"Loading data from {args.load_from}...")
        with open(args.load_from, 'rb') as f:
            data_from_workers = pickle.load(f)
    else:
        # Construction des cas à traiter
        use_cases = []
        for root, dirs, files in os.walk(prime_rsi_target_dir):
            for file in files:
                target_file = os.path.join(str(root), str(file))
                assert os.path.exists(target_file)
                prime_rsi_args = Namespace(verbose=False, target_files=[target_file], clip=False, hide_zero_signal=False)
                use_cases.append({'indicator': 'prime_rsi', 'args': prime_rsi_args})
        # Variables partagées
        use_cases__shared, master_cmd__shared = Queue(99999), Value("i", 0)
        out__shared = [Queue(1) for k in range(0, nb_worker)]
        # Lancement des workers
        for k in range(0, nb_worker):
            p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k],))
            p.start()
        # Envoie les informations aux workers pour traitement
        # Préparation des lots de travail
        for use_case in use_cases:
            use_cases__shared.put(use_case)
        # Autoriser les workers à traiter
        with master_cmd__shared.get_lock():
            master_cmd__shared.value = 1
        # Récupération des résultats
        for k in range(0, nb_worker):
            data_from_workers.extend(out__shared[k].get())

        # ==========================================
        # SAVE TO FILE
        # ==========================================
        if args.save_to:
            print(f"Saving data to {args.save_to}...")
            with open(args.save_to, 'wb') as f:
                pickle.dump(data_from_workers, f)

    results = data_from_workers

    # # Pass hide_zero_signal dynamically from the main args
    # autotune_args = Namespace(
    #     verbose=True,
    #     target_dir=autotune_target_dir,
    #     clip=False,
    #     hide_zero_signal=args.hide_zero_signal
    # )
    # results.extend(autotune_player(autotune_args))
    #
    # dgdr_args = Namespace(
    #     verbose=True,
    #     target_dir=dgdr_target_dir,
    #     clip=False,
    #     hide_zero_signal=False
    # )
    # results.extend(dgdr_player(dgdr_args))
    #
    # oerh_args = Namespace(
    #     verbose=True,
    #     target_dir=oerh_target_dir,
    #     clip=False,
    #     hide_zero_signal=False
    # )
    # results.extend(oerh_player(oerh_args))
    #
    # # Prime RSI
    # for root, dirs, files in os.walk(prime_rsi_target_dir):
    #     prime_rsi_args = Namespace(
    #         verbose=True,
    #         target_dir=root,
    #         clip=False,
    #         hide_zero_signal=False
    #     )
    #     results.extend(prime_rsi_player(prime_rsi_args))

    # Helper to parse percentage (allows passing 80 instead of 0.8)
    def get_min_rate(val):
        if val is None:
            return None
        return val / 100.0 if val > 1 else val

    min_val_rate = get_min_rate(args.min_val_rate)
    min_train_rate = get_min_rate(args.min_train_rate)

    # Print results
    headers = ["Info", "Signal", "Current Price", "Current Date", "Target Price", "Target Date", "Train Win Rate", "Val Win Rate", "Optimize Target", "Method", "Threshold", "Indicator"]
    table_rows, mapped_table_rows = [], []

    for res in results:
        # Extract raw data first
        info = res["info"]
        signal = res["signal"]
        current_price = res['current_price']
        current_date = res['current_date']
        target_price = res['target_price']
        target_date = res['target_date']
        train_win_rate = res['train_win_rate']
        val_win_rate = res['val_win_rate']
        optimize = str(res["optimize_target"])
        method = str(res["method"])
        threshold = str(res["threshold"])
        indicator = str(res["app"])

        # ==========================================
        # APPLY FILTERS
        # ==========================================
        if args.hide_zero_signal:
            assert signal in [-1, 0, 1]
            if signal == 0:
                continue

        if min_val_rate is not None and val_win_rate < min_val_rate:
            continue

        if min_train_rate is not None and train_win_rate < min_train_rate:
            continue

        if args.signal is not None and signal not in args.signal:
            continue

        if args.indicator is not None and indicator not in args.indicator:
            continue

        if args.method is not None and method not in args.method:
            continue

        if args.optimize_target is not None and optimize not in args.optimize_target:
            continue

        if args.info is not None and not all(sub in info for sub in args.info):
            continue

        if args.threshold is not None and not any(sub in threshold for sub in args.threshold):
            continue

        # Format strings for display (only for rows that passed filters)
        current_price_str = f"{current_price:.2f}"
        current_date_str = f"{current_date.strftime('%Y-%m-%d')}"
        target_price_str = f"{target_price:.2f}"
        target_date_str = f"{target_date.strftime('%Y-%m-%d')}"
        train_win_rate_str = f"{train_win_rate:.4%}"
        val_win_rate_str = f"{val_win_rate:.4%}"
        mapped_table_rows.append({"info": info, "signal": signal, "current_price_str": current_price_str,
                                  "current_date_str": current_date_str, "target_price_str": target_price_str, "target_date_str": target_date_str,
                                  "train_win_rate_str": train_win_rate_str, "val_win_rate_str": val_win_rate_str, "optimize": optimize,
                                  "method": method, "threshold": threshold, "indicator": indicator})
        table_rows.append([info, signal, current_price_str, current_date_str, target_price_str, target_date_str, train_win_rate_str, val_win_rate_str, optimize, method, threshold, indicator])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            try:
                col_widths[i] = max(col_widths[i], len(str(cell)))
            except Exception:
                col_widths[i] = max(col_widths[i], len(f"{cell}"))

    col_widths = [w + 2 for w in col_widths]
    total_width = sum(col_widths)

    # Formatting helper
    def format_row(cells):
        return "".join(f"{str(cell):<{w}}" for cell, w in zip(cells, col_widths)).rstrip()

    print("\n" + "=" * total_width)
    print(f"{'ALL INDICATORS RESULTS SUMMARY':^{total_width}}")
    print("=" * total_width)
    print(format_row(headers))
    print("-" * total_width)
    for row in table_rows:
        print(format_row(row))
    print("=" * total_width)
    return mapped_table_rows


if __name__ == "__main__":
    freeze_support()
    args = parse_args()
    entry(args)