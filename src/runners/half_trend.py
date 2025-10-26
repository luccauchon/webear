try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # print(parent_dir)
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from constants import FYAHOO__OUTPUTFILENAME
import os
import argparse
from algorithms.trade_prime_half_trend import trade_prime_half_trend_strategy
import json
import pickle
import pprint
from tqdm import tqdm
from algorithms.trade_prime_half_trend import trade_prime_half_trend_strategy
from datetime import datetime, timedelta, date
import os
import time
from utils import format_execution_time, get_weekdays
from copy import deepcopy
from multiprocessing import freeze_support, Lock, Process, Queue, Value
from constants import FYAHOO__OUTPUTFILENAME, NB_WORKERS
import pickle
import psutil


def _update_alerts_ui_print_only(setup_log):
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    GRAY = '\033[90m'

    # Clear screen (optional, for cleaner output)
    print("\033[2J\033[H", end="")  # Clears terminal and moves cursor to top-left

    now = datetime.now()
    print(f"{GRAY}@{now.strftime('%Y-%m-%d %H:%M')}{RESET}")

    # Sort alerts newest first
    for alert in sorted(setup_log, key=lambda x: x['time'], reverse=True):
        the_time = (alert['time'] + timedelta(hours=0)).strftime('%Y-%m-%d')
        signal_at = alert['time'].strftime('%H:%M')
        enter_at = datetime.now().strftime('%H:%M')

        # Determine emoji and color
        if alert['type'] == "BUY":
            color = GREEN
            emoji = "✅" if alert['actual'] >= alert['close'] else "⚠️"
        else:  # SELL
            color = RED
            emoji = "✅" if alert['actual'] <= alert['close'] else "⚠️"

        line = (
            f"[{the_time}>>{signal_at}] "
            f"{alert['type']:>4} → {alert['ticker']} , dst:{alert['distance']}h >> "
            f"Entry:{alert['close']:.2f}@{enter_at}  "
            f"SL:{alert['stop_loss']:.2f}  TP:{alert['take_profit']:.2f}  "
            f"Actual:{alert['actual']:.2f} {emoji}"
        )
        print(f"{color}{line}{RESET}")


def _worker_processor(stocks__shared, master_cmd__shared, out__shared, ):
    _debug = False
    today, yesterday, before_yesterday = get_weekdays()
    timestamp = datetime.fromtimestamp(os.path.getmtime(FYAHOO__OUTPUTFILENAME)).strftime('%Y-%m-%d %H:%M:%S')
    if _debug:
        print(f"[{os.getpid()}:{datetime.now()}] Reading FYAHOO data source from {FYAHOO__OUTPUTFILENAME} ({timestamp})")
    with open(FYAHOO__OUTPUTFILENAME, 'rb') as f:
        data_cache = pickle.load(f)

    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.1)
    results = []
    # Effectuer le traitement
    while True:
        # Arrêter le traitement
        with master_cmd__shared.get_lock():
            if 3 == master_cmd__shared.value:
                break

        try:
            tmp = stocks__shared.get(timeout=0.1)
            assert 1 == len(tmp)
            stock = list(tmp.keys())[0]
            if _debug:
                print(f"[{os.getpid()}:{datetime.now()}] Processing stock: {stock}")
        except:
            break  # Terminé

        for buy_setup in [True, False]:
            kwargs = {'use__entry_type': 'Hard',
                      'use__higher_timeframe_strong_trend': {'enable': False},
                      'use__volume_confirmed': {'enable': False},
                      'use__relative_strength_vs_benchmark': {'enable_vix': False, 'vix_dataframe': data_cache["^VIX"].copy(), 'period_vix': 10 * 7,
                                                              'enable_spx': False, 'spx_dataframe': data_cache["^GSPC"].copy(),
                                                              }
                      }
            df = trade_prime_half_trend_strategy(ticker_df=data_cache[stock].copy(), ticker_name=stock, buy_setup=buy_setup, **kwargs)
            recent_signals = df[df[('custom_signal', stock)]]
            if not recent_signals.empty:
                last = recent_signals.iloc[-1]
                if last.name.date() == today or last.name.date() == yesterday or last.name.date() == before_yesterday:
                    alert = {'time': last.name, 'ticker': stock, 'type': 'BUY' if buy_setup else 'SELL', 'close': last[('Close', stock)],
                             'distance': last[('triggered_distance', stock)], 'entry_point': last[('custom_signal', stock)], 'actual': df[('Close', stock)].iloc[-1],
                             'stop_loss': last[('stop_loss', stock)], 'take_profit': last[('take_profit', stock)]}
                    results.append(alert)
    out__shared.put(results)
    if _debug:
        print(f"[{os.getpid()}:{datetime.now()} Terminating")


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('-c', '--config', required=False, default=r'D:\Temp2\test.json', help='Path to the configuration file')
    return parser.parse_args()


def entry():
    args = parse_args()
    config_file_path = args.config

    # Check if the config file exists
    if not os.path.exists(config_file_path):
        print(f"Config file '{config_file_path}' not found.")
        return

    #timestamp = datetime.fromtimestamp(os.path.getmtime(FYAHOO__OUTPUTFILENAME)).strftime('%Y-%m-%d %H:%M:%S')
    #print(f"Reading FYAHOO data source from {FYAHOO__OUTPUTFILENAME} ({timestamp})")
    with open(FYAHOO__OUTPUTFILENAME, 'rb') as f:
        data_cache = pickle.load(f)

    #timestamp = datetime.fromtimestamp(os.path.getmtime(config_file_path)).strftime('%Y-%m-%d %H:%M:%S')
    #print(f"Reading configuration setup from {config_file_path} ({timestamp})")
    with open(config_file_path, 'r') as f:
        configuration_setup = json.load(f)

    pprint.pprint(configuration_setup)

    # Add dataset to configuration
    configuration_setup['use__relative_strength_vs_benchmark'].update({'vix_dataframe': data_cache["^VIX"].copy()})
    configuration_setup['use__relative_strength_vs_benchmark'].update({'spx_dataframe': data_cache["^GSPC"].copy()})

    stocks = list(data_cache.keys())
    # Variables partagées pour transmettre l'information aux workers
    stocks__shared, master_cmd__shared = Queue(len(stocks)), Value("i", 0)
    out__shared = [Queue(1) for k in range(0, NB_WORKERS)]

    # Lancement des workers
    for k in range(0, NB_WORKERS):
        p = Process(target=_worker_processor, args=(stocks__shared, master_cmd__shared, out__shared[k],))
        p.start()
        pid = p.pid
        p_obj = psutil.Process(pid)
        p_obj.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    # Envoie les informations aux workers pour traitement
    # print(f"Preparations de {len(data_cache.items())} annotations...")
    for k, v in data_cache.items():
        stocks__shared.put({k: deepcopy(v)})

    # Autoriser les workers à traiter
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1

    # Récupération des résultats
    data_from_workers = []
    for k in range(0, NB_WORKERS):
        data_from_workers.extend(out__shared[k].get())

    # Affichage des résultats
    _update_alerts_ui_print_only(data_from_workers)

if __name__ == "__main__":
    freeze_support()
    entry()