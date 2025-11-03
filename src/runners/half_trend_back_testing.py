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
import os
import argparse
import pathlib
import json
import numpy as np
import pickle
import pprint
from tqdm import tqdm
from algorithms.trade_prime_half_trend import trade_prime_half_trend_strategy, get_entry_type, get_volume_confirmed, get_higher_timeframe_strong_trend, get_relative_strength_vs_benchmark, get_candlestick_confirmation_pattern
from algorithms.trade_prime_half_trend import set_volumed_confirmed
from datetime import datetime, timedelta, date
import os
import time
from utils import format_execution_time, get_weekdays
from copy import deepcopy
from multiprocessing import freeze_support, Lock, Process, Queue, Value
from constants import FYAHOO__OUTPUTFILENAME, NB_WORKERS
import pickle
import psutil


def _update_alerts_ui_print_only(setup_log, configuration_setup):
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    GRAY = '\033[90m'
    YELLOW = '\033[93m'

    # Clear screen (optional, for cleaner output)
    print("\033[2J\033[H", end="")  # Clears terminal and moves cursor to top-left

    now = datetime.now()
    today_date = now.date()
    print(f"{GRAY}@{now.strftime('%Y-%m-%d %H:%M')}{RESET}")

    line = f"{get_entry_type(**configuration_setup)}"
    volume_confirmed__enabled, volume_confirmed__window_size = get_volume_confirmed(**configuration_setup)
    line += f"{(', Volume Confirmed: '+str(volume_confirmed__window_size)+'h' if volume_confirmed__enabled else '')}"
    higher_timeframe_strong_trend__enabled, higher_timeframe_strong_trend__length = get_higher_timeframe_strong_trend(**configuration_setup)
    line += f"{(', Higher TimeFrame Strong Trend:'+str(higher_timeframe_strong_trend__length)+'h' if higher_timeframe_strong_trend__enabled else '')}"
    (use_vix, pd_v, vix_values), (use_spx, pd_s, spx_values) = get_relative_strength_vs_benchmark(**configuration_setup)
    line += f"{(', VIX bias ('+str(pd_v)+' hours)' if use_vix else '')}"
    line += f"{(', SPX bias ('+str(pd_s)+' hours)' if use_spx else '')}"
    use__candlestick_formation_pattern = get_candlestick_confirmation_pattern(**configuration_setup)
    line += f"{(', Candlestick Pattern' if use__candlestick_formation_pattern else '')}"
    print(f"{YELLOW}Config:{RESET} {line}")
    print()  # Blank line for separation
    line_results = ''
    # Get all ticker
    all_tickers = sorted(list(set([list(set(one_setup['ticker_name'].values))[0] for one_setup in setup_log])))
    for ticker_name in all_tickers:
        trade_success__buy_setup, trade_fail__buy_setup, p_and_l__sell_setup, p_and_l__buy_setup = 0, 0, [], []
        trade_success__sell_setup, trade_fail__sell_setup = 0, 0
        assert 2 == len([slg for slg in setup_log if slg['ticker_name'].values[0] == ticker_name])  # A buy setup and a sell setup
        for one_setup in [slg for slg in setup_log if slg['ticker_name'].values[0] == ticker_name]:
            setup_triggered = one_setup[one_setup[('setup_triggered', ticker_name)]].copy()
            custom_signal = one_setup[one_setup[('custom_signal', ticker_name)]].copy()
            assert len(setup_triggered) >= len(custom_signal)  # Not all triggers become trade
            the_setup_mode = None  # Buy or Sell
            for i in range(0, len(custom_signal)):
                the_date = custom_signal.index[i]
                is_buy_setup = custom_signal.iloc[i]['is_buy_setup'].values[0] == True
                if the_setup_mode is None:
                    the_setup_mode = is_buy_setup
                else:
                    assert the_setup_mode == is_buy_setup
                entry_point = custom_signal.iloc[i]['entry_price'].values[0]
                execution_price = custom_signal.iloc[i]['execution_price'].values[0]
                stop_loss = custom_signal.iloc[i]['stop_loss'].values[0]
                take_profit = custom_signal.iloc[i]['take_profit'].values[0]
                assert 1 == len(custom_signal.iloc[i]['execution_price'].values)  # Sanity check
                assert (stop_loss - entry_point <= 0) if is_buy_setup else (stop_loss - entry_point >= 0)
                if np.isnan(execution_price):
                    if is_buy_setup:
                        trade_fail__buy_setup += 1
                        assert stop_loss - entry_point <= 0
                        p_and_l__buy_setup.append((stop_loss - entry_point) / entry_point)
                    else:
                        trade_fail__sell_setup += 1
                        assert entry_point - stop_loss <= 0
                        p_and_l__sell_setup.append((entry_point - stop_loss) / entry_point)
                else:
                    assert (take_profit - entry_point >= 0) if is_buy_setup else (take_profit - entry_point <= 0)
                    if is_buy_setup:
                        trade_success__buy_setup += 1
                        assert take_profit - entry_point >= 0
                        p_and_l__buy_setup.append((take_profit - entry_point) / entry_point)
                    else:
                        trade_success__sell_setup += 1
                        assert entry_point - take_profit >= 0
                        p_and_l__sell_setup.append((entry_point - take_profit) / entry_point)
                # print(f"[{ticker_name} : {'BUY' if is_buy_setup else 'SELL'}]  {the_date}  {entry_point:0.2f}$ >> {execution_price:0.2f}$  ")
            if the_setup_mode:
                p_and_l = np.array(p_and_l__buy_setup).mean()*100.
                assert len(np.array(p_and_l__buy_setup)[np.array(p_and_l__buy_setup) < 0]) == trade_fail__buy_setup
                assert len(np.array(p_and_l__buy_setup)[np.array(p_and_l__buy_setup) > 0]) == trade_success__buy_setup
                line_results += f"[{ticker_name} : {len(p_and_l__buy_setup)} BUY SETUP]     P/L : {trade_success__buy_setup/(trade_fail__buy_setup+trade_success__buy_setup)*100:0.2f}%  >> {p_and_l:0.2f}%" + "\n"
            else:
                p_and_l = np.array(p_and_l__sell_setup).mean()*100.
                assert len(np.array(p_and_l__sell_setup)[np.array(p_and_l__sell_setup) < 0]) == trade_fail__sell_setup
                assert len(np.array(p_and_l__sell_setup)[np.array(p_and_l__sell_setup) > 0]) == trade_success__sell_setup
                line_results += f"[{ticker_name} : {len(p_and_l__sell_setup)} SELL SETUP]     P/L : {trade_success__sell_setup / (trade_fail__sell_setup + trade_success__sell_setup) * 100:0.2f}%  >> {p_and_l:0.2f}%" + "\n"
    print(line_results)


def _update_alerts_ui_print_only_2(setup_log, configuration_setup):
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    GRAY = '\033[90m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'

    # Clear screen (optional)
    print("\033[2J\033[H", end="")

    now = datetime.now()
    print(f"{GRAY}@ {now.strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")

    # Build and print config line
    config_parts = [get_entry_type(**configuration_setup)]

    volume_enabled, vol_window = get_volume_confirmed(**configuration_setup)
    if volume_enabled:
        config_parts.append(f"Volume Confirmed: {vol_window}h")

    htft_enabled, htft_len = get_higher_timeframe_strong_trend(**configuration_setup)
    if htft_enabled:
        config_parts.append(f"HTF Strong Trend: {htft_len}h")

    (use_vix, pd_v, _), (use_spx, pd_s, _) = get_relative_strength_vs_benchmark(**configuration_setup)
    if use_vix:
        config_parts.append(f"VIX bias ({pd_v}h)")
    if use_spx:
        config_parts.append(f"SPX bias ({pd_s}h)")

    if get_candlestick_confirmation_pattern(**configuration_setup):
        config_parts.append("Candlestick Pattern")

    print(f"{YELLOW}Config:{RESET} {', '.join(config_parts)}\n")

    # Process results per ticker
    all_tickers = sorted({list(one_setup['ticker_name'].values)[0] for one_setup in setup_log})
    result_lines = []

    for ticker_name in all_tickers:
        setups = [slg for slg in setup_log if slg['ticker_name'].values[0] == ticker_name]
        assert len(setups) == 2, f"Expected 2 setups (buy + sell) for {ticker_name}"

        for one_setup in setups:
            setup_triggered = one_setup[one_setup[('setup_triggered', ticker_name)]].copy()
            custom_signal = one_setup[one_setup[('custom_signal', ticker_name)]].copy()
            assert len(setup_triggered) >= len(custom_signal)

            is_buy = custom_signal.iloc[0]['is_buy_setup'].values[0] if len(custom_signal) > 0 else None
            if is_buy is None:
                continue

            trades = []
            for i in range(len(custom_signal)):
                row = custom_signal.iloc[i]
                entry = row['entry_price'].values[0]
                exec_price = row['execution_price'].values[0]
                sl = row['stop_loss'].values[0]
                tp = row['take_profit'].values[0]
                date = custom_signal.index[i]

                # print(f"[{ticker_name} : {'BUY' if is_buy else 'SELL'}]  {date.strftime('%Y-%m-%d %H:%M')}  {entry:6.2f}$ ➔ {exec_price:6.2f}$")

                if np.isnan(exec_price):
                    pnl = ((sl - entry) / entry) if is_buy else ((entry - sl) / entry)
                    trades.append((False, pnl))
                else:
                    pnl = ((tp - entry) / entry) if is_buy else ((entry - tp) / entry)
                    trades.append((True, pnl))

            n_trades = len(trades)
            successes = sum(1 for t in trades if t[0])
            failures = n_trades - successes
            win_rate = (successes / n_trades * 100) if n_trades > 0 else 0
            avg_pnl = np.mean([t[1] for t in trades]) * 100 if n_trades > 0 else 0

            side = "BUY" if is_buy else "SELL"
            color = GREEN if avg_pnl >= 0 else RED
            result_line = (
                f"{CYAN}[{ticker_name}]{RESET} "
                f"{n_trades:2d} {side:<4} setups | "
                f"Win Rate: {win_rate:5.1f}% | "
                f"Avg P/L: {color}{avg_pnl:+6.2f}%{RESET}"
            )
            result_lines.append(result_line)

    print("\n" + "=" * 70)
    print(f"{YELLOW}SUMMARY{RESET}".center(70))
    print("=" * 70)
    for line in result_lines:
        print(line)


def _worker_processor(stocks__shared, master_cmd__shared, out__shared, configuration_setup__shared, ):
    _debug = False
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
    configuration_setup = configuration_setup__shared.get()

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
            kwargs = deepcopy(configuration_setup)
            df = trade_prime_half_trend_strategy(ticker_df=data_cache[stock].copy(), ticker_name=stock, buy_setup=buy_setup, **kwargs)
            results.append(df.copy())
    out__shared.put(results)
    if _debug:
        print(f"[{os.getpid()}:{datetime.now()} Terminating")


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description')
    # Compute default config path relative to this script's directory
    script_dir = pathlib.Path(__file__).resolve().parent.parent
    default_config = os.path.join(script_dir, "config", "half_trend", "default.json")
    parser.add_argument('-c', '--config',
                        required=False,
                        default=str(default_config),
                        help='Path to the configuration file (default: config/half_trend/default.json relative to script)')
    parser.add_argument('-s', '--rmstock',
                        required=False,
                        default=None,
                        help='Use only the specified stock')
    # Handle both --volumed_confirmed and --volumed_confirmed=20
    parser.add_argument('--volume_confirmed',
                        nargs='?',
                        const=20,  # Default value when flag is used without =value
                        type=int,
                        default=None,  # Default when flag is not used at all
                        help='Enable volume confirmation with optional threshold (default: 20 if flag used)')

    return parser.parse_args()


def entry():
    args = parse_args()
    config_file_path = args.config
    rmstock = args.rmstock
    volume_confirmed = args.volume_confirmed

    # Check if the config file exists
    if not os.path.exists(config_file_path):
        # Look in directory config/half_trend/
        config_file_path = os.path.join(parent_dir, 'config', 'half_trend', f'{config_file_path}.json')
        if not os.path.exists(config_file_path):
            print(f"Config file '{config_file_path}' not found.")
            return

    with open(FYAHOO__OUTPUTFILENAME, 'rb') as f:
        data_cache = pickle.load(f)

    with open(config_file_path, 'r') as f:
        configuration_setup = json.load(f)

    # pprint.pprint(configuration_setup)

    # Add dataset to configuration
    configuration_setup['use__relative_strength_vs_benchmark'].update({'vix_dataframe': data_cache["^VIX"].copy()})
    configuration_setup['use__relative_strength_vs_benchmark'].update({'spx_dataframe': data_cache["^GSPC"].copy()})

    # Remove all keys that are not 'rmstock'
    if rmstock is not None:
        if rmstock in data_cache:
            data_cache = {rmstock: data_cache[rmstock], "^VIX": data_cache["^VIX"], "^GSPC": data_cache["^GSPC"]}

    stocks = list(data_cache.keys())
    # Variables partagées pour transmettre l'information aux workers
    stocks__shared, configuration_setup__shared, master_cmd__shared = Queue(len(stocks)), [Queue(1) for k in range(0, NB_WORKERS)], Value("i", 0)
    out__shared = [Queue(1) for k in range(0, NB_WORKERS)]

    # Lancement des workers
    for k in range(0, NB_WORKERS):
        p = Process(target=_worker_processor, args=(stocks__shared, master_cmd__shared, out__shared[k], configuration_setup__shared[k], ))
        p.start()
        pid = p.pid
        p_obj = psutil.Process(pid)
        p_obj.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    # Envoie les informations aux workers pour traitement
    for k, v in data_cache.items():
        stocks__shared.put({k: deepcopy(v)})
    for k in range(0, NB_WORKERS):
        configuration_setup__shared[k].put(configuration_setup)

    # Autoriser les workers à traiter
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1

    # Récupération des résultats
    data_from_workers = []
    for k in range(0, NB_WORKERS):
        data_from_workers.extend(out__shared[k].get())

    # Affichage des résultats
    _update_alerts_ui_print_only(data_from_workers, configuration_setup)
    _update_alerts_ui_print_only_2(data_from_workers, configuration_setup)


if __name__ == "__main__":
    freeze_support()
    entry()