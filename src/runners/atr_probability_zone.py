try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import os
import argparse
import math
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import optuna
import joblib
import time
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import argparse
import sys
import psutil
from argparse import Namespace
import os
import sys
import time
from runners.atr import entry as atr_entry
from utils import WEBEARStyle, send_html_email


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dataset-id", type=str, default="day")
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')

    return parser


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
        try:
            item = use_cases__shared.get(timeout=0.1)
            use_case_batch.append(item)
        except:
            break  # Queue is empty or no more items within timeout
        if 0 == len(use_case_batch):
            break
        for use_case in use_case_batch:
            atr_config = use_case
            result = atr_entry(args=atr_config)
            predicted_high = math.ceil((result['realtime']['predicted_high'] + 0.01) / 5) * 5
            predicted_low = math.floor((result['realtime']['predicted_low'] + 0.01) / 5) * 5
            vix_regime = result['realtime']['vix_regime']
            probability_predicted_high = result['regime_metrics'][vix_regime]['Borne Haute Respectée']
            probability_predicted_low = result['regime_metrics'][vix_regime]['Borne Basse Respectée']
            actual_high, actual_low, actual_close, actual_open = result['realtime']['actual_high'], result['realtime']['actual_low'], result['realtime']['actual_close'], result['realtime']['actual_open']
            atr_config.dataframe=None
            all_results_computed.append({'actual_high': actual_high, 'actual_low': actual_low, 'actual_close': actual_close, 'actual_open': actual_open,
                                         'predicted_high': predicted_high, 'predicted_low': predicted_low,
                                         'probability_predicted_high': probability_predicted_high, 'probability_predicted_low': probability_predicted_low,
                                         'atr_config': atr_config})
    out__shared.put(all_results_computed)


def entry(args):
    nb_worker = 15
    verbose = True
    n_trials = 999

    # Obtention du dataframe en realtime
    atr_config = Namespace(ticker=args.ticker, dataset_id=args.dataset_id, dataframe=None, verbose=False, n_trials=9, use_realtime_data=True, atr_window=14,
                           n_split=0.9, tightness_weight=0., use_close_for_range=True, clip_n=0, timeout=9999)
    result = atr_entry(args=atr_config)
    dataframe = result['dataframe_and_cols']

    # Données des workers
    data_from_workers = []

    # Construction des cas à traiter
    use_cases = []
    tightness_weights = np.linspace(0, 1, 30).tolist()
    for tightness_weight in tightness_weights + [2., 4., 8., 16.]:
        atr_config = Namespace(ticker=args.ticker, dataset_id=args.dataset_id, dataframe=dataframe, verbose=False, n_trials=n_trials, use_realtime_data=True, atr_window=14,
                               n_split=0.9, tightness_weight=tightness_weight, use_close_for_range=False, clip_n=0, timeout=9999)
        use_cases.append(atr_config)

    use_cases__shared, master_cmd__shared = Queue(256000), Value("i", 0)
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
    actual_high, actual_low, actual_close, actual_open = next(({k: v for k, v in item.items() if k in ['actual_high', 'actual_low', 'actual_close', 'actual_open']}.values() for item in data_from_workers))
    subject = (f"[REALTIME @{datetime.now().strftime("%Y%m%d_%H%M")}]   O:{actual_open:.0f} H:{actual_high:.0f} L:{actual_low:.0f} C:{actual_close:.0f}   [BREAK EVEN ON 5-POINT WIDE SPREAD]")
    string_generated = subject + "\n"
    for col_for_sort in ["predicted_low", "predicted_high"]:
        # 1. Tri du plus GRAND au plus PETIT
        liste_triee = sorted(data_from_workers, key=lambda x: x[f"probability_{col_for_sort}"], reverse=True)
        # 2. Le dictionnaire garde la DERNIÈRE valeur lue (donc la plus basse)
        dictionnaire_unique = {elem[col_for_sort]: elem for elem in liste_triee}
        # 3. Récupération du résultat final sous forme de liste
        sorted_probabilities_for_predicted_ = list(dictionnaire_unique.values())
        # Liste pour stocker les éléments filtrés
        filtered_probabilities = []
        for i in range(len(sorted_probabilities_for_predicted_)):
            current = sorted_probabilities_for_predicted_[i]

            # Si on n'est pas au premier élément, on compare avec le précédent (i - 1)
            if i > 0:
                previous = sorted_probabilities_for_predicted_[i - 1]

                same_low = current['probability_predicted_low'] == previous['probability_predicted_low']
                same_high = current['probability_predicted_high'] == previous['probability_predicted_high']

                # Si les deux probabilités sont identiques au précédent, on l'ignore
                if same_low and same_high:
                    continue

            # On garde l'élément s'il est unique par rapport au précédent
            filtered_probabilities.append(current)
        spread_width = 500.0
        actual_high, actual_low, actual_close, actual_open = next(({k: v for k, v in item.items() if k in ['actual_high', 'actual_low', 'actual_close', 'actual_open']}.values() for item in filtered_probabilities))
        if 'low' in col_for_sort:
            string_generated += (f"\t::  {WEBEARStyle.BOLD}Low  @P    BE$   {WEBEARStyle.END}")+ "\n"
        else:
            string_generated += (f"\t::  {' ':<16}{WEBEARStyle.BOLD}High @P    BE${WEBEARStyle.END}")+ "\n"
        for sorted_probability in filtered_probabilities:
            predicted_low, predicted_high = int(sorted_probability['predicted_low']), int(sorted_probability['predicted_high'])
            probability_predicted_low, probability_predicted_high = int(sorted_probability['probability_predicted_low']), int(sorted_probability['probability_predicted_high'])
            assert actual_high == sorted_probability['actual_high']
            assert actual_low == sorted_probability['actual_low']
            assert actual_close == sorted_probability['actual_close']
            assert actual_open == sorted_probability['actual_open']
            breakeven_high = int((1.0 - probability_predicted_high/100.) * spread_width)
            breakeven_low = int((1.0 - probability_predicted_low/100.) * spread_width)
            if 'low' in col_for_sort:
                string_generated += (f"\t"                      
                      f"{' ':<4}{predicted_low:04d} @{probability_predicted_low:02d}% {breakeven_low:03d}$")+ "\n"
            else:
                string_generated += (f"\t"                      
                      f"{' ':<20}{predicted_high:04d} @{probability_predicted_high:02d}% {breakeven_high:03d}$")+ "\n"
    print(string_generated)
    send_html_email(destinataire="luccauchon@gmail.com", sujet=subject, corps=string_generated)
    return None


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)