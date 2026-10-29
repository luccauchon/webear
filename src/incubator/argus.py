'''
En référence au géant de la mythologie grecque doté de cent yeux. Parfait pour un script qui surveille simultanément une multitude d'indicateurs de marché.
'''
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
import time
from argparse import Namespace
from multiprocessing import freeze_support, Lock, Process, Queue, Value
from constants import TITLE_WEBEAR, HISTORIAN_ARGUS
from runners.atr_probability_zone import entry as atr_probability_zone
from optimizers.quantloop.quant_morning_alpha_miner import entry as quant_morning_alpha_miner
from optimizers.quantloop.quant_morning_beta_miner_realtime_and_backtesting import entry as quant_morning_beta_miner
from runners.streak_probability_informer import entry as streak_probability_entry
from crusaders.SPX_drop.player import entry as spx_drop_player_entry
from runners.conditional_gap_persistence import calculate_sp500_probabilities as conditional_gap_persistence_entry
from optimizers.quantloop.quant_morning_alpha_miner import TopNEnsembleModel
from optimizers.vera.realtime_and_backtest_hyperparameter_search_optuna import entry as vera_entry_point
import sys
import re
import os
import random
import subprocess
from datetime import datetime
from utils import get_next_step, get_month_range,get_week_range, minutes_jusqua_16h, get_quarter_range, get_year_range
from fetchers.data_factory import factory_load_data
import schedule
import traceback
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
from scipy.integrate import trapezoid
import pyarrow as pa
import pyarrow.parquet as pq
import copy
from datetime import datetime, timedelta


def clean_console():
    if os.name == 'nt':
        subprocess.run(['cls'], shell=True)
    else:
        subprocess.run(['clear'])


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared):
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 1 == master_cmd__shared.value:
                break
        time.sleep(0.333)

    # Traitement des requêtes
    request_batch_size = 1
    while True:
        use_case_batch = []
        while len(use_case_batch) < request_batch_size:
            try:
                item = use_cases__shared.get(timeout=0.123)
                use_case_batch.append(item)
            except:
                break  # Queue is empty or no more items within timeout
        with master_cmd__shared.get_lock():
            if 2 == master_cmd__shared.value:
                break  # On quitte
        if 0 == len(use_case_batch):
            #out__shared.put({"info_computing": f""})
            continue
        all_results_computed = []
        for use_case in use_case_batch:
            try:
                assert 1 == len(use_case)
                k, v = next(iter(use_case.items()))
                #print(f"[{os.getpid()}] ------Processing {k} --------------{v}")
                #out__shared.put({"info_computing": f"[{os.getpid()}] Processing {k}:{v}"})
                if k == "bouclier_spx":
                    res_bouclier_spx = spx_drop_player_entry(args_player=Namespace(update_dataset=v["update_dataset"], production_setup=False, verbose=False, send_email=False))
                    all_results_computed.append({"bouclier_spx": res_bouclier_spx})
                if k == "trend_computation":
                    df__60min  = factory_load_data(_dataset_id=f"intraday-60min", _ticker=v["ticker"], _args={"realtime": True})
                    df__240min = factory_load_data(_dataset_id=f"intraday-240min", _ticker=v["ticker"], _args={"realtime": True})
                    df__day    = factory_load_data(_dataset_id=f"day", _ticker=v["ticker"], _args={"realtime": True})
                    df__week   = factory_load_data(_dataset_id=f"week", _ticker=v["ticker"], _args={"realtime": True})
                    df__month  = factory_load_data(_dataset_id=f"month", _ticker=v["ticker"], _args={"realtime": True})
                    all_results_computed.append({"trend_computation":{"df__60min": df__60min, "df__240min": df__240min, "df__day": df__day, "df__week": df__week, "df__month": df__month}})
                if k == "realtime_values":
                    df = factory_load_data(_dataset_id="day", _ticker=v["ticker"], _args={"realtime": True})
                    open_col = ('Open', v["ticker"])
                    close_col = ('Close', v["ticker"])
                    high_col = ('High', v["ticker"])
                    low_col = ('Low', v["ticker"])
                    volume_col = ('Volume', v["ticker"])
                    dict_tmp = {"realtime_values": {'close_t_m1': df[close_col].iloc[-2], 'open': df[open_col].iloc[-1], 'high': df[high_col].iloc[-1],
                                                    'low': df[low_col].iloc[-1], 'close': df[close_col].iloc[-1],}}
                    df = factory_load_data(_dataset_id="intraday-1min", _ticker=v["ticker"], _args={"realtime": True})
                    # 1. Filtrer le DataFrame pour la dernière journée à partir de 9h30
                    last_day_df = df.loc[df.index.date == df.index.date[-1]].between_time("09:30", "16:00").copy()

                    # 2. Extraire les colonnes nécessaires pour cette journée
                    high_series = last_day_df[high_col]
                    low_series = last_day_df[low_col]
                    close_series = last_day_df[close_col]
                    volume_series = last_day_df[volume_col]

                    # 3. Calculer le Prix Typique (Typical Price)
                    typical_price = (high_series + low_series + close_series) / 3

                    # 4. Calculer le VWAP réinitialisé (les cumsum repartent de zéro à 9h30)
                    vwap_reset = (typical_price * volume_series).cumsum() / volume_series.cumsum()

                    # 5. Calculer l'écart entre le prix de clôture et le nouveau VWAP
                    delta = (close_series - vwap_reset).to_numpy()

                    # 6. Isoler les écarts (au-dessus vs en dessous)
                    delta_above = np.where(delta > 0, delta, 0)
                    delta_below = np.where(delta < 0, np.abs(delta), 0)

                    # 7. Calcul de l'aire sous la courbe (dx=1 pour un pas de 1 minute)
                    aire_au_dessus = trapezoid(delta_above, dx=1)
                    aire_en_dessous = trapezoid(delta_below, dx=1)

                    #print(f"Aire quand le prix est AU-DESSUS du VWAP : {aire_au_dessus:.4f}")
                    #print(f"Aire quand le prix est EN DESSOUS du VWAP : {aire_en_dessous:.4f}")
                    dict_tmp["realtime_values"].update({"energy_above_vwap": aire_au_dessus, "energy_below_vwap": aire_en_dessous})
                    all_results_computed.append(dict_tmp)
                if k == "horizontal_lines":
                    data_from_atr_probability_zone = atr_probability_zone(args=Namespace(production_setup=False, ticker=v["ticker"], dataset_id=v["dataset_id"],
                                                                                         n_split=v["n_split"], n_trials=v["n_trials"],
                                                                                         use_close_for_range=True, no_email=True, verbose=False,
                                                                                         tightness_weights=v["tightness_weights"], nb_worker=v["nb_worker"]))
                    all_results_computed.append({v["key_id"]: data_from_atr_probability_zone})
                if k == "vera":
                    min_16h = minutes_jusqua_16h()
                    time.sleep(60)  # Make sure that we don't get "Exception: Realtime mode activated. Please run after 10:20:00." exception
                    vera_results = vera_entry_point(args=Namespace(realtime=True, intraday_candle_space=min_16h, ticker=v["ticker"], verbose_print_results=False,
                                                                   tightness_weight=v["tightness_weight"], n_trials=v["n_trials"], n_split=v["n_split"], ))
                    assert 1 == len(vera_results["inf_resulting"])  # Realtime
                    vera_results["inf_resulting"][0].update({"minutes_jusqua_16h": min_16h})
                    all_results_computed.append({k: vera_results["inf_resulting"][0]})
                if k == "gap":
                    gap_results = conditional_gap_persistence_entry(args=Namespace(ticker=v["ticker"], epsilon=0., dataset_id="day", use_realtime_data=True, verbose=False))
                    all_results_computed.append({k: gap_results["tech_info"]})
                if k == "day_streaker":
                    streaker_results = streak_probability_entry(args=Namespace(ticker=v["ticker"], dataset_id=v["dataset_id"], send_email=False, realtime=v["realtime"], production_setup=False, clip_n=v["clip_n"]))
                    all_results_computed.append({k: (streaker_results["proba_txt"] , {"realtime": v["realtime"]}) })
                if k == "week_streaker":
                    streaker_results = streak_probability_entry(args=Namespace(ticker=v["ticker"], dataset_id=v["dataset_id"], send_email=False, realtime=v["realtime"], production_setup=False, clip_n=v["clip_n"]))
                    all_results_computed.append({k: (streaker_results["proba_txt"] , {"realtime": v["realtime"]}) })
                if k == "quant":
                    result_quant = quant_morning_beta_miner(args=Namespace(realtime=True, ticker=v["ticker"], st1=v["st1"], st2=v["st2"], et1=v["et1"], et2=v["et2"], seuil_pos=v["seuil_pos"], models_dir=v["models_dir"]))
                    action, strike_price, win_rate, timestamp = result_quant["action"], result_quant["strike_price"], result_quant["win_rate"], datetime.now().time()
                    tmp_lst = [{'strike_price': int(strike_price), 'win_rate': win_rate, 'action': action, 'timestamp': timestamp}]
                    all_results_computed.append({k: (str(int(strike_price)), tmp_lst)})
                if k == "cleaner":
                    pass
            except Exception as eee:
                print(eee)
                traceback.print_exc()
        out__shared.put(all_results_computed)

    # Quit


def entry():
    ticker = "^GSPC"
    open_col  = ('Open', ticker)
    close_col = ('Close', ticker)
    high_col  = ('High', ticker)
    low_col   = ('Low', ticker)

    starter_auth = {'bouclier_spx': True, "realtime": True, "trend_ema": True, "horizontal_lines": True, "vera": True,
                    "splash_screen": True,
                    "gap": False,  # Pas utilise
                    "day_streaker": True, "week_streaker": True, "historian": True, "quant": True,
                    "activer_console": True, "activer_random_update": False, }

    vera__tightness_weight, vera__n_trials, vera__n_split = 0.33, 250, 0.9
    horizontal_lines__nb_worker, horizontal_lines__tightness_weights, horizontal_lines__n_trials = 4, [0., 0.3, 0.99, 2, 16], 500
    horizontal_lines__max_horizontal_lines, horizontal_lines__max_horizontal_lines_h0 = 3, 5
    historian__output_dir, historian__accumulator = HISTORIAN_ARGUS, []
    quant__st2 = ("09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:30", "13:00","13:30", "14:00", "15:00")
    quant__seuil_pos = (0.999, 0.9975, 0.9950, 0.9925, 0.9900, 1.001, 1.0025, 1.0050, 1.0075, 1.0100)
    argus__nb_worker = 2

    if starter_auth["activer_console"]: clean_console()
    argus__information_displayed = {}  # Data used to create and update the display
    use_cases__shared, master_cmd__shared = Queue(640), Value("i", 0)
    out__shared = [Queue(640) for k in range(0, argus__nb_worker)]
    # Lancement des workers
    for k in range(0, argus__nb_worker):
        p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k],))
        p.start()
    # Autoriser les workers à traiter
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1

    ###########################################################################
    #
    ###########################################################################
    def make_display(info):
        console_length = 160
        actual_close, actual_open, actual_high, actual_low = float(info['close']), float(info['open']), float(info['high']), float(info['low'])
        first_indent = " " * 12
        second_indent = " " * 24
        second_indent_pp = " " * 36
        saut_ligne = "\n"
        gras = "\033[1m"
        bleu = "\033[34m"
        vert = "\033[32m"
        rouge = "\033[31m"
        rouge_clair = "\033[91m"
        jaune = "\033[33m"
        jaune_clair = "\033[93m"
        jaune_fonce = "\033[38;5;142m"  # Jaune plus sombre/moutarde (mode 256 couleurs)
        # Orange
        orange = "\033[38;5;208m"  # Orange standard
        orange_clair = "\033[38;5;215m"  # Orange pastel/clair
        orange_fonce = "\033[38;5;166m"  # Orange brique/sombre
        barre = "\033[9m"
        gris = "\033[90m"
        gris_moyen = "\033[38;5;244m"
        gris_fonce = "\033[38;5;239m"
        plus_ou_moins = "±"
        reset = "\033[0m"
        # Flèches simples
        fleche_haut = "↑"
        fleche_bas = "↓"

        # Flèches pleines (géométriques)
        fleche_haut_pleine = "▲"
        fleche_bas_pleine = "▼"

        previous_close = float(info["close_t_m1"])
        def compute_distance(a, b):
            if float(a) == float(b): return 0
            if 0 == float(b): return 0
            return (float(a)-float(b))/float(b)

        def trier_par_prix_decroissant(liste_dicts):
            return sorted(liste_dicts, key=lambda x: x["price"], reverse=True)

        def trier_par_prix_croissant(liste_dicts):
            return sorted(liste_dicts, key=lambda x: x["price"], reverse=False)

        def format_price(price):
            return f"{price:>5}"

        def format_proba(proba):
            return f"{proba:>4.0%}"

        def determine_color(actual_price, limit_price, dte, situation):
            price_a = actual_price if situation == "lower" else limit_price
            price_b = limit_price  if situation == "lower" else actual_price
            if dte == "h0":
                if compute_distance(price_a, price_b) < 0: return rouge_clair
                if compute_distance(price_a, price_b) < 0.001: return orange_fonce
                if compute_distance(price_a, price_b) < 0.0025: return orange_clair
                if compute_distance(price_a, price_b) < 0.0050: return jaune
            if dte == "h5":
                if compute_distance(price_a, price_b) < 0: return rouge_clair
            if dte == "hweek":
                if compute_distance(price_a, price_b) < 0: return rouge_clair
            if dte == "h10":
                if compute_distance(price_a, price_b) < 0: return rouge_clair
            if dte == "h15":
                if compute_distance(price_a, price_b) < 0: return rouge_clair
            if dte == "h20":
                if compute_distance(price_a, price_b) < 0: return rouge_clair
            if dte == "hmonth":
                if compute_distance(price_a, price_b) < 0: return rouge_clair
            return ""

        def make_bloc_line(texte_ln_c1, texte_ln_c2, avec_saut_ligne=True):
            return f"{first_indent*2}{texte_ln_c1}{second_indent_pp}{texte_ln_c2}{saut_ligne if avec_saut_ligne else ''}"

        def make_bloc_line_3(texte_ln_c1, texte_ln_c2, texte_ln_c3, avec_saut_ligne=True):
            return f"{first_indent*2}{texte_ln_c1}{second_indent_pp}{texte_ln_c2}{' '*8}{texte_ln_c3}{saut_ligne if avec_saut_ligne else ''}"

        def make_upper_limits(info):
            #######################################################################
            # QUARTER/YEAR/H80/H240
            #######################################################################
            def quarter_year_h80_h240():
                _bloc, _bloc_length = "", (console_length // 7, console_length // 7)
                dates, ee = [], 0
                for hquarter_data, h80_data, hyear_data, h240_data in zip(trier_par_prix_decroissant(info["hquarter_up"])[:info['max_horizontal_lines']],
                                                                          trier_par_prix_decroissant(info["h80_up"])[:info['max_horizontal_lines']],
                                                                          trier_par_prix_decroissant(info["hyear_up"])[:info['max_horizontal_lines']],
                                                                          trier_par_prix_decroissant(info["h240_up"])[:info['max_horizontal_lines']],
                                                                          ):
                    ee += 1
                    #  C1
                    limit_price, proba, date = hquarter_data["price"], hquarter_data["proba"], hquarter_data["date"]
                    dates.append(date)
                    texte_l1_c1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="hquarter", situation="upper")}{texte_l1_c1}{reset}"

                    #  C2
                    limit_price, proba, date = h80_data["price"], h80_data["proba"], h80_data["date"]
                    dates.append(date)
                    texte_l1_c2 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c2 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h80", situation="upper")}{texte_l1_c2}{reset}"

                    #  C3
                    limit_price, proba, date = hyear_data["price"], hyear_data["proba"], hyear_data["date"]
                    dates.append(date)
                    texte_l1_c3 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c3 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="hyear", situation="upper")}{texte_l1_c3}{reset}"

                    #  C4
                    limit_price, proba, date = h240_data["price"], h240_data["proba"], h240_data["date"]
                    dates.append(date)
                    texte_l1_c4 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c4 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h240", situation="upper")}{texte_l1_c4}{reset}"

                    _bloc += f"  {texte_l1_c1}{' '*int(console_length//10)}{texte_l1_c2}{' '*int(console_length//5)}{texte_l1_c3}{' '*int(console_length//11)}{texte_l1_c4}{saut_ligne if ee < info['max_horizontal_lines'] else ''}"
                _bloc_result  = f" QUARTER ({dates[0]}){' '*int(console_length//18)}H80 ({dates[1]}){' '*int(console_length//6)}"
                _bloc_result += f" YEAR ({dates[2]}){' ' * int(console_length // 18)}H240 ({dates[3]}){' ' * int(console_length // 6)}"
                _bloc_result += saut_ligne + _bloc
                return _bloc_result

            #######################################################################
            # MONTH/H20
            #######################################################################
            def month_h20():
                bloc_monthH20, bloc_monthH20_length = "", (console_length//7, console_length//7)
                dates = []
                for hmonth_data, h20_data in zip(trier_par_prix_decroissant(info["hmonth_up"])[:info['max_horizontal_lines']], trier_par_prix_decroissant(info["h20_up"])[:info['max_horizontal_lines']]):
                    #  C1
                    limit_price, proba, date = hmonth_data["price"], hmonth_data["proba"], hmonth_data["date"]
                    dates.append(date)
                    texte_l1_c1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="hmonth", situation="upper")}{texte_l1_c1}{reset}"

                    #  C2
                    limit_price, proba, date = h20_data["price"], h20_data["proba"], h20_data["date"]
                    dates.append(date)
                    texte_l1_c2 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c2 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h20", situation="upper")}{texte_l1_c2}{reset}"

                    bloc_monthH20 += make_bloc_line(texte_l1_c1, texte_l1_c2)
                bloc_monthH20 = f"{first_indent}{' ' * int(bloc_monthH20_length[0]//2)}MONTH ({dates[0]}){second_indent}{' ' * int(bloc_monthH20_length[1] // 3 - 2)}H20 ({dates[1]}){' ' * (int(bloc_monthH20_length[1]//2) - 3)}" +saut_ligne+ bloc_monthH20
                return bloc_monthH20

            #######################################################################
            # H10/H15
            #######################################################################
            def h10_h15():
                bloc_H10H15, bloc_H10H15_length = "", (console_length//7, console_length//7)
                dates = []
                for h10_data, h15_data in zip(trier_par_prix_decroissant(info["h10_up"])[:info['max_horizontal_lines']], trier_par_prix_decroissant(info["h15_up"])[:info['max_horizontal_lines']]):
                    #  C1
                    limit_price, proba, date = h10_data["price"], h10_data["proba"], h10_data["date"]
                    dates.append(date)
                    texte_l1_c1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h10", situation="upper")}{texte_l1_c1}{reset}"

                    #  C2
                    limit_price, proba, date = h15_data["price"], h15_data["proba"], h15_data["date"]
                    dates.append(date)
                    texte_l1_c2 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c2 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h15", situation="upper")}{texte_l1_c2}{reset}"

                    bloc_H10H15 += make_bloc_line(texte_l1_c1, texte_l1_c2)
                bloc_H10H15 = f"{first_indent}{' ' * int(bloc_H10H15_length[0] // 2)}H10 ({dates[0]}){second_indent}{' ' * int(bloc_H10H15_length[1] // 3)}H15 ({dates[1]}){' ' * (int(bloc_H10H15_length[1] // 2) - 3)}" + saut_ligne + bloc_H10H15
                return bloc_H10H15

            #######################################################################
            # WEEK/H5
            #######################################################################
            def week_h5():
                bloc_weekH5, bloc_weekH5_length = "", (console_length//7, console_length//7)
                quant_dataset = [[],[],[]]  # Chaque colonne est représentée par une liste
                liste_des_strike_prices_tries = sorted(list(info["quant"].keys()), key=int)
                for e, strike_price in enumerate(liste_des_strike_prices_tries):
                    information_for_price_level = sorted(info["quant"][f"{strike_price}"], key=lambda x: x['timestamp'])
                    most_recent_information_for_price_level = information_for_price_level[-1]
                    strike_price, win_rate, action, timestamp = most_recent_information_for_price_level['strike_price'], most_recent_information_for_price_level['win_rate'], most_recent_information_for_price_level['action'], most_recent_information_for_price_level['timestamp']
                    if e in [0, 1, 2, 3, 4]: quant_dataset[1].append((strike_price, win_rate, action, timestamp))
                    if e in [5, 6, 7, 8, 9]: quant_dataset[2].append((strike_price, win_rate, action, timestamp))
                dates, ee = [], 0
                for hweek_data, h5_data, quant_data in zip(trier_par_prix_decroissant(info["hweek_up"])[:info['max_horizontal_lines']], trier_par_prix_decroissant(info["h5_up"])[:info['max_horizontal_lines']], quant_dataset):
                    ee += 1
                    #  C1
                    limit_price, proba, date = hweek_data["price"], hweek_data["proba"], hweek_data["date"]
                    dates.append(date)
                    texte_l1_c1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="hweek", situation="upper")}{texte_l1_c1}{reset}"

                    #  C2
                    limit_price, proba, date = h5_data["price"], h5_data["proba"], h5_data["date"]
                    dates.append(date)
                    texte_l1_c2 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c2 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h5", situation="upper")}{texte_l1_c2}{reset}"

                    # C3
                    texte_l1_c3 = ""
                    for e, a_quant_sub_data in enumerate(quant_data):
                        strike_price, win_rate, action, timestamp = a_quant_sub_data
                        tmp_str = f"{strike_price} ({format_proba(win_rate)})"
                        if action == "PUT_CREDIT_SPREAD": tmp_str = f"{vert}{fleche_haut_pleine}{tmp_str}{reset}"
                        elif action == "CALL_CREDIT_SPREAD": tmp_str = f"{rouge_clair}{fleche_bas_pleine}{tmp_str}{reset}"
                        elif action == "PASS": tmp_str = f"{gris_fonce}{barre}{fleche_haut_pleine}{tmp_str}{reset}"
                        texte_l1_c3 += f"{tmp_str}" if e == len(quant_data)-1 else f"{tmp_str} | "
                    # C1 + C2 + C3
                    bloc_weekH5 += make_bloc_line_3(texte_l1_c1, texte_l1_c2, texte_l1_c3, True if ee < info['max_horizontal_lines'] else False)
                qtrend_info = f"{rouge_clair}{fleche_bas_pleine}{reset}=call credit spread  {vert}{fleche_haut_pleine}{reset}=put credit spread" if info["display_qtrend_info"] else ""
                bloc_weekH5 = (f"{first_indent}{' ' * int(bloc_weekH5_length[0] // 2)}WEEK ({dates[0]}){second_indent}{' ' * int(bloc_weekH5_length[1] // 3 - 1)}H5 ({dates[1]}){' ' * (int(bloc_weekH5_length[1] // 2)-4)}" +
                               f"QTREND |{len(liste_des_strike_prices_tries)}| (updated @{info['quant_meta_info']['timestamp']}) {qtrend_info}" + saut_ligne + bloc_weekH5)
                return bloc_weekH5

            return quarter_year_h80_h240() + saut_ligne + month_h20() + saut_ligne + h10_h15() + saut_ligne + week_h5()

        def make_lower_limits(info):
            #######################################################################
            # QUARTER/YEAR/H80/H240
            #######################################################################
            def quarter_year_h80_h240():
                _bloc, _bloc_length = "", (console_length // 7, console_length // 7)
                dates, ee = [], 0
                for hquarter_data, h80_data, hyear_data, h240_data in zip(trier_par_prix_decroissant(info["hquarter_down"])[:info['max_horizontal_lines']],
                                                                          trier_par_prix_decroissant(info["h80_down"])[:info['max_horizontal_lines']],
                                                                          trier_par_prix_decroissant(info["hyear_down"])[:info['max_horizontal_lines']],
                                                                          trier_par_prix_decroissant(info["h240_down"])[:info['max_horizontal_lines']],
                                                                          ):
                    ee += 1
                    #  C1
                    limit_price, proba, date = hquarter_data["price"], hquarter_data["proba"], hquarter_data["date"]
                    dates.append(date)
                    texte_l1_c1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="hquarter", situation="upper")}{texte_l1_c1}{reset}"

                    #  C2
                    limit_price, proba, date = h80_data["price"], h80_data["proba"], h80_data["date"]
                    dates.append(date)
                    texte_l1_c2 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c2 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h80", situation="upper")}{texte_l1_c2}{reset}"

                    #  C3
                    limit_price, proba, date = hyear_data["price"], hyear_data["proba"], hyear_data["date"]
                    dates.append(date)
                    texte_l1_c3 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c3 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="hyear", situation="upper")}{texte_l1_c3}{reset}"

                    #  C4
                    limit_price, proba, date = h240_data["price"], h240_data["proba"], h240_data["date"]
                    dates.append(date)
                    texte_l1_c4 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c4 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h240", situation="upper")}{texte_l1_c4}{reset}"

                    _bloc += f"  {texte_l1_c1}{' ' * int(console_length // 10)}{texte_l1_c2}{' ' * int(console_length // 5)}{texte_l1_c3}{' ' * int(console_length // 11)}{texte_l1_c4}{saut_ligne if ee < info['max_horizontal_lines'] else ''}"
                _bloc_result = f" QUARTER ({dates[0]}){' ' * int(console_length // 18)}H80 ({dates[1]}){' ' * int(console_length // 6)}"
                _bloc_result += f" YEAR ({dates[2]}){' ' * int(console_length // 18)}H240 ({dates[3]}){' ' * int(console_length // 6)}"
                _bloc_result += saut_ligne + _bloc
                return _bloc_result

            #######################################################################
            # WEEK/H5
            #######################################################################
            def week_h5():
                bloc_weekH5, bloc_weekH5_length = "", (console_length//7, console_length//7)
                dates = []
                for hweek_data, h5_data in zip(trier_par_prix_decroissant(info["hweek_down"])[:info['max_horizontal_lines']], trier_par_prix_decroissant(info["h5_down"])[:info['max_horizontal_lines']]):
                    #  C1
                    limit_price, proba, date = hweek_data["price"], hweek_data["proba"], hweek_data["date"]
                    dates.append(date)
                    texte_l1_c1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="hweek", situation="lower")}{texte_l1_c1}{reset}"

                    #  C2
                    limit_price, proba, date = h5_data["price"], h5_data["proba"], h5_data["date"]
                    dates.append(date)
                    texte_l1_c2 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c2 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h5", situation="lower")}{texte_l1_c2}{reset}"

                    bloc_weekH5 += make_bloc_line(texte_l1_c1, texte_l1_c2)
                bloc_weekH5 = f"{first_indent}{' ' * int(bloc_weekH5_length[0] // 2)}WEEK ({dates[0]}){second_indent}{' ' * int(bloc_weekH5_length[1] // 3 - 1)}H5 ({dates[1]}){' ' * (int(bloc_weekH5_length[1] // 2) - 2)}" + saut_ligne + bloc_weekH5
                return bloc_weekH5

            #######################################################################
            # H10/H15
            #######################################################################
            def h10_h15():
                bloc_H10H15, bloc_H10H15_length = "", (console_length//7, console_length//7)
                dates = []
                for h10_data, h15_data in zip(trier_par_prix_decroissant(info["h10_down"])[:info['max_horizontal_lines']], trier_par_prix_decroissant(info["h15_down"])[:info['max_horizontal_lines']]):
                    #  C1
                    limit_price, proba, date = h10_data["price"], h10_data["proba"], h10_data["date"]
                    dates.append(date)
                    texte_l1_c1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h10", situation="lower")}{texte_l1_c1}{reset}"

                    #  C2
                    limit_price, proba, date = h15_data["price"], h15_data["proba"], h15_data["date"]
                    dates.append(date)
                    texte_l1_c2 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c2 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h15", situation="lower")}{texte_l1_c2}{reset}"

                    bloc_H10H15 += make_bloc_line(texte_l1_c1, texte_l1_c2)
                bloc_H10H15 = f"{first_indent}{' ' * int(bloc_H10H15_length[0] // 2)}H10 ({dates[0]}){second_indent}{' ' * int(bloc_H10H15_length[1] // 3)}H15 ({dates[1]}){' ' * (int(bloc_H10H15_length[1] // 2) - 3)}" + saut_ligne + bloc_H10H15
                return bloc_H10H15

            #######################################################################
            # MONTH/H20
            #######################################################################
            def month_h20():
                bloc_monthH20, bloc_monthH20_length = "", (console_length//7, console_length//7)
                dates, ee = [], 0
                for hmonth_data, h20_data in zip(trier_par_prix_decroissant(info["hmonth_down"])[:info['max_horizontal_lines']], trier_par_prix_decroissant(info["h20_down"])[:info['max_horizontal_lines']]):
                    ee += 1
                    #  C1
                    limit_price, proba, date = hmonth_data["price"], hmonth_data["proba"], hmonth_data["date"]
                    dates.append(date)
                    texte_l1_c1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="hmonth", situation="lower")}{texte_l1_c1}{reset}"

                    #  C2
                    limit_price, proba, date = h20_data["price"], h20_data["proba"], h20_data["date"]
                    dates.append(date)
                    texte_l1_c2 = f"{format_price(limit_price)} ({format_proba(proba)})"
                    texte_l1_c2 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h20", situation="lower")}{texte_l1_c2}{reset}"

                    bloc_monthH20 += make_bloc_line(texte_l1_c1, texte_l1_c2, True if ee < info['max_horizontal_lines'] else False)
                bloc_monthH20 = f"{first_indent}{' ' * int(bloc_monthH20_length[0] // 2)}MONTH ({dates[0]}){second_indent}{' ' * int(bloc_monthH20_length[1] // 3 - 2)}H20 ({dates[1]}){' ' * (int(bloc_monthH20_length[1] // 2) - 3)}" + saut_ligne + bloc_monthH20
                return bloc_monthH20

            return week_h5() + saut_ligne + h10_h15() + saut_ligne + month_h20() + saut_ligne + quarter_year_h80_h240()

        def make_upper_H0_limits(info):
            bloc_H0 = f"{first_indent * 7}H0:: "
            for e, h0_data in enumerate(trier_par_prix_croissant(info["h0_up"])[:info['max_horizontal_lines_h0']]):
                limit_price, proba, date = h0_data["price"], h0_data["proba"], h0_data["date"]
                texte_l1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                texte_l1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h0", situation="upper")}{texte_l1}{reset}"
                bloc_H0 += f"{texte_l1} "
                if e < info['max_horizontal_lines_h0']-1: bloc_H0 += f"| "
            return bloc_H0

        def make_lower_H0_limits(info):
            bloc_H0 = f"{first_indent * 7}H0:: "
            for e, h0_data in enumerate(trier_par_prix_decroissant(info["h0_down"])[:info['max_horizontal_lines_h0']]):
                limit_price, proba, date = h0_data["price"], h0_data["proba"], h0_data["date"]
                texte_l1 = f"{format_price(limit_price)} ({format_proba(proba)})"
                texte_l1 = f"{determine_color(actual_price=actual_close, limit_price=limit_price, dte="h0", situation="lower")}{texte_l1}{reset}"
                bloc_H0 += f"{texte_l1} "
                if e < info['max_horizontal_lines_h0']-1: bloc_H0 += f"| "
            return bloc_H0

        def make_trend(info):
            str_trend, trend_length = "", ["60min", "240min", "day", "week", "month"]
            for e, length_ in enumerate(trend_length):
                df__ = info[f'df__{length_}']
                close_col = ('Close', ticker)
                str_trend += "1h    " if "60min" == length_ else ("4h    " if "240min" == length_ else ("DAY   " if "day" == length_ else ("WEEK  " if "week" == length_ else "MONTH ")))
                for ema_length in [5, 10, 20, 50, 100, 200]:
                    if df__ is None:
                        val_formatted = f"{'-'* 10}"  # Le '>' aligne à droite, le '10' impose une largeur de 10 caractères
                        str_trend += f"{vert}{val_formatted}{reset}({ema_length} EMA)  "
                        continue
                    # 1. Calculer la EMA sur la colonne de clôture
                    df__[(f'EMA_{ema_length}', ticker)] = df__[close_col].ewm(span=ema_length, adjust=False).mean()
                    # 2. Calculer le pourcentage d'écart par rapport à la EMA
                    # Un résultat positif signifie que le prix est au-dessus, négatif s'il est en dessous
                    df__[(f'Pct_from_EMA_{ema_length}', ticker)] = ((df__[close_col] - df__[(f'EMA_{ema_length}', ticker)]) / df__[(f'EMA_{ema_length}', ticker)])
                    val_ = df__[(f'Pct_from_EMA_{ema_length}', ticker)].iloc[-1]
                    # 1. Créez d'abord le texte de la valeur sans les couleurs, avec une largeur fixe (ex: 8 caractères, aligné à droite)
                    val_text = f"↑{val_:.4%}" if val_ > 0 else f"↓{abs(val_):.4%}"
                    val_formatted = f"{val_text:>10}"  # Le '>' aligne à droite, le '10' impose une largeur de 10 caractères
                    if val_ > 0: str_trend += f"{vert}{val_formatted}{reset}({ema_length} EMA)  "
                    else: str_trend += f"{rouge_clair}{val_formatted}{reset}({ema_length} EMA)  "
                str_trend += saut_ligne if e < len(trend_length) - 1 else ""
            return str_trend

        print(f"{gras}{f'THE {TITLE_WEBEAR} COMPANY'.center(console_length, '=')}{reset}")
        bouclier_spx = f"{gras}SPX:{info['bouclier_spx']}{reset}"
        tech_info_trie = dict(sorted(info["persistance_conditionelle"].items(),key=lambda item: item[1]['proba'],reverse=True))
        persistance_conditionnelle = ""
        for e, (k, v) in enumerate(tech_info_trie.items()):
            less_than = True if v["sign"] == "<" else False
            threshold_value = v["thres"]
            proba = v["proba"]
            type_fleche = fleche_bas_pleine if less_than else fleche_haut_pleine
            persistance_conditionnelle += f'{gras}{actual_close}{reset}{type_fleche}{threshold_value:.0f}={proba:.0%}' + (',' if e < len(tech_info_trie)-1 else '')
        print(bouclier_spx)
        #print(bouclier_spx + f"{' | GAP:'}" + persistance_conditionnelle)
        print(f"{gras}{f'=' * console_length}{reset}")
        print(make_trend(info))
        print(make_upper_limits(info))
        color_day_streaker, tmr_day_streaker = f"{rouge_clair}" if info['day_streaker']['dir']=='down' and info["close"] > info["close_t_m1"] else (f"{rouge_clair}" if info['day_streaker']['dir']=='up' and info["close"] < info["close_t_m1"] else f"{vert}"), info["day_streaker"]["realtime"]
        ligne_previous_close_bar_day_streaker  = f"{color_day_streaker}{info['day_streaker']['proba']:.0%}{fleche_haut if info['day_streaker']['dir']=='up' else fleche_bas}(D{info['day_streaker']['b']}){reset}"
        color_week_streaker, tmr_week_streaker = f"{rouge_clair}" if info['week_streaker']['dir'] == 'down' and info["close"] > info["close_t_m1"] else (f"{rouge_clair}" if info['week_streaker']['dir'] == 'up' and info["close"] < info["close_t_m1"] else f"{vert}"), info["week_streaker"]["realtime"]
        ligne_previous_close_bar_week_streaker = f"{color_week_streaker}{info['week_streaker']['proba']:.0%}{fleche_haut if info['week_streaker']['dir'] == 'up' else fleche_bas}(W{info['week_streaker']['b']}){reset}"
        ligne_previous_close_bar = (f"({gras}{info['previous_now_txt']}) CLOSE:{int(previous_close)}{reset} | {ligne_previous_close_bar_day_streaker} : {ligne_previous_close_bar_week_streaker} | "
                                    f"E-VWAP {fleche_haut_pleine}{info["energy_above_vwap"]}/{fleche_bas_pleine}{info["energy_below_vwap"]}").center(console_length + len(gras) + len(reset), "=")
        if previous_close > actual_close: print(ligne_previous_close_bar)
        else: print(f"{'='*console_length}")
        print(make_upper_H0_limits(info).ljust(console_length, ' '))
        _diff = (actual_close - previous_close) / actual_close if 0 != previous_close else 0
        _diff = f"{rouge_clair}{_diff:.2%}{fleche_bas}{reset}" if _diff < 0 else f"{vert}{_diff:.2%}{fleche_haut}{reset}"
        ligne_centrale = f"[REALTIME] OPEN:{int(actual_open)}  HIGH:{int(actual_high)}  LOW:{int(actual_low)}  {' '*8}{gras}CLOSE:{int(actual_close)}{reset} ({_diff})   {info['now_txt']}"
        length_of_ligne_centrale = len(ligne_centrale) - len(gras) - len(reset)
        ligne_centrale_vera = f"{info['vera']['minutes_jusqua_16h']:03}minutes:: Put@{info['vera']['atr_predicted_low']}:{info['vera']['atr_prob_down']:.0%} | Call@{info['vera']['atr_predicted_high']}:{info['vera']['atr_prob_up']:.0%}"
        print(ligne_centrale+f"{' '*2} VERA::"+ligne_centrale_vera)
        print(make_lower_H0_limits(info).ljust(console_length, ' '))
        if previous_close <= actual_close: print(ligne_previous_close_bar)
        else:print(f"{'=' * console_length}")
        print(make_lower_limits(info))

    ###########################################################################
    #
    ###########################################################################
    def initialize_info(ticker):
        now = datetime.now()
        now_txt = now.strftime('%Y-%m-%d %Hh%Mm%Ss')
        previous_now = get_next_step(now, "day", -1)
        previous_now_txt = previous_now.strftime('%Y-%m-%d')
        info = {'now': now, 'now_txt': now_txt, 'previous_now': previous_now, 'previous_now_txt': previous_now_txt, 'ticker': ticker,
                'max_horizontal_lines': horizontal_lines__max_horizontal_lines, 'max_horizontal_lines_h0': horizontal_lines__max_horizontal_lines_h0, 'display_qtrend_info': False,
                'df__60min': None, 'df__240min': None, 'df__day': None, 'df__week': None, 'df__month': None,
                'open': 0, 'high': 0, 'low': 0, 'close': 0, 'energy_above_vwap': 0, 'energy_below_vwap': 0,
                'bouclier_spx': '-',
                'vera': {'atr_predicted_low': 0, 'atr_prob_down': 0, 'atr_predicted_high':0, 'atr_prob_up': 0, 'minutes_jusqua_16h': 0},
                'persistance_conditionelle': {1: {'close_t': 0, 'thres': 0, 'sign': '<', 'proba': 0}, 3: {'close_t': 0, 'thres': 0, 'sign': '<', 'proba': 0},
                                              6: {'close_t': 0, 'thres': 0, 'sign': '<', 'proba': 0}, 8: {'close_t': 0, 'thres': 0, 'sign': '<', 'proba': 0}},
                'close_t_m1': 0,
                'h240_up': [],
                'hyear_up': [],
                'h80_up': [],
                'hquarter_up': [],
                'h20_up': [],
                'hmonth_up': [],
                'h5_up': [],
                'hweek_up': [],
                'h10_up': [],
                'h15_up': [],
                'h240_down': [],
                'hyear_down': [],
                'h80_down': [],
                'hquarter_down': [],
                'h20_down': [],
                'hmonth_down': [],
                'h5_down': [],
                'hweek_down': [],
                'h10_down': [],
                'h15_down': [],
                'h0_down': [],
                'h0_up': [],
                'day_streaker':  {'b': 0, 'proba': 0, 'dir': 'up', 'realtime': False},
                'week_streaker': {'b': 0, 'proba': 0, 'dir': 'up', 'realtime': False},
                'quant': {},
                'quant_meta_info': {'timestamp': 0}
                }
        return info

    def random_update_info(info, **kwargs):
        now = datetime.now()
        now_txt = now.strftime('%Y-%m-%d %Hh%Mm%Ss')
        previous_now = get_next_step(now, "day", -1)
        previous_now_txt = previous_now.strftime('%Y-%m-%d')
        info.update({'now': now, 'now_txt': now_txt, 'previous_now': previous_now, 'previous_now_txt': previous_now_txt, 'ticker': ticker,})
        info.update({'df__60min': None, 'df__240min': None, 'df__day': None, 'df__week': None, 'df__month': None,})
        info.update({'energy_above_vwap': random.randint(0, 999), 'energy_below_vwap': random.randint(0, 999),})
        info.update({'open': random.randint(7750, 7850), 'high': random.randint(7750, 7850), 'low': random.randint(7750, 7850), 'close': random.randint(7750, 7850),
                'close_t_m1': random.randint(7750, 7850),
                'bouclier_spx': '*',})
        info.update({
            'h240_up': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                       {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                       {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                       {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')}, ],
            'h80_up': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                       {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                       {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                       {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')}, ],
                'h20_up': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                           {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                           {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                           {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')}, ],
            'hyear_up': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')}],
            'hquarter_up': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                          {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                          {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                          {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')}],
                'hmonth_up': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                              {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                              {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                              {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')}],
                'h5_up': [{"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 5).strftime('%Y-%m-%d')},
                          {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 5).strftime('%Y-%m-%d')},
                          {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 5).strftime('%Y-%m-%d')},
                          {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 5).strftime('%Y-%m-%d')}, ],
                'hweek_up': [{"price": random.randint(7700, 7900), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')}],
                'h10_up': [{"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 10).strftime('%Y-%m-%d')},
                           {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 10).strftime('%Y-%m-%d')},
                           {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 10).strftime('%Y-%m-%d')},
                           {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 10).strftime('%Y-%m-%d')}, ],
                'h15_up': [{"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 15).strftime('%Y-%m-%d')},
                           {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 15).strftime('%Y-%m-%d')},
                           {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 15).strftime('%Y-%m-%d')},
                           {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 15).strftime('%Y-%m-%d')}],
                'h20_down': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                             {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                             {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                             {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')}, ],
            'h240_down': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                         {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                         {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                         {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')}, ],
            'h80_down': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                         {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                         {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                         {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')}, ],
            'hquarter_down': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')}],
            'hyear_down': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                            {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')}],
                'hmonth_down': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                                {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                                {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                                {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')}],
                'h5_down': [{"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 5).strftime('%Y-%m-%d')},
                            {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 5).strftime('%Y-%m-%d')},
                            {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 5).strftime('%Y-%m-%d')},
                            {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 5).strftime('%Y-%m-%d')}, ],
                'hweek_down': [{"price": random.randint(7700, 7900), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                               {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                               {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')},
                               {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_month_range(now)[1].strftime('%Y-%m-%d')}],
                'h10_down': [{"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 10).strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 10).strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 10).strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 10).strftime('%Y-%m-%d')}, ],
                'h15_down': [{"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 15).strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 15).strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 15).strftime('%Y-%m-%d')},
                             {"price": random.randint(7700, 7900), "proba": random.random(), "date": get_next_step(now, "day", 15).strftime('%Y-%m-%d')}],
                'h0_down': [{"price": random.randint(7700, 7750), "proba": random.random(), "date": get_next_step(now, "day", 0).strftime('%Y-%m-%d')},
                            {"price": random.randint(7700, 7750), "proba": random.random(), "date": get_next_step(now, "day", 0).strftime('%Y-%m-%d')},
                            {"price": random.randint(7700, 7750), "proba": random.random(), "date": get_next_step(now, "day", 0).strftime('%Y-%m-%d')},
                            {"price": random.randint(7700, 7750), "proba": random.random(), "date": get_next_step(now, "day", 0).strftime('%Y-%m-%d')}],
                'h0_up': [{"price": random.randint(7800, 7850), "proba": random.random(), "date": get_next_step(now, "day", 0).strftime('%Y-%m-%d')},
                          {"price": random.randint(7800, 7850), "proba": random.random(), "date": get_next_step(now, "day", 0).strftime('%Y-%m-%d')},
                          {"price": random.randint(7800, 7850), "proba": random.random(), "date": get_next_step(now, "day", 0).strftime('%Y-%m-%d')},
                          {"price": random.randint(7800, 7850), "proba": random.random(), "date": get_next_step(now, "day", 0).strftime('%Y-%m-%d')}]})
        info.update({'vera': {'atr_predicted_low': random.randint(7700, 7900), 'atr_prob_down': random.random(), 'atr_predicted_high':random.randint(7700, 7900), 'atr_prob_up': random.random(), 'minutes_jusqua_16h': random.randint(0, 199)},
                     'persistance_conditionelle': {1: {'close_t': np.float64(7718.93994140625), 'thres': np.float64(7814.14990234375), 'sign': '<', 'proba': np.float64(0.815251084934904)},
                                                   3: {'close_t': np.float64(7718.93994140625), 'thres': np.float64(7786.10986328125), 'sign': '<', 'proba': np.float64(0.38561686298822073)}, 6: {'close_t': np.float64(7718.93994140625), 'thres': np.float64(7805.4501953125), 'sign': '<', 'proba': np.float64(0.6531308121512709)},
                                                   8: {'close_t': np.float64(7718.93994140625), 'thres': np.float64(7805.4501953125), 'sign': '<', 'proba': np.float64(0.7561436672967864)}}})
        info.update({'day_streaker': {'b':random.randint(2, 10), 'proba': random.random(), 'dir': 'up' if random.random() > 0.5 else 'down', 'realtime':random.choice([True, False]),},
                     'week_streaker': {'b': random.randint(2, 10), 'proba': random.random(), 'dir': 'up' if random.random() > 0.5 else 'down', 'realtime':random.choice([True, False]),}})
        if not kwargs.get("skip_quant", False):
            nb_quant_values = 10
            strike_prices = [random.randint(int(info["open"]-100), int(info["open"])) for _ in range(nb_quant_values//2)] + [random.randint(int(info["open"]), int(info["open"])+100) for _ in range(nb_quant_values//2)]
            opt = ["PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD", "PASS"]
            info.update({'quant': {}})
            for j in range(nb_quant_values):
                tmp_lst = []
                tmp_lst.append({'strike_price': strike_prices[j], 'win_rate': random.random(), 'action': random.choice(opt), 'timestamp': datetime.now().time()})
                time.sleep(0.456)
                tmp_lst.append({'strike_price': strike_prices[j], 'win_rate': random.random(), 'action': random.choice(opt), 'timestamp': datetime.now().time()})
                time.sleep(0.456)
                tmp_lst.append({'strike_price': strike_prices[j], 'win_rate': random.random(), 'action': random.choice(opt), 'timestamp': datetime.now().time()})
                time.sleep(0.456)
                info['quant'][str(strike_prices[j])] = tmp_lst
            info['display_qtrend_info'] = random.choice([True, False])
            info['quant_meta_info'] = {'timestamp': now_txt}
        return info

    argus__information_displayed = initialize_info(ticker)
    argus__information_displayed = random_update_info(argus__information_displayed, **{"skip_quant": True})  # Quant accumule les mesures
    scheduler = BackgroundScheduler()

    ###########################################################################
    # Realtime value
    ###########################################################################
    def setup_realtime_values():
        def request_for_realtime_values(**kwargs):
            use_cases__shared.put({"realtime_values": kwargs})
        kwargs = {"ticker": ticker}
        scheduler.add_job(request_for_realtime_values, 'date', kwargs=kwargs)
        # 1. Gère la première heure incomplète (de 9h30 à 9h59)
        scheduler.add_job(request_for_realtime_values,'cron',day_of_week='mon-fri',hour=9,minute='30-59',kwargs=kwargs)
        # 2. Gère les heures pleines (toutes les minutes de 10h00 à 15h59)
        scheduler.add_job(request_for_realtime_values,'cron',day_of_week='mon-fri',hour='10-15',minute='*',kwargs=kwargs)
        # 3. Gère la minute finale (16h00 pile)
        scheduler.add_job(request_for_realtime_values,'cron',day_of_week='mon-fri',hour=16,minute=0,kwargs=kwargs)
    if starter_auth["realtime"]: setup_realtime_values()

    ###########################################################################
    # Bouclier SPX
    ###########################################################################
    def setup_bouclier_spx():
        def request_for_bouclier_spx(**kwargs):
            use_cases__shared.put({"bouclier_spx": kwargs})
        # 1. Gère les exécutions de la journée (toutes les 4 heures à partir de 9h30)
        scheduler.add_job(request_for_bouclier_spx,'cron',day_of_week='mon-fri',hour='9,13',minute=30,kwargs={'update_dataset': False})
        # 2. Gère l'exécution finale à 16h00 pile
        scheduler.add_job(request_for_bouclier_spx,'cron',day_of_week='mon-fri',hour=16,minute=0,kwargs={'update_dataset': False})
        scheduler.add_job(request_for_bouclier_spx, 'date', kwargs={'update_dataset': False})
        scheduler.add_job(request_for_bouclier_spx,'cron',day_of_week='tue,fri',hour=6,minute=15,kwargs={'update_dataset': True})
    if starter_auth["bouclier_spx"]: setup_bouclier_spx()

    ###########################################################################
    # Trend EMA
    ###########################################################################
    def setup_trend_ema():
        def request_for_trend_computation(**kwargs):
            use_cases__shared.put({"trend_computation": kwargs})
        # 1. Gère les déclenchements de 9h30, puis toutes les heures de 10h30 à 15h30
        scheduler.add_job(request_for_trend_computation,'cron',day_of_week='mon-fri',hour='9-15',minute='0,30',kwargs={"ticker": ticker})
        # 2. Gère le déclenchement final à 16h00 pile
        scheduler.add_job(request_for_trend_computation,'cron',day_of_week='mon-fri',hour=16,minute=0,kwargs={"ticker": ticker})
        scheduler.add_job(request_for_trend_computation,'date',kwargs={'ticker': ticker})
    if starter_auth["trend_ema"]: setup_trend_ema()

    ###########################################################################
    # Horizontal Lines
    ###########################################################################
    def setup_horizontal_lines():
        def request_for_horizontal_lines(**kwargs):
            use_cases__shared.put({"horizontal_lines": kwargs})
        for _dataset_id, _key_id in zip(["year",  "extraday-240days", "quarter",  "extraday-80days", "extraday-20days", "month",  "extraday-15days", "extraday-10days", "extraday-5days", "week",  "day"],
                                                      ["hyear", "h240",             "hquarter", "h80",             "h20",             "hmonth", "h15",             "h10",             "h5",             "hweek", "h0"]):
            n_split = 0.7 if _key_id in ["hyear", "h240", ] else (0.8 if _key_id in ["hquarter", "h80"] else 0.9)
            kwargs = {'dataset_id': _dataset_id, "key_id": _key_id, "ticker": ticker, "n_split": n_split, "n_trials": horizontal_lines__n_trials,
                      "tightness_weights": horizontal_lines__tightness_weights, "nb_worker": horizontal_lines__nb_worker}
            scheduler.add_job(request_for_horizontal_lines, 'cron', day_of_week='mon-fri', hour=7, minute=0, kwargs=kwargs)
            scheduler.add_job(request_for_horizontal_lines, 'cron', day_of_week='mon-fri', hour=9, minute=31, kwargs=kwargs)
            scheduler.add_job(request_for_horizontal_lines, 'date', kwargs=kwargs)
    if starter_auth["horizontal_lines"]: setup_horizontal_lines()

    ###########################################################################
    # Vera
    ###########################################################################
    def setup_vera():
        def request_for_vera(**kwargs):
            # Vérification stricte de l'heure courante
            maintenant = datetime.now().time()
            heure_debut = datetime.strptime("09:45", "%H:%M").time()
            heure_fin = datetime.strptime("16:00", "%H:%M").time()
            if maintenant < heure_debut or maintenant > heure_fin:
                return  # Ignore l'exécution si on est hors de la plage
            use_cases__shared.put({"vera": kwargs})
        kwargs = {"ticker": ticker, "tightness_weight": vera__tightness_weight, "n_trials": vera__n_trials, "n_split": vera__n_split}
        scheduler.add_job(request_for_vera,'cron',day_of_week='mon-fri',hour='9-16',minute='*/5',kwargs=kwargs)
    if starter_auth["vera"]: setup_vera()

    ###########################################################################
    # Persistance conditionelle
    ###########################################################################
    def setup_gap():
        def request_for_gap(**kwargs):
            use_cases__shared.put({"gap": kwargs})
        kwargs = {"ticker": ticker, }
        scheduler.add_job(request_for_gap,'cron',day_of_week='mon-fri',hour=9,minute=32,kwargs=kwargs)
        scheduler.add_job(request_for_gap, 'date', kwargs=kwargs)
    if starter_auth["gap"]: setup_gap()

    ###########################################################################
    # Streakers
    ###########################################################################
    def setup_day_streaker():
        def request_for_day_streaker(**kwargs):
            realtime = True if datetime.now().time() > datetime.strptime("15:29", "%H:%M").time() else False
            kwargs.update({"realtime": realtime})
            use_cases__shared.put({"day_streaker": kwargs})
        kwargs = {"ticker": ticker, "dataset_id": "day", "realtime": False, "clip_n": 0}
        scheduler.add_job(request_for_day_streaker,'cron',day_of_week='mon-fri',hour='9-15',minute='30',kwargs=kwargs)
        scheduler.add_job(request_for_day_streaker, 'date', kwargs=kwargs)
    if starter_auth["day_streaker"]: setup_day_streaker()
    def setup_week_streaker():
        def request_for_week_streaker(**kwargs):
            use_cases__shared.put({"week_streaker": kwargs})
        kwargs = {"ticker": ticker, "dataset_id": "week", "realtime": False, "clip_n": 0}
        scheduler.add_job(request_for_week_streaker,'cron',day_of_week='mon-fri',hour='9-15',minute='19',kwargs=kwargs)
        scheduler.add_job(request_for_week_streaker, 'date', kwargs=kwargs)
    if starter_auth["week_streaker"]: setup_week_streaker()

    ###########################################################################
    # Quant
    ###########################################################################
    def setup_quant():
        def request_for_quant(**kwargs):
            use_cases__shared.put({"quant": kwargs})
        kwargs = {"ticker": ticker, "st1":"09:30", "et1":"15:59", "et2":"16:00", "realtime": True, "models_dir": r"D:\Finance\compiled_models\taurus\v1\2026.10.22\quant\miner1__credit_spread_2"}
        for st2 in quant__st2:
            for seuil_pos in quant__seuil_pos:
                # 1. Convertir la chaîne "HH:MM" en objet datetime pour manipuler le temps facilement
                time_obj = datetime.strptime(st2, "%H:%M")
                # 2. Ajouter 1 minute à cet horaire
                cron_time = time_obj + timedelta(minutes=1)
                # 3. Extraire l'heure et la minute calculées (sous forme d'entiers)
                cron_hour = cron_time.hour
                cron_minute = cron_time.minute
                scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour=cron_hour, minute=cron_minute, kwargs={**kwargs, "st2":st2, "seuil_pos":seuil_pos})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 0.999})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 0.9975})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 0.9950})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 0.9925})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 0.99})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 1.001})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 1.0025})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 1.0050})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 1.0075})
        # scheduler.add_job(request_for_quant, 'cron', day_of_week='mon-fri', hour="11", minute="*/1", kwargs={**kwargs, "st2": "11:30", "seuil_pos": 1.01})
    if starter_auth["quant"]: setup_quant()

    ###########################################################################
    # Historian
    ###########################################################################
    def setup_historian():
        def request_for_historian_dump(**kwargs):
            info, output_dir, acc = kwargs["info"], kwargs["output_dir"], kwargs["accumulator_historian"]
            info = copy.deepcopy(info)
            info["df__60min"], info["df__240min"], info["df__day"], info["df__week"], info["df__month"] = [], [], [], [], []
            acc.append(info)
            maintenant = datetime.now()
            timestamp_cle = maintenant.strftime("%Y-%m-%d %H:%M:%S")
            def stringifier_cles_rec(obj):
                """Convertit récursivement toutes les clés de dictionnaires en str."""
                if isinstance(obj, dict):
                    return {str(cle): stringifier_cles_rec(valeur) for cle, valeur in obj.items()}
                elif isinstance(obj, list):
                    return [stringifier_cles_rec(element) for element in obj]
                return obj
            info_propre = stringifier_cles_rec(info)
            # On ajoute l'année, le mois et le jour comme colonnes de partitionnement
            dict_pour_parquet = {
                "year": [maintenant.year],
                "month": [maintenant.month],
                "day": [maintenant.day],
                "timestamp": [timestamp_cle],
                "info": [info_propre], }
            table = pa.Table.from_pydict(dict_pour_parquet)
            # write_to_dataset va créer automatiquement l'arborescence :
            # dossier/year=2026/month=10/day=16/xxxx.parquet
            pq.write_to_dataset(table, root_path=output_dir, partition_cols=["year", "month", "day"], use_dictionary=True, compression="zstd", compression_level=12)
            del kwargs["accumulator_historian"][:]
        def request_for_historian(**kwargs):
            info, output_dir, acc = kwargs["info"], kwargs["output_dir"], kwargs["accumulator_historian"]
            info = copy.deepcopy(info)
            info["df__60min"], info["df__240min"], info["df__day"], info["df__week"], info["df__month"]=[],[],[],[],[]
            acc.append(info)
        kwargs = {"info": argus__information_displayed, "output_dir": historian__output_dir, "accumulator_historian": historian__accumulator}
        scheduler.add_job(request_for_historian, 'cron', day_of_week='mon-fri', hour='9', minute='30-59', kwargs=kwargs)
        scheduler.add_job(request_for_historian, 'cron', day_of_week='mon-fri', hour='10-15', minute='*', kwargs=kwargs)
        scheduler.add_job(request_for_historian, 'cron', day_of_week='mon-fri', hour='16', minute='0', kwargs=kwargs)
        scheduler.add_job(request_for_historian_dump, 'cron', day_of_week='mon-fri', hour='11,13,16', minute='0', kwargs=kwargs)
    if starter_auth["historian"]: setup_historian()

    def setup_cleaner():
        def request_for_cleaner(**kwargs):
            use_cases__shared.put({"cleaner": kwargs})
            argus__information_displayed.update({'quant': {}})
        scheduler.add_job(request_for_cleaner,'cron',hour=0,minute=0,kwargs={})  # Tous les jours à minuit
    setup_cleaner()

    scheduler.start()

    if starter_auth["splash_screen"]:
        # =============================================================================
        # Print Human-Readable Cron Schedules extracted directly from the Scheduler
        # =============================================================================
        print("\n" + "=" * 95)
        print("📅 ACTIVE SCHEDULES OVERVIEW".center(95))
        print("=" * 95)

        def format_trigger(trigger):
            s = str(trigger)
            if s.startswith('cron'):
                content = s[5:-1]
                parts = []
                for p in content.split(', '):
                    if '=' in p:
                        k, v = p.split('=', 1)
                        v = v.strip("'\"")
                        if k == 'day_of_week':
                            parts.append(f"Days: {v}")
                        elif k == 'hour':
                            parts.append(f"Hours: {v}")
                        elif k == 'minute':
                            parts.append(f"Mins: {v}")
                        elif k == 'month':
                            parts.append(f"Months: {v}")
                        elif k == 'day':
                            parts.append(f"DaysOfMonth: {v}")
                        else:
                            parts.append(f"{k}: {v}")
                return ", ".join(parts)
            elif s.startswith('date'):
                return f"Startup (Once)"
            return s

        ACTIVITY_MAPPING = {
            'request_for_realtime_values': ('realtime', "⏱️  Realtime Values"),
            'request_for_bouclier_spx': ('bouclier_spx', "🛡️  Bouclier SPX"),
            'request_for_trend_computation': ('trend_ema', "📈 Trend EMA"),
            'request_for_horizontal_lines': ('horizontal_lines', "📏 Horizontal Lines"),
            'request_for_vera': ('vera', "🔮 Vera Model"),
            'request_for_gap': ('gap', "🕳️  Gap Persistence"),
            'request_for_day_streaker': ('day_streaker', "🔥 Day Streaker"),
            'request_for_week_streaker': ('week_streaker', "🔥 Week Streaker"),
            'request_for_historian': ('historian', "🏛️  Historian"),
            'request_for_historian_dump': ('historian_dump', "🏛️  Historian (Dump)"),
            'request_for_quant': ('quant', "📊 Quant"),
            'request_for_cleaner': ('cleaner', "🧹 Cleaner"),
        }

        grouped_jobs = {}
        for job in scheduler.get_jobs():
            func_name = job.func.__name__ if hasattr(job.func, '__name__') else str(job.func)
            if func_name in ACTIVITY_MAPPING:
                key, title = ACTIVITY_MAPPING[func_name]
                if key not in grouped_jobs:
                    grouped_jobs[key] = {'title': title, 'triggers': []}

                trigger_str = format_trigger(job.trigger)
                if trigger_str not in grouped_jobs[key]['triggers']:
                    grouped_jobs[key]['triggers'].append(trigger_str)

        ordered_keys = [
            'realtime', 'bouclier_spx', 'trend_ema', 'horizontal_lines',
            'vera', 'gap', 'day_streaker', 'week_streaker', 'historian', 'historian_dump', 'quant', 'cleaner'
        ]

        for key in ordered_keys:
            if key in grouped_jobs:
                title = grouped_jobs[key]['title']
                triggers = grouped_jobs[key]['triggers']

                if key == 'quant':
                    cron_triggers = [t for t in triggers if t != "Startup (Once)"]
                    if cron_triggers:
                        print(f"{title:<25} : {len(cron_triggers)} specific schedules from {cron_triggers[0]} to {cron_triggers[-1]}")
                else:
                    trigger_str = " | ".join(triggers)
                    print(f"{title:<25} : {trigger_str}")

        print("\n🚀 *Note: 'Startup (Once)' means the task executes immediately at program start.*")
        print("=" * 95)

        try:
            print("\nLancement de la boucle de surveillance dans : ", end="", flush=True)
            for i in range(5, -1, -1):
                print(f"{i}...", end="", flush=True)
                time.sleep(1)
                # Supprime le chiffre précédent pour afficher le suivant au même endroit
                if i > 0:
                    print("\b\b\b\b", end="", flush=True)
            print("\nBoucle démarrée !")
        except KeyboardInterrupt:
            print("\n\nDémarrage annulé par l'utilisateur.")
            sys.exit(0)

    # =============================================================================
    if starter_auth["activer_console"]: clean_console()
    while True:
        if starter_auth["activer_console"]: print("\033[H", end="")  # déplace le curseur textuel tout en haut à gauche du terminal (la position "Home")
        if starter_auth["activer_random_update"]: argus__information_displayed = random_update_info(info=argus__information_displayed)
        resuls_from_worker = []
        # Get the computed data, if any
        for k in range(0, argus__nb_worker):
            try:
                resuls_from_worker.extend(out__shared[k].get(timeout=0.1))
            except:
                pass
        # Add values to info
        for one_result_from_worker in resuls_from_worker:
            assert 1 == len(one_result_from_worker)
            k, v = next(iter(one_result_from_worker.items()))
            if k == "bouclier_spx":
                if v['avg_score'] <= 3.5:
                    argus__information_displayed['bouclier_spx'] = "🟢 Environnement favorable"
                # synthese_globale = "🟢 Environnement favorable. Les indicateurs de risque sont faibles. La poursuite des stratégies standards est justifiée."
                elif v['avg_score'] <= 6.5:
                    argus__information_displayed['bouclier_spx'] = "🟡 Prudence requise"
                else:
                    argus__information_displayed['bouclier_spx'] = "🔴 Alerte de risque élevée"
            if k == "trend_computation":
                for k2, v2 in v.items():
                    argus__information_displayed[k2] = v2
            if k == "realtime_values":
                for k2, v2 in v.items():
                    argus__information_displayed[k2] = f"{v2:.0f}"
            if k in ["hyear","h240","hquarter","h80","h20", "hmonth", "h15", "h10", "h5", "hweek", "h0"]:
                argus__information_displayed[f"{k}_down"] = []
                argus__information_displayed[f"{k}_up"] = []
                if k == "h240": dte_date = get_next_step(datetime.now(), "day", 240).strftime('%Y-%m-%d')
                if k == "h80": dte_date = get_next_step(datetime.now(), "day", 80).strftime('%Y-%m-%d')
                if k == "h20": dte_date = get_next_step(datetime.now(), "day", 20).strftime('%Y-%m-%d')
                if k == "h15": dte_date = get_next_step(datetime.now(), "day", 15).strftime('%Y-%m-%d')
                if k == "h10": dte_date = get_next_step(datetime.now(), "day", 10).strftime('%Y-%m-%d')
                if k == "h5": dte_date = get_next_step(datetime.now(), "day", 5).strftime('%Y-%m-%d')
                if k == "h0": dte_date = datetime.now().strftime('%Y-%m-%d')
                if k == "hyear": dte_date = get_year_range(datetime.now())[1].strftime('%Y-%m-%d')
                if k == "hquarter": dte_date = get_quarter_range(datetime.now())[1].strftime('%Y-%m-%d')
                if k == "hmonth": dte_date = get_month_range(datetime.now())[1].strftime('%Y-%m-%d')
                if k == "hweek": dte_date = get_week_range(datetime.now())[1].strftime('%Y-%m-%d')
                #print(f"{dte_date}   {k} :: {v}\n\n\n")
                for a_dict in v['low']: argus__information_displayed[f"{k}_down"].append({"price": a_dict['predicted_low'], "proba": float(a_dict['probability_predicted_low'])/100., "date": dte_date})
                while len(argus__information_displayed[f"{k}_down"]) < argus__information_displayed["max_horizontal_lines"]: argus__information_displayed[f"{k}_down"].append({"price": 1, "proba": 0, "date": dte_date})
                for a_dict in v['high']: argus__information_displayed[f"{k}_up"].append({"price": a_dict['predicted_high'], "proba": float(a_dict['probability_predicted_high'])/100., "date": dte_date})
                while len(argus__information_displayed[f"{k}_up"]) < argus__information_displayed["max_horizontal_lines"]: argus__information_displayed[f"{k}_up"].append({"price": 99999, "proba": 0, "date": dte_date})
            if k == "vera":
                argus__information_displayed["vera"] = v
            if k == "gap":
                argus__information_displayed.update({'persistance_conditionelle': v})
            def extraire_info_streaker(texte):
                # Expression régulière pour capturer :
                # 1. Le pourcentage (ex: 28.9)
                # 2. La flèche (↑ ou ↓)
                # 3. Le chiffre après le 'B' (ex: 2)
                motif = r"(\d+(?:\.\d+)?)%([↑↓])\s*\(B(\d+)\)"

                match = re.search(motif, texte)
                assert match
                pourcentage_str, fleche, valeur_b = match.groups()
                #print(texte)
                #print(f"{pourcentage_str}, {fleche}, {valeur_b}")
                # a) Conversion du pourcentage en probabilité (0 à 1)
                proba = float(pourcentage_str) / 100.0

                # b) Détermination de 'up' ou 'down' selon la flèche
                direction = "up" if fleche.strip() == "↑" else "down"

                # c) Conversion de la valeur après le B en entier
                valeur_b = int(valeur_b)

                return {
                    "proba": proba,
                    "direction": direction,
                    "valeur_b": valeur_b
                }
            if k == "day_streaker":
                dd_tmp = extraire_info_streaker(v[0])
                argus__information_displayed.update({'day_streaker': {'b': dd_tmp["valeur_b"], 'proba': dd_tmp["proba"], 'dir': dd_tmp["direction"]} | v[1], })
            if k == "week_streaker":
                dd_tmp = extraire_info_streaker(v[0])
                argus__information_displayed.update({'week_streaker': {'b': dd_tmp["valeur_b"], 'proba': dd_tmp["proba"], 'dir': dd_tmp["direction"]} | v[1], })
            if k == "quant":
                strike_price_str, tmp_dict = v
                argus__information_displayed["quant"].setdefault(strike_price_str, []).extend(tmp_dict)
                argus__information_displayed['quant_meta_info'].update({'timestamp': datetime.now().strftime('%Hh%Mm%Ss')})
        now = datetime.now()
        now_txt = now.strftime('%Y-%m-%d %Hh%Mm%Ss')
        previous_now = get_next_step(now, "day", -1)
        previous_now_txt = previous_now.strftime('%Y-%m-%d')
        argus__information_displayed.update({'now': now, 'now_txt': now_txt, 'previous_now': previous_now, 'previous_now_txt': previous_now_txt, 'ticker': ticker, 'max_horizontal_lines': 3,})

        # Display info
        if starter_auth["activer_console"]:
            if 0 != len(resuls_from_worker): clean_console()
            make_display(info=argus__information_displayed)

        # Wait
        time.sleep(3)


if __name__ == "__main__":
    freeze_support()
    entry()