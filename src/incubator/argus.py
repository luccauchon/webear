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
from utils import get_next_step, get_month_range,get_week_range, minutes_jusqua_16h
from fetchers.data_factory import factory_load_data
import schedule
import traceback
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
from scipy.integrate import trapezoid
import pyarrow as pa
import pyarrow.parquet as pq
import copy


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
    quant_cached_models = []
    while True:
        use_case_batch = []
        while len(use_case_batch) < request_batch_size:
            try:
                item = use_cases__shared.get(timeout=0.1)
                use_case_batch.append(item)
            except:
                break  # Queue is empty or no more items within timeout
        with master_cmd__shared.get_lock():
            if 2 == master_cmd__shared.value:
                break  # On quitte
        if 0 == len(use_case_batch): continue
        all_results_computed = []
        for use_case in use_case_batch:
            try:
                assert 1 == len(use_case)
                k, v = next(iter(use_case.items()))
                #print(f"[{os.getpid()}] ------Processing {k} --------------{v}")
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
                                                                                         use_close_for_range=True, n_trials=999, no_email=True, verbose=False,
                                                                                         tightness_weights=v["tightness_weights"], nb_worker=4))
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
                    streaker_results = streak_probability_entry(args=Namespace(ticker=v["ticker"], dataset_id=v["dataset_id"], send_email=False, realtime=v["realtime"], production_setup=False))
                    all_results_computed.append({k: streaker_results["proba_txt"]})
                if k == "week_streaker":
                    streaker_results = streak_probability_entry(args=Namespace(ticker=v["ticker"], dataset_id=v["dataset_id"], send_email=False, realtime=v["realtime"], production_setup=False))
                    all_results_computed.append({k: streaker_results["proba_txt"]})
                if k == "qaunt":
                    if (0 == len(v) or 0 == len(quant_cached_models)) or (0 != len(v) and 0 == len(quant_cached_models)):
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

    # data_from_quant = quant_morning_alpha_miner(args=Namespace(ticker=ticker, nb_worker=1, realtime=True, verbose=False, modelfile=r"..\optimizers\quantloop\quant\miner1\_top_5_ensemble_model__0.9995__0930-1000__155500-160000_High_0_Open__921145____tf1_0.51757085.pkl"))
    # print(data_from_quant)
    # print(data_from_quant["result"]["median_probability"].values[0])
    # print(data_from_quant["result"]["std_probability"].values[0])
    # print(data_from_quant["result"]["predicted_class"].values[0])
    starter_auth = {'bouclier_spx': False, "realtime": True, "trend_ema": False, "horizontal_lines": False, "vera": False,
                    "gap": False, # Pas utilise
                    "day_streaker": True, "week_streaker": True, "historian": True,
                    "activer_console": True, "activer_random_update": True, "quant": True}

    output_dir_historian, accumulator_historian = HISTORIAN_ARGUS, []
    clean_console()
    information_displayed = {}  # Data used to create and update the display
    nb_worker = 2
    use_cases__shared, master_cmd__shared = Queue(640), Value("i", 0)
    out__shared = [Queue(640) for k in range(0, nb_worker)]
    # Lancement des workers
    for k in range(0, nb_worker):
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

        def make_bloc_line(texte_ln_c1, texte_ln_c2):
            return f"{first_indent*2}{texte_ln_c1}{second_indent_pp}{texte_ln_c2}{saut_ligne}"

        def make_bloc_line_3(texte_ln_c1, texte_ln_c2, texte_ln_c3):
            return f"{first_indent*2}{texte_ln_c1}{second_indent_pp}{texte_ln_c2}{' '*8}{texte_ln_c3}{saut_ligne}"

        def make_upper_limits(info):
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
                quant_dataset = [[],[],[]]
                dates = []
                for hweek_data, h5_data, quant_data in zip(trier_par_prix_decroissant(info["hweek_up"])[:info['max_horizontal_lines']], trier_par_prix_decroissant(info["h5_up"])[:info['max_horizontal_lines']], quant_dataset):
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
                    ppp, mmm = 0.56, .04
                    texte_l1_c3 = f"|{fleche_bas_pleine}|{gris_fonce}7555({ppp:03.0%}{plus_ou_moins}{mmm:03.0%}){reset},{vert}7800({ppp:03.0%}{plus_ou_moins}{mmm:03.0%}){reset}{vert}7810(45%){reset},{vert}7800(45%){reset},{gris_fonce}7555(0%){reset}"

                    bloc_weekH5 += make_bloc_line_3(texte_l1_c1, texte_l1_c2, texte_l1_c3)
                bloc_weekH5 = f"{first_indent}{' ' * int(bloc_weekH5_length[0] // 2)}WEEK ({dates[0]}){second_indent}{' ' * int(bloc_weekH5_length[1] // 3 - 1)}H5 ({dates[1]}){' ' * (int(bloc_weekH5_length[1] // 2) - 2)}" + saut_ligne + bloc_weekH5
                return bloc_weekH5

            return month_h20() + saut_ligne + h10_h15() + saut_ligne + week_h5()

        def make_lower_limits(info):
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
                dates = []
                for hmonth_data, h20_data in zip(trier_par_prix_decroissant(info["hmonth_down"])[:info['max_horizontal_lines']], trier_par_prix_decroissant(info["h20_down"])[:info['max_horizontal_lines']]):
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

                    bloc_monthH20 += make_bloc_line(texte_l1_c1, texte_l1_c2)
                bloc_monthH20 = f"{first_indent}{' ' * int(bloc_monthH20_length[0] // 2)}MONTH ({dates[0]}){second_indent}{' ' * int(bloc_monthH20_length[1] // 3 - 2)}H20 ({dates[1]}){' ' * (int(bloc_monthH20_length[1] // 2) - 3)}" + saut_ligne + bloc_monthH20
                return bloc_monthH20

            return week_h5() + saut_ligne + h10_h15() + saut_ligne + month_h20()

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
            str_trend = ""
            for length_ in ["60min", "240min", "day", "week", "month"]:
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
                str_trend += saut_ligne
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
        ligne_previous_close_bar_day_streaker  = f"{info['day_streaker']['proba']:.0%}{fleche_haut if info['day_streaker']['dir']=='up' else fleche_bas}(D{info['day_streaker']['b']})"
        ligne_previous_close_bar_week_streaker = f"{info['week_streaker']['proba']:.0%}{fleche_haut if info['week_streaker']['dir'] == 'up' else fleche_bas}(W{info['week_streaker']['b']})"
        ligne_previous_close_bar = (f"{gras}CLOSE:{previous_close} ({info['previous_now_txt']}){reset} | {ligne_previous_close_bar_day_streaker} : {ligne_previous_close_bar_week_streaker} | "
                                    f"E-VWAP {fleche_haut_pleine}{info["energy_above_vwap"]}/{fleche_bas_pleine}{info["energy_below_vwap"]}").center(console_length + len(gras) + len(reset), "=")
        if previous_close > actual_close: print(ligne_previous_close_bar)
        else: print(f"{'='*console_length}")
        print(make_upper_H0_limits(info).ljust(console_length, ' '))
        _diff = (actual_close - previous_close) / actual_close if 0 != previous_close else 0
        _diff = f"{rouge_clair}{_diff:.2%}{fleche_bas}{reset}" if _diff < 0 else f"{vert}{_diff:.2%}{fleche_haut}{reset}"
        ligne_centrale = f"{first_indent}OPEN:{actual_open}  HIGH:{actual_high}  LOW:{actual_low}  {gras}CLOSE:{actual_close}{reset}({_diff})   {info['now_txt']}"
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
                'max_horizontal_lines': 3, 'max_horizontal_lines_h0': 5,
                'df__60min': None, 'df__240min': None, 'df__day': None, 'df__week': None, 'df__month': None,
                'open': 0, 'high': 0, 'low': 0, 'close': 0, 'energy_above_vwap': 0, 'energy_below_vwap': 0,
                'bouclier_spx': '-',
                'vera': {'atr_predicted_low': 0, 'atr_prob_down': 0, 'atr_predicted_high':0, 'atr_prob_up': 0, 'minutes_jusqua_16h': 0},
                'persistance_conditionelle': {1: {'close_t': 0, 'thres': 0, 'sign': '<', 'proba': 0}, 3: {'close_t': 0, 'thres': 0, 'sign': '<', 'proba': 0},
                                              6: {'close_t': 0, 'thres': 0, 'sign': '<', 'proba': 0}, 8: {'close_t': 0, 'thres': 0, 'sign': '<', 'proba': 0}},
                'close_t_m1': 0,
                'h20_up': [],
                'hmonth_up': [],
                'h5_up': [],
                'hweek_up': [],
                'h10_up': [],
                'h15_up': [],
                'h20_down': [],
                'hmonth_down': [],
                'h5_down': [],
                'hweek_down': [],
                'h10_down': [],
                'h15_down': [],
                'h0_down': [],
                'h0_up': [],
                'day_streaker': {'b':0, 'proba': 0, 'dir': 'up'},
                'week_streaker': {'b': 0, 'proba': 0, 'dir': 'up'},
                'quant_call_credit_spread': {},
                'quant_put_credit_spread': {},
                }
        return info
    information_displayed = initialize_info(ticker)

    def random_update_info(info):
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
        info.update({'h20_up': [{"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                           {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                           {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')},
                           {"price": random.randint(8000, 9000), "proba": random.random(), "date": get_next_step(now, "day", 20).strftime('%Y-%m-%d')}, ],
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
        info.update({'day_streaker': {'b':random.randint(2, 10), 'proba': random.random(), 'dir': 'up' if random.random() > 0.5 else 'down'},
                     'week_streaker': {'b': random.randint(2, 10), 'proba': random.random(), 'dir': 'up' if random.random() > 0.5 else 'down'},})
        return info

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
        for _dataset_id, _key_id in zip(["extraday-20days", "month", "extraday-15days", "extraday-10days", "extraday-5days", "week", "day"], ["h20", "hmonth", "h15", "h10", "h5", "hweek", "h0"]):
            kwargs = {'dataset_id': _dataset_id, "key_id": _key_id, "ticker": ticker, "tightness_weights": [0., 0.33, 0.44, 0.55, 0.66, 0.99, 2, 4]}
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
        kwargs = {"ticker": ticker, "tightness_weight": 0.33, "n_trials": 500, "n_split": 0.9}
        scheduler.add_job(request_for_vera,'cron',day_of_week='mon-fri',hour='9-16',minute='*/5',kwargs=kwargs)
        scheduler.add_job(request_for_vera, 'date', kwargs=kwargs)
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
            use_cases__shared.put({"day_streaker": kwargs})
        kwargs = {"ticker": ticker, "dataset_id": "day", "realtime": True}
        scheduler.add_job(request_for_day_streaker,'cron',day_of_week='mon-fri',hour='9-16',minute='0,30',kwargs=kwargs)
        scheduler.add_job(request_for_day_streaker, 'date', kwargs=kwargs)
    if starter_auth["day_streaker"]: setup_day_streaker()
    def setup_week_streaker():
        def request_for_week_streaker(**kwargs):
            use_cases__shared.put({"week_streaker": kwargs})
        kwargs = {"ticker": ticker, "dataset_id": "week", "realtime": True}
        scheduler.add_job(request_for_week_streaker,'cron',day_of_week='mon-fri',hour='9-16',minute='0,30',kwargs=kwargs)
        scheduler.add_job(request_for_week_streaker, 'date', kwargs=kwargs)
    if starter_auth["week_streaker"]: setup_week_streaker()

    ###########################################################################
    # Quant
    ###########################################################################
    def setup_quant():
        def request_for_quant(**kwargs):
            use_cases__shared.put({"quant": kwargs})
        kwargs = {"ticker": ticker, "dataset_id": "week", "realtime": True}
        # S'assurer que la cache est peuplée sur tout les workers
        for _ in range(2*nb_worker): scheduler.add_job(request_for_quant, 'date', kwargs={})
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
        kwargs = {"info": information_displayed, "output_dir": output_dir_historian, "accumulator_historian": accumulator_historian}
        scheduler.add_job(request_for_historian, 'cron', day_of_week='mon-fri', hour='9', minute='30-59', kwargs=kwargs)
        scheduler.add_job(request_for_historian, 'cron', day_of_week='mon-fri', hour='10-15', minute='*', kwargs=kwargs)
        scheduler.add_job(request_for_historian, 'cron', day_of_week='mon-fri', hour='16', minute='0', kwargs=kwargs)
        scheduler.add_job(request_for_historian_dump, 'cron', day_of_week='mon-fri', hour='11,13,16', minute='0', kwargs=kwargs)
    if starter_auth["historian"]: setup_historian()

    scheduler.start()
    # =============================================================================
    # Print Human-Readable Cron Schedules
    # =============================================================================
    print("\n" + "=" * 75)
    print("📅 ACTIVE SCHEDULES OVERVIEW".center(75))
    print("=" * 75)

    schedules = [
        ("realtime", "⏱️ Realtime Values", "Every minute from 09:30 to 16:00 (Mon-Fri)"),
        ("bouclier_spx", "🛡️ Bouclier SPX", "08:15 (Mon) | 09:30, 13:30, 16:00 (Mon-Fri)"),
        ("trend_ema", "📈 Trend EMA", "Every 30 mins from 09:00 to 16:00 (Mon-Fri)"),
        ("horizontal_lines", "📏 Horizontal Lines", "07:00, 09:31 (Mon-Fri) for all timeframes"),
        ("vera", "🔮 Vera Model", "Every 5 mins from 09:45 to 16:00 (Mon-Fri)"),
        ("gap", "🕳️  Gap Persistence", "09:32 (Mon-Fri)"),
        ("day_streaker", "🔥 Day Streaker", "Every 30 mins from 09:00 to 16:30 (Mon-Fri)"),
        ("week_streaker", "🔥 Week Streaker", "Every 30 mins from 09:00 to 16:30 (Mon-Fri)"),
        ("historian", "🏛️ Historian", "Every minutes from 09:30 to 16:00 (Mon-Fri)"),
    ]
    for key, title, desc in schedules:
        if starter_auth.get(key):
            print(f"{title:<25} : {desc}")

    print("\n🚀 *Note: All scheduled tasks will also execute immediately at startup.*")
    print("=" * 75)

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
        print("\033[H", end="")
        if starter_auth["activer_random_update"]: information_displayed = random_update_info(info=information_displayed)
        resuls_from_worker = []
        # Get the computed data, if any
        for k in range(0, nb_worker):
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
                    information_displayed['bouclier_spx'] = "🟢 Environnement favorable"
                # synthese_globale = "🟢 Environnement favorable. Les indicateurs de risque sont faibles. La poursuite des stratégies standards est justifiée."
                elif v['avg_score'] <= 6.5:
                    information_displayed['bouclier_spx'] = "🟡 Prudence requise"
                else:
                    information_displayed['bouclier_spx'] = "🔴 Alerte de risque élevée"
            if k == "trend_computation":
                for k2, v2 in v.items():
                    information_displayed[k2] = v2
            if k == "realtime_values":
                for k2, v2 in v.items():
                    information_displayed[k2] = f"{v2:.0f}"
            if k in ["h20", "hmonth", "h15", "h10", "h5", "hweek", "h0"]:
                information_displayed[f"{k}_down"] = []
                information_displayed[f"{k}_up"] = []
                if k == "h20": dte_date = get_next_step(datetime.now(), "day", 20).strftime('%Y-%m-%d')
                if k == "h15": dte_date = get_next_step(datetime.now(), "day", 15).strftime('%Y-%m-%d')
                if k == "h10": dte_date = get_next_step(datetime.now(), "day", 10).strftime('%Y-%m-%d')
                if k == "h5": dte_date = get_next_step(datetime.now(), "day", 5).strftime('%Y-%m-%d')
                if k == "h0": dte_date = datetime.now().strftime('%Y-%m-%d')
                if k == "hmonth": dte_date = get_month_range(datetime.now())[1].strftime('%Y-%m-%d')
                if k == "hweek": dte_date = get_week_range(datetime.now())[1].strftime('%Y-%m-%d')
                for a_dict in v['low']: information_displayed[f"{k}_down"].append({"price": a_dict['predicted_low'], "proba": float(a_dict['probability_predicted_low'])/100., "date": dte_date})
                while len(information_displayed[f"{k}_down"]) < information_displayed["max_horizontal_lines"]: information_displayed[f"{k}_down"].append({"price": 1111, "proba": 0, "date": "--------"})
                for a_dict in v['high']: information_displayed[f"{k}_up"].append({"price": a_dict['predicted_high'], "proba": float(a_dict['probability_predicted_high'])/100., "date": dte_date})
                while len(information_displayed[f"{k}_up"]) < information_displayed["max_horizontal_lines"]: information_displayed[f"{k}_up"].append({"price": 9999, "proba": 0, "date": "--------"})
            if k == "vera":
                information_displayed["vera"] = v
            if k == "gap":
                information_displayed.update({'persistance_conditionelle': v})
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
                dd_tmp = extraire_info_streaker(v)
                information_displayed.update({'day_streaker': {'b': dd_tmp["valeur_b"], 'proba': dd_tmp["proba"], 'dir': dd_tmp["direction"]}, })
            if k == "week_streaker":
                dd_tmp = extraire_info_streaker(v)
                information_displayed.update({'week_streaker': {'b': dd_tmp["valeur_b"], 'proba': dd_tmp["proba"], 'dir': dd_tmp["direction"]}, })
        now = datetime.now()
        now_txt = now.strftime('%Y-%m-%d %Hh%Mm%Ss')
        previous_now = get_next_step(now, "day", -1)
        previous_now_txt = previous_now.strftime('%Y-%m-%d')
        information_displayed.update({'now': now, 'now_txt': now_txt, 'previous_now': previous_now, 'previous_now_txt': previous_now_txt, 'ticker': ticker, 'max_horizontal_lines': 3,})

        # Display info
        if starter_auth["activer_console"]:
            if 0 != len(resuls_from_worker): clean_console()
            make_display(info=information_displayed)

        # Wait
        time.sleep(3)


if __name__ == "__main__":
    freeze_support()
    entry()