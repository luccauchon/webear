try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import time
from utils import str2bool, format_duration, round_price_for_put_credit_spread, round_price_for_call_credit_spread
from optimizers.quantloop.quant_morning_alpha_miner import entry as entry_quant_morning_alpha_miner
from optimizers.quantloop.quant_morning_alpha_miner import TopNEnsembleModel
from fetchers.data_factory import factory_load_data
import os
import sys
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pickle
import json
from collections import Counter
import io
from multiprocessing import Process, Queue, freeze_support, Value
import warnings
import argparse
from argparse import Namespace
import os
import time
from utils import format_execution_time
from functools import lru_cache

def _parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Verbose output')
    # Ticker de l'actif financier (ex: ^GSPC)
    parser.add_argument('--ticker', type=str, default="^GSPC", help="Ticker de l'actif à analyser.")

    # Identifiant du dataset à charger
    parser.add_argument('--dataset-id', type=str, default="intraday_1min", help="ID du dataset à charger.")

    parser.add_argument("--realtime", action="store_true", default=False, help="")
    parser.add_argument("--models-dir", type=str, default="quant", help="Directory where all the models are located")

    # Fenêtres d'analyse matinale (listes)
    parser.add_argument('--st1', type=str, nargs="+", default=["09:30"], help="Liste des heures de début de la fenêtre d'analyse (HH:MM).")
    parser.add_argument('--st2', type=str, nargs="+", default=["09:45"], help="Liste des heures de fin de la fenêtre d'analyse (HH:MM).")

    # Fenêtres cibles après-midi (listes)
    parser.add_argument('--et1', type=str, nargs="+", default=["15:59"], help="Liste des heures de début de la fenêtre cible (HH:MM).")
    parser.add_argument('--et2', type=str, nargs="+", default=["16:00"], help="Liste des heures de fin de la fenêtre cible (HH:MM).")

    # Définition des cibles (liste de seuils)
    parser.add_argument('--seuil-pos', type=float, nargs="+", default=[0.999], help="Liste des seuils multiplicateurs du Daily Open pour la cible.")
    parser.add_argument('--nb-worker', type=int, default=12, help="Nombre de workers pour l'évaluation parallèle.")
    return parser.parse_args()


def _realtime(ticker, st1, st2, et1, et2, seuil_pos, models_dir, verbose):
    df_1minute = factory_load_data(_dataset_id="intraday-1min", _ticker=ticker, _args={"realtime": True})
    if isinstance(df_1minute.columns, pd.MultiIndex): df_1minute.columns = df_1minute.columns.get_level_values(0)
    df_1minute = df_1minute.sort_index()
    winrate_file = os.path.join(models_dir, "win_rate.json")
    if os.path.exists(winrate_file):
        with open(winrate_file, "r", encoding="utf-8") as fichier:
            computed_win_rate = json.load(fichier)
    else: computed_win_rate = None
    all_models = _read_models(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos, models_dir=models_dir)
    pred_results = _compute_prediction(df_=df_1minute, all_models_=all_models, args_=Namespace(et1=et1, et2=et2, st1=st1, st2=st2, seuil_pos=seuil_pos, models_dir=models_dir, verbose=verbose))
    result_pcs, result_ccs, best_model_ccs, best_model_pcs = pred_results['result_pcs'], pred_results['result_ccs'], pred_results['best_model_ccs'], pred_results['best_model_pcs']
    groupes_par_jour = list(df_1minute.groupby(df_1minute.index.date))
    date_derniere_journee, df_dernier_jour = groupes_par_jour[-1]
    open_derniere_journee = df_dernier_jour["Open"].iloc[0]
    strike_price = int(best_model_pcs.get_adjusted_strike_price(open_derniere_journee))
    key_dict = _get_key_from_parameters(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos)
    win_rate = [0.] if computed_win_rate is None else [v for k, v in computed_win_rate.items() if key_dict==k]
    assert len(win_rate) == 1, f"La donnees {key_dict} devrait exister dans le fichier {os.path.join(models_dir, 'win_rate.json')}"
    win_rate = win_rate[0]
    win_dict = _compute_win_situation_from_computed_predictions(result_pcs=result_pcs, result_ccs=result_ccs, future_price_max=None, future_price_min=None, strike_price=strike_price)
    dict_returned = {"action": win_dict["action"], "strike_price": strike_price, "win_rate": win_rate}
    if verbose: print(f"{dict_returned=}")
    return dict_returned


def _get_models_from_parameters(all_models, **kwargs):
    results = []
    for a_model, a_filename in all_models:
        keep_model = True
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                if not hasattr(a_model.args, k) or str(getattr(a_model.args, k)) != str(v[0]): keep_model = False
            else:
                if not hasattr(a_model.args, k) or str(getattr(a_model.args, k)) != str(v): keep_model = False
        if keep_model: results.append((a_model, a_filename))
    return results


def _get_best_model(models):
    # Le protocole idéal pour faire votre choix :
    # 1. Comparez les scores F1 sur le jeu de test de tous vos modèles entraînés.
    # 2. Éliminez les modèles qui montrent un écart trop grand entre la CV et le test (signe de surapprentissage).
    # 3. Sélectionnez le modèle avec le F1-score de test le plus élevé.
    # 4. Bonus : Si deux modèles ont un score de test presque identique (ex: 0.852 vs 0.851), choisissez celui qui a le score de CV le plus stable (variance la plus faible)
    best_model, best_ecart_cv_tf1, best_std_cv = None, 0, 0
    for a_model_, a_filename_ in models:
        # Sélectionnez le modèle avec le F1-score de test le plus élevé.
        if best_model is None or a_model_.get_test_f1_score() > best_model[0].get_test_f1_score():
            best_model        = a_model_, a_filename_
            best_ecart_cv_tf1 = abs(a_model_.get_mean_std_cv_f1_score()[0] - a_model_.get_test_f1_score())
            best_std_cv       = a_model_.get_mean_std_cv_f1_score()[1]
        # Si deux modèles ont un score de test identique, choisissez celui qui a le score de CV le plus stable (variance la plus faible)
        elif a_model_.get_test_f1_score() == best_model[0].get_test_f1_score() and a_model_.get_mean_std_cv_f1_score()[1] < best_std_cv:
            best_model        = a_model_, a_filename_
            best_ecart_cv_tf1 = abs(a_model_.get_mean_std_cv_f1_score()[0] - a_model_.get_test_f1_score())
            best_std_cv       = a_model_.get_mean_std_cv_f1_score()[1]
        # Score de test identique + même std en CV, choisir celui avec le plus de sous-modèles
        elif a_model_.get_test_f1_score() == best_model[0].get_test_f1_score() and a_model_.get_mean_std_cv_f1_score()[1] == best_std_cv and a_model_.n_models_ > best_model[0].n_models_:
            best_model        = a_model_, a_filename_
            best_ecart_cv_tf1 = abs(a_model_.get_mean_std_cv_f1_score()[0] - a_model_.get_test_f1_score())
            best_std_cv       = a_model_.get_mean_std_cv_f1_score()[1]
    return best_model


def _compute_prediction(df_, all_models_, args_):
    models_for_ccs = _get_models_from_parameters(all_models_, **{"st1": args_.st1, "st2": args_.st2, "et1": args_.et1, "et2": args_.et2, "seuil_pos": args_.seuil_pos, "et_gt_open": 0})
    if 0 == len(models_for_ccs): raise Exception(f"Aucun modèle pour le CALL CREDIT SPREAD (et-gt-open=0) dans {getattr('models_dir', args_.models_dir, None)}. Revoir les paramètres utilisés.")
    models_for_pcs = _get_models_from_parameters(all_models_, **{"st1": args_.st1, "st2": args_.st2, "et1": args_.et1, "et2": args_.et2, "seuil_pos": args_.seuil_pos, "et_gt_open": 1})
    if 0 == len(models_for_pcs): raise Exception(f"Aucun modèle pour le PUT CREDIT SPREAD (et-gt-open=1) dans {getattr('models_dir', args_.models_dir, None)}. Revoir les paramètres utilisés.")
    best_model_ccs, file_model_ccs = _get_best_model(models=models_for_ccs)
    best_model_pcs, file_model_pcs = _get_best_model(models=models_for_pcs)
    if getattr(args_, "verbose", False): print(f"{'='*80}\nCALL CREDIT SPREAD\n{'='*80}")
    result_ccs = entry_quant_morning_alpha_miner(args=Namespace(modelfile=file_model_ccs, realtime=True, daily_df_precomputed=df_.copy(), verbose=getattr(args_, "verbose", False)))
    if getattr(args_, "verbose", False): print(f"{'='*80}\nPUT CREDIT SPREAD\n{'='*80}")
    result_pcs = entry_quant_morning_alpha_miner(args=Namespace(modelfile=file_model_pcs, realtime=True, daily_df_precomputed=df_.copy(), verbose=getattr(args_, "verbose", False)))
    return {'result_ccs': result_ccs, 'best_model_ccs': best_model_ccs, 'result_pcs': result_pcs, 'best_model_pcs': best_model_pcs, 'file_model_ccs': file_model_ccs, 'file_model_pcs': file_model_pcs}


def _read_models(st1, st2, et1, et2, seuil_pos, models_dir):
    restrict_model = None
    all_models = []
    file_paths = [Path(root) / file for root, _, files in os.walk(models_dir) for file in files if file.endswith(".joblib")]
    key_to_look_for = f"_ensemble_model__{seuil_pos}__{st1}-{st2}__{et1}00-{et2}00".replace(':', '')
    if isinstance(seuil_pos, (list, tuple)):
        key_to_look_for = f"_ensemble_model__{seuil_pos[0]}__{st1[0]}-{st2[0]}__{et1[0]}00-{et2[0]}00".replace(':', '')
    for ensemble_filename in file_paths:
        ensemble_filename = str(ensemble_filename)
        if key_to_look_for in ensemble_filename:
            if restrict_model is not None:
                if restrict_model not in ensemble_filename:
                    continue
            try:
                model = TopNEnsembleModel.load(ensemble_filename)
                if model is not None: all_models.append((model, ensemble_filename))
            except Exception:
                pass # Pour l'instant.
    return all_models


def _get_key_from_parameters(st1, st2, et1, et2, seuil_pos):
    if isinstance(st1, (list, tuple)):
        return f"{st1[0]} :: {st2[0]} | {et1[0]} :: {et2[0]} | {seuil_pos[0]}"
    return f"{st1} :: {st2} | {et1} :: {et2} | {seuil_pos}"


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared):
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)

    # Traitement des requêtes
    resultats_dct, test_date_t1 = {}, None
    while True:
        t1 = time.time()
        _use_case = None
        try:
            item = use_cases__shared.get(timeout=2)
            _use_case = item
        except:
            break  # Queue is empty or no more items within timeout
        st1, st2, et1, et2, seuil_pos, df_1minute, models_dir, lookback_jours, verbose = _use_case
        t2 = time.time()
        key_dict = _get_key_from_parameters(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos)
        all_models = _read_models(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos, models_dir=models_dir)
        t3 = time.time()
        groupes_par_jour = list(df_1minute.groupby(df_1minute.index.date))
        total_jours = len(groupes_par_jour)

        if test_date_t1 is None:
            test_date_t1 = pd.to_datetime(sorted(list(set([am[0].train_test_dates['test_dates'][0] for am in all_models])))[-1])

        # On sélectionne uniquement les jours restants (on exclut les 'lookback_jours' derniers jours)
        jours_restants = groupes_par_jour if 0 == lookback_jours else groupes_par_jour[:-lookback_jours]
        if 0 == len(jours_restants): continue
        date_derniere_journee, df_dernier_jour = jours_restants[-1]

        # 2. Valeur de l'Open de la dernière journée (première ligne du jour)
        open_derniere_journee = df_dernier_jour["Open"].iloc[0]
        #print(f"Open du {date_derniere_journee} : {open_derniere_journee}")

        # Affichage des infos globales pour ce lookback
        df_filtre = pd.concat([df_jour for _, df_jour in jours_restants])
        if df_filtre.index[-1].tz_localize(None) < test_date_t1: continue
        date_debut = jours_restants[0][0]
        date_fin = jours_restants[-1][0]
        if verbose: print(f"[{os.getpid()}] >>> {date_debut}::{date_fin} --> {len(jours_restants)} jours restants")
        if verbose: print(f"[{os.getpid()}]  \t >>> {df_filtre.index[0]} --> {df_filtre.index[-1]}")
        if lookback_jours > total_jours: continue
        t4 = time.time()
        pred_results = _compute_prediction(df_=df_filtre, all_models_=all_models, args_=Namespace(et1=et1, et2=et2, st1=st1, st2=st2, seuil_pos=seuil_pos, models_dir=models_dir))
        result_pcs, result_ccs, best_model_ccs, best_model_pcs = pred_results['result_pcs'], pred_results['result_ccs'], pred_results['best_model_ccs'], pred_results['best_model_pcs']
        t5 = time.time()

        #print(f"PUT CREDIT SPREAD  : {result_pcs['result']['predicted_class'].values[0]}  {result_pcs['result']['median_probability'].values[0]:.0%}")
        #print(f"CALL CREDIT SPREAD : {result_ccs['result']['predicted_class'].values[0]}  {result_ccs['result']['median_probability'].values[0]:.0%}")

        #print(f"Open du {date_derniere_journee} : {open_derniere_journee}")
        #print(f"ccs: {open_derniere_journee} --> {best_model_ccs.get_adjusted_strike_price(open_derniere_journee)}")
        #print(f"pcs: {open_derniere_journee} --> {best_model_pcs.get_adjusted_strike_price(open_derniere_journee)}")
        #print(f"Toujours la même valeur parce que même seuil_pos")
        assert best_model_ccs.get_adjusted_strike_price(open_derniere_journee) == best_model_pcs.get_adjusted_strike_price(open_derniere_journee)
        strike_price = best_model_pcs.get_adjusted_strike_price(open_derniere_journee)

        # 3. Valeurs entre 15h.. et 16h00 de la dernière journée
        et1, et2       = best_model_ccs.get_et1_et2()
        df_fin_journee = df_dernier_jour.between_time(et1, et2)
        # Affichage des valeurs (par exemple la colonne 'Open' ou 'Close' selon vos besoins)
        # print(f"[{os.getpid()}]  \t >>>\nValeurs entre {et1} et {et2} pour le {date_derniere_journee} :")
        # print(f"[{os.getpid()}]  \t >>> {df_fin_journee['Close']}")
        # 2. On extrait le minimum et le maximum (Exemple basé sur les colonnes High et Low)
        future_price_min = df_fin_journee["Low"].min()
        future_price_max = df_fin_journee["High"].max()
        # print(f"[{os.getpid()}]  \t >>>Entre {et1} et {et2} le {date_derniere_journee} :")
        # print(f"[{os.getpid()}]  \t >>> - Prix Minimum (Low) : {prix_min}")
        # print(f"[{os.getpid()}]  \t >>> - Prix Maximum (High) : {prix_max}")
        assert best_model_ccs.get_adjusted_strike_price(open_derniere_journee) == best_model_pcs.get_adjusted_strike_price(open_derniere_journee)

        win_dict = _compute_win_situation_from_computed_predictions(result_pcs=result_pcs, result_ccs=result_ccs, future_price_max=future_price_max, future_price_min=future_price_min, strike_price=strike_price)

        if lookback_jours not in resultats_dct: resultats_dct.update({lookback_jours: {}})
        if key_dict not in resultats_dct[lookback_jours]: resultats_dct[lookback_jours].update({key_dict: []})
        resultats_dct[lookback_jours][key_dict].append({date_derniere_journee: win_dict["win"]})

        t6 = time.time()
        if verbose: print(f"[{os.getpid()}]  \t >>> {format_execution_time(t2 - t1)} / {format_execution_time(t3 - t2)} / {format_execution_time(t4 - t3)} "
                          f"/ {format_execution_time(t5 - t4)} / {format_execution_time(t6 - t5)}")
    out__shared.put(resultats_dct)


def _compute_win_situation_from_computed_predictions(result_pcs, result_ccs, future_price_max, future_price_min, strike_price):
    pcs__predicted_class    = result_pcs["result"]["predicted_class"].values[0]
    pcs__median_probability = result_pcs["result"]["median_probability"].values[0]
    pcs__std_probability    = result_pcs["result"]["std_probability"].values[0]
    ccs__predicted_class    = result_ccs["result"]["predicted_class"].values[0]
    ccs__median_probability = result_ccs["result"]["median_probability"].values[0]
    ccs__std_probability    = result_ccs["result"]["std_probability"].values[0]

    #######################################################################
    #######################################################################
    # Si PCS==1 et CCS==0 --> valeur à la fin de la journée devrait être supérieure à strike_price
    if ((1 == pcs__predicted_class and pcs__median_probability > 0.65 and pcs__std_probability < 0.1) and
            (0 == ccs__predicted_class and ccs__median_probability < 0.35 and ccs__std_probability < 0.1)):
        action = "PUT_CREDIT_SPREAD"
        if future_price_max is not None:
            win = 1 if future_price_max >= strike_price else 0
        else: win = 1
    #######################################################################
    # Si PCS==0 et CCS==1 --> valeur à la fin de la journée devrait être inférieure à strike_price
    elif ((0 == pcs__predicted_class and pcs__median_probability < 0.35 and pcs__std_probability < 0.1) and
          (1 == ccs__predicted_class and ccs__median_probability > 0.65 and ccs__std_probability < 0.1)):
        action = "CALL_CREDIT_SPREAD"
        if future_price_max is not None:
            win = 1 if future_price_min <= strike_price else 0
        else: win = 1
    else:
        action = "PASS"
        win = -1
    #######################################################################
    #######################################################################

    return {"win": win, "action": action}


def entry(args):
    if args.realtime:
        return _realtime(ticker=args.ticker, st1=args.st1, st2=args.st2, et1=args.et1, et2=args.et2, seuil_pos=args.seuil_pos, models_dir=args.models_dir, verbose=getattr(args, "verbose", False))

    ###########################################################################
    # Lecture des données
    ###########################################################################
    df_1minute = factory_load_data(_dataset_id="intraday-1min", _ticker=args.ticker, _args={})
    if isinstance(df_1minute.columns, pd.MultiIndex): df_1minute.columns = df_1minute.columns.get_level_values(0)
    df_1minute = df_1minute.sort_index()
    close_col = 'Close'
    nb_worker = getattr(args, "nb_worker", 12)

    ###########################################################################
    # Boucle pour effectuer le backtest
    ###########################################################################
    assert len(args.st1)==len(args.st2)==len(args.et1)==len(args.et2)==len(args.seuil_pos)

    use_cases, tmp_lst, total_jours = [], [], 365
    for st1, st2, et1, et2, seuil_pos in tqdm(zip(args.st1, args.st2, args.et1, args.et2, args.seuil_pos)):
        key_dict = _get_key_from_parameters(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos)
        if key_dict in tmp_lst: continue  # Pass it. Already computed (because generated cases for et-gt-open==0 and ==1)
        for lookback_jours in range(0, total_jours - 3):
            use_cases.append((st1, st2, et1, et2, seuil_pos, df_1minute, args.models_dir, lookback_jours, args.verbose))
        tmp_lst.append(key_dict)
    print(f"Got {len(use_cases)} use cases to process with {nb_worker} workers...")
    data_from_workers = []
    use_cases__shared, master_cmd__shared = Queue(), Value("i", 0)
    out__shared = [Queue(1) for k in range(0, nb_worker)]

    # Lancement des workers
    for k in range(0, nb_worker):
        p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k],))
        p.start()
    # Envoie les informations aux workers pour traitement
    for use_case in use_cases:
        use_cases__shared.put(use_case)
    # Autoriser les workers à traiter
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1
    # Récupération des résultats
    for k in range(0, nb_worker):
        data_from_workers.append(out__shared[k].get())
    filtered_data = [d for d in data_from_workers if d]
    key_dict__2__results, key_dict__2__win_rate = {}, {}
    for _fd in filtered_data:
        for lookback_jours, _tmp_dict in _fd.items():
            for key_dict, _pv in _tmp_dict.items():
                assert 1 == len(_pv)
                assert 1 == len(_pv[0])
                date_derniere_journee, win = next(iter(_pv[0].items()))
                if key_dict not in key_dict__2__results: key_dict__2__results.update({key_dict: []})
                key_dict__2__results[key_dict].append(win)
    for k, v in key_dict__2__results.items():
        stats = (lambda d: {"total": len(d), "wins": d.count(1), "losses": d.count(0), "passes": d.count(-1),
                            "test_density": round((d.count(1) + d.count(0)) / (d.count(1) + d.count(0) + d.count(-1)), 4) if (d.count(1) + d.count(0)) else 0})(v)
        stats["test_win_rate"] = stats['wins'] / (stats['total'] - stats['passes'])
        print(f"{k=}   {stats=}")
        key_dict__2__win_rate.update({k: stats["test_win_rate"]})
    print(key_dict__2__win_rate)
    if not os.path.exists(os.path.join(args.models_dir, "win_rate.json")):
        with open(os.path.join(args.models_dir, "win_rate.json"), "w", encoding="utf-8") as f:
            json.dump(key_dict__2__win_rate, f, indent=4)
    else:
        with open(os.path.join(args.models_dir, "NEW__win_rate.json"), "w", encoding="utf-8") as f:
            json.dump(key_dict__2__win_rate, f, indent=4)
    return None


if __name__ == "__main__":
    freeze_support()
    args = _parse_arguments()
    entry(args=args)