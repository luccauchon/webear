from llvmlite.binding import passmanagers

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

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

    return parser.parse_args()


def realtime(ticker, st1, st2, et1, et2, seuil_pos, models_dir):
    df_1minute = factory_load_data(_dataset_id="intraday-1min", _ticker=ticker, _args={"realtime": True})
    if isinstance(df_1minute.columns, pd.MultiIndex): df_1minute.columns = df_1minute.columns.get_level_values(0)
    df_1minute = df_1minute.sort_index()
    with open(os.path.join(models_dir, "win_rate.json"), "r", encoding="utf-8") as fichier:
        computed_win_rate = json.load(fichier)
    all_models = read_models(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos, models_dir=models_dir)
    pred_results = compute_prediction(df_=df_1minute, all_models_=all_models, args_=Namespace(et1=et1, et2=et2, st1=st1, st2=st2, seuil_pos=seuil_pos, models_dir=models_dir))
    result_pcs, result_ccs, best_model_ccs, best_model_pcs = pred_results['result_pcs'], pred_results['result_ccs'], pred_results['best_model_ccs'], pred_results['best_model_pcs']
    groupes_par_jour = list(df_1minute.groupby(df_1minute.index.date))
    date_derniere_journee, df_dernier_jour = groupes_par_jour[-1]
    open_derniere_journee = df_dernier_jour["Open"].iloc[0]
    strike_price = best_model_pcs.get_adjusted_strike_price(open_derniere_journee)
    key_dict = get_key_from_parameters(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos)
    win_rate = [cwr[1] for cwr in computed_win_rate if key_dict==cwr[0]]
    assert len(win_rate) == 1
    win_rate = win_rate[0]
    # Repris du code du backtesting
    if 1 == result_pcs["result"]["predicted_class"].values[0] and 0 == result_ccs["result"]["predicted_class"].values[0]:
        return {"action": "PUT_CREDIT_SPREAD", "strike_price": strike_price, "win_rate": win_rate}
    if 0 == result_pcs["result"]["predicted_class"].values[0] and 1 == result_ccs["result"]["predicted_class"].values[0]:
        return {"action": "CALL_CREDIT_SPREAD", "strike_price": strike_price, "win_rate": win_rate}
    return {"action": "PASS", "strike_price": strike_price}


def get_models_from_parameters(all_models, **kwargs):
    results = []
    for a_model, a_filename in all_models:
        keep_model = True
        for k, v in kwargs.items():
            if not hasattr(a_model.args, k) or str(getattr(a_model.args, k)) != str(v): keep_model = False
        if keep_model: results.append((a_model, a_filename))
    return results


def get_best_model(models):
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


def compute_prediction(df_, all_models_, args_):
    models_for_ccs = get_models_from_parameters(all_models_, **{"st1": args_.st1, "st2": args_.st2, "et1": args_.et1, "et2": args_.et2, "seuil_pos": args_.seuil_pos, "et_gt_open": 0})
    if 0 == len(models_for_ccs): raise Exception(f"Aucun modèle pour le CALL CREDIT SPREAD (et-gt-open=0) dans {getattr('models_dir', args_.models_dir, None)}. Revoir les paramètres utilisés.")
    models_for_pcs = get_models_from_parameters(all_models_, **{"st1": args_.st1, "st2": args_.st2, "et1": args_.et1, "et2": args_.et2, "seuil_pos": args_.seuil_pos, "et_gt_open": 1})
    if 0 == len(models_for_pcs): raise Exception(f"Aucun modèle pour le PUT CREDIT SPREAD (et-gt-open=1) dans {getattr('models_dir', args_.models_dir, None)}. Revoir les paramètres utilisés.")
    best_model_ccs, file_model_ccs = get_best_model(models=models_for_ccs)
    best_model_pcs, file_model_pcs = get_best_model(models=models_for_pcs)
    result_ccs = entry_quant_morning_alpha_miner(args=Namespace(modelfile=file_model_ccs, realtime=True, daily_df_precomputed=df_, verbose=False))
    result_pcs = entry_quant_morning_alpha_miner(args=Namespace(modelfile=file_model_pcs, realtime=True, daily_df_precomputed=df_, verbose=False))
    return {'result_ccs': result_ccs, 'best_model_ccs': best_model_ccs, 'result_pcs': result_pcs, 'best_model_pcs': best_model_pcs, 'file_model_ccs': file_model_ccs, 'file_model_pcs': file_model_pcs}


def read_models(st1, st2, et1, et2, seuil_pos, models_dir):
    all_models = []
    file_paths = [Path(root) / file for root, _, files in os.walk(models_dir) for file in files if file.endswith(".pkl")]
    for file_path in file_paths:
        ensemble_filename = str(file_path)
        if f"_ensemble_model__{seuil_pos}__{st1}-{st2}__{et1}00-{et2}00".replace(':', '') in ensemble_filename:
            model = TopNEnsembleModel.load(ensemble_filename)
            all_models.append((model, ensemble_filename))
    return all_models


def get_key_from_parameters(st1, st2, et1, et2, seuil_pos):
    return f"{st1} :: {st2} | {et1} :: {et2} | {seuil_pos}"


def entry(args):
    if args.realtime:
        return realtime(ticker=args.ticker, st1=args.st1, st2=args.st2, et1=args.et1, et2=args.et2, seuil_pos=args.seuil_pos, models_dir=args.models_dir)

    ###########################################################################
    # Lecture des données
    ###########################################################################
    df_1minute = factory_load_data(_dataset_id="intraday-1min", _ticker=args.ticker, _args={})
    if isinstance(df_1minute.columns, pd.MultiIndex): df_1minute.columns = df_1minute.columns.get_level_values(0)
    df_1minute = df_1minute.sort_index()
    close_col = 'Close'

    ###########################################################################
    # Boucle pour effectuer le backtest
    ###########################################################################
    assert len(args.st1)==len(args.st2)==len(args.et1)==len(args.et2)==len(args.seuil_pos)
    resultats_dct = {}
    for st1, st2, et1, et2, seuil_pos in tqdm(zip(args.st1, args.st2, args.et1, args.et2, args.seuil_pos)):
        ###########################################################################
        # Lecture des modèles requis
        ###########################################################################
        all_models = read_models(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos, models_dir=args.models_dir)
        #print(f" Fenêtre d\'analyse : {st1} :: {st2}")
        #print(f" Fenêtre cible     : {et1} :: {et2}")
        #print(f" Niveau de prix    : {seuil_pos}")
        #print(f" Nombre de modèles : {len(all_models)}")
        groupes_par_jour = list(df_1minute.groupby(df_1minute.index.date))
        total_jours = len(groupes_par_jour)
        date_2_result = {}
        # 2. On itère sur le nombre de jours à reculer (ex: de 1 jour à (total - 3) jours)
        for lookback_jours in tqdm(range(0, total_jours - 3)):
            # On sélectionne uniquement les jours restants (on exclut les 'lookback_jours' derniers jours)
            jours_restants = groupes_par_jour if 0 == lookback_jours else groupes_par_jour[:-lookback_jours]
            date_derniere_journee, df_dernier_jour = jours_restants[-1]

            # 2. Valeur de l'Open de la dernière journée (première ligne du jour)
            open_derniere_journee = df_dernier_jour["Open"].iloc[0]
            #print(f"Open du {date_derniere_journee} : {open_derniere_journee}")

            # Affichage des infos globales pour ce lookback
            date_debut = jours_restants[0][0]
            date_fin   = jours_restants[-1][0]
            #print(f" >>> {date_debut}::{date_fin} --> {len(jours_restants)} jours restants")
            df_filtre = pd.concat([df_jour for _, df_jour in jours_restants])
            #print(f" \t >>> {df_filtre.index[0]} --> {df_filtre.index[-1]}")
            pred_results = compute_prediction(df_=df_filtre, all_models_=all_models, args_=Namespace(et1=et1, et2=et2, st1=st1, st2=st2, seuil_pos=seuil_pos, models_dir=args.models_dir))
            result_pcs, result_ccs, best_model_ccs, best_model_pcs = pred_results['result_pcs'], pred_results['result_ccs'], pred_results['best_model_ccs'], pred_results['best_model_pcs']

            #print(f"PUT CREDIT SPREAD  : {result_pcs['result']['predicted_class'].values[0]}  {result_pcs['result']['median_probability'].values[0]:.0%}")
            #print(f"CALL CREDIT SPREAD : {result_ccs['result']['predicted_class'].values[0]}  {result_ccs['result']['median_probability'].values[0]:.0%}")

            #print(f"Open du {date_derniere_journee} : {open_derniere_journee}")
            #print(f"ccs: {open_derniere_journee} --> {best_model_ccs.get_adjusted_strike_price(open_derniere_journee)}")
            #print(f"pcs: {open_derniere_journee} --> {best_model_pcs.get_adjusted_strike_price(open_derniere_journee)}")
            #print(f"Toujours la même valeur parce que même seuil_pos")
            assert best_model_ccs.get_adjusted_strike_price(open_derniere_journee) == best_model_pcs.get_adjusted_strike_price(open_derniere_journee)
            strike_price = best_model_pcs.get_adjusted_strike_price(open_derniere_journee)

            # 3. Valeurs entre 15h.. et 16h00 de la dernière journée
            et1, et2 = best_model_ccs.get_et1_et2()
            df_fin_journee = df_dernier_jour.between_time(et1, et2)
            # Affichage des valeurs (par exemple la colonne 'Open' ou 'Close' selon vos besoins)
            #print(f"\nValeurs entre {et1} et {et2} pour le {date_derniere_journee} :")
            #print(df_fin_journee["Close"])
            # 2. On extrait le minimum et le maximum (Exemple basé sur les colonnes High et Low)
            prix_min = df_fin_journee["Low"].min()
            prix_max = df_fin_journee["High"].max()
            #print(f"Entre {et1} et {et2} le {date_derniere_journee} :")
            #print(f" - Prix Minimum (Low) : {prix_min}")
            #print(f" - Prix Maximum (High) : {prix_max}")
            assert best_model_ccs.get_adjusted_strike_price(open_derniere_journee) == best_model_pcs.get_adjusted_strike_price(open_derniere_journee)
            win = -1
            # Si PCS==1 et CCS==0 --> valeur à la fin de la journée devrait être supérieure à strike_price
            if   1 == result_pcs["result"]["predicted_class"].values[0] and 0 == result_ccs["result"]["predicted_class"].values[0]:
                win = 1 if prix_max >= strike_price else 0
            # Si PCS==0 et CCS==1 --> valeur à la fin de la journée devrait être inférieure à strike_price
            elif 0 == result_pcs["result"]["predicted_class"].values[0] and 1 == result_ccs["result"]["predicted_class"].values[0]:
                win = 1 if prix_min <= strike_price else 0
            else:
                win = -1
            date_2_result.update({date_derniere_journee: win})
            #input("ENTER")
        # 1. On compte les occurrences de chaque résultat (1, 0, -1) dans le dictionnaire
        statistiques = Counter(date_2_result.values())
        # 2. Affichage des résultats
        #print(f"--- Statistiques globales ({len(date_2_result)} jours analysés) ---")
        #print(f"Win  (1)      : {statistiques.get(1, 0)} jours")
        #print(f"Loss (0)      : {statistiques.get(0, 0)} jours")
        #print(f"Fait rien (-1): {statistiques.get(-1, 0)} jours")
        total_signal_activated = statistiques.get(1, 0) + statistiques.get(0, 0)
        win_rate = statistiques.get(1, 0) / total_signal_activated
        key_dict = get_key_from_parameters(st1=st1, st2=st2, et1=et1, et2=et2, seuil_pos=seuil_pos)
        print(f"Win Rate: {win_rate:.2%}  ({key_dict})")
        assert key_dict not in resultats_dct
        resultats_dct.update({key_dict: win_rate})
    resultats_dct_trie = sorted(resultats_dct.items(), key=lambda x: x[1])
    with open(os.path.join(args.models_dir, "win_rate.json"), "w", encoding="utf-8") as f:
        json.dump(resultats_dct_trie, f, indent=4)
    return None


if __name__ == "__main__":
    freeze_support()
    args = parse_arguments()
    entry(args=args)