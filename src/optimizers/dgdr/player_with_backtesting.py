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
import argparse
import pathlib
from argparse import Namespace
import os
from datetime import datetime
from optimizers.dgdr.realtime_and_backtest_hyperparameter_search_optuna import entry as dgdr_entry_point
from utils import get_next_step, factory_load_data
from tqdm import tqdm
from pathlib import Path
class NoMoreDataException(Exception):
    """Exception pour interrompre instantanément toutes les boucles imbriquées."""
    pass


def entry():
    # --- Configuration d'argparse pour la saisie utilisateur ---
    parser = argparse.ArgumentParser(description="Compilation de modèles avec backtesting.")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=r".\models",
        help="Chemin vers le dossier contenant les modèles .pkl (par défaut: .\\models)"
    )
    parser.add_argument('--verbose-per-study', action=argparse.BooleanOptionalAction, default=False, help='')
    parser.add_argument('--n-back', type=int, default=365, help='Number of steps (bars) back')
    args = parser.parse_args()

    # Récupération du chemin saisi ou par défaut
    models_dir = args.models_dir

    # Vérification de l'existence du dossier
    if not os.path.isdir(models_dir):
        print(f"[ERREUR] Le dossier spécifié n'existe pas : {models_dir}")
        return

    # --- File Logging Setup ---
    log_filename = f"player_with_backtesting__compilation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def dual_print(message=""):
        """Prints a message to both the console and the log file."""
        print(message)
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")

    # Structure enrichie pour stocker les statistiques globales et par modèle
    compilation = {
        "global": {"success": 0, "failure": 0},
        "by_model": {}  # Permet de voir quel modèle .pkl performe le mieux
    }
    dataset_id = "day"
    ticker = "^GSPC"

    df_not_clipped = factory_load_data(_dataset_id=dataset_id, _ticker=ticker, _args={"clip_n": 0})
    # 1. Statistiques Globales
    t1 = df_not_clipped.index[-1]
    t2 = df_not_clipped.index[-args.n_back]
    dual_print(f"\n🌍 STATS GLOBALES :")
    dual_print(f"  • Dataset     : {dataset_id}")
    dual_print(f"  • Ticker      : {ticker}")
    dual_print(f"  • Dates       : {t1.strftime('%Y-%m-%d_%H%M')} :: {t2.strftime('%Y-%m-%d_%H%M')}")
    try:
        for clip_n in tqdm(range(0, args.n_back), desc="Clips"):
            # Récupération préalable de la liste des fichiers pour connaître la taille totale
            file_list = [os.path.join(root, file) for root, dirs, files in os.walk(models_dir) for file in files if file.endswith('.pkl')]

            # Deuxième barre tqdm imbriquée
            for target_file in tqdm(file_list, desc=f"Clip {clip_n}", leave=False):
                model_name = Path(target_file).stem  # Nom du fichier pour le suivi individuel

                # Traitement du fichier
                config = Namespace(real_time=True, use_realtime_data=False, clip_n=clip_n, model_path=target_file, verbose=False, seed=123)
                live_result = dgdr_entry_point(config)
                if live_result['buy_signal_detected'] or live_result['sell_signal_detected']:
                    assert live_result['ticker'] == ticker
                    assert live_result['dataset_id'] == dataset_id
                    assert (live_result['buy_signal_detected'] and not live_result['sell_signal_detected']) or (not live_result['buy_signal_detected'] and live_result['sell_signal_detected'])
                    signal = "BUY" if live_result['buy_signal_detected'] else "SELL"
                    close_col = live_result['close_col']
                    lookahead = live_result['lookahead']
                    df_realtime = live_result['df_realtime']
                    bar_on_which_signal_was_triggered = df_realtime.index[-1]
                    bar_on_which_credit_spread_expired = get_next_step(the_date=bar_on_which_signal_was_triggered, dataset_id=dataset_id, nn=lookahead)
                    skip_this = False
                    while True:
                        try:
                            values_of_bar_on_which_credit_spread_expired = df_not_clipped.loc[bar_on_which_credit_spread_expired]
                            break
                        except KeyError:
                            bar_on_which_credit_spread_expired = get_next_step(the_date=bar_on_which_credit_spread_expired, dataset_id=dataset_id, nn=lookahead)
                            now = datetime.now()
                            if bar_on_which_credit_spread_expired > now:
                                skip_this = True
                                break  # We don't have this data yet :)
                    if skip_this:
                        continue
                    price_at_expiration = values_of_bar_on_which_credit_spread_expired[close_col]
                    entry_price = live_result['current_price']
                    assert signal in ["BUY", "SELL"]
                    put_strike_pct, call_strike_pct = live_result['put_strike_pct'], live_result['call_strike_pct']
                    is_success = (entry_price * put_strike_pct) < price_at_expiration if signal== "BUY" else (entry_price * call_strike_pct) > price_at_expiration

                    # Initialisation des stats pour ce modèle spécifique si premier passage
                    if model_name not in compilation["by_model"]:
                        compilation["by_model"][model_name] = {"success": 0, "failure": 0}

                    # Enregistrement du résultat
                    if is_success:
                        compilation["global"]["success"] += 1
                        compilation["by_model"][model_name]["success"] += 1
                    else:
                        compilation["global"]["failure"] += 1
                        compilation["by_model"][model_name]["failure"] += 1

                    entry_date = live_result['current_date']
                    target_date = live_result['target_date']
                    train_win_rate = live_result['train_win_rate']
                    test_win_rate = live_result['val_win_rate']
                    train_trade_density = live_result['train_trade_density']
                    test_trade_density = live_result['val_trade_density']
                    if args.verbose_per_study:
                        dual_print(f"📊 {signal} @ {entry_price:.2f}$ on {entry_date.strftime('%Y-%m-%d_%H%M')} "
                              f"| Price at Expiration: {price_at_expiration:.2f}$ ({target_date.strftime('%Y-%m-%d_%H%M')}) "
                              f"| {'Success' if is_success else 'Failure'} "
                              f"| {train_win_rate:.2%} probability of success in TRAIN , Density: {train_trade_density:.2%} "
                              f"| {test_win_rate:.2%} probability of success in TEST , Density: {test_trade_density:.2%}")
    except NoMoreDataException:
        dual_print("\n[INFO] Fin prématurée détectée ('no more data'). Génération des statistiques...")

    # --- Affichage des Statistiques Finales ---
    dual_print("\n" + "=" * 50)
    dual_print(" STATISTIQUES FINALES DE COMPILATION ".center(50, "="))
    dual_print("=" * 50)

    # 2. Statistiques par Modèle (.pkl)
    if compilation["by_model"]:
        dual_print(f"\n📊 STATS PAR MODÈLE :")
        for m_name, m_stats in compilation["by_model"].items():
            m_total = m_stats["success"] + m_stats["failure"]
            m_density = float(m_total) / float(args.n_back)
            m_wr = (m_stats["success"] / m_total) * 100 if m_total > 0 else 0
            dual_print(f"  • {m_name:<30} -> Total: {m_total:<4} | Density: {m_density:.2%} | WR: {m_wr:.2f}%")

    dual_print("=" * 50)


if __name__ == "__main__":
    entry()
