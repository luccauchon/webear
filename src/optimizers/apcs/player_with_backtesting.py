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
from pathlib import Path
from argparse import Namespace
import os
from datetime import datetime
from optimizers.apcs.realtime_and_backtest_hyperparameter_search_optuna import entry as apcs_entry_point
from utils import get_next_step, factory_load_data
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
        for clip_n in range(0, args.n_back):
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if not file.endswith('.pkl'):
                        continue
                    target_file = os.path.join(root, file)
                    model_name = file  # Nom du fichier pour le suivi individuel

                    config = Namespace(realtime=True, use_realtime_data=False, clip_n=clip_n, model_file=target_file, verbose=False)
                    live_result = apcs_entry_point(config)['local_results']
                    if live_result['reason'] is None:
                        assert live_result['Signal'] in ["BUY", "SELL"]
                        test_wr = live_result['model_info']['test_wr']
                        close_col = live_result['close_col']
                        open_col = live_result['open_col']
                        lookahead = live_result['model_info']['lookahead']
                        dataset_id = live_result['model_info']['dataset_id']
                        df_realtime = live_result['df_realtime']
                        df_realtime_not_clipped = live_result['df_realtime_not_clipped']
                        density = live_result['model_info']['test_den']
                        bar_on_which_signal_was_triggered = df_realtime.index[-1]
                        bar_on_which_credit_spread_expired = get_next_step(the_date=bar_on_which_signal_was_triggered, dataset_id=dataset_id, nn=lookahead)
                        bar_on_which_entry_is_made = get_next_step(the_date=bar_on_which_signal_was_triggered, dataset_id=dataset_id, nn=1)
                        while True:
                            try:
                                values_of_bar_on_which_credit_spread_expired = df_realtime_not_clipped.loc[bar_on_which_credit_spread_expired]
                                break
                            except KeyError:
                                bar_on_which_credit_spread_expired = get_next_step(the_date=bar_on_which_credit_spread_expired, dataset_id=dataset_id, nn=lookahead)
                        while True:
                            try:
                                values_of_bar_on_which_entry_is_made = df_realtime_not_clipped.loc[bar_on_which_entry_is_made]
                                break
                            except KeyError:
                                values_of_bar_on_which_entry_is_made = None
                                break
                        assert live_result['Entry_Execution'] == 'NextOpen'
                        if values_of_bar_on_which_entry_is_made is None:
                            continue
                        entry_price = values_of_bar_on_which_entry_is_made[open_col]*live_result['model_info']['buy_offset'] if live_result['Signal']=="BUY" else values_of_bar_on_which_entry_is_made[open_col]*live_result['model_info']['sell_offset']
                        entry_date = values_of_bar_on_which_entry_is_made.name
                        exit_price = values_of_bar_on_which_credit_spread_expired[close_col]
                        exit_date = values_of_bar_on_which_credit_spread_expired.name
                        is_success = entry_price < exit_price if live_result['Signal'] == "BUY" else entry_price > exit_price

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
                        if args.verbose_per_study:
                            assert live_result['Date'] == bar_on_which_signal_was_triggered
                            assert exit_date == bar_on_which_credit_spread_expired
                            dual_print(f"[{Path(file).stem}] | {live_result['Signal']} ({live_result['type_option']}) | Triggered: {live_result['Date'].strftime('%Y-%m-%d_%H%M')} | Entry: {entry_price:.2f} ({entry_date.strftime('%Y-%m-%d_%H%M')}) "
                                  f"| Exit: {exit_price:.2f} ({bar_on_which_credit_spread_expired.strftime('%Y-%m-%d_%H%M')}) "
                                  f"| {'Success' if is_success else 'Failure'} "
                                  f"| twr: {test_wr:.2%} | Density: {density:.2%}")
                    elif 'no more data' in live_result['reason']:
                        raise NoMoreDataException()
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
