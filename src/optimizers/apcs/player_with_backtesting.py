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
from optimizers.apcs.realtime_and_backtest_hyperparameter_search_optuna import entry as apcs_entry_point
from utils import get_next_step
class NoMoreDataException(Exception):
    """Exception pour interrompre instantanément toutes les boucles imbriquées."""
    pass


def entry():
    # --- Configuration d'argparse pour la saisie utilisateur ---
    parser = argparse.ArgumentParser(description="Compilation de modèles avec backtesting.")
    parser.add_argument(
        "--models_dir",
        type=str,
        default=r".\models",
        help="Chemin vers le dossier contenant les modèles .pkl (par défaut: .\\models)"
    )
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

    try:
        for clip_n in range(0, 999999):
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if not file.endswith('.pkl'):
                        continue
                    target_file = os.path.join(root, file)
                    model_name = file  # Nom du fichier pour le suivi individuel

                    config = Namespace(realtime=True, use_realtime_data=False, clip_n=clip_n, model_file=target_file, verbose=False)
                    live_result = apcs_entry_point(config)['local_results']

                    if live_result['reason'] is None:
                        # print(live_result)
                        test_wr = live_result['model_info']['test_wr']
                        close_col = live_result['close_col']
                        lookahead = live_result['model_info']['lookahead']
                        dataset_id = live_result['model_info']['dataset_id']
                        df_realtime = live_result['df_realtime']
                        df_realtime_not_clipped = live_result['df_realtime_not_clipped']
                        density = live_result['model_info']['test_den']
                        bar_on_which_signal_was_triggered = df_realtime.index[-1]
                        bar_on_which_credit_spread_expired = get_next_step(the_date=bar_on_which_signal_was_triggered, dataset_id=dataset_id, nn=lookahead)

                        while True:
                            try:
                                values_of_bar_on_which_credit_spread_expired = df_realtime_not_clipped.loc[bar_on_which_credit_spread_expired]
                                break
                            except KeyError:
                                bar_on_which_credit_spread_expired = get_next_step(the_date=bar_on_which_credit_spread_expired, dataset_id=dataset_id, nn=lookahead)

                        entry_price = live_result['Price']
                        price_at_expiration = values_of_bar_on_which_credit_spread_expired[close_col]
                        is_success = entry_price < price_at_expiration if live_result['Signal'] == "BUY" else entry_price > price_at_expiration

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

                        dual_print(f"{live_result['Signal']} ({live_result['type_option']}) at Entry: {entry_price:.2f} on {live_result['Date'].strftime('%Y-%m-%d')} "
                              f"| Price at Expiration: {price_at_expiration:.2f} ({bar_on_which_credit_spread_expired.strftime('%Y-%m-%d')}) "
                              f"| {'Success' if is_success else 'Failure'} "
                              f"| {test_wr:.2%} probability of success | Density: {density:.2%}")

                    elif 'no more data' in live_result['reason']:
                        raise NoMoreDataException()
    except NoMoreDataException:
        dual_print("\n[INFO] Fin prématurée détectée ('no more data'). Génération des statistiques...")

    # --- Affichage des Statistiques Finales ---
    dual_print("\n" + "=" * 50)
    dual_print(" STATISTIQUES FINALES DE COMPILATION ".center(50, "="))
    dual_print("=" * 50)

    # 1. Statistiques Globales
    g_success = compilation["global"]["success"]
    g_failure = compilation["global"]["failure"]
    g_total = g_success + g_failure

    if g_total > 0:
        g_wr = (g_success / g_total) * 100
        dual_print(f"\n🌍 STATS GLOBALES :")
        dual_print(f"  • Total d'essais     : {g_total}")
        dual_print(f"  • Succès (Win)       : {g_success}")
        dual_print(f"  • Échecs (Loss)      : {g_failure}")
        dual_print(f"  • Taux de réussite   : {g_wr:.2f}%")
    else:
        dual_print("\n🌍 STATS GLOBALES : Aucun trade simulé")

    # 2. Statistiques par Modèle (.pkl)
    if compilation["by_model"]:
        dual_print(f"\n📊 STATS PAR MODÈLE :")
        for m_name, m_stats in compilation["by_model"].items():
            m_total = m_stats["success"] + m_stats["failure"]
            m_wr = (m_stats["success"] / m_total) * 100 if m_total > 0 else 0
            dual_print(f"  • {m_name:<30} -> Total: {m_total:<4} | Win: {m_stats['success']:<4} | Loss: {m_stats['failure']:<4} | WR: {m_wr:.2f}%")

    dual_print("=" * 50)


if __name__ == "__main__":
    entry()
