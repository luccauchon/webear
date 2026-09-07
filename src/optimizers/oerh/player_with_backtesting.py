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
import math
from argparse import Namespace
import os
from datetime import datetime
from optimizers.oerh.realtime_and_backtest_hyperparameter_search_optuna import entry as oerh_entry_point
from utils import get_next_step
from fetchers.data_factory import factory_load_data
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
    parser.add_argument('--dual-print', action=argparse.BooleanOptionalAction, default=False, help='')
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
        if args.dual_print:
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(str(message) + "\n")

    # Structure enrichie pour stocker les statistiques globales et par modèle
    compilation = {
        "global": {"success": 0, "failure": 0},
        "by_model": {}  # Permet de voir quel modèle .pkl performe le mieux
    }
    dataset_id = "day"
    ticker = "^GSPC"
    close_col = ('Close', ticker)
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
                config = Namespace(realtime=True, use_realtime_data=False, clip_n=clip_n, model_path=target_file, return_values_as_dict=True)
                live_result = oerh_entry_point(config)

                # Cas: Long + any half
                if 1 == live_result['signal']:
                    assert 'long_accuracy::any_half_B' == live_result['metric_target_type']
                    lookahead_bars = live_result['lookahead_bars']
                    if clip_n < lookahead_bars:
                        if live_result['current_date'] not in df_not_clipped.index or get_next_step(the_date=live_result['current_date'], dataset_id=live_result['dataset_id'], nn=lookahead_bars) not in df_not_clipped.index:
                            continue  # In the future.
                    entry_price    = live_result['current_price'] * (1+live_result['threshold_pct'])
                    target_price   = live_result['target_price']
                    assert math.isclose(entry_price, target_price, abs_tol=0.1)
                    is_success = False
                    for ppp in range(lookahead_bars//2, lookahead_bars+1):
                        a_date = get_next_step(the_date=live_result['current_date'], dataset_id=live_result['dataset_id'], nn=ppp)
                        try:
                            if entry_price <= df_not_clipped.loc[a_date][close_col]:
                                is_success = True
                                break
                        except:
                            pass
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
