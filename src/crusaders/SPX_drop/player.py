try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from datetime import datetime
import os
import argparse
import pathlib
from argparse import Namespace
import traceback
from utils import send_html_email, get_and_clean_stub_dir
from crusaders.SPX_drop.SPX_drop_anticipation_model_2 import entry as entry_model_2
from crusaders.SPX_drop.SPX_drop_anticipation_model_3 import run_once as entry_model_3
from crusaders.SPX_drop.SPX_drop_anticipation_model_4 import run_live_scan as entry_model_4
from constants import TITLE_WEBEAR, GET_EMAILS
from fetchers.download_list_of_spx500_html import entry as download_list_of_spx500
from fetchers.sp500_download import entry as sp500_download


def parse_args():
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )
    parser.add_argument("--production-setup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--update-dataset", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def entry(args_player):
    destinataires = GET_EMAILS() if args_player.production_setup else GET_EMAILS(dev=True)
    signature_for_prod = TITLE_WEBEAR if args_player.production_setup else "DEV"

    # Update dataset
    if args.update_dataset:
        output_file = os.path.join(get_and_clean_stub_dir("spx_drop_working_dir"), "List of S&P 500 companies - Wikipedia.html")
        download_list_of_spx500(args=Namespace(output_file=output_file))
        sp500_download(args=Namespace(html_file=output_file))

    # Compute models
    result_model_2 = entry_model_2()
    result_model_3 = entry_model_3()
    result_model_4 = entry_model_4()

    # Extraction propre des données pour le courriel
    score_2, reco_2 = result_model_2['score'], result_model_2['recommendation']
    score_3, reco_3 = result_model_3['score'], result_model_3['recommendation']
    score_4, reco_4 = result_model_4['score'], result_model_4['recommendation']

    # Calcul d'un score moyen pour la synthèse (ajustez la pondération si nécessaire)
    # Note: Le modèle 2 est sur 8, les modèles 3 et 4 sont sur 10. On normalise sur 10.
    avg_score = ((score_2 / 8 * 10) + score_3 + score_4) / 3

    # Détermination du message de synthèse global
    if avg_score <= 3.5:
        synthese_globale = "🟢 Environnement favorable. Les indicateurs de risque sont faibles. La poursuite des stratégies standards est justifiée."
    elif avg_score <= 6.5:
        synthese_globale = "🟡 Prudence requise. Des divergences ou des tensions macroéconomiques sont détectées. Il est recommandé de réduire la taille des positions de 30 à 50 %."
    else:
        synthese_globale = "🔴 Alerte de risque élevée. La probabilité d'une correction est significative. Évitez toute nouvelle prise de risque (ex: vente de credit put spreads) et privilégiez la liquidité."

    # Construction du corps du courriel
    email_body = f"""Bonjour,

Voici le rapport quotidien du Bouclier SPX, notre système de détection précoce des risques de correction sur l'indice S&P 500.

📊 RÉSULTATS DES MODÈLES
--------------------------------------------------
• Modèle 2 (Régime de marché)   : Score {score_2}/8  | {reco_2.strip('| ')}
• Modèle 3 (Scanner technique)   : Score {score_3}/10 | {reco_3}
• Modèle 4 (Validation historique): Score {score_4}/10 | {reco_4}

💡 SYNTHÈSE GLOBALE
--------------------------------------------------
{synthese_globale}

🛡️ RAPPEL DE NOTRE APPROCHE
--------------------------------------------------
Conformément à notre mandat de préservation du capital, ce système ne vise pas à prédire le marché, mais à filtrer les environnements à haut risque. En cas de score moyen ou élevé, l'abstention ou la réduction drastique de l'exposition est la stratégie la plus rationnelle.

---
Ce rapport est généré automatiquement par le système {signature_for_prod}. 
Il est fourni à titre informatif pour la gestion des risques et ne constitue pas un conseil financier personnalisé.

Cordialement,
Le système
"""

    subject = f"[{signature_for_prod} @{datetime.now().strftime('%Y%m%d_%H%M')}] | Le Bouclier SPX | {synthese_globale}"

    print("\n" + "="*60)
    print("APERÇU DU COURRIEL GÉNÉRÉ :")
    print("="*60)
    print(f"Objet : {subject}")
    print(email_body)
    print("="*60)

    send_html_email(destinataires=destinataires, sujet=subject, corps=email_body)

    return {'subject': subject, 'body': email_body}


if __name__ == "__main__":
    args = parse_args()
    entry(args_player=args)