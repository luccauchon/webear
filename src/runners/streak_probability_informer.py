try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
from argparse import Namespace
from graphics.plotly_close_oriented import main as plotly_close_oriented
from utils import send_html_email
from pathlib import Path
from constants import GET_EMAILS, TITLE_WEBEAR


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument('--dataset-id', type=str, default='day', help='Dataset identifier')
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    parser.add_argument('--realtime', action='store_true', default=False, help='')
    parser.add_argument("--production-setup", action=argparse.BooleanOptionalAction, default=False)
    return parser


def entry(args):
    configuration = Namespace(dataset_id=args.dataset_id, ticker=args.ticker, limit=12, generate_image=True, realtime=args.realtime, generate_html=False)
    result = plotly_close_oriented(configuration)
    output_file = Path(result['output_filename'])
    target_date = result['target_date']
    destinataires = GET_EMAILS() if args.production_setup else GET_EMAILS(dev=True)
    subject=f"{TITLE_WEBEAR} | Analyse statistique des séries | {args.ticker} | {args.dataset_id} | {target_date.strftime('%Y-%m-%d %H:%M')}"
    string_generated = "Bonjour,\n"
    tt1 = "jours" if args.dataset_id in ["day"] else ("semaines"if args.dataset_id in ["week"] else ("mois" if args.dataset_id in ["month"] else "?"))
    string_generated += (f"Quand le prix d'une action baisse (ou monte) plusieurs {tt1} de suite, "
                         "quelles sont les chances mathématiques qu'elle continue dans le même sens le lendemain ?\n")
    tt1 = "la journée" if args.dataset_id in ["day"] else ("la semaine" if args.dataset_id in ["week"] else ("le mois" if args.dataset_id in ["month"] else "?"))
    string_generated += (f"🟢 Une chandelle verte signifie que le prix a fini {tt1} plus haut que la clôture précédente.\n"
                         f"🔴 Une chandelle rouge signifie que le prix a fini {tt1} plus bas que la clôture précédente.\n")
    tt1 = "le jour" if args.dataset_id in ["day"] else ("la semaine" if args.dataset_id in ["week"] else ("le mois" if args.dataset_id in ["month"] else "?"))
    string_generated += (f"Le graphique prédit {tt1} d'après : il calcule le pourcentage de chances que la chandelle suivante soit encore rouge (la baisse continue) "
                         "ou qu'elle tourne au vert (le prix rebondit).")
    send_html_email(destinataires=destinataires, sujet=subject, corps=string_generated, pieces_jointes=[output_file], cc="luccauchon@gmail.com", cci="luccauchon@gmail.com")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)