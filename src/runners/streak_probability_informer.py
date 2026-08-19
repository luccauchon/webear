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
from constants import GET_EMAILS


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument('--dataset-id', type=str, default='day', help='Dataset identifier')
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    parser.add_argument('--realtime', action='store_true', default=False, help='')
    return parser


def entry(args):
    configuration = Namespace(dataset_id=args.dataset_id, ticker=args.ticker, limit=8, generate_image=True, realtime=args.realtime, generate_html=False)
    result = plotly_close_oriented(configuration)
    output_file = Path(result['output_filename'])
    target_date = result['target_date']
    destinataires = GET_EMAILS()
    subject=f"Probabilité pour {target_date.strftime('%Y-%m-%d')}"
    string_generated=""
    send_html_email(destinataires=destinataires, sujet=subject, corps=string_generated, pieces_jointes=[output_file])


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)