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
from constants import TITLE_WEBEAR, GET_EMAILS
from runners.conditional_gap_persistence import calculate_sp500_probabilities as entry_conditional_gap_persistence


def parse_args():
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )
    parser.add_argument("--production-setup", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def entry(args_player):
    destinataires = GET_EMAILS() if args_player.production_setup else GET_EMAILS(dev=True)
    signature_for_prod = TITLE_WEBEAR if args_player.production_setup else "DEV"

    result = entry_conditional_gap_persistence(args=Namespace(dataset_id="day",ticker="^GSPC",epsilon=0,use_realtime_data=False,display_all=False))
    result = result['string_generated']
    subject = f"[{signature_for_prod} @{datetime.now().strftime('%Y%m%d_%H%M')}] | Persistance conditionnelle de l'écart | "
    # Construction du corps du courriel
    email_body = f"{result}"

    send_html_email(destinataires=destinataires, sujet=subject, corps=email_body)

    return {'subject': subject, 'body': email_body}


if __name__ == "__main__":
    args = parse_args()
    entry(args_player=args)