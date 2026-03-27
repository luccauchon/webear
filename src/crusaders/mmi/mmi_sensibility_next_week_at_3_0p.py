try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
from argparse import Namespace
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool, get_weekday_range
import copy
import numpy as np
from datetime import datetime, timedelta
from crusaders.mmi.mmi_next import main as MMI_next
from crusaders.mmi.mmi_next_week_at_3_0p import CONFIGURATION_FOR_MMI_NEXT_WEEK, BEST_SCORE
from rich.console import Console
from rich.table import Table
from rich import box

def main(args):
    console = Console()

    config_dict = vars(CONFIGURATION_FOR_MMI_NEXT_WEEK)
    config_dict.update({'ticker': args.ticker, 'col': args.col, 'verbose': False, 'older_dataset': args.older_dataset, 'keep_last_step': args.keep_last_step})

    sensitivities = np.arange(1.004, 0.9968, -0.000675)

    date_prediction_step = MMI_next(Namespace(**config_dict))["date_prediction_step"]

    # Create Table
    tt1, tt2 = get_weekday_range(date_prediction_step)
    table = Table(title=f"Sensitivity {args.ticker} for next week ({tt1}:{tt2})", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Sensitivity", justify="right", style="cyan")
    table.add_column("Old Price", justify="right", style="orange_red1")
    table.add_column("New Price", justify="right", style="orange_red1")
    table.add_column("Change", justify="right")
    table.add_column("Thresholds", justify="center")

    for sensitivity in sensitivities:
        config_dict.update({'modify_closing_price': sensitivity})
        configuration = Namespace(**config_dict)
        result = MMI_next(configuration)

        old_p, new_p = result["var_closing_prices"][0], result["var_closing_prices"][1]
        lower, upper = result['prices_threshold'][0], result['prices_threshold'][1]

        change = new_p - old_p
        if change > 0:
            change_str = f"[green]+{change:.0f}[/green]"
        elif change < 0:
            change_str = f"[red]{change:.0f}[/red]"
        else:
            change_str = f"{change:.0f}"

        is_baseline = np.allclose(1, sensitivity, atol=0.0001)
        style = "bold yellow" if is_baseline else ""
        sens_display = f"★ {sensitivity:.4f}" if is_baseline else f"  {sensitivity:.4f}"

        table.add_row(
            sens_display,
            f"{old_p:.0f}",
            f"{new_p:.0f}",
            change_str,
            f"{lower:.0f}:{upper:.0f}",
            style=style
        )
    print(f"Historical performance of {BEST_SCORE * 100}% (overall accuracy)")
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument('--keep_last_step', type=str2bool, default=True)
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)
