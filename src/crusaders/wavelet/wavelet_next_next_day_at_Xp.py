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
from utils import str2bool
from  crusaders.wavelet._wavelet_next_step import main as wavelet_next_step_main


def main(args):
    configuration = Namespace(
        ticker=args.ticker, col=args.col,
        older_dataset=args.older_dataset,
        keep_last_step=args.keep_last_step,
        percentage=None,
        vix_modulation=({10: 1., 12: 0.9388, 15: 0.9002, 18: 0.8665, 20: 0.8509, 22: 0.8375, 25: 0.8223, 28: 0.8099, 30: 0.8057, 9999: 0.7923}),
        lower_performance=None,
        upper_performance=None,
        n_forecast_length=2,
        n_forecast_length_in_training=4,
        n_models_to_keep=60,
        upper_multiplier=None,
        lower_multiplier=(0.99,),
        step_type="day",
        verbose=args.verbose,
        put_side=True,
        call_side=False,
        which_output_to_use="last_of_mean_forecast",
    )
    wavelet_next_step_main(configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument('--keep_last_step', type=str2bool, default=True)
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)