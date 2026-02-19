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
from  crusaders.wavelet._wavelet_next_step import main as wavelet_main


def main(args):
    configuration = Namespace(
        ticker=args.ticker, col=args.col,
        older_dataset=args.older_dataset,
        keep_last_step=args.keep_last_step,
        percentage=(1,2,3,4,5),
        vix_modulation=None,
        lower_performance=(0.6767,0.7583,0.8350,0.8717,0.8950),
        upper_performance=(0.5033,0.6300,0.7233,0.8117,0.8717),
        n_forecast_length=1,
        n_forecast_length_in_training=6,
        n_models_to_keep=330,
        upper_multiplier=(1.01,1.02,1.03,1.04,1.05),
        lower_multiplier=(0.99,0.98,0.97,0.96,0.95),
        step_type="month",
        verbose=args.verbose,
        put_side=True,
        call_side=True,
        which_output_to_use="last_of_mean_forecast",
    )
    wavelet_main(configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument('--keep_last_step', type=str2bool, default=True)
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)