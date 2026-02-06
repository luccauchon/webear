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
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool, next_week, next_day, get_next_week_range
from  crusaders.wavelet._wavelet_next_week import main as wavelet_main


def main(args):
    configuration = Namespace(
        ticker=args.ticker, col=args.col,
        older_dataset=args.older_dataset,
        keep_last_step=args.keep_last_step,
        percentage=(1,2,3,4),
        lower_performance=(0.7392,0.8483,0.9150,0.9492),
        upper_performance=(0.6333,0.8150,0.9067,0.9533),
        n_forecast_length=1,
        n_forecast_length_in_training=4,
        n_models_to_keep=60,
        upper_multiplier=(1.01,1.02,1.03,1.04),
        lower_multiplier=(0.99,0.98,0.97,0.96),
        verbose=args.verbose,
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