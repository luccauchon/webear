# Local custom modules
try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from multiprocessing import freeze_support
from argparse import Namespace
import numpy as np
import argparse
from utils import DATASET_AVAILABLE, str2bool
from runners.VIX_realtime_and_backtest import main as VVIX_realtime_and_backtest
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")


def main(args):
    if args.verbose:
        # --- Nicely print the arguments ---
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)
    configuration = Namespace(
        dataset_id="day", col=args.col, ticker=args.ticker,
        look_ahead=args.look_ahead,
        put=True,
        call=True,
        iron_condor=False,
        step_back_range=args.step_back_range,
        ##
        use_directional_var=True,
        use_directional_var__vix3m=False,
        ##
        upper_side_scale_factor=1.,
        lower_side_scale_factor=1.,

        ## EMA
        adj_call__ema=False,
        adj_call__ema_factor=1.,
        adj_put__ema=False,
        adj_put__ema_factor=1.,
        ema_short=21,
        ema_long=50,

        ## SMA
        adj_call__sma=False,
        adj_put__sma=False,
        sma_period=50,
        adj_put__sma_factor=1.,
        adj_call__sma_factor=1.,

        ## RSI
        adj_call__rsi=False,
        adj_put__rsi=False,
        rsi_period=14,
        adj_call__rsi_factor=1.,
        adj_put__rsi_factor=1.,

        ## MACD
        adj_call__macd=True,
        adj_put__macd=True,
        macd_fast_period=12,
        macd_slow_period=26,
        macd_signal_period=9,
        adj_call__macd_factor=1.,
        adj_put__macd_factor=1.,

        ##
        adj_call_and_put__contango=True,
        verbose=True, verbose_lower_vix=False)
    results = VVIX_realtime_and_backtest(configuration)
    print(results)

if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--ticker', type=str, default='^GSPC')
    parser.add_argument('--col', type=str, default='Close',
                        choices=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    parser.add_argument('--look_ahead', type=int, default=1)
    parser.add_argument('--step_back_range', type=int, default=99999)
    parser.add_argument('--verbose', type=str2bool, default=True)

    args = parser.parse_args()
    main(args)