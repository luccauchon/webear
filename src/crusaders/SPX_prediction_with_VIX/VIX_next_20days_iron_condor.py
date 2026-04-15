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
from  runners.VIX_realtime_and_backtest import main as VIX_realtime_and_backtest

# ================================================================================
#
# 🏆 Optimization Finished!
#
BEST_SCORE = 0.886424
TYPE_OF = "IRON_CONDOR"
# Best Parameters:
#     ema_short............................... 12
#     ema_long................................ 200
#     sma_period.............................. 24
#     rsi_period.............................. 5
#     macd_fast_period........................ 5
#     macd_slow_period........................ 29
#     macd_signal_period...................... 5
#
# ================================================================================

def main(args):
    configuration = Namespace(
        ticker='^GSPC', col='Close', look_ahead=20, dataset_id='day',
        step_back_range=0,
        adj_call__ema=True, adj_put__ema=True, ema_short=12, ema_long=200, adj_call__ema_factor=1.01, adj_put__ema_factor=0.99,
        adj_call__rsi=True, adj_put__rsi=True, rsi_period=5, adj_call__rsi_factor=1.01, adj_put__rsi_factor=0.99,
        adj_call__macd=True, adj_put__macd=True, macd_fast_period=5, macd_slow_period=29, macd_signal_period=5, adj_call__macd_factor=1.01, adj_put__macd_factor=0.99,
        verbose_arguments=False, verbose=False, verbose_results=True,
        put=False, call=False, iron_condor=True,
        use_directional_var=True,
        use_directional_var__vix3m=False,
        upper_side_scale_factor=1., lower_side_scale_factor=1.,
        adj_balanced=False, sharpen_x_side_scale_factor=False,
        adj_call__sma=True, adj_call__sma_factor=1.01, adj_put__sma=True,  adj_put__sma_factor=0.99, sma_period=24,
        adj_call_and_put__contango=True, adj_call_and_put__contango_factor=0.01
    )
    print("*" * 80, flush=True)
    print(f" Best Score ({TYPE_OF}): {BEST_SCORE*100:.1f}%")
    VIX_realtime_and_backtest(configuration)
    print("*" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    args = parser.parse_args()
    main(args)