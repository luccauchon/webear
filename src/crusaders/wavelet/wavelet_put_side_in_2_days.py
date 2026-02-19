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


# 2026.02.16
#
# (PY311) PS C:\PYCHARMPROJECTS\webear\src\runners> python .\wavelet_backtest.py --n_forecast_length_in_training=38 --dataset_id=day --n_forecast_length=2 --warrior_pred_scale_factor=-0.0225 --step-back-range=10000 --backtest_strategy=warrior --n_models_to_keep=96 --warrior_spread=put --warrior_gt_range_for_success=0.1 --verbose --use_vix
#
# Warrior Strategy Success Rate: 83.34% (7579/9094)
#
# [VIX < 10] Warrior Strategy Success Rate: 100.0% (68/68)
#
# [VIX < 12] Warrior Strategy Success Rate: 96.13% (770/801)
#
# [VIX < 15] Warrior Strategy Success Rate: 93.42% (2753/2947)
#
# [VIX < 18] Warrior Strategy Success Rate: 90.8% (4332/4771)
#
# [VIX < 20] Warrior Strategy Success Rate: 89.25% (5088/5701)
#
# [VIX < 22] Warrior Strategy Success Rate: 87.78% (5746/6546)
#
# [VIX < 25] Warrior Strategy Success Rate: 86.37% (6484/7507)
#
# [VIX < 28] Warrior Strategy Success Rate: 85.19% (6891/8089)
#
# [VIX < 30] Warrior Strategy Success Rate: 84.8% (7089/8360)
#
# [VIX < 32] Warrior Strategy Success Rate: 84.46% (7239/8571)
#
# [VIX < 35] Warrior Strategy Success Rate: 84.22% (7362/8741)
#
# [VIX < 38] Warrior Strategy Success Rate: 84.06% (7433/8842)
#
# [VIX < 40] Warrior Strategy Success Rate: 83.94% (7459/8886)
#
# [VIX < 42] Warrior Strategy Success Rate: 83.85% (7485/8927)
#
# [VIX < 45] Warrior Strategy Success Rate: 83.75% (7520/8979)
#
# [VIX < 48] Warrior Strategy Success Rate: 83.68% (7540/9011)
#
# [VIX < 50] Warrior Strategy Success Rate: 83.65% (7544/9019)
#
# [VIX < 52] Warrior Strategy Success Rate: 83.63% (7546/9023)
#
# [VIX < 55] Warrior Strategy Success Rate: 83.54% (7551/9039)
#
# [VIX < 58] Warrior Strategy Success Rate: 83.49% (7556/9050)
#
# [VIX < 60] Warrior Strategy Success Rate: 83.48% (7560/9056)
#
# [VIX < 200] Warrior Strategy Success Rate: 83.34% (7579/9094)

def main(args):
    configuration = Namespace(
        ticker=args.ticker, col=args.col,
        older_dataset=args.older_dataset,
        keep_last_step=args.keep_last_step,
        percentage=None,
        vix_modulation=({10: 1., 12: 0.9613, 15: 0.9342, 18: 0.9080, 20: 0.8925, 22: 0.8778, 25: 0.8637, 28: 0.8519, 30: 0.8480, 32: 0.8446, 35: 0.8422, 38: 0.8406, 40: 0.8394, 42: 0.8385, 45: 0.8375, 48: 0.8368, 50: 0.8365, 52: 0.8363, 55: 0.8354, 58: 0.8349, 60: 0.8348, 200: 0.8334}),
        lower_performance=None,
        upper_performance=None,
        n_forecast_length=2,
        n_forecast_length_in_training=38,
        n_models_to_keep=96,
        upper_multiplier=None,
        lower_multiplier=(0.9775,),
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