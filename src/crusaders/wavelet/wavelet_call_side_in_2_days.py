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

# 2026.02.17 CASIR
#
# export ENV_EXEC_CODE__WEBEAR=2
#
# (sam_3_box) Apptainer> PYTHONIOENCODING=UTF-8 python wavelet_backtest.py --n_forecast_length_in_training=38 --dataset_id=day --n_forecast_length=2 --warrior_pred_scale_factor=0.0225 --step-back-range=5000 --backtest_strategy=warrior --n_models_to_keep=96 --warrior_spread=call --warrior_gt_range_for_success=0.1 --verbose --use_vix
#
# Warrior Strategy Success Rate: 91.86% (4593/5000)
#
# [VIX < 10] Warrior Strategy Success Rate: 100.0% (63/63)
#
# [VIX < 12] Warrior Strategy Success Rate: 100.0% (440/440)
#
# [VIX < 15] Warrior Strategy Success Rate: 99.29% (1687/1699)
#
# [VIX < 18] Warrior Strategy Success Rate: 98.16% (2714/2765)
#
# [VIX < 20] Warrior Strategy Success Rate: 96.9% (3162/3263)
#
# [VIX < 22] Warrior Strategy Success Rate: 96.45% (3534/3664)
#
# [VIX < 25] Warrior Strategy Success Rate: 95.64% (3952/4132)
#
# [VIX < 28] Warrior Strategy Success Rate: 94.93% (4211/4436)
#
# [VIX < 30] Warrior Strategy Success Rate: 94.44% (4300/4553)
#
# [VIX < 32] Warrior Strategy Success Rate: 93.96% (4372/4653)
#
# [VIX < 35] Warrior Strategy Success Rate: 93.66% (4446/4747)
#
# [VIX < 38] Warrior Strategy Success Rate: 93.33% (4478/4798)
#
# [VIX < 40] Warrior Strategy Success Rate: 93.22% (4494/4821)
#
# [VIX < 42] Warrior Strategy Success Rate: 93.01% (4510/4849)
#
# [VIX < 45] Warrior Strategy Success Rate: 92.7% (4531/4888)
#
# [VIX < 48] Warrior Strategy Success Rate: 92.48% (4547/4917)
#
# [VIX < 50] Warrior Strategy Success Rate: 92.45% (4553/4925)
#
# [VIX < 52] Warrior Strategy Success Rate: 92.43% (4556/4929)
#
# [VIX < 55] Warrior Strategy Success Rate: 92.34% (4566/4945)
#
# [VIX < 58] Warrior Strategy Success Rate: 92.31% (4575/4956)
#
# [VIX < 60] Warrior Strategy Success Rate: 92.24% (4577/4962)
#
# [VIX < 200] Warrior Strategy Success Rate: 91.86% (4593/5000)

def main(args):
    configuration = Namespace(
        ticker=args.ticker, col=args.col,
        older_dataset=args.older_dataset,
        keep_last_step=args.keep_last_step,
        percentage=None,
        vix_modulation=({10: 1., 12: 1., 15: 0.9929, 18: 0.9816, 20: 0.9690, 22: 0.9645, 25: 0.9564, 28: 0.9493, 30: 0.9444, 32: 0.9396, 35: 0.9366, 38: 0.9333, 40: 0.9322, 42: 0.9301, 45: 0.9270, 48: 0.9248, 50: 0.9245, 52: 0.9243, 55: 0.9234, 58: 0.9231, 60: 0.9224, 200: 0.9186}),
        lower_performance=None,
        upper_performance=None,
        n_forecast_length=2,
        n_forecast_length_in_training=38,
        n_models_to_keep=96,
        upper_multiplier=(1.0225,),
        lower_multiplier=None,
        step_type="day",
        verbose=args.verbose,
        put_side=False,
        call_side=True,
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