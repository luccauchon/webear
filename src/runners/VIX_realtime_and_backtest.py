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
import pickle
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool, next_weekday
import copy
import numpy as np
from tqdm import tqdm
from runners.MMI_realtime import main as MMI_realtime
from argparse import Namespace
import pandas as pd


def main(args):
    if args.verbose:
        # --- Nicely print the arguments ---
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            if 'master_data_cache' in arg:
                print(f"    {arg:.<40} {value.index[0].strftime('%Y-%m-%d')} to {value.index[-1].strftime('%Y-%m-%d')} ({args.one_dataset_filename})")
                continue
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)
    assert args.look_ahead >= 1.
    assert args.dataset_id in ['day']
    one_dataset_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    close_col = (args.col, args.ticker)
    vix__master_data_cache = copy.deepcopy(master_data_cache['^VIX'].sort_index()[('Close', '^VIX')])
    vix1d__master_data_cache = copy.deepcopy(master_data_cache['^VIX1D'].sort_index()[('Close', '^VIX1D')])
    vix3m__master_data_cache = copy.deepcopy(master_data_cache['^VIX3M'].sort_index()[('Close', '^VIX3M')])
    master_data_cache = copy.deepcopy(master_data_cache[args.ticker].sort_index())[close_col]
    put_credit = args.put
    call_credit = args.call
    iron_condor = args.iron_condor
    # Adjust VIX
    if vix__master_data_cache.index[-1] != master_data_cache.index[-2]:
        print(f"Removing last element of VIX DF.")
        vix__master_data_cache = vix__master_data_cache.iloc[:-1]
    assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix__master_data_cache.index[-1].strftime('%Y-%m-%d')
    assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix1d__master_data_cache.index[-1].strftime('%Y-%m-%d')
    assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix3m__master_data_cache.index[-1].strftime('%Y-%m-%d')
    lower_side_scale_factor, upper_side_scale_factor = args.lower_side_scale_factor, args.upper_side_scale_factor
    results_realtime, results_backtest, _str_tmp_put_credit, _str_tmp_call_credit, _str_tmp_iron_condor_credit = [], [], '', '', ''
    step_back_range = args.step_back_range if args.step_back_range <len(master_data_cache) else len(master_data_cache)
    used_past_point = 0
    for step_back in tqdm(range(0, step_back_range + 1)) if args.verbose else range(0, step_back_range + 1):
        if 0 == step_back:
            past_df = master_data_cache
            future_df = master_data_cache
            vix_df = vix__master_data_cache
            vix3m_df = vix3m__master_data_cache
        else:
            past_df = master_data_cache.iloc[:-step_back]
            future_df = master_data_cache.iloc[-step_back:]
            vix_df = vix__master_data_cache.iloc[:-step_back]
            vix3m_df = vix3m__master_data_cache.iloc[:-step_back]

        if args.look_ahead > len(future_df) or 0 == len(past_df):
            continue
        if 0 == len(vix_df):
            continue
        if args.use_directional_var:
            if 0 == len(vix3m_df):
                continue

        # print(f"")
        # print(f"MASTER:{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} --> {future_df.index[0].strftime('%Y-%m-%d')}/{future_df.index[-1].strftime('%Y-%m-%d')}")
        # print(f"MASTER:{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} :: {future_df.index[args.look_ahead - 1].strftime('%Y-%m-%d')}")
        # print(f"VIX   : {vix_df.index[0].strftime('%Y-%m-%d')}/{vix_df.index[-1].strftime('%Y-%m-%d')}")
        # print(f"Using {past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} to predict {future_df.index[args.look_ahead - 1].strftime('%Y-%m-%d')}")
        # print(f"\n\n")

        assert vix_df.index[-1].strftime('%Y-%m-%d') == past_df.index[-1].strftime('%Y-%m-%d')
        if args.use_directional_var:
            assert vix3m_df.index[-1].strftime('%Y-%m-%d') == past_df.index[-1].strftime('%Y-%m-%d')
        used_past_point += 1
        current_price = past_df.iloc[-1]
        vix           = vix_df.iloc[-1]
        contango, backwardation = None, None
        trend_bias, momentum_bias, vol_structure = '', '', ''
        # --- DIRECTIONAL VARIABLES ---
        if args.use_directional_var:
            # The Signal:
            # Contango (VIX < VIX3M): Normal market. Usually bullish or neutral.
            # Backwardation (VIX > VIX3M): Fear is immediate. This is a strong Bearish signal. When short-term volatility spikes above long-term, the SPX often drops.
            vix3m         = vix3m_df.iloc[-1]
            contango      = vix <= vix3m
            backwardation = vix > vix3m

            # 1. Trend Filter: 50-Day Simple Moving Average
            sma_50 = past_df.rolling(window=50).mean().iloc[-1]
            trend_bias = 'BULLISH' if current_price > sma_50 else 'BEARISH'

            # 2. Momentum Filter: 14-Day RSI
            delta = past_df.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]

            momentum_bias = 'NEUTRAL'
            if rsi_val > 70:
                momentum_bias = 'OVERBOUGHT (Bearish Risk)'
            elif rsi_val < 30:
                momentum_bias = 'OVERSOLD (Bullish Opportunity)'

            # 3. VIX Term Structure (Requires ^VIX3M in your cache)
            # Assuming you load vix3m_df similar to vix_df
            if args.use_directional_var:
                vix_ratio = vix / vix3m_df.iloc[-1]
                vol_structure = 'BACKWARDATION (Fear)' if vix_ratio > 1.0 else 'CONTANGO (Normal)'

        # VIX indices are annualized standard deviations.
        # To get 1-day expected move: (Index / 100) * sqrt(1 / 252)
        trading_days_per_year = 252.0

        vix_implied_daily = (vix / 100.0) * np.sqrt(1. / trading_days_per_year) * np.sqrt(args.look_ahead)
        upper_limit = upper_side_scale_factor * current_price * (1 + vix_implied_daily)
        lower_limit = lower_side_scale_factor * current_price * (1 - vix_implied_daily)

        if 0 == step_back:
            # Real time
            assert np.all(past_df.index == future_df.index)
            results_realtime.append({
                'date': past_df.index[-1],
                'target_date': next_weekday(input_date=past_df.index[-1], nn=args.look_ahead),
                'current_price': current_price,
                'vix': vix,
                'upper_limit': upper_limit,
                'lower_limit': lower_limit,
                'vix_3m': {'contango': contango, 'backwardation': backwardation},
                'directional_var__trend_bias': trend_bias,
                'directional_var__momentum_bias': momentum_bias,
                'directional_var__vol_structure': vol_structure,
            })
        else:
            # Backtest
            assert 0 == len(past_df.index.intersection(future_df.index))
            target_price = future_df.iloc[args.look_ahead - 1]
            success_iron_condor = True if lower_limit <= target_price <= upper_limit else False
            succes_put_credit = True if lower_limit <= target_price else False
            succes_call_credit = True if target_price <= upper_limit else False
            actual_return = (target_price - current_price) / current_price

            # if args.use_directional_var:
            #     if trend_bias == 'BEARISH':
            #         succes_put_credit = True  # Don't sell put credit in a downtrend
            #     if momentum_bias == 'OVERBOUGHT':
            #         succes_call_credit = True  # Don't sell call credit when market is extended up

            results_backtest.append({
                'date': past_df.index[-1],
                'current_price': current_price,
                'target_price': target_price,
                'actual_return': actual_return,
                'vix': vix,
                'delta_lower_side':target_price - lower_limit,
                'delta_upper_side': upper_limit - target_price,
                'success_iron_condor': success_iron_condor,
                'success_put_credit':  succes_put_credit,
                'success_call_credit':  succes_call_credit,
                'vix_3m': {'contango': contango, 'backwardation': backwardation},
            })
    if len(results_realtime) > 0:
        assert 1 == len(results_realtime)
        if args.verbose:
            current_date = results_realtime[0]['date']
            prediction_date = results_realtime[0]['target_date']
            _str_directional_var = (f'Trend bias:{results_realtime[0]["directional_var__trend_bias"]}  '
                                    f'Momentum bias:{results_realtime[0]["directional_var__momentum_bias"]}  '
                                    f'Vol structure:{results_realtime[0]["directional_var__vol_structure"]}')
            print(f"[{current_date.strftime('%Y-%m-%d')}], the prediction for {prediction_date.strftime('%Y-%m-%d')} is "
                  f"[{results_realtime[0]['lower_limit']:.0f} :: {results_realtime[0]['upper_limit']:.0f}] , {_str_directional_var}")
    if len(results_backtest) > 0:
        print(f"Backtested on {used_past_point} steps.")
        df_results = pd.DataFrame(results_backtest)
        vixes = [999]
        if args.verbose_lower_vix:
            vixes = [10, 12, 15, 18, 20, 30, 40, 50, 999]
        if put_credit:
            for uuu in vixes:
                tmp_df = df_results[df_results['vix'] <= uuu]
                if np.isnan(tmp_df['delta_lower_side'].mean()):
                    continue
                delta_lower_side = tmp_df[tmp_df['delta_lower_side'] > 0]['delta_lower_side']
                delta_lower_side = delta_lower_side.mean() / tmp_df['target_price'].mean() * 100
                if args.verbose:
                    print(f"\t[VIX<={uuu}] ,     PUT : {tmp_df['success_put_credit'].mean()*100:.1f}% success")
        if call_credit:
            for uuu in vixes:
                tmp_df = df_results[df_results['vix'] <= uuu]
                delta_upper_side = tmp_df['delta_upper_side'].mean()
                if np.isnan(delta_upper_side):
                    continue
                if args.verbose:
                    print(f"\t[VIX<={uuu}] ,     CALL : {tmp_df['success_call_credit'].mean()*100:.1f}% success")
        if iron_condor:
            for uuu in vixes:
                tmp_df = df_results[df_results['vix'] <= uuu]
                if 0 == len(tmp_df):
                    continue
                print(f"\t[VIX<={uuu}] IRON CONDOR : {tmp_df['success_iron_condor'].mean()*100:.1f}% success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
    parser.add_argument("--look_ahead", type=int, default=1)
    parser.add_argument("--lower_side_scale_factor", type=float, default=1)
    parser.add_argument("--upper_side_scale_factor", type=float, default=1)
    parser.add_argument('--put', type=str2bool, default=True)
    parser.add_argument('--call', type=str2bool, default=True)
    parser.add_argument('--iron_condor', type=str2bool, default=True)
    parser.add_argument('--step-back-range', type=int, default=5,
                        help="Number of historical time windows to simulate (rolling backtest depth).")
    parser.add_argument('--use_directional_var', type=str2bool, default=False)
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--verbose_lower_vix', type=str2bool, default=False)
    args = parser.parse_args()

    main(args)