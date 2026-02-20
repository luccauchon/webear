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
    if vix__master_data_cache.index[-1] != master_data_cache.index[-1]:
        print(f"Removing last element of VIX DF.")
        vix__master_data_cache = vix__master_data_cache.iloc[:-1]
    assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix__master_data_cache.index[-1].strftime('%Y-%m-%d')
    assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix1d__master_data_cache.index[-1].strftime('%Y-%m-%d')
    assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix3m__master_data_cache.index[-1].strftime('%Y-%m-%d')
    results_realtime, results_backtest, _str_tmp_put_credit, _str_tmp_call_credit, _str_tmp_iron_condor_credit = [], [], '', '', ''
    step_back_range = args.step_back_range if args.step_back_range < len(master_data_cache) else len(master_data_cache)
    used_past_point, real_time_date, backstep_t1, backstep_t2 = 0, None, None, None
    returned_results = {}
    for step_back in tqdm(range(0, step_back_range + 1)) if args.verbose else range(0, step_back_range + 1):
        if 0 == step_back:
            past_df = master_data_cache
            future_df = master_data_cache
            vix_df = vix__master_data_cache
            if args.use_directional_var__vix3m and args.use_directional_var:
                vix3m_df = vix3m__master_data_cache
        else:
            past_df = master_data_cache.iloc[:-step_back]
            future_df = master_data_cache.iloc[-step_back:]
            vix_df = vix__master_data_cache.iloc[:-step_back]
            if args.use_directional_var__vix3m and args.use_directional_var:
                vix3m_df = vix3m__master_data_cache.iloc[:-step_back]

        if args.look_ahead > len(future_df) or 0 == len(past_df):
            continue
        if 0 == len(vix_df):
            continue
        if args.use_directional_var__vix3m and args.use_directional_var:
            if 0 == len(vix3m_df):
                continue

        # print(f"")
        # print(f"MASTER:{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} --> {future_df.index[0].strftime('%Y-%m-%d')}/{future_df.index[-1].strftime('%Y-%m-%d')}")
        # print(f"MASTER:{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} :: {future_df.index[args.look_ahead - 1].strftime('%Y-%m-%d')}")
        # print(f"VIX   : {vix_df.index[0].strftime('%Y-%m-%d')}/{vix_df.index[-1].strftime('%Y-%m-%d')}")
        # print(f"Using {past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} to predict {future_df.index[args.look_ahead - 1].strftime('%Y-%m-%d')}")
        # print(f"\n\n")

        assert vix_df.index[-1].strftime('%Y-%m-%d') == past_df.index[-1].strftime('%Y-%m-%d')
        if args.use_directional_var__vix3m and args.use_directional_var:
            assert vix3m_df.index[-1].strftime('%Y-%m-%d') == past_df.index[-1].strftime('%Y-%m-%d')
        used_past_point += 1 if 0 != step_back else 0
        current_price = past_df.iloc[-1]
        vix           = vix_df.iloc[-1]
        contango, backwardation = None, None
        ema_trend_bias, sma_trend_bias, momentum_bias, vol_structure = '', '', '', ''
        macd_bias = ''
        # --- DIRECTIONAL VARIABLES ---
        if args.use_directional_var:
            if args.use_directional_var__vix3m:
                # The Signal:
                # Contango (VIX < VIX3M): Normal market. Usually bullish or neutral.
                # Backwardation (VIX > VIX3M): Fear is immediate. This is a strong Bearish signal. When short-term volatility spikes above long-term, the SPX often drops.
                vix3m         = vix3m_df.iloc[-1]
                contango      = vix <= vix3m
                backwardation = vix > vix3m

            if args.adj_call__sma or args.adj_put__sma:
                # 1. Trend Filter: 50-Day Simple Moving Average
                sma_50 = past_df.rolling(window=args.sma_period).mean().iloc[-1]
                sma_trend_bias = 'BULLISH' if current_price > sma_50 else 'BEARISH'

            if args.adj_call__rsi or args.adj_put__rsi:
                # 2. Momentum Filter: 14-Day RSI
                delta = past_df.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=args.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=args.rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1]

                momentum_bias = 'NEUTRAL'
                if rsi_val > 70:
                    momentum_bias = 'OVERBOUGHT (Bearish Risk)'
                elif rsi_val < 30:
                    momentum_bias = 'OVERSOLD (Bullish Opportunity)'

            if args.use_directional_var__vix3m:
                # 3. VIX Term Structure (Requires ^VIX3M in your cache)
                # Assuming you load vix3m_df similar to vix_df
                vix_ratio = vix / vix3m_df.iloc[-1]
                vol_structure = 'BACKWARDATION (Fear)' if vix_ratio > 1.0 else 'CONTANGO (Normal)'

            if args.adj_call__ema or args.adj_put__ema:
                # 4. --- 🚀 NEW: EMA DIRECTIONAL LOGIC 🚀 ---
                # Calculate EMAs on the PAST data only (no look-ahead bias)
                # adjust=False makes it behave like standard trading view EMA
                ema_short_series = past_df.ewm(span=args.ema_short, adjust=False).mean()
                ema_long_series  = past_df.ewm(span=args.ema_long, adjust=False).mean()

                ema_short = ema_short_series.iloc[-1]
                ema_long  = ema_long_series.iloc[-1]

                # Determine Trend Bias
                ema_trend_bias = "NEUTRAL"
                if not np.isnan(ema_long):
                    if current_price > ema_long and ema_short > ema_long:
                        ema_trend_bias = "BULLISH 🟢"
                    elif current_price < ema_long and ema_short < ema_long:
                        ema_trend_bias = "BEARISH 🔴"
                    elif current_price > ema_long:
                        ema_trend_bias = "WEAK BULL 🟢"
                    elif current_price < ema_long:
                        ema_trend_bias = "WEAK BEAR 🔴"

            if args.adj_call__macd or args.adj_put__macd:
                # 5. --- 🚀 NEW: MACD MOMENTUM LOGIC 🚀 ---
                # Calculate MACD on the PAST data only (no look-ahead bias)
                ema_fast = past_df.ewm(span=args.macd_fast_period, adjust=False).mean()
                ema_slow = past_df.ewm(span=args.macd_slow_period, adjust=False).mean()

                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=args.macd_signal_period, adjust=False).mean()
                macd_histogram = macd_line - signal_line

                macd_line_val = macd_line.iloc[-1]
                signal_line_val = signal_line.iloc[-1]
                macd_histogram_val = macd_histogram.iloc[-1]

                # Determine MACD Bias
                macd_bias = "NEUTRAL"
                if not np.isnan(macd_line_val) and not np.isnan(signal_line_val):
                    if macd_line_val > signal_line_val and macd_histogram_val > 0:
                        macd_bias = "BULLISH 🟢"
                    elif macd_line_val < signal_line_val and macd_histogram_val < 0:
                        macd_bias = "BEARISH 🔴"
                    elif macd_line_val > signal_line_val:
                        macd_bias = "WEAK BULL 🟢"
                    elif macd_line_val < signal_line_val:
                        macd_bias = "WEAK BEAR 🔴"
                    # Check for crossovers
                    if len(macd_line) >= 2:
                        prev_macd = macd_line.iloc[-2]
                        prev_signal = signal_line.iloc[-2]
                        if prev_macd <= prev_signal and macd_line_val > signal_line_val:
                            macd_bias = "BULLISH CROSSOVER 🟢"
                        elif prev_macd >= prev_signal and macd_line_val < signal_line_val:
                            macd_bias = "BEARISH CROSSOVER 🔴"

        # VIX indices are annualized standard deviations.
        # To get 1-day expected move: (Index / 100) * sqrt(1 / 252)
        trading_days_per_year = 252.0
        vix_implied_daily = (vix / 100.0) * np.sqrt(1. / trading_days_per_year) * np.sqrt(args.look_ahead)
        upper_limit = args.upper_side_scale_factor * current_price * (1 + vix_implied_daily)
        lower_limit = args.lower_side_scale_factor * current_price * (1 - vix_implied_daily)

        if args.adj_call__ema:
            assert args.use_directional_var
            if ema_trend_bias in ["WEAK BULL 🟢", "BULLISH 🟢"]:
                assert args.adj_call__ema_factor >= 1.
                upper_limit = upper_limit * args.adj_call__ema_factor
        if args.adj_put__ema:
            assert args.use_directional_var
            if ema_trend_bias in ["WEAK BEAR 🔴", "BEARISH 🔴"]:
                assert args.adj_put__ema_factor <= 1.
                lower_limit = lower_limit * args.adj_put__ema_factor
        if args.adj_call__sma:
            assert args.use_directional_var
            if sma_trend_bias in ["BULLISH"]:
                assert args.adj_call__sma_factor >= 1.
                upper_limit = upper_limit * args.adj_call__sma_factor
        if args.adj_put__sma:
            assert args.use_directional_var
            if sma_trend_bias in ["BEARISH"]:
                assert args.adj_put__sma_factor <= 1.
                lower_limit = lower_limit * args.adj_put__sma_factor
        if args.adj_call__rsi:
            assert args.use_directional_var
            if momentum_bias in ['OVERSOLD (Bullish Opportunity)']:
                assert args.adj_call__rsi_factor >= 1.
                upper_limit = upper_limit * args.adj_call__rsi_factor
        if args.adj_put__rsi:
            assert args.use_directional_var
            if momentum_bias in ['OVERBOUGHT (Bearish Risk)']:
                assert args.adj_put__rsi_factor <= 1.
                lower_limit = lower_limit * args.adj_put__rsi_factor
        # --- NEW: MACD Adjustments ---
        if args.adj_call__macd:
            assert args.use_directional_var
            if macd_bias in ["BULLISH 🟢", "BULLISH CROSSOVER 🟢", "WEAK BULL 🟢"]:
                assert args.adj_call__macd_factor >= 1.
                upper_limit = upper_limit * args.adj_call__macd_factor
        if args.adj_put__macd:
            assert args.use_directional_var
            if macd_bias in ["BEARISH 🔴", "BEARISH CROSSOVER 🔴", "WEAK BEAR 🔴"]:
                assert args.adj_put__macd_factor <= 1.
                lower_limit = lower_limit * args.adj_put__macd_factor
        if args.adj_call_and_put__contango:
            assert args.use_directional_var
            if vol_structure in ['BACKWARDATION (Fear)']:
                assert 0 <= args.adj_call_and_put__contango_factor <= 0.1
                upper_limit = upper_limit * (1 + args.adj_call_and_put__contango_factor)
                lower_limit = lower_limit * (1 - args.adj_call_and_put__contango_factor)

        if 0 == step_back:
            # Real time
            assert np.all(past_df.index == future_df.index)
            real_time_date = next_weekday(input_date=past_df.index[-1], nn=args.look_ahead)
            results_realtime.append({
                'date': past_df.index[-1],
                'target_date': next_weekday(input_date=past_df.index[-1], nn=args.look_ahead),
                'current_price': current_price,
                'vix': vix,
                'upper_limit': upper_limit,
                'lower_limit': lower_limit,
                'vix_3m': {'contango': contango, 'backwardation': backwardation},
                'directional_var__sma_trend_bias': sma_trend_bias,
                'directional_var__ema_trend_bias': ema_trend_bias,
                'directional_var__momentum_bias': momentum_bias,
                'directional_var__vol_structure': vol_structure,
                'directional_var__macd_bias': macd_bias,
            })
        else:
            # Backtest
            assert 0 == len(past_df.index.intersection(future_df.index))
            target_price = future_df.iloc[args.look_ahead - 1]
            success_iron_condor = True if lower_limit  <= target_price <= upper_limit else False
            succes_put_credit   = True if lower_limit  <= target_price else False
            succes_call_credit  = True if target_price <= upper_limit else False
            actual_return = (target_price - current_price) / current_price
            lower_diff = (current_price - lower_limit) / current_price
            assert lower_diff > 0
            upper_diff = (upper_limit - current_price) / current_price
            assert upper_diff > 0
            if 1 == step_back:
                backstep_t1 = past_df.index[-1]
            backstep_t2 = past_df.index[-1]
            results_backtest.append({
                'date': past_df.index[-1],
                'current_price': current_price,
                'target_price': target_price,
                'actual_return': actual_return,
                'vix': vix, 'upper_diff': upper_diff, 'lower_diff': lower_diff,
                'delta_lower_side': target_price - lower_limit,
                'delta_upper_side': upper_limit - target_price,
                'success_iron_condor': success_iron_condor,
                'success_put_credit': succes_put_credit,
                'success_call_credit': succes_call_credit,
                'vix_3m': {'contango': contango, 'backwardation': backwardation},
            })
    if len(results_realtime) > 0:
        assert 1 == len(results_realtime)
        if args.verbose:
            current_date = results_realtime[0]['date']
            prediction_date = results_realtime[0]['target_date']
            _ema_trend = f'EMA trend bias: {results_realtime[0]["directional_var__ema_trend_bias"]}' if args.use_directional_var else ''
            _sma_trend = f'SMA trend bias: {results_realtime[0]["directional_var__sma_trend_bias"]}' if args.use_directional_var else ''
            _rsi       = f'RSI           : {results_realtime[0]["directional_var__momentum_bias"]}' if args.use_directional_var else ''
            _macd      = f'MACD          : {results_realtime[0]["directional_var__macd_bias"]}' if args.use_directional_var else ''
            _vol_struct = f'VIX/VIX3m     : {results_realtime[0]["directional_var__vol_structure"]}' if args.use_directional_var else ''
            _str_directional_var = f'\n\t{_ema_trend}\n\t{_sma_trend}\n\t{_rsi}\n\t{_macd}\n\t{_vol_struct}\n\n'
            print(f"Today the {current_date.strftime('%Y-%m-%d')} (SPX is @{results_realtime[0]['current_price']:.0f}, VIX is @{results_realtime[0]['vix']:.1f}), "
                  f"the prediction for {prediction_date.strftime('%Y-%m-%d')} is "
                  f"[{results_realtime[0]['lower_limit']:.0f} :: {results_realtime[0]['upper_limit']:.0f}] , {_str_directional_var}")
    if len(results_backtest) > 0:
        if args.verbose:
            print(f"Backtested on {used_past_point} steps (from {backstep_t1.strftime('%Y-%m-%d')} to {backstep_t2.strftime('%Y-%m-%d')}), collecting {len(results_backtest)} results. "
                  f"Prediction is for {real_time_date.strftime('%Y-%m-%d')}.")
        df_results = pd.DataFrame(results_backtest)
        vixes = [999]
        if args.verbose_lower_vix:
            vixes = [10, 12, 15, 18, 20, 30, 40, 50, 999]
        if put_credit:
            returned_results.update({'put': {}})
            for uuu in vixes:
                tmp_df = df_results[df_results['vix'] <= uuu]
                if np.isnan(tmp_df['delta_lower_side'].mean()):
                    continue
                delta_lower_side = tmp_df[tmp_df['delta_lower_side'] > 0]['delta_lower_side']
                delta_lower_side = delta_lower_side.mean() / tmp_df['target_price'].mean() * 100
                returned_results['put'].update({f'success_rate__vix{uuu}': tmp_df['success_put_credit'].mean()})
                if args.verbose:
                    print(f"\t[VIX<={uuu}] ,     PUT : {tmp_df['success_put_credit'].mean() * 100:.1f}% success")
        if call_credit:
            returned_results.update({'call': {}})
            for uuu in vixes:
                tmp_df = df_results[df_results['vix'] <= uuu]
                delta_upper_side = tmp_df['delta_upper_side'].mean()
                if np.isnan(delta_upper_side):
                    continue
                returned_results['call'].update({f'success_rate__vix{uuu}': tmp_df['success_call_credit'].mean()})
                if args.verbose:
                    print(f"\t[VIX<={uuu}] ,     CALL : {tmp_df['success_call_credit'].mean() * 100:.1f}% success")
        if iron_condor:
            returned_results.update({'iron_condor': {}})
            for uuu in vixes:
                tmp_df = df_results[df_results['vix'] <= uuu]
                if 0 == len(tmp_df):
                    continue
                returned_results['iron_condor'].update({f'success_rate__vix{uuu}': tmp_df['success_iron_condor'].mean()})
                if args.verbose:
                    print(f"\t[VIX<={uuu}] IRON CONDOR : {tmp_df['success_iron_condor'].mean() * 100:.1f}% success")
    return returned_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
    parser.add_argument("--look_ahead", type=int, default=1)
    #
    parser.add_argument("--lower_side_scale_factor", type=float, default=1, help="Gets multiplied againts the current price to obtain the lower side price")
    parser.add_argument("--upper_side_scale_factor", type=float, default=1, help="Gets multiplied againts the current price to obtain the upper side price")
    parser.add_argument('--put', type=str2bool, default=True)
    parser.add_argument('--call', type=str2bool, default=True)
    parser.add_argument('--iron_condor', type=str2bool, default=False)

    parser.add_argument('--step-back-range', type=int, default=99999,
                        help="Number of historical time windows to simulate (rolling backtest depth).")
    #
    parser.add_argument('--use_directional_var', type=str2bool, default=False)
    parser.add_argument('--use_directional_var__vix3m', type=str2bool, default=True)
    # RSI
    parser.add_argument("--rsi_period", type=float, default=14, help="")
    # SMA
    parser.add_argument("--sma_period", type=int, default=50, help="Period used to calculate SMA trend")
    # EMA
    parser.add_argument("--ema_short", type=int, default=21, help="Short EMA span for trend confirmation (default: 21)")
    parser.add_argument("--ema_long", type=int, default=50, help="Long EMA span for trend baseline (default: 50)")
    # MACD
    parser.add_argument("--macd_fast_period", type=int, default=12, help="Fast EMA period for MACD (default: 12)")
    parser.add_argument("--macd_slow_period", type=int, default=26, help="Slow EMA period for MACD (default: 26)")
    parser.add_argument("--macd_signal_period", type=int, default=9, help="Signal line EMA period for MACD (default: 9)")

    # Adjust call and put
    parser.add_argument('--adj_call__ema', type=str2bool, default=False, help="If True, augment the upper limit by a small value if EMA trend is against us")
    parser.add_argument('--adj_call__ema_factor', type=float, default=1.01, help="Value used to augment the upper limit")
    parser.add_argument('--adj_put__ema', type=str2bool, default=False, help="If True, reduce the lower limit by a small value if EMA trend is against us")
    parser.add_argument('--adj_put__ema_factor', type=float, default=0.99, help="Value used to reduce the lower limit")

    parser.add_argument('--adj_call__sma', type=str2bool, default=False, help="If True, augment the upper limit by a small value if SMA trend is against us")
    parser.add_argument('--adj_call__sma_factor', type=float, default=1.01, help="Value used to augment the upper limit")
    parser.add_argument('--adj_put__sma', type=str2bool, default=False, help="If True, reduce the lower limit by a small value if SMA trend is against us")
    parser.add_argument('--adj_put__sma_factor', type=float, default=0.99, help="Value used to reduce the lower limit")

    parser.add_argument('--adj_call__rsi', type=str2bool, default=False, help="If True, augment the upper limit by a small value if RSI value is against us")
    parser.add_argument('--adj_call__rsi_factor', type=float, default=1.01, help="Value used to augment the upper limit")
    parser.add_argument('--adj_put__rsi', type=str2bool, default=False, help="If True, reduce the lower limit by a small value if RSI value is against us")
    parser.add_argument('--adj_put__rsi_factor', type=float, default=0.99, help="Value used to reduce the lower limit")

    # MACD Adjustments
    parser.add_argument('--adj_call__macd', type=str2bool, default=False, help="If True, augment the upper limit if MACD is bullish")
    parser.add_argument('--adj_call__macd_factor', type=float, default=1.01, help="Value used to augment the upper limit based on MACD")
    parser.add_argument('--adj_put__macd', type=str2bool, default=False, help="If True, reduce the lower limit if MACD is bearish")
    parser.add_argument('--adj_put__macd_factor', type=float, default=0.99, help="Value used to reduce the lower limit based on MACD")

    parser.add_argument('--adj_call_and_put__contango', type=str2bool, default=False, help="If True, augment the upper and lower limits by a small value if VIX > VIX3m")
    parser.add_argument('--adj_call_and_put__contango_factor', type=float, default=0.01, help="Value used to change the upper and lower limits")

    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--verbose_lower_vix', type=str2bool, default=False)
    args = parser.parse_args()

    main(args)