"""
ATR-VRM (Average True Range Volatility Regime Model)
"""
import numpy as np
import pandas as pd
from fetchers.data_factory import factory_load_data
from utils import get_next_step
from runners.atr import entry as atr_entry
from argparse import Namespace
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


def entry():
    ###########################################################################
    # User defined parameters
    ###########################################################################
    g_ticker = "^GSPC"
    g_realtime = False
    g_dataset_id = "day"

    back_in_time_n_bars_ranges = list(range(1,3999))  # Backtesting ranges
    lookahead_ranges = [1, 2, 3, 4, 5]  # Number of days that we go into the future
    strategy_3_conf = {'up_bias': 1.01, 'down_bias': 0.99}
    n_split_for_atr = 0.8
    starting_n_trials_for_atr = 50
    tightness_weight_for_atr = 0.33
    enforce_code_expansion = True   # Ensure that high/low are monotonic accross all predictions
    # TODO use back_in_time_n_bars=0 and g_realtime=True for realtime

    ###########################################################################
    #
    ###########################################################################
    g_df_spx_full = factory_load_data(_dataset_id=g_dataset_id, _ticker=g_ticker, _args={})  # Contain all data, so that we can evaluate the results "in the future"
    compiled_predictions = {}  # All the results of backtesting used as input to evaluate different strategies
    strategies = defaultdict(dict)  # All the results extracted from backtesting evaluation
    close_col = ('Close', g_ticker)
    open_col = ('Open', g_ticker)
    high_col = ('High', g_ticker)
    low_col = ('Low', g_ticker)

    ###########################################################################
    # Compute Backtesting
    ###########################################################################
    for back_in_time_n_bars in tqdm(back_in_time_n_bars_ranges, desc="Backtesting"):  # Backtesting
        the_date_of_projection_is_made = None
        lookahead_prediction, ground_truth_data_available = {}, True
        for lookahead in lookahead_ranges:  # Projection in the future
            assert g_dataset_id in ["day"]
            dataset_id = f"extraday-{lookahead}days"
            n_split = n_split_for_atr
            n_trials = starting_n_trials_for_atr
            tightness_weight = tightness_weight_for_atr
            the_date_in_the_future = None
            extra_pen_cone_expansion = 0  # Penalty for not respecting cone expansion
            extra_tw_cone_expansion = 0  # Some slack for not respecting cone expansion
            for max_retry in range(0, 5):
                extra_pen_band_limits = 0  # Penalty for not respecting band limits
                # Estimation of bands for lookahead
                while True:
                    bands_estimation = {}
                    for use_close_for_range in [True, False]:
                        tw = tightness_weight - extra_tw_cone_expansion if tightness_weight - extra_tw_cone_expansion > 0 else 0
                        atr_results = atr_entry(Namespace(
                                    ticker=g_ticker, dataframe=None, dataset_id=dataset_id, use_realtime_data=g_realtime, verbose=False,
                                    atr_window=14, tightness_weight=tw,
                                    clip_n_before=back_in_time_n_bars,
                                    n_split=n_split, n_trials=n_trials + extra_pen_band_limits + extra_pen_cone_expansion, timeout=9999, use_close_for_range=use_close_for_range
                                ))

                        df_bt, vix_col, atr_col = atr_results['dataframe_and_cols']

                        dataset_configuration = atr_results['dataset_configuration']
                        #print(f"Train from {dataset_configuration['train_info']['start_date']}::{dataset_configuration['train_info']['end_date']}")
                        #print(f"Test  from {dataset_configuration['test_info']['start_date']}::{dataset_configuration['test_info']['end_date']}")

                        regime = atr_results['realtime']['vix_regime']
                        atr_prob_up = atr_results['regime_metrics'][regime]['Borne Haute Respectée'] / 100.
                        atr_prob_down = atr_results['regime_metrics'][regime]['Borne Basse Respectée'] / 100.
                        predicted_high = atr_results['realtime']['predicted_high']
                        predicted_low = atr_results['realtime']['predicted_low']
                        actual_low = atr_results['realtime']['actual_low']
                        actual_high = atr_results['realtime']['actual_high']
                        actual_close = atr_results['realtime']['actual_close']
                        actual_open= atr_results['realtime']['actual_open']

                        # Just a reminder
                        if not use_close_for_range:
                            assert df_bt.iloc[-1][close_col] == actual_close
                            assert df_bt.iloc[-1][open_col] == actual_open
                            assert df_bt.iloc[-1][high_col] == actual_high
                            assert df_bt.iloc[-1][low_col] == actual_low
                        if use_close_for_range:
                            assert df_bt.iloc[-1][close_col] == actual_close
                            assert df_bt.iloc[-1][open_col] == actual_open
                            assert df_bt.iloc[-1][close_col] == actual_high
                            assert df_bt.iloc[-1][close_col] == actual_low

                        actual_day_of_the_prediction = df_bt.index[-1]
                        if the_date_of_projection_is_made is None:
                            the_date_of_projection_is_made = actual_day_of_the_prediction
                        else:
                            if the_date_of_projection_is_made != actual_day_of_the_prediction:
                                print(f"\nFUCK!!  {the_date_of_projection_is_made} == {actual_day_of_the_prediction}   {back_in_time_n_bars=}TODO FIXME")
                            # assert the_date_of_projection_is_made == actual_day_of_the_prediction, f"{the_date_of_projection_is_made} == {actual_day_of_the_prediction}"
                        day_predicted_in_the_future = get_next_step(actual_day_of_the_prediction, "day", lookahead)
                        assert the_date_of_projection_is_made < day_predicted_in_the_future
                        if the_date_in_the_future is None:
                            the_date_in_the_future = day_predicted_in_the_future
                        else:
                            assert the_date_in_the_future == day_predicted_in_the_future
                        # print(f"{actual_day_of_the_prediction=}  {day_predicted_in_the_future=}   {predicted_low=}({atr_prob_down:.0%})  {predicted_high=}({atr_prob_up:.0%})")
                        if use_close_for_range:
                            bands_estimation.update({"predicted_upper_close": (predicted_high, atr_prob_up)})
                            bands_estimation.update({"predicted_lower_close": (predicted_low, atr_prob_down)})
                        else:
                            bands_estimation.update({"predicted_high": (predicted_high, atr_prob_up)})
                            bands_estimation.update({"predicted_low" : (predicted_low, atr_prob_down)})
                        bands_estimation.update({'the_date_in_the_future': the_date_in_the_future})
                        # Sanity check
                        assert the_date_in_the_future not in df_bt
                    if (bands_estimation['predicted_upper_close'][0] > bands_estimation['predicted_lower_close'][0] and
                            bands_estimation['predicted_upper_close'][0] < bands_estimation['predicted_high'][0] and
                            bands_estimation['predicted_lower_close'][0] > bands_estimation['predicted_low'][0] and
                            bands_estimation['predicted_high'][0] > bands_estimation['predicted_low'][0]):
                        break
                    else:
                        extra_pen_band_limits += 25
                        #print(f"\nFUCKKKKKKKKKKKKKKKKKKKKKKKK!!!!!!!!!!!!!!!!! {extra_pen_band_limits=}")
                # Sanity check
                assert bands_estimation['predicted_upper_close'][0] > bands_estimation['predicted_lower_close'][0]
                assert bands_estimation['predicted_upper_close'][0] < bands_estimation['predicted_high'][0]
                assert bands_estimation['predicted_lower_close'][0] > bands_estimation['predicted_low'][0]
                assert bands_estimation['predicted_high'][0] > bands_estimation['predicted_low'][0]
                if 0 == len(lookahead_prediction) or not enforce_code_expansion:
                    break
                # Enforce that the high/low prediction shall be greater/lower than those of previous bar
                if (lookahead_prediction[lookahead-1]['predicted_high'][0] > bands_estimation['predicted_high'][0]
                        or lookahead_prediction[lookahead - 1]['predicted_low'][0] < bands_estimation['predicted_low'][0]):
                    # Actual high and/or low is/are not higher/lower than previous values
                    extra_pen_cone_expansion += 25
                    extra_tw_cone_expansion -= 0.1
                    #print(f"\nFUCKKKKKKKKKKKKKKKKKKKKKKKK!!!!!!!!!!!!!!!!! {extra_pen_cone_expansion=}")
                    continue
                break
            lookahead_prediction.update({lookahead: bands_estimation})
            lookahead_prediction[lookahead].update({"date_projection_is_made": the_date_of_projection_is_made})
            #print(f"{the_date_of_projection_is_made}-->{the_date_in_the_future}")
            if the_date_in_the_future not in g_df_spx_full.index:
                ground_truth_data_available = False
                break
        if not ground_truth_data_available:# No ground truth data to evaluate the projection (all or some part of it is unavailable)
            continue
        compiled_predictions.update({back_in_time_n_bars: {}})
        compiled_predictions[back_in_time_n_bars].update({"the_date_of_projection_is_made": the_date_of_projection_is_made,
                                                          'closing_price_on_the_date_the_projection_is_made': g_df_spx_full.loc[the_date_of_projection_is_made][close_col],
                                                          'closing_price_at_end_of_projection': g_df_spx_full.loc[get_next_step(the_date_of_projection_is_made, "day", len(lookahead_ranges))][close_col],
                                                          'day_at_end_of_projection': get_next_step(the_date_of_projection_is_made, "day", len(lookahead_ranges)),
                                                          'lookahead_ranges': len(lookahead_ranges)},)
        compiled_predictions[back_in_time_n_bars].update({"lookahead_prediction": lookahead_prediction})
        assert len(lookahead_ranges) == len(lookahead_prediction)
        assert compiled_predictions[back_in_time_n_bars]['day_at_end_of_projection'] == compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['the_date_in_the_future']

    ###########################################################################
    # Evaluate Strategies
    ###########################################################################
    for back_in_time_n_bars in tqdm(back_in_time_n_bars_ranges, desc="Evaluate Strategies"):
        if back_in_time_n_bars not in compiled_predictions:
            continue  # No data for this
        #######################################################################
        # Strategy #1: price is in range of predicted closing price
        #######################################################################
        closing_price_ground_truth = compiled_predictions[back_in_time_n_bars]['closing_price_at_end_of_projection']
        upper_closing_price_prediction = compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_upper_close']
        lower_closing_price_prediction = compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_lower_close']
        #print(f"{closing_price_ground_truth} == {upper_closing_price_prediction} :: {lower_closing_price_prediction}")
        win_put_side = lower_closing_price_prediction[0] <= closing_price_ground_truth
        win_call_side = closing_price_ground_truth <= upper_closing_price_prediction[0]
        win = lower_closing_price_prediction[0] <= closing_price_ground_truth <= upper_closing_price_prediction[0]
        strategies["strategy_1"].setdefault('description', "price is in range of predicted closing price")
        strategies["strategy_1"].setdefault('stats_put_side', []).append(win_put_side)
        strategies["strategy_1"].setdefault('stats_call_side', []).append(win_call_side)
        strategies["strategy_1"].setdefault('stats_range', []).append(win)
        strategies["strategy_1"].setdefault('back_in_time_n_bars', []).append(back_in_time_n_bars)

        #######################################################################
        # Strategy #2: price is in range of predicted high/low prices
        #######################################################################
        closing_price_ground_truth = compiled_predictions[back_in_time_n_bars]['closing_price_at_end_of_projection']
        high_price_prediction = compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_high']
        low_price_prediction = compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_low']
        #print(f"{closing_price_ground_truth} == {high_price_prediction} :: {low_price_prediction}")
        win_put_side = low_price_prediction[0] <= closing_price_ground_truth
        win_call_side = closing_price_ground_truth <= high_price_prediction[0]
        win = low_price_prediction[0] <= closing_price_ground_truth <= high_price_prediction[0]
        strategies["strategy_2"].setdefault('description', "price is in range of predicted high/low prices")
        strategies["strategy_2"].setdefault('stats_put_side', []).append(win_put_side)
        strategies["strategy_2"].setdefault('stats_call_side', []).append(win_call_side)
        strategies["strategy_2"].setdefault('stats_range', []).append(win)
        strategies["strategy_2"].setdefault('back_in_time_n_bars', []).append(back_in_time_n_bars)

        #######################################################################
        # Strategy #3: Energy driven: add bias on the risky side
        #######################################################################
        #print(f"------------------------------------------------------")
        closing_price_ground_truth = compiled_predictions[back_in_time_n_bars]['closing_price_at_end_of_projection']
        closing_price_at_t_0 = compiled_predictions[back_in_time_n_bars]['closing_price_on_the_date_the_projection_is_made']
        #print(closing_price_at_t_0)
        #print(compiled_predictions[back_in_time_n_bars]['lookahead_prediction'])
        energy_ratio = []
        for lookahead_t, payload in compiled_predictions[back_in_time_n_bars]['lookahead_prediction'].items():
            #print(f"{lookahead_t} == {payload}")
            v1 = payload['predicted_upper_close'][0] - closing_price_at_t_0
            v2 = closing_price_at_t_0 - payload['predicted_lower_close'][0]
            #print(f"{v1=}  {v2=}")
            energy_ratio.append(float(v1)/float(v2))
        mean_energy_ratio = np.mean(energy_ratio)
        #print(f"{energy_ratio=}  {mean_energy_ratio=}")
        upper_closing_price_prediction = list(compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_upper_close'])
        lower_closing_price_prediction = list(compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_lower_close'])
        if mean_energy_ratio > 1:
            upper_closing_price_prediction[0] = compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_upper_close'][0] * strategy_3_conf["up_bias"]
        else:
            lower_closing_price_prediction[0] = compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_lower_close'][0] * strategy_3_conf["down_bias"]
        win_put_side  = lower_closing_price_prediction[0] <= closing_price_ground_truth
        win_call_side = closing_price_ground_truth <= upper_closing_price_prediction[0]
        win_range = lower_closing_price_prediction[0] <= closing_price_ground_truth <= upper_closing_price_prediction[0]
        if win_range: assert win_put_side and win_call_side
        if not win_range: assert win_put_side or win_call_side
        strategies["strategy_3"].setdefault('description', "Energy driven: add bias on the risky side")
        strategies["strategy_3"].setdefault('final_range', []).append((lower_closing_price_prediction[0], closing_price_ground_truth, upper_closing_price_prediction[0]))
        strategies["strategy_3"].setdefault('stats_put_side', []).append(win_put_side)
        strategies["strategy_3"].setdefault('stats_call_side', []).append(win_call_side)
        strategies["strategy_3"].setdefault('stats_range', []).append(win_range)
        strategies["strategy_3"].setdefault('back_in_time_n_bars', []).append(back_in_time_n_bars)
        #print(f"------------------------------------------------------")

        #######################################################################
        # Strategy #4: win if price breaks entry price in half end side
        #######################################################################
        closing_price_at_t_0 = compiled_predictions[back_in_time_n_bars]['closing_price_on_the_date_the_projection_is_made']
        closing_price_ground_truth = compiled_predictions[back_in_time_n_bars]['closing_price_at_end_of_projection']
        predicted_low_closing_prices, predicted_high_closing_prices, ground_truth_closing_prices = [], [], []
        put_side_results, call_side_results = [], []
        for lookahead_tn, payload in compiled_predictions[back_in_time_n_bars]['lookahead_prediction'].items():
            predicted_low_closing_prices.append(payload['predicted_low'][0])
            predicted_high_closing_prices.append(payload['predicted_high'][0])
            ground_truth_closing_prices.append(g_df_spx_full.loc[payload['the_date_in_the_future']][close_col])
        # 1. Détermination de l'indice de départ pour la deuxième moitié des données
        indice_milieu = len(ground_truth_closing_prices) // 2
        deuxieme_moitie = ground_truth_closing_prices[indice_milieu:]
        # 2. Application des conditions logiques
        # PUT SIDE : True si AU MOINS UNE valeur de la 2e moitié est STRICTEMENT SUPÉRIEURE à t_0
        put_side = any(prix > closing_price_at_t_0 for prix in deuxieme_moitie)
        # CALL SIDE : True si AU MOINS UNE valeur de la 2e moitié est STRICTEMENT INFÉRIEURE à t_0
        call_side = any(prix < closing_price_at_t_0 for prix in deuxieme_moitie)
        upper_closing_price_prediction = list(compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_upper_close'])
        lower_closing_price_prediction = list(compiled_predictions[back_in_time_n_bars]['lookahead_prediction'][len(lookahead_ranges)]['predicted_lower_close'])
        win_put_side  = put_side  or lower_closing_price_prediction[0] <= closing_price_ground_truth
        win_call_side = call_side or closing_price_ground_truth <= upper_closing_price_prediction[0]
        strategies["strategy_4"].setdefault('description', "win if price breaks entry price in half end side")
        strategies["strategy_4"].setdefault('final_range', []).append((lower_closing_price_prediction[0], closing_price_ground_truth, upper_closing_price_prediction[0]))
        strategies["strategy_4"].setdefault('stats_put_side', []).append(win_put_side)
        strategies["strategy_4"].setdefault('stats_call_side', []).append(win_call_side)
        strategies["strategy_4"].setdefault('stats_range', []).append(win_call_side and win_put_side)
        strategies["strategy_4"].setdefault('back_in_time_n_bars', []).append(back_in_time_n_bars)

    ###########################################################################
    # Display the results
    ###########################################################################
    for strategy_id, payload in strategies.items():
        print(f"{strategy_id} ({payload['description']}): Put:{np.mean(payload['stats_put_side']):.0%}   Call:{np.mean(payload['stats_call_side']):.0%}   IC:{np.mean(payload['stats_range']):.0%}")


if __name__ == "__main__":
    entry()
