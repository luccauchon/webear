try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from argparse import Namespace
import pickle
from utils import get_filename_for_dataset, next_day
from optimizers.roulette.realtime_and_backtest import main as roulette


def entry():
    # ================================================================================
    # 🏆 Optimization Finished!
    # Best Score (pos_seq__f1): 0.88512940
    # Best Parameters:
    #     sma_windows_tuple....................... 110,170
    #     sma_shifts_tuple........................
    #     rsi_windows_tuple....................... 3
    #     rsi_shifts_tuple........................ 2
    #     vwap_window............................. 21
    #     shift_seq_col........................... 3
    # ================================================================================
    #
    # {'sma_windows_tuple': '110,170', 'sma_shifts_tuple': '', 'rsi_windows_tuple': '3', 'rsi_shifts_tuple': '2', 'vwap_window': 21, 'shift_seq_col': 3}
    # To run the best experiment (roulette):
    # python realtime_and_backtest.py --ticker "^GSPC" --dataset_id day --look_ahead 1 --step_back_range 260 --epsilon 0.01 --target POS_SEQ --convert_price_level_with_baseline fraction --verbose true --older_dataset none --enable_ema false --enable_sma true --sma_windows 110 170 --enable_rsi true --rsi_windows 3 --shift_rsi_col 2 --enable_macd false --enable_vwap true --vwap_window 21 --enable_day_data True --shift_seq_col 3 --min_percentage_to_keep_class 4.0 --base_models xgb --add_only_vwap_z_and_vwap_triggers True
    # To recompile the model, added the filename:
    # (PY312_HT) PS D:\PyCharmProjects\webear\src\optimizers\roulette> python realtime_and_backtest.py --ticker "^GSPC" --dataset_id day --look_ahead 1 --step_back_range 99999 --epsilon 0.01 --target POS_SEQ --convert_price_level_with_baseline fraction --verbose true --older_dataset none --enable_ema false --enable_sma true --sma_windows 110 170 --enable_rsi true --rsi_windows 3 --shift_rsi_col 2 --enable_macd false --enable_vwap true --vwap_window 21 --enable_day_data True --shift_seq_col 3 --min_percentage_to_keep_class 4.0 --base_models xgb --add_only_vwap_z_and_vwap_triggers True --save_model_path 2026.03.20.M1.model
    # Xs are:
    # [('SMA_110', '^GSPC'), ('SMA_170', '^GSPC'), ('RSI_3', '^GSPC'), ('SHIFTED_RSI3_2', '^GSPC'), ('VWAP_21_z', '^GSPC'), ('VWAP_21_above_1sigma', '^GSPC'),
    # ('VWAP_21_below_1sigma', '^GSPC'), ('VWAP_21_above_2sigma', '^GSPC'), ('VWAP_21_below_2sigma', '^GSPC'), ('VWAP_21_above_3sigma', '^GSPC'),
    # ('VWAP_21_below_3sigma', '^GSPC'), ('day_sin', '^GSPC'), ('day_cos', '^GSPC'), ('SHIFTED_1_POS_SEQ', '^GSPC'), ('SHIFTED_1_NEG_SEQ', '^GSPC'),
    # ('SHIFTED_2_POS_SEQ', '^GSPC'), ('SHIFTED_2_NEG_SEQ', '^GSPC'), ('SHIFTED_3_POS_SEQ', '^GSPC'), ('SHIFTED_3_NEG_SEQ', '^GSPC'), ('POS_SEQ', '^GSPC'),
    # ('NEG_SEQ', '^GSPC'), ('STREAK_SEQ', '^GSPC'), ('Close', '^GSPC')]
    # Ys are: [('POS_SEQ', '^GSPC')]
    output_filename_for_dataset = r"D:\Finance\data\roulette\2026.03.20.M1.data.npz"
    model_filename = r'D:\Finance\compiled_models\roulette\2026.03.20.M1.model'

    print(f"Construction du dataset dans {output_filename_for_dataset}...")
    configuration = Namespace(ticker="^GSPC", dataset_id="day", step_back_range=260, look_ahead=1, epsilon=0.01, target="POS_SEQ", convert_price_level_with_baseline="fraction",
                              verbose=False, older_dataset=None, enable_ema=False, enable_sma=True, sma_windows=[110,170], enable_rsi=True, rsi_windows=[3], shift_rsi_col=[2],
                              enable_macd=False, enable_vwap=True, vwap_window=21, enable_day_data=True, shift_seq_col=3,min_percentage_to_keep_class=4.0,
                              base_models="xgb", add_only_vwap_z_and_vwap_triggers=True, compiled_dataset_filename=None, real_time_only=True, macd_params=None,
                              add_close_diff=None,shift_sma_col=[], specific_wanted_class=[], load_model_path=None, shift_macd_col=[], real_time_only_num_classes=2,
                              model_overrides="{}", save_dataset_to_file_and_exit=output_filename_for_dataset, drop_when_out_of_range=False, optimization_strategy="classification")
    roulette(configuration)

    print(f"Excution du modèle {model_filename}...")
    configuration.load_model_path = model_filename
    configuration.real_time_only = True
    configuration.compiled_dataset_filename = output_filename_for_dataset
    predicted_classes, prediction_probabilities, date_of_data_point_x_used, model_data = roulette(configuration)
    print(f"Chargement des données...")
    one_dataset_filename = get_filename_for_dataset("day", older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    close_col = ("Close", "^GSPC")
    master_data_cache = master_data_cache['^GSPC'].sort_index()[close_col]
    assert master_data_cache.index[-1] == date_of_data_point_x_used
    close_value = master_data_cache.iloc[-1]
    lower_close_value, upper_close_value = 0.99 * close_value, 1.01*close_value
    if 0 == predicted_classes:
        print(f"[POS SEQ={predicted_classes}] There is {prediction_probabilities[predicted_classes]*100:.0f}% chance of closing price be above {lower_close_value:.0f} on {next_day(date_of_data_point_x_used)}")
    elif 1 == predicted_classes:
        print(f"[POS SEQ={predicted_classes}] There is {prediction_probabilities[predicted_classes]*100:.0f}% chance of closing price be above {lower_close_value:.0f} on {next_day(date_of_data_point_x_used)}")
    else:
        assert False


if __name__ == "__main__":
    entry()
