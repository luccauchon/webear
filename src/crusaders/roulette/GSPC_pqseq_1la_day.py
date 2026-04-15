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
from argparse import Namespace


def entry(args):
    # To generate the model:
    # (PY312_HT) D:\PyCharmProjects\webear\src\optimizers\roulette>python realtime_and_backtest.py --ticker "^GSPC" --dataset_id day --look_ahead 1 --step_back_range 1111 --epsilon 0.0 --convert_price_level_with_baseline fraction --verbose true --older_dataset none --enable_ema true --ema_windows 7 11 --enable_sma true --sma_windows 75 155 --shift_sma_col 1 --enable_rsi true --rsi_windows 9 --shift_rsi_col 3 --enable_macd True --macd_params "{\"fast\": 16, \"slow\": 24, \"signal\": 8}" --enable_vwap true --vwap_window 9 --enable_day_data True --shift_seq_col 2 --min_percentage_to_keep_class 4.0 --base-models xgb --add_only_vwap_z_and_vwap_triggers False --add_close_diff True --save-model-path D:\Finance\compiled_models\roulette\2026.04.06.M1.model

    # ================================================================================
    # 🏆 Optimization Finished!
    # Best Score (pos_seq__f1): 0.62567720
    output_filename_for_dataset = r"D:\Finance\data\roulette\2026.04.06.M1.data.npz"
    model_filename = r'D:\Finance\compiled_models\roulette\2026.04.06.M1.model'

    print(f"Construction du dataset dans {output_filename_for_dataset}...")
    configuration = Namespace(ticker="^GSPC", dataset_id="day", step_back_range=260, look_ahead=1, epsilon=0., target="SEQ_F1", convert_price_level_with_baseline="fraction",
                              verbose=False, older_dataset=None, enable_ema=True, enable_sma=True, sma_windows=[75,155], enable_rsi=True, rsi_windows=[9], shift_rsi_col=[3],
                              shift_ema_col=[], ema_windows=[7,11],
                              enable_macd=True, enable_vwap=True, vwap_window=9, enable_day_data=True, shift_seq_col=2,min_percentage_to_keep_class=4.0,
                              base_models="xgb", add_only_vwap_z_and_vwap_triggers=False, compiled_dataset_filename=None, real_time_only=True, macd_params='{"fast":16,"slow":24,"signal":8}',
                              add_close_diff=True,shift_sma_col=[1], specific_wanted_class=[], load_model_path=None, shift_macd_col=[], real_time_only_num_classes=2,
                              model_overrides="{}", save_dataset_to_file_and_exit=output_filename_for_dataset, drop_when_out_of_range=False, optimization_strategy="classification")
    roulette(configuration)
    if args.verbose:
        print(f"Excution du modèle {model_filename}...")
    configuration.load_model_path = model_filename
    configuration.real_time_only = True
    configuration.compiled_dataset_filename = output_filename_for_dataset
    predicted_classes, prediction_probabilities, date_of_data_point_x_used, model_data = roulette(configuration)
    if args.verbose:
        print(f"Chargement des données...")
    one_dataset_filename = get_filename_for_dataset("day", older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    close_col = ("Close", "^GSPC")
    master_data_cache = master_data_cache['^GSPC'].sort_index()[close_col]
    assert master_data_cache.index[-1] == date_of_data_point_x_used
    close_value = master_data_cache.iloc[-1]
    pqseq = model_data['class_id_pred__2__tuple'].get(predicted_classes)
    assert 2 == len(pqseq) and (0 == pqseq[0] or 0 == pqseq[1])
    _tmp_str = "below" if 0 == pqseq[0] else "above"
    if args.verbose:
        print(f"[POS SEQ={pqseq}] There is {prediction_probabilities[predicted_classes]*100:.0f}% chance of closing price be {_tmp_str} {close_value:.0f} on {next_day(date_of_data_point_x_used)}")
    return pqseq, prediction_probabilities[predicted_classes], close_value, next_day(date_of_data_point_x_used)


if __name__ == "__main__":
    entry(args=Namespace(verbose=True))
