try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import os
import json
import random
import argparse
from datetime import datetime
from multiprocessing import freeze_support, Lock, Process, Queue, Value
from optimizers.wavelet_opt import main as wavelet_optimizer_entry_point
from argparse import Namespace
import matplotlib.pyplot as plt
import pickle
from constants import FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_DAY


def main(args):
    #######################################################################
    # Extraction des bons paramètres pour le forecasting
    #######################################################################
    configuration = Namespace(
        master_data_cache=args.master_data_cache.copy(),
        ticker=args.ticker,
        col= args.col,
        output_dir = args.output_dir,
        temp_filename=os.path.join(args.output_dir, 'real_time.pkl'),
        older_dataset=None,
        dataset_id=args.dataset_id,
        number_of_step_back=1,  # FIXME TODO permettre plusieurs steps
        n_forecast_length=args.n_forecast_length,  # FIXME TODO un length pour ici et un autre pour le forecast voulu? si !=, alors, boucle de retroaction
        n_models_to_keep=60,
        plot_graph=False,
        show_graph=False,
        save_graph=False,
        do_not_close_graph=True,
        thresholds_ep = args.thresholds_ep,
        quiet = True,
        floor_and_ceil = 5,
        maintenance_margin = 2000,
        real_time = False,
        use_given_gt_truth = None,
        display_tqdm = args.display_tqdm,
    )
    # Get the best parameters for the real time processor
    _, _, _, _, parameters_best_models, _ = wavelet_optimizer_entry_point(configuration)
    assert 1 == len(parameters_best_models)

    #######################################################################
    # Projection dans le futur
    #######################################################################
    # Adjust some parameters
    configuration.real_time = True
    configuration.real_time_use_cases = parameters_best_models[0]
    configuration.plot_graph = True
    if args.use_given_gt_truth is not None:
        configuration.use_given_gt_truth = args.use_given_gt_truth
    # Run the real time processor
    _, _, _, description_of_what_user_shall_do, _, misc_returned = wavelet_optimizer_entry_point(configuration)
    # Only a step back when we do a real forecast
    assert 1 == len(description_of_what_user_shall_do) and 1 == len(misc_returned)

    #######################################################################
    # Display
    #######################################################################
    if args.plot_graph:
        user_instruction = description_of_what_user_shall_do[0]['description']
        # Get current figure (assumes wavelet_optimizer_entry_point already plotted something)
        fig = plt.gcf()

        # Add nicely styled text box with the instruction
        fig.text(
            0.5, 0.02,  # x, y position (bottom center)
            user_instruction,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=12,
            color='black',
            fontfamily='sans-serif',
            bbox=dict(
                facecolor='lightgray',
                edgecolor='darkgray',
                boxstyle='round,pad=0.5',
                alpha=0.9
            )
        )

        plt.tight_layout(rect=[0, 0.1, 1, 1])  # Make room for the text at the bottom
        plt.show()
        plt.close()
    return description_of_what_user_shall_do[0], misc_returned[0]


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Wavelet-based stock real time estimator.")
    args = parser.parse_args()

    ticker = '^GSPC'
    col = 'Close'
    dataset_id = 'day'
    output_dir = f"../../stubs/wavelet_realtime_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}/"
    os.makedirs(output_dir, exist_ok=True)
    n_forecast_length   = 1
    thresholds_ep = "(0.0125, 0.012)"

    if dataset_id == 'day':
        df_filename = FYAHOO__OUTPUTFILENAME_DAY
    elif dataset_id == 'week':
        df_filename = FYAHOO__OUTPUTFILENAME_WEEK
    with open(df_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = master_data_cache[ticker].copy()

    args = Namespace(
        master_data_cache=master_data_cache.copy(),
        ticker=ticker,
        col=col,
        output_dir=output_dir,
        temp_filename=os.path.join(output_dir, "real_time.pkl"),
        dataset_id=dataset_id,
        n_forecast_length=n_forecast_length,
        thresholds_ep=thresholds_ep,
        plot_graph = True,
        use_given_gt_truth = None,
        display_tqdm = False,
    )

    main(args)