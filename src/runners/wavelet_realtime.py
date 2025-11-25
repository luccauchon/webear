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
from constants import FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_DAY, OUTPUT_DIR_WAVLET_BASED_STOCK_FORECAST
from utils import transform_path


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
        older_dataset=None,  # Don't modify
        dataset_id=args.dataset_id,
        number_of_step_back=1,  # FIXME TODO permettre plusieurs steps , même dans le real time
        n_forecast_length=args.n_forecast_length,  # FIXME TODO un length pour ici et un autre pour le forecast voulu? si !=, alors, boucle de retroaction
        n_forecast_length_in_training=args.n_forecast_length_in_training,
        n_models_to_keep=args.n_models_to_keep,
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
        strategy_for_exit=args.strategy_for_exit,
    )
    # Get the best parameters for the real time processor
    _, _, _, _, parameters_best_models, _ = wavelet_optimizer_entry_point(configuration)
    assert 1 == len(parameters_best_models)

    #######################################################################
    # Projection dans le futur
    #######################################################################
    # Adjust some parameters
    configuration.real_time           = True
    configuration.real_time_use_cases = parameters_best_models[0]
    configuration.plot_graph          = args.plot_graph
    configuration.quiet               = not args.verbose
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
        # Save and show AFTER adding text
        if older_dataset is not None:
            today_str = datetime.today().strftime('%Y-%m-%d')
            os.makedirs(os.path.join(OUTPUT_DIR_WAVLET_BASED_STOCK_FORECAST, args.older_dataset), exist_ok=True)
            figure_filename = os.path.join(OUTPUT_DIR_WAVLET_BASED_STOCK_FORECAST, args.older_dataset, f"forecast__{args.ticker}_{args.dataset_id}__.png")
            plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    return description_of_what_user_shall_do[0], misc_returned[0]


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Wavelet-based stock real time estimator.")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument("--dataset_id", type=str, default="week", choices=['week', 'day'])
    parser.add_argument("--n_forecast_length", type=int, default=4)
    parser.add_argument("--n_forecast_length_in_training", type=int, default=4)
    parser.add_argument("--n_models_to_keep", type=int, default=60)
    parser.add_argument('--thresholds_ep', type=str, default="(0.0125, 0.0125)")
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    output_dir = f"../../stubs/wavelet_realtime_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}/"
    os.makedirs(output_dir, exist_ok=True)

    if args.dataset_id == 'day':
        df_filename = FYAHOO__OUTPUTFILENAME_DAY
    elif args.dataset_id == 'week':
        df_filename = FYAHOO__OUTPUTFILENAME_WEEK
    older_dataset = None if args.older_dataset == "None" else args.older_dataset
    one_dataset_filename = df_filename if older_dataset is None else transform_path(df_filename, older_dataset)

    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = master_data_cache[args.ticker].copy()

    args = Namespace(
        master_data_cache=master_data_cache.copy(),
        ticker=args.ticker,
        col=args.col,
        output_dir=output_dir,
        temp_filename=os.path.join(output_dir, "real_time.pkl"),
        dataset_id=args.dataset_id,
        older_dataset=args.older_dataset,
        n_forecast_length=args.n_forecast_length,
        n_forecast_length_in_training=args.n_forecast_length_in_training,
        n_models_to_keep=args.n_models_to_keep,
        thresholds_ep=args.thresholds_ep,
        plot_graph = True,
        use_given_gt_truth = None,
        display_tqdm = False,
        strategy_for_exit = 'hold_until_the_end_with_roll',
        verbose = args.verbose,
    )
    # --- Nicely print the arguments ---
    print("🔧 Arguments:")
    for arg, value in vars(args).items():
        if 'master_data_cache' in arg:
            print(f"    {arg:.<40} {value.index[0].strftime('%Y-%m-%d')} to {value.index[-1].strftime('%Y-%m-%d')} ({one_dataset_filename})")
            continue
        print(f"    {arg:.<40} {value}")
    print("-" * 80, flush=True)
    main(args)