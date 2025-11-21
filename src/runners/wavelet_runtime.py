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
import argparse
from multiprocessing import freeze_support, Lock, Process, Queue, Value
from optimizers.wavelet_opt import main as wavelet_optimizer_entry_point
from argparse import Namespace
import matplotlib.pyplot as plt


def main(args):
    ticker = args.ticker
    col = args.col
    dataset_id = 'day'
    output_dir = "../../stubs/wavelet_runtime_/"
    number_of_step_back = 1
    n_forecast_length =1
    n_models_to_keep = 60
    thresholds_ep = "(0.0125, 0.012)"
    configuration = Namespace(
        ticker=ticker,
        col=col,
        output_dir = output_dir,
        older_dataset=None,
        dataset_id=dataset_id,
        number_of_step_back=number_of_step_back,
        n_forecast_length=n_forecast_length,
        n_models_to_keep=n_models_to_keep,
        plot_graph=False,
        save_graph=False,
        do_not_close_graph=True,
        thresholds_ep = thresholds_ep,
        quiet = True,
        floor_and_ceil = 5,
        maintenance_margin = 2000,
    )
    _, _, _, description_of_what_user_shall_do = wavelet_optimizer_entry_point(configuration)
    assert 1 == len(description_of_what_user_shall_do)
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


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Wavelet-based stock real time estimator.")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    args = parser.parse_args()

    main(args)