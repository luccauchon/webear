try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

import argparse
import pandas as pd
from campaigns.universal_playback import BacktestIterator, BacktestStep
from tqdm import tqdm
from runners.atr import calculate_atr
from runners.atr import entry as atr_entry_point
from argparse import Namespace
import pickle
from utils import get_filename_for_dataset
import datetime


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Backtesting framework for ATR-based options strategies (Iron Condor, Call/Put Spreads) with VIX regime filtering.",
        # Automatically appends "(default: ...)" to the end of all help strings!
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--step-back-range",
        type=int,
        default=8,
        help="Number of historical days to use as the lookback window for each backtest step."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output and display the tqdm progress bar."
    )
    parser.add_argument(
        "--atr-window",
        type=int,
        default=14,
        help="Window size (in periods) for calculating the Average True Range (ATR)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="^GSPC",
        help="Ticker symbol to run the backtest on (e.g., '^GSPC' for S&P 500)."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="day",
        help="Granularity/ID of the dataset to load (e.g., 'day', 'hour')."
    )
    parser.add_argument(
        "--tightness-weight",
        type=float,
        default=0.9,
        help="Weight factor applied to the tightness of the predicted bounds."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=500,
        help="Number of optimization trials to run."
    )
    parser.add_argument(
        "--n-split",
        type=float,
        default=0.8,
        help="Train/test split ratio for the data (e.g., 0.8 means 80%% training, 20%% testing)."
    )

    return parser


def entry(args):
    open_col = ('Open', args.ticker)
    close_col = ('Close', args.ticker)
    high_col = ('High', args.ticker)
    low_col = ('Low', args.ticker)

    one_dataset_filename = get_filename_for_dataset(dataset_choice=args.dataset_id, older_dataset=None)

    with open(one_dataset_filename, 'rb') as f:
        _master_data_cache = pickle.load(f)

    df_ticker = _master_data_cache[args.ticker].sort_index().copy()

    try:
        df_vix = _master_data_cache["^VIX_MEAN"].sort_index()
    except:
        # DAY data has not the VIX_MEAN dataset.
        df_vix = _master_data_cache["^VIX"].sort_index()

    df_ticker, atr_col, prev_close_col = calculate_atr(
        df=df_ticker, close_col=close_col, high_col=high_col, low_col=low_col,
        ticker=args.ticker, window=args.atr_window
    )
    vix_col = next((col for col in df_vix.columns if isinstance(col, tuple) and 'Close' in col), None)
    df_bt = df_ticker.join(df_vix[[vix_col]], how='inner').dropna().copy()
    # Before joining or calculating rolling rank, shift VIX by 1
    df_bt[vix_col] = df_bt[vix_col].shift(1)
    df_bt = df_bt.dropna()  # Drop the first row where VIX is now NaN
    # Calculates the rank of the current element within its rolling window automatically
    df_bt['VIX_Rolling_Rank'] = df_bt[vix_col].rolling(window=252, min_periods=20).rank(pct=True).copy()

    # 1️⃣ Create the iterator
    bt_iter = BacktestIterator(
        df=df_bt.copy(),
        step_back_range=args.step_back_range,
        verbose=False
    )

    # 2️⃣ Wrap with tqdm only if verbose (tqdm uses __len__ for accurate progress)
    iterator = tqdm(bt_iter) if args.verbose else bt_iter

    # Initialize statistics counters
    total_trades = 0
    iron_condor_wins = 0
    call_spread_wins = 0
    put_spread_wins = 0

    # 3️⃣ Run multiple algorithms cleanly
    for step in iterator:
        past_df = step.past_df
        future_df = step.future_df
        if future_df is not None:
            assert past_df.index.intersection(future_df.index).empty

        config = Namespace(dataframe=past_df.copy(),
                           ticker=args.ticker,
                           n_split=args.n_split,
                           atr_window=args.atr_window,
                           dataset_id=args.dataset_id,
                           n_trials=args.n_trials,
                           tightness_weight=args.tightness_weight,
                           verbose=False)
        realtime_results = atr_entry_point(args=config)

        predicted_high = realtime_results["realtime"]["predicted_high"]
        predicted_low = realtime_results["realtime"]["predicted_low"]
        actual_high = realtime_results["realtime"]["actual_high"]
        actual_low = realtime_results["realtime"]["actual_low"]
        vix_regime = realtime_results["realtime"]["vix_regime"]
        vix_rank = realtime_results["realtime"]["vix_rank"]
        last_date = realtime_results["realtime"]["last_date"]

        # Ignore rows with NaNs/missing bounds (ensures we only evaluate valid setups)
        if pd.isna(predicted_high) or pd.isna(predicted_low) or pd.isna(actual_high) or pd.isna(actual_low):
            continue

        total_trades += 1

        # Evaluate Win Conditions
        is_high_win = predicted_high > actual_high
        is_low_win = predicted_low < actual_low
        is_total_win = is_high_win and is_low_win

        if is_high_win:
            call_spread_wins += 1
        if is_low_win:
            put_spread_wins += 1
        if is_total_win:
            iron_condor_wins += 1

    # 4️⃣ Display Statistics & Calculate Rates
    if total_trades > 0:
        iron_condor_wr = (iron_condor_wins / total_trades) * 100
        call_spread_wr = (call_spread_wins / total_trades) * 100
        put_spread_wr = (put_spread_wins / total_trades) * 100
    else:
        iron_condor_wr = call_spread_wr = put_spread_wr = 0.0

    if args.verbose:
        print("\n" + "=" * 60)
        print(" BACKTEST STATISTICS ".center(60, "="))
        print("=" * 60)

        if total_trades > 0:
            print(f"Total Valid Trades      : {total_trades}")
            print("-" * 60)

            print(f"Iron Condor Win Rate   : {iron_condor_wr:6.2f}% ({iron_condor_wins}/{total_trades})")
            print(f"  -> Win if: Predicted High > Actual High")
            print(f"             AND Predicted Low < Actual Low")
            print("-" * 60)

            print(f"Call Credit Spread WR  : {call_spread_wr:6.2f}% ({call_spread_wins}/{total_trades})")
            print(f"  -> Win if: Predicted High > Actual High")
            print("-" * 60)

            print(f"Put Credit Spread WR   : {put_spread_wr:6.2f}% ({put_spread_wins}/{total_trades})")
            print(f"  -> Win if: Predicted Low < Actual Low")

        else:
            print("No valid trades evaluated.")

        print("=" * 60 + "\n")

    # 5️⃣ Save Results to TXT File
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean up ticker for filename (remove special characters like '^')
    clean_ticker = args.ticker.replace("^", "")
    filename = f"backtest_results_{clean_ticker}_{args.dataset_id}__atr{args.atr_window}__tightness{args.tightness_weight}__ic{iron_condor_wr}__{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("BACKTEST PARAMETERS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Ticker              : {args.ticker}\n")
        f.write(f"Dataset ID          : {args.dataset_id}\n")
        f.write(f"Step Back Range     : {args.step_back_range}\n")
        f.write(f"ATR Window          : {args.atr_window}\n")
        f.write(f"Tightness Weight    : {args.tightness_weight}\n")
        f.write(f"Number of Trials    : {args.n_trials}\n")
        f.write(f"Train/Test Split    : {args.n_split}\n")
        f.write("=" * 60 + "\n\n")

        f.write("BACKTEST RESULTS\n")
        f.write("=" * 60 + "\n")

        if total_trades > 0:
            f.write(f"Total Valid Trades      : {total_trades}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Iron Condor Win Rate   : {iron_condor_wr:6.2f}% ({iron_condor_wins}/{total_trades})\n")
            f.write(f"  -> Win if: Predicted High > Actual High\n")
            f.write(f"             AND Predicted Low < Actual Low\n")
            f.write("-" * 60 + "\n")
            f.write(f"Call Credit Spread WR  : {call_spread_wr:6.2f}% ({call_spread_wins}/{total_trades})\n")
            f.write(f"  -> Win if: Predicted High > Actual High\n")
            f.write("-" * 60 + "\n")
            f.write(f"Put Credit Spread WR   : {put_spread_wr:6.2f}% ({put_spread_wins}/{total_trades})\n")
            f.write(f"  -> Win if: Predicted Low < Actual Low\n")
        else:
            f.write("No valid trades evaluated.\n")

        f.write("=" * 60 + "\n")

    print(f"\n[INFO] Results successfully saved to: {filename}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)