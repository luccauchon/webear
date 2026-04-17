#!/usr/bin/env python3
"""
Market & Macro Data Collector
=============================
Downloads S&P 500, VIX, and FRED macroeconomic indicators, resamples them to
the specified frequency (monthly or weekly), validates against a cached dataset,
and saves the combined result as a pickle file.

Usage:
    python script.py [OPTIONS]
"""
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
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

from constants import FRED_API_KEY
from utils import get_filename_for_dataset


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="market_macro_collector",
        description=(
            "Download, resample, and combine S&P 500, VIX, and FRED macroeconomic "
            "data into a monthly or weekly dataset. Outputs a pickle file containing "
            "both market and macro data aligned to period-end."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python script.py --freq month --output-dir ./data --start-date 2000-01-01\n"
            "  python script.py --freq week -m FEDFUNDS,UNRATE,CPIAUCSL -e 2023-12-31\n"
            "  python script.py --freq week --vix-method last\n"
        )
    )

    parser.add_argument(
        "-o", "--output-dir", type=str, default=".",
        help="Directory to save the combined dataset (default: current working directory)"
    )
    parser.add_argument(
        "-s", "--start-date", type=str, default="1927-01-01",
        help="Start date for FRED data retrieval in YYYY-MM-DD format (default: 1927-01-01)"
    )
    parser.add_argument(
        "-e", "--end-date", type=str, default=None,
        help="End date for FRED data retrieval in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "-m", "--macro-indicators", type=str, default="FEDFUNDS,UNRATE,CPIAUCSL,T10Y2Y",
        help="Comma-separated list of FRED macroeconomic indicators to download (default: FEDFUNDS,UNRATE,CPIAUCSL,T10Y2Y)"
    )
    parser.add_argument(
        "-f", "--filename", type=str, default=None,
        help="Name of the output pickle file (default: auto-generated based on frequency)"
    )
    parser.add_argument(
        "--freq", type=str, choices=["month", "week"], default="month",
        help="Resampling frequency: 'month' (month-end) or 'week' (Friday close) (default: month)"
    )
    parser.add_argument(
        "-v", "--vix-method", type=str, choices=["mean", "last"], default="mean",
        help="VIX aggregation method: 'mean' for period average, 'last' for period-end close (default: mean)"
    )

    return parser.parse_args()


def entry_point(args: argparse.Namespace) -> None:
    """
    Main execution flow: fetches market & macro data, resamples to specified frequency,
    validates against a cached dataset, and saves the combined result.

    Args:
        args: Parsed command-line arguments from argparse.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Map human-readable frequency to pandas offset alias
    pd_freq = "ME" if args.freq == "month" else "W-FRI"
    freq_label = "Monthly" if args.freq == "month" else "Weekly"
    dataset_id = "month" if args.freq == "month" else "week"
    # Load cached master data for validation (cache is expected to be monthly)
    cache_filename = get_filename_for_dataset(dataset_id, older_dataset=None)
    with open(cache_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    df_spx500 = master_data_cache["^GSPC"].sort_index()
    df_vix = master_data_cache["^VIX_MEAN"].sort_index()
    if args.vix_method == "last":
        df_vix = master_data_cache["^VIX"].sort_index()

    # 1. Download market data (S&P 500 and VIX)
    print("📥 Downloading S&P 500 and VIX data from Yahoo Finance...")
    sp500 = yf.download("^GSPC ^VIX", period="max", interval="1d", auto_adjust=False)
    online_data = sp500['Close'].copy()

    # 2. Resample to target frequency
    print(f"📊 Resampling to {freq_label.lower()} frequency ({pd_freq})...")
    market_data = online_data['^GSPC'].resample(pd_freq).last().to_frame(name='Close')
    market_data['VIX'] = online_data['^VIX'].resample(pd_freq).mean()
    if args.vix_method == "last":
        market_data['VIX'] = online_data['^VIX'].resample(pd_freq).last()

    # Validation against cached monthly data (adjusted window size for frequency)
    val_window = 33 if args.freq == "month" else 8  # ~1 year of periods
    spx_close_col = ('Close', '^GSPC')
    print(f"🔍 Validating downloaded data against cache (last {val_window} periods)...")

    # Safe slicing for validation
    cached_spx = df_spx500[spx_close_col].iloc[-val_window - 1:-1]
    new_spx = market_data['Close'].iloc[-val_window - 1:-1]
    assert val_window == np.count_nonzero(cached_spx.values == new_spx.values), f"SPX validation failed for {freq_label}"
    # df_vix[vix_close_col].index[-1], market_data['VIX'].index[-1]
    vix_close_col = ('Close', '^VIX')
    cached_vix = df_vix[vix_close_col].iloc[-val_window - 1:-1]
    new_vix = market_data['VIX'].iloc[-val_window - 1:-1]
    assert np.allclose(new_vix.values, cached_vix.values), f"VIX validation failed for {freq_label}"

    # Prepare date range
    start_date = args.start_date
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')

    # 3. Download FRED macroeconomic data
    macro_indicators = [ind.strip() for ind in args.macro_indicators.split(',')]
    print(f"📥 Downloading FRED data for indicators: {', '.join(macro_indicators)}...")
    macro_data = web.DataReader(macro_indicators, 'fred', start_date, end_date, api_key=FRED_API_KEY)

    # 4. Resample and align macro data
    # .ffill() is added to propagate the last known value through weeks without releases
    macro_resampled = macro_data.resample(pd_freq).last().ffill()

    # Combine datasets
    combined_data = {
        "market_data": market_data,
        "macro_data": macro_resampled
    }

    # Determine output filename
    if args.filename:
        final_dataset_filename = os.path.join(args.output_dir, args.filename)
    else:
        if args.freq == "week":
            final_dataset_filename = os.path.join(args.output_dir, "combined_weekly_macro.data")
        else:
            final_dataset_filename = os.path.join(args.output_dir, "combined_monthly_macro.data")

    # Save combined dataset
    print(f"💾 Saving combined dataset to {final_dataset_filename}...")
    with open(final_dataset_filename, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"✅ {freq_label} data combined and saved successfully to: {final_dataset_filename}")

    # Verification load & print date ranges
    with open(final_dataset_filename, 'rb') as f:
        loaded_data = pickle.load(f)

    df_market = loaded_data["market_data"]
    df_macro = loaded_data["macro_data"]

    print(f"📅 Market Data Dates:  {df_market.index[0].strftime('%Y-%m-%d')} :: {df_market.index[-1].strftime('%Y-%m-%d')}")
    print(f"📅 Macro Data Dates:   {df_macro.index[0].strftime('%Y-%m-%d')} :: {df_macro.index[-1].strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    try:
        entry_point(parse_args())
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)