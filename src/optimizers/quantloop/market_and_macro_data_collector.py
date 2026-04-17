#!/usr/bin/env python3
"""
Market & Macro Data Collector
=============================
Downloads S&P 500, VIX, and FRED macroeconomic indicators, resamples them to
monthly frequency, validates against a cached dataset, and saves the combined
result as a pickle file.

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
            "data into a monthly dataset. Outputs a pickle file containing both market "
            "and macro data aligned to month-end."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python script.py --output-dir ./data --start-date 2000-01-01\n"
            "  python script.py -m FEDFUNDS,UNRATE,CPIAUCSL -e 2023-12-31\n"
            "  python script.py -f my_dataset.pkl --vix-method last\n"
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
        "-f", "--filename", type=str, default="combined_monthly_macro.data",
        help="Name of the output pickle file (default: combined_monthly_macro.data)"
    )
    parser.add_argument(
        "-v", "--vix-method", type=str, choices=["mean", "last"], default="mean",
        help="VIX aggregation method: 'mean' for monthly average, 'last' for month-end close (default: mean)"
    )

    return parser.parse_args()


def entry_point(args: argparse.Namespace) -> None:
    """
    Main execution flow: fetches market & macro data, resamples to monthly,
    validates against a cached dataset, and saves the combined result.

    Args:
        args: Parsed command-line arguments from argparse.
    """
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load cached master data for validation
    one_dataset_filename = get_filename_for_dataset("month", older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    df_spx500 = master_data_cache["^GSPC"].sort_index()
    df_vix = master_data_cache["^VIX_MEAN"].sort_index()
    if args.vix_method == "last":
        df_vix = master_data_cache["^VIX"].sort_index()

    # 1. Download market data (S&P 500 and VIX)
    print("📥 Downloading S&P 500 and VIX data from Yahoo Finance...")
    sp500 = yf.download("^GSPC ^VIX", period="max", interval="1d", auto_adjust=False)
    data = sp500['Close'].copy()

    # 2. Resample to monthly frequency
    print("📊 Resampling to monthly frequency...")
    monthly = data['^GSPC'].resample('ME').last().to_frame(name='Close')
    monthly['VIX'] = data['^VIX'].resample('ME').mean()

    # Validation against cached data
    close_col = ('Close', '^GSPC')
    print("🔍 Validating downloaded data against cache...")
    assert 132 == np.count_nonzero(df_spx500[close_col][-133:-1] == monthly['Close'][-133:-1]), "SPX validation failed"
    assert np.allclose(monthly['VIX'].iloc[-133:-1].values, df_vix[('Close', '^VIX')].iloc[-133:-1].values), "VIX validation failed"

    # Prepare date range
    start_date = args.start_date
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')

    # 3. Download FRED macroeconomic data
    macro_indicators = [ind.strip() for ind in args.macro_indicators.split(',')]
    print(f"📥 Downloading FRED data for indicators: {', '.join(macro_indicators)}...")
    macro_data = web.DataReader(macro_indicators, 'fred', start_date, end_date, api_key=FRED_API_KEY)

    # 4. Resample and align macro data
    macro_monthly = macro_data.resample('ME').last()

    # Combine datasets
    combined_data = {
        "market_data": monthly,
        "macro_data": macro_monthly
    }

    # Save combined dataset
    final_dataset_filename = os.path.join(args.output_dir, args.filename)
    print(f"💾 Saving combined dataset to {final_dataset_filename}...")
    with open(final_dataset_filename, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"✅ Données combinées sauvegardées avec succès dans : {final_dataset_filename}")

    # Verification load & print date ranges
    with open(final_dataset_filename, 'rb') as f:
        loaded_data = pickle.load(f)

    df_monthly = loaded_data["market_data"]
    df_macro = loaded_data["macro_data"]

    print(f"📅 Market Data Dates:  {df_monthly.index[0].strftime('%Y-%m-%d')} :: {df_monthly.index[-1].strftime('%Y-%m-%d')}")
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