import yfinance as yf
import pandas as pd
from utils import get_df_SPY_and_VIX_virgin_at_minutes
import os
import schedule
import time


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


def get_spx_futures(years, interval):
    """
    Fetch CONTINUOUS futures data (ES=F) aligned with SPX Index (^GSPC),
    VIX (^VIX), SPY ETF, and GPSC.

    Backbone Logic:
    - Futures data is NEVER dropped (24/5 backbone).
    - Index, VIX, SPY, and GPSC have NaNs outside market hours.

    Parameters:
        years (int): Years of historical data (default=1)
        interval (str): Data interval ('1h', '30m', etc.)

    Returns:
        pd.DataFrame:
          - SPX_Futures: COMPLETE series (all trading hours)
          - SPX_Index: Values ONLY during market hours (NaN otherwise)
          - VIX: Values ONLY during market hours (NaN otherwise)
          - SPY: Values ONLY during market hours (NaN otherwise)
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * years)
    chunk_size = timedelta(days=55)  # Stay under Yahoo's intraday limit
    current_start = start_date

    futures_chunks = []
    index_chunks = []
    vix_chunks = []
    spy_chunks = []  # New

    chunk_num = 0

    print(f"🚀 Downloading {years} years of HOURLY data (Futures 24/5 | Assets Market Hours)")
    print(f"   Period: {start_date.date()} to {end_date.date()} | Interval: {interval}\n")
    print(f"   Assets: ES=F, ^GSPC, ^VIX, SPY\n")

    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)

        # --- 1. DOWNLOAD FUTURES (PRIORITY: NEVER SKIP) ---
        fut_success = False
        for attempt in range(2):
            try:
                fut = yf.download(
                    'ES=F',
                    start=current_start,
                    end=current_end,
                    interval=interval,
                    auto_adjust=True,
                    timeout=15,
                    progress=False
                )
                if not fut.empty and 'Close' in fut.columns:
                    if fut.index.tz is None:
                        fut.index = fut.index.tz_localize('US/Eastern').tz_convert('UTC')
                    else:
                        fut.index = fut.index.tz_convert('UTC')
                    futures_chunks.append(fut[['Close']].rename(columns={'Close': 'SPX_Futures'}))
                    fut_success = True
                    break
            except Exception as e:
                if attempt == 1:
                    print(f"⚠️  Futures chunk failed ({current_start.date()}): {str(e)[:60]}")
            time.sleep(0.8)

        # --- 2. DOWNLOAD INDEX (^GSPC) ---
        try:
            idx = yf.download(
                '^GSPC',
                start=current_start,
                end=current_end,
                interval=interval,
                auto_adjust=True,
                timeout=15,
                progress=False
            )
            if not idx.empty and 'Close' in idx.columns:
                if idx.index.tz is None:
                    idx.index = idx.index.tz_localize('US/Eastern').tz_convert('UTC')
                else:
                    idx.index = idx.index.tz_convert('UTC')
                index_chunks.append(idx[['Close']].rename(columns={'Close': 'SPX_Index'}))
        except Exception as e:
            print(f"ℹ️  Index chunk skipped ({current_start.date()}): {str(e)[:50]}")

        # --- 3. DOWNLOAD VIX (^VIX) ---
        try:
            vix = yf.download(
                '^VIX',
                start=current_start,
                end=current_end,
                interval=interval,
                auto_adjust=True,
                timeout=15,
                progress=False
            )
            if not vix.empty and 'Close' in vix.columns:
                if vix.index.tz is None:
                    vix.index = vix.index.tz_localize('US/Eastern').tz_convert('UTC')
                else:
                    vix.index = vix.index.tz_convert('UTC')
                vix_chunks.append(vix[['Close']].rename(columns={'Close': 'VIX'}))
        except Exception as e:
            print(f"ℹ️  VIX chunk skipped ({current_start.date()}): {str(e)[:50]}")

        # --- 4. DOWNLOAD SPY (New) ---
        try:
            spy = yf.download(
                'SPY',
                start=current_start,
                end=current_end,
                interval=interval,
                auto_adjust=True,
                timeout=15,
                progress=False
            )
            if not spy.empty and 'Close' in spy.columns:
                if spy.index.tz is None:
                    spy.index = spy.index.tz_localize('US/Eastern').tz_convert('UTC')
                else:
                    spy.index = spy.index.tz_convert('UTC')
                spy_chunks.append(spy[['Close']].rename(columns={'Close': 'SPY'}))
        except Exception as e:
            print(f"ℹ️  SPY chunk skipped ({current_start.date()}): {str(e)[:50]}")

        time.sleep(1.0)  # Rate limiting safety
        chunk_num += 1
        if fut_success:
            print(f"✓ Chunk {chunk_num}: {current_start.date()} → {current_end.date()} | Futures: {len(futures_chunks[-1])} pts")
        current_start = current_end

    # ===== COMBINE WITH FUTURES AS BACKBONE =====
    if not futures_chunks:
        raise RuntimeError("No futures data retrieved. Check network/Yahoo status.")

    # 1. Merge ALL futures chunks (priority backbone)
    df_futures = pd.concat(futures_chunks).sort_index()
    df_futures = df_futures[~df_futures.index.duplicated(keep='first')]

    # 2. Merge Market Hour Assets
    def merge_market_asset(base_df, chunks, col_name, direction):
        if not chunks:
            base_df[col_name] = np.nan
            return base_df

        df_asset = pd.concat(chunks).sort_index()
        df_asset = df_asset[~df_asset.index.duplicated(keep='first')]

        return pd.merge_asof(
            base_df.sort_index(),
            df_asset.sort_index(),
            left_index=True,
            right_index=True,
            direction=direction
        )

    # Start with Futures
    df_merged = df_futures

    # Merge Index
    df_merged = merge_market_asset(df_merged, index_chunks, 'SPX_Index', direction='backward')

    # Merge VIX
    df_merged = merge_market_asset(df_merged, vix_chunks, 'VIX', direction='backward')

    # Merge SPY
    df_merged = merge_market_asset(df_merged, spy_chunks, 'SPY', direction='backward')

    # Convert final index to New York time for readability
    df_merged = df_merged.tz_convert("America/New_York")

    return df_merged


def dump_spx_futures_to_disk(output_dir=r"D:\Finance\data\futures"):
    _today = pd.Timestamp.now().date()
    print(f"Running the extraction for Futures @minutes for  {_today}")
    df_merged = get_spx_futures(years=2,interval='1h')
    # Get current date in YYYY_MM_DD format
    current_date = datetime.now().strftime("%Y_%m_%d")

    # Save intermediate with date in filename
    os.makedirs(output_dir, exist_ok=True)
    df_merged.to_csv(os.path.join(output_dir, f"spx_es_multi_asset_merged_{current_date}.csv"))


if __name__ == '__main__':
    dump_spx_futures_to_disk()
    schedule.every().day.at("18:00").do(dump_spx_futures_to_disk)  # Run every day at 17:00

    while True:
        schedule.run_pending()
        time.sleep(1)