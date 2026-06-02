import pandas as pd
from datetime import datetime, timedelta
from futu import OpenQuoteContext, KLType, AuType, RET_OK
import time
import os


def download_spy_slow(start_year, end_year, host='127.0.0.1', port=11111,
                      sleep_per_page=4.0, sleep_per_year=10.0, max_retries=3):
    """
    Downloads SPY historical K-line data chunked by year with conservative delays.
    Automatically handles pagination, retries, and incremental CSV saving.
    """
    quote_ctx = OpenQuoteContext(host=host, port=port)
    symbol = 'US.SPY'
    output_file = 'SPY_1min_history.csv'
    file_exists = os.path.exists(output_file)

    try:
        for year in range(start_year, end_year + 1):
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            print(f"\n📅 Processing {year} ({year_start} to {year_end})...")

            all_klines_year = []
            page_req_key = None
            max_pages = 500
            page_count = 0

            while page_count < max_pages:
                # Retry logic for transient API errors
                success = False
                for retry in range(max_retries):
                    ret, data, page_req_key = quote_ctx.request_history_kline(
                        code=symbol,
                        start=year_start,
                        end=year_end,
                        ktype=KLType.K_1M,  # Change to KLType.K_DAY for 40-year coverage
                        autype=AuType.QFQ,
                        max_count=1000,
                        page_req_key=page_req_key
                    )

                    if ret == RET_OK:
                        success = True
                        break
                    print(f"   ❌ API Error (attempt {retry + 1}/{max_retries}): {data}")
                    time.sleep(sleep_per_page * (retry + 1))

                if not success:
                    print("   ⛔ Max retries reached. Skipping to next year.")
                    break

                # If first page is empty, likely no 1-min data exists for this year
                if data.empty and page_count == 0:
                    print(f"   ⚠️ No 1-min data available for {year}. Stopping year loop.")
                    break
                elif data.empty:
                    print("   ✅ No more pages for this year.")
                    break

                all_klines_year.append(data)
                total = sum(len(d) for d in all_klines_year)
                print(f"   📄 Page {page_count + 1}: {len(data)} candles | Year total: {total}")

                if page_req_key is None:
                    break

                time.sleep(sleep_per_page)
                page_count += 1

            # Save yearly chunk incrementally
            if all_klines_year:
                df_year = pd.concat(all_klines_year, ignore_index=True)
                df_year.to_csv(output_file, mode='a', header=not file_exists, index=False)
                file_exists = True
                print(f"   💾 Saved {len(df_year)} rows for {year} to {output_file}")
            else:
                print(f"   ℹ️ No data retrieved for {year}.")

            # Cooling down between years to respect rate limits
            if year < end_year:
                print(f"   ⏳ Cooling down {sleep_per_year}s before next year...")
                time.sleep(sleep_per_year)

    except KeyboardInterrupt:
        print("\n⏸️ Interrupted by user. Partial data saved.")
    finally:
        quote_ctx.close()

    print(f"\n🎉 Finished! Check {output_file}")


if __name__ == '__main__':
    # 🕰️ Set your range (40 years example: 1984-2024)
    START_YEAR = 2020
    END_YEAR = 2026

    print(f"📥 Downloading SPY 1-min data from {START_YEAR} to {END_YEAR}...")
    print("⚠️ Note: Moomoo typically limits 1-min data to ~1-2 years. Older years will be skipped automatically.")

    download_spy_slow(
        start_year=START_YEAR,
        end_year=END_YEAR,
        sleep_per_page=4.0,  # Seconds between pagination requests
        sleep_per_year=10.0,  # Seconds between yearly chunks
        max_retries=3
    )