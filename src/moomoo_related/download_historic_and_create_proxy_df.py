import os

import pandas as pd
import pytz
import time
from datetime import datetime, timedelta
from moomoo import OpenQuoteContext, KLType, AuType, RET_OK
from utils import is_weekday
import os


def entry():
    # Connexion à OpenD
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

    # Remplacement par les proxies hautement liquides (ETFs)
    symbols = ['US.SPY', 'US.VIXY']
    kline_type = KLType.K_1M
    autype = AuType.QFQ

    # Variables temporelles
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 8)
    delta_step = timedelta(days=1)
    symbol__2__filename = {}
    for symbol in symbols:
        clean_name = "US_SPX_PROXY" if symbol == 'US.SPY' else symbol.replace('.', '_')
        filename = f"{clean_name}_1min_RTH.csv"
        if os.path.exists(filename):
            continue
        print(f"\n==========================================")
        print(f"Début du téléchargement pour {symbol}...")
        print(f"==========================================")

        df_list = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + delta_step, end_date)

            # Vérification si la période actuelle concerne un jour de la semaine (Lundi-Vendredi)
            if not is_weekday(current_start):
                print(f"[{symbol}] Skip week-end : {current_start.strftime('%Y-%m-%d')}")
                current_start = current_end
                continue

            start_str = current_start.strftime("%Y-%m-%d %H:%M:%S")
            end_str = current_end.strftime("%Y-%m-%d %H:%M:%S")

            ret, data, page_req_key = quote_ctx.request_history_kline(
                code=symbol,
                start=start_str,
                end=end_str,
                ktype=kline_type,
                autype=autype,
                max_count=1000
            )

            if ret == RET_OK:
                if not data.empty:
                    df_list.append(data)
                    print(f"[{symbol}] OK : {start_str[:10]} -> {end_str[:10]} ({len(data)} lignes)")
            else:
                print(f"[{symbol}] Erreur {start_str[:10]} : {data}")

            current_start = current_end
            time.sleep(0.6)  # Pause rate-limiting Moomoo

        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            df.drop_duplicates(subset=['time_key'], inplace=True)

            # 1. Ajustement mathématique pour coller au S&P 500 (Uniquement pour SPY)
            if symbol == 'US.SPY':
                print(f"\n[Ajustement] Application du multiplicateur dynamique SPY -> SPX...")
                multiplier = 10.0

                price_cols = ['open', 'close', 'high', 'low', 'last_close']
                for col in price_cols:
                    if col in df.columns:
                        df[col] = df[col] * multiplier

            # Conversion du fuseau horaire vers New York
            df['time_dt'] = pd.to_datetime(df['time_key'])
            if df['time_dt'].dt.tz is None:
                df['time_dt'] = df['time_dt'].dt.tz_localize('America/New_York')
            else:
                df['time_dt'] = df['time_dt'].dt.tz_convert('America/New_York')

            # Filtrage RTH (09:30 à 16:00 EST)
            time_only = df['time_dt'].dt.time
            rth_start = datetime.strptime("09:30:00", "%H:%M:%S").time()
            rth_end = datetime.strptime("16:00:00", "%H:%M:%S").time()

            df_rth = df[(time_only >= rth_start) & (time_only <= rth_end)].copy()
            df_rth.drop(columns=['time_dt'], inplace=True)
            df_rth.sort_values(by=['time_key'], inplace=True)

            # Sauvegarde CSV avec nom explicite
            df_rth.to_csv(filename, index=False)
            print(f"Terminé pour {symbol} ! {len(df_rth)} bougies RTH enregistrées dans {filename}")
            symbol__2__filename.update({symbol: filename})
        else:
            print(f"Aucune donnée récupérée pour {symbol}.")

    quote_ctx.close()

    tuples = [
        ("Adj Close", "^GSPC"),
        ("Close", "^GSPC"),
        ("High", "^GSPC"),
        ("Low", "^GSPC"),
        ("Open", "^GSPC"),
        ("Volume", "^GSPC"),
    ]
    df_moomoo_spx = pd.read_csv(symbol__2__filename['US.SPY'], parse_dates=["time_key"])
    df_moomoo_spx.set_index("time_key", inplace=True)
    df_moomoo_spx = df_moomoo_spx[['close', 'close', 'high', 'low', 'open', 'volume', ]]
    df_moomoo_spx = df_moomoo_spx.rename_axis('Datetime')
    df_moomoo_spx.index = pd.to_datetime(df_moomoo_spx.index)
    df_moomoo_spx.index = df_moomoo_spx.index.tz_localize('America/New_York')
    new_columns = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
    df_moomoo_spx.columns = new_columns
    df_moomoo_spx = df_moomoo_spx.sort_index()

    tuples = [
        ("Adj Close", "^VIX"),
        ("Close", "^VIX"),
        ("High", "^VIX"),
        ("Low", "^VIX"),
        ("Open", "^VIX"),
        ("Volume", "^VIX"),
    ]
    df_moomoo_vix = pd.read_csv(symbol__2__filename['US.VIXY'], parse_dates=["time_key"])
    df_moomoo_vix.set_index("time_key", inplace=True)
    df_moomoo_vix = df_moomoo_vix[['close', 'close', 'high', 'low', 'open', 'volume', ]]
    df_moomoo_vix = df_moomoo_vix.rename_axis('Datetime')
    df_moomoo_vix.index = pd.to_datetime(df_moomoo_vix.index)
    df_moomoo_vix.index = df_moomoo_vix.index.tz_localize('America/New_York')
    new_columns = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
    df_moomoo_vix.columns = new_columns
    df_moomoo_vix = df_moomoo_vix.sort_index()

    # 2. Jointure "left" sur les index + ffill automatique sur le VIX
    df_fusionne = df_moomoo_spx.join(df_moomoo_vix, how="left").ffill()

    print(df_fusionne.head())
    print(df_fusionne.tail())

    df_fusionne.to_parquet("df_proxy_spx_moomoo.parquet")


if __name__ == "__main__":
    entry()
