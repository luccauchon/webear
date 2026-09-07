from constants import FYAHOO__OUTPUTFILENAME_DAY, FYAHOO__OUTPUTFILENAME_MONTH, FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_QUARTER, FYAHOO__OUTPUTFILENAME_YEAR
from constants import BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR, EMAIL_SENDER_WEBEAR, PWD_GOOGLE_API, MOOMOO__PROXY_SPX_FILENAME
from utils import DATASET_AVAILABLE, get_filename_for_dataset
import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm
import re
import pickle
from collections import defaultdict
from fetchers.serialize_fyahoo import realtime as fyahoo_realtime
import yfinance as yf
from datetime import datetime, timedelta
import copy


def _build_cols_dict(ticker):
    """Génère le dictionnaire de clés MultiIndex pour un ticker donné."""
    if ticker == "^VIX":
        return {
            "open_col": ("Open", ticker),
            "high_col": ("High", ticker),
            "low_col": ("Low", ticker),
            "close_col": ("Close", ticker)
        }
    return {
        "open_col": ("Open", ticker),
        "high_col": ("High", ticker),
        "low_col": ("Low", ticker),
        "close_col": ("Close", ticker),
        "volume_col": ("Volume", ticker)
    }


def _is_dataset_heikin_ashi(_dataset_id):
    try:
        # Style : intraday_3min_heikinashi
        type_candle = _dataset_id.split("_")[2]
        if type_candle == "heikinashi":
            return True
    except:
        try:
            # Style :  month_heikinashi
            try:
                type_candle = _dataset_id.split("_")[1]
            except:
                type_candle = _dataset_id.split("-")[1]
            if type_candle == "heikinashi":
                return True
        except:
            pass
        return False


def _get_dataset_timeframe(_dataset_id):
    try:
        if _dataset_id.startswith("intraday"):
            try:
                _n_minutes = int(_dataset_id.split("_")[1][:-3])
                return _n_minutes
            except:
                _n_minutes = int(_dataset_id.split("-")[1][:-3])
                return _n_minutes
        return None
    except:
        return None


def factory_load_data(_dataset_id, _ticker, _args):
    """
       Charge, rééchantillonne et transforme les données financières d'un ticker.

       Parameters:
       -----------
       _dataset_id : str
           Identifiant du jeu de données (ex: "intraday" ou historique).
       _ticker : str
           Le symbole boursier de l'actif (ex: "^GSPC").
       _args : dict
           Dictionnaire de configuration contenant :
           - 'clip_n' (int) : Nombre de lignes à tronquer à la fin du DataFrame (défaut: 0).
           - 'n_minutes' (int) : Taille de la fenêtre de rééchantillonnage en minutes (défaut: 0).
           - 'convert_to_heikin_ashi' (bool) : Activer le calcul Heikin Ashi (défaut: False).
           - 'overwrite_col' (bool) : Écraser les colonnes OHLC d'origine si HA est activé (défaut: False).

       Returns:
       --------
       pd.DataFrame
           Le DataFrame Pandas transformé et trié chronologiquement.
    """
    _reduce_n = _args.get("reduce_n", 0)
    _clip_n = _args.get("clip_n", 0)
    _convert_to_heikin_ashi = _is_dataset_heikin_ashi(_dataset_id)
    _overwrite_col = _args.get("overwrite_col", True)
    _realtime_data = _args.get("realtime", False)
    _get_vix = _args.get("get_vix", False)
    _proxy_spx = _args.get("proxy_spx", False)
    _filter_per_day = _args.get("filter_per_day", [])
    _replace_last_candle = _args.get("new_values_for_last_candles", {})
    _meta_info = ""
    if _proxy_spx:  # Préséance sur realtime
        assert _ticker in ["^GSPC"]
        assert _dataset_id.startswith("intraday")
        df_all = pd.read_parquet(MOOMOO__PROXY_SPX_FILENAME)
        _n_minutes = _get_dataset_timeframe(_dataset_id)
        tuples = [("Adj Close", "^GSPC"),("Close", "^GSPC"),("High", "^GSPC"),("Low", "^GSPC"),("Open", "^GSPC"),("Volume", "^GSPC"),]
        df_main = df_all[tuples].copy()
        if _n_minutes > 1:
            df_main = resample_candles(df=df_main, n_minutes=_n_minutes, ticker=_ticker)
        if _get_vix:
            tuples = [("Adj Close", "^VIX"),("Close", "^VIX"),("High", "^VIX"),("Low", "^VIX"),("Open", "^VIX"),("Volume", "^VIX"),]
            df_vix = df_all[tuples].copy()
            if _n_minutes > 1:
                df_vix = resample_candles(df=df_vix, n_minutes=_n_minutes, ticker="^VIX")
    else:
        def _resample_dataset(df, n_minutes, ticker):
            df = resample_candles_enhanced(df=df, n_minutes=n_minutes, ticker=ticker)
            return df
        if _realtime_data:
            if _dataset_id.startswith("intraday"):
                assert _ticker in ["^GSPC"]
                df_spx_main_local = get_1_minute_df(verbose=False, SPX=True)
                df_vix_main_local = get_1_minute_df(verbose=False, SPX=False, VIX=True)
                df_realtime = factory_df_SPY_SPX_VIX_NDX_at_minutes(period='2d', vix=True, spy=False, spx=True, ndx=False)
                df_spx_main_realtime = df_realtime['spx']
                df_vix_main_realtime = df_realtime['vix']
                # print(f"LOCAL     {df_spx_main_local.index[0]}::{df_spx_main_local.index[-1]}  {len(df_spx_main_local)}")
                # print(f"REALTIME  {df_spx_main_realtime.index[0]}::{df_spx_main_realtime.index[-1]}  {len(df_spx_main_realtime)}")
                def _combine_2df(df_a, df_b):
                    # 1. Fusionner les deux DataFrames (l'un en dessous de l'autre)
                    df_combined = pd.concat([df_a, df_b])

                    # 2. Supprimer les doublons basés sur l'index (la date/heure)
                    # 'keep="last"' conserve la donnée en temps réel la plus récente en cas de chevauchement
                    df_new = df_combined[~df_combined.index.duplicated(keep="last")]

                    # 3. Optionnel : Trier par ordre chronologique pour s'assurer que le flux reste linéaire
                    df_new = df_new.sort_index()
                    return df_new
                df_main = _combine_2df(df_a=df_spx_main_local, df_b=df_spx_main_realtime)
                # print("========================================================")
                # print(f"FUSIONNED {df_main.index[0]}::{df_main.index[-1]}  {len(df_main)}")

                df_vix = _combine_2df(df_a=df_vix_main_local, df_b=df_vix_main_realtime)

                _n_minutes = _get_dataset_timeframe(_dataset_id)
                if _n_minutes > 1:
                    df_main = _resample_dataset(df=df_main, n_minutes=_n_minutes, ticker=_ticker)
                if _get_vix:
                    if _n_minutes > 1:
                        df_vix = resample_candles(df=df_vix, n_minutes=_n_minutes, ticker="^VIX")
            else:
                assert _ticker in ["^GSPC"]
                assert _dataset_id in ["day", "week", "month", "quarter", "year"]
                daily_data_cache, weekly_data_cache, monthly_data_cache, quaterly_data_cache, yearly_data_cache = fyahoo_realtime()
                the_vix = None
                if _dataset_id == "day":
                    df_main = daily_data_cache[_ticker].sort_index().copy()
                    the_vix = daily_data_cache["^VIX"]
                if _dataset_id == "week":
                    df_main = weekly_data_cache[_ticker].sort_index().copy()
                    the_vix = weekly_data_cache["^VIX_MEAN"]
                if _dataset_id == "month":
                    df_main = monthly_data_cache[_ticker].sort_index().copy()
                    the_vix = monthly_data_cache["^VIX_MEAN"]
                if _dataset_id == "quarter":
                    df_main = quaterly_data_cache[_ticker].sort_index().copy()
                    the_vix = quaterly_data_cache["^VIX"]
                if _dataset_id == "year":
                    df_main = yearly_data_cache[_ticker].sort_index().copy()
                    the_vix = yearly_data_cache["^VIX"]
                if _get_vix:
                    df_vix = the_vix.sort_index().copy()
        else:
            if _dataset_id.startswith("intraday"):
                assert _ticker in ["^GSPC", "SPY"]
                _n_minutes = _get_dataset_timeframe(_dataset_id)
                df_main = get_1_minute_df(verbose=False, SPX=_ticker in ["^GSPC"], SPY=_ticker in ["SPY"], NDX=False, VIX=False)
                if _n_minutes > 1:
                    df_main = _resample_dataset(df=df_main, n_minutes=_n_minutes, ticker=_ticker)
                if _get_vix:
                    df_vix = get_1_minute_df(verbose=False, SPX=False, SPY=False, VIX=True)
                    if _n_minutes > 1:
                        df_vix = resample_candles(df=df_vix, n_minutes=_n_minutes, ticker="^VIX")
            else:
                if _dataset_id not in DATASET_AVAILABLE:
                    # Style: day_heikinashi or day_2B
                    # Extra information is already extracted
                    _dataset_id, _meta_info = _dataset_id.split("_")
                    assert _dataset_id in DATASET_AVAILABLE
                with open(get_filename_for_dataset(_dataset_id, older_dataset=None), 'rb') as f:
                    _master_data_cache = pickle.load(f)
                assert _master_data_cache is not None
                df_main = _master_data_cache[_ticker].sort_index().copy()
                if _get_vix:
                    df_vix = _master_data_cache["^VIX"].sort_index().copy()
    assert df_main is not None
    if _convert_to_heikin_ashi:
        _tmp_n1 = len(df_main.dropna())
        df_main = convert_to_heikin_ashi(df=df_main, ticker=_ticker, overwrite=True)
        assert _tmp_n1 == len(df_main.dropna())
    if _clip_n > 0:
        if _realtime_data:
            print(f"\033[1m[WARNING] <<_realtime_data>> is ON but <<clip_n>> is also set to {_clip_n} : Realtime data might have been discarded.\033[0m")
        df_main = df_main.iloc[:-_clip_n]
    if _reduce_n > 0:
        df_main = df_main.iloc[_reduce_n:]
    if len(_filter_per_day) > 0:
        # Le vendredi correspond au jour de la semaine 4 (Lundi = 0, Dimanche = 6)
        assert len([n for n in _filter_per_day if 0 <= n <= 6]) == len(_filter_per_day)
        df_main = df_main[df_main.index.dayofweek.isin(_filter_per_day)]
        if _get_vix:
            df_vix = df_vix[df_vix.index.dayofweek.isin(_filter_per_day)]
    if bool(re.match(r"^[1-9]\d?B$", _meta_info)):
        n_bars = int(_meta_info[:-1])
        df_main = resample_macro_candles(df=df_main, n=n_bars, timeframe=_dataset_id, ticker=_ticker)
        if _get_vix:
            df_vix = resample_macro_candles(df=df_main, n=n_bars, timeframe=_dataset_id, ticker="^VIX")
    if len(_replace_last_candle) > 0:
        # # Cibler la dernière ligne (-1) pour chaque colonne spécifique
        # df.loc[df.index[-1], ("Open", ticker)] = new_open
        # df.loc[df.index[-1], ("High", ticker)] = new_high
        # df.loc[df.index[-1], ("Low", ticker)] = new_low
        # df.loc[df.index[-1], ("Close", ticker)] = new_close
        # df.loc[df.index[-1], ("Volume", ticker)] = new_volume
        pass
    ###########################################################################
    # Retour des valeurs
    ###########################################################################
    if _get_vix:
        return df_main.copy(), df_vix.copy()
    return df_main.copy()


def convert_to_heikin_ashi(df, ticker, overwrite=False):
    """
    Compute Heikin Ashi candles.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    ticker : str
        Ticker used by _build_cols_dict().
    overwrite : bool, default=False
        If True, replace the original OHLC columns.
        Otherwise, create new *_HA columns.

    Returns
    -------
    pd.DataFrame
    """
    df_ha = df.copy()
    cols = _build_cols_dict(ticker)

    # Original OHLC as numpy arrays (fast)
    o = df_ha[cols["open_col"]].to_numpy(dtype=float)
    h = df_ha[cols["high_col"]].to_numpy(dtype=float)
    l = df_ha[cols["low_col"]].to_numpy(dtype=float)
    c = df_ha[cols["close_col"]].to_numpy(dtype=float)

    n = len(df_ha)

    ha_open = np.empty(n, dtype=float)
    ha_close = (o + h + l + c) / 4.0

    # First candle
    ha_open[0] = (o[0] + c[0]) / 2.0

    # Recursive computation
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    if overwrite:
        df_ha[cols["open_col"]] = ha_open
        df_ha[cols["high_col"]] = ha_high
        df_ha[cols["low_col"]] = ha_low
        df_ha[cols["close_col"]] = ha_close
    else:
        df_ha[("Open_HA", ticker)] = ha_open
        df_ha[("High_HA", ticker)] = ha_high
        df_ha[("Low_HA", ticker)] = ha_low
        df_ha[("Close_HA", ticker)] = ha_close

    return df_ha


def resample_macro_candles(df, n, timeframe, ticker):
    """
    Convertit un DataFrame de bougies (ex: Day, Week) en bougies de taille 'n * timeframe'.

    Paramètres:
    -----------
    df : pandas.DataFrame
        DataFrame avec un DateTimeIndex.
    n : int
        Le multiplicateur (ex: 3 pour 3 jours, 4 pour 4 semaines).
    timeframe : str
        Le type de bougie d'origine/cible: 'day', 'week', 'month', 'quarter', 'annual'.
    ticker : str
        Le symbole boursier pour mapper les colonnes.
    """
    cols = _build_cols_dict(ticker)

    # Définition du dictionnaire d'agrégation standard
    agg_dict = {
        cols["open_col"]: "first",
        cols["high_col"]: "max",
        cols["low_col"]: "min",
        cols["close_col"]: "last"
    }
    if ticker != "^VIX" and "volume_col" in cols:
        agg_dict[cols["volume_col"]] = "sum"

    # Mappage des timeframes vers les codes d'ancrage Pandas
    tf_mapping = {
        "day": "D",
        "week": "W",  # Aligné sur le dimanche par défaut ou fin de semaine
        "month": "ME",  # Month End (fin de mois)
        "quarter": "QE",  # Quarter End (fin de trimestre)
        "annual": "YE"  # Year End (fin d'année)
    }

    tf_lower = timeframe.lower()
    if tf_lower not in tf_mapping:
        raise ValueError("timeframe doit être: 'day', 'week', 'month', 'quarter' ou 'annual'")

    # Construction de la règle (ex: '3D', '4W', '2ME')
    rule = f"{n}{tf_mapping[tf_lower]}"

    # Rééchantillonnage et agrégation
    df_resampled = df.resample(rule).agg(agg_dict)

    # Nettoyage des périodes sans données (ex: weekends ou jours fériés)
    df_resampled = df_resampled.dropna(subset=[cols["open_col"]])

    return df_resampled.copy()


def resample_candles_enhanced(
    df,
    n_minutes,
    ticker,
    session_start="09:30:00",
    session_end="16:00:00",
    label="left",
    include_close=False,
):
    """
    Resample 1-minute candles to n-minute candles on a per-day basis,
    forcing the last candle of each day to end at session_end (default 16:00).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex containing 1-minute candles.
    n_minutes : int
        Target candle size in minutes.
    ticker : str
        Ticker used to build the column names via _build_cols_dict().
    session_start : str or pd.Timedelta, default "09:30:00"
        Regular session start time.
    session_end : str or pd.Timedelta, default "16:00:00"
        Regular session end time.
    label : {"left", "right"}, default "left"
        - "left": the resulting index is the candle OPEN time.
                  This matches your current behaviour:
                  for 15m, the last candle is labelled 15:45 and closes at 16:00.
        - "right": the resulting index is the candle CLOSE time.
                  The last candle of each day will be labelled 16:00.
    include_close : bool, default False
        - False assumes your 1-minute bars are start-labelled and the last
          regular bar is 15:59, so 16:00 is excluded.
        - True includes a 16:00 timestamp in the last candle if present.

    Returns
    -------
    pd.DataFrame
        Resampled n-minute candles.
    """

    # Validate n_minutes
    if n_minutes <= 0:
        raise ValueError("n_minutes must be positive.")
    if int(n_minutes) != n_minutes:
        raise ValueError("n_minutes must be an integer number of minutes.")
    n_minutes = int(n_minutes)

    if label not in ("left", "right"):
        raise ValueError("label must be either 'left' or 'right'.")

    session_start = pd.Timedelta(session_start)
    session_end = pd.Timedelta(session_end)

    if session_start >= session_end:
        raise ValueError("session_start must be before session_end.")

    cols = _build_cols_dict(ticker)

    if ticker == "^VIX":
        agg_dict = {
            cols["open_col"]: "first",
            cols["high_col"]: "max",
            cols["low_col"]: "min",
            cols["close_col"]: "last",
        }
    else:
        agg_dict = {
            cols["open_col"]: "first",
            cols["high_col"]: "max",
            cols["low_col"]: "min",
            cols["close_col"]: "last",
            cols["volume_col"]: "sum",
        }

    if df.empty:
        return pd.DataFrame(columns=list(agg_dict.keys())).rename_axis(df.index.name)

    df = df.sort_index()
    td = pd.Timedelta(minutes=n_minutes)

    def _session_ts(day_ts, delta):
        """
        Build a wall-clock session timestamp for the given day.

        This is useful if the index is timezone-aware, especially around DST.
        For naive indexes, it simply returns midnight + delta.
        """
        naive = pd.Timestamp(day_ts.date()) + delta

        tz = getattr(day_ts, "tz", None)
        if tz is None:
            tz = getattr(day_ts, "tzinfo", None)

        if tz is None:
            return naive

        # Try to use a timezone name if available.
        tz_name = getattr(tz, "zone", None) or getattr(tz, "key", None) or str(tz)

        try:
            return naive.tz_localize(tz_name)
        except Exception:
            # Fallback: absolute arithmetic. Good enough for fixed offsets / non-DST cases.
            return day_ts + delta

    parts = []

    # Process each trading day independently.
    for day, g in df.groupby(df.index.normalize()):

        start_ts = _session_ts(day, session_start)
        end_ts = _session_ts(day, session_end)

        # Keep only regular session data.
        if include_close:
            g = g[(g.index >= start_ts) & (g.index <= end_ts)]
        else:
            g = g[(g.index >= start_ts) & (g.index < end_ts)]

        if g.empty:
            continue

        # Build candle edges backwards from the session close.
        # Example for n_minutes=13:
        # edges = [..., 15:34, 15:47, 16:00]
        edges = [end_ts]
        while edges[-1] > start_ts:
            edges.append(edges[-1] - td)

        # Put edges in ascending order.
        edges = list(reversed(edges))

        # If the backward grid starts before the open, replace that edge by the open.
        # This makes the first candle partial when n_minutes does not divide the session.
        if edges[0] < start_ts:
            edges[0] = start_ts

        edges = pd.DatetimeIndex(edges).drop_duplicates().sort_values()

        if len(edges) < 2:
            continue

        # Assign each 1-minute timestamp to its candle interval:
        # interval i is [edges[i], edges[i+1])
        pos = edges.searchsorted(g.index, side="right") - 1

        # If include_close=True and there is a 16:00 timestamp,
        # clip puts it into the last candle instead of creating a new one.
        pos = np.clip(pos, 0, len(edges) - 2)

        if label == "left":
            # Candle index = open/start time of the candle.
            group_labels = edges[pos]
        else:
            # Candle index = close/end time of the candle.
            group_labels = edges[pos + 1]

        part = g.groupby(group_labels).agg(agg_dict)
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=list(agg_dict.keys())).rename_axis(df.index.name)

    df_resampled = pd.concat(parts).sort_index()
    df_resampled.index.name = df.index.name

    # Remove empty candles, same idea as your original dropna().
    df_resampled = df_resampled.dropna(subset=[cols["open_col"]])

    return df_resampled.copy()


def resample_candles(df, n_minutes, ticker):
    """
    Convertit un DataFrame de bougies 1-minute en bougies de n-minutes
    en commençant l'alignement à 09h30.
    """
    cols = _build_cols_dict(ticker)
    if ticker == "^VIX":
        agg_dict = {
            cols["open_col"]: "first",
            cols["high_col"]: "max",
            cols["low_col"]: "min",
            cols["close_col"]: "last"
        }
    else:
        agg_dict = {
            cols["open_col"]: "first",
            cols["high_col"]: "max",
            cols["low_col"]: "min",
            cols["close_col"]: "last",
            cols["volume_col"]: "sum"}

    rule = f"{n_minutes}min"

    df_resampled = df.resample(
        rule,
        origin='start_day',
        offset=pd.Timedelta(hours=9, minutes=30)
    ).agg(agg_dict)

    # Supprime les barres vides (ex: avant 9h30 ou après la fermeture)
    df_resampled = df_resampled.dropna(subset=[cols["open_col"]])

    return df_resampled.copy()


def get_1_minute_df(SPY=False, NDX=False, VIX=False, SPX=True, only_active_trading_hours=True, verbose=False):
    # Group files by date
    files_by_date = defaultdict(list)

    for file in glob.glob(os.path.join(BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR, '*.pkl')):
        filename = os.path.basename(file)
        if "__SPY" in filename and not SPY:
            continue
        if "__NDX" in filename and not NDX:
            continue
        if "__VIX" in filename and not VIX:
            continue
        if "__SPX" in filename and not SPX:
            continue
        date = filename.split('_')[0]  # "2026-03-31"
        files_by_date[date].append(file)

    merged_all = []

    # Merge all symbols per day
    trq = tqdm(files_by_date.items()) if verbose else files_by_date.items()
    for date, files in trq:
        dfs = [pd.read_pickle(f) for f in files]

        # Outer merge all DataFrames for that day
        merged_day = dfs[0]
        for df in dfs[1:]:
            merged_day = pd.merge(
                merged_day, df,
                left_index=True,
                right_index=True,
                how='outer'  # IMPORTANT
            )

        merged_all.append(merged_day)

    # Combine all days
    merged_df = pd.concat(merged_all)

    # Remove duplicates just in case
    merged_df = merged_df[~merged_df.index.duplicated()]
    merged_df = merged_df.sort_index()
    if only_active_trading_hours:
        merged_df = merged_df.between_time("09:30", "16:00").dropna()
    if verbose:
        print(f"{merged_df.index[0]} → {merged_df.index[-1]} | rows: {len(merged_df)}")
    return merged_df


def get_df_SPY_and_VIX_virgin_at_minutes():
    _tmp = factory_df_SPY_SPX_VIX_NDX_at_minutes(period='max', vix=True, spy=True, spx=True, ndx=True)
    return _tmp['spy'], _tmp['spx'], _tmp['vix'], _tmp['ndx']


def factory_df_SPY_SPX_VIX_NDX_at_minutes(period='max', vix=False, spy=False, spx=True, ndx=False):
    df_spy, df_spx, df_vix, df_ndx = None, None, None, None
    if vix:
        df_vix       = yf.download("^VIX", period=period, interval='1m', auto_adjust=False, progress=False)
        df_vix       = df_vix.drop("Volume", axis=1)
        df_vix.index = df_vix.index.tz_convert('US/Eastern')
    if spy:
        df_spy       = yf.download("SPY", period=period, interval='1m', auto_adjust=False, progress=False)
        df_spy.index = df_spy.index.tz_convert('US/Eastern')
    if spx:
        df_spx       = yf.download("^GSPC", period=period, interval='1m', auto_adjust=False, progress=False)
        df_spx.index = df_spx.index.tz_convert('US/Eastern')
    if ndx:
        df_ndx       = yf.download("^NDX", period=period, interval='1m', auto_adjust=False, progress=False)
        df_ndx.index = df_ndx.index.tz_convert('US/Eastern')

    return  {'spy': df_spy, 'spx': df_spx, 'vix': df_vix, 'ndx': df_ndx}


def get_df_SPY_and_VIX_virgin_at_30minutes():
    # Calculate start date (30 days ago)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

    df_vix       = yf.download("^VIX", start=start_date, end=end_date, interval='30m', auto_adjust=False, progress=False)
    df_vix       = df_vix.drop("Volume", axis=1)
    df_vix.index = df_vix.index.tz_convert('US/Eastern')

    df_spy       = yf.download("SPY", start=start_date, end=end_date, interval='30m', auto_adjust=False, progress=False)
    df_spy.index = df_spy.index.tz_convert('US/Eastern')

    df_spx       = yf.download("^GSPC", start=start_date, end=end_date, interval='30m', auto_adjust=False, progress=False)
    df_spx.index = df_spx.index.tz_convert('US/Eastern')

    df_ndx       = yf.download("^NDX", start=start_date, end=end_date, interval='30m', auto_adjust=False, progress=False)
    df_ndx.index = df_ndx.index.tz_convert('US/Eastern')

    return  df_spy, df_spx, df_vix, df_ndx


def get_df_SPY_and_VIX(interval="1d", add_moving_averages=True, _window_sizes=(2,3,4,5)):
    df_vix    = yf.download("^VIX", period="max", interval="1d", auto_adjust=False)
    df_vix    = df_vix.drop("Volume", axis=1)
    df_spy    = yf.download("SPY", period="max", interval="1d", auto_adjust=False)
    merged_df = pd.merge(df_spy, df_vix, on='Date', how='left')
    assert 0 == np.sum(merged_df.isna().sum().values)
    if interval in ['1mo', '1wk']:
        # Define the list of symbols
        symbols = ['SPY', '^VIX']
        # Define the aggregation functions for each column type
        agg_funcs = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',}
        agg_dict = {(col, symbol): agg_funcs[col] for symbol in symbols for col in agg_funcs}
        df_1jil = None
        if interval == '1wk':
            df_1jil = copy.deepcopy(merged_df.resample('W-FRI').agg(agg_dict))
            # Resample the volume data into 5-minute mean volumes
            volume_candles = merged_df['Volume'].resample('W-FRI').mean()
            # Add the volume column to the candles DataFrame
            df_1jil['Volume'] = volume_candles
            if merged_df.index[-1].weekday() < 5:  # 5 represents Saturday
                df_1jil = df_1jil.drop(df_1jil.index[-1])
        if interval == '1mo':
            df_1jil = copy.deepcopy(merged_df.resample('ME').agg(agg_dict))
            # Resample the volume data into monthly mean volumes
            volume_candles = merged_df['Volume'].resample('ME').mean()
            # Add the volume column to the candles DataFrame
            df_1jil['Volume'] = volume_candles
            if merged_df.index[-1].day < 28:
                df_1jil = df_1jil.drop(df_1jil.index[-1])
        assert 0 == np.sum(df_1jil.isna().sum().values)
        merged_df = copy.deepcopy(df_1jil)
    merged_df['day_of_week']  = merged_df.index.dayofweek + 1
    merged_df['week_of_year'] = merged_df.index.isocalendar().week + 1
    merged_df['unique_week']  = merged_df.index.year * 1000 + merged_df['week_of_year']
    merged_df['month_of_year'] = merged_df.index.month
    #merged_df[('Close_direction', 'SPY')]  = merged_df.apply(lambda row: 1 if row[('Close', 'SPY')] > row[('Open', 'SPY')] else -1, axis=1)
    #merged_df[('Close_direction', '^VIX')] = merged_df.apply(lambda row: 1 if row[('Close', '^VIX')] > row[('Open', '^VIX')] else -1, axis=1)

    if add_moving_averages:
        for window_size in _window_sizes:  # Define the window size for the moving average
            do_ma_on_those = [('Close', 'SPY'), ('High', 'SPY'), ('Low', 'SPY'), ('Open', 'SPY'), ('Volume', 'SPY'),
                              ('Close', '^VIX'), ('High', '^VIX'), ('Low', '^VIX'), ('Open', '^VIX')]
            new_cols = []
            for col in merged_df.columns:
                if not col in do_ma_on_those:
                    continue
                col_title = (col[0] + f'_MA{window_size}', col[1])
                merged_df[col_title] = merged_df[col].rolling(window=window_size, center=True).mean()
                merged_df[col_title] = merged_df[col_title].shift(window_size // 2)
                new_cols.append(col_title)

                col_title = (col[0] + f'_EMA{window_size}', col[1])
                merged_df[col_title] = merged_df[col].ewm(span=window_size, adjust=False).mean()
                new_cols.append(col_title)

            #print(new_cols)
        merged_df = merged_df.dropna()

    merged_df = merged_df.sort_index(ascending=True)

    # merged_df[['column1', 'column2']] = merged_df[['column1', 'column2']].astype(int)
    # float_cols = merged_df.select_dtypes(include=['float64']).columns
    # merged_df[float_cols] = merged_df[float_cols].astype(int)

    return copy.deepcopy(merged_df), f'spy_vix_multicol_reverse_rc1__direction_at_{interval}'