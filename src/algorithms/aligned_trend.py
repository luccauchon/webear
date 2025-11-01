import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class EMATrendAnalyzer:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def _print(self, *args, **kwargs):
        """Helper method to conditionally print based on self.verbose."""
        if self.verbose:
            print(*args, **kwargs)

    def get_4h_data(self, df_1h, ticker_name):
        """Resample 1-hour data to 4-hour OHLC."""
        df_4h = df_1h.resample('4h').agg({
            ('Open', ticker_name): 'first',
            ('High', ticker_name): 'max',
            ('Low', ticker_name): 'min',
            ('Close', ticker_name): 'last',
            ('Volume', ticker_name): 'sum'
        }).dropna()
        return df_4h

    def add_sma_and_signals(self, df, ticker_name, periods=[20, 50, 200]):
        """Add SMA columns and signals for trend direction and price vs SMA."""
        for period in periods:
            sma_col = f'SMA_{period}'
            df[sma_col] = df[('Close', ticker_name)].rolling(window=period).mean()
            # Trend: True if SMA is rising (current > previous)
            df[f'SMA_{period}_Up'] = df[sma_col] > df[sma_col].shift(1)
            # Price vs SMA: True if Close > SMA
            df[f'Close_Above_SMA_{period}'] = df[('Close', ticker_name)] > df[sma_col]
        return df

    def add_ema_and_signals(self, df, ticker_name, periods=[20, 50, 200]):
        """Add EMA columns and signals for trend direction and price vs EMA."""
        for period in periods:
            ema_col = f'EMA_{period}'
            df[ema_col] = df[('Close', ticker_name)].ewm(span=period, adjust=False).mean()
            # Trend: True if EMA is rising (current > previous)
            df[f'EMA_{period}_Up'] = df[ema_col] > df[ema_col].shift(1)
            # Price vs EMA: True if Close > EMA
            df[f'Close_Above_EMA_{period}'] = df[('Close', ticker_name)] > df[ema_col]
        return df

    def analyze_last_row(self, df, name):
        """Analyze and return trend info for the latest available row."""
        if df.empty:
            self._print(f"{name}: No data available.")
            return None

        last = df.iloc[-1]
        t1, t2 = df.index[0], df.index[-1]
        self._print(f"\n=== {name} (Latest Bar) === ({t1} >> {t2})")

        trends = {}
        for period in [20, 50, 200]:
            ema_up_key = f'EMA_{period}_Up'
            close_above_key = f'Close_Above_EMA_{period}'

            ema_up = last.get(ema_up_key, np.nan)
            close_above = last.get(close_above_key, np.nan)

            # Handle MultiIndex columns: if tuple, get scalar value
            if hasattr(ema_up, 'iloc'):
                ema_up = ema_up.iloc[0]
            if hasattr(close_above, 'iloc'):
                close_above = close_above.iloc[0]

            if pd.isna(ema_up) or pd.isna(close_above):
                self._print(f"  EMA {period}: Insufficient data")
                trends[period] = None
            else:
                trend = "UP" if ema_up else "DOWN"
                position = "ABOVE" if close_above else "BELOW"
                self._print(f"  EMA {period}: Trend = {trend}, Price = {position}")
                trends[period] = ema_up  # Store boolean

        return trends

    def analyze(self, ticker_name='AAPL'):
        """Main analysis method that fetches data and runs EMA trend analysis."""
        self._print(f"Fetching data for: {ticker_name}")
        end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=200)).strftime('%Y-%m-%d')

        all_trends = {}

        # === 1-Day Data ===
        df_1d = yf.download(ticker_name, interval='1d', start=start_date, end=end_date,
                            auto_adjust=False, ignore_tz=True, progress=False)
        if not df_1d.empty:
            df_1d = self.add_ema_and_signals(df_1d, ticker_name)
            all_trends['1-Day'] = self.analyze_last_row(df_1d, "1-Day")
        else:
            self._print("1-Day: Failed to download data.")
            all_trends['1-Day'] = None

        # === 1-Hour Data ===
        df_1h = yf.download(ticker_name, interval='1h', start=start_date, end=end_date,
                            auto_adjust=False, ignore_tz=True, progress=False)
        if not df_1h.empty:
            # === 4-Hour Data (resampled from 1h) ===
            df_4h = self.get_4h_data(df_1h, ticker_name)
            df_4h = self.add_ema_and_signals(df_4h, ticker_name)
            all_trends['4-Hour'] = self.analyze_last_row(df_4h, "4-Hour")

            df_1h = self.add_ema_and_signals(df_1h, ticker_name)
            all_trends['1-Hour'] = self.analyze_last_row(df_1h, "1-Hour")
        else:
            self._print("1-Hour: Failed to download data.")
            all_trends['4-Hour'] = None
            all_trends['1-Hour'] = None

        # === Consensus Check Across Timeframes ===
        self._print("\n" + "=" * 50)
        self._print("CONSENSUS ANALYSIS: Are all EMA trends aligned?")

        # Flatten all EMA trend booleans across timeframes
        all_values = []
        valid = True

        for tf, trends in all_trends.items():
            if trends is None:
                self._print(f"  {tf}: No data → skipping consensus")
                valid = False
                continue
            for period, val in trends.items():
                if val is None:
                    self._print(f"  {tf} EMA {period}: Insufficient data → skipping consensus")
                    valid = False
                else:
                    all_values.append(val)

        if not valid or len(all_values) == 0:
            print("  → Cannot determine consensus due to missing data.")
        elif all(all_values):
            print(f"  ✅ {ticker_name} STRONG BULLISH CONSENSUS: All EMA trends are UP across all timeframes.")
        elif not any(all_values):  # i.e., all are False
            print(f"  🚨 {ticker_name} STRONG BEARISH CONSENSUS: All EMA trends are DOWN across all timeframes.")
        else:
            print(f"  ⚠️  {ticker_name} MIXED SIGNALS: Trends are not aligned across timeframes.")

        return all_trends
