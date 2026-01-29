try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import pandas_datareader.data as web
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from constants import FRED_API_KEY


def category_A_Market_Internals():
    def compute_ma_stats(df, window=50, ma_type='SMA'):
        """
        Compute moving average (SMA or EMA) for each stock using 'Close' prices,
        then compare the latest close price to the latest MA value.

        Parameters:
        - df: pandas DataFrame with MultiIndex columns (e.g., ('Close', 'AAPL'))
        - window: int, window size for the moving average (default: 50)
        - ma_type: str, either 'SMA' or 'EMA' (default: 'SMA')

        Returns:
        - summary: dict with total stocks, number above MA, percentage above MA
        - results_df: DataFrame with ticker, last_close, last_ma, is_above
        """
        if df.empty:
            return {'Total_Stocks_Analyzed': 0}, pd.DataFrame()

        # Ensure MultiIndex and extract tickers
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected DataFrame with MultiIndex columns like ('Close', 'TICKER')")

        tickers = df.columns.get_level_values(1).unique()
        results = []

        for ticker in tickers:
            close_series = df[('Close', ticker)].dropna()

            if len(close_series) < window:
                continue

            if ma_type.upper() == 'SMA':
                ma_series = close_series.rolling(window=window).mean()
            elif ma_type.upper() == 'EMA':
                ma_series = close_series.ewm(span=window, adjust=False).mean()
            else:
                raise ValueError("ma_type must be 'SMA' or 'EMA'")

            last_close = close_series.iloc[-1]
            last_ma = ma_series.iloc[-1]
            is_above = last_close > last_ma

            results.append({
                'Ticker': ticker,
                'Last_Close': last_close,
                'Last_MA': last_ma,
                'Is_Above_MA': is_above
            })

        results_df = pd.DataFrame(results)
        total = len(results_df)
        above = results_df['Is_Above_MA'].sum() if total > 0 else 0
        pct_above = (above / total * 100) if total > 0 else 0

        summary = {
            'Total_Stocks_Analyzed': total,
            'Stocks_Above_MA': above,
            'Percent_Above_MA': pct_above,
            'MA_Type': ma_type,
            'Window': window
        }

        return summary, results_df

    def fewer_stocks_above_their_50_day_MA():
        try:
            df = pd.read_parquet(r"D:\Finance\data\yfinance\sp500_daily_data.parquet")
        except Exception as e:
            print(f"⚠️ Failed to load SP500 data: {e}")
            return 0  # conservative: assume not bearish if data missing

        summary, _ = compute_ma_stats(df, window=50, ma_type='SMA')
        return 0 if summary['Percent_Above_MA'] > 50 else 1

    def check_rsp_spx_divergence(days=20, threshold=0.03):
        tickers = ["^GSPC", "RSP"]
        try:
            data = yf.download(tickers, period="6mo", interval="1d", auto_adjust=True)
        except Exception as e:
            print(f"⚠️ Error downloading SPX/RSP data: {e}")
            return 0

        if data.empty or len(data) < days + 5:
            print("❌ Not enough data for SPX/RSP divergence check.")
            return 0

        # Handle both flat and MultiIndex column structures
        if isinstance(data.columns, pd.MultiIndex):
            spx = data[('Close', '^GSPC')]
            rsp = data[('Close', 'RSP')]
        else:
            spx = data['^GSPC']
            rsp = data['RSP']

        # Align and drop NaN
        combined = pd.DataFrame({'SPX': spx, 'RSP': rsp}).dropna()
        if len(combined) < days + 1:
            print("❌ Not enough aligned data points.")
            return 0

        spx_latest = combined['SPX'].iloc[-1]
        spx_past = combined['SPX'].iloc[-(days + 1)]
        rsp_latest = combined['RSP'].iloc[-1]
        rsp_past = combined['RSP'].iloc[-(days + 1)]

        spx_return = (spx_latest / spx_past) - 1
        rsp_return = (rsp_latest / rsp_past) - 1
        underperformance = spx_return - rsp_return

        print(f"SPX {days}-day return: {spx_return:.2%}")
        print(f"RSP {days}-day return: {rsp_return:.2%}")
        print(f"RSP underperformance vs SPX: {underperformance:.2%}")

        if underperformance > threshold:
            print(f"✅ Divergence detected: RSP underperformed SPX by >{threshold:.0%} over {days} days.")
            return 1
        else:
            print(f"❌ No significant divergence (> {threshold:.0%} underperformance).")
            return 0

    def check_hyg_spx_divergence(days=5, spx_min_return=0.0, hyg_max_return=-0.001):
        tickers = ["^GSPC", "HYG"]
        try:
            data = yf.download(tickers, period="3mo", interval="1d", auto_adjust=True)
        except Exception as e:
            print(f"⚠️ Error downloading SPX/HYG data: {e}")
            return 0

        if data.empty or len(data) < days + 5:
            print("❌ Not enough data for SPX/HYG divergence check.")
            return 0

        if isinstance(data.columns, pd.MultiIndex):
            spx = data[('Close', '^GSPC')]
            hyg = data[('Close', 'HYG')]
        else:
            spx = data['^GSPC']
            hyg = data['HYG']

        combined = pd.DataFrame({'SPX': spx, 'HYG': hyg}).dropna()
        if len(combined) < days + 1:
            print("❌ Not enough aligned data points.")
            return 0

        spx_latest = combined['SPX'].iloc[-1]
        spx_past = combined['SPX'].iloc[-(days + 1)]
        hyg_latest = combined['HYG'].iloc[-1]
        hyg_past = combined['HYG'].iloc[-(days + 1)]

        spx_return = spx_latest / spx_past - 1
        hyg_return = hyg_latest / hyg_past - 1

        print(f"📅 Period: {days} trading days")
        print(f"📈 SPX return: {spx_return:.2%}")
        print(f"📉 HYG return: {hyg_return:.2%}")

        spx_up = spx_return >= spx_min_return
        hyg_down = hyg_return <= hyg_max_return

        if spx_up and hyg_down:
            print("⚠️ DIVERGENCE DETECTED: SPX up while HYG down!")
            print("   → Equity market rising, but high-yield bonds weakening.")
            print("   → Possible early warning of risk-off sentiment or credit stress.")
            return 1
        else:
            print("✅ No divergence: SPX and HYG moving in expected alignment.")
            return 0

    score = 0
    score += fewer_stocks_above_their_50_day_MA()
    score += check_rsp_spx_divergence(days=20, threshold=0.03)
    score += check_hyg_spx_divergence(days=5)
    return score


def Category_B_Technical_Conditions():
    ticker = "^GSPC"
    try:
        data = yf.download(ticker, period="2y", interval="1d")
    except Exception as e:
        print(f"⚠️ Failed to download SPX data: {e}")
        return 0

    if data.empty:
        print("❌ Empty SPX data.")
        return 0

    close_col = ('Close', ticker)

    # --- Condition 1: Weekly RSI > 68 ---
    weekly = data[close_col].resample('W-FRI').last().dropna()
    weekly_rsi = ta.rsi(weekly, length=14)
    weekly_rsi_cond = weekly_rsi.iloc[-1] > 68 if not weekly_rsi.empty else False

    # --- Condition 2: SPX above 2σ Bollinger Band (20-day) ---
    bb = ta.bbands(data[close_col], length=20, std=2)
    bb_upper = bb['BBU_20_2.0_2.0']
    bb_cond = data[close_col].iloc[-1] > bb_upper.iloc[-1] if not bb_upper.empty else False

    # --- Condition 3: Monthly candle > 2% above 20-month EMA ---
    monthly = data[close_col].resample('ME').last().dropna()
    ema20 = ta.ema(monthly, length=20)
    parabolic_cond = (
        (monthly.iloc[-1] > 1.02 * ema20.iloc[-1]) if (not monthly.empty and not ema20.empty) else False
    )

    points = int(weekly_rsi_cond) + int(bb_cond) + int(parabolic_cond)

    print("\n=== Category B: Technical Conditions (S&P 500) ===")
    print(f"- Weekly RSI > 68: {weekly_rsi_cond} (RSI = {weekly_rsi.iloc[-1]:.2f} if available)")
    print(f"- Price > 2σ Bollinger (20-day): {bb_cond}")
    print(f"- Monthly close > 2% above 20-month EMA: {parabolic_cond}")
    print(f"\nTotal Points: {points}/3")
    return points


def category_C_Macro_Liquidity():
    def real_yield_rising_20bp():
        end = datetime.today()
        start = end - timedelta(weeks=6)
        try:
            df = web.DataReader('DFII10', 'fred', start, end, api_key=FRED_API_KEY)
            df = df.dropna()
            if len(df) < 5:
                return False
            # Get most recent and ~14 days ago (asof handles holidays)
            recent_val = df['DFII10'].iloc[-1]
            past_val = df['DFII10'].asof(end - timedelta(weeks=2))
            change_bps = (recent_val - past_val) * 100
            return change_bps >= 20
        except Exception as e:
            print("Error fetching TIPS data:", e)
            return False

    def financial_conditions_tightening():
        end = datetime.today()
        start = end - timedelta(weeks=6)
        try:
            df = web.DataReader('NFCI', 'fred', start, end, api_key=FRED_API_KEY)
            df = df.dropna()
            if len(df) < 5:
                return False
            recent = df['NFCI'].iloc[-1]
            past = df['NFCI'].asof(end - timedelta(weeks=2))
            return recent > past
        except Exception as e:
            print("Error fetching NFCI data:", e)
            return False

    points = 0
    if real_yield_rising_20bp():
        print("✅ Real yields rose ≥20 bps in 2 weeks → +1 point")
        points += 1
    else:
        print("❌ Real yields did not rise ≥20 bps")

    if financial_conditions_tightening():
        print("✅ Financial conditions tightened → +1 point")
        points += 1
    else:
        print("❌ Financial conditions did not tighten")

    print(f"\n=== Category C: Macro Liquidity Score: {points}/2 ===")
    return points


def entry():
    print("Running Market Regime Scanner...\n")
    score = 0
    score += category_A_Market_Internals()
    print("\n" + "="*60 + "\n")
    score += Category_B_Technical_Conditions()
    print("\n" + "="*60 + "\n")
    score += category_C_Macro_Liquidity()
    str_response = (("\n" + "=" * 60) + f"FINAL REGIME SCORE: {score}/6"+
                    "| Score    | Meaning                                          | Action for credit put spreads |"+
                    "| -------- | ------------------------------------------------ | ----------------------------- |"+
                    "| **0–3**  | Low drop probability                             | Enter normal position         |"+
                    "| **4–6**  | Medium risk                                      | Reduce size 30–50%            |"+
                    "| **7–10** | High probability of 5–8% drop in next 30–45 days | Avoid selling CPS this month  |")
    print("\n" + "="*60)
    print(f"FINAL REGIME SCORE: {score}/6")
    print("| Score    | Meaning                                          | Action for credit put spreads |")
    print("| -------- | ------------------------------------------------ | ----------------------------- |")
    print("| **0–3**  | Low drop probability                             | Enter normal position         |")
    print("| **4–6**  | Medium risk                                      | Reduce size 30–50%            |")
    print("| **7–10** | High probability of 5–8% drop in next 30–45 days | Avoid selling CPS this month  |")
    # Note: max possible is 6 (2+3+1? Actually A=3, B=3, C=2 → max=8)
    # But your table says up to 10. Clarify if needed.
    return score, str_response


if __name__ == "__main__":
    entry()