import pandas as pd
import numpy as np
from fetchers.serialize_fyahoo import entry as fyahoo_entry
from algorithms.trade_prime_half_trend import trade_prime_half_trend_strategy as trade_prime_half_trend_strategy
from datetime import datetime, timedelta


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    ticker = "MSFT"
    ticker = "SPY"
    ticker = "^GSPC"
    today = datetime.today()
    end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=725)).strftime('%Y-%m-%d')  # Max 730 for FYahoo!

    df = fyahoo_entry(
    use_all_tickers=False,
    start_date=start_date,
    end_date=end_date,
    skip_hourly=False,
    skip_daily=True,
    skip_weekly=True,
    skip_monthly=True,
    skip_quarterly=True,
    skip_yearly=True,
    skip_economic=True,
    auto_adjust=True,
    dont_serialize_return_df_instead=True,
    specify_tickers=(ticker,))[ticker]
    print(f"{ticker} : {df.index[0].strftime('%Y-%m-%d_%H%M')}::{df.index[-1].strftime('%Y-%m-%d_%H%M')}")
    df_result = trade_prime_half_trend_strategy(ticker_df=df.copy(), ticker_name=ticker, buy_setup=True,
                                                **{'lookahead': 7*1, 'stop_loss_atr_factor': 1.5, 'take_profit_atr_factor': 0})
    recent_signals = df_result[df_result[('custom_signal', ticker)]]
    print(recent_signals)
    n_trades = len(recent_signals[('win_setup', ticker)])
    nb_ones = sum(recent_signals[('win_setup', ticker)])
    win_rate = nb_ones / n_trades
    print(f"{win_rate=}")
