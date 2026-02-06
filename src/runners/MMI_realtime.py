try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
import pickle
from utils import get_filename_for_dataset, DATASET_AVAILABLE, str2bool
import copy
import numpy as np
from datetime import datetime, timedelta


def sma(prices, window):
    if len(prices) < window:
        return None
    return prices.rolling(window).mean().iloc[-1]


def ema(prices, window):
    if len(prices) < window:
        return None
    return prices.ewm(span=window, adjust=False).mean().iloc[-1]


def mmi(prices, period):
    if len(prices) < period:
        return None

    subset = prices[-period:]
    median = np.median(subset)
    count = 0

    for i in range(1, len(subset)):
        if (subset.iloc[i] > median and subset.iloc[i] > subset.iloc[i - 1]) or \
           (subset.iloc[i] < median and subset.iloc[i] < subset.iloc[i - 1]):
            count += 1
    if len(subset) < 2:
        return None
    return 100.0 * count / (len(subset) - 1)


def main(args):
    if args.verbose:
        # --- Nicely print the arguments ---
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            if 'master_data_cache' in arg:
                print(f"    {arg:.<40} {value.index[0].strftime('%Y-%m-%d')} to {value.index[-1].strftime('%Y-%m-%d')} ({args.one_dataset_filename})")
                continue
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)
    master_data_cache = None
    if 'master_data_cache' in args:
        master_data_cache = args.master_data_cache
    else:
        one_dataset_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None if args.older_dataset == "None" else args.older_dataset)
        with open(one_dataset_filename, 'rb') as f:
            master_data_cache = pickle.load(f)
        master_data_cache = copy.deepcopy(master_data_cache[args.ticker])
        master_data_cache = master_data_cache.sort_index()
    close_col = (args.col, args.ticker)
    close_prices = master_data_cache[close_col]
    # =========================
    # Indicators
    # =========================
    mmi_val    = mmi(close_prices, period=int(args.mmi_period))
    ma_50      = sma(close_prices, window=int(args.sma_period))
    ma_50_prev = sma(close_prices[:-1], window=int(args.sma_period))
    if args.use_ema:
        ma_50      = ema(close_prices, window=int(args.sma_period))
        ma_50_prev = ema(close_prices[:-1], window=int(args.sma_period))
    if mmi_val is None or ma_50 is None or ma_50_prev is None or 0 == len(close_prices):
        return {
        "date": close_prices.index[-1],
        "signal": None,
        "mmi": None,
        "ma_slope": None,
    }
    assert not (np.isnan(ma_50) or np.isnan(ma_50_prev))

    ma_slope = (ma_50 - ma_50_prev) / ma_50_prev
    trending = mmi_val < args.mmi_trend_max

    # =========================
    # Signal
    # =========================
    signal = "Choppy"
    if trending:
        if close_prices.iloc[-1] > ma_50 and ma_slope > 0:
            signal = "Bull"
        elif close_prices.iloc[-1] < ma_50 and ma_slope < 0:
            signal = "Bear"
    assert args.return_threshold >= 0
    prices_threshold = [close_prices.iloc[-1] * (1 + -args.return_threshold), close_prices.iloc[-1] * (1 + args.return_threshold)]
    prices_threshold = [None, prices_threshold[1]] if signal == 'Bull' else prices_threshold
    prices_threshold = [prices_threshold[0], None] if signal == 'Bear' else prices_threshold
    return {
        "date": close_prices.index[-1],
        "signal": signal,
        "prices_threshold": prices_threshold,
        "prices": close_prices.iloc[-1],
        "mmi": mmi_val,
        "ma_slope": ma_slope,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
    parser.add_argument("--mmi_period", type=float, default=100)
    parser.add_argument("--mmi_trend_max", type=float, default=70)
    parser.add_argument("--return_threshold", type=float, default=0.01)
    parser.add_argument("--sma_period", type=float, default=50)
    parser.add_argument('--use_ema', type=str2bool, default=False)
    parser.add_argument('--verbose', type=str2bool, default=True)
    args = parser.parse_args()
    result = main(args)
    if args.verbose:
        print(result)
        prediction_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        if args.dataset_id == "week":
            prediction_date = (datetime.today() + timedelta(weeks=1)).strftime('%Y-%m-%d')
        if result['signal'] == "Choppy":
            lower, upper = result['prices_threshold'][0], result['prices_threshold'][1]
            print(f"For {prediction_date}, price should be between {lower:0.2f} and {upper:0.2f}")