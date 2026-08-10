from argparse import Namespace
try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import argparse
from optimizers.prime_rsi.realtime_and_backtest_hyperparameter_search_optuna import entry as realtime_and_backtest_hyperparameter_search_optuna


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    data_group = parser.add_argument_group('Data & Symbol')
    data_group.add_argument('--dataset-id', type=str, default='day', help='Dataset identifier')
    data_group.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    data_group.add_argument("--clip", action="store_true", help="Exclude incomplete current bar in real-time")
    data_group.add_argument("--clip-n", type=int, default=0, help="Number of most recent bars to clip from the dataset.")
    data_group.add_argument("--reduce-n", type=int, default=0, help="Number of most oldest bars to clip from the dataset.")

    strat_group = parser.add_argument_group('Strategy & P&L Parameters')
    strat_group.add_argument('--lookahead-bars', type=int, default=1, dest='lookahead_bars', help='Forward-looking window')
    strat_group.add_argument('--method', type=str, default='final_close', choices=['touched', 'final_close'], help='Strike evaluation method')
    strat_group.add_argument('--min-signal-density', type=float, default=0.0001, help='Min signal frequency threshold')
    strat_group.add_argument('--put-strike-pct', type=float, default=0.99999, help='Base put strike multiplier')
    strat_group.add_argument('--call-strike-pct', type=float, default=1.00001, help='Base call strike multiplier')
    strat_group.add_argument('--profile', type=str, default="institutional_pullback", help='', choices=["institutional_pullback",
                                                                                                        "structural_confluence", "exhaustion_reversal"])

    opt_group = parser.add_argument_group('Optimization & Execution')
    opt_group.add_argument('--optimize-target', type=str, default='buy_wr', choices=['combined_wr', 'buy_wr', 'sell_wr'],
                           help='Metric to maximize during optimization')
    opt_group.add_argument('--n-trials', type=int, default=99999, help='Optuna trials per run')
    opt_group.add_argument('--timeout', type=int, default=86400, help='Max runtime (seconds)')
    opt_group.add_argument('--output-dir', type=str, default='models', help='Output directory')
    opt_group.add_argument('--optuna-db', type=str, default=None, help='Database URL for Optuna persistence (e.g., sqlite:///optuna.db or postgresql://user:pass@host/db)')
    opt_group.add_argument('--train-ratio', type=float, default=0.9,
                           help='Ratio of data to use for training (rest for validation). Use 1.0 to disable split.')

    return parser


def print_configuration_nice(configuration: Namespace, profile_name: str):
    """Helper function to pretty-print the configuration before optimization."""
    print("\n" + "=" * 70)
    print(f" 🚀 OPTIMIZATION PROFILE: {profile_name.upper().replace('_', ' ')}")
    print("=" * 70)

    # Define profile-specific keys to highlight
    profile_keys = [
        'buy_confluence_range', 'sell_confluence_range',
        'use_ma_conf_buy', 'use_vwap_buy', 'use_vol_buy',
        'use_fib_rsi_buy', 'use_macd_buy',
        'use_reg_bull_div', 'use_ema_cross_buy', 'use_bb_buy'
    ]

    print(" 📊 Active Profile Parameters:")
    config_dict = vars(configuration)
    for key in profile_keys:
        if key in config_dict:
            val = config_dict[key]
            # Highlight boolean True values or non-None ranges
            if val is True or val is not None:
                print(f"   • {key:<25}: {val}")

    print("-" * 70)
    print(" ⚙️  Full Configuration Namespace:")
    # Print all other configuration parameters neatly sorted
    for key in sorted(config_dict.keys()):
        if key not in profile_keys:
            print(f"   • {key:<25}: {config_dict[key]}")
    print("=" * 70 + "\n")


def entry(args):
    configuration = Namespace(seed=52, verbose=True, ticker=args.ticker, real_time=False, dataset_id=args.dataset_id, clip_n=args.clip_n, reduce_n=args.reduce_n, optimize=True,
                              timeout=args.timeout, n_trials=args.n_trials, put_strike_pct=args.put_strike_pct, call_strike_pct=args.call_strike_pct,
                              lookahead_bars=args.lookahead_bars, method=args.method, min_signal_density=args.min_signal_density,
                              model_path=None, optuna_db=args.optuna_db, verbose_optuna_progression=True,
                              sanity_check=False, output_dir=args.output_dir, plot=False, train_ratio=args.train_ratio,
                              optimize_target=args.optimize_target,
                                      )
    if args.profile == "institutional_pullback":
        ###########################################################################
        # Combination A: "The Institutional Pullback" (Highest Win Rate for Credit Spreads)
        # Logic: Buy the dip in an established uptrend when price retests a value area with volume confirmation.
        # This is the safest setup for ensuring the price stays above your short put strike.
        # Components: use_ma_conf_buy + use_vwap_buy + use_vol_buy
        # Literature Backing: VWAP is the primary benchmark for institutional execution. A bounce off rolling VWAP with a volume spike indicates smart money defending a position.
        ###########################################################################
        configuration.buy_confluence_range = (3,3)
        configuration.sell_confluence_range = (3,3)
        configuration.use_ma_conf_buy = True
        configuration.use_vwap_buy = True
        configuration.use_vol_buy = True
        configuration.optuna_db = f"journal://institutional_pullback.db"

    if args.profile == "structural_confluence":
        # Combination B: "The Structural Confluence" (Best Risk/Reward)
        # Logic: Price enters a universally watched algorithmic support zone, and momentum confirms the turn.
        # Components: use_fib_rsi_buy + use_macd_buy + use_vol_buy
        # Literature Backing: The 0.5–0.618 Fibonacci zone is a well-documented self-fulfilling prophecy in quant literature.
        # Pairing it with MACD (which measures the rate of change of the trend) filters out "falling knife" scenarios.
        configuration.buy_confluence_range = (3, 3)
        configuration.sell_confluence_range = (3, 3)
        configuration.use_fib_rsi_buy = True
        configuration.use_macd_buy = True
        configuration.use_vol_buy = True
        configuration.optuna_db = f"journal://structural_confluence.db"

    if args.profile == "exhaustion_reversal":
        # Combination C: "The Exhaustion Reversal" (Moderate Win Rate, High Frequency)
        # Logic: Sellers are exhausted, and a sharp mean-reversion bounce is imminent.
        # Components: use_reg_bull_div + use_ema_cross_buy (or use_bb_buy)
        # Literature Backing: Bulkowski’s statistical analysis shows Regular Bullish Divergence is one of the most reliable predictors of short-term trend exhaustion.
        configuration.buy_confluence_range = (2, 3)
        configuration.sell_confluence_range = (2, 3)
        configuration.use_reg_bull_div = True
        configuration.use_ema_cross_buy = True
        configuration.use_bb_buy = True
        configuration.optuna_db = f"journal://exhaustion_reversal.db"

    # Bonus SOTA Tip: Hidden vs. Regular Divergence
    # If you include divergence, note that Hidden Bullish Divergence (use_hid_bull_div) is statistically more reliable for trend continuation
    # (which is what you want for a Put Credit Spread in an uptrend),
    # while Regular Bullish Divergence (use_reg_bull_div) is for reversals.
    # If your use_ma_conf_buy is active, pair it with Hidden Divergence, not Regular.

    # 🎨 PRINT NICELY BEFORE CALLING OPTUNA
    print_configuration_nice(configuration, args.profile)

    # Call the optimization function
    realtime_and_backtest_hyperparameter_search_optuna(args=configuration)


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)
