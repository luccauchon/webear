try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from constants import FYAHOO__OUTPUTFILENAME_YEAR, FYAHOO__OUTPUTFILENAME_QUARTER, FYAHOO__OUTPUTFILENAME_MONTH, FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_DAY
from utils import str2bool
import pickle
import copy
import numpy as np
import pandas as pd
import argparse
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import pandas as pd


def add_sequence_columns(df, col_name, epsilon=0.0):
    """
    Add POS_SEQ and NEG_SEQ columns to dataframe representing
    consecutive positive/negative close-to-close sequences.

    POS_SEQ: Counts consecutive positive returns (>= epsilon)
    NEG_SEQ: Counts consecutive negative returns (<= -epsilon)
    Values: 0 = no sequence, 1 = first step, 2 = second step, etc.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with price data
    col_name : str or tuple
        Column name for close prices
    epsilon : float
        Threshold for neutral returns (default=0.0)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added POS_SEQ and NEG_SEQ columns
    """
    df = df.copy()

    # Compute returns
    returns = df[col_name].pct_change()

    # Determine positive/negative masks
    pos_mask = returns >= epsilon
    neg_mask = returns <= -epsilon

    # Initialize sequence columns
    pos_seq = np.zeros(len(df), dtype=int)
    neg_seq = np.zeros(len(df), dtype=int)

    # Track consecutive sequences
    for i in range(1, len(df)):
        if pos_mask.iloc[i]:
            pos_seq[i] = pos_seq[i - 1] + 1
        else:
            pos_seq[i] = 0

        if neg_mask.iloc[i]:
            neg_seq[i] = neg_seq[i - 1] + 1
        else:
            neg_seq[i] = 0

    df['POS_SEQ'] = pos_seq
    df['NEG_SEQ'] = neg_seq

    return df


def add_sequence_columns_vectorized(df, col_name, epsilon=0.0):
    """
    Vectorized version of add_sequence_columns for better performance on large datasets.
    """
    df = df.copy()

    # Compute returns
    returns = df[col_name].pct_change()

    # Determine positive/negative masks
    pos_mask = (returns >= epsilon).astype(int)
    neg_mask = (returns <= -epsilon).astype(int)

    # Create groups where sequence breaks
    pos_groups = (pos_mask != pos_mask.shift(1)).cumsum()
    neg_groups = (neg_mask != neg_mask.shift(1)).cumsum()

    # Count within each group, but only where mask is True
    pos_seq = pos_mask.groupby(pos_groups).cumsum() * pos_mask
    neg_seq = neg_mask.groupby(neg_groups).cumsum() * neg_mask

    # Fill NaN with 0
    df['POS_SEQ'] = pos_seq.fillna(0).astype(int)
    df['NEG_SEQ'] = neg_seq.fillna(0).astype(int)

    return df


def compute_stats_close_to_close(df, close_col, label, verbose):
    """
    Computes close-to-close returns, mean, and std.
    """
    df = copy.deepcopy(df)
    df['Return'] = df[close_col].pct_change()
    df = df.dropna()

    mean_return = df['Return'].mean()
    std_return = df['Return'].std()
    if verbose:
        print(f"\n{'='*50}")
        print(f"{label.upper()} RETURN STATISTICS".center(50))
        print(f"{'='*50}")
        print(f"{'Mean Return:':<20} {mean_return*100:>+8.2f}%")
        print(f"{'Std Dev:':<20} {std_return*100:>+8.2f}%")
        print(f"{'Observations:':<20} {len(df):>8}")
        print(f"{'='*50}")

    return df['Return'], mean_return, std_return


def conditional_next_return_stats_fast(
    returns: pd.Series,
    n_consecutive=3,
    direction="pos",
    delta_for_n_1_pct_change=0.0,
    delta_for_last_pct_change=0.0,
    epsilon=0.,
):
    r = returns.astype(float)

    if n_consecutive == 0:
        mask_pre = pd.Series(True, index=r.index)
    else:
        if direction == "pos":
            sign_mask = r >= epsilon
        else:
            sign_mask = r <= -epsilon

        # All previous n_consecutive satisfy sign
        mask_pre = (
            sign_mask
            .rolling(n_consecutive)
            .sum()
            .eq(n_consecutive)
        )

        # Compounded return of previous n_consecutive
        compounded = (
            (1 + r)
            .rolling(n_consecutive)
            .apply(np.prod, raw=True)
            - 1
        )

        if direction == "pos":
            mask_pre &= compounded >= delta_for_n_1_pct_change
        else:
            mask_pre &= compounded <= delta_for_n_1_pct_change

    # Next-period return
    next_r = r.shift(-1)

    if direction == "pos":
        mask_next = next_r >= delta_for_last_pct_change
    else:
        mask_next = next_r <= delta_for_last_pct_change

    final_mask = mask_pre & mask_next

    next_returns = next_r[final_mask].dropna()

    if next_returns.empty:
        return {
            "total": len(r),
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
        }

    return {
        "total": len(r),
        "count": next_returns.size,
        "mean": next_returns.mean(),
        "std": next_returns.std(),
    }


def restricted_conditional_next_return_stats_fast(
    returns_series,
    n_consecutive=3,
    direction='pos',
    delta_for_n_1_pct_change=0.0,
    delta_for_last_pct_change=0.0,
    epsilon=0.0,
):
    """
    Fast vectorized version. Avoids Python loops.
    """
    r = returns_series.astype(float)
    total = len(r)

    if n_consecutive == 0:
        mask_pre = pd.Series(True, index=r.index)
        compounded = pd.Series(0.0, index=r.index)  # dummy
    else:
        # Sign condition
        if direction == "pos":
            sign_ok = r >= epsilon
        else:
            sign_ok = r <= -epsilon

        # Rolling window: all n_consecutive signs correct?
        mask_pre = sign_ok.rolling(n_consecutive).sum() == n_consecutive

        # Compounded return over n_consecutive
        compounded = (1 + r).rolling(n_consecutive).apply(np.prod, raw=True) - 1

        # Apply compounded threshold
        if direction == "pos":
            mask_pre &= compounded >= delta_for_n_1_pct_change
        else:
            mask_pre &= compounded <= delta_for_n_1_pct_change

    # Next return
    next_r = r.shift(-1)

    # Next return condition
    if direction == "pos":
        mask_next = next_r >= delta_for_last_pct_change
    else:
        mask_next = next_r <= delta_for_last_pct_change

    final_mask = mask_pre & mask_next

    # Count n_consecutive occurrences (regardless of next)
    number_of_n_consecutive_step = mask_pre.sum()

    # Count valid n+1 sequences
    number_of_n_plus_one_consecutive_step = final_mask.sum()

    return {
        "number_of_n_consecutive_step": int(number_of_n_consecutive_step),
        "number_of_n_plus_one_consecutive_step": int(number_of_n_plus_one_consecutive_step),
    }


def restricted_conditional_next_return_stats(returns_series, n_consecutive=3, direction='pos',
                                             delta_for_n_1_pct_change=0., delta_for_last_pct_change=0., epsilon=0.):
    """
    Analyze returns following N+1 consecutive positive/negative returns, given the realization of N consecutive positive/negative returns
    """
    pre_returns, next_returns, windows_accumulator = [], [], []

    # Compute the number of sequence of "n_consecutive" step
    number_of_n_consecutive_step, indexes = 0, []
    for i in tqdm(range(n_consecutive, len(returns_series))):
        window = returns_series.iloc[i - n_consecutive: i]
        assert len(window) == n_consecutive
        if direction == 'pos':
            if (window >= epsilon).all():
                number_of_n_consecutive_step += 1
                indexes.append((i - n_consecutive, i))
        else:
            if (window <= -epsilon).all():
                number_of_n_consecutive_step += 1
                indexes.append((i - n_consecutive, i))

    # Compute the number of times the next step is correct, with respect to the specification
    number_of_n_plus_one_consecutive_step = 0
    for i in tqdm(indexes):
        idx1, idx2 = i
        window = returns_series.iloc[idx1: idx2+1]
        assert len(window) == n_consecutive + 1
        if direction == 'pos':
            if window.iloc[-1] >= delta_for_last_pct_change:
                if np.prod(1 + window.iloc[:-1]) >= (1 + delta_for_n_1_pct_change) or n_consecutive == 0:
                    next_returns.append(window.iloc[-1])
                    pre_returns.append(window.iloc[:-1] if len(window.iloc[:-1]) > 0 else pd.Series([0]))
                    windows_accumulator.append(window)
                    number_of_n_plus_one_consecutive_step += 1
        else:
            if window.iloc[-1] <= delta_for_last_pct_change:
                if np.prod(1 + window.iloc[:-1]) <= (1 + delta_for_n_1_pct_change) or n_consecutive == 0:
                    next_returns.append(window.iloc[-1])
                    pre_returns.append(window.iloc[:-1] if len(window.iloc[:-1]) > 0 else pd.Series([0]))
                    windows_accumulator.append(window)
                    number_of_n_plus_one_consecutive_step += 1

    if not next_returns:
        return {
            "number_of_n_plus_one_consecutive_step": 0, "number_of_n_consecutive_step": 0
        }

    assert len(next_returns) == len(windows_accumulator) == len(pre_returns)
    return {
        "number_of_n_plus_one_consecutive_step": number_of_n_plus_one_consecutive_step,
        "number_of_n_consecutive_step": number_of_n_consecutive_step,
    }


def conditional_next_return_stats(returns_series, n_consecutive=3, direction='pos',
                                 delta_for_n_1_pct_change=0., delta_for_last_pct_change=0., epsilon=0):
    """
    Analyze returns following N consecutive positive/negative returns.
    """
    pre_returns, next_returns, windows_accumulator = [], [], []
    for i in tqdm(range(n_consecutive, len(returns_series))):
        window = returns_series.iloc[i - n_consecutive: i+1]
        assert len(window) == n_consecutive + 1
        if direction == 'pos':
            if (window.iloc[:-1] >= epsilon).all():
                if window.iloc[-1] >= delta_for_last_pct_change:
                    if np.prod(1 + window.iloc[:-1]) >= (1 + delta_for_n_1_pct_change) or n_consecutive == 0:
                        next_returns.append(window.iloc[-1])
                        pre_returns.append(window.iloc[:-1] if len(window.iloc[:-1]) > 0 else pd.Series([0]))
                        windows_accumulator.append(window)
        else:
            if (window.iloc[:-1] <= -epsilon).all():
                if window.iloc[-1] <= delta_for_last_pct_change:
                    if np.prod(1 + window.iloc[:-1]) <= (1 + delta_for_n_1_pct_change) or n_consecutive == 0:
                        next_returns.append(window.iloc[-1])
                        pre_returns.append(window.iloc[:-1] if len(window.iloc[:-1]) > 0 else pd.Series([0]))
                        windows_accumulator.append(window)

    if not next_returns:
        return {
            "count": 0, "mean": np.nan, "std": np.nan, "total": len(returns_series),
            "pre_window_mean": 0, "pre_window_std": 0, "windows": []
        }

    assert len(next_returns) == len(windows_accumulator) == len(pre_returns)
    next_returns = np.array(next_returns)
    return {
        "total": len(returns_series),
        "count": len(next_returns),
        "mean": next_returns.mean(),
        "std": next_returns.std(),
        "pre_window_mean": np.mean([np.mean(k.values) for k in pre_returns]),
        "pre_window_std": np.std([np.std(k.values) for k in pre_returns]),
        "windows": windows_accumulator,
    }


def get_label(data_frequency):
    labels = {
        'day': 'Days',
        'week': 'Weeks',
        'month': 'Months',
        'quarter': 'Quarters',
        'year': 'Years'
    }
    return labels[data_frequency]


def load_data(data_frequency, ticker):
    mapping = {
        'day': FYAHOO__OUTPUTFILENAME_DAY,
        'week': FYAHOO__OUTPUTFILENAME_WEEK,
        'month': FYAHOO__OUTPUTFILENAME_MONTH,
        'quarter': FYAHOO__OUTPUTFILENAME_QUARTER,
        'year': FYAHOO__OUTPUTFILENAME_YEAR,
    }
    with open(mapping[data_frequency], 'rb') as f:
        master_data_cache = pickle.load(f)
    return master_data_cache[ticker]


def new_main(args, bring_my_own_df=None):
    ticker = "^GSPC"
    close_col = ('Close', ticker)
    data_frequency = args.frequency
    direction = args.direction
    max_n = args.max_n
    min_n = args.min_n
    delta = args.delta
    verbose = args.verbose
    debug_speeding = args.debug_verify_speeding
    if bring_my_own_df is None:
        _spx500 = load_data(data_frequency, ticker)
    else:
        _spx500 = bring_my_own_df

    _returns, _mean, _std = compute_stats_close_to_close(_spx500, close_col=close_col, label=get_label(data_frequency), verbose=verbose)
    direction_label = "Positive" if direction == "pos" else "Negative"
    time_label = get_label(data_frequency)
    if verbose:
        print(f"\n{'='*50}")
        print(f"CONDITIONAL RETURN ANALYSIS".center(50))
        print(f"{'='*50}")
        print(f"Frequency: {time_label}")
        print(f"Direction: {direction_label}")
        print(f"Min streak: {min_n}")
        print(f"Max streak: {max_n}")
        print(f"{'='*50}")
    assert delta >= 0
    if direction == 'neg':
        delta = -delta
    returned_results = {}
    _returns = copy.deepcopy(_returns.dropna())
    for NN in range(min_n, max_n + 1):
        stats = conditional_next_return_stats_fast(_returns,
            direction=direction,
            n_consecutive=NN,
            delta_for_n_1_pct_change=0.,
            delta_for_last_pct_change=delta,
            epsilon=args.epsilon,
        )
        if debug_speeding:
            stats2 = conditional_next_return_stats(
                _returns,
                direction=direction,
                n_consecutive=NN,
                delta_for_n_1_pct_change=0.,
                delta_for_last_pct_change=delta,
                epsilon=args.epsilon,
            )
            assert np.allclose(stats['count'], stats2['count'], atol=1)
            assert np.allclose(stats['total'], stats2['total'], atol=1)
            if not np.isnan(stats2['mean']):
                assert np.allclose(stats['mean'], stats2['mean'], atol=0.01), f"{stats['mean']} == {stats2['mean']}"
            if not np.isnan(stats2['std']) and stats2['std'] != 0:
                assert np.allclose(stats['std'], stats2['std'], atol=0.04), f"{stats['std']}  ==  {stats2['std']}"
            for small_delta in [0., 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
                stats3 = restricted_conditional_next_return_stats_fast(_returns,
                                                                  direction=direction,
                                                                  n_consecutive=NN,
                                                                  delta_for_n_1_pct_change=0.,
                                                                  delta_for_last_pct_change=small_delta, epsilon=args.epsilon,)
                stats4 = restricted_conditional_next_return_stats(_returns,
                                                                  direction=direction,
                                                                  n_consecutive=NN,
                                                                  delta_for_n_1_pct_change=0.,
                                                                  delta_for_last_pct_change=small_delta, epsilon=args.epsilon,)
                if stats4['number_of_n_plus_one_consecutive_step'] == 0:
                    assert np.allclose(stats3['number_of_n_plus_one_consecutive_step'], stats4['number_of_n_plus_one_consecutive_step'], atol=1)
                    break
                assert np.allclose(stats3['number_of_n_consecutive_step'],          stats4['number_of_n_consecutive_step'], atol=1)
                assert np.allclose(stats3['number_of_n_plus_one_consecutive_step'], stats4['number_of_n_plus_one_consecutive_step'], atol=1)
        pct = stats['count'] / stats['total'] if stats['total'] != 0 else 0
        if verbose:
            print(f"\n▶ After {NN:2d} Consecutive {direction_label} {time_label}")
            print("-" * 50)
            tmpstr = '' if delta==0 else f'with a delta of {delta*100:.1f}%'
            print(f"Probability of occurence of next {direction_label} {time_label} is {pct*100:6.2f}% ({stats['count']}/{stats['total']}) {tmpstr}")
            if stats['count'] > 0:
                for cond_delta in [0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.040, 0.045, 0.05, 0.06]:
                    if direction == 'neg':
                        cond_delta = -cond_delta
                    restricted_stats = restricted_conditional_next_return_stats_fast(_returns,
                                                                                     direction=direction,
                                                                                     n_consecutive=NN,
                                                                                     delta_for_n_1_pct_change=0.,
                                                                                     delta_for_last_pct_change=cond_delta, epsilon=args.epsilon,)
                    cond_prob = restricted_stats['number_of_n_plus_one_consecutive_step'] / restricted_stats['number_of_n_consecutive_step'] if 0 != restricted_stats['number_of_n_plus_one_consecutive_step'] else 0
                    print(f"\tConditional Next {direction_label} is {cond_prob * 100:.1f}% with @delta of {cond_delta*100:.1f}%  "
                          f"({restricted_stats['number_of_n_plus_one_consecutive_step']}/{restricted_stats['number_of_n_consecutive_step']})")
                    if cond_prob*100. < 5:  # Below 5%, we stop
                        break
            # if stats['count'] > 0:
            #     print(f"{'Next Return μ:':<20} {stats['mean']*100:+8.2f}%")
            #     print(f"{'Next Return σ:':<20} {stats['std']*100:8.2f}%")
            #     if 'pre_window_mean' in stats:
            #         print(f"{'Pre-Window μ:':<20} {stats['pre_window_mean']*100:+8.2f}%")
            #         print(f"{'Pre-Window σ:':<20} {stats['pre_window_std']*100:8.2f}%")
            # else:
            #     print(f"{'Next Return μ:':<20} {'N/A':>8}")
            #     print(f"{'Next Return σ:':<20} {'N/A':>8}")
        returned_results.update({NN: {'NN': NN, 'frequency': data_frequency, 'prob': pct, 'count': stats['count'], 'total_streaks': stats['total'], 'delta': delta}})
        if stats['count'] == 0:
            if verbose:
                print("Stopping loop since 0 elements where found")
            break
    return returned_results


def main(direction: str, method: str, older_dataset: str, bold: int, frequency: str, delta: float, ticker_name: str, verbose: bool, bring_my_own_df=None,):
    assert direction in ["pos", "neg"]
    assert -1 <= delta <= 1
    assert method in ['prev_close']
    args = Namespace(
        frequency=frequency,
        direction=direction,
        max_n=12,
        min_n=0,
        delta=delta,
        verbose=verbose,
        debug_verify_speeding=False,
        epsilon=0.,
    )
    return new_main(args, bring_my_own_df=bring_my_own_df,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze conditional returns for ^GSPC after N consecutive positive/negative periods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--frequency",
        choices=["day", "week", "month", "quarter", "year"],
        default="day",
        help="Time frequency of the data"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.,
        help="Minimum variation in last element of streak to consider the streak sequence as valid"
    )
    parser.add_argument(
        "--direction",
        choices=["pos", "neg"],
        default="pos",
        help="Direction of streaks to analyze"
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=10,
        help="Maximum number of consecutive periods to test"
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=0,
        help="Minimum number of consecutive periods to test"
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Display verbose output"
    )
    parser.add_argument(
        "--debug_verify_speeding",
        type=str2bool,
        default=False,
        help=""
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.00005,
        help="Threshold for neutral returns (|return| < epsilon breaks streaks). Default=0.00005 (0.005%%)"
    )

    args = parser.parse_args()
    new_main(args)