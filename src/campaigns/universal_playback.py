try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from dataclasses import dataclass
from typing import Iterator, Optional, Any
import pandas as pd
from tqdm import tqdm

@dataclass(frozen=True)
class BacktestStep:
    past_df: pd.DataFrame
    future_df: Optional[pd.DataFrame]
    close_col: Any
    open_col: Any
    high_col: Any
    low_col: Any
    volume_col: Any


class BacktestIterator:
    def __init__(self, df: pd.DataFrame, step_back_range: int,
                 close_col: Any, open_col: Any, high_col: Any,
                 low_col: Any, volume_col: Any, verbose: bool = False):
        # Clean once to avoid repeated dropna() overhead
        self.df = df.dropna().copy()
        self.step_back_range = step_back_range
        self.cols = (close_col, open_col, high_col, low_col, volume_col)
        self.verbose = verbose
        self._current_step = -1

    def __iter__(self) -> Iterator[BacktestStep]:
        self._current_step = -1  # Reset state for re-iteration
        return self

    def __next__(self) -> BacktestStep:
        self._current_step += 1
        if self._current_step > self.step_back_range:
            raise StopIteration

        step = self._current_step
        if step == 0:
            past_df = self.df.copy()
            future_df = None
        else:
            past_df = self.df.iloc[:-step].copy()
            future_df = self.df.iloc[-step:].copy()

        if self.verbose:
            self._print_step_info(step, past_df, future_df)

        return BacktestStep(
            past_df=past_df,
            future_df=future_df,
            close_col=self.cols[0],
            open_col=self.cols[1],
            high_col=self.cols[2],
            low_col=self.cols[3],
            volume_col=self.cols[4]
        )

    def __len__(self) -> int:
        return self.step_back_range + 1

    def _print_step_info(self, step: int, past_df: pd.DataFrame, future_df: Optional[pd.DataFrame]):
        if future_df is not None and len(future_df) > 0:
            print(f"\nstep={step}  PAST[{len(past_df)}]: {past_df.index[0].strftime('%Y-%m-%d')} >> {past_df.index[-1].strftime('%Y-%m-%d')}   "
                  f"FUTURE[{len(future_df)}]: {future_df.index[0].strftime('%Y-%m-%d')} >> {future_df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"\nstep={step}  PAST[{len(past_df)}]: {past_df.index[0].strftime('%Y-%m-%d')} >> {past_df.index[-1].strftime('%Y-%m-%d')}   FUTURE: None")


if __name__ == "__main__":
    from argparse import Namespace
    import pickle
    from utils import get_filename_for_dataset

    ticker = '^GSPC'
    dataset_id = 'day'
    one_dataset_filename = get_filename_for_dataset(dataset_choice=dataset_id, older_dataset=None)

    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)

    master_data_cache = master_data_cache[ticker].sort_index().copy()

    cols = {
        "open_col": ("Open", ticker),
        "high_col": ("High", ticker),
        "low_col": ("Low", ticker),
        "close_col": ("Close", ticker),
        "volume_col": ("Volume", ticker)
    }

    args = Namespace(
        df=master_data_cache,
        step_back_range=10,
        verbose=True,
        **cols
    )

    # 1️⃣ Create the iterator
    bt_iter = BacktestIterator(
        df=args.df,
        step_back_range=args.step_back_range,
        close_col=args.close_col,
        open_col=args.open_col,
        high_col=args.high_col,
        low_col=args.low_col,
        volume_col=args.volume_col,
        verbose=args.verbose
    )

    # 2️⃣ Wrap with tqdm only if verbose (tqdm uses __len__ for accurate progress)
    iterator = tqdm(bt_iter) if args.verbose else bt_iter

    # 3️⃣ Run multiple algorithms cleanly
    for step in iterator:
        # Example: Pass to different backtesting functions
        # run_strategy_a(step)
        # run_strategy_b(step)
        # run_ml_model(step)
        pass