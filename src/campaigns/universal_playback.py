from tqdm import tqdm
from argparse import Namespace
from utils import get_filename_for_dataset
import pickle


def entry_point(args):
    # Load dataset
    one_dataset_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    master_data_cache = master_data_cache[args.ticker].sort_index().dropna()
    # Define column tuples
    open_col = ("Open", args.ticker)
    high_col = ("High", args.ticker)
    low_col = ("Low", args.ticker)
    close_col = ("Close", args.ticker)
    volume_col = ("Volume", args.ticker)

    for step_back in (tqdm(range(0, args.step_back_range + 1)) if args.verbose else range(0, args.step_back_range + 1)):
        if 0 == step_back:
            past_df = master_data_cache.copy().dropna()
            future_df = None
        else:
            past_df = master_data_cache.iloc[:-step_back].copy().dropna()
            future_df = master_data_cache.iloc[-step_back:].copy().dropna()
        if args.verbose:
            try:
                print(f"\n{step_back=}  PAST[{len(past_df)}]:{past_df.index[0].strftime('%Y-%m-%d')} >> {past_df.index[-1].strftime('%Y-%m-%d')}   "
                      f"FUTURE[{len(future_df)}]:{future_df.index[0].strftime('%Y-%m-%d')} >> {future_df.index[-1].strftime('%Y-%m-%d')}")
            except:
                print(f"\n{step_back=}  PAST[{len(past_df)}]:{past_df.index[0].strftime('%Y-%m-%d')} >> {past_df.index[-1].strftime('%Y-%m-%d')}   "
                      f"FUTURE: None")

if __name__ == "__main__":
    args = Namespace(step_back_range=10, verbose=True, dataset_id='day', ticker='^GSPC')
    entry_point(args)