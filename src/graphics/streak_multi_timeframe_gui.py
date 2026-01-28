try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

import tkinter as tk
from tkinter import ttk, messagebox
from argparse import Namespace
import copy
from datetime import datetime
import pickle
from runners.streak_multi_timeframe import main as streak_multi_timeframe
from utils import get_filename_for_dataset

# --- Global reference to skip indicator lights ---
skip_lights = {'day': None, 'week': None, 'month': None}


# --- Core auto-detection logic ---
def compute_streak_config(ticker: str):
    dataset__2__data_cache = {}
    dataset_ends_on = {}

    for dataset in ['day', 'week', 'month']:
        filename = get_filename_for_dataset(dataset)
        try:
            with open(filename, 'rb') as f:
                data_cache = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filename}")
        if ticker not in data_cache:
            raise KeyError(f"Ticker '{ticker}' not found in {filename}")
        dataset__2__data_cache[dataset] = data_cache
        last_date = data_cache[ticker].sort_index().index[-1]
        dataset_ends_on[dataset] = last_date

    dataset_2__info = {}
    now = datetime.now()
    skipped_datasets = {'day': False, 'week': False, 'month': False}  # Track skips

    for dataset in ['day', 'week', 'month']:
        data = dataset__2__data_cache[dataset][ticker]
        df = copy.deepcopy(data[[('Open', ticker), ('High', ticker), ('Low', ticker), ('Close', ticker)]])
        df.columns = ['Open', 'High', 'Low', 'Close']
        df.index.name = 'Date'
        df.sort_index(inplace=True)
        assert df.index.is_monotonic_increasing
        assert df.index[-1] > df.index[0]

        if dataset == 'day' and now < df.index[-1]:
            if skip_day_var.get():
                df = copy.deepcopy(df.iloc[:-1])
                skipped_datasets['day'] = True
        elif dataset == 'week' and now < df.index[-1]:
            if skip_week_var.get():
                df = copy.deepcopy(df.iloc[:-1])
                skipped_datasets['week'] = True
        elif dataset == 'month' and now < df.index[-1]:
            if skip_month_var.get():
                df = copy.deepcopy(df.iloc[:-1])
                skipped_datasets['month'] = True

        df['PrevClose'] = df['Close'].shift(1)
        df['UpDay'] = df['Close'] > df['PrevClose']
        df = df.dropna(subset=['PrevClose'])
        if df.empty:
            raise ValueError(f"DataFrame for {dataset} is empty after preprocessing.")

        last_idx = df.index[-1]
        direction_str = "positive" if df.loc[last_idx, 'UpDay'] else "negative"

        streak_length = 0
        n = len(df)
        for i in range(n):
            idx = df.index[-(i + 1)]
            if df.loc[idx, 'UpDay'] == (direction_str == "positive"):
                streak_length += 1
            else:
                break

        dataset_2__info[dataset] = {
            'direction': direction_str,
            'streak_length': streak_length,
        }

    return {
        'direction_day': 'pos' if dataset_2__info['day']['direction'] == 'positive' else 'neg',
        'direction_week': 'pos' if dataset_2__info['week']['direction'] == 'positive' else 'neg',
        'direction_month': 'pos' if dataset_2__info['month']['direction'] == 'positive' else 'neg',
        'nn_day': dataset_2__info['day']['streak_length'],
        'nn_week': dataset_2__info['week']['streak_length'],
        'nn_month': dataset_2__info['month']['streak_length'],
    }, skipped_datasets


# --- GUI Functions ---
def auto_detect():
    try:
        ticker = ticker_var.get().strip()
        if not ticker:
            messagebox.showwarning("Input Needed", "Please enter a ticker symbol first.")
            return

        config, skipped = compute_streak_config(ticker)

        # Update GUI variables
        dir_day_var.set("Positive" if config['direction_day'] == 'pos' else "Negative")
        dir_week_var.set("Positive" if config['direction_week'] == 'pos' else "Negative")
        dir_month_var.set("Positive" if config['direction_month'] == 'pos' else "Negative")

        nn_day_var.set(str(config['nn_day']))
        nn_week_var.set(str(config['nn_week']))
        nn_month_var.set(str(config['nn_month']))

        # Map timeframe keys to their skip vars for easy access
        skip_vars = {
            'day': skip_day_var,
            'week': skip_week_var,
            'month': skip_month_var
        }

        # Update indicator lights AND uncheck skip if light turns green
        for tf in ['day', 'week', 'month']:
            is_skipped = skipped[tf]
            color = "red" if is_skipped else "green"
            if skip_lights[tf] is not None:
                skip_lights[tf].config(fg=color)
                # 🔁 If light is green → data was NOT skipped → uncheck the checkbox
                if not is_skipped:
                    skip_vars[tf].set(False)  # uncheck

    except Exception as e:
        messagebox.showerror("Auto-Detect Error", f"Failed to compute streaks:\n{str(e)}")


def launch_analysis():
    try:
        ticker = ticker_var.get().strip()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        limit = int(limit_var.get())

        # Build configuration from GUI selections
        config = Namespace(
            ticker=ticker,
            limit=limit,
            direction_day='pos' if dir_day_var.get() == "Positive" else 'neg',
            direction_week='pos' if dir_week_var.get() == "Positive" else 'neg',
            direction_month='pos' if dir_month_var.get() == "Positive" else 'neg',
            nn_day=int(nn_day_var.get()),
            nn_week=int(nn_week_var.get()),
            nn_month=int(nn_month_var.get()),
            skip_last_day=skip_day_var.get(),
            skip_last_week=skip_week_var.get(),
            skip_last_month=skip_month_var.get(),
            verbose=True
        )

        # Call your core function
        streak_multi_timeframe(config)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to run analysis:\n{str(e)}")


if __name__ == "__main__":
    # --- GUI Setup ---
    root = tk.Tk()
    root.title("📈 Multi-Timeframe Streak Analyzer")
    root.geometry("620x580")
    root.resizable(True, True)

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Ticker & Limit
    ttk.Label(frame, text="Ticker:").grid(row=0, column=0, sticky=tk.W, pady=5)
    ticker_var = tk.StringVar(value="^GSPC")
    ttk.Entry(frame, textvariable=ticker_var, width=15).grid(row=0, column=1, sticky=tk.W, padx=10)

    ttk.Label(frame, text="Limit (recent points):").grid(row=1, column=0, sticky=tk.W, pady=5)
    limit_var = tk.StringVar(value="50")
    ttk.Entry(frame, textvariable=limit_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=10)

    # Auto-Detect Button
    ttk.Button(frame, text="🔍 Auto-Detect Streaks", command=auto_detect).grid(row=2, column=0, columnspan=2, pady=10)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)

    # Variables
    dir_day_var = tk.StringVar(value="Positive")
    dir_week_var = tk.StringVar(value="Positive")
    dir_month_var = tk.StringVar(value="Positive")

    nn_day_var = tk.StringVar(value="1")
    nn_week_var = tk.StringVar(value="1")
    nn_month_var = tk.StringVar(value="1")

    skip_day_var = tk.BooleanVar(value=True)
    skip_week_var = tk.BooleanVar(value=True)
    skip_month_var = tk.BooleanVar(value=True)

    timeframes = [
        ("Daily", dir_day_var, nn_day_var, skip_day_var),
        ("Weekly", dir_week_var, nn_week_var, skip_week_var),
        ("Monthly", dir_month_var, nn_month_var, skip_month_var)
    ]

    # Timeframe Controls with indicator lights
    for i, (label, dir_var, nn_var, skip_var) in enumerate(timeframes, start=4):
        tf_key = {'Daily': 'day', 'Weekly': 'week', 'Monthly': 'month'}[label]

        ttk.Label(frame, text=f"{label} Direction:").grid(row=i, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(frame, textvariable=dir_var, values=["Positive", "Negative"], state="readonly", width=10).grid(row=i, column=1, sticky=tk.W, padx=10)

        ttk.Label(frame, text=f"{label} Streak Length (nn):").grid(row=i+3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=nn_var, width=8).grid(row=i+3, column=1, sticky=tk.W, padx=10)

        # Frame for checkbox + light indicator
        skip_frame = ttk.Frame(frame)
        skip_frame.grid(row=i+6, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Checkbutton(skip_frame, text=f"Skip last {label.lower()} (if uncompleted)", variable=skip_var).pack(side=tk.LEFT)

        light_label = tk.Label(skip_frame, text="●", font=("Arial", 10), fg="gray")
        light_label.pack(side=tk.LEFT, padx=(5, 0))
        skip_lights[tf_key] = light_label

    # Launch Button
    ttk.Button(frame, text="🚀 Launch Multi Time Frame", command=launch_analysis).grid(row=16, column=0, columnspan=2, pady=20)

    # Grid weights for resizing
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    root.mainloop()