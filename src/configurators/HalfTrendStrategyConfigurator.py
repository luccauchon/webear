try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # print(parent_dir)
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import tkinter as tk
import copy
from tkinter import ttk, messagebox, filedialog
import json
import argparse
import sys

# --- Your config (copied for reference) ---
HALF_TREND_DEFAULT_CONFIG = {
    'use__entry_type': 'Hard',
    'use__volume_confirmed': {'enable': False, 'period': 20},
    'use__higher_timeframe_strong_trend': {'enable': False, 'length': 21, 'min_rate': 0.95},
    'use__relative_strength_vs_benchmark': {
        'enable_vix': False,
        'enable_spx': False,
        'period_vix': 10*7,
        'period_spx': 200*7,
        'vix_dataframe': None,
        'spx_dataframe': None
    },
    'use__candlestick_confirmation_pattern': {'enable': False}
}

def set_nested_value(config, flat_key, value_str):
    """
    Set a value in nested dict using flat key with '__' as separator.
    Tries to auto-convert value_str to bool/int/float if possible.
    """
    keys = flat_key.split('__')
    current = config

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            raise KeyError(f"Key path invalid: {'__'.join(keys)} (missing '{key}')")
        current = current[key]

    final_key = keys[-1]

    if final_key not in current:
        raise KeyError(f"Leaf key '{final_key}' not found in config path: {'__'.join(keys)}")

    # Auto-parse value
    if value_str.lower() in ('true', 'false'):
        value = value_str.lower() == 'true'
    else:
        try:
            # Try int first
            value = int(value_str)
        except ValueError:
            try:
                # Then float
                value = float(value_str)
            except ValueError:
                # Keep as string
                value = value_str

    current[final_key] = value

def parse_args_into_config(default_config):
    parser = argparse.ArgumentParser(description="HalfTrend Strategy Configurator")
    parser.add_argument(
        '--config',
        type=str,
        help="Path to a JSON config file to load initially"
    )
    # Allow any --key__subkey value
    parser.add_argument('overrides', nargs='*', help="Override config keys using flat notation, e.g., -- use__entry_type Hard use__volume_confirmed__enable true")

    # Parse known args to avoid tkinter interference
    args, unknown = parser.parse_known_args()

    config = copy.deepcopy(default_config)

    # Load from file if specified
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
            # Deep merge? For simplicity, we'll just update top-level keys
            config.update(file_config)
        except Exception as e:
            print(f"Error loading config file: {e}", file=sys.stderr)

    # Handle overrides: they come as a list like ['key1', 'val1', 'key2', 'val2']
    if args.overrides:
        if len(args.overrides) % 2 != 0:
            raise ValueError("Overrides must be in key-value pairs.")
        for i in range(0, len(args.overrides), 2):
            key = args.overrides[i]
            val = args.overrides[i+1]
            try:
                set_nested_value(config, key, val)
            except Exception as e:
                print(f"Warning: failed to set {key}={val}: {e}", file=sys.stderr)

    return config

class HalfTrendConfigGUI:
    def __init__(self, root, initial_config):
        self.root = root
        self.root.title("HalfTrend Strategy Configurator")
        self.root.geometry("1024x768")
        self.config = initial_config  # Now comes from args

        # Main frame with scroll support
        canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Build UI
        self.entries = {}
        self.build_ui(scrollable_frame)

        # Buttons at bottom
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Load Config", command=self.load_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save Config", command=self.save_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset to Default", command=self.reset_to_default).pack(side="left", padx=5)

    def build_ui(self, parent):
        row = 0

        # Entry Type
        ttk.Label(parent, text="Entry Type:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        self.entry_type_var = tk.StringVar(value=self.config['use__entry_type'])
        ttk.Combobox(parent, textvariable=self.entry_type_var, values=["Soft", "Hard"], state="readonly", width=10).grid(row=row, column=1, sticky="w")
        row += 1

        # Volume Confirmed
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        ttk.Label(parent, text="Volume Confirmation").grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(0,5))
        row += 1

        self.vol_enable = tk.BooleanVar(value=self.config['use__volume_confirmed']['enable'])
        ttk.Checkbutton(parent, text="Enable", variable=self.vol_enable).grid(row=row, column=0, sticky="w", padx=20)
        ttk.Label(parent, text="Period (hours):").grid(row=row, column=1, sticky="e")
        self.vol_period = tk.IntVar(value=self.config['use__volume_confirmed']['period'])
        ttk.Entry(parent, textvariable=self.vol_period, width=8).grid(row=row, column=2, sticky="w")
        row += 1

        # Higher Timeframe Strong Trend
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        ttk.Label(parent, text="Higher Timeframe Strong Trend").grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(0,5))
        row += 1

        self.ht_enable = tk.BooleanVar(value=self.config['use__higher_timeframe_strong_trend']['enable'])
        ttk.Checkbutton(parent, text="Enable", variable=self.ht_enable).grid(row=row, column=0, sticky="w", padx=20)
        ttk.Label(parent, text="Length (hours):").grid(row=row+1, column=1, sticky="e")
        ttk.Label(parent, text="Min Rate:").grid(row=row+2, column=1, sticky="e")

        self.ht_length = tk.IntVar(value=self.config['use__higher_timeframe_strong_trend']['length'])
        self.ht_min_rate = tk.DoubleVar(value=self.config['use__higher_timeframe_strong_trend']['min_rate'])

        ttk.Entry(parent, textvariable=self.ht_length, width=8).grid(row=row+1, column=2, sticky="w")
        ttk.Entry(parent, textvariable=self.ht_min_rate, width=8).grid(row=row+2, column=2, sticky="w")
        row += 3

        # Relative Strength vs Benchmark
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        ttk.Label(parent, text="Relative Strength vs Benchmark").grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(0,5))
        row += 1

        self.rs_vix = tk.BooleanVar(value=self.config['use__relative_strength_vs_benchmark']['enable_vix'])
        self.rs_spx = tk.BooleanVar(value=self.config['use__relative_strength_vs_benchmark']['enable_spx'])
        ttk.Checkbutton(parent, text="Enable VIX Filter", variable=self.rs_vix).grid(row=row, column=0, sticky="w", padx=20)
        ttk.Checkbutton(parent, text="Enable SPX Filter", variable=self.rs_spx).grid(row=row+1, column=0, sticky="w", padx=20)

        ttk.Label(parent, text="VIX Period (hours):").grid(row=row, column=1, sticky="e")
        ttk.Label(parent, text="SPX Period (hours):").grid(row=row+1, column=1, sticky="e")

        self.vix_period = tk.IntVar(value=self.config['use__relative_strength_vs_benchmark']['period_vix'])
        self.spx_period = tk.IntVar(value=self.config['use__relative_strength_vs_benchmark']['period_spx'])

        ttk.Entry(parent, textvariable=self.vix_period, width=8).grid(row=row, column=2, sticky="w")
        ttk.Entry(parent, textvariable=self.spx_period, width=8).grid(row=row+1, column=2, sticky="w")
        row += 2

        # Candlestick Confirmation
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        ttk.Label(parent, text="Candlestick Confirmation Pattern").grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(0,5))
        row += 1

        self.candle_enable = tk.BooleanVar(value=self.config['use__candlestick_confirmation_pattern']['enable'])
        ttk.Checkbutton(parent, text="Enable", variable=self.candle_enable).grid(row=row, column=0, sticky="w", padx=20)
        row += 1

    def collect_config(self):
        return {
            'use__entry_type': self.entry_type_var.get(),
            'use__volume_confirmed': {
                'enable': self.vol_enable.get(),
                'period': self.vol_period.get()
            },
            'use__higher_timeframe_strong_trend': {
                'enable': self.ht_enable.get(),
                'length': self.ht_length.get(),
                'min_rate': self.ht_min_rate.get()
            },
            'use__relative_strength_vs_benchmark': {
                'enable_vix': self.rs_vix.get(),
                'enable_spx': self.rs_spx.get(),
                'period_vix': self.vix_period.get(),
                'period_spx': self.spx_period.get(),
                'vix_dataframe': None,
                'spx_dataframe': None
            },
            'use__candlestick_confirmation_pattern': {
                'enable': self.candle_enable.get()
            }
        }

    def apply_config(self, cfg):
        try:
            self.entry_type_var.set(cfg['use__entry_type'])
            vc = cfg['use__volume_confirmed']
            self.vol_enable.set(vc['enable'])
            self.vol_period.set(vc['period'])

            ht = cfg['use__higher_timeframe_strong_trend']
            self.ht_enable.set(ht['enable'])
            self.ht_length.set(ht['length'])
            self.ht_min_rate.set(ht['min_rate'])

            rs = cfg['use__relative_strength_vs_benchmark']
            self.rs_vix.set(rs['enable_vix'])
            self.rs_spx.set(rs['enable_spx'])
            self.vix_period.set(rs['period_vix'])
            self.spx_period.set(rs['period_spx'])

            cp = cfg['use__candlestick_confirmation_pattern']
            self.candle_enable.set(cp['enable'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    def save_config(self):
        cfg = self.collect_config()
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(cfg, f, indent=4)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{e}")

    def load_config(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    cfg = json.load(f)
                self.apply_config(cfg)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load:\n{e}")

    def reset_to_default(self):
        self.apply_config(HALF_TREND_DEFAULT_CONFIG)
        messagebox.showinfo("Reset", "Configuration reset to default.")

if __name__ == "__main__":
    try:
        initial_config = parse_args_into_config(HALF_TREND_DEFAULT_CONFIG)
    except Exception as e:
        print(f"Argument parsing error: {e}", file=sys.stderr)
        initial_config = HALF_TREND_DEFAULT_CONFIG.copy()

    root = tk.Tk()
    app = HalfTrendConfigGUI(root, initial_config)
    root.mainloop()