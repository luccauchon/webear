import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json

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

class HalfTrendConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HalfTrend Strategy Configurator")
        self.root.geometry("1024x768")
        self.config = HALF_TREND_DEFAULT_CONFIG.copy()

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
                'vix_dataframe': None,  # Not editable in GUI (dataframe)
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
                # messagebox.showinfo("Success", "Config saved successfully!")
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
                # messagebox.showinfo("Success", "Config loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load:\n{e}")

    def reset_to_default(self):
        self.apply_config(HALF_TREND_DEFAULT_CONFIG)
        messagebox.showinfo("Reset", "Configuration reset to default.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HalfTrendConfigGUI(root)
    root.mainloop()