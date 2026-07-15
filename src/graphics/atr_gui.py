"""
Market Prediction GUI Dashboard
Requires: pip install customtkinter matplotlib pandas numpy optuna
Note: Ensure your original script is saved as 'runners.atr' or imported correctly.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import threading
import argparse
import numpy as np
import pandas as pd
import sys
import os

try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
# Import the core logic module
try:
    import runners.atr as mp
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Module Not Found", "Please ensure 'runners.atr' is correctly configured in your Python path.")
    sys.exit(1)
import traceback

# --- Monkey Patching to Capture Data ---
captured_data = {}
original_display_report = mp.display_report_with_vix
original_display_realtime = mp.display_realtime_prediction
original_display_dataset = getattr(mp, 'display_dataset_info', None)


def mock_display_report(global_metrics, regime_metrics, total_bars):
    captured_data['global_metrics'] = global_metrics
    captured_data['regime_metrics'] = regime_metrics
    captured_data['total_bars'] = total_bars
    original_display_report(global_metrics, regime_metrics, total_bars)


def mock_display_realtime(df_bt, vix_col, open_col, close_col, atr_col, high_col, low_col, ticker, optimized_params, use_close_for_range=False, verbose=True):
    captured_data['df_bt'] = df_bt
    captured_data['ticker'] = ticker
    captured_data['optimized_params'] = optimized_params
    captured_data['open_col'] = open_col
    captured_data['close_col'] = close_col
    captured_data['high_col'] = high_col
    captured_data['low_col'] = low_col
    captured_data['atr_col'] = atr_col
    captured_data['use_close_for_range'] = use_close_for_range
    captured_data['last_row'] = df_bt.iloc[-1]
    captured_data['last_date'] = df_bt.index[-1]

    # Determine regime safely
    vix_rank = df_bt.iloc[-1]['VIX_Rolling_Rank']
    if hasattr(vix_rank, '__len__') and not isinstance(vix_rank, str):
        vix_rank = vix_rank.values[0] if len(vix_rank) == 1 else vix_rank.iloc[0]

    if vix_rank < 0.30:
        regime = 'Low'
    elif vix_rank > 0.70:
        regime = 'High'
    else:
        regime = 'Normal'
    captured_data['regime'] = regime

    original_display_realtime(df_bt, vix_col, open_col, close_col, atr_col, high_col, low_col, ticker, optimized_params, use_close_for_range, verbose)


def mock_display_dataset_info(train_info, test_info, ticker, dataset_id, atr_window):
    captured_data['train_info'] = train_info
    captured_data['test_info'] = test_info
    captured_data['dataset_ticker'] = ticker
    captured_data['dataset_id'] = dataset_id
    captured_data['atr_window'] = atr_window
    if original_display_dataset:
        original_display_dataset(train_info, test_info, ticker, dataset_id, atr_window)


mp.display_report_with_vix = mock_display_report
mp.display_realtime_prediction = mock_display_realtime
mp.display_dataset_info = mock_display_dataset_info

# --- GUI Application ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Market Prediction Dashboard")
        self.geometry("1400x900")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL (SETTINGS) ---
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="⚙️ Parameters", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 20))

        # Variables
        self.ticker_var = ctk.StringVar(value="^GSPC")
        self.dataset_var = ctk.StringVar(value="day")
        self.realtime_var = ctk.BooleanVar(value=True)
        self.close_range_var = ctk.BooleanVar(value=False)
        self.atr_var = ctk.StringVar(value="14")
        self.trials_var = ctk.StringVar(value="50")  # Lower default for GUI speed

        # Inputs
        self.create_input("Ticker Symbol", self.ticker_var, 1)
        self.create_dropdown("Dataset Frequency", self.dataset_var, ["day", "week", "month"], 3)
        ctk.CTkCheckBox(self.sidebar, text="Use Real-time Data", variable=self.realtime_var).grid(row=5, column=0, padx=20, pady=10, sticky="w")
        ctk.CTkCheckBox(self.sidebar, text="Use Close for Range", variable=self.close_range_var).grid(row=6, column=0, padx=20, pady=10, sticky="w")
        self.create_input("ATR Window", self.atr_var, 8)

        # Sliders
        self.split_var = ctk.DoubleVar(value=0.80)
        self.sld_split = ctk.CTkSlider(self.sidebar, from_=0.5, to=0.9, command=self.update_split_label)
        self.sld_split.set(0.80)
        self.lbl_split = ctk.CTkLabel(self.sidebar, text="Train Split: 0.80")
        self.lbl_split.grid(row=10, column=0, padx=20, pady=(10, 0), sticky="w")
        self.sld_split.grid(row=11, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.tight_var = ctk.DoubleVar(value=0.3)
        self.sld_tight = ctk.CTkSlider(self.sidebar, from_=0.0, to=4.0, command=self.update_tight_label)
        self.sld_tight.set(0.3)
        self.lbl_tight = ctk.CTkLabel(self.sidebar, text="Tightness Weight: 0.30")
        self.lbl_tight.grid(row=12, column=0, padx=20, pady=(10, 0), sticky="w")
        self.sld_tight.grid(row=13, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.create_input("Optuna Trials", self.trials_var, 15)

        # Action Button
        self.run_btn = ctk.CTkButton(self.sidebar, text="🚀 Run Analysis", command=self.run_analysis, height=45, font=ctk.CTkFont(size=16, weight="bold"))
        self.run_btn.grid(row=17, column=0, padx=20, pady=(30, 10), sticky="ew")

        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready", text_color="gray")
        self.status_label.grid(row=18, column=0, padx=20, pady=10)

        # --- RIGHT PANEL (RESULTS) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)  # Chart row expands

        # Dataset Info Textbox
        self.dataset_frame = ctk.CTkFrame(self.main_frame)
        self.dataset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.dataset_text = ctk.CTkTextbox(self.dataset_frame, height=110, state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.dataset_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Metrics Cards
        self.metrics_frame = ctk.CTkFrame(self.main_frame)
        self.metrics_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.metrics_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.card1 = self.create_card(self.metrics_frame, "Test Hit Rate", "-- %", 0)
        self.card2 = self.create_card(self.metrics_frame, "Test High Bound Reliability", "-- %", 1)
        self.card3 = self.create_card(self.metrics_frame, "Test Low Bound Reliability", "-- %", 2)

        # Regime Textbox
        self.regime_frame = ctk.CTkFrame(self.main_frame)
        self.regime_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.regime_text = ctk.CTkTextbox(self.regime_frame, height=120, state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.regime_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Matplotlib Chart
        self.chart_frame = ctk.CTkFrame(self.main_frame)
        self.chart_frame.grid(row=3, column=0, sticky="nsew")
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tooltip state ---
        self.annot = None
        self.hover_line = None
        self.chart_x = None
        self.chart_highs = None
        self.chart_lows = None
        self.chart_closes = None
        self.chart_dates = None
        self.pred_high = None
        self.pred_low = None
        self.be_high = 0.0
        self.be_low = 0.0
        self._last_hover_idx = -1
        self.use_close_for_range = False

        # Connect mouse motion event once
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def create_input(self, label, var, row):
        ctk.CTkLabel(self.sidebar, text=label).grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        ctk.CTkEntry(self.sidebar, textvariable=var).grid(row=row + 1, column=0, padx=20, pady=(0, 10), sticky="ew")

    def create_dropdown(self, label, var, values, row):
        ctk.CTkLabel(self.sidebar, text=label).grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        ctk.CTkOptionMenu(self.sidebar, variable=var, values=values).grid(row=row + 1, column=0, padx=20, pady=(0, 10), sticky="ew")

    def create_card(self, parent, title, value, col):
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color="lightblue").pack(pady=(15, 5))
        lbl_value = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=28, weight="bold"))
        lbl_value.pack(pady=(0, 15))
        return lbl_value

    def update_split_label(self, value):
        self.lbl_split.configure(text=f"Train Split: {float(value):.2f}")

    def update_tight_label(self, value):
        self.lbl_tight.configure(text=f"Tightness Weight: {float(value):.2f}")

    def run_analysis(self):
        self.run_btn.configure(state="disabled", text="Running...")
        self.status_label.configure(text="Fetching data & optimizing...", text_color="orange")
        self.clear_results()

        global captured_data
        captured_data = {}

        thread = threading.Thread(target=self.worker)
        thread.start()

    def worker(self):
        try:
            args = argparse.Namespace(
                dataset_id=self.dataset_var.get(),
                ticker=self.ticker_var.get().upper(),
                n_trials=int(self.trials_var.get()),
                use_realtime_data=self.realtime_var.get(),
                atr_window=int(self.atr_var.get()),
                n_split=self.sld_split.get(),
                tightness_weight=self.sld_tight.get(),
                use_close_for_range=self.close_range_var.get(),
                dataframe=None, verbose=True,
            )
            mp.entry(args)
            self.after(0, self.update_ui_success)
        except Exception as e:
            traceback.print_exc()
            self.after(0, self.update_ui_error, str(e))

    def clear_results(self):
        self.card1.configure(text="-- %")
        self.card2.configure(text="-- %")
        self.card3.configure(text="-- %")

        self.dataset_text.configure(state="normal")
        self.dataset_text.delete("1.0", tk.END)
        self.dataset_text.configure(state="disabled")

        self.regime_text.configure(state="normal")
        self.regime_text.delete("1.0", tk.END)
        self.regime_text.configure(state="disabled")

        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_title("Waiting for data...", color='white')
        self.canvas.draw()
        # Reset tooltip state
        self.chart_x = None
        self.annot = None
        self.hover_line = None

    def update_ui_success(self):
        self.run_btn.configure(state="normal", text="🚀 Run Analysis")
        self.status_label.configure(text="Analysis Complete!", text_color="green")

        if not captured_data: return

        # Dataset Info
        train_info = captured_data.get('train_info', {})
        test_info = captured_data.get('test_info', {})
        ticker = captured_data.get('dataset_ticker', 'N/A')
        dataset_id = captured_data.get('dataset_id', 'N/A')
        atr_window = captured_data.get('atr_window', 'N/A')

        self.dataset_text.configure(state="normal")
        self.dataset_text.delete("1.0", tk.END)
        info_text = (
            f"📂 DATASET SPLIT: {ticker} ({dataset_id.upper()}) | ATR Window: {atr_window}\n"
            f"{'─' * 76}\n"
            f"🟢 TRAIN SET : {train_info.get('start_date', '')} ➔ {train_info.get('end_date', '')} | "
            f"{train_info.get('bars', 0)} bars ({train_info.get('split_ratio', 0):.0%})\n"
            f"🔵 TEST SET  : {test_info.get('start_date', '')} ➔ {test_info.get('end_date', '')} | "
            f"{test_info.get('bars', 0)} bars ({test_info.get('split_ratio', 0):.0%})"
        )
        self.dataset_text.insert(tk.END, info_text)
        self.dataset_text.configure(state="disabled")

        gm = captured_data.get('global_metrics', {})
        self.card1.configure(text=f"{gm.get('Hit Rate Global (Range Tenu)', 0):.2f} %")
        self.card2.configure(text=f"{gm.get('Borne Haute Respectée', 0):.2f} %")
        self.card3.configure(text=f"{gm.get('Borne Basse Respectée', 0):.2f} %")

        rm = captured_data.get('regime_metrics', {})
        self.regime_text.configure(state="normal")
        self.regime_text.delete("1.0", tk.END)
        text = f"{'REGIME':<10} | {'HIT RATE':<10} | {'HIGH REL':<10} | {'LOW REL':<10} | {'K_UP':<5} | {'K_DOWN':<5} | {'BARS'}\n" + "-" * 80 + "\n"
        for regime, metrics in rm.items():
            text += f"{regime:<10} | {metrics['Hit Rate Global (Range Tenu)']:>6.2f} %   | {metrics['Borne Haute Respectée']:>6.2f} %   | {metrics['Borne Basse Respectée']:>6.2f} %   | {metrics['k_up']:>5.3f} | {metrics['k_down']:>5.3f}  | {metrics['Count']}\n"
        self.regime_text.insert(tk.END, text)
        self.regime_text.configure(state="disabled")

        self.draw_chart()

    def update_ui_error(self, err_msg):
        self.run_btn.configure(state="normal", text="🚀 Run Analysis")
        self.status_label.configure(text=f"Error: {err_msg}", text_color="red")

    def draw_chart(self):
        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')

        if 'df_bt' not in captured_data or captured_data['df_bt'].empty: return

        df = captured_data['df_bt'].tail(30)
        open_col = captured_data['open_col']
        close_col = captured_data['close_col']
        high_col = captured_data['high_col']
        low_col = captured_data['low_col']
        last_row = captured_data['last_row']
        opt_params = captured_data['optimized_params']
        regime = captured_data['regime']
        self.use_close_for_range = captured_data.get('use_close_for_range', False)

        k_up = opt_params[f'k_up_{regime.lower()}']
        k_down = opt_params[f'k_down_{regime.lower()}']

        current_open = last_row[open_col]
        current_atr = last_row[captured_data['atr_col']]
        pred_high = current_open + (current_atr * k_up)
        pred_low = current_open - (current_atr * k_down)

        act_high = last_row[high_col]
        act_low = last_row[low_col]
        act_close = last_row[close_col]

        x = np.arange(len(df))
        opens = df[open_col].values
        highs = df[high_col].values
        lows = df[low_col].values
        closes = df[close_col].values

        # Draw historical candles
        for i in range(len(df) - 1):
            self.ax.plot([x[i], x[i]], [lows[i], highs[i]], color='gray', linewidth=1.5)
            self.ax.scatter(x[i], opens[i], color='white', s=15, zorder=5)

        # Prediction Box
        i = len(df) - 1
        rect = Rectangle((x[i] - 0.4, pred_low), 0.8, pred_high - pred_low, color='cyan', alpha=0.2, zorder=2, label='Predicted Range')
        self.ax.add_patch(rect)
        self.ax.plot([x[i] - 0.5, x[i] + 0.5], [pred_high, pred_high], color='cyan', linestyle='--', linewidth=2)
        self.ax.plot([x[i] - 0.5, x[i] + 0.5], [pred_low, pred_low], color='cyan', linestyle='--', linewidth=2)

        # Last Bar Actuals
        self.ax.plot([x[i], x[i]], [lows[i], highs[i]], color='orange', linewidth=2)
        self.ax.scatter(x[i], opens[i], color='yellow', s=30, zorder=6)

        if self.use_close_for_range:
            if pd.notna(act_close):
                self.ax.scatter(x[i], act_close, color='magenta', s=60, marker='o', zorder=7, label=f'Actual Close ({act_close:.2f})')
        else:
            if pd.notna(act_high): self.ax.scatter(x[i], act_high, color='red', s=60, marker='v', zorder=7, label=f'Actual High ({act_high:.2f})')
            if pd.notna(act_low): self.ax.scatter(x[i], act_low, color='green', s=60, marker='^', zorder=7, label=f'Actual Low ({act_low:.2f})')

        # Formatting
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([dt.strftime('%m-%d') for dt in df.index], rotation=45, ha='right', fontsize=9, color='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.set_ylabel("Price", color='white', fontsize=12)
        self.ax.set_title(f"Real-Time Prediction | {captured_data['ticker']} | Regime: {regime.upper()} VIX", color='white', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.2, color='gray')
        self.ax.legend(loc='upper left', facecolor='#2b2b2b', edgecolor='white', labelcolor='white')

        # --- Store data for tooltip ---
        self.chart_x = x
        self.chart_highs = highs
        self.chart_lows = lows
        self.chart_closes = closes
        self.chart_dates = [dt.strftime('%Y-%m-%d') for dt in df.index]
        self.pred_high = pred_high
        self.pred_low = pred_low
        self._last_hover_idx = -1

        # --- Calculate Break-even Premiums ---
        rm = captured_data.get('regime_metrics', {})
        metrics = rm.get(regime, {})

        # Use reliability (high/low bound respected) as win_rate, falling back to global hit rate
        win_rate_high = metrics.get('Borne Haute Respectée', metrics.get('Hit Rate Global (Range Tenu)', 0)) / 100.0
        win_rate_low = metrics.get('Borne Basse Respectée', metrics.get('Hit Rate Global (Range Tenu)', 0)) / 100.0

        spread_width = 500.0
        self.be_high = (1.0 - win_rate_high) * spread_width
        self.be_low = (1.0 - win_rate_low) * spread_width

        # --- Create tooltip annotation ---
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(-164, 20),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc="#1a1a2e",
                ec="#00d4ff",
                lw=1.5,
                alpha=0.92
            ),
            fontsize=9,
            color='#e0e0e0',
            fontfamily='monospace',
            linespacing=1.5,
            zorder=100
        )
        self.annot.set_visible(False)

        # --- Create hover indicator line ---
        self.hover_line = self.ax.axvline(
            x=-1, color='#00d4ff', linestyle=':', alpha=0, linewidth=0.8, zorder=50
        )

        self.fig.tight_layout()
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Handle mouse motion over the chart to display bar tooltips."""
        # Guard: no data loaded yet
        if self.chart_x is None or self.annot is None:
            return

        # Mouse left the axes area → hide tooltip
        if event.inaxes != self.ax:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                if self.hover_line:
                    self.hover_line.set_alpha(0)
                self.canvas.draw_idle()
            return

        mouse_x = event.xdata
        if mouse_x is None:
            return

        # Snap to nearest bar
        idx = int(round(mouse_x))

        # Out of range or too far from any bar center
        if idx < 0 or idx >= len(self.chart_x) or abs(mouse_x - idx) > 0.5:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                if self.hover_line:
                    self.hover_line.set_alpha(0)
                self.canvas.draw_idle()
            return

        # Update hover indicator line
        if self.hover_line:
            self.hover_line.set_xdata([idx, idx])
            self.hover_line.set_alpha(0.4)

        # --- Build tooltip text ---
        date_str = self.chart_dates[idx]

        if idx == len(self.chart_x) - 1:
            # Last bar: show date + actual high/low/close + predicted high/low + BE Premiums
            if self.use_close_for_range:
                text = (
                    f"  Date: {date_str}\n"
                    f"  ─────────────────\n"
                    f"  Actual Close: {self.chart_closes[idx]:>10.2f}\n"
                    f"  ─────────────────\n"
                    f"  Pred High   : {self.pred_high:>10.2f}\n"
                    f"  Pred Low    : {self.pred_low:>10.2f}\n"
                    f"  ─────────────────\n"
                    f"  BE Prem (H) : {self.be_high:>10.2f} $\n"
                    f"  BE Prem (L) : {self.be_low:>10.2f} $"
                )
            else:
                text = (
                    f"  Date: {date_str}\n"
                    f"  ─────────────────\n"
                    f"  Actual High : {self.chart_highs[idx]:>10.2f}\n"
                    f"  Actual Low  : {self.chart_lows[idx]:>10.2f}\n"
                    f"  ─────────────────\n"
                    f"  Pred High   : {self.pred_high:>10.2f}\n"
                    f"  Pred Low    : {self.pred_low:>10.2f}\n"
                    f"  ─────────────────\n"
                    f"  BE Prem (H) : {self.be_high:>10.2f} $\n"
                    f"  BE Prem (L) : {self.be_low:>10.2f} $"
                )
        else:
            # Other bars: show date + high + low
            text = (
                f"  Date: {date_str}\n"
                f"  ─────────────────\n"
                f"  High : {self.chart_highs[idx]:>10.2f}\n"
                f"  Low  : {self.chart_lows[idx]:>10.2f}"
            )

        self.annot.set_text(text)
        self.annot.xy = (idx, event.ydata)
        self.annot.set_visible(True)

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = App()
    app.mainloop()