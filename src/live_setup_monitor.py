import tkinter as tk
from datetime import date
import matplotlib.pyplot as plt
from datetime import timedelta
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import threading
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import time
import requests
from bs4 import BeautifulSoup
from algorithms import trade_prime_half_trend_strategy, trade_prime_half_trend_strategy_plus_volume_confirmation_and_atr_stop_loss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

def get_all_tickers():
    sp500 = []#get_sp500_tickers()
    custom = ["^GSPC", "SPY", "QQQ", "ADBE", "AMD", "AMZN", "AAPL", "COST", "DBRG", "GOOG", "INTC",
              "MA", "META", "MSFT", "NFLX", "NVDA", "ORCL", "PINS", "PLTR", "RDDT",
              "RKT", "TSLA", "V", "WMT"]
    return list(set(sp500 + custom))

# -------------------------------
# Main App Class
# -------------------------------

class SetupMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 Live Buy/Sell Setup Monitor")
        self.root.geometry("1400x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.is_running = False
        self.data_cache = {}
        self.setup_log = []  # List of dicts: {'time', 'ticker', 'type'}

        self.create_widgets()
        self.tickers = get_all_tickers()
        print(f"Loaded {len(self.tickers)} tickers.")

    def create_widgets(self):
        # Control Frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10, fill='x', padx=10)

        self.start_btn = ttk.Button(control_frame, text="▶ Start", command=self.start_monitoring)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", command=self.stop_monitoring, state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        # Algorithm Selection
        ttk.Label(control_frame, text="Algo:").pack(side='left', padx=(10, 2))
        self.selected_algo = tk.StringVar(value="Half Trend")
        algo_options = ["Half Trend", "Half Trend + Volume & ATR"]
        self.algo_combo = ttk.Combobox(control_frame, textvariable=self.selected_algo, values=algo_options, width=25, state="readonly")
        self.algo_combo.pack(side='left', padx=5)

        # Hard/Soft Entry Toggle
        self.entry_mode = tk.StringVar(value="Hard")
        self.entry_btn = ttk.Button(
            control_frame,
            text=f"Entry: {self.entry_mode.get()}",
            command=self.toggle_entry_mode,
            width=12
        )
        self.entry_btn.pack(side='left', padx=5)

        self.status_label = ttk.Label(control_frame, text="Idle", foreground="gray")
        self.status_label.pack(side='left', padx=10)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # Alerts Tab
        self.alerts_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.alerts_frame, text="🔔 Alerts")

        self.alerts_text = tk.Text(self.alerts_frame, wrap='word', font=("Consolas", 14))
        self.alerts_scroll = ttk.Scrollbar(self.alerts_frame, orient='vertical', command=self.alerts_text.yview)
        self.alerts_text.configure(yscrollcommand=self.alerts_scroll.set)
        self.alerts_text.pack(side='left', fill='both', expand=True)
        self.alerts_scroll.pack(side='right', fill='y')

        # Chart Tab
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="📊 Chart")

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Bind double-click on alerts to show chart
        self.alerts_text.bind("<Double-1>", self.on_alert_double_click)

    def toggle_entry_mode(self):
        current = self.entry_mode.get()
        new_mode = "Soft" if current == "Hard" else "Hard"
        self.entry_mode.set(new_mode)
        self.entry_btn.config(text=f"Entry: {new_mode}")

    def start_monitoring(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Running...", foreground="green")
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Stopped", foreground="red")

    def on_closing(self):
        self.is_running = False
        self.root.destroy()

    def monitor_loop(self):
        max_time = 15 * 60
        while self.is_running:
            self.status_label.config(text=f"Running...", foreground="green")
            self.check_setups()
            for st in range(max_time):
                if not self.is_running:
                    break
                time.sleep(1)
                self.status_label.config(text=f"Waiting {max_time-st} seconds...", foreground="green")

    def check_setups(self, use_yahoo=True):
        end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=62)).strftime('%Y-%m-%d')
        today = datetime.today().date()
        algo_name = self.selected_algo.get()
        entry_type = self.entry_mode.get()  # "Hard" or "Soft"

        for e, ticker in enumerate(self.tickers):
            if not self.is_running and use_yahoo:
                break
            try:
                if use_yahoo or ticker not in self.data_cache:
                    data = yf.download(ticker, start=start_date, end=end_date, interval='1h', auto_adjust=False, ignore_tz=True)
                    self.data_cache[ticker] = data
                else:
                    data = self.data_cache[ticker]
                self.status_label.config(text=f"Running... (Last value:{data.index[-1]}) [{ticker}, {e+1}/{len(self.tickers)}]", foreground="green")
                if use_yahoo:
                    time.sleep(0.3)

                for buy_setup in [True, False]:
                    df = trade_prime_half_trend_strategy(ticker=self.data_cache[ticker].copy(), ticker_name=ticker, buy_setup=buy_setup)
                    if algo_name == "Half Trend":
                        df = trade_prime_half_trend_strategy(ticker=self.data_cache[ticker].copy(), ticker_name=ticker, buy_setup=buy_setup)
                    elif algo_name == "Half Trend + Volume & ATR":
                        df = trade_prime_half_trend_strategy_plus_volume_confirmation_and_atr_stop_loss(ticker=self.data_cache[ticker].copy(), ticker_name=ticker, buy_setup=buy_setup)
                    recent_signals = df[df[('custom_signal', ticker)]]
                    if not recent_signals.empty:
                        # last = recent_signals[[('Close', ticker), ('triggered_distance', ticker), ('custom_signal', ticker)]].iloc[-1]
                        last = recent_signals.iloc[-1]
                        if last.name.date() == today:
                            alert = {'time': last.name, 'ticker': ticker, 'type': 'BUY' if buy_setup else 'SELL', 'algo_name': algo_name, 'close':last[('Close', ticker)],
                                     'distance': last[('triggered_distance', ticker)], 'entry_point': last[('custom_signal', ticker)],
                                     'stop_loss': last[('stop_loss', ticker)], 'take_profit': last[('take_profit', ticker)]}
                            self.log_alert(alert)

            except Exception as e:
                print(f"Error with {ticker}: {e}")
                time.sleep(1)

        self.update_alerts_display()

    def log_alert(self, alert):
        # Avoid duplicates in last 1 hour
        now = datetime.now()
        if any(
            a['ticker'] == alert['ticker'] and a['type'] == alert['type'] and a['distance'] == alert['distance'] and a['algo_name'] == alert['algo_name']
            for a in self.setup_log
        ):
            return

        self.setup_log.append(alert)
        #self.update_alerts_display()

    def update_alerts_display(self):
        self.root.after(0, self._update_alerts_ui)

    def _update_alerts_ui(self):
        self.alerts_text.delete(1.0, tk.END)
        # Configure tags for colors
        self.alerts_text.tag_config('buy', foreground='green')
        self.alerts_text.tag_config('sell', foreground='red')
        today = date.today()
        for alert in sorted(self.setup_log, key=lambda x: x['time'], reverse=True):
            color = "green" if alert['type'] == "BUY" else "red"
            #the_time = (alert['time'] + timedelta(hours=0)).strftime('%Y-%m-%d %H:%M')
            the_time = (alert['time'] + timedelta(hours=0)).strftime('%Y-%m-%d')
            triggered_at = (alert['time'] - timedelta(hours=int(alert['distance']))).strftime('%H:%M')
            signal_at = (alert['time']).strftime('%H:%M')
            line = (f"[{the_time}  >>{signal_at}][{alert['algo_name']}] {alert['type']:>4} → {alert['ticker']} , dst:{alert['distance']} hours >> "
                    f"Entry:{alert['close']:.2f}  SL:{alert['stop_loss']:.2f}  TP:{alert['take_profit']:.2f}\n")
            # self.alerts_text.insert(tk.END, line)
            # Insert text with tags
            if alert['type'] == "BUY":
                self.alerts_text.insert(tk.END, line, 'buy')
            else:
                self.alerts_text.insert(tk.END, line, 'sell')
        self.alerts_text.see(tk.END)

    def on_alert_double_click(self, event):
        try:
            index = self.alerts_text.index(f"@{event.x},{event.y}")
            line = self.alerts_text.get(index + " linestart", index + " lineend")
            parts = line.split("→")
            if len(parts) == 2:
                ticker = parts[1].strip().split(',')[0].strip()
                self.plot_ticker(ticker)
        except Exception as e:
            print(f"Chart error: {e}")

    def plot_ticker(self, ticker):
        if ticker not in self.data_cache:
            return
        data = self.data_cache[ticker].copy()
        open = ('Open', ticker.upper())
        high = ('High', ticker.upper())
        low = ('Low', ticker.upper())
        close = ('Close', ticker.upper())
        volume = ('Volume', ticker.upper())

        self.ax.clear()
        self.ax.plot(data.index, data['Close'], label='Close', color='blue')
        self.ax.set_title(f"{ticker} - 1H Chart")
        self.ax.grid(True)
        self.ax.legend()
        self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %Hh'))
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        self.canvas.draw()

# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = SetupMonitorApp(root)
    root.mainloop()