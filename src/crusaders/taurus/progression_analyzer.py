import sys
import os
import json
import glob
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
                               QStatusBar, QFrame, QComboBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ==========================================
# Core Analysis Logic (Backend)
# ==========================================
class TaurusAnalyzer:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.daily_data = {}
        self.load_files()

    def load_files(self):
        pattern = os.path.join(self.dir_path, "taurus_visualization_day_*.json")
        files = glob.glob(pattern)
        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                now_str = data.get("now", "")
                if not now_str: continue
                current_date = self.parse_date(now_str)
                self.daily_data[current_date] = data
            except Exception:
                pass

    def parse_date(self, date_str):
        for fmt in ("%Y_%m_%d", "%Y.%m.%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Date format not recognized: {date_str}")

    def analyze(self, start_date_str, target_price, future_bar, optimize_type, indicator_name):
        try:
            start_date = self.parse_date(start_date_str)
        except ValueError:
            raise ValueError("Invalid start date format. Use YYYY-MM-DD, YYYY.MM.DD, or YYYY_MM_DD.")

        if start_date not in self.daily_data:
            raise ValueError(f"Start date {start_date_str} not found in loaded files.")

        start_data = self.daily_data[start_date]
        bar_str = str(future_bar)
        if bar_str not in start_data:
            raise ValueError(f"Future bar {future_bar} not found in start date file.")

        bar_data = start_data[bar_str]
        candidates = []

        indicator_lower = indicator_name.lower()

        # Only look in the selected optimize_type block AND matching indicator
        if optimize_type in bar_data:
            for thresh, preds in bar_data[optimize_type].items():
                for pred in preds:
                    if pred.get("indicator", "").lower() == indicator_lower:
                        candidates.append(pred)

        if not candidates:
            raise ValueError(f"No '{optimize_type}' predictions found for indicator '{indicator_name}' on the start date.")

        def get_target_price(pred):
            try:
                return float(str(pred.get("target_price_str", "0")).replace(",", ""))
            except:
                return 0.0

        anchor = min(candidates, key=lambda p: abs(get_target_price(p) - target_price))
        anchor_target_date_str = anchor.get("target_date_str", "")
        if not anchor_target_date_str:
            raise ValueError("Anchor prediction missing 'target_date_str'.")

        anchor_target_date = self.parse_date(anchor_target_date_str)

        sorted_dates = sorted([d for d in self.daily_data.keys() if start_date <= d <= anchor_target_date])

        progression_dates = []
        progression_rates = []
        progression_prices = []

        def get_val_win_rate(pred):
            try:
                return float(str(pred.get("val_win_rate_str", "0")).replace("%", ""))
            except:
                return 0.0

        for d in sorted_dates:
            data = self.daily_data[d]
            cands = []
            for i in range(1, 21):
                b = str(i)
                if b in data:
                    # Only look in the selected optimize_type block AND matching indicator
                    if optimize_type in data[b]:
                        for thresh, preds in data[b][optimize_type].items():
                            for pred in preds:
                                if pred.get("target_date_str") == anchor_target_date_str:
                                    if pred.get("indicator", "").lower() == indicator_lower:
                                        cands.append(pred)

            if not cands: continue

            best = min(cands, key=lambda p: abs(get_target_price(p) - target_price))

            progression_dates.append(d)
            progression_rates.append(get_val_win_rate(best))
            progression_prices.append(get_target_price(best))

        if not progression_dates:
            raise ValueError(f"No progression data found for '{optimize_type}' / '{indicator_name}'.")

        return progression_dates, progression_rates, progression_prices, anchor_target_date_str


# ==========================================
# Matplotlib Canvas for PySide6
# ==========================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#1e1e2e')
        super().__init__(fig)
        self.figure = fig


# ==========================================
# Main GUI Window
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taurus Progression Analyzer")
        self.resize(1200, 850)

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Top Panel for inputs (Card Style)
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QVBoxLayout(input_frame)

        # Row 1: Directory
        row1 = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select directory containing JSON files...")
        self.dir_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse Directory")
        browse_btn.clicked.connect(self.browse_directory)
        row1.addWidget(QLabel("Directory:"))
        row1.addWidget(self.dir_edit, 1)
        row1.addWidget(browse_btn)
        input_layout.addLayout(row1)

        # Row 2: Date, Price, Bar
        row2 = QHBoxLayout()
        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("YYYY-MM-DD")
        row2.addWidget(QLabel("Start Date (t):"))
        row2.addWidget(self.date_edit)

        self.price_spin = QDoubleSpinBox()
        self.price_spin.setRange(0, 1000000)
        self.price_spin.setDecimals(2)
        self.price_spin.setValue(7400.00)
        row2.addWidget(QLabel("Target Price:"))
        row2.addWidget(self.price_spin)

        self.bar_spin = QSpinBox()
        self.bar_spin.setRange(1, 20)
        self.bar_spin.setValue(1)
        row2.addWidget(QLabel("Future Bar (N):"))
        row2.addWidget(self.bar_spin)
        input_layout.addLayout(row2)

        # Row 3: Optimize, Indicator
        row3 = QHBoxLayout()
        self.optimize_combo = QComboBox()
        self.optimize_combo.addItems(["buy_wr", "sell_wr"])
        self.optimize_combo.setCurrentText("buy_wr")
        row3.addWidget(QLabel("Optimize:"))
        row3.addWidget(self.optimize_combo)

        self.indicator_edit = QLineEdit()
        self.indicator_edit.setText("Prime RSI")
        row3.addWidget(QLabel("Indicator:"))
        row3.addWidget(self.indicator_edit)
        row3.addStretch()  # Pushes items to the left
        input_layout.addLayout(row3)

        # Row 4: Generate Button
        row4 = QHBoxLayout()
        row4.addStretch()
        self.generate_btn = QPushButton("Generate Progression Plot")
        self.generate_btn.setObjectName("generateBtn")
        self.generate_btn.clicked.connect(self.generate_plot)
        row4.addWidget(self.generate_btn)
        row4.addStretch()
        input_layout.addLayout(row4)

        main_layout.addWidget(input_frame)

        # Plot Area
        self.canvas = MplCanvas(self, width=12, height=6, dpi=100)
        main_layout.addWidget(self.canvas, 1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Please select a directory and enter parameters.")

        # Apply CSS Stylesheet (Catppuccin Dark Theme)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: "Segoe UI", Arial, sans-serif; font-size: 13px; }
            QFrame#inputFrame { background-color: #313244; border-radius: 10px; padding: 15px; border: 1px solid #45475a; }

            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { 
                background-color: #1e1e2e; border: 1px solid #45475a; border-radius: 6px; padding: 8px; color: #cdd6f4; 
            }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { 
                background-color: #1e1e2e; color: #cdd6f4; selection-background-color: #45475a; 
            }

            QPushButton { background-color: #89b4fa; color: #1e1e2e; border-radius: 6px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #b4befe; }
            QPushButton#generateBtn { background-color: #a6e3a1; padding: 12px 24px; font-size: 15px; }
            QPushButton#generateBtn:hover { background-color: #94e2d5; }
            QStatusBar { background-color: #181825; color: #a6adc8; }
            QLabel { background-color: transparent; }
        """)

        self.analyzer = None
        self.clear_plot()

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.dir_edit.setText(dir_path)
            self.status_bar.showMessage(f"Selected directory: {dir_path}")

    def clear_plot(self):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Awaiting Data...', horizontalalignment='center',
                verticalalignment='center', fontsize=20, color='#6c7086',
                transform=ax.transAxes)
        ax.set_facecolor('#1e1e2e')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.figure.patch.set_facecolor('#1e1e2e')
        self.canvas.draw()

    def generate_plot(self):
        dir_path = self.dir_edit.text()
        if not os.path.isdir(dir_path):
            QMessageBox.warning(self, "Error", "Please select a valid directory.")
            return

        t_str = self.date_edit.text().strip()
        if not t_str:
            QMessageBox.warning(self, "Error", "Please enter a start date.")
            return

        target_price = self.price_spin.value()
        future_bar = self.bar_spin.value()
        optimize_type = self.optimize_combo.currentText()
        indicator_name = self.indicator_edit.text().strip()

        if not indicator_name:
            QMessageBox.warning(self, "Error", "Please enter an indicator name.")
            return

        self.status_bar.showMessage("Processing files... please wait.")
        QApplication.processEvents()

        try:
            # Reuse analyzer if directory hasn't changed (Caching)
            if not self.analyzer or self.analyzer.dir_path != dir_path:
                self.analyzer = TaurusAnalyzer(dir_path)
                self.status_bar.showMessage("Files loaded. Analyzing progression...")
                QApplication.processEvents()

            dates, rates, prices, target_date_str = self.analyzer.analyze(t_str, target_price, future_bar, optimize_type, indicator_name)
            self.plot_data(dates, rates, prices, target_date_str, optimize_type, indicator_name)
            self.status_bar.showMessage(f"Success! Plot generated for '{optimize_type}' / '{indicator_name}' on target date {target_date_str}.")
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", str(e))
            self.status_bar.showMessage("Error during analysis.")
            self.clear_plot()

    def plot_data(self, dates, rates, prices, target_date_str, optimize_type, indicator_name):
        self.canvas.figure.clear()

        # Color Palette matching the UI
        bg_color = '#1e1e2e'
        text_color = '#cdd6f4'
        grid_color = '#45475a'
        line1_color = '#89b4fa'
        line2_color = '#f38ba8'

        fig = self.canvas.figure
        fig.patch.set_facecolor(bg_color)

        ax1 = fig.add_subplot(111)
        ax1.set_facecolor(bg_color)

        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
        x = np.arange(len(date_strs))

        # Plot Win Rate
        line1 = ax1.plot(x, rates, marker='o', linestyle='-', color=line1_color,
                         linewidth=2, markersize=8, label='Val Win Rate (%)')[0]
        ax1.set_ylabel('Val Win Rate (%)', color=line1_color, fontweight='bold', fontsize=12)
        ax1.tick_params(axis='y', labelcolor=line1_color, colors=text_color)
        ax1.tick_params(axis='x', colors=text_color)
        ax1.grid(True, linestyle='--', alpha=0.5, color=grid_color)

        # Annotations
        for i, txt in enumerate(rates):
            ax1.annotate(f'{txt:.1f}%', (x[i], rates[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10,
                         color=text_color, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc=bg_color, ec=line1_color, lw=1, alpha=0.9))

        ax1.set_xticks(x)
        ax1.set_xticklabels(date_strs, rotation=45, ha='right')

        # Twin axis for price
        ax2 = ax1.twinx()
        line2 = ax2.plot(x, prices, marker='x', linestyle='--', color=line2_color,
                         linewidth=2, markersize=8, label='Target Price')[0]
        ax2.set_ylabel('Target Price', color=line2_color, fontweight='bold', fontsize=12)
        ax2.tick_params(axis='y', labelcolor=line2_color, colors=text_color)

        fig.suptitle(f'Val Win Rate Progression ({optimize_type} | {indicator_name})\nTarget Date: {target_date_str}',
                     fontsize=16, fontweight='bold', color=text_color, y=0.98)

        # Legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.90),
                   facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)

        # Clean up spines
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_color(grid_color)
            ax.spines['left'].set_color(grid_color)
            ax.spines['right'].set_color(grid_color)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.canvas.draw()


# ==========================================
# App Entry Point
# ==========================================
def main():
    app = QApplication(sys.argv)

    # High DPI scaling
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()