import os
import re
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from utils import get_and_clean_stub_dir
from fetchers.data_factory import factory_load_data
from datetime import timedelta, datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ==========================================
# 1. Configuration
# ==========================================
TICKER = "^GSPC"

# Sequence configuration
N_BARS = 20  # Input sequence length
M_BARS = 15  # Prediction horizon length

# Model configuration
HIDDEN_SIZE = 64
NUM_LAYERS = 8
DROPOUT = 0.2
BATCH_SIZE = 64 // 4
EPOCHS = 250
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-3

# Train/test split by trading day
NUMBER_OF_DAYS_FOR_VALIDATION = 4
BETWEEN_TIME = ("14:30", "16:00")
NUMBER_DAYS_FOR_DATASET = 25
NUMBER_OF_DAYS_FOR_TEST = 2

# Multitask loss weights
REG_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 1.0
SLOPE_LOSS_WEIGHT = 0.1

# Indicator settings
RSI_PERIOD = 14
VOL_PERIOD = 14

# Input features.
FEATURE_COLS = [
    "log_ret",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ret_vol_14",
    "parkinson_vol_14",
    "atr_pct_14",
    "prev_day_close",
    "vwap",
    "vix",
]

# Warmup bars to avoid using early unstable indicator values.
WARMUP_BARS = max(26, RSI_PERIOD, VOL_PERIOD) + 1

# Dynamic input size
INPUT_SIZE = len(FEATURE_COLS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 50  # Stop if val loss doesn't improve

GENERATE_PLOT = True


# ==========================================
# 2. Data Fetching & Preprocessing
# ==========================================
def fetch_and_clean_data():
    df_ticker, df_vix = factory_load_data(_dataset_id="intraday_1min", _ticker=TICKER, _args={"get_vix": True})
    # df = yf.download(TICKER,period="5d",interval="1m",progress=False,prepost=False,auto_adjust=True, ignore_tz=True,)

    if df_ticker is None or len(df_ticker) == 0:
        raise RuntimeError("No data returned from data factory.")

    if isinstance(df_ticker.columns, pd.MultiIndex):
        df_ticker.columns = df_ticker.columns.get_level_values(0)
        if df_vix is not None and isinstance(df_vix.columns, pd.MultiIndex):
            df_vix.columns = df_vix.columns.get_level_values(0)

    df_ticker = df_ticker.between_time(BETWEEN_TIME[0], BETWEEN_TIME[1])

    # --- VIX Integration ---
    if df_vix is not None and len(df_vix) > 0:
        df_vix = df_vix.between_time(BETWEEN_TIME[0], BETWEEN_TIME[1])
        # Extract VIX close price and rename to 'vix'
        vix_series = df_vix["Close"].rename("vix")
        # Left join to keep all df_ticker rows, forward fill missing VIX values
        df_ticker = df_ticker.join(vix_series, how="left")
        # Forward fill, then backward fill for any leading NaNs, then 0.0 as last resort
        df_ticker["vix"] = df_ticker["vix"].ffill().bfill().fillna(0.0)
    else:
        df_ticker["vix"] = 0.0
    # -----------------------

    # Added "vix" to required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume", "vix"]
    df_ticker = df_ticker[required_cols].dropna()

    df_ticker["Date"] = df_ticker.index.date

    num_days = df_ticker["Date"].nunique()
    print(f"Loaded {len(df_ticker)} minute bars across {num_days} trading days.")

    return df_ticker


# ==========================================
# 3. Feature Engineering
# ==========================================
def _add_day_features(day: pd.DataFrame) -> pd.DataFrame:
    g = day.copy()

    close = g["Close"].astype(float)
    high = g["High"].astype(float)
    low = g["Low"].astype(float)
    volume = g["Volume"].astype(float)

    # Log return
    prev_close = close.shift(1)
    log_ret = np.log(close / prev_close)
    log_ret = log_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta.clip(upper=0.0)).fillna(0.0)

    avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()

    eps = 1e-12
    rs = avg_gain / (avg_loss + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.replace([np.inf, -np.inf], 50.0)
    rsi[(avg_loss.abs() < eps) & (avg_gain.abs() < eps)] = 50.0
    rsi = rsi.fillna(50.0).clip(0.0, 100.0)

    # MACD
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    macd = macd.fillna(0.0)
    macd_signal = macd_signal.fillna(0.0)
    macd_hist = macd_hist.fillna(0.0)

    # Volatility metrics
    ret_vol = log_ret.rolling(window=VOL_PERIOD, min_periods=2).std().fillna(0.0)

    low_safe = low.replace(0.0, np.nan)
    hl = np.log(high / low_safe)
    hl = hl.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    parkinson_var = (hl ** 2) / (4.0 * np.log(2.0))
    parkinson_vol = parkinson_var.rolling(window=VOL_PERIOD, min_periods=1).mean().pow(0.5).fillna(0.0)

    prev_close_tr = close.shift(1).fillna(close)
    tr = pd.concat([high - low, (high - prev_close_tr).abs(), (low - prev_close_tr).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=VOL_PERIOD, min_periods=1).mean()
    atr_pct = (atr / close).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # VWAP (Volume Weighted Average Price)
    typical_price = (high + low + close) / 3.0
    # Avoid division by zero if volume is 0
    safe_volume = volume.replace(0.0, np.nan)
    cum_tp_vol = (typical_price * safe_volume).cumsum()
    cum_vol = safe_volume.cumsum()
    vwap = cum_tp_vol / cum_vol
    vwap = vwap.fillna(close)  # Fallback to close price if volume is missing/zero

    g["log_ret"] = log_ret
    g["rsi_14"] = rsi
    g["macd"] = macd
    g["macd_signal"] = macd_signal
    g["macd_hist"] = macd_hist
    g["ret_vol_14"] = ret_vol
    g["parkinson_vol_14"] = parkinson_vol
    g["atr_pct_14"] = atr_pct
    g["vwap"] = vwap

    return g


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    daily_close = df.groupby('Date')['Close'].last()
    prev_day_close_map = daily_close.shift(1)
    df['prev_day_close'] = df['Date'].map(prev_day_close_map)

    first_day_close = df.groupby('Date')['Close'].first()
    df['prev_day_close'] = df['prev_day_close'].fillna(df['Date'].map(first_day_close))

    df = df.groupby("Date", group_keys=False).apply(_add_day_features)
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], 0.0)

    fill_values = {col: 0.0 for col in FEATURE_COLS}
    fill_values["rsi_14"] = 50.0
    fill_values["prev_day_close"] = df["Close"].mean()
    fill_values["vwap"] = df["Close"].mean()  # Fallback for VWAP
    fill_values["vix"] = df["vix"].mean() if "vix" in df.columns else 15.0  # Fallback for VIX

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(fill_values)

    return df


# ==========================================
# 4. Sequence Generation
# ==========================================
def create_intraday_sequences(df: pd.DataFrame, n: int, m: int):
    X = []
    y_logret = []
    y_dir = []
    y_slope_dir = []  # NEW: Slope direction of the target
    y_future_close = []
    y_last_close = []
    y_future_times = []

    for date, group in df.groupby("Date"):
        group = group.sort_index()

        if len(group) < WARMUP_BARS + n + m:
            continue

        features = group[FEATURE_COLS].values.astype(np.float32)
        close = group["Close"].values.astype(np.float64)
        times = group.index

        max_start = len(features) - n - m

        for i in range(WARMUP_BARS, max_start + 1):
            x_seq = features[i: i + n]

            last_close = close[i + n - 1]
            future_close = close[i + n: i + n + m]
            future_times = times[i + n: i + n + m]

            if last_close > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    future_logret = np.log(future_close / last_close)
            else:
                future_logret = np.zeros(m, dtype=np.float64)

            future_logret = np.where(np.isfinite(future_logret), future_logret, 0.0).astype(np.float32)
            direction = (future_logret > 0.0).astype(np.float32)

            # NEW: Compute slope direction (1 if last point > first point, else 0)
            slope_dir = 1.0 if future_close[-1] > future_close[0] else 0.0

            X.append(x_seq)
            y_logret.append(future_logret)
            y_dir.append(direction)
            y_slope_dir.append(slope_dir)
            y_future_close.append(future_close)
            y_last_close.append(last_close)
            y_future_times.append(future_times)

    if len(X) == 0:
        return (
            np.empty((0, n, INPUT_SIZE), dtype=np.float32),
            np.empty((0, m), dtype=np.float32),
            np.empty((0, m), dtype=np.float32),
            np.empty((0,), dtype=np.float32),  # y_slope_dir
            np.empty((0, m), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=object),
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y_logret, dtype=np.float32),
        np.array(y_dir, dtype=np.float32),
        np.array(y_slope_dir, dtype=np.float32),
        np.array(y_future_close, dtype=np.float64),
        np.array(y_last_close, dtype=np.float64),
        np.array(y_future_times, dtype=object),
    )


# ==========================================
# 5. PyTorch Dataset
# ==========================================
class MultiTaskSPXDataset(Dataset):
    def __init__(self, X, y_reg, y_dir, y_slope):
        self.X = torch.from_numpy(X).float()
        self.y_reg = torch.from_numpy(y_reg).float()
        self.y_dir = torch.from_numpy(y_dir).float()
        self.y_slope = torch.from_numpy(y_slope).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_dir[idx], self.y_slope[idx]


# ==========================================
# 6. Multitask LSTM Model
# ==========================================
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, m_steps: int, bidirectional: bool = True):
        super(MultiTaskLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.m_steps = m_steps
        self.bidirectional = bidirectional

        # When bidirectional is True, the LSTM output hidden size is doubled
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.shared_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, m_steps),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, m_steps),
        )

        self.slope_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        shared = self.shared_head(last_hidden)

        reg_out = self.regression_head(shared)
        cls_logits = self.classification_head(shared)
        slope_logits = self.slope_head(shared)  # Shape: (batch_size, 1)

        return reg_out, cls_logits, slope_logits


# ==========================================
# 7. Training Pipeline
# ==========================================
def train_model():
    df = fetch_and_clean_data()
    df = add_features(df)
    print(f"Dataframe dates: {df.index[0].strftime('%Y-%m-%d_%H%M')} :: {df.index[-1].strftime('%Y-%m-%d_%H%M')}")
    unique_dates = sorted(df["Date"].unique())
    if len(unique_dates) < 2:
        raise RuntimeError("Need at least two trading days for train/test splitting.")

    assert NUMBER_DAYS_FOR_DATASET > 0 and NUMBER_DAYS_FOR_DATASET < len(unique_dates)
    unique_dates = unique_dates[-NUMBER_DAYS_FOR_DATASET:]
    assert len(unique_dates) == NUMBER_DAYS_FOR_DATASET

    assert NUMBER_OF_DAYS_FOR_VALIDATION > 0
    split_idx = len(unique_dates) - (NUMBER_OF_DAYS_FOR_VALIDATION + NUMBER_OF_DAYS_FOR_TEST)
    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:split_idx+NUMBER_OF_DAYS_FOR_VALIDATION]
    assert len(val_dates) == NUMBER_OF_DAYS_FOR_VALIDATION
    test_dates = unique_dates[split_idx+NUMBER_OF_DAYS_FOR_VALIDATION:]
    assert len(test_dates) == NUMBER_OF_DAYS_FOR_TEST

    df_train = df[df["Date"].isin(train_dates)]
    df_val   = df[df["Date"].isin(val_dates)]
    df_test  = df[df["Date"].isin(test_dates)]
    print(f"Train days:{train_dates}\nVal days:{val_dates}\nTest days:{test_dates}")
    print(f"Train dates: {len(train_dates)} ({df_train.index[0].strftime('%Y-%m-%d_%H%M')} :: {df_train.index[-1].strftime('%Y-%m-%d_%H%M')}) | "
          f"Val dates: {len(val_dates)} ({df_val.index[0].strftime('%Y-%m-%d_%H%M')} :: {df_val.index[-1].strftime('%Y-%m-%d_%H%M')}) | "
          f"Test dates: {len(test_dates)} ({df_test.index[0].strftime('%Y-%m-%d_%H%M')} :: {df_test.index[-1].strftime('%Y-%m-%d_%H%M')})")

    X_train, y_train_logret, y_train_dir, y_train_slope, _, _, _ = create_intraday_sequences(df_train, N_BARS, M_BARS)

    (
        X_val, y_val_logret, y_val_dir, y_val_slope,
        y_val_future_close, y_val_last_close, y_val_future_times
    ) = create_intraday_sequences(df_val, N_BARS, M_BARS)

    (
        X_test, y_test_logret, y_test_dir, y_test_slope,
        y_test_future_close, y_test_last_close, y_test_future_times
    ) = create_intraday_sequences(df_test, N_BARS, M_BARS)

    print(f"Created {len(X_train)} train sequences.")
    print(f"Created {len(X_val)} test sequences.")
    print(f"Created {len(X_test)} test sequences.")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError("No sequences created. Reduce N_BARS/M_BARS/WARMUP_BARS or fetch more data.")

    input_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train.reshape(-1, INPUT_SIZE)).reshape(X_train.shape)
    X_val_scaled = input_scaler.transform(X_val.reshape(-1, INPUT_SIZE)).reshape(X_val.shape)
    X_test_scaled = input_scaler.transform(X_test.reshape(-1, INPUT_SIZE)).reshape(X_test.shape)

    target_scaler = StandardScaler()
    y_train_reg_scaled = target_scaler.fit_transform(y_train_logret.reshape(-1, 1)).reshape(y_train_logret.shape)
    y_val_reg_scaled = target_scaler.transform(y_val_logret.reshape(-1, 1)).reshape(y_val_logret.shape)
    y_test_reg_scaled = target_scaler.transform(y_test_logret.reshape(-1, 1)).reshape(y_test_logret.shape)

    train_dataset = MultiTaskSPXDataset(X_train_scaled, y_train_reg_scaled, y_train_dir, y_train_slope)
    val_dataset = MultiTaskSPXDataset(X_val_scaled, y_val_reg_scaled, y_val_dir, y_val_slope)
    test_dataset = MultiTaskSPXDataset(X_test_scaled, y_test_reg_scaled, y_test_dir, y_test_slope)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        m_steps=M_BARS,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Input size: {INPUT_SIZE} | Model parameters: {param_count:,}")

    reg_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Training on {DEVICE}...")
    epochs_no_improve = 0
    # Track the best validation loss
    best_val_loss = float('inf')
    best_model = None
    for epoch in range(EPOCHS):
        model.train()
        train_total_loss, train_reg_loss, train_cls_loss, train_slope_loss = 0.0, 0.0, 0.0, 0.0
        train_dir_acc, train_slope_acc = 0.0, 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for X_batch, y_reg_batch, y_dir_batch, y_slope_batch in train_bar:
            X_batch = X_batch.to(DEVICE)
            y_reg_batch = y_reg_batch.to(DEVICE)
            y_dir_batch = y_dir_batch.to(DEVICE)
            y_slope_batch = y_slope_batch.to(DEVICE)

            optimizer.zero_grad()
            pred_reg, pred_cls, pred_slope = model(X_batch)

            loss_reg = reg_criterion(pred_reg, y_reg_batch)
            loss_cls = cls_criterion(pred_cls, y_dir_batch)
            loss_slope = cls_criterion(pred_slope, y_slope_batch.unsqueeze(1))

            loss = REG_LOSS_WEIGHT * loss_reg + CLS_LOSS_WEIGHT * loss_cls + SLOPE_LOSS_WEIGHT * loss_slope
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total_loss += loss.item()
            train_reg_loss += loss_reg.item()
            train_cls_loss += loss_cls.item()
            train_slope_loss += loss_slope.item()

            pred_dir = (torch.sigmoid(pred_cls) > 0.5).float()
            train_dir_acc += (pred_dir == y_dir_batch).float().mean().item()

            pred_slope_dir = (torch.sigmoid(pred_slope) > 0.5).float().squeeze()
            train_slope_acc += (pred_slope_dir == y_slope_batch).float().mean().item()

        scheduler.step()
        n_train_batches = max(1, len(train_loader))

        model.eval()
        val_total_loss, val_reg_loss, val_cls_loss, val_slope_loss = 0.0, 0.0, 0.0, 0.0
        val_dir_acc, val_slope_acc = 0.0, 0.0

        with torch.no_grad():
            for X_batch, y_reg_batch, y_dir_batch, y_slope_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_reg_batch = y_reg_batch.to(DEVICE)
                y_dir_batch = y_dir_batch.to(DEVICE)
                y_slope_batch = y_slope_batch.to(DEVICE)

                pred_reg, pred_cls, pred_slope = model(X_batch)

                loss_reg = reg_criterion(pred_reg, y_reg_batch)
                loss_cls = cls_criterion(pred_cls, y_dir_batch)
                loss_slope = cls_criterion(pred_slope, y_slope_batch.unsqueeze(1))

                loss = REG_LOSS_WEIGHT * loss_reg + CLS_LOSS_WEIGHT * loss_cls + SLOPE_LOSS_WEIGHT * loss_slope

                val_total_loss += loss.item()
                val_reg_loss += loss_reg.item()
                val_cls_loss += loss_cls.item()
                val_slope_loss += loss_slope.item()

                pred_dir = (torch.sigmoid(pred_cls) > 0.5).float()
                val_dir_acc += (pred_dir == y_dir_batch).float().mean().item()

                pred_slope_dir = (torch.sigmoid(pred_slope) > 0.5).float().squeeze()
                val_slope_acc += (pred_slope_dir == y_slope_batch).float().mean().item()

        n_val_batches = max(1, len(val_loader))
        current_val_loss = val_total_loss / n_val_batches
        print_info = False
        # Save best model if validation loss improves
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model, "best_model.pth")
            print(f"  [Epoch {epoch + 1:03d} | Saved Best Model] Validation loss improved to {best_val_loss:.5f}")
            print_info = True
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  [Early Stopping] No improvement for {PATIENCE} epochs. Stopping training.")
                break
        if epoch % 50 == 0 or epoch == EPOCHS - 1 or print_info:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train Loss: {train_total_loss / n_train_batches:.5f} "
                f"(reg: {train_reg_loss / n_train_batches:.4f}, cls: {train_cls_loss / n_train_batches:.4f}, slope: {train_slope_loss / n_train_batches:.4f}) | "
                f"Train Acc: Dir {train_dir_acc / n_train_batches:.4f} | Train Slope Acc {train_slope_acc / n_train_batches:.4f} | "
                f"Val Loss: {current_val_loss:.5f} | "
                f"Val Dir Acc: {val_dir_acc / n_val_batches:.4f} | Val Slope Acc: {val_slope_acc / n_val_batches:.4f}"
            )
    # Run best model on test data
    print(f"Loading best model...")
    best_model = torch.load("best_model.pth", map_location=DEVICE, weights_only=False)
    best_model.eval()
    test_total_loss, test_reg_loss, test_cls_loss, test_slope_loss = 0.0, 0.0, 0.0, 0.0
    test_dir_acc, test_slope_acc = 0.0, 0.0
    with torch.no_grad():
        for X_batch, y_reg_batch, y_dir_batch, y_slope_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_reg_batch = y_reg_batch.to(DEVICE)
            y_dir_batch = y_dir_batch.to(DEVICE)
            y_slope_batch = y_slope_batch.to(DEVICE)

            pred_reg, pred_cls, pred_slope = best_model(X_batch)

            loss_reg = reg_criterion(pred_reg, y_reg_batch)
            loss_cls = cls_criterion(pred_cls, y_dir_batch)
            loss_slope = cls_criterion(pred_slope, y_slope_batch.unsqueeze(1))

            loss = REG_LOSS_WEIGHT * loss_reg + CLS_LOSS_WEIGHT * loss_cls + SLOPE_LOSS_WEIGHT * loss_slope

            test_total_loss += loss.item()
            test_reg_loss += loss_reg.item()
            test_cls_loss += loss_cls.item()
            test_slope_loss += loss_slope.item()

            pred_dir = (torch.sigmoid(pred_cls) > 0.5).float()
            test_dir_acc += (pred_dir == y_dir_batch).float().mean().item()

            pred_slope_dir = (torch.sigmoid(pred_slope) > 0.5).float().squeeze()
            test_slope_acc += (pred_slope_dir == y_slope_batch).float().mean().item()
    n_test_batches = max(1, len(test_loader))
    print(
        f"Test Loss: {test_total_loss / n_test_batches:.5f} "
        f"(reg: {test_reg_loss / n_test_batches:.4f}, cls: {test_cls_loss / n_test_batches:.4f}, slope: {test_slope_loss / n_test_batches:.4f}) | "
        f"Test Dir Acc: {test_dir_acc / n_test_batches:.4f} | Test Slope Acc: {test_slope_acc / n_test_batches:.4f} | "
    )
    return (
        model, input_scaler, target_scaler, X_test_scaled,
        y_test_logret, y_test_dir, y_test_slope,
        y_test_future_close, y_test_last_close, y_test_future_times,
    )


# ==========================================
# 8. Inference & Visualization
# ==========================================
def plot_predictions(
        model, target_scaler, X_abc,
        y_abc_logret_raw, y_abc_dir_raw, y_abc_slope_raw,
        y_abc_future_close, y_abc_last_close, y_abc_future_times,
):
    if len(X_abc) == 0:
        print("No test sequences available for plotting.")
        return

    output_dir = get_and_clean_stub_dir("lstm_output__all")
    output_dir_match = get_and_clean_stub_dir("lstm_output__slope_match")
    output_dir_no_match = get_and_clean_stub_dir("lstm_output__no_slope_match")
    model.eval()

    n_samples = len(X_abc)
    batch_size = BATCH_SIZE

    all_pred_logret = []
    all_pred_up_prob = []
    all_pred_slope_prob = []  # NEW

    for i in range(0, n_samples, batch_size):
        X_batch = X_abc[i:i + batch_size]
        X_tensor = torch.from_numpy(X_batch).float().to(DEVICE)

        with torch.no_grad():
            pred_reg_scaled, pred_cls_logits, pred_slope_logits = model(X_tensor)

        pred_reg_scaled_np = pred_reg_scaled.cpu().numpy()
        pred_logret_batch = target_scaler.inverse_transform(
            pred_reg_scaled_np.reshape(-1, 1)
        ).reshape(pred_reg_scaled_np.shape)

        pred_up_prob_batch = torch.sigmoid(pred_cls_logits).cpu().numpy()
        pred_slope_prob_batch = torch.sigmoid(pred_slope_logits).cpu().numpy()

        all_pred_logret.append(pred_logret_batch)
        all_pred_up_prob.append(pred_up_prob_batch)
        all_pred_slope_prob.append(pred_slope_prob_batch)

    all_pred_logret = np.concatenate(all_pred_logret, axis=0)
    all_pred_up_prob = np.concatenate(all_pred_up_prob, axis=0)
    all_pred_slope_prob = np.concatenate(all_pred_slope_prob, axis=0)

    print(f"Generating plots for all {n_samples} test sequences in {output_dir}...")

    for idx in tqdm(range(n_samples)):
        actual_close = y_abc_future_close[idx]
        last_close = y_abc_last_close[idx]
        future_times = y_abc_future_times[idx]

        pred_close = last_close * np.exp(all_pred_logret[idx])

        actual_dir = y_abc_dir_raw[idx]
        pred_prob = all_pred_up_prob[idx]

        # NEW: Extract slope targets and predictions
        actual_slope = y_abc_slope_raw[idx]
        pred_slope_prob = all_pred_slope_prob[idx][0]
        pred_slope_dir = 1 if pred_slope_prob >= 0.5 else 0

        start_time = future_times[0].strftime('%Hh%Mm')
        end_time = future_times[-1].strftime('%Hh%Mm')
        date_str = future_times[0].strftime('%Y-%m-%d')
        time_str = f"{date_str} | {start_time} to {end_time}"

        plot_title = f"SPX {M_BARS}-Minute Multitask Prediction {time_str}"
        safe_filename = re.sub(r'[^A-Za-z0-9_.-]', '_', plot_title) + ".png"
        safe_filename = re.sub(r'_+', '_', safe_filename)

        minutes = np.arange(1, len(actual_close) + 1)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=True)

        # Determine colors for slope visualization
        actual_slope_text = "UP" if actual_slope == 1 else "DOWN"
        pred_slope_text = "UP" if pred_slope_dir == 1 else "DOWN"
        actual_color = "green" if actual_slope == 1 else "red"
        pred_color = "green" if pred_slope_dir == 1 else "red"

        # Price plot
        axes[0].plot(minutes, actual_close, label="Actual Close", marker="o", color="blue", markersize=4)
        axes[0].plot(minutes, pred_close, label="Predicted Close", marker="x", linestyle="--", color="orange", markersize=4)

        # NEW: Visual integration of slope direction
        # Draw actual slope line
        axes[0].plot([1, len(actual_close)], [actual_close[0], actual_close[-1]],
                     color=actual_color, linestyle="-", linewidth=2.5, alpha=0.6,
                     label=f"Actual Slope ({actual_slope_text})")

        # Draw predicted slope line
        axes[0].plot([1, len(pred_close)], [pred_close[0], pred_close[-1]],
                     color=pred_color, linestyle="--", linewidth=2.5, alpha=0.8,
                     label=f"Pred Slope ({pred_slope_text}, {pred_slope_prob:.1%})")

        # Add a clear text annotation box for the slope
        slope_match = (actual_slope == pred_slope_dir)
        box_color = "green" if slope_match else "orange"
        slope_info = f"Slope → Actual: {actual_slope_text} | Pred: {pred_slope_text} ({pred_slope_prob:.1%})"

        axes[0].text(0.02, 0.02, slope_info, transform=axes[0].transAxes,
                     fontsize=10, fontweight='bold', verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=box_color, linewidth=1.5))

        axes[0].set_title(f"SPX {M_BARS}-Minute Multitask Prediction\n{time_str}")
        axes[0].set_ylabel("Price")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # Direction / probability plot
        axes[1].plot(minutes, actual_dir, drawstyle="steps-mid", label="Actual Direction (1=Up)", color="blue", alpha=0.7)
        axes[1].plot(minutes, pred_prob, marker="x", linestyle="--", label="Predicted Up Probability", color="red")
        axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.6)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_yticks([0.0, 0.5, 1.0])
        axes[1].set_xlabel("Minutes Ahead")
        axes[1].set_ylabel("Direction / Probability")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(output_dir, safe_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        if slope_match:
            output_path = os.path.join(output_dir_match, safe_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            output_path = os.path.join(output_dir_no_match, safe_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ==========================================
# 9. Run
# ==========================================
if __name__ == "__main__":
    (
        model, input_scaler, target_scaler, X_test_scaled,
        y_test_logret, y_test_dir, y_test_slope,
        y_test_future_close, y_test_last_close, y_test_future_times,
    ) = train_model()
    if GENERATE_PLOT:
        plot_predictions(
            model=model,
            target_scaler=target_scaler,
            X_abc=X_test_scaled,
            y_abc_logret_raw=y_test_logret,
            y_abc_dir_raw=y_test_dir,
            y_abc_slope_raw=y_test_slope,
            y_abc_future_close=y_test_future_close,
            y_abc_last_close=y_test_last_close,
            y_abc_future_times=y_test_future_times,
        )