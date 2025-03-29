debug_level = "DEBUG"
direction='up'

type_margin = 'relative'
margin      = 0.

wanted_pos_weight = 1.

lr_decay_iters= 4001
max_iters     = 4001
learning_rate = 1e-3
min_lr        = 1e-4
weight_decay  = 0.1

tav_dates      = ["2024-01-01", "2025-03-22"]
mes_dates      = ["2025-03-23", "2025-03-29"]

log_interval  = 1000
eval_interval = 1

force_download_data = False

x_seq_length = 4
y_seq_length = 1

#x_cols = ['Close', 'High', 'Low', 'Open'] + ['day_of_week']  # For SPY and VIX
x_cols = ['Close'] + ['day_of_week']  # For SPY and VIX
#x_cols = ['Close_MA2','High_MA2','Low_MA2','Open_MA2']
#x_cols += ['Close_MA3','High_MA3','Low_MA3','Open_MA3']
#x_cols += ['Close_MA5','High_MA5','Low_MA5','Open_MA5']
x_cols += ['Close_MA3','Close_MA5']
x_cols_to_norm = []
y_cols = [('Close_MA3', 'SPY')]

batch_size=1024

jump_ahead=0

data_augmentation=False
frequency_of_noise=0.5
power_of_noise=2.5