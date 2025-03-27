debug_level = "DEBUG"
direction='down'

type_margin = 'fixed'
margin      = 1.

wanted_pos_weight = 1.5

lr_decay_iters= 3001
max_iters     = 300199
learning_rate = 1e-4
min_lr        = 1e-5
weight_decay  = 0.1

tav_dates      = ["2019-01-01", "2025-03-01"]
mes_dates      = ["2025-03-02", "2025-03-08"]

log_interval  = 500
eval_interval = 10

force_download_data = False

x_seq_length = 10
y_seq_length = 1

x_cols = ['Close_MA2','Close_MA5','Close', 'High', 'Low', 'Open'] + ['day_of_week']  # For SPY and VIX
x_cols_to_norm = []
y_cols = [('Close_MA2', 'SPY')]

batch_size=4096