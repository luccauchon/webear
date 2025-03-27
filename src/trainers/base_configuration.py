debug_level = "INFO"

tav_dates      = ["2019-01-01", "2025-03-15"]
mes_dates      = ["2025-03-16", "2025-03-22"]

fast_execution_for_debugging = False

class_id__2__sideways = {'0':(-99,-5), '1': (-5, -0.5), '2': (-0.5, 0.5), '3': (0.5, 5), '4': (5, 99)}

eval_interval = 10
log_interval  = 5000
lr_decay_iters= 15000
max_iters     = 15000
learning_rate = 1e-3
min_lr        = 1e-6
weight_decay  = 0.1

direction = 'down'

type_margin = 'fixed'
margin = 1.5
jump_ahead = 0 # Number of days ahead we want to predict. 0 to predict for tomorrow.

frequency_of_noise  = 0.25
power_of_noise      = 0.001

wanted_pos_weight=1


force_download_data = False
data_augmentation   = False

x_cols = ['Close', 'High', 'Low', 'Open'] + ['Volume'] + ['day_of_week']  # For SPY and VIX
x_cols_to_norm = ['Close', 'High', 'Low', 'Open', 'Volume']
y_cols = [('Close_MA5', 'SPY')]
x_seq_length=10
y_seq_length = 1

batch_size = 1024