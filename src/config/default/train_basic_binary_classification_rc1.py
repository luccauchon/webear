toda__seed_offset          = 123
toda__debug_level          = "DEBUG"
toda__device               = 'cuda'

toda__data_augmentation    = False
toda__force_download_data  = False

toda__tav_dates            = ["2019-01-01", "2025-01-31"]
toda__mes_dates            = ["2025-02-01", "2099"]

toda__run_id               = 123
toda__version              = "rc1"

toda__x_cols               = ['Close', 'High', 'Low', 'Open'] + ['Volume'] + ['day_of_week']  # For SPY and VIX
toda__x_cols_to_norm       = ['Close', 'High', 'Low', 'Open', 'Volume']
toda__x_seq_length         = 30
toda__y_cols               = [('Close', 'SPY')]
toda__y_seq_length         = 1
toda__cutoff_days          = [1]  # Monday

