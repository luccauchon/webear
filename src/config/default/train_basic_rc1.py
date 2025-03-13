toda__data_augmentation    = False
toda__debug_level          = "DEBUG"
toda__device               = 'cuda'
toda__feature_cols         = ['Close', 'High', 'Low', 'Open', 'Volume', 'Close_direction'] + ['day_of_week']  # For SPY and VIX
toda__mes_dates            = ["2025-01-01", "2099-12-31"]
toda__run_id               = 123
toda__seed_offset          = 123
toda__precision_spread     = 1.
toda__target_col           = [('Close', 'SPY')]
toda__tav_dates            = ["2019-12-01", "2024-12-31"]
toda__version              = "rc1"
toda__x_cols_to_norm       = ['Close', 'High', 'Low', 'Open', 'Volume']
toda__x_seq_length         = 10
toda__y_cols_to_norm       = [('Close', 'SPY')]
toda__y_seq_length         = 1

toda__num_input_features   = 2 * len(toda__feature_cols) + -1 + -1  # For SPY and VIX + -1 for day_of_week and -1 No volume for VIX