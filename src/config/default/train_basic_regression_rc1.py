toda__seed_offset          = 123
toda__debug_level          = "DEBUG"
toda__device               = 'cuda'

toda__data_augmentation    = False
toda__force_download_data  = False

toda__tav_dates            = ["2019-12-01", "2024-12-31"]
toda__mes_dates            = ["2025-01-01", "2099-12-31"]

toda__run_id               = 123
toda__version              = "rc1"

toda__precision_spread     = 1.

toda__x_cols               = ['Close', 'High', 'Low', 'Open'] + ['Volume'] + ['day_of_week']  # For SPY and VIX
toda__x_cols_to_norm       = ['Close', 'High', 'Low', 'Open', 'Volume']
toda__x_seq_length         = 15
toda__y_cols               = [('Close', 'SPY'), ('Open', 'SPY'), ('High', 'SPY'), ('Low', 'SPY')]
toda__y_cols_to_norm       = [('Close', 'SPY'), ('Open', 'SPY'), ('High', 'SPY'), ('Low', 'SPY')]
toda__y_seq_length         = 5

# toda__target_col           = [('Close_MA16', 'SPY')]
# toda__x_cols_to_norm       = ['Close_MA16', 'High_MA16', 'Low_MA16', 'Open_MA16', 'Volume_MA16']
# toda__feature_cols         = ['Close_MA16', 'High_MA16', 'Low_MA16', 'Open_MA16'] + ['Volume_MA16'] + ['day_of_week']  # For SPY and VIX
# toda__y_cols_to_norm       = [('Close_MA16', 'SPY')]
