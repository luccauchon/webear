trainer__df_source=None
trainer__data_interval = '1d'
trainer__force_download_data = False
trainer__run_id  = 1
trainer__version = "v1"

trainer__tav_dates      = ["2024-01-01", "2025-04-05"]
trainer__mes_dates      = ["2025-04-01", "2025-04-12"]

trainer__x_seq_length   = 15
trainer__y_seq_length   = 1

trainer__x_cols_mutable = ['Close_SPY', 'Volume_SPY', 'Close_^VIX']
trainer__x_cols_fixed   = ['day_of_week']

trainer__batch_size = 2048

trainer__learning_rate = 1e-3
trainer__min_lr        = 1e-5
trainer__weight_decay  = 0.1

trainer__lr_decay_iters= 999001
trainer__max_iters     = 999001

trainer__decay_lr      = True
trainer__eval_interval = 5
trainer__log_interval  = 10