fast_execution_for_debugging     = True
skip_training_with_already_computed_results = [r"D:\PyCharmProjects\webear\stubs\kepler__0329_10h38m\__trainers__"]

debug_level = "DEBUG"

tav_dates      = ["2024-01-01", "2025-03-22"]
mes_dates      = ["2025-03-23", "2025-03-29"]
inf_dates      = ["2025-03-31", "2025-03-31"]

runner__lr_decay_iters= 4001
runner__max_iters     = 4001
runner__learning_rate = 1e-3
runner__min_lr        = 1e-4
runner__weight_decay  = 0.1
runner__log_interval  = 1000
runner__eval_interval = 1
runner__x_seq_length  = 4
runner__y_seq_length  = 1
runner__x_cols        = ['Close', 'Close_MA3','Close_MA5'] + ['day_of_week']  # For SPY and VIX
runner__x_cols_to_norm= []
runner__y_cols        = [('Close_MA3', 'SPY')]
runner__batch_size    =1024
runner__jump_ahead    =0
runner__type_margin   ='relative'
runner__margin        =0.2
