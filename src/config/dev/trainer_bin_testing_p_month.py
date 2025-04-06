trainer__debug_level = "DEBUG"
trainer__run_id=123
trainer__version="22xs"

trainer__direction='down'

trainer__type_margin = 'relative'
##################################
trainer__margin      = 0.1
##################################

trainer__wanted_pos_weight = 1.

trainer__lr_decay_iters= 4001
trainer__max_iters     = 4001
trainer__learning_rate = 1e-3
trainer__min_lr        = 1e-5
trainer__weight_decay  = 0.1
trainer__decay_lr      = True

trainer__tav_dates      = ["2018-01-01", "2025-03-29"]
trainer__mes_dates      = ["2024-12-01", "2025-04-05"]
trainer__data_interval  = "1mo"
trainer__number_of_timestep_for_validation = 15

trainer__log_interval  = 200
trainer__eval_interval = 10

trainer__force_download_data = False


##################################
trainer__x_seq_length = 9
##################################
trainer__y_seq_length = 1

trainer__x_cols = ['Close', 'High', 'Low', 'Close_MA3'] + ['month_of_year']

##################################
trainer__y_cols = [('Close_EMA4', 'SPY')]
##################################

trainer__batch_size=1024

trainer__jump_ahead=0
trainer__shuffle_indices=False
trainer__save_checkpoint=False
trainer__data_augmentation=True
trainer__frequency_of_noise=0.5
trainer__power_of_noise=5.5
trainer__df_source=None
