trainer__debug_level = "DEBUG"
trainer__run_id=123
trainer__version="12xs"

trainer__direction='up'

trainer__type_margin = 'relative'
##################################
trainer__margin      = 0.1
##################################

trainer__wanted_pos_weight = 1.

trainer__lr_decay_iters= 2001
trainer__max_iters     = 4001
trainer__learning_rate = 1e-3
trainer__min_lr        = 1e-5
trainer__weight_decay  = 0.1
trainer__decay_lr      = True

trainer__tav_dates      = ["2018-01-01", "2025-03-29"]
trainer__mes_dates      = ["2025-01-01", "2025-04-05"]
trainer__data_interval  = "1wk"
trainer__number_of_timestep_for_validation = 24

trainer__log_interval  = 1000
trainer__eval_interval = 10

trainer__force_download_data = False


##################################
trainer__x_seq_length = 3
##################################
trainer__y_seq_length = 1

trainer__x_cols = ['Close', 'Open', 'High', 'Low', 'Close_MA3', 'Open_MA3', 'High_MA3', 'Low_MA3']

##################################
trainer__y_cols = [('Close_MA3', 'SPY')]
##################################

trainer__batch_size=1024

trainer__jump_ahead=0
trainer__shuffle_indices=False
trainer__save_checkpoint=False
trainer__data_augmentation=True
trainer__frequency_of_noise=0.5
trainer__power_of_noise=2.5
trainer__df_source=None
