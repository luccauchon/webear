runner__debug_level         = "INFO"

runner__fast_execution_for_debugging                = False
runner__skip_training_with_already_computed_results = []
runner__fetch_new_dataframe = False
runner__master_df_source    = None
runner__inf_power_of_noise  = 1.
runner__inf_frequency_of_noise = 0.
runner__nb_iter_test_in_inference = 5
runner__download_data_for_inf = True

runner__tav_dates      = ["2018-01-01", "2025-03-29"]
runner__mes_dates      = ["2025-01-01", "2025-04-05"]
runner__inf_dates      = ["2025-04-07", "2025-04-11"]
runner__skip_monday    = False


###############################################################################
# Trainer
###############################################################################
trainer__lr_decay_iters= 2001
trainer__max_iters     = 4001
trainer__learning_rate = 1e-3
trainer__min_lr        = 1e-5
trainer__weight_decay  = 0.1
trainer__log_interval  = 1000
trainer__eval_interval = 1
trainer__x_seq_length  = 3
trainer__y_seq_length  = 1
trainer__x_cols        = ['Close', 'Close_MA3','Close_MA5'] + ['day_of_week']  # For SPY and VIX
trainer__y_cols        = [('Close_MA3', 'SPY')]
trainer__batch_size    =1024
trainer__jump_ahead    =0
trainer__type_margin   ='relative'
trainer__margin        =0.1
trainer__wanted_pos_weight = 1.
trainer__data_interval = "1wk"
trainer__number_of_timestep_for_validation = 24
trainer__data_augmentation=True
trainer__frequency_of_noise=0.5
trainer__power_of_noise=2.5