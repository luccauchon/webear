@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "RANGE=666"
set "DATASET_ID=day"
set "EPSILON=0.0"
set "TIMEOUT=40000"
set "CONDA_ACTIVATE=C:\Users\cj3272\AppData\Local\miniconda3\Scripts\activate.bat"
set "CONDA_ENV=PY312"
set "WORK_DIR=C:\Projets\webear\src\optimizers"

:: Use "call" for the activation script specifically.
:: I added quotes around the path to handle potential spaces.
start "Roulette Search" cmd /k ^
    "call "%CONDA_ACTIVATE%" %CONDA_ENV% && ^
    cd /D "%WORK_DIR%" && ^
    python roulette_hyperparameter_search_optuna.py ^
    --step_back_range %RANGE% ^
    --dataset_id %DATASET_ID% ^
    --optimize_target pos_seq__f1 ^
    --specific_wanted_class 0 1 2 ^
    --timeout %TIMEOUT% ^
    --epsilon %EPSILON% ^
    --activate_ema_space_search true ^
    --max_ema_slots 5 ^
    --ema_min 2 ^
    --ema_max 10 ^
    --ema_step 2 ^
    --max_ema_shift_slots 3 ^
    --ema_shift_min 1 ^
    --ema_shift_max 5 ^
    --activate_sma_space_search false ^
    --activate_rsi_space_search false ^
    --activate_macd_space_search false ^
    --activate_vwap_space_search false ^
    --add_only_vwap_z_and_vwap_triggers false"

sleep 1



start "Roulette Search" cmd /k ^
    "call conda activate %CONDA_ENV% && ^
    cd /D "%WORK_DIR%" && ^
    python roulette_hyperparameter_search_optuna.py ^
        --step_back_range %RANGE% ^
        --dataset_id %DATASET_ID% ^
        --optimize_target pos_seq__f1 ^
        --specific_wanted_class 0 1 2 ^
        --timeout %TIMEOUT% ^
        --epsilon %EPSILON% ^
        --activate_ema_space_search false ^
        --max_ema_slots 5 ^
        --ema_min 2 ^
        --ema_max 10 ^
        --ema_step 2 ^
        --max_ema_shift_slots 3 ^
        --ema_shift_min 1 ^
        --ema_shift_max 5 ^
        --activate_sma_space_search true ^
        --max_sma_slots 5 ^
        --sma_min 5 ^
        --sma_max 50 ^
        --sma_step 5 ^
        --max_sma_shift_slots 3 ^
        --sma_shift_min 1 ^
        --sma_shift_max 5 ^
        --activate_rsi_space_search false ^
        --activate_macd_space_search false ^
        --activate_vwap_space_search false ^
        --add_only_vwap_z_and_vwap_triggers false"

sleep 1


echo All processes have been dispatched.
pause