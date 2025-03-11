import os
from datetime import datetime

dp__debug_level       = 'DEBUG'
dp__seed_offset       = 123

dp__list_of_symbols   = [{"symbol":"^VIX", "period":"max", "interval":"1d", "drop": ["Volume"]},
                         {"symbol":"SPY", "period":"max", "interval":"1d"}]

dp__day_of_week       = True

dp__output_dir        = os.path.join("stubs", "data_preparation", f"{datetime.now().strftime('run_%Y%m%d_%Hh%Mm%Ss')}")
dp__output_filename   = os.path.join(dp__output_dir, "multi_cols_revers_rc1.pkl")
dp__final_destination = None  # Copy "dp__output_filename" to that place