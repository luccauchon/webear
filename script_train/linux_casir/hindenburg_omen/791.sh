#!/bin/bash
cd ../../../src/optimizers/hindenburg_omen
python use_case_generator.py --nb-workers 128 --timeout 85000 --output-dir ~/14b/cj3272/experiences/output_hindenburg_omen/ \
     --experience-id 791 --forward-days 10 11 12 13 14 15 16 17 18 19 20 \
     --thresholds -0.03 -0.04 -0.05 -0.06 -0.07 -0.08 -0.09 -0.1 -0.11 -0.12 -0.13 --penalties 0.999 \
     --lookback-years 15 --base-signals simple_ma,hull_ma_cross,supertrend,atr_expansion,bb_extreme

