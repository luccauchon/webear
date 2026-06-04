#!/bin/bash
sh _laucher_12x.sh --trials 99999 --threshold -0.005   --lookahead 1 --metric long_accuracy
sh _laucher_12x.sh --trials 99999 --threshold -0.0025  --lookahead 1 --metric long_accuracy
sh _laucher_12x.sh --trials 99999 --threshold -0.00125 --lookahead 1 --metric long_accuracy