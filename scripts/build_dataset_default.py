from pathlib import Path
import runpy
import sys
from os.path import join, dirname, normpath,abspath
import argparse
import os
from tkinter import OFF
from freqtrade.data.history.history_utils import load_pair_history

from ritl import add
add(__file__, "..")
from CGrbi.datatools.wtseries_training import WTSeriesTraining
from CGrbi.datatools.preprocessing import AutoNormalize
from CGrbi.datatools.indice import Indice, add_indicators
from CGrbi.envtools.project import Project
from CGrbi.envtools.path import get_dir_prgm

INPUT_SIZE = 50
TARGET_SIZE = 1
OFFSET = 5
COLUMN_INDEX = "date"
TEST_VAL_PROP = 0.1
VAL_PROP = 0.2
WINDOW_STEP=1
N_INTERVAL=4
LIST_INDICE = [Indice.ICHIMOKU, Indice.RSI, Indice.STOCH_SLOW, Indice.STOCH_FAST, 
               Indice.ADX, Indice.SMA, Indice.AROON, Indice.EMA]

HOME_USER =  normpath(join(dirname(__file__), "..", "user_data"))
USER_DATA_PATH = normpath(join(HOME_USER, "data"))

def run_download_freqtrade(sysargs):
    sys.argv = [sys.argv[0]]
    sys.argv.append("download-data")
    sys.argv.extend(["--userdir", HOME_USER])
    sys.argv.extend(sysargs[1:])
    try:
        runpy.run_module("freqtrade", run_name="__main__")
    except SystemExit as exception:
        return exception.code

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", type=str)
    parser.add_argument("--days", type=str)
    parser.add_argument("--timeframe", type=str, nargs="+")
    parser.add_argument("--pairs", type=str, nargs="+")
    args = parser.parse_args()
    
    run_download_freqtrade(sys.argv)
    
    dataset = WTSeriesTraining(INPUT_SIZE,
                               TARGET_SIZE,
                               column_index=COLUMN_INDEX)
    for pair in args.pairs:
        for tf in args.timeframe:
                pair_history = load_pair_history(pair, 
                                                 tf, 
                                                 Path(join(USER_DATA_PATH, args.exchange)))
                pair_history = add_indicators(pair_history, LIST_INDICE)
                dataset.add_time_serie(pair_history, 
                                       test_val_prop=TEST_VAL_PROP,
                                       val_prop=VAL_PROP,
                                       do_shuffle=False,
                                       offset=OFFSET,
                                       window_step=WINDOW_STEP,
                                       preprocess=AutoNormalize)
    dataset.write(USER_DATA_PATH, "default_dataset")
              
if __name__ == "__main__":
    main()