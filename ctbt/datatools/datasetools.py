import sys, runpy
from pathlib import Path
from typing import List

from freqtrade.data.history.history_utils import load_pair_history

### CTBT modules ###
from ctbt.datatools.wtseries_training import WTSeriesTraining
from ctbt.datatools.preprocessing import WTSeriesPreProcess
from ctbt.datatools.indice import Indice, add_indicators
from ctbt.envtools.hometools import CtbtHome

def run_download_freqtrade(pairs : List[str], 
                           timeframes : List[str], 
                           days : int, 
                           exchange : str, 
                           userdir : Path):
    old_sysargv = sys.argv
    sys.argv = [sys.argv[0]]
    sys.argv.append("download-data")
    sys.argv.append("--pairs")
    sys.argv.extend(pairs)
    sys.argv.append("-t")
    sys.argv.extend(timeframes)
    sys.argv.extend(["--days", str(days)])
    sys.argv.extend(["--exchange", exchange])
    sys.argv.extend(["--userdir", str(userdir)])
    try:
        runpy.run_module("freqtrade", run_name="__main__")
    except SystemExit as exception:
        sys.argv = old_sysargv
        return exception.code
    sys.argv = old_sysargv
    

def write_wtstdataset_from_raw_data(project : CtbtHome,
                                    rawdata_dir : Path,
                                    pairs : List[str],
                                    timeframes : List[str],
                                    dataset_name : str,
                                    input_size : int,
                                    target_size : int,
                                    offset : int,
                                    window_step : int,
                                    n_interval : int,
                                    index_column : str,
                                    prop_tv : float,
                                    prop_v : float,
                                    indices : List[Indice],
                                    preprocess : WTSeriesPreProcess):
    
    dataset = WTSeriesTraining(input_size,
                               target_size,
                               index_column,
                               project=project)
    for pair in pairs:
        for tf in timeframes:
                project.log.info(f"Loading data ({pair}-{tf}) in {rawdata_dir}")
                pair_history = load_pair_history(pair, 
                                                 tf, 
                                                 rawdata_dir)
                if indices:
                    project.log.info(f"Adding indicators : {[i.value for i in indices]}")
                    pair_history = add_indicators(pair_history, indices)
                    
                dataset.add_time_serie(pair_history, 
                                       prop_tv=prop_tv,
                                       prop_v=prop_v,
                                       do_shuffle=False,
                                       n_interval=n_interval,
                                       offset=offset,
                                       window_step=window_step,
                                       preprocess=preprocess)
                
    dataset.write(project.data_dir, dataset_name)