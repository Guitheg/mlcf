
import argparse
from pathlib import Path
from typing import List

### CG-RBI modules ###
from CGrbi.datatools.indice import Indice
from CGrbi.datatools.datasetools import write_wtstdataset_from_raw_data, run_download_freqtrade
from CGrbi.datatools.preprocessing import WTSeriesPreProcess
from CGrbi.envtools.project import CGrbi


def build_dataset(project : CGrbi,
                  userdir : Path,
                  pairs : List[str],
                  timeframes : List[str],
                  days : int,
                  exchange : str,
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
                  preprocess : WTSeriesPreProcess,
                  *args,
                  **kwargs):
    rawdata_dir : Path = userdir.joinpath("data", exchange)
    project.log.info(f"Download data with freqtrade. Pairs : {pairs}, Timeframes : {timeframes}, "+
                     f"Days of historic : {days}, Market : {exchange}. Save here : {rawdata_dir} ")
    run_download_freqtrade(pairs=pairs, 
                           timeframes=timeframes, 
                           days=days, 
                           exchange=exchange, 
                           userdir=userdir)
    write_wtstdataset_from_raw_data(project=project,
                                    rawdata_dir=rawdata_dir,
                                    dataset_name=dataset_name,
                                    pairs=pairs,
                                    timeframes=timeframes,
                                    input_size=input_size,
                                    target_size=target_size,
                                    offset=offset,
                                    window_step=window_step,
                                    n_interval=n_interval,
                                    index_column=index_column,
                                    prop_tv=prop_tv,
                                    prop_v=prop_v,
                                    indices=indices,
                                    preprocess=preprocess)
    
    