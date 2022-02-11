
import argparse
from pathlib import Path
from typing import List

### CG-RBI modules ###
from CGrbi.datatools.indice import Indice
from CGrbi.datatools.datasetools import write_wtstdataset_from_raw_data, run_download_freqtrade
from CGrbi.datatools.preprocessing import WTSeriesPreProcess


def build_dataset(userdir : Path,
                  datadir : Path,
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
                  preprocess : WTSeriesPreProcess):
    
    run_download_freqtrade(pairs=pairs, 
                           timeframes=timeframes, 
                           days=days, 
                           exchange=exchange, 
                           userdir=userdir)
    
    rawdata_dir : Path = userdir.joinpath("data", exchange)
    
    write_wtstdataset_from_raw_data(wtstdataset_dir=datadir,
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
    
    