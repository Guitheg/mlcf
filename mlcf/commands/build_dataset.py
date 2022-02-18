from pathlib import Path
from typing import List

from mlcf.datatools.datasetools import write_wtstdataset_from_raw_data


# MLCF modules
from mlcf.datatools.indice import Indice
from mlcf.datatools.preprocessing import WTSeriesPreProcess
from mlcf.envtools.hometools import MlcfHome


def build_dataset(
    project: MlcfHome,
    rawdata_dir: Path,
    dataset_name: str,
    input_size: int,
    target_size: int,
    offset: int,
    window_step: int,
    n_interval: int,
    index_column: str,
    prop_tv: float,
    prop_v: float,
    indices: List[Indice],
    preprocess: WTSeriesPreProcess,
    *args,
    **kwargs
):

    write_wtstdataset_from_raw_data(
        project=project,
        rawdata_dir=rawdata_dir,
        dataset_name=dataset_name,
        input_size=input_size,
        target_size=target_size,
        offset=offset,
        window_step=window_step,
        n_interval=n_interval,
        index_column=index_column,
        prop_tv=prop_tv,
        prop_v=prop_v,
        indices=indices,
        preprocess=preprocess
    )
    del args
    del kwargs
