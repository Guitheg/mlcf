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
    pairs: List[str],
    timeframes: List[str],
    input_width: int,
    target_width: int,
    offset: int,
    window_step: int,
    n_interval: int,
    index_column: str,
    prop_tv: float,
    prop_v: float,
    indices: List[Indice],
    preprocess: WTSeriesPreProcess,
    merge_pairs: bool,
    n_category: int,
    standardize: bool,
    selected_columns: List[str],
    unselected_columns: List[str],
    *args,
    **kwargs
):

    write_wtstdataset_from_raw_data(
        project=project,
        rawdata_dir=rawdata_dir,
        dataset_name=dataset_name,
        pairs=pairs,
        timeframes=timeframes,
        input_width=input_width,
        target_width=target_width,
        offset=offset,
        window_step=window_step,
        n_interval=n_interval,
        index_column=index_column,
        prop_tv=prop_tv,
        prop_v=prop_v,
        indices=indices,
        preprocess=preprocess,
        merge_pairs=merge_pairs,
        n_category=n_category,
        standardize=standardize,
        selected_columns=selected_columns,
        unselected_columns=unselected_columns
    )
    del args
    del kwargs
