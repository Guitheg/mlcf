from functools import reduce
from pathlib import Path
from typing import Dict, List
import pandas as pd

from mlcf.datatools.indice import Indice, add_indicators
from mlcf.datatools.preprocessing import WTSeriesPreProcess

# MLCF modules
from mlcf.datatools.wtseries_training import WTSeriesTraining, read_wtseries_training
from mlcf.envtools.hometools import MlcfHome
from mlcf.envtools.hometools import ProjectHome


def read_wtseries_training_from_file(path: Path, project: ProjectHome):
    return read_wtseries_training(path, project)


def read_ohlcv_json_rawdata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise Exception(f"The path {path} lead to any file")
    data = pd.read_json(path).values
    if data.shape[1] != 6:
        raise Exception("It's seems it is not OHLCV data - because we don't have 6 columns")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data, columns=columns)
    data['date'] = pd.to_datetime(data["date"], unit="ms")
    return data


def select_raw_data_based_on_name(pairs, timeframes, file):
    pair, tf = file.stem.split("-")
    return pair in [pair.replace("/", "_") for pair in pairs] and tf in timeframes


def merge_dict_of_dataframe(
    dict_of_dataframe: Dict[str, pd.DataFrame],
    index_column: str
) -> pd.DataFrame:
    dataframe = reduce(
        lambda item1, item2: pd.merge(
            item1[1],
            item2[1],
            on=index_column,
            suffixes=(f"_{item1[0]}", f"_{item2[0]}")
        ),
        dict_of_dataframe.items()
    )
    return dataframe


def write_wtstdataset_from_raw_data(
    project: MlcfHome,
    rawdata_dir: Path,
    dataset_name: str,
    pairs: List[str],
    timeframes: List[str],
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
    merge_pairs: bool,
    *args, **kwargs
):

    dataset = WTSeriesTraining(
        input_size=input_size,
        target_size=target_size,
        index_column=index_column,
        project=project
    )

    rawdata_set: Dict[str, Dict[str, pd.DataFrame]] = {}
    for pair in pairs:
        for tf in timeframes:
            filename = f"{pair.replace('/','_')}-{tf}.json"
            raw_data = read_ohlcv_json_rawdata(rawdata_dir.joinpath(filename))
            if indices:
                project.log.info(f"Adding indicators: {[i.value for i in indices]}")
                raw_data = add_indicators(raw_data, indices)
            if tf not in rawdata_set:
                rawdata_set[tf] = {pair: raw_data}
            else:
                rawdata_set[tf][pair] = raw_data

    if merge_pairs:
        for tf in rawdata_set:
            data_to_add = merge_dict_of_dataframe(rawdata_set[tf], index_column=index_column)
            project.log.debug(f"List features of the data : {data_to_add.columns}")
            dataset.add_time_serie(
                data_to_add,
                prop_tv=prop_tv,
                prop_v=prop_v,
                do_shuffle=False,
                n_interval=n_interval,
                offset=offset,
                window_step=window_step,
                preprocess=preprocess
            )

    else:
        for tf in rawdata_set:
            for pair in rawdata_set[tf]:
                project.log.debug(f"List features of the data : {rawdata_set[tf][pair].columns}")
                dataset.add_time_serie(
                    rawdata_set[tf][pair],
                    prop_tv=prop_tv,
                    prop_v=prop_v,
                    do_shuffle=False,
                    n_interval=n_interval,
                    offset=offset,
                    window_step=window_step,
                    preprocess=preprocess
                )

    project.log.debug(f"Dataset built:{dataset}")
    dataset.write(project.data_dir, dataset_name)
