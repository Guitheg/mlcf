from pathlib import Path
from typing import Generator, List
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


def read_all_ohlcv_rawdata_from_dir(dir: Path) -> Generator:
    if not dir.is_dir():
        raise Exception(f"The path {dir} lead to any directory")
    for file in dir.iterdir():
        if file.is_file() and file.suffix == ".json":
            data = read_ohlcv_json_rawdata(file)
            yield data


def write_wtstdataset_from_raw_data(
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
    *args, **kwargs
):

    dataset = WTSeriesTraining(
        input_size=input_size,
        target_size=target_size,
        index_column=index_column,
        project=project
    )

    for raw_data in read_all_ohlcv_rawdata_from_dir(rawdata_dir):
        if indices:
            project.log.info(f"Adding indicators: {[i.value for i in indices]}")
            raw_data = add_indicators(raw_data, indices)
        dataset.add_time_serie(
            raw_data,
            prop_tv=prop_tv,
            prop_v=prop_v,
            do_shuffle=False,
            n_interval=n_interval,
            offset=offset,
            window_step=window_step,
            preprocess=preprocess,
        )
    project.log.debug(f"Dataset built:{dataset}")
    dataset.write(project.data_dir, dataset_name)
