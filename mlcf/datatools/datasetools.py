from functools import reduce
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

from mlcf.datatools.indice import Indice, add_indicators
from mlcf.datatools.preprocessing import WTSeriesPreProcess

# MLCF modules
from mlcf.datatools.wtst_dataset import WTSTrainingDataset
from mlcf.envtools.hometools import MlcfHome


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
    if len(dict_of_dataframe.keys()) == 1:
        return dict_of_dataframe[list(dict_of_dataframe.keys())[0]]
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
    selected_columns: List[str] = [],
    unselected_columns: List[str] = []
):

    dataset = WTSTrainingDataset(
        dataset_path=project.data_dir.joinpath(dataset_name),
        input_width=input_width,
        target_width=target_width,
        index_column=index_column,
        project=project
    )

    list_to_std: Set[str] = set()
    rawdata_set: Dict[str, Dict[str, pd.DataFrame]] = {}
    for pair in pairs:
        for tf in timeframes:
            filename = f"{pair.replace('/','_')}-{tf}.json"
            raw_data = read_ohlcv_json_rawdata(rawdata_dir.joinpath(filename))
            if indices:
                project.log.info(f"Adding indicators: {[i.value for i in indices]}")
                raw_data = add_indicators(
                    raw_data,
                    indices,
                    dropna=True,
                    standardize=standardize,
                    list_to_std=list_to_std)

            # set selected columns and set features of the dataset
            if not selected_columns:
                selected_columns = list(
                    set(list(raw_data.columns)) - set(unselected_columns) - set([index_column])
                )
            if not dataset.features_has_been_set:
                dataset.set_features(selected_columns)

            if tf not in rawdata_set:
                rawdata_set[tf] = {pair: raw_data}
            else:
                rawdata_set[tf][pair] = raw_data

    if standardize:
        list_to_std.add("open")
        list_to_std.add("close")
        list_to_std.add("low")
        list_to_std.add("high")
        list_to_std.add("volume")

    list_to_std = list_to_std & set(selected_columns)

    if merge_pairs:
        for i, tf in enumerate(rawdata_set):
            data_to_add = merge_dict_of_dataframe(rawdata_set[tf], index_column=index_column)
            project.log.info(
                f"List features: {list(data_to_add.columns)} " +
                f"{'(the data will be balance)' if n_category > 1 else ''}")
            project.log.info(f"Dataset built at {i/len(rawdata_set):.0%}")
            dataset.add_time_serie(
                data_to_add,
                prop_tv=prop_tv,
                prop_v=prop_v,
                n_interval=n_interval,
                offset=offset,
                window_step=window_step,
                preprocess=preprocess,
                n_category=n_category,
                list_to_std=list(list_to_std)
            )
    else:
        for i, tf in enumerate(rawdata_set):
            for j, pair in enumerate(rawdata_set[tf]):
                project.log.info(
                    f"List features: {list(rawdata_set[tf][pair].columns)} " +
                    f"{'(the data will be balance)' if n_category > 1 else ''}")
                p = (i+((len(rawdata_set[tf])-1)*j)) / (len(rawdata_set)*len(rawdata_set[tf]))
                project.log.info(f"Dataset built at {p:.0%}")
                dataset.add_time_serie(
                    rawdata_set[tf][pair],
                    prop_tv=prop_tv,
                    prop_v=prop_v,
                    n_interval=n_interval,
                    offset=offset,
                    window_step=window_step,
                    preprocess=preprocess,
                    n_category=n_category,
                    list_to_std=list(list_to_std)
                )

    project.log.debug(f"Dataset built:{dataset}")
