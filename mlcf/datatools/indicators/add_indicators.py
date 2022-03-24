"""Add indicators module.
This module provide function to add intern or extern indicators as features
on a time series dataframe.
"""

from typing import List

from pathlib import Path
from mlcf.datatools import read_json_file
import pandas as pd
import numpy as np

from mlcf.datatools.indicators.indicators_fct import INDICE_DICT

# TODO: (doc)

__all__ = [
    "FeatureAlreadyExistException",
    "add_extern_indicator",
    "add_intern_indicator"
]


class FeatureAlreadyExistException(Exception):
    pass


def add_extern_indicator(
    data: pd.DataFrame,
    file_path: Path,
    feature_names: List[str],
    time_index_name: str = "date"
) -> pd.DataFrame:
    """This function allows to add new features with extern indicators (provided by a json file).
    The data's file must have a time column as index. The data's features will be merged to the
    given {data} pandas.DataFrame.

    Args:
        data (pd.DataFrame): The pandas.DataFrame time series OHLCV data
        file_path (Path): The file path to the extern indicators file.
        feature_names (List[str]): The names of the features.
        time_index_name (str, optional): The name of the time index column. Defaults to "date".

    Returns:
        pd.DataFrame: The new extern features merged to the time series OHLCV dataframe
    """
    dataframe = data.copy()
    feature_columns: List[str] = feature_names
    if time_index_name in feature_names:
        feature_columns.remove(time_index_name)

    f_exist = list(
        np.array(feature_columns)[[i in list(dataframe.columns) for i in feature_columns]])
    if f_exist:
        raise FeatureAlreadyExistException(f"The features {f_exist} already exist in the dataframe")

    columns: List[str] = [time_index_name] + feature_columns
    data = read_json_file(
        file_path,
        index_name=time_index_name,
        col_names=columns,
        columns_to_time={time_index_name: 'ms'}
    )
    dataframe.loc[:, feature_columns] = data.loc[:, feature_columns]
    return dataframe


def add_intern_indicator(
    data: pd.DataFrame,
    indice_name: str,
    *args, **kwargs
) -> pd.DataFrame:
    """This function allows to add new features with intern indicators (provided by processing the
    OHLCV data).
    The new data's features will be merged to the given {data} pandas.DataFrame.

    Args:
        data (pd.DataFrame): The OHLCV pandas.DataFrame
        indice_name (str): The name of the function which will add new features to the dataframe

    Returns:
        pd.DataFrame: The dataframe merged with the new features
    """
    dataframe = data.copy()
    dataframe = INDICE_DICT[indice_name](dataframe, *args, **kwargs)
    return dataframe
