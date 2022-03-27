"""Indicators Module.

This module provide function to add intern or extern indicators as features
on a time series dataframe.

    Example:

    .. code-block:: python

        from mlcf.indicators.add_indicators import add_intern_indicator

        # you can add yoursel your own indicators or features
        data["return"] = data["close"].pct_change(1)
        data.dropna(inplace=True)  # make sure to drop nan values

        # you can add intern indicator
        data = add_intern_indicator(data, indice_name="adx")
"""

from typing import List

from pathlib import Path
from mlcf.datatools import read_json_file
import pandas as pd
import numpy as np

from mlcf.indicators.indicators_fct import indice_dict

# TODO (doc) correct English

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
    """
    This function allows to add new features with extern indicators (provided by a json file).
    The data file must have as index a time column that fits with your data.
    The data format must be such as : {column -> {index -> value}}.
    The new features will be merged to the given {data} pandas.DataFrame.

    Args:
        data (pd.DataFrame): The pandas.DataFrame time series OHLCV data

        file_path (Path): The file path to the extern indicators file.

        feature_names (List[str]): The names of the new features.

        time_index_name (str, optional): The name of the time index column of the file data.
            Defaults to "date".

    Raises:
        FeatureAlreadyExistException: Raise this exception if a feature in the extern file
            indicator already exist in the given dataframe.

    Returns:
        pd.DataFrame: The passed data plus the new extern features.
    """
    dataframe = data.copy()
    feature_columns: List[str] = feature_names
    if time_index_name in feature_names:
        feature_columns.remove(time_index_name)

    f_exist = list(
        np.array(feature_columns)[[i in list(dataframe.columns) for i in feature_columns]])
    if f_exist:
        raise FeatureAlreadyExistException(
            f"The features {f_exist} already exist in the dataframe")

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
    """
    This function allows adding new features with internally calculated indicators.
    (provided by processing the OHLCV data).
    The new features will be merged into the given {data} pandas.DataFrame.

    Args:
        data (pd.DataFrame): The OHLCV pandas.DataFrame

        indice_name (str): It is the name of the calculated indicator which corresponds to a key in
            :obj:`INDICE_DICT <mlcf.indicators.indicators_fct.indice_dict>`
            that refers the corresponding indicator function.


    Returns:
        pd.DataFrame: The dataframe merged with the new features.
    """
    dataframe = data.copy()
    dataframe = indice_dict(indice_name)(dataframe, *args, **kwargs)
    return dataframe
