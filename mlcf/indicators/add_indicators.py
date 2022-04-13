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
        data = add_intern_indicator(data, indicator_name="adx")
"""

from typing import List

from pathlib import Path
from mlcf.datatools import read_json_file
import pandas as pd
import numpy as np

from mlcf.indicators.indicators_fct import (
    TA_FEATURES,
    TSFRESH_FEATURES,
    CUSTOM_FEATURES,
    add_ta_feature,
    add_tsfresh_feature,
    add_custom_feature
)

# TODO (doc) correct English

__all__ = [
    "FeatureAlreadyExistException",
    "add_extern_indicator",
    "add_intern_indicator"
]


LIST_INTERN_INDICATORS = (
    list(TSFRESH_FEATURES.keys()) +
    list(TA_FEATURES.keys()) +
    list(CUSTOM_FEATURES.keys())
)


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
    indicator_name: str,
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    This function allows adding new features with internally calculated indicators.
    (provided by processing the OHLCV data).
    The new features will be merged into the given {{data}} pandas.DataFrame.

    Args:
        data (pd.DataFrame): The OHLCV pandas.DataFrame

        indicator_name (str): It is the name of the calculated feature which corresponds to a key
            in :obj:`indicator_dict <mlcf.indicators.indicators_fct.indicator_dict>`
            that refers the corresponding indicator function. It can also refers to a TSfresh
            features:
            :obj:`tsfresh_features
            <mlcf.indicators.indicators_fct.statistical_indicators.TSFRESH_FEATURES>`.
            Here the list of available features:
            {feature_list}


    Returns:
        pd.DataFrame: The dataframe merged with the new features.
    """
    dataframe = data.copy()
    if indicator_name in TSFRESH_FEATURES:
        dataframe = add_tsfresh_feature(
            dataframe,
            indicator_name,
            *args,
            **kwargs
        )
    elif indicator_name in TA_FEATURES:
        dataframe = add_ta_feature(
            dataframe,
            indicator_name,
            *args,
            **kwargs
        )
    elif indicator_name in CUSTOM_FEATURES:
        dataframe = add_custom_feature(
            dataframe,
            indicator_name,
            *args,
            **kwargs
        )
    else:
        raise KeyError(
            f"This indicator ({indicator_name}) name doesn't exist in any dictionnary. " +
            f"Please choose between: {LIST_INTERN_INDICATORS}")
    return dataframe


add_intern_indicator.__doc__ = str(add_intern_indicator.__doc__).format(
    feature_list=LIST_INTERN_INDICATORS
)
