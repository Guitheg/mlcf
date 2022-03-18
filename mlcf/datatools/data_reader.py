"""Data reader module.
This module allows to read data from file to convert it into pandas.DataFrame
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


__all__ = [
    "OHLCVUncompatibleFileException",
    "UncompatibleFileException",
    "NoFileException",
    "NoDirectoryException",
    "read_ohlcv_json_from_file",
    "read_ohlcv_json_from_dir",
    "read_json_file"]


class OHLCVUncompatibleFileException(Exception):
    pass


class UncompatibleFileException(Exception):
    pass


class NoFileException(Exception):
    pass


class NoDirectoryException(Exception):
    pass


def read_ohlcv_json_from_file(path: Path) -> pd.DataFrame:
    """
    Read a OHLCV Json file give by {path} and return a data frame.
    The OHLCV Json File must respect the format : {column -> {index -> value}}

    Args:
        path (Path): The file path to the OHLCV JSON file.

    Raises:
        OHLCVUncompatibleFileException: Raise this exception if the given file hasn't a OHLCV data.
        NoFileException: If the path lead to any file

    Returns:
        pd.DataFrame: return the data frame built from the OHLCV data's json file.
    """

    try:
        data = read_json_file(
            path,
            index_name="date",
            col_names=["date", "open", "high", "low", "close", "volume"],
            columns_to_time={'date': 'ms'}
        )
    except UncompatibleFileException:
        raise OHLCVUncompatibleFileException("This file hasn't OHLCV data")
    return data


def read_ohlcv_json_from_dir(
    directory: Path,
    pair: str,
    timeframe: str,
    filename_format: str = "{pair}-{tf}.json",
) -> pd.DataFrame:
    """Given a directory, a pair and a timeframe returns the data of the finded OHLCV JSON file

    Args:
        directory (Path): Directory where the pairs are stored.
        pair (str): The wanted pair.
        timeframe (str): The wanted timeframe
        filename_format (str, optional): The filename string fromat. Defaults to "{pair}-{tf}.json".

    Raises:
        NoDirectoryException: If the path lead to any directory

    Returns:
        pd.DataFrame: return the data frame built from the OHLCV data's json file.
    """
    if not directory.is_dir():
        raise NoDirectoryException(f"The path {directory} lead to any directory")
    file_path = directory.joinpath(filename_format.format(pair=pair, tf=timeframe))
    return read_ohlcv_json_from_file(file_path)


def read_json_file(
    file_path: Path,
    index_name: Optional[str] = None,
    col_names: Optional[List[str]] = None,
    columns_to_time: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Read a JSON file to return a DataFrame. We can give an {index_name} to set the index on
    the {index_name} column. We can provide the column_names if the header is not provided in the
    JSON file. The size of columns_names list must be equal to the number of columns of the data in
    the file.
    The JSON file must respect the format: {column -> {index -> value}}

    Args:
        file_path (Path): The file path to the JSON file.
        index_name (Optional[str], optional): The column name which will be set as the new index.
        Defaults to None.
        col_names (Optional[List[str]], optional): The exhaustive list of column name of the data.
        If there are an index column, it should be there too. Defaults to None.
        columns_to_time (Optional[List[str]], optional): A dict of column to convert to time
        such as : Dict(key: column name, value: time unit). example : {'date': 'ms'}.
        Defaults to None.

    Raises:
        NoFileException: If the path lead to any file
        UncompatibleFileException: If len(col_names) != data.shape[1]

    Returns:
        pd.DataFrame: return the data frame built from the data's json file.
    """
    if not file_path.is_file():
        raise NoFileException(f"The path {file_path} lead to any file")
    data = pd.read_json(file_path).values

    if col_names:
        if len(col_names) != data.shape[1]:
            raise UncompatibleFileException(
                "The column names list hast not the same size" +
                f"of the number of column of the data: ({len(col_names)} != {data.shape[1]}")
        df_data = pd.DataFrame(data, columns=col_names)
    else:
        df_data = pd.DataFrame(data)
    if columns_to_time:
        for time_column, unit in columns_to_time.items():
            df_data[time_column] = pd.to_datetime(df_data[time_column], unit=unit)
    if index_name:
        df_data.set_index(index_name, inplace=True)
    return df_data
