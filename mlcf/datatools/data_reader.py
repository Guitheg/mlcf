"""Data Reader Module.

This module reads data from a file and converts it into pandas.DataFrame.

    Example:

    .. code-block:: python

        from pathlib import Path
        from mlcf.datatools.data_reader import (
            read_ohlcv_json_from_file,
            read_ohlcv_json_from_dir,
            read_json_file
        )

        # from a ohlcv json file
        data = read_ohlcv_json_from_file(Path("tests/testdata/ETH_BUSD-15m.json"))

        # from a directory, a pair, and a timeframe
        pair = "ETH_BUSD"
        tf = "15m"
        data = read_ohlcv_json_from_dir(Path("tests/testdata/"), pair=pair, timeframe=tf)

        # read a json file (but not necessary a OHLCV file)
        data = read_json_file(Path("tests/testdata/meteo.json"), 'time', ["time", "Temperature"])
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


__all__ = [
    "read_ohlcv_json_from_file",
    "read_ohlcv_json_from_dir",
    "read_json_file",
    "OHLCVUncompatibleFileException",
    "UncompatibleFileException",
    "NoFileException",
    "NoDirectoryException"
]


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
    This function reads a OHLCV JSON file given by a {path} and returns a pandas.DataFrame.
    The OHLCV JSON file must respect the following format: {column -> {index -> value}}.

    Args:
        path (Path): The file path to the OHLCV JSON file

    Raises:
        OHLCVUncompatibleFileException: Raises this exception if the given file doesn't have OHLCV
            data.

        NoFileException: If the file is not found in the given path

    Returns:
        pd.DataFrame: A pandas.DataFrame built from the JSON file of OHLCV data
    """

    try:
        data = read_json_file(
            path,
            index_name="date",
            col_names=["date", "open", "high", "low", "close", "volume"],
            columns_to_time={'date': 'ms'}
        )
    except UncompatibleFileException:
        raise OHLCVUncompatibleFileException("This file doesn't have OHLCV data")
    return data


def read_ohlcv_json_from_dir(
    directory: Path,
    pair: str,
    timeframe: str,
    filename_format: str = "{pair}-{timeframe}.json",
) -> pd.DataFrame:
    """
    Given a {directory}, this function returns the data in the OHLCV JSON file corresponding to the
    {pair} and the {timeframe}.

    Args:
        directory (Path): Directory where OHLCV JSON files are stored

        pair (str): The desired pair

        timeframe (str): The desired timeframe

        filename_format (str, optional): Filename in String format. "{pair}-{timeframe}.json"
            by default

    Raises:
        NoDirectoryException: If the directory is not found in the given path

    Returns:
        pd.DataFrame: A pandas.DataFrame built from the JSON file of OHLCV data
    """
    if not directory.is_dir():
        raise NoDirectoryException(f"The {directory} of given path isn't found")
    file_path = directory.joinpath(filename_format.format(pair=pair, timeframe=timeframe))
    return read_ohlcv_json_from_file(file_path)


def read_json_file(
    file_path: Path,
    index_name: Optional[str] = None,
    col_names: Optional[List[str]] = None,
    columns_to_time: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    This function reads a JSON file and returns a pandas.DataFrame. {index_name} specifies the
    desired column to use as an index. {col_names} determines the header if it is not provided in
    JSON. The size of {col_names} must be equal to the number of columns of data in the file.
    Moreover, the function is order sensitive; thus, the columns in {col_names} must be provided in
    the correct order. The JSON file must have the following format: {column -> {index -> value}}.

    Args:
        file_path (Path): The file path to the JSON file

        index_name (Optional[str], optional): The column name which will be set as the new index.
            Defaults to None.

        col_names (Optional[List[str]], optional): The exhaustive list of column names of data.
            {col_names} must contain the column that is chosen as the new index. Defaults to None.

        columns_to_time (Optional[List[str]], optional): A dictionary that contains a pair of column
            name and time unit. It is used to determine which time unit is used to interpret the
            timestamp values of a column.
            such as : Dict(key: column name, value: time unit). example : {'date': 'ms'}.
            Defaults to None.

    Raises:
        NoFileException: If the file is not found in the given path

        UncompatibleFileException: If len({col_names}) != {data}.shape[1]

    Returns:
        pd.DataFrame: A pandas.DataFrame built from the JSON file of OHLCV data
    """
    if not file_path.is_file():
        raise NoFileException(f"The file is not found in the given path: {file_path}")
    data = pd.read_json(file_path).values

    if col_names:
        if len(col_names) != data.shape[1]:
            raise UncompatibleFileException(
                "The list of column names doesn't have the same size as " +
                f"the number of column in JSON file data: ({len(col_names)} != {data.shape[1]}")
        df_data = pd.DataFrame(data, columns=col_names)
    else:
        df_data = pd.DataFrame(data)
    if columns_to_time:
        for time_column, unit in columns_to_time.items():
            df_data[time_column] = pd.to_datetime(df_data[time_column], unit=unit)
    if index_name:
        df_data.set_index(index_name, inplace=True)
    return df_data
