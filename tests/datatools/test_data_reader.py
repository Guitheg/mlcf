from pathlib import Path
from pandas import Timestamp
import pytest
from mlcf.datatools.data_reader import (
    NoDirectoryException,
    NoFileException,
    OHLCVUncompatibleFileException,
    UncompatibleFileException,
    read_json_file,
    read_ohlcv_json_from_dir,
    read_ohlcv_json_from_file
)


def test_read_ohlcv_json_from_file(ohlcv_data_path, uncompatible_file_path):
    with pytest.raises(OHLCVUncompatibleFileException):
        data = read_ohlcv_json_from_file(uncompatible_file_path)

    with pytest.raises(NoFileException):
        data = read_ohlcv_json_from_file(Path("/testdata/nofile.json"))

    data = read_ohlcv_json_from_file(ohlcv_data_path)
    assert list(data.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(data.index[0], Timestamp)
    assert data.index.name == "date"


def test_read_ohlcv_json_from_dir(testdatadir):
    with pytest.raises(NoDirectoryException):
        data = read_ohlcv_json_from_dir(Path("unknowndirectory/"), "OHLCV", "data")
    data = read_ohlcv_json_from_dir(testdatadir, "OHLCV", "data")
    assert list(data.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(data.index[0], Timestamp)
    assert data.index.name == "date"


def test_read_json_file(uncompatible_file_path):
    index_name = "date"
    header = ["date", "a", "b", "c", "d"]
    with pytest.raises(NoFileException):
        data = read_json_file(Path("/testdata/nofile.json"), index_name, header)
    with pytest.raises(UncompatibleFileException):
        data = read_json_file(uncompatible_file_path, index_name, header[:-1])

    data = read_json_file(uncompatible_file_path, index_name, header)
    assert list(data.columns) == header[1:]
    assert data.index.name == index_name
