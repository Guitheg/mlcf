
import pytest
from pathlib import Path

import random
from mlcf.datatools.data_intervals import DataInIntervals

from mlcf.datatools.data_reader import read_ohlcv_json_from_file
random.seed(0)


@pytest.fixture
def testdatadir() -> Path:
    return (Path(__file__).parent / "testdata").resolve()


@pytest.fixture
def rawdata_btc_path(testdatadir):
    return Path(testdatadir / "BTC_BUSD-15m.json")


@pytest.fixture
def rawdata_eth_path(testdatadir):
    return Path(testdatadir / "ETH_BUSD-15m.json")


@pytest.fixture
def ohlcv_data_path(testdatadir):
    return Path(testdatadir / "OHLCV-data.json")


@pytest.fixture
def uncompatible_file_path(testdatadir):
    return Path(testdatadir / "UncompatibleFile.json")


@pytest.fixture
def ohlcv_btc(rawdata_btc_path):
    return read_ohlcv_json_from_file(rawdata_btc_path)
