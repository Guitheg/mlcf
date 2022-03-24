
import pytest
from pathlib import Path

import random
import numpy as np
from mlcf.datatools.data_intervals import DataIntervals

from mlcf.datatools.data_reader import read_ohlcv_json_from_file
from mlcf.datatools.indicators.indicators_fct import add_adx
from mlcf.datatools.utils import labelize
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


@pytest.fixture
def ohlcvr_btc(ohlcv_btc):
    dataframe = ohlcv_btc.copy()
    dataframe["return"] = dataframe["close"].pct_change(1)
    return dataframe.dropna()


@pytest.fixture
def ohlcvra_btc(ohlcvr_btc):
    dataframe = add_adx(ohlcvr_btc)
    return dataframe.dropna()


@pytest.fixture
def ohlcvrl_btc(ohlcvr_btc):
    mean = ohlcvr_btc["return"].mean()
    std = ohlcvr_btc["return"].std()
    return labelize(
        ohlcvr_btc,
        "return",
        10,
        (mean - std, mean + std),
        label_col_name="label"
    )
