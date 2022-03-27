
import pytest
import pandas as pd
import numpy as np
from mlcf.indicators.add_indicators import (
    FeatureAlreadyExistException,
    add_extern_indicator,
    add_intern_indicator
)


def test_add_extern_indicator_exception(ohlcv_btc, rawdata_eth_path):
    with pytest.raises(FeatureAlreadyExistException):
        add_extern_indicator(
            ohlcv_btc,
            rawdata_eth_path,
            ["open", "high", "low", "close", "volume"])


def test_add_extern_indicator(ohlcv_btc, rawdata_eth_path):
    dataframe = add_extern_indicator(
        ohlcv_btc,
        rawdata_eth_path,
        ["Eopen", "Ehigh", "Elow", "Eclose", "Evolume"])

    assert list(dataframe.columns) == [
        "open", "high", "low", "close", "volume", "Eopen", "Ehigh", "Elow", "Eclose", "Evolume"]
    # first value of ETH
    assert dataframe.loc[pd.to_datetime(1571644800000, unit="ms"), "Eopen"] == 175.0
    # value before the first value of ETH
    assert np.isnan(dataframe.loc[pd.to_datetime(1571643900000, unit="ms"), "Eopen"])


def test_add_intern_indicator(ohlcv_btc):
    dataframe = add_intern_indicator(ohlcv_btc, "adx")
    assert len(dataframe.columns) == len(ohlcv_btc.columns) + 1
