
import pytest
import pandas as pd
import numpy as np
from mlcf.indicators.add_indicators import (
    FeatureAlreadyExistException,
    add_all_intern_indicators,
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


@pytest.mark.parametrize(
    "test_input",
    [
        {
            "indicator_name": "adx"
        },
        {
            "indicator_name": "aroon",
        },
        {
            "indicator_name": "abs_energy",
            "column": "volume",
            "timeperiod": 10,
        },
        {
            "indicator_name": "autocorrelation",
            "column": "close",
            "timeperiod": 10,
            "lag": 5
        },
        {
            "indicator_name": "returns",
            "column": "close",
        }
    ]
)
def test_add_intern_indicator(ohlcv_btc, test_input):
    dataframe = add_intern_indicator(ohlcv_btc, **test_input)
    assert len(dataframe)


@pytest.mark.parametrize(
    "indicator_list",
    [
        [
            ("adx", {"custom_column_name": "ADX"}),
            ("aroon", {"custom_column_name": "aroon"}),
            ("abs_energy", {"column": "volume", "timeperiod": 10, "custom_column_name": "abs"}),
            ("autocorrelation", {"column": "close", "timeperiod": 10, "lag": 5}),
            ("returns", {"column": "close", "custom_column_name": "r"})
        ]
    ]
)
def test_add_all_intern_indicators(ohlcv_btc, indicator_list):
    dataframe = add_all_intern_indicators(ohlcv_btc, indicator_list)
    assert len(dataframe)
