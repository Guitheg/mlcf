import pytest
from mlcf.datatools.utils import build_forecast_ts_training_dataset
from mlcf.datatools.wtseries import WTSwindowSizeException
from mlcf.datatools.wtst import (
    Field,
    Partition,
    WTSTColumnIndexException,
    WTSTFeaturesException,
    WTSTraining)
import numpy as np
import pandas as pd


def test_WTSeriesTraining(btc_ohlcv):
    ts_data = WTSTraining(9)
    ts_data.add_time_serie(btc_ohlcv.iloc[0:1000], prop_tv=0.2)
    assert len(ts_data) == 800 - 10 + 1


def test_WTSeriesTraining_balance(btc_ohlcv):
    ts_data = WTSTraining(9)
    import random
    random.seed(0)
    ts_data.add_time_serie(btc_ohlcv.iloc[0:1000], prop_tv=0.2, n_category=10)
    assert len(ts_data) == 523


def test_WTSeriesTraining_copy(btc_ohlcv):
    ts_data = WTSTraining(9, index_column="date")
    ts_data.add_time_serie(btc_ohlcv.iloc[0:1000], prop_tv=0.2)

    ts_data_copy = ts_data.copy()
    assert ts_data_copy.input_width == ts_data.input_width
    assert ts_data_copy.features == ts_data.features
    assert ts_data_copy.index_column == ts_data.index_column

    ts_data_filter = ts_data.copy(filter=[True, True, False, False, False])
    assert ts_data_filter.features == ["open", "high"]


def test_WTSeriesTraining_call(btc_ohlcv):
    ts_data = WTSTraining(9, index_column="date")
    ts_data.add_time_serie(btc_ohlcv.iloc[0:1000], prop_tv=0.2)
    ts_data("test")
    assert np.all(ts_data.ts_data[Partition.TEST.value][Field.INPUT.value][0] == ts_data[0][0])


def test_WTSeriesTraining_add_wtsdata(btc_ohlcv):
    wtsdata = build_forecast_ts_training_dataset(
        btc_ohlcv[["close", "open"]].iloc[0:1000],
        input_width=9,
        prop_tv=0.2,
        prop_v=0.2)
    wtstraining = WTSTraining(input_width=9, target_width=1, features=["open"], index_column="date")
    with pytest.raises(WTSTColumnIndexException):
        wtstraining.add_wtseries(*wtsdata)

    wtstraining = WTSTraining(input_width=9, target_width=1, features=["open"])
    with pytest.raises(WTSTFeaturesException):
        wtstraining.add_wtseries(*wtsdata)

    wtstraining = WTSTraining(input_width=8, target_width=1)
    with pytest.raises(WTSwindowSizeException):
        wtstraining.add_wtseries(*wtsdata)

    wtstraining = WTSTraining(input_width=9, target_width=1)
    wtstraining.add_wtseries(*wtsdata)


def test_WTSeriesTraining_get(btc_ohlcv):
    ts_data = WTSTraining(9, index_column="date")
    ts_data.add_time_serie(btc_ohlcv.iloc[0:1000], prop_tv=0.2)
    inp, targ = ts_data[0:5]
    assert len(inp) == 5 and isinstance(inp[0], pd.DataFrame)
    assert len(targ) == 5 and isinstance(targ[0], pd.DataFrame)
    ts_data.set_selected_features(["close"])
    inp, targ = ts_data[0:5]
    assert list(inp[0].columns) == ["close"]
    assert list(targ[0].columns) == ["close"]
