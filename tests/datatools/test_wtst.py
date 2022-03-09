import pytest
from mlcf.datatools.utils import build_forecast_ts_training_dataset
from mlcf.datatools.wtseries import WTSwindowSizeException
from mlcf.datatools.wtst import (
    Partition,
    WTSTColumnIndexException,
    WTSTFeaturesException,
    WTSTraining)


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

    assert ts_data(Partition.TRAIN) == ts_data("train")


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
