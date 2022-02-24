from mlcf.datatools.wtst import Partition, WTSTraining


def test_WTSeriesTraining(btc_ohlcv):
    ts_data = WTSTraining(9)
    ts_data.add_time_serie(btc_ohlcv.iloc[0:1000], prop_tv=0.2)
    assert len(ts_data) == 800 - 10 + 1


def test_io_WTSeriesTraining(btc_ohlcv):
    ts_data = WTSTraining(9, index_column='date')
    ts_data.add_time_serie(btc_ohlcv.iloc[0:1000], prop_tv=0.2)


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
