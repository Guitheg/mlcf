import pandas as pd
from mlcf.datatools.wtst import Partition, WTSTraining


def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] = pd.to_datetime(data["date"], unit="ms")
    return data


data = init_data()


def test_WTSeriesTraining():
    ts_data = WTSTraining(9)
    ts_data.add_time_serie(data.iloc[0:1000], prop_tv=0.2)
    assert len(ts_data) == 800 - 10 + 1


def test_io_WTSeriesTraining():
    ts_data = WTSTraining(9, index_column='date')
    ts_data.add_time_serie(data.iloc[0:1000], prop_tv=0.2)


def test_WTSeriesTraining_copy():
    ts_data = WTSTraining(9, index_column="date")
    ts_data.add_time_serie(data.iloc[0:1000], prop_tv=0.2)

    ts_data_copy = ts_data.copy()
    assert ts_data_copy.input_size == ts_data.input_size
    assert ts_data_copy.features == ts_data.features
    assert ts_data_copy.index_column == ts_data.index_column

    ts_data_filter = ts_data.copy(filter=[True, True, False, False, False])
    assert ts_data_filter.features == ["open", "high"]


def test_WTSeriesTraining_call():
    ts_data = WTSTraining(9, index_column="date")
    ts_data.add_time_serie(data.iloc[0:1000], prop_tv=0.2)

    assert ts_data(Partition.TRAIN) == ts_data("train")
