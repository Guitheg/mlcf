import pytest
import pandas as pd
from mlcf.datatools.wtseries_training import Partition, Field, WTSeriesTraining
from mlcf.datatools.wtseries_tensor import WTSeriesTensor
import numpy as np


def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] = pd.to_datetime(data["date"], unit="ms")
    return data


data = init_data()


def test_WTSeriesTensor():
    ts_data = WTSeriesTraining(9, index_column="date")
    with pytest.raises(ValueError):
        dataset = WTSeriesTensor(Partition.TRAIN, ts_data=ts_data)
    ts_data.add_time_serie(data.iloc[0:3000])
    assert ts_data.index_column == "date"
    dataset = WTSeriesTensor(Partition.TRAIN, ts_data=ts_data)
    assert ts_data.len(Partition.TRAIN) == 2391
    assert ts_data.len(Partition.VALIDATION) == 171
    assert ts_data.len(Partition.TEST) == 411
    for i in range(len(dataset)):
        inp, tar = dataset[i]
        assert np.all(np.array(inp) == np.array(ts_data(Partition.TRAIN,
                                                        Field.INPUT)[i].astype(np.float32)))
        assert np.all(np.array(tar) == np.array(ts_data(Partition.TRAIN,
                                                        Field.TARGET)[i].astype(np.float32)))
    assert dataset.input_data.size() == ts_data(Partition.TRAIN, Field.INPUT).size()
