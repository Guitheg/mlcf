import pytest
import pandas as pd
from datatools.wtseries_training import TARGET, WTSeriesTraining, TRAIN, INPUT
from datatools.wtseries_tensor import WTSeriesTensor
from torch import tensor
import numpy as np

def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data
data = init_data()

def test_WTSeriesTensor():
    ts_data = WTSeriesTraining(9, column_index="date")
    with pytest.raises(ValueError):
        dataset = WTSeriesTensor(TRAIN, ts_data=ts_data)
    ts_data.add_time_serie(data.iloc[0:3000])
    dataset = WTSeriesTensor(TRAIN, ts_data=ts_data)
    assert "Input size: 9, Target size: 1, "+\
        "Index name: 'date'\nData :\nLength Train: 2391, "+\
            "Length Validation: 171, Length Test: 411" == ts_data.__str__()
    for i in range(len(dataset)):
        inp, tar = dataset[i]
        assert np.all(np.array(inp) == np.array(ts_data(TRAIN, INPUT)[i].astype(np.float32)))
        assert np.all(np.array(tar) == np.array(ts_data(TRAIN, TARGET)[i].astype(np.float32)))
    assert dataset.input_data.size() == ts_data(TRAIN, INPUT).shape()