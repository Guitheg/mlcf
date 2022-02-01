import pytest
import pandas as pd
from dataset.time_series import TARGET, Time_Series, TRAIN, INPUT
from dataset.time_series_dataset import Time_Series_Dataset
from torch import tensor
import numpy as np

def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data
data = init_data()

def test_Time_Series_Dataset():
    ts_data = Time_Series(9, column_index="date")
    ts_data.add_time_serie(data.iloc[0:3000])
    dataset = Time_Series_Dataset(TRAIN, ts_data=ts_data)
    for i in range(len(dataset)):
        inp, tar = dataset[i]
        assert np.all(np.array(inp) == np.array(ts_data(TRAIN, INPUT)[i]))
        assert np.all(np.array(tar) == np.array(ts_data(TRAIN, TARGET)[i]))
    assert dataset.input_data.size() == ts_data(TRAIN, INPUT).shape()