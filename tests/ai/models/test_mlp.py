from ai.models.mlp import MLP
from datatools.wtseries_training import WTSeriesTraining
import pandas as pd
import numpy as np
import os
from torch.optim import SGD
from torch.nn import L1Loss

def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data
data = init_data()

def create_path_mock(*paths):
    return os.path.join(*paths)

def test_mlp(mocker):
    mocker.patch(
        "ai.super_module.create_path",
        side_effect=create_path_mock
    )
    ts_data = WTSeriesTraining(20, column_index="date")
    ts_data.add_time_serie(data.iloc[0:3000])
    module = MLP(features=ts_data.n_features(),window_size=ts_data.input_size)
    module.init(loss = L1Loss(), 
                optimizer=SGD(module.parameters(), lr=0.1))
    module.fit(ts_data, 1, 20)
    