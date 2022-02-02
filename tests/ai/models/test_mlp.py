from ai.models.mlp import MLP
from datatools.wtseries_training import WTSeriesTraining
from datatools.wtseries_tensor import WTSeriesTensor
import pandas as pd
import numpy as np
import os
from ai.metrics import L2
from torch.optim import SGD
from torch.nn import L1Loss

def init_data():
    data = np.arange(10000)
    columns = ["value"]
    data = pd.DataFrame(data, columns=columns)
    return data
data = init_data()

def create_path_mock(*paths):
    return os.path.join(*paths)

def test_mlp(mocker):
    mocker.patch(
        "ai.super_module.create_path",
        side_effect=create_path_mock
    )
    mocker.patch(
        "ai.super_module.SuperModule.save",
        return_value=None
    )
    ts_data = WTSeriesTraining(20)
    ts_data.add_time_serie(data)
    module = MLP(features=ts_data.n_features(), window_size=ts_data.input_size)
    module.init(loss = L1Loss(), 
                optimizer=SGD(module.parameters(), lr=0.1),
                metrics=[L2])
    module.summary()
    module.fit(ts_data, 1, 20)
    tensor_data = WTSeriesTensor("test", ts_data=ts_data)
    i,l = tensor_data[0]
    y = module.predict(i.view(1,-1))