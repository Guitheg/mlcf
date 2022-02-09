import pandas as pd
import numpy as np
from torch.optim import SGD
from torch.nn import L1Loss

from CGrbi.ai.metrics import L2
from CGrbi.ai.models.mlp import MLP
from CGrbi.datatools.wtseries_training import WTSeriesTraining, TEST
from CGrbi.datatools.wtseries_tensor import WTSeriesTensor

def init_data():
    data = np.arange(1000)
    columns = ["value"]
    data = pd.DataFrame(data, columns=columns)
    return data
data = init_data()

def test_mlp(mocker):
    ts_data = WTSeriesTraining(20)
    ts_data.add_time_serie(data)
    module = MLP(features=ts_data.ndim(), window_width=ts_data.input_size)
    module.init(loss = L1Loss(), 
                optimizer=SGD(module.parameters(), lr=0.1),
                metrics=[L2])
    module.summary()
    module.fit(ts_data, 1, 20)
    tensor_data = WTSeriesTensor(TEST, ts_data=ts_data)
    i,l = tensor_data[0]
    y = module.predict(i.view(1,-1))