import pandas as pd
import numpy as np
from torch.optim import SGD
from torch.nn import L1Loss

from ctbt.aitools.metrics import L2
from ctbt.datatools.wtseries_training import WTSeriesTraining, TEST
from ctbt.datatools.wtseries_tensor import WTSeriesTensor


from torch import nn, tensor, sigmoid

### CTBT modules ###
from ctbt.aitools.super_module import SuperModule

class MLP(SuperModule):
    def __init__(self, features, window_width, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.n_features = features*window_width
        
        self.layer = nn.Linear(self.n_features, 1)
        
    def forward(self, x):
        out = sigmoid(self.layer(x))
        return out.view(-1)
    
    def transform_x(self, x : tensor):
        x = x.view(-1, self.n_features)
        mean = x.mean(axis=1).view(-1, 1).expand(len(x), self.n_features)
        std = x.std(axis=1).view(-1, 1).expand(len(x), self.n_features)
        return (x - mean) / std
    
    def transform_y(self, y : tensor):
        return y[:,:,0].view(-1)

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