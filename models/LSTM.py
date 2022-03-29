
from torch import nn, relu, tensor, sigmoid, sum
import torch
from mlcf.aitools.super_module import SuperModule
from mlcf.aitools.metrics import L2
from torch.optim import SGD
from torch.nn import L1Loss
from torchvision import transforms
from mlcf.datatools.wtseries_tensor import WTSeriesTensor, select_features
from mlcf.datatools.wtst import WTSTraining
import numpy as np


class LSTM(SuperModule):

    def __init__(self, PARAMS):
    #     self,  seq_len, bool_list_features: np.array, index_list_target: list,
    #     feature_dim=20, hidden_size=20, num_layers=2,
    #     n_epoch=1, batch_size=5, learning_rate=0.1, loss=L1Loss(), metrics=[L2],
    #     optimizer=None, *args, **kwargs
    # ):

        super(LSTM, self).__init__(*args, **kwargs)

        # self.index_list_features = np.where(bool_list_features)[0]
        # self.index_list_target = index_list_target
        self.feature_dim = len(self.index_list_features)
        self.hidden_size = PARAMS("hidden_size")
        self.num_layers = PARAMS("num_layers")
        # self.n_epoch = n_epoch
        # self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.optimizer = None
        self.seq_len = seq_len

        # nn_model specification here
        self.lstm = nn.LSTM(
            self.feature_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True)
        self.layer1 = nn.Linear(self.hidden_size*self.seq_len, (self.hidden_size*self.seq_len)//2)
        self.layer2 = nn.Linear((self.hidden_size*self.seq_len)//2, 1)
        # input : seq_len, batch_size, feature_dim
        # D = 1 -> monodirectional, 2 if bidirectional
        # h_n : D*num_layer , batch size, hidden size when proj_size <= 0
        # c_n : D∗num_layers, batch size ,hidden size

    def prepare_data(input_width, data):
        ts_data = WTSTraining(input_width)
        ts_data.add_time_serie(data)
        return ts_data
    
    def set_optimizer(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, hidden = self.lstm(x)
        out = out.reshape(-1, self.hidden_size*self.seq_len)
        fc = self.layer1(out)
        x = relu(fc)
        fc = self.layer2(x)
        x = sigmoid(fc)
        return x.view(-1, 1,)

    def transform_x(self, x: tensor):
        new_x = select_features(x, self.index_list_features)
        return new_x

    def transform_y(self, y: tensor):
        new_y = select_features(y, self.index_list_target)
        return new_y

    # def select_features(x: tensor, index_list_features):
    #     return torch.index_select(x, 1, index_list_features)
