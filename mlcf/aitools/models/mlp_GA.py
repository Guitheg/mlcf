
from torch import nn, relu, tensor, sigmoid, sum
from mlcf.aitools.super_module import SuperModule
from mlcf.aitools.metrics import L2
from torch.optim import SGD
from torch.nn import L1Loss
from torchvision import transforms
from mlcf.datatools.wtseries_training import WTSeriesTraining, Partition as P


class MLPGA(SuperModule):

    def __init__(
        self, features, window_width,
        input_size=20, n_epoch=1, batch_size=5,
        learning_rate=0.1, loss=L1Loss(), metrics=[L2],
        optimizer=None,
        *args, **kwargs
    ):

        super(MLPGA, self).__init__(*args, **kwargs)

        self.window_width = window_width
        self.n_features = features*window_width
        self.input_size = input_size
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)

        # nn_model specification here
        self.layer = nn.Linear(self.n_features, 1)

    def prepare_data(input_size, data):
        ts_data = WTSeriesTraining(input_size)
        ts_data.add_time_serie(data)
        return ts_data

    def forward(self, x):
        out = sigmoid(self.layer(x))
        return out.view(-1)

    def transform_x(self, x: tensor):
        x = x.view(-1, self.n_features)
        mean = x.mean(axis=1).view(-1, 1).expand(len(x), self.n_features)
        std = x.std(axis=1).view(-1, 1).expand(len(x), self.n_features)
        return (x - mean) / std

    def transform_y(self, y: tensor):
        return y[:, :, 0].view(-1)
