
from torch import Tensor, nn, relu
from mlcf.aitools.super_module import SuperModule


class LSTM(SuperModule):
    def __init__(self, window_width, list_columns, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        self.n_features = len(list_columns)
        self.seq_len = window_width
        self.list_columns = list_columns
        self.hu_1 = 32
        self.lstm_1 = nn.LSTM(input_size=self.n_features,
                              hidden_size=self.hu_1,
                              dropout=0.2,
                              num_layers=2,
                              batch_first=True)
        self.linear_1 = nn.Linear(self.hu_1*self.seq_len, 32)
        self.linear_2 = nn.Linear(32, 16)
        self.linear_3 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm_1(x)
        out = out.reshape(-1, self.hu_1*self.seq_len)
        out = relu(self.linear_1(out))
        out = relu(self.linear_2(out))
        out = self.linear_3(out)
        return out

    def transform_x(self, x: Tensor):
        return x

    def transform_y(self, y: Tensor):
        return y[:, 3]
