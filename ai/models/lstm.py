
from torch import nn, relu, tensor, sigmoid, sum
from ai.super_module import SuperModule
from torchvision import transforms

class LSTM(SuperModule):
    def __init__(self, features, window_width, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        self.n_features = features
        self.seq_len = window_width
        
        self.hu_1 = 128
        self.lstm_1 = nn.LSTM(input_size=self.n_features, 
                              hidden_size=self.hu_1, 
                              dropout=0.2, 
                              num_layers=4,
                              batch_first=True)
        self.linear_1 = nn.Linear(self.hu_1*self.seq_len, 128)
        self.linear_2 = nn.Linear(128, 32)
        self.linear_3 = nn.Linear(32, 1)
        
        
    def forward(self, x):
        out, _ = self.lstm_1(x)
        out = out.reshape(-1, self.hu_1*self.seq_len)
        out = relu(self.linear_1(out))
        out = relu(self.linear_2(out))
        out = self.linear_3(out)
        return out
    
    def transform_x(self, x : tensor):
        return x
    
    def transform_y(self, y : tensor):
        return y[:,:,3].view(-1, 1)
        