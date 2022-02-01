from torch import nn, relu, tensor
from ai.super_module import SuperModule


class MLP(SuperModule):
    def __init__(self, features, window_size, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.n_features = features*window_size
        
        self.layer = nn.Linear(self.n_features, 1)
        
    def forward(self, x):
        x = x.view(-1, self.n_features)
        out = relu(self.layer(x))
        return out.view(-1)
    
    # def transform_x(self, x : tensor):
    #     return x.float()
    
    def transform_y(self, y : tensor):
        return y[:,:,0].view(-1)
        