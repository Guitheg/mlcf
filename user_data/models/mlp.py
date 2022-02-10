
from torch import nn, tensor, sigmoid

### CG-RBI modules ###
from CGrbi.ai.super_module import SuperModule

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
        