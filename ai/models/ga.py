from pandas import DataFrame
import pandas as pd
from torch import nn, relu, tensor, sigmoid, sum
from ai.super_module import SuperModule
from geneticalgorithm2 import geneticalgorithm2 as ga
from datatools.wtseries_training import WTSeriesTraining, Partition as P
from torchvision import transforms
from ai.metrics import L2
from torch.optim import SGD
from torch.nn import L1Loss

class GA(SuperModule):
    def __init__(self, data:DataFrame, nn_model:SuperModule, loss_function, nn_model_max_loss=10000, test_data = True, test_nn_model_dim = True, *args, **kwargs):
        super(GA, self).__init__(*args, **kwargs)

        if test_data:
            for collum in data.columns:
                pd.to_numeric(data[collum], errors='raise').notnull().all() #maybe set raise to coerce (replace wrong value by NaN)

        self.data = data
        self.n_dim = data.shape[1]
        
        self.nn_model = nn_model

        self.loss_function = loss_function

        self.nn_model_max_loss = nn_model_max_loss #setable at init or after

        def set_nn_model_max_loss(new_nn_max_loss):
            self.nn_model_max_loss = new_nn_max_loss
        
        def run_nn_model(input_size=20, n_epoch = 1, batch_size= 5,learning_rate = 0.1, loss=L1Loss(),metrics=[L2]):
            # print(data.columns)
            ts_data = WTSeriesTraining(input_size)
            ts_data.add_time_serie(self.data)
            module = nn_model(features=self.ndim, window_width=input_size)
            module.init(loss = loss, 
                        optimizer=SGD(module.parameters(), lr=learning_rate),
                        metrics=metrics)
            # module.summary()
            logs, log_eval = module.fit(ts_data, n_epoch, batch_size, talkative=False)
            return logs,log_eval
        
        def loss_function(X):
            if X.sum() == 0.0:
                loss = 1000
                return loss

            selected_data = (data.copy()).loc[:, (X != 0.0)]
            # print(selected_data)
            loss = 0
            
            logs,log_eval = self.run_nn_model()

            return loss

        self.ga_model = ga(function=loss_function, variable_type='bool', dimension=self.n_dim)

        def run():
            self.ga_model.run()

        def get_convergence():
            convergence = self.ga_model.report
            return convergence

        def get_solution():
            solution = self.model.output_dict
            return solution

       

    
