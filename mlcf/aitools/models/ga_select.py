from pandas import DataFrame
import pandas as pd
from mlcf.aitools.super_module import SuperModule
from geneticalgorithm2 import geneticalgorithm2 as ga
from mlcf.datatools import wtseries_training
from mlcf.datatools.wtseries_training import WTSeriesTraining, Partition as P
from torchvision import transforms
from mlcf.aitools.metrics import L2
from torch.optim import SGD
from torch.nn import L1Loss


glb_raw_data = pd.DataFrame()
glb_wts_training = None
glb_nn_model = None
glb_nn_model_max_loss = None


def run_nn_model(
    ts_data: WTSeriesTraining,
    ndim=glb_raw_data.shape[1],
    input_size=20,
    n_epoch=1,
    batch_size=5,
    learning_rate=0.1,
    loss=L1Loss(),
    metrics=[L2]
):
    # print(data.columns)
    module = glb_nn_model(features=ndim, window_width=input_size)
    module.init(
        loss=loss,
        optimizer=SGD(module.parameters(), lr=learning_rate),
        metrics=metrics
    )
    # module.summary()
    logs, log_eval = module.fit(ts_data, n_epoch, batch_size, talkative=False)
    return logs, log_eval


class GA_SELECT(SuperModule):
    def __init__(
        self,
        data: DataFrame,
        nn_model: SuperModule,
        function=None,
        nn_model_max_loss=10000,
        dimension=3,
        variable_boundaries=None,
        variable_type_mixed=False,
        function_timeout=360000,  # default is 100 hour
        test_data=True,  # test if data are well formated
        test_nn_model_dim=True,
        max_num_iteration=None,  # algo param starts here
        population_size=100,
        mutation_probability=0.1,
        elit_ratio=0.01,
        crossover_probability=0.5,
        parents_portion=0.3,
        crossover_type='uniform',
        mutation_type='uniform_by_center',
        selection_type='roulette',
        max_iteration_without_improv=100,
        *args,
        **kwargs
    ):
        super(GA_SELECT, self).__init__(*args, **kwargs)

        if test_data:
            for collum in data.columns:
                pd.to_numeric(data[collum], errors='raise').notnull().all()  # maybe set 'raise' to
                # 'coerce' (replace wrong value by NaN)

        global glb_raw_data
        glb_raw_data = data
        self.data = data
        self.ndim = data.shape[1]
        global glb_nn_model
        self.nn_model = nn_model
        glb_nn_model = nn_model
        global glb_nn_model_max_loss
        self.nn_model_max_loss = nn_model_max_loss  # setable at init or after
        glb_nn_model_max_loss = nn_model_max_loss
        self.function_timeout = function_timeout

        self.max_num_iteration = max_num_iteration,  # algo param starts here
        self.population_size = population_size,
        self.mutation_probability = mutation_probability,
        self.elit_ratio = elit_ratio,
        self.crossover_probability = crossover_probability,
        self.parents_portion = parents_portion,
        self.crossover_type = crossover_type,
        self.mutation_type = mutation_type,
        self.selection_type = selection_type,
        self.max_iteration_without_improv = max_iteration_without_improv

        # self.loss_function = self.loss_function #setable after init to customize

        print("population_size", population_size)

    def loss_function(X):
        global raw_data
        if X.sum() == 0.0:
            loss = 1000
            return loss

        selected_data = (raw_data.copy()).loc[:, (X != 0.0)]
        print(selected_data)
        loss = 0
        logs, log_eval = run_nn_model()
        return loss

    def create_algorithm_parameters(
        self,
        max_num_iteration=None,
        population_size=100,
        mutation_probability=0.1,
        elit_ratio=0.01,
        crossover_probability=0.5,
        parents_portion=0.3,
        crossover_type='uniform',
        mutation_type='uniform_by_center',
        selection_type='roulette',
        max_iteration_without_improv=None
    ):
        alg_param = {
            'max_num_iteration': max_num_iteration,
            'population_size': population_size,
            'mutation_probability': mutation_probability,
            'elit_ratio': elit_ratio,
            'crossover_probability': crossover_probability,
            'parents_portion': parents_portion,
            'crossover_type': crossover_type,
            'mutation_type': mutation_type,
            'selection_type': selection_type,
            'max_iteration_without_improv': max_iteration_without_improv
        }
        return alg_param

    def create_ga_model(self):
        alg_param = self.create_algorithm_parameters(
            self.max_num_iteration,
            self.population_size,
            self.mutation_probability,
            self.elit_ratio,
            self.crossover_probability,
            self.parents_portion,
            self.crossover_type,
            self.mutation_type,
            self.selection_type,
            self.max_iteration_without_improv
        )

        ga_model = ga(
            function=self.loss_function, variable_type='bool',
            dimension=self.ndim,
            function_timeout=self.function_timeout,
            algorithm_parameters=alg_param
        )
        return ga_model

    def create_ts_data(self, input_size, data=None):
        if data is None:
            data = self.data
        ts_data = WTSeriesTraining(input_size)
        ts_data.add_time_serie(data)
        return ts_data

    def set_nn_model_max_loss(self, new_nn_max_loss):
        self.nn_model_max_loss = new_nn_max_loss

    def set_loss_function(self, f):
        self.loss_function = f

    def run_ga(self):
        self.ga_model.run()

    def get_convergence(self):
        convergence = self.ga_model.report
        return convergence

    def get_solution(self):
        solution = self.model.output_dict
        return solution
