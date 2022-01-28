import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga # for creating and running optimization model
# from geneticalgorithm2 import Generation, AlgorithmParams, MiddleCallbackData # classes for comfortable parameters setting and getting
# from geneticalgorithm2 import Crossover, Mutations, Selection # classes for specific mutation and crossover behavior
# from geneticalgorithm2 import Population_initializer # for creating better start population
# from geneticalgorithm2 import np_lru_cache # for cache function (if u want)
# from geneticalgorithm2 import plot_pop_scores # for plotting population scores, if u want
# from geneticalgorithm2 import Callbacks # simple callbacks
# from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks # middle callbacks

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def f(X):
    loss = 0
    for i in range(0,len(X),2):
        loss += X[i]
    for i in range(0,len(X),3):
        loss -= X[i]

    return loss
     
varbound = np.array([[0,10]]*10)

model = ga(function=f, variable_type='bool', dimension=10)

model.run()

convergence = model.report

solution = model.output_dict