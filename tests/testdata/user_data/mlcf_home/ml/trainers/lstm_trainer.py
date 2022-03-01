import torch

from ritl import add
add(__file__, "..")

from models.lstm import LSTM


def train(
    project,
    training_name,
    wtst_data,
    *args,
    **kwargs,
):
    list_columns = wtst_data.features
    lstm = LSTM(30, list_columns)
    lstm.init(torch.nn.L1Loss(),
              torch.optim.SGD(lstm.parameters(), lr=0.01, momentum=0.9),
              training_name=training_name,
              project=project)
    lstm.fit(wtst_data, 5, 20, False, True, True, False, True)
