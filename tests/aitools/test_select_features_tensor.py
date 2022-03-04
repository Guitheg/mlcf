from mlcf.datatools.wtst import Partition, WTSTraining
from mlcf.datatools.wtseries_tensor import WTSeriesTensor
import numpy as np


def test_WTSeriesTensor_select(wts_tensor, ts_data_no_date: WTSTraining):

    bool_list_features = np.random.choice([True, False], size=len(ts_data_no_date.features))
    index_list_features = np.where(bool_list_features)[0]
    dataset = WTSeriesTensor(ts_data=ts_data_no_date, partition=Partition.TRAIN)
    inputs, targets = ts_data_no_date(Partition.TRAIN)
    for i in range(len(dataset)):
        inp, tar = dataset[i]
        tensor_input = WTSeriesTensor.select_features(inp, index_list_features)
        tensor_target = WTSeriesTensor.select_features(tar, index_list_features)
        training_input = inputs[i].loc[:, bool_list_features]
        target_input = targets[i].loc[:, bool_list_features]
        assert np.all(np.array(tensor_input) == np.array(training_input.astype(np.float32)))
        assert np.all(np.array(tensor_target) == np.array(target_input.astype(np.float32)))
