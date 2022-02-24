from mlcf.datatools.wtst import Partition, WTSTraining
from mlcf.datatools.wtseries_tensor import WTSeriesTensor
import numpy as np


def test_WTSeriesTensor(btc_ohlcv):
    ts_data = WTSTraining(9, index_column="date")
    ts_data.add_time_serie(btc_ohlcv.iloc[0:3000])
    assert ts_data.index_column == "date"
    dataset = WTSeriesTensor(ts_data=ts_data, partition=Partition.TRAIN)
    assert ts_data.len(Partition.TRAIN) == 2391
    assert ts_data.len(Partition.VALIDATION) == 171
    assert ts_data.len(Partition.TEST) == 411
    for i in range(len(dataset)):
        inp, tar = dataset[i]
        inputs, targets = ts_data(Partition.TRAIN)
        assert np.all(np.array(inp) == np.array(inputs[i].astype(np.float32)))
        assert np.all(np.array(tar) == np.array(targets[i].astype(np.float32)))
    inputs, targets = ts_data(Partition.TRAIN)
    assert dataset.x_size() == inputs.size()[1:3]
    assert dataset.y_size() == targets.size()[1:3]
