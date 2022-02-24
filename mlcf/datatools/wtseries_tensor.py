import numpy as np
from typing import Callable
from torch.utils.data.dataloader import DataLoader
from torch import tensor

# MLCF modules
from mlcf.datatools.wtst import Partition, WTSTraining


class WTSeriesTensor():
    def __init__(self,
                 ts_data: WTSTraining,
                 partition: Partition,
                 transform_x: Callable = None,
                 transform_y: Callable = None,
                 *args, **kwargs):
        """WTSeriesTensor provide an indexer on time series tensor data which is compatible
        to a dataloader use in machine learning traning

        Args:
            partition (str): the part 'train', 'validation' or 'test'
            ts_data (WTSTraining, optional): 'The time series data'. Defaults to None.
            dir_ts_data (Path, optional): 'the dir to the time series file'. Defaults to None.

        Raises:
            ValueError: You should to have a WTSTraining (ts_data) or at least the dir to a
            TimeSeries (dir_ts_data)
            NotImplementedError: read a time serie data from a file is not implemented yet
            Warning: read a time serie data from a file is not implemented yet
        """
        self.partition = partition
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.ts_data = ts_data
        self.ts_data.set_partition(self.partition)

    def x_size(self, *args, **kwargs):
        return self[0][0].size(*args, **kwargs)

    def y_size(self, *args, **kwargs):
        return self[0][1].size(*args, **kwargs)

    def __len__(self):
        return len(self.ts_data)

    def __getitem__(self, index):
        input_data, target_data = self.ts_data[index]
        input_data = tensor(np.array(input_data).astype(np.float32))
        target_data = tensor(np.array(target_data).astype(np.float32))
        if self.transform_x is not None:
            input_data = self.transform_x(input_data)
        if self.transform_y is not None:
            target_data = self.transform_y(target_data)
        return input_data, target_data

    def copy(self):
        return WTSeriesTensor(
            ts_data=self.ts_data,
            partition=self.partition,
            transform_x=self.transform_x,
            transform_y=self.transform_y
        )

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)
