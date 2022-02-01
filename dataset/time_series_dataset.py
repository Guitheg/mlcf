from pathlib import Path
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch import tensor
from dataset.time_series import Time_Series, INPUT, TARGET
import numpy as np

class Time_Series_Dataset(TensorDataset):
    def __init__(self, 
                 partition : str,
                 ts_data : Time_Series = None,  
                 dir_ts_data : Path = None,
                 *args, **kwargs):
        """Time_Series_Dataset provide an indexer on time series tensor data which is compatible
        to a dataloader use in machine learning traning

        Args:
            partition (str): the part 'train', 'validation' or 'test'
            ts_data (Time_Series, optional): 'The time series data'. Defaults to None.
            dir_ts_data (Path, optional): 'the dir to the time series file'. Defaults to None.

        Raises:
            ValueError: You should to have a Time_Series (ts_data) or at least the dir to a 
            TimeSeries (dir_ts_data)
            NotImplementedError: read a time serie data from a file is not implemented yet
            Warning: read a time serie data from a file is not implemented yet
        """
        if ts_data is None and dir_ts_data is None:
            raise ValueError("You should to have a Time_Series (ts_data) or "+\
                "at least the dir to a TimeSeries (dir_ts_data).")
        if not dir_ts_data is None and ts_data is None:
            raise NotImplementedError("read a time serie data from a file is not implemented yet")
        if not ts_data is None:
            if not dir_ts_data is None:
                raise Warning("read a time serie data from a file is not implemented yet")
            self.ts_data : Time_Series = ts_data
        
        self.input_data, self.target_data = self._ts_data_to_tensor(partition)
        super(Time_Series_Dataset, self).__init__(*[self.input_data, self.target_data], 
                                                  *args, **kwargs)
    
    def get_input_size(self):
        return self.ts_data.get_input_size()
    
    def get_target_size(self):
        return self.ts_data.get_target_size()
    
    def get_n_features(self):
        return self.ts_data.n_features()
    
    def _ts_data_to_tensor(self, partition):
        input_data = self.ts_data(partition, INPUT)
        target_data = self.ts_data(partition, TARGET)
        return tensor(np.array(input_data)), tensor(np.array(target_data))
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)