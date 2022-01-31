from pathlib import Path
from torch.utils.data import Dataset
from torch import tensor
from dataset.time_series import Time_Series, INPUT, TARGET

class Time_Series_Dataset(Dataset):
    def __init__(self, 
                 partition : str,
                 ts_data : Time_Series = None,  
                 dir_ts_data : Path = None,
                 *args, **kwargs):
        super(Time_Series_Dataset, self).__init__(*args, **kwargs)
        if ts_data is None and dir_ts_data is None:
            raise ValueError("You should to have a Time_Series (ts_data) or "+\
                "at least the dir to a TimeSeries (dir_ts_data).")
        if not dir_ts_data is None and ts_data is None:
            raise NotImplementedError("read a time serie data from a file is not implemented yet")
        if not ts_data is None:
            if not dir_ts_data is None:
                raise Warning("read a time serie data from a file is not implemented yet")
            self.ts_data = ts_data
            
        self.input_data, self.target_data = self._ts_data_to_tensor(partition)
    
    def _ts_data_to_tensor(self, partition):
        input_data = self.ts_data[partition][INPUT]()
        target_data = self.ts_data[partition][TARGET]()
        
        return tensor(input_data), tensor(target_data)
        