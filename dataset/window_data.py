from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from freqtrade.data.history.history_utils import load_pair_history
import pandas as pd
from torch.utils.data import Dataset


from typing import List, Tuple
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_data(dataframe : pd.DataFrame, 
                window_size : int,
                window_step : int = 1) -> List[pd.DataFrame]:
        """Window the data given a {window_size} and a {window_step}.

        Args:
            dataframe (pd.DataFrame): The dataframe we want to window
            window_size (int): the windows size
            window_step (int, optional): the window_step between each window. Defaults to 1.

        Returns:
            List[pd.DataFrame]: The list of asscreated windows of size {window_size} and 
            selected every window_step {window_step}
        """
        data = dataframe.copy()
        if len(data) == 0:
            return [pd.DataFrame(columns=data.columns)]
        if len(data) < window_size:
            raise Warning("The length of data is smaller than the window size (return Empty DataFrame)")
            return [pd.DataFrame(columns=data.columns)]
        n_windows = ((len(data.index)-window_size) // window_step) + 1
        n_columns = len(data.columns)
        
        # Slid window on all data
        windowed_data : np.ndarray = sliding_window_view(data, 
                                            window_shape=(window_size, len(data.columns)))

        # Take data every window_step
        windowed_data = windowed_data[::window_step]
        
        # Reshape windowed data
        windowed_data_shape : Tuple[int, int, int]= (n_windows, window_size, n_columns)
        windowed_data = np.reshape(windowed_data, newshape = windowed_data_shape)
        
        # Make list of dataframe
        list_data : List[pd.DataFrame] = []
        for idx in range(n_windows):
            list_data.append(
                pd.DataFrame(windowed_data[idx], 
                            index = data.index[idx*window_step : (idx*window_step)+window_size],
                            columns = data.columns)
                )
        return list_data

class Window_Data(object):
    def __init__(self, 
                 window_size : int, 
                 data : pd.DataFrame = None, 
                 window_step : int = 1,
                 *args, **kwargs):
        super(Window_Data, self).__init__(*args, **kwargs)
        
        self.raw_data : pd.DataFrame = data 
        self._window_size : int = window_size
        self.features_has_been_set = False
        if not data is None:
            self.win_data : List[pd.DataFrame] = window_data(self.raw_data, 
                                                            self.window_size(), 
                                                            window_step)
            self.set_features(self.raw_data.columns)
        else:
            self.win_data : List = []
        if not self.is_empty() and len(self.win_data[0].index) != self.window_size():
            raise Exception("The window size is suppose to be equal "+\
                            "to the number of row if it is not empty")
            
    def __len__(self):
        return len(self.win_data)
    
    def set_features(self, columns : List[str]):
        self.features = columns
        self.features_has_been_set = True
        
    def get_features(self):
        return self.features
    
    def window_size(self):
        return self._window_size
    
    def n_features(self):
        if self.features_has_been_set:
            return len(self.features)
        return 0
    
    def shape(self):
        return (len(self), self.window_size(), self.n_features())
    
    def is_empty(self):
        return len(self) == 0 or len(self.win_data[0].index) == 0
    
    def __str__(self):
        return f"window n°1:\n{str(self[0])}\n" +\
               f"window n°2:\n{str(self[1])}\n" +\
               f"Number of windows : {len(self)}, " +\
               f"window's size : {self.window_size()}, " +\
               f"number of features : {self.n_features()}"
    
    def __getitem__(self, index : int):
        return self.win_data[index]
    
    def __call__(self, index : int = None):
        if not index is None:
            return self[index]
        return self.win_data
    
    def add_data(self, data : pd.DataFrame, 
                 window_step : int = 1, 
                 ignore_data_empty : bool = False):
        if len(data) != 0:
            if self.features_has_been_set:
                if len(data.columns) != self.n_features():
                    raise Exception ("The number of features is supposed to be the same")
            data_to_add : List[pd.DataFrame] = window_data(data, self.window_size(), window_step)
            self.win_data.extend(data_to_add)
            if not self.features_has_been_set:
                self.set_features(data.columns)
        else:
            if not ignore_data_empty:
                raise Exception("Data is empty")
            
    def add_one_window(self, window : pd.DataFrame, ignore_data_empty : bool = False):
        if len(window) != 0:
            if self.features_has_been_set:
                if len(window.columns) != self.n_features():
                    raise Exception ("The number of features is supposed to be the same")
            if len(window.index) != self.window_size():
                raise Exception ("The window size is supposed to be the same")
            data = window.copy()
            self.win_data.append(data)
            if not self.features_has_been_set:
                self.set_features(data.columns)
        else:
            if not ignore_data_empty:
                raise Exception("Data is empty")
                
    def merge_window_data(self,
                          window_data : Window_Data, 
                          ignore_data_empty : bool = False):
        if not window_data.is_empty():
            if self.features_has_been_set:
                if window_data.n_features() != self.n_features():
                    raise Exception ("The number of features is supposed to be the same")
            if window_data.window_size() != self.window_size():
                raise Exception ("The window size is supposed to be the same")
            self.win_data.extend(window_data())
            if not self.features_has_been_set:
                self.set_features(window_data.get_features())
        else:
            if not ignore_data_empty:
                raise Exception("Data is empty")
            
def main():
    path = Path("./user_data/data/binance")
    pair = "BTC/BUSD"
    timeframe = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"] 
    col = ["all"]
    t = timeframe[4]
    
    pair_history = load_pair_history(pair, t, path)
    pair_history3 = load_pair_history("ETH/BUSD", t, path)
    pair_history2 = load_pair_history(pair, "1d", path)
    # pair_history.set_index("date", inplace=True)
    dataframe = pair_history[["close", "open", "volume"]]
   


    win = Window_Data(dataframe, 5)
    win2 = Window_Data(dataframe, 5, 3)
    print(win.shape())
    print(win2.shape())
    win.merge_window_data(win2)
    print(win.shape())

if __name__ == "__main__":
    main()