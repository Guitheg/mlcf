from pathlib import Path
from typing import List, Tuple
from freqtrade.data.history.history_utils import load_pair_history
import pandas as pd
from torch.utils.data import Dataset
from datatools import build_forecast_ts_training_dataset

class Time_Series_Dataset(object):
    def __init__(self, 
                 input_size : int,
                 label_size : int = 1,
                 selected_columns : List[str] = None,
                 column_index : str = None):
        
        self.data : List[pd.DataFrame] = []
        self.input_size : int = input_size
        self.label_size : int = label_size
        self.columns : List[str] = selected_columns
        self.column_index : str = column_index
         
        self.train_data : Tuple[List[pd.DataFrame], List[pd.DataFrame]] = ([], [])
        self.val_data : Tuple[List[pd.DataFrame], List[pd.DataFrame]] = ([], [])
        self.test_data : Tuple[List[pd.DataFrame], List[pd.DataFrame]] = ([], [])
        
    def add_time_serie(self, dataframe : pd.DataFrame, 
                       test_val_prop : float = 0.2,
                       val_prop : float = 0.3,
                       do_shuffle : bool = False,
                       n_interval : int = 1,
                       offset : int = 0,
                       step_window : int = 1,):
        data = dataframe.copy()
        data.set_index(self.column_index, inplace=True)
        self.data.append(data)
        
        training_dataset = build_forecast_ts_training_dataset(data[self.columns], 
                                                              input_width=self.input_size,
                                                              label_width=self.label_size,
                                                              offset=offset,
                                                              step=step_window,
                                                              n_interval=n_interval,
                                                              test_val_prop=test_val_prop,
                                                              val_prop=val_prop,
                                                              do_shuffle=do_shuffle)
        self.train_data = (self.train_data[0].extend(training_dataset[0]), 
                           self.train_data[1].extend(training_dataset[1]))
        
        self.val_data = (self.val_data[0].extend(training_dataset[2]), 
                         self.val_data[1].extend(training_dataset[3]))
        
        self.test_data = (self.test_data[0].extend(training_dataset[4]), 
                          self.test_data[1].extend(training_dataset[5]))
        

def main():
    path = Path("./user_data/data/binance")
    pair = "BTC/BUSD"
    timeframe = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"] 
    col = ["all"]
    t = timeframe[4]
    
    pair_history = load_pair_history(pair, t, path)
    # pair_history.set_index("date", inplace=True)
    dataframe = pair_history[["close", "open", "volume"]]
    x,y,val,valy,test,testy = build_forecast_ts_training_dataset(dataframe, 
                                                                  n_interval=2,
                                                                  input_width=5,
                                                                  label_width=3,
                                                                  test_val_prop=0.1,
                                                                  val_prop = 0.0,
                                                                  step=1,
                                                                  do_shuffle=False)

    ts_dataset = Time_Series_Dataset(input_size=20, column_index="date", selected_columns=["close"])
    ts_dataset.add_time_serie(pair_history)

    print(ts_dataset.train_data)
    
if __name__ == "__main__":
    main()