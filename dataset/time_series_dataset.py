from pathlib import Path
from typing import Dict, List, Tuple
from freqtrade.data.history.history_utils import load_pair_history
import pandas as pd
from torch.utils.data import Dataset
from datatools import build_forecast_ts_training_dataset, make_commmon_shuffle

class Time_Series_Dataset(object):
    def __init__(self, 
                 input_size : int,
                 label_size : int = 1,
                 selected_columns : List[str] = None,
                 column_index : str = None):
        
        self.TRAIN : str = "train"
        self.VALIDATION : str = "validation"
        self.TEST : str = "test"
        self.INPUT : str = "input"
        self.TARGET : str = "target"
        
        self.raw_data : List[pd.DataFrame] = []
        self.input_size : int = input_size
        self.label_size : int = label_size
        self.columns : List[str] = selected_columns
        self.column_index : str = column_index
        
        self.train_data : Dict[List[pd.DataFrame], List[pd.DataFrame]] = {self.INPUT : [], 
                                                                          self.TARGET : []}
        
        self.val_data : Dict[List[pd.DataFrame], List[pd.DataFrame]] = {self.INPUT : [], 
                                                                        self.TARGET : []}
        
        self.test_data : Dict[List[pd.DataFrame], List[pd.DataFrame]] = {self.INPUT : [], 
                                                                        self.TARGET : []}
        
        self.ts_data : Dict = {self.TRAIN : self.train_data,
                               self.VALIDATION : self.val_data,
                               self.TEST : self.test_data}
    
    def __getitem__(self, index, partition : str = ""):
        pass    
    
    def _add_ts_data(self, 
                     input_ts_data : List[pd.DataFrame],
                     target_ts_data : List[pd.DataFrame],
                     partition : str,
                     do_shuffle : bool = False,
                     ignore_empty_dataframe : bool = False):
        
        if len(input_ts_data) != 0 and len(input_ts_data[0]) != 0:
            self.ts_data[partition][self.INPUT].extend(input_ts_data)
            self.ts_data[partition][self.TARGET].extend(target_ts_data)
            if do_shuffle:
                (input_data_shuffle, target_data_shuffle) = make_commmon_shuffle(
                    self.ts_data[partition][self.INPUT],
                    self.ts_data[partition][self.TARGET]) 
                self.ts_data[partition][self.INPUT] = input_data_shuffle
                self.ts_data[partition][self.TARGET] = target_data_shuffle    
        else:
            if not ignore_empty_dataframe:
                raise Exception("The data is empty")
    
    def add_time_serie(self, dataframe : pd.DataFrame, 
                       test_val_prop : float = 0.2,
                       val_prop : float = 0.3,
                       do_shuffle : bool = False,
                       n_interval : int = 1,
                       offset : int = 0,
                       step_window : int = 1,):
        data = dataframe.copy()
        data.set_index(self.column_index, inplace=True)
        self.raw_data.append(data)
        
        training_dataset : Tuple = build_forecast_ts_training_dataset(data[self.columns], 
                                                              input_width=self.input_size,
                                                              label_width=self.label_size,
                                                              offset=offset,
                                                              step=step_window,
                                                              n_interval=n_interval,
                                                              test_val_prop=test_val_prop,
                                                              val_prop=val_prop,
                                                              do_shuffle=do_shuffle)

        self._add_ts_data(input_ts_data=training_dataset[0],
                          target_ts_data=training_dataset[1],
                          partition=self.TRAIN,
                          do_shuffle=do_shuffle,
                          ignore_empty_dataframe=True)
        
        self._add_ts_data(input_ts_data=training_dataset[2],
                          target_ts_data=training_dataset[3],
                          partition=self.VALIDATION,
                          do_shuffle=do_shuffle,
                          ignore_empty_dataframe=True)
        
        self._add_ts_data(input_ts_data=training_dataset[4],
                          target_ts_data=training_dataset[5],
                          partition=self.TEST,
                          do_shuffle=do_shuffle,
                          ignore_empty_dataframe=True)
        
    def x_train(self, index : int = None):
        if index is None:
            return self.train_data[0]
        return self.train_data[0][index]
    
    def y_train(self, index : int = None):
        if index is None:
            return self.train_data[1]
        return self.train_data[1][index]
    
    def x_val(self, index : int = None):
        if index is None:
            return self.val_data[0]
        return self.val_data[0][index]
        
    def y_val(self, index : int = None):
        if index is None:
            return self.val_data[1]
        return self.val_data[1][index]
    
    def x_test(self, index : int = None):
        if index is None:
            return self.test_data[0]
        return self.test_data[0][index]
    
    def y_test(self, index : int = None):
        if index is None:
            return self.test_data[1]
        return self.test_data[1][index]

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
    x,y,val,valy,test,testy = build_forecast_ts_training_dataset(dataframe, 
                                                                  n_interval=2,
                                                                  input_width=5,
                                                                  label_width=3,
                                                                  test_val_prop=0.1,
                                                                  val_prop = 0.0,
                                                                  step=1,
                                                                  do_shuffle=False)

    ts_dataset = Time_Series_Dataset(input_size=20, column_index="date", selected_columns=["close"])
    ts_dataset.add_time_serie(pair_history, test_val_prop=0)
    ts_dataset.add_time_serie(pair_history2, test_val_prop=0)
    ts_dataset.add_time_serie(pair_history3, test_val_prop=1)
    
    print(pair_history3.iloc[0:20])
    print(ts_dataset.x_test(0))
if __name__ == "__main__":
    main()