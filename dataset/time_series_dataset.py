from typing import Dict, List, Tuple
import pandas as pd
from dataset.datatools import build_forecast_ts_training_dataset, make_commmon_shuffle
from dataset.window_data import Window_Data

class Time_Series_Dataset(object):
    def __init__(self, 
                 input_size : int,
                 target_size : int = 1,
                 column_index : str = None,
                 *args, **kwargs):
        super(Time_Series_Dataset, self).__init__(*args, **kwargs)
        
        self.TRAIN : str = "train"
        self.VALIDATION : str = "validation"
        self.TEST : str = "test"
        self.INPUT : str = "input"
        self.TARGET : str = "target"
        
        self.raw_data : List[pd.DataFrame] = []
        self.input_size : int = input_size
        self.target_size : int = target_size
        self.column_index : str = column_index
        
        self.train_data = {self.INPUT : Window_Data(self.input_size), 
                           self.TARGET : Window_Data(self.target_size)}
        
        self.val_data = {self.INPUT : Window_Data(self.input_size), 
                         self.TARGET : Window_Data(self.target_size)}
        
        self.test_data = {self.INPUT : Window_Data(self.input_size), 
                          self.TARGET : Window_Data(self.target_size)}
        
        self.ts_data : Dict = {self.TRAIN : self.train_data,
                               self.VALIDATION : self.val_data,
                               self.TEST : self.test_data}

    def _add_ts_data(self, 
                     input_ts_data : Window_Data,
                     target_ts_data : Window_Data,
                     partition : str,
                     do_shuffle : bool = False):
        
        self.ts_data[partition][self.INPUT].merge_window_data(input_ts_data, 
                                                              ignore_data_empty=True)
        self.ts_data[partition][self.TARGET].merge_window_data(target_ts_data,
                                                               ignore_data_empty=True)
        if do_shuffle:
            (input_data_shuffle, target_data_shuffle) = make_commmon_shuffle(
                self.ts_data[partition][self.INPUT],
                self.ts_data[partition][self.TARGET]) 
            self.ts_data[partition][self.INPUT] = input_data_shuffle
            self.ts_data[partition][self.TARGET] = target_data_shuffle    
    
    def add_time_serie(self, dataframe : pd.DataFrame, 
                       test_val_prop : float = 0.2,
                       val_prop : float = 0.3,
                       do_shuffle : bool = False,
                       n_interval : int = 1,
                       offset : int = 0,
                       window_step : int = 1,):
        data = dataframe.copy()
        if not self.column_index is None:
            data.set_index(self.column_index, inplace=True)
        self.raw_data.append(data)
        
        training_dataset : Tuple = build_forecast_ts_training_dataset(data, 
                                                              input_width=self.input_size,
                                                              target_width=self.target_size,
                                                              offset=offset,
                                                              window_step=window_step,
                                                              n_interval=n_interval,
                                                              test_val_prop=test_val_prop,
                                                              val_prop=val_prop,
                                                              do_shuffle=do_shuffle)

        self._add_ts_data(input_ts_data=training_dataset[0],
                          target_ts_data=training_dataset[1],
                          partition=self.TRAIN,
                          do_shuffle=do_shuffle)
        
        self._add_ts_data(input_ts_data=training_dataset[2],
                          target_ts_data=training_dataset[3],
                          partition=self.VALIDATION,
                          do_shuffle=do_shuffle)
        
        self._add_ts_data(input_ts_data=training_dataset[4],
                          target_ts_data=training_dataset[5],
                          partition=self.TEST,
                          do_shuffle=do_shuffle)
        
    def x_train(self, index : int = None):
        if index is None:
            return self.train_data[self.INPUT]
        return self.train_data[self.INPUT][index]
    
    def y_train(self, index : int = None):
        if index is None:
            return self.train_data[self.TARGET]
        return self.train_data[self.TARGET][index]
    
    def x_val(self, index : int = None):
        if index is None:
            return self.val_data[self.INPUT]
        return self.val_data[self.INPUT][index]
        
    def y_val(self, index : int = None):
        if index is None:
            return self.val_data[self.TARGET]
        return self.val_data[self.TARGET][index]
    
    def x_test(self, index : int = None):
        if index is None:
            return self.test_data[self.INPUT]
        return self.test_data[self.INPUT][index]
    
    def y_test(self, index : int = None):
        if index is None:
            return self.test_data[self.TARGET]
        return self.test_data[self.TARGET][index]
