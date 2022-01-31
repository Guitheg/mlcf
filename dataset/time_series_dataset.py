from typing import Dict, List, Tuple, Union
import pandas as pd
from dataset.datatools import build_forecast_ts_training_dataset, make_commmon_shuffle
from dataset.window_data import Window_Data

class Time_Series_Dataset(object):
    def __init__(self, 
                 input_size : int,
                 target_size : int = 1,
                 column_index : str = None,
                 columns : list[str] = None,
                 *args, **kwargs):
        """Time_Series_Dataset allow to handle time series data in a machine learning training.
        The component of the Time_Series_Dataset is the Window_Data which is a list of window
        extract from window sliding of a time series data. 

        Args:
            input_size (int): The number of available time / the input width for a ml model
            target_size (int, optional): the size of the target / 
            the size of the output for a ml model. Defaults to 1.
            column_index (str, optional): the name of the column we want to index the data. In
            general it's "Date". Defaults to None.
        """
        super(Time_Series_Dataset, self).__init__(*args, **kwargs)
        
        self.TRAIN : str = "train"
        self.VALIDATION : str = "validation"
        self.TEST : str = "test"
        self.INPUT : str = "input"
        self.TARGET : str = "target"
        
        self.features_has_been_set = False
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
        if not columns is None:
            self.set_features(columns)

    def _add_ts_data(self, 
                     input_ts_data : Window_Data,
                     target_ts_data : Window_Data,
                     partition : str,
                     do_shuffle : bool = False):
        """_add_ts_data add a Input ts data and a target ts data to the train, val or test part.
        In function of the {partition} parameter which is the name of the part 
        (train, validation or test)

        Args:
            input_ts_data (Window_Data): A window data refferring to the input data
            target_ts_data (Window_Data): A window data refferring to the target data
            partition (str): the name of the part : 'train', 'validation' or 'test'
            do_shuffle (bool, optional): perform a shuffle if True. Defaults to False.
        """
        
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
    
    def add_time_serie(self, 
                       dataframe : pd.DataFrame, 
                       test_val_prop : float = 0.2,
                       val_prop : float = 0.3,
                       do_shuffle : bool = False,
                       n_interval : int = 1,
                       offset : int = 0,
                       window_step : int = 1,):
        """extend the time series data by extracting the window data from a input dataframe

        Args:
            dataframe (pd.DataFrame): A input raw dataframe
            test_val_prop (float, optional): The percentage of test and val part. Defaults to 0.2.
            val_prop (float, optional): The percentage of val part in 
            the union of test and val part. Defaults to 0.3.
            do_shuffle (bool, optional): do a shuffle if True. Defaults to False.
            n_interval (int, optional): A number of interval to divide the raw data 
            before windowing. Allow to homogenized the ts data. Defaults to 1.
            offset (int, optional): the width time between input and the target. Defaults to 0.
            window_step (int, optional): the step of each window. Defaults to 1.
        """
        data = dataframe.copy()
        if not self.column_index is None:
            data.set_index(self.column_index, inplace=True)
        if self.features_has_been_set:
            selected_data = data[self.features]
        else:
            selected_data = data
            self.set_features(data.columns)
        self.raw_data.append(data)
        training_dataset : Tuple = build_forecast_ts_training_dataset(selected_data, 
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
    
    def __call__(self, part : str = None, field : str = None) -> Union[Dict[str, Dict], 
                                                                       Dict[str, Window_Data], 
                                                                       Window_Data]:
        """return the time series data (a dict format) if None arguments has been filled.
        If part is filled, return the partition (train, validation, or test) (with a dict format).
        If field is filled, return the field (input or target) window data
        
        Args:
            part (str, optional): The partition ("train", "validation" or "test") we want to return. 
            Defaults to None.
            field (str, optional): The field ("input", or "target") we want to return.
            Defaults to None.

        Raises:
            Exception: You should fill part if field is filled

        Returns:
            Union[Dict, Dict[Window_Data], Window_Data]: 
            A dict of Dict of window data (all the time series data),
            or a dict of window data (a part 'train', 'validation' or 'test'),
            or a window data (a field 'input', 'target')
        """
        if not field is None and part is None:
            raise Exception("You should fill part if field is filled")
        elif not field is None:
            return self.ts_data[part][field]
        if not part is None and field is None:
            return self.ts_data[part]
        return self.ts_data 
            
    def __str__(self) -> str:
        return f"Input size: {self.input_size}, Target size: {self.target_size},"+\
               f"Index name: {self.column_index}.\nData : Length Train: {len(self.train_data)}, "+\
               f"Length Validation: {len(self.val_data)}, Length Test: {len(self.test_data)}"
    
    def set_features(self, features : List[str]):
        self.features = features
        self.features_has_been_set = True
         
    def n_features(self) -> int:
        if self.features_has_been_set:
            return len(self.features)
        return 0
    
    def get_input_size(self) -> int:
        return self.input_size
    
    def get_target_size(self) -> int:
        return self.target_size
                   
    def x_train(self, index : int = None) -> Union[Dict[str, Window_Data], Window_Data]:
        if index is None:
            return self.train_data[self.INPUT]
        return self.train_data[self.INPUT][index]
    
    def y_train(self, index : int = None) -> Union[Dict[str, Window_Data], Window_Data]:
        if index is None:
            return self.train_data[self.TARGET]
        return self.train_data[self.TARGET][index]
    
    def x_val(self, index : int = None) -> Union[Dict[str, Window_Data], Window_Data]:
        if index is None:
            return self.val_data[self.INPUT]
        return self.val_data[self.INPUT][index]
        
    def y_val(self, index : int = None) -> Union[Dict[str, Window_Data], Window_Data]:
        if index is None:
            return self.val_data[self.TARGET]
        return self.val_data[self.TARGET][index]
    
    def x_test(self, index : int = None) -> Union[Dict[str, Window_Data], Window_Data]:
        if index is None:
            return self.test_data[self.INPUT]
        return self.test_data[self.INPUT][index]
    
    def y_test(self, index : int = None) -> Union[Dict[str, Window_Data], Window_Data]:
        if index is None:
            return self.test_data[self.TARGET]
        return self.test_data[self.TARGET][index]
