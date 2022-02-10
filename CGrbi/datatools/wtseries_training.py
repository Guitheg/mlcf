from typing import Dict, List, Tuple, Union
import pandas as pd
from enum import Enum, unique

from os.path import isdir, join
from pathlib import Path
import pickle

### CG-RBI modules ###
from CGrbi.datatools.preprocessing import Identity, WTSeriesPreProcess
from CGrbi.datatools.utils import build_forecast_ts_training_dataset, make_commmon_shuffle
from CGrbi.datatools.wtseries import WTSeries

@unique
class Partition(Enum):
    TRAIN : str = "train"
    VALIDATION : str = "validation"
    TEST : str = "test"

@unique  
class Field(Enum):
    INPUT : str = "input"
    TARGET : str = "target"

TRAIN : str = Partition.TRAIN.value
VALIDATION : str = Partition.VALIDATION.value
TEST : str = Partition.TEST.value
INPUT : str = Field.INPUT.value
TARGET : str = Field.TARGET.value
EXTENSION_FILE = ".wtst"

def read_wtseries_training(path : Path):
    if not isinstance(path, Path):
        if isinstance(path, str):
            dir = Path(path)
        else:
            raise Exception(f"path instance is not attempted : {type(path)}")
    if not path.is_file():
        raise Exception("The given file path is unknown")
    if path.suffix != EXTENSION_FILE:
        raise Exception("The given file is not a WTST FILE (.wtst)")
    with open(path, "rb") as read_file:
        wtseries_training : WTSeriesTraining = pickle.load(read_file)
    return wtseries_training

class WTSeriesTraining(object):
    def __init__(self, 
                 input_size : int,
                 target_size : int = 1,
                 index_column : str = None,
                 features : List[str] = None,
                 *args, **kwargs):
        """WTSeriesTraining allow to handle time series data in a machine learning training.
        The component of the WTSeriesTraining is the WTSeries which is a list of window
        extract from window sliding of a time series data. 

        Args:
            input_size (int): The number of available time / the input width for a ml model
            target_size (int, optional): the size of the target / 
            the size of the output for a ml model. Defaults to 1.
            index_column (str, optional): the name of the column we want to index the data. In
            general it's "Date". Defaults to None.
        """
        super(WTSeriesTraining, self).__init__(*args, **kwargs)
        
        self.features_has_been_set = False
        self.raw_data : List[pd.DataFrame] = []
        self.input_size : int = input_size
        self.target_size : int = target_size
        self.index_column : str = index_column
        self.features : List[str] = None
        
        self.train_data = {INPUT : WTSeries(self.input_size), 
                           TARGET : WTSeries(self.target_size)}
        
        self.val_data = {INPUT : WTSeries(self.input_size), 
                         TARGET : WTSeries(self.target_size)}
        
        self.test_data = {INPUT : WTSeries(self.input_size), 
                          TARGET : WTSeries(self.target_size)}
        
        self.ts_data : Dict = {TRAIN : self.train_data,
                               VALIDATION : self.val_data,
                               TEST : self.test_data}
        if not features is None:
            self._set_features(features)
    
    def write(self, dir : Path, name : str):
        if not isinstance(dir, Path):
            if isinstance(dir, str):
                dir = Path(dir)
            else:
                raise Exception(f"dir instance is not attempted : {type(dir)}")
        if not dir.is_dir():
            raise Exception("The given directory is unknown")
        path : Path = dir.joinpath(name).with_suffix(EXTENSION_FILE)
        with open(path, "wb") as write_file:
            pickle.dump(self, write_file, pickle.HIGHEST_PROTOCOL)

    def _add_ts_data(self, 
                     input_ts_data : WTSeries,
                     target_ts_data : WTSeries,
                     partition : Partition,
                     do_shuffle : bool = False):
        """_add_ts_data add a Input ts data and a target ts data to the train, val or test part.
        In function of the {partition} parameter which is the name of the part 
        (train, validation or test)

        Args:
            input_ts_data (WTSeries): A window data refferring to the input data
            target_ts_data (WTSeries): A window data refferring to the target data
            partition (Partition): the name of the part : 'train', 'validation' or 'test'
            do_shuffle (bool, optional): perform a shuffle if True. Defaults to False.
        """
        
        self.ts_data[partition][INPUT].merge_window_data(input_ts_data, 
                                                              ignore_data_empty=True)
        self.ts_data[partition][TARGET].merge_window_data(target_ts_data,
                                                               ignore_data_empty=True)
        if do_shuffle:
            (input_data_shuffle, target_data_shuffle) = make_commmon_shuffle(
                self.ts_data[partition][INPUT],
                self.ts_data[partition][TARGET]) 
            self.ts_data[partition][INPUT] = input_data_shuffle
            self.ts_data[partition][TARGET] = target_data_shuffle
    
    def add_time_serie(self, 
                       dataframe : pd.DataFrame, 
                       prop_tv : float = 0.2,
                       prop_v : float = 0.3,
                       do_shuffle : bool = False,
                       n_interval : int = 1,
                       offset : int = 0,
                       window_step : int = 1,
                       preprocess : WTSeriesPreProcess = Identity):
        """extend the time series data by extracting the window data from a input dataframe

        Args:
            dataframe (pd.DataFrame): A input raw dataframe
            prop_tv (float, optional): The percentage of test and val part. Defaults to 0.2.
            prop_v (float, optional): The percentage of val part in 
            the union of test and val part. Defaults to 0.3.
            do_shuffle (bool, optional): do a shuffle if True. Defaults to False.
            n_interval (int, optional): A number of interval to divide the raw data 
            before windowing. Allow to homogenized the ts data. Defaults to 1.
            offset (int, optional): the width time between input and the target. Defaults to 0.
            window_step (int, optional): the step of each window. Defaults to 1.
        """
        data = dataframe.copy()
        if not self.index_column is None:
            data.set_index(self.index_column, inplace=True)
        if self.features_has_been_set:
            selected_data = data[self.features]
        else:
            selected_data = data
            self._set_features(data.columns)
        self.raw_data.append(data)
        training_dataset : Tuple = build_forecast_ts_training_dataset(selected_data, 
                                                              input_width=self.input_size,
                                                              target_width=self.target_size,
                                                              offset=offset,
                                                              window_step=window_step,
                                                              n_interval=n_interval,
                                                              prop_tv=prop_tv,
                                                              prop_v=prop_v,
                                                              do_shuffle=do_shuffle,
                                                              preprocess=preprocess)

        self._add_ts_data(input_ts_data=training_dataset[0],
                          target_ts_data=training_dataset[1],
                          partition=TRAIN,
                          do_shuffle=do_shuffle)
        
        self._add_ts_data(input_ts_data=training_dataset[2],
                          target_ts_data=training_dataset[3],
                          partition=VALIDATION,
                          do_shuffle=do_shuffle)
        
        self._add_ts_data(input_ts_data=training_dataset[4],
                          target_ts_data=training_dataset[5],
                          partition=TEST,
                          do_shuffle=do_shuffle)
    
    def __call__(self, 
                 part : Partition = None, 
                 field : Field = None) -> Union[Dict[str, Dict], Dict[str, WTSeries], WTSeries]:
        """return the time series data (a dict format) if None arguments has been filled.
        If part is filled, return the partition (train, validation, or test) (with a dict format).
        If field is filled, return the field (input or target) window data
        
        Args:
            part (Partition, optional): The partition ("train", "validation" or "test") we want to return. 
            Defaults to None.
            field (Field, optional): The field ("input", or "target") we want to return.
            Defaults to None.

        Raises:
            Exception: You should fill part if field is filled

        Returns:
            Union[Dict, Dict[WTSeries], WTSeries]: 
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
        return f"Input size: {self.input_size}, Target size: {self.target_size}, "+\
               f"Index name: '{self.index_column}' - Data : "+\
               f"Length Train: {self.len(TRAIN)}, "+\
               f"Length Validation: {self.len(VALIDATION)}, "+\
               f"Length Test: {self.len(TEST)}"
    
    def len(self, part : Partition = None) -> int:
        """Return the length of a partition 'train', 'val' or 'test'

        Args:
            part (Partition, optional): 'train', 'val' or 'test'. Defaults to None.

        Returns:
            int: The sum of the 3 parts length if is None. Else return the length of the part
        """
        if not part is None:
            return len(self(part, INPUT))
        return len(self(TRAIN, INPUT)) + len(self(VALIDATION, INPUT)) + len(self(TEST, INPUT))
    
    def width(self, field : Field = None) -> Union[int, Tuple[int, int]]:
        """return the width of windows data given 'input' or 'target'

        Args:
            field (Field, optional): 'input' or 'target'. Defaults to None.

        Raises:
            ValueError: Only 'input' or 'target' are allowed

        Returns:
            Union[int, Tuple[int, int]]: (input or target width)
            or respectively both if field is None
        """
        if not field is None:
            if field == INPUT:
                return self.input_size
            elif field == TARGET:
                return self.target_size
            else:
                raise ValueError("Only 'input' or 'target' are allowed")
        else:
            return self.input_size, self.target_size
    
    def ndim(self) -> int:
        if self.features_has_been_set:
            return len(self.features)
        return 0
    
    def __len__(self):
        return len()
    
    def _set_features(self, features : List[str]):
        self.features = features
        self.features_has_been_set = True
                   
    def x_train(self, index : int = None) -> Union[Dict[str, WTSeries], WTSeries]:
        if index is None:
            return self.train_data[INPUT]
        return self.train_data[INPUT][index]
    
    def y_train(self, index : int = None) -> Union[Dict[str, WTSeries], WTSeries]:
        if index is None:
            return self.train_data[TARGET]
        return self.train_data[TARGET][index]
    
    def x_val(self, index : int = None) -> Union[Dict[str, WTSeries], WTSeries]:
        if index is None:
            return self.val_data[INPUT]
        return self.val_data[INPUT][index]
        
    def y_val(self, index : int = None) -> Union[Dict[str, WTSeries], WTSeries]:
        if index is None:
            return self.val_data[TARGET]
        return self.val_data[TARGET][index]
    
    def x_test(self, index : int = None) -> Union[Dict[str, WTSeries], WTSeries]:
        if index is None:
            return self.test_data[INPUT]
        return self.test_data[INPUT][index]
    
    def y_test(self, index : int = None) -> Union[Dict[str, WTSeries], WTSeries]:
        if index is None:
            return self.test_data[TARGET]
        return self.test_data[TARGET][index]
