from __future__ import annotations
from typing import List, Tuple, Union
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

class WTSeries(object):

    def __init__(self, 
                 window_size : int, 
                 data : pd.DataFrame = None, 
                 window_step : int = 1,
                 *args, **kwargs):
        """A Window Data is a list of DataFrame (windows) extract from a raw DataFrame.
        A slinding window has been apply to the raw DataFrame and every dataframe windows has been 
        added in a list.
        Window Data propose simple ways to add more window, merge data, access data and etc.

        Args:
            window_size (int): it is the size of the window apply on the DataFrame and therefore
            the size of every window dataframe
            data (pd.DataFrame, optional): The raw DataFrame to apply the sliding window and 
            extract the data. Defaults to None.
            window_step (int, optional): The step of each slide of the window. Defaults to 1.

        Raises:
            Exception: [description]
        """
        super(WTSeries, self).__init__(*args, **kwargs)
        
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
            
    def __len__(self) -> int:
        """Return the number of window

        Returns:
            [int]: The number of window
        """
        return len(self.win_data)
    
    def set_features(self, columns : List[str]) -> None:
        """Set the features (the columns, the dimension) of the data.

        Args:
            columns (List[str]): the list of columns names
        """
        self.features = columns
        self.features_has_been_set = True
        
    def get_features(self) -> List[str]:
        """Return the list of columns names

        Returns:
            List[str]: list of columns names
        """
        return self.features
    
    def window_size(self) -> int:
        """Return the size of a window

        Returns:
            int: the width/size of a window
        """
        return self._window_size
    
    def n_features(self) -> int:
        """Return the number of columns / features

        Returns:
            int: The number of columns / features
        """
        if self.features_has_been_set:
            return len(self.features)
        return 0
    
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of Window Data
        
        Returns:
            Tuple[int,int,int]: Respectively : 
            the number of window, 
            the size of window, 
            the number of features
        """
        return (len(self), self.window_size(), self.n_features())
    
    def is_empty(self) -> bool:
        """check if this window data is empty

        Returns:
            bool: True if it's empty
        """
        return len(self) == 0 or len(self.win_data[0].index) == 0
    
    def __str__(self) -> str:
        """str() function

        Returns:
            str: the text to print in print()
        """
        return f"window n°1:\n{str(self[0])}\n" +\
               f"window n°2:\n{str(self[1])}\n" +\
               f"Number of windows : {len(self)}, " +\
               f"window's size : {self.window_size()}, " +\
               f"number of features : {self.n_features()}"
    
    def __getitem__(self, index : int) -> pd.DataFrame:
        """return a window (a dataframe) given the index of the list of {win_data}

        Args:
            index (int): an index of a window

        Returns:
            pd.DataFrame: the window
        """
        return self.win_data[index]
    
    def __call__(self, index : int = None) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """return the list of {win_data} or a window if an index is given

        Args:
            index (int, optional): an index of a window. Defaults to None.

        Returns:
            Union[List[pd.DataFrame], pd.DataFrame]: the list of {win_data} or 
            a window if an index is given
        """
        if not index is None:
            return self[index]
        return self.win_data
    
    def add_data(self, data : pd.DataFrame, 
                 window_step : int = 1, 
                 ignore_data_empty : bool = False):
        """From a raw data, perform the sliding window and add the windows to the list of 
        window : {win_data}

        Args:
            data (pd.DataFrame): The raw data
            window_step (int, optional): The step of the sliding. Defaults to 1.
            ignore_data_empty (bool, optional): ignore the empty exception if it's True. 
            Defaults to False.

        Raises:
            Exception: [The number of features is supposed to be the same]
            Exception: [Data is empty]
        """
        if len(data) != 0:
            if self.features_has_been_set:
                if len(data.columns) != self.n_features():
                    raise Exception("The number of features is supposed to be the same")
            data_to_add : List[pd.DataFrame] = window_data(data, self.window_size(), window_step)
            self.win_data.extend(data_to_add)
            if not self.features_has_been_set:
                self.set_features(data.columns)
        else:
            if not ignore_data_empty:
                raise Exception("")
            
    def add_one_window(self, window : pd.DataFrame, ignore_data_empty : bool = False):
        """Add a dataframe (a window) (its length = to the {window_size}) to the list of window : 
        {win_data}

        Args:
            window (pd.DataFrame): A dataframe (a window)
            ignore_data_empty (bool, optional): ignore the empty exception if it's True. 
            Defaults to False.

        Raises:
            Exception: [The number of features is supposed to be the same]
            Exception: [The window size is supposed to be the same]
            Exception: [Data is empty]
        """
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
                          window_data : WTSeries, 
                          ignore_data_empty : bool = False):
        """merge the input WTSeries to the current WTSeries

        Args:
            window_data (WTSeries): the input WTSeries to merge into this one
            ignore_data_empty (bool, optional): ignore the empty exception if it's True.
            Defaults to False.

        Raises:
            Exception: [The number of features is supposed to be the same]
            Exception: [The window size is supposed to be the same]
            Exception: [Data is empty]
        """
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