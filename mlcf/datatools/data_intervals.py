"""Data Interval Module
This is a data structure that divides the input data into n intervals and then into 3 set such as 
train set, validation set and test set. This module provide DataIntervals allowing us to handle the 
Nx3 intervals data structure."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from mlcf.datatools.windowing.filter import WindowFilter
from mlcf.datatools.windowing.tseries import WTSeries

from mlcf.datatools.utils import split_train_val_test
from mlcf.datatools.standardize_fct import StandardisationFct, standardize


__all__ = [
    "DataIntervals"
]


# TODO: (enhancement) datainterval with multi-index dataframe
class DataIntervals():
    """
    DataIntervals is a class which provide tools to divide a dataframe in Nx3 parts and then handeling it.

    Attributes:
        raw_data (pandas.DataFrame): The raw time series dataframe (not splited).
        
        n_intervals (int): The number of intervals the dataframe has been splited.
        
        train_intervals (List[pandas.DataFrame]): A list of {n_intervals} dataframe which correspond to the {n_intervals} train set.
        
        val_intervals (List[pandas.DataFrame]): A list of {n_intervals} dataframe which correspond to the {n_intervals} validation set.
        
        test_intervals (List[pandas.DataFrame]): A list of {n_intervals} dataframe which correspond to the {n_intervals} test set.
        
        intervals: (Dict[str, List[pandas.DataFrame]): A dictionnary which regroup the 3 intervals set. With the 'train', 'val' and 'test' keys
        returning the corresponding list of dataframe.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        n_intervals: int,
        prop_val_test: float = 0.2,
        prop_val: float = 0.3
    ):
        """
        Create a DataInterval object given a time series dataframe.
        The {data} will be splited into {n_intervals} parts.
        Then theses parts will be splited into 3 set such as 'train', 'val' and 'test' set.

        Args:
            data (pd.DataFrame): The time series dataframe on which the splitted operation will be perform.
            
            n_intervals (int): The number of intervals to split the {data}.
            
            prop_val_test (float, optional): The val and test rows proportion.
                The proportion of train is equal to: 1-{prop_val_test}. Defaults to 0.2.
                
            prop_val (float, optional): The val set proportion amoung the test set.
                The proportion of val rows is : {prop_val_test}*{prop_val}. 
                The proportion of test part is : {prop_val_test}*(1-{prop_val}). Defaults to 0.3.
        """
        self.raw_data: pd.DataFrame = data
        self.n_intervals: int = n_intervals
        self.train_intervals, self.val_intervals, self.test_intervals = \
            self.split_train_val_test(
                list_intervals=self.create_list_interval(
                    data=self.raw_data,
                    n_intervals=self.n_intervals),
                prop_val_test=prop_val_test,
                prop_val=prop_val
            )
        self.intervals: Dict = {
            "train": self.train_intervals,
            "val": self.val_intervals,
            "test": self.test_intervals
        }

    def get(self, set_name: str) -> List[pd.DataFrame]:
        """
        Given a {set_name} return the corresponding set (list of dataframe).
        The {set_name} string value must be 'train', 'val' or 'test'.

        Args:
            set_name (str): The string key corresponding to the set we want.
                Choose between 'train', 'val', 'test'.

        Raises:
            ValueError: If the {set_name} correspond to any set in the self.intervals dictionnary.

        Returns:
            List[pd.DataFrame]: The corresponding set.
        """
        if set_name not in self.intervals:
            raise ValueError(
                f"{set_name} is not a set name. Choose between {list(self.intervals.keys())}")
        return self.intervals[set_name]

    def __call__(self, set_name: str) -> List[pd.DataFrame]:
        """
        Call the get(self, set_name) function and return its result.

        Args:
            set_name (str): The string key corresponding to the set we want.
                Choose between 'train', 'val', 'test'.

        Returns:
            List[pd.DataFrame]: The corresponding set.
        """
        return self.get(set_name)

    def __getitem__(self, set_name: str) -> List[pd.DataFrame]:
        """
        Call the get(self, set_name) function and return its result.

        Args:
            set_name (str): The string key corresponding to the set we want.
                Choose between 'train', 'val', 'test'.

        Returns:
            List[pd.DataFrame]: The corresponding set.
        """
        return self.get(set_name)

    def __iter__(self) -> Dict:
        """
        Iterate over the {self.intervals} dictionnary

        Returns:
            Dict: The dictionnary which contains all the set.
        """
        return self.intervals

    def __next__(self):
        return next(self.intervals)

    @classmethod
    def create_list_interval(
        self,
        data: pd.DataFrame,
        n_intervals: int = 1
    ) -> List[pd.DataFrame]:
        """
        Split the {data} in a number of {n_intervals}. Returns the list of these intervals.

        Args:
            data (pd.DataFrame): The time series dataframe.

            n_intervals (int, optional): The number of intervals to split the dataframe. Defaults to 1.

        Raises:
            ValueError: If the dataframe is empty.
            TypeError: If {n_intervals} is not an Integer.
            ValueError: If the number of intervals is less than or equal to 0

        Returns:
            List[pd.DataFrame]: The list of dataframe (intervals)
        """

        dataframe = data.copy()
        if dataframe.empty:
            raise ValueError("The dataframe is empty.")
        if not isinstance(n_intervals, int):
            raise TypeError("n_intervals must be a int greater than 0")
        if n_intervals < 1:
            raise ValueError("n_intervals must be greater than 0")

        k, m = divmod(len(dataframe), n_intervals)
        list_intervals: List[pd.DataFrame] = [
            dataframe.iloc[i*k+min(i, m):(i+1)*k+min(i+1, m)]
            for i in range(n_intervals)
        ]
        return list_intervals

    # TODO: (refactoring) list comprehension to create split list interval
    # TODO: (enhancement) handle if test or val is null
    @classmethod
    def split_train_val_test(
        self,
        list_intervals: List[pd.DataFrame],
        prop_val_test: float,
        prop_val: float = 0.0
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Split each interval (dataframe) of the {list_intervals} in 3 set part 'train', 'val' and 'test'.
        The proportion of train, val and test set are given by the {prop_val_test} and the {prop_val} parameters.

        Args:
            list_intervals (List[pd.DataFrame]): A list of dataframe

            prop_val_test (float): The val and test rows proportion.
                The proportion of train is equal to: 1-{prop_val_test}.

            prop_val (float, optional): The val set proportion amoung the test set.
                The proportion of val rows is : {prop_val_test}*{prop_val}. 
                The proportion of test part is : {prop_val_test}*(1-{prop_val}). Defaults to 0.0.

        Returns:
            Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]: _description_
        """
        splited_list_interval: Tuple[
            List[pd.DataFrame],
            List[pd.DataFrame],
            List[pd.DataFrame]] = ([], [], [])

        for interval in list_intervals:
            train, val, test = split_train_val_test(
                data=interval,
                prop_val_test=prop_val_test,
                prop_val=prop_val)
            splited_list_interval[0].append(train)
            splited_list_interval[1].append(val)
            splited_list_interval[2].append(test)

        return splited_list_interval

    def standardize(
        self,
        std_by_feature: Dict[str, StandardisationFct]
    ) -> None:
        """
        An Inplace operation. Apply a standardisation over all the dataframes contains in DataIntervals.
        The fit operation of the standardisation is perform only to the 'train' set.
        The transform operation of the standardisation is perform to every set.
        
        Args:
            std_by_feature (Dict[str, StandardisationFct]): A dict which give us the information about what standardisation apply to each
                columns. The dict format must be : {key (string) -> value (StandardisationFct)}.
                The key must correspond to a column name (a feature) of the dataframes.
                The standardisation function must inherit from StandardisationFct class.
        """
        self.std_by_feature = std_by_feature

        self.train_intervals = standardize(
            std_by_feature=self.std_by_feature,
            fit_data=self.train_intervals,
            transform_data=self.train_intervals
        )

        self.val_intervals = standardize(
            std_by_feature=self.std_by_feature,
            fit_data=[],
            transform_data=self.val_intervals
        )

        self.test_intervals = standardize(
            std_by_feature=self.std_by_feature,
            fit_data=[],
            transform_data=self.test_intervals
        )

    def data_windowing(
        self,
        window_width: int,
        window_step: int = 1,
        selected_columns: Optional[List[str]] = None,
        filter_by_dataset: Optional[Dict[str, WindowFilter]] = None,
        std_by_feature: Optional[Dict[str, StandardisationFct]] = None
    ) -> Dict[str, WTSeries]:
        """
        It perform a windowing operation over all intervals producing for each set a WTSeries (a windowed time series).
        We can provide a window filtering operation and a window standardisation operation.
        
        Args:
            window_width (int): The window width
            
            window_step (int, optional): The step between each window. Defaults to 1.
            
            selected_columns (Optional[List[str]], optional): The list of name of features we want to keep in the WTSeries.
                If None is given, then every feature will be kept. Defaults to None.
                
            filter_by_dataset (Optional[Dict[str, WindowFilter]], optional): A dict which give the information about what window filtering will be applied
                on each set.
                The dict format is such as : {key (string) -> value (WindowFilter)}.
                The key is the name of the set ('train', 'val' and 'test').
                The value is a class which inherit from the WindowFilter class.
                If None, any window filtering is apply. Defaults to None.
                
            std_by_feature (Optional[Dict[str, StandardisationFct]], optional): 
                A dict which give us the information about what standardisation apply to each
                columns. 
                Here, the standardisation are perform per window independently.
                The dict format must be : {key (string) -> value (StandardisationFct)}.
                The key must correspond to a column name (a feature) of the dataframes.
                The standardisation function must inherit from StandardisationFct class.
                If None, any standardisation is apply. Defaults to None.

        Returns:
            Dict[str, WTSeries]: A dict such as {key (string) -> value (WTSeries)} where the key correspond to a set name.
        """

        data_windowed: Dict[str, WTSeries] = {}
        for key, intervals in self.intervals.items():
            if filter_by_dataset and key in filter_by_dataset:
                window_filter = filter_by_dataset[key]
            else:
                window_filter = None
            for interval in intervals:
                if len(interval) >= window_width:
                    wtseries = WTSeries.create_wtseries(
                        dataframe=interval,
                        window_width=window_width,
                        window_step=window_step,
                        selected_columns=selected_columns,
                        window_filter=window_filter,
                        std_by_feature=std_by_feature
                    )
                    if key in data_windowed:
                        data_windowed[key] = data_windowed[key].merge(wtseries)
                    else:
                        data_windowed[key] = wtseries
        return data_windowed
