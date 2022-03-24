"""Data Interval Module
This is a data structure that divides the input data into n intervals and applies certain function
to each of these intervals. It provides a data structure to tag (True or False) rows according to
certain conditions."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from mlcf.datatools.windowing.filter import WindowFilter
from mlcf.datatools.windowing.tseries import WTSeries

from mlcf.datatools.utils import split_train_val_test
from mlcf.datatools.standardize_fct import StandardisationFct, standardize

# TODO: (doc)

__all__ = [
    "DataIntervals"
]


# TODO: (enhancement) datainterval with multi-index dataframe
class DataIntervals():
    def __init__(
        self,
        data: pd.DataFrame,
        n_intervals: int,
        prop_val_test: float = 0.2,
        prop_val: float = 0.3
    ):
        self.raw_data = data
        self.n_intervals = n_intervals
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

    def get(self, set_name: str):
        if set_name not in self.intervals:
            raise ValueError(
                f"{set_name} is not a set name. Choose between {list(self.intervals.keys())}")
        return self.intervals[set_name]

    def __call__(self, set_name: str):
        return self.get(set_name)

    def __getitem__(self, set_name: str):
        return self.get(set_name)

    def __iter__(self):
        return self.intervals

    def __next__(self):
        return next(self.intervals)

    @classmethod
    def create_list_interval(
        self,
        data: pd.DataFrame,
        n_intervals: int = 1
    ) -> List[pd.DataFrame]:

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
    @classmethod
    def split_train_val_test(
        self,
        list_intervals: List[pd.DataFrame],
        prop_val_test: float,
        prop_val: float = 0.0
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
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
