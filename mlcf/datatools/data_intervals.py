"""__summary__
"""

from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import random
from mlcf.datatools.sliding_windows import data_windowing

from mlcf.datatools.utils import split_train_val_test
from mlcf.datatools.standardize_fct import StandardisationFct, standardize

__all__ = [

]


class AnyStepTag(Exception):
    pass


class HaveAlreadyAStepTag(Exception):
    pass


class TagCreator():
    def __call__(
        self,
        data: pd.DataFrame,
        columns: List[str],
        *args, **kwargs
    ) -> pd.Series:
        return pd.Series([True]*len(data[columns]))


class BalanceTag(TagCreator):
    def __call__(
        self,
        data: pd.DataFrame,
        columns: List[str],
        max_count: int = None,
        sample_function: Callable = random.sample,
        *args, **kwargs
    ) -> pd.Series:
        dataframe = data.loc[data["step_tag"]].copy()
        idx_cat = dataframe[columns].value_counts().index
        value_count = dataframe[columns].value_counts()
        if not max_count:
            max_count = np.mean([value_count[min(idx_cat)], value_count[max(idx_cat)]]).astype(int)
        tag_col = data["step_tag"]
        for idx in sorted(value_count.index):
            if value_count[idx] > max_count:
                tag_col.loc[
                    sorted(sample_function(
                        list(dataframe[dataframe[columns] == idx].index),
                        k=value_count[idx]-max_count)
                    )
                ] = False
        return tag_col


class DataInIntervals():
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
        self.step_tag: int = 0

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
            self.std_by_feature,
            fit_data=self.train_intervals,
            transform_data=self.train_intervals)
        self.val_intervals = standardize(
            self.std_by_feature,
            fit_data=[],
            transform_data=self.val_intervals)
        self.test_intervals = standardize(
            self.std_by_feature,
            fit_data=[],
            transform_data=self.test_intervals)

    def add_step_tag(self, step: int):
        if self.step_tag:
            raise HaveAlreadyAStepTag(
                "Cannot recompute a new step tag considering there " +
                f"are already a step tag ({self.step_tag})"
            )
        if step < 1:
            raise ValueError("The step value must be greater than 0")
        for _, intervals in self.intervals.items():
            for interval in intervals:
                array = np.zeros(len(interval), dtype=bool)
                array[::step] = True
                interval["step_tag"] = array
        self.step_tag = step

    def add_tag(
        self,
        tag_name: str,
        list_partitions: List[str],
        columns: List[str],
        tag_creator: TagCreator,
        *args, **kwargs
    ):
        if not self.step_tag:
            raise AnyStepTag("There are any step tag. We need a step tag to add a tag")
        for partition in list_partitions:
            for interval in self.intervals[partition]:
                interval[tag_name] = tag_creator(
                    data=interval,
                    columns=columns,
                    *args,
                    **kwargs
                )

    def data_windowing(
        self,
        window_width: int,
        selected_columns: Optional[List[str]] = None,
        predicate_row_selection: Optional[Callable] = None,
        std_by_feature: Optional[Dict[str, StandardisationFct]] = None
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        data_windowed: Dict[str, List[pd.DataFrame]] = {
            key: reduce(
                    list.__add__,
                    [
                        data_windowing(
                            dataframe=interval,
                            window_width=window_width,
                            window_step=self.step_tag,
                            selected_columns=selected_columns,
                            predicate_row_selection=predicate_row_selection,
                            std_by_feature=std_by_feature
                        )
                        for interval in intervals
                    ]
                )
            for key, intervals in self.intervals.items()}
        return data_windowed
