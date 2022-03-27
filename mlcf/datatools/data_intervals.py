"""Data Interval Module.

It is a data structure that divides the input data into n intervals and then into 3 sets such as
the train set, the validation set and the test set.
It provides DataIntervals class allowing us to handle the Nx3 intervals data structure.

    Example:

    .. code-block:: python

        from mlcf.datatools.data_intervals import DataIntervals
        from mlcf.datatools.standardize_fct import ClassicStd, MinMaxStd
        from mlcf.datatools.windowing.filter import LabelBalanceFilter

        # We define a dict which give us the information about what standardization apply to each
        # columns.
        std_by_feautures = {
            "close": ClassicStd(),
            "return": ClassicStd(with_mean=False),  # to avoid to shift we don't center
            "adx": MinMaxStd(minmax=(0, 100))  # the value observed in the adx are between
                                               # 0 and 100 and we
                                               # want to set it between 0 and 1.
        }
        data_intervals = DataIntervals(data, n_intervals=10)
        data_intervals.standardize(std_by_feautures)

        # We can apply a filter the dataset we want. Here we will filter the values in order
        # to balance the histogram of return value. For this, we use the label previously process
        # on return.
        filter_by_set = {
            "train": LabelBalanceFilter("label")  # the column we will balance the data is 'label
                                                # the max count will be automatically process
        }

        # dict_train_val_test is a dict with the key 'train', 'val', 'test'.
        # The value of the dict is a  WTSeries (a windowed time series).
        dict_train_val_test = data_intervals.windowing(
            window_width=30,
            window_step=1,
            selected_columns=["close", "return", "adx"],
            filter_by_dataset=filter_by_set,
            std_by_feature=None  # Here we can pass the same kind of dict previously introduce
                                 # to apply the standardization independtly on each window
        )
"""

from typing import Dict, Iterator, List, Optional, Tuple
import pandas as pd
from mlcf.windowing.filtering import WindowFilter
from mlcf.windowing.iterator import WTSeries

from mlcf.datatools.utils import split_train_val_test
from mlcf.datatools.standardisation import StandardisationModule, standardize


__all__ = [
    "DataIntervals"
]


class DataIntervals():
    """Data Intervals Class.
    It provides tools to divide a data frame into Nx3 parts and handles them.

    Attributes:
        raw_data (pandas.DataFrame): Raw time series data frame (unsplit).

        n_intervals (int): Number of intervals created after splitting the data frame.

        train_intervals (List[pandas.DataFrame]): List of data frame corresponding to training set
            whose length is {n_intervals}

        val_intervals (List[pandas.DataFrame]): List of data frame corresponding to validation set
            whose length is {n_intervals}

        test_intervals (List[pandas.DataFrame]): List of data frame corresponding to test set
            whose length is {n_intervals}

        intervals: (Dict[str, List[pandas.DataFrame]): A dictionary of keys 'train', 'val', 'test'
            that groups 3 sets of intervals.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        n_intervals: int,
        prop_val_test: float = 0.2,
        prop_val: float = 0.3
    ):
        """It creates a DataInterval object from a time series data frame.

        The {data} will be split into {n_intervals}.
        Then these parts will be split into 3 sets such as 'train', 'val' and 'test'.

        Args:
            data (pd.DataFrame): The time series data frame on which the split operation
                will be performed.

            n_intervals (int): The number of intervals to split the {data}.

            prop_val_test (float, optional): The proportion of val and test rows.
                The proportion of train is equal to 1-{prop_val_test}. It is set to 0.2 by default.

            prop_val (float, optional): The proportion of val set amoung the test set.
                The proportion of val rows is : {prop_val_test}*{prop_val}.
                The proportion of test part is : {prop_val_test}*(1-{prop_val}).
                It is set to 0.3 by default.
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
        """Given a {set_name}, it returns the corresponding set (list of dataframe).
        The {set_name} string value must be 'train', 'val' or 'test'.

        Args:
            set_name (str): The string key corresponds to the desired set.
                It must be chosen between 'train', 'val', 'test'.

        Raises:
            ValueError: If the {set_name} does not match any set name
                in the self.intervals dictionnary.

        Returns:
            List[pd.DataFrame]: The corresponding list of dataframe.
        """
        if set_name not in self.intervals:
            raise ValueError(
                f"{set_name} is not a set name. Choose between {list(self.intervals.keys())}")
        return self.intervals[set_name]

    def __call__(self, set_name: str) -> List[pd.DataFrame]:
        """It calls the :py:func:`~get` function and returns its result.

        Args:
            set_name (str): The string key corresponding to the desired set.
                It must be chosen between 'train', 'val', 'test'.

        Returns:
            List[pd.DataFrame]: The corresponding set.
        """
        return self.get(set_name)

    def __getitem__(self, set_name: str) -> List[pd.DataFrame]:
        """It calls the :py:func:`~get` function and returns its result.

        Args:
            set_name (str): The string key corresponding to the desired set.
                It must be chosen between 'train', 'val', 'test'.

        Returns:
            List[pd.DataFrame]: The corresponding set.
        """
        return self.get(set_name)

    def __iter__(self) -> Iterator:
        """It iterates over the {self.intervals} dictionary.

        Returns:
            Dict: The dictionary iterator
        """
        return self.intervals.__iter__()

    def __next__(self):
        return next(self.intervals)

    @classmethod
    def create_list_interval(
        self,
        data: pd.DataFrame,
        n_intervals: int = 1
    ) -> List[pd.DataFrame]:
        """It splits the {data} into {n_intervals} and returns the list of these intervals.

        Args:
            data (pd.DataFrame): The time series data frame.

            n_intervals (int, optional): The number of intervals to split the data frame.
                Default to 1.

        Raises:
            ValueError: If the data frame is empty.
            TypeError: If {n_intervals} is not an Integer.
            ValueError: If the number of intervals is less than or equal to 0.

        Returns:
            List[pd.DataFrame]: The list of data frame (intervals).
        """

        dataframe = data.copy()
        if dataframe.empty:
            raise ValueError("The data frame is empty.")
        if not isinstance(n_intervals, int):
            raise TypeError("n_intervals must be an Integer and greater than 0")
        if n_intervals < 1:
            raise ValueError("n_intervals must be greater than 0.")

        k, m = divmod(len(dataframe), n_intervals)
        list_intervals: List[pd.DataFrame] = [
            dataframe.iloc[i*k+min(i, m):(i+1)*k+min(i+1, m)]
            for i in range(n_intervals)
        ]
        return list_intervals

    # TODO (refactoring) list comprehension to create split list interval
    # TODO (enhancement) handle if test or val is null
    @classmethod
    def split_train_val_test(
        self,
        list_intervals: List[pd.DataFrame],
        prop_val_test: float,
        prop_val: float = 0.0
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        """It splits each interval (data frame) of {list_intervals} in 3 sets:
        'train', 'val' and 'test'.
        The proportions of train, val and test sets are given by
        {prop_val_test} and {prop_val} parameters.

        Args:
            list_intervals (List[pd.DataFrame]): A list of data frame.

            prop_val_test (float): The proportion of val and test rows.
                The proportion of train is equal to 1-{prop_val_test}.

            prop_val (float, optional): The proportion of val set amoung the test set.
                The proportion of val rows is : {prop_val_test}*{prop_val}.
                The proportion of test part is : {prop_val_test}*(1-{prop_val}).
                Default to 0.0.

        Returns:
            Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
                A tuple of lists corresponding to the train, validation and test parts
                of the given list data frames.
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
        std_by_feature: Dict[str, StandardisationModule]
    ) -> None:
        """An inplace operation applying a standardisation over all the data
        frame intervals of DataIntervals.
        Fit operation of the standardisation is performed only on the 'train' set.
        Transform operation of the standardisation is performed on every set.

        Args:
            std_by_feature (Dict[str, StandardisationModule]): A dictionary
                prodiving the standardisation to be applied on each column.
                The dictionary format must be as following:
                {string -> :py:class:`StandardisationModule
                <mlcf.datatools.standardisation.StandardisationModule>`}.
                The key must correspond to a column name (a feature) of the data frame.
                The value is any object inheriting from the
                :py:class:`StandardisationModule
                <mlcf.datatools.standardisation.StandardisationModule>` class.
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

    def windowing(
        self,
        window_width: int,
        window_step: int = 1,
        selected_columns: Optional[List[str]] = None,
        filter_by_dataset: Optional[Dict[str, WindowFilter]] = None,
        std_by_feature: Optional[Dict[str, StandardisationModule]] = None
    ) -> Dict[str, WTSeries]:
        """It performs the windowing operation over all intervals producing for each set a
        :py:class:`WTSeries <mlcf.windowing.iterator.tseries.WTSeries>`
        (a windowed time series).
        We can provide a dictionnary indicating the standardisation used for each feature.
        We can provide a dictionary indicating the window filtering operation applied to each set.

        Args:
            window_width (int): The window width.

            window_step (int, optional): The step between each window. Defaults to 1.

            selected_columns (Optional[List[str]], optional): The list of names of the selected
                features. If None is given, then all features will be kept.
                The default value is None.

            filter_by_dataset (Optional[Dict[str, WindowFilter]], optional): A dictionary
                that gives information about the type of window filtering applied on each set.
                The format of the dictionary is such as: {key (string) ->
                :py:class:`WindowFilter <mlcf.windowing.filtering.filter.WindowFilter>`}.
                The key is the name of a set ('train', 'val' and 'test').
                The value is any object inheriting from the
                :py:class:`WindowFilter <mlcf.windowing.filtering.filter.WindowFilter>` class.
                If None is set then no window filtering is applied. The default value is None.

            std_by_feature (Optional[Dict[str, StandardisationModule]], optional):
                A dictionary prodiving the standardisation to be applied on each column.
                Here, the standardisation is done independently on each window.
                The dictionary format must be as following:
                {string -> :py:class:`StandardisationModule
                <mlcf.datatools.standardisation.StandardisationModule>`}.
                The key must correspond to a column name (a feature) of the data frame.
                The value is any object inheriting from the
                :py:class:`StandardisationModule
                <mlcf.datatools.standardisation.StandardisationModule>` class.

        Returns:
            Dict[str, WTSeries]:
                A dictionnary such as {key (string) ->
                :py:class:`WTSeries <mlcf.windowing.iterator.tseries.WTSeries>` }
                where the key correspond to a set name.
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
