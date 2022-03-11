from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from random import shuffle


class WTSwindowSizeException(Exception):
    pass


class WTSfeatureException(Exception):
    pass


class WTSemptyDataException(Exception):
    pass


def window_data(dataframe: pd.DataFrame,
                window_width: int,
                window_step: int = 1,
                selected_columns: List[str] = None,
                balance_tag: str = None) -> List[pd.DataFrame]:
    """
    Window the data given a {window_width} and a {window_step}.

    Args:
        dataframe (pd.DataFrame): The dataframe we want to window
        window_width (int): the windows size
        window_step (int, optional): the window_step between each window. Defaults to 1.

    Returns:
        List[pd.DataFrame]: The list of asscreated windows of size {window_width} and
        selected every window_step {window_step}
    """
    data = dataframe.copy()
    if len(data) == 0 or len(data) < window_width:
        return [pd.DataFrame(columns=data.columns)]
    n_windows = ((len(data.index)-window_width) // window_step) + 1
    n_columns = len(data.columns)

    # Slid window on all data
    windowed_data: np.ndarray = sliding_window_view(data,
                                                    window_shape=(window_width,
                                                                  len(data.columns)))

    # Take data every window_step
    windowed_data = windowed_data[::window_step]

    # Reshape windowed data
    windowed_data_shape: Tuple[int, int, int] = (n_windows, window_width, n_columns)
    windowed_data = np.reshape(windowed_data, newshape=windowed_data_shape)

    # Make list of dataframe
    list_windows: List[pd.DataFrame] = [
        pd.DataFrame(
            window,
            index=data.index[idx*window_step: (idx*window_step)+window_width],
            columns=data.columns)[selected_columns if selected_columns else data.columns]
        for idx, window in enumerate(windowed_data)
        if not balance_tag or data.loc[data.index[(idx*window_step)+window_width-1], balance_tag]
    ]

    return list_windows


class WTSeries(Iterable):

    def __init__(self,
                 window_width: int,
                 raw_data: Optional[pd.DataFrame] = None,
                 features: Optional[List[str]] = None,
                 window_step: int = 1,
                 tag_name: Optional[str] = None,
                 *args, **kwargs):
        """A Window Data is a list of DataFrame (windows) extract from a raw DataFrame.
        A slinding window has been apply to the raw DataFrame and every dataframe windows has been
        added in a list.
        Window Data propose simple ways to add more window, merge data, access data and etc.

        Args:
            window_width (int): it is the size of the window apply on the DataFrame and therefore
            the size of every window dataframe
            raw_data (pd.DataFrame, optional): The raw DataFrame to apply the sliding window and
            extract the data. Defaults to None.
            window_step (int, optional): The step of each slide of the window. Defaults to 1.

        Raises:
            Exception: [description]
        """
        super(WTSeries, self).__init__()
        self.index = 0
        self._window_width: int = window_width
        self.features_has_been_set = False
        if features:
            self.set_features(features)
        self.window_step = window_step

        if raw_data is not None:
            self.data: List[pd.DataFrame] = window_data(
                raw_data,
                self.width(),
                window_step,
                selected_columns=features,
                balance_tag=tag_name)
            if not self.features_has_been_set:
                self.set_features(raw_data.columns)
        else:
            self.data = []
        if not self.is_empty() and len(self.data[0].index) != self.width():
            raise Exception("The window size is suppose to be equal " +
                            "to the number of row if it is not empty")

    def __len__(self) -> int:
        """Return the number of window

        Returns:
            [int]: The number of window
        """
        return len(self.data)

    def set_features(self, features: List[str]) -> None:
        """Set the features (the columns, the dimension) of the data.

        Args:
            columns (List[str]): the list of columns names
        """
        self.features = list(features)
        self.features_has_been_set = True

    def make_common_shuffle(self, other: WTSeries):
        """perform a common shuffle on two dataframe

        Args:
            data_1 (pd.DataFrame): A DataFrame
            data_2 (pd.DataFrame): A DataFrame

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The two given dataframes shuffled in parallel
        """

        data_1_2 = list(zip(self.data.copy(), other.data.copy()))
        shuffle(data_1_2)
        data_1_shuffled, data_2_shuffled = zip(*data_1_2)
        self.data = data_1_shuffled
        other.data = data_2_shuffled

    def get_features(self) -> List[str]:
        """Return the list of columns names

        Returns:
            List[str]: list of columns names
        """
        return self.features

    def width(self) -> int:
        """Return the size of a window

        Returns:
            int: the width/size of a window
        """
        return self._window_width

    def ndim(self) -> int:
        """Return the number of columns / features

        Returns:
            int: The number of columns / features
        """
        if self.features_has_been_set:
            return len(self.features)
        return 0

    def shape(self) -> Tuple[int, int]:
        """Return the shape of a Window

        Returns:
            Tuple[int, int]: ({_window_width}, {n_features})
        """
        return self.width(), self.ndim()

    def size(self) -> Tuple[int, int, int]:
        """Return the size of Window Data

        Returns:
            Tuple[int,int,int]: Respectively:
            the number of window,
            the size of window,
            the number of features
        """
        return (len(self), self.width(), self.ndim())

    def is_empty(self) -> bool:
        """check if this window data is empty

        Returns:
            bool: True if it's empty
        """
        return len(self) == 0 or len(self.data[0].index) == 0

    def __str__(self) -> str:
        """str() function

        Returns:
            str: the text to print in print()
        """
        return f"window n°1:\n{str(self[0])}\n" +\
               f"window n°2:\n{str(self[1])}\n" +\
               f"Number of windows: {len(self)}, " +\
               f"window's size: {self.width()}, " +\
               f"number of features: {self.ndim()}"

    def __setitem__(self, index: int, value: pd.DataFrame):
        self.data[index] = value

    def __iter__(self):
        return iter(self.data)

    def __next__(self):
        r = self[self.index]
        self.index += 1
        if self.index >= len(self):
            raise StopIteration()
        return r

    def __getitem__(self, index: int) -> pd.DataFrame:
        """return a window (a dataframe) given the index of the list of {win_data}

        Args:
            index (int): an index of a window

        Returns:
            pd.DataFrame: the window
        """
        return self.data[index]

    def __call__(self, index: int = None) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """return the list of {win_data} or a window if an index is given

        Args:
            index (int, optional): an index of a window. Defaults to None.

        Returns:
            Union[List[pd.DataFrame], pd.DataFrame]: the list of {win_data} or
            a window if an index is given
        """
        if index is not None:
            return self[index]
        return self.data

    def add_data(self, data: pd.DataFrame,
                 ignore_data_empty: bool = True,
                 window_step: int = None):
        """From a raw data, perform the sliding window and add the windows to the list of
        window: {win_data}

        Args:
            data (pd.DataFrame): The raw data
            window_step (int, optional): The step of the sliding. Defaults to 1.
            ignore_data_empty (bool, optional): ignore the empty exception if it's True.
            Defaults to False.

        Raises:
            Exception: [The number of features is supposed to be the same]
            Exception: [Data is empty]
        """
        if window_step:
            w_step = window_step
        else:
            w_step = self.window_step

        if len(data) != 0:
            if self.features_has_been_set:
                if len(data.columns) != self.ndim():
                    raise WTSfeatureException("The number of features is supposed to be the same")
            data_to_add: List[pd.DataFrame] = window_data(data, self.width(), w_step)
            self.data.extend(data_to_add)
            if not self.features_has_been_set:
                self.set_features(data.columns)
        else:
            if not ignore_data_empty:
                raise WTSemptyDataException("Data is empty")

    def add_one_window(self, window: pd.DataFrame, ignore_data_empty: bool = False):
        """Add a dataframe (a window) (its length = to the {window_width}) to the list of window:
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
                if len(window.columns) != self.ndim():
                    raise WTSfeatureException("The number of features is supposed to be the same")
            if len(window.index) != self.width():
                raise WTSwindowSizeException("The window size is supposed to be the same")
            data = window.copy()
            self.data.append(data)
            if not self.features_has_been_set:
                self.set_features(data.columns)
        else:
            if not ignore_data_empty:
                raise WTSemptyDataException("Data is empty")

    def add_window_data(
        self,
        window_data: WTSeries,
        ignore_data_empty: bool = False
    ):
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
                if window_data.ndim() != self.ndim():
                    raise WTSfeatureException("The number of features is supposed to be the same")
            if window_data.width() != self.width():
                raise WTSwindowSizeException("The window size is supposed to be the same")
            self.data.extend(window_data())
            if not self.features_has_been_set:
                self.set_features(window_data.get_features())
        else:
            if not ignore_data_empty:
                raise WTSemptyDataException("Data is empty")

    def copy(self, filter: Optional[List[Union[bool, str]]] = None):
        wtseries_copy = WTSeries(self._window_width, raw_data=None, window_step=self.window_step)

        if filter:
            wtseries_copy.data = list(map(lambda window: window.loc[:, filter], self.data))
        else:
            wtseries_copy.data = self.data

        return wtseries_copy
