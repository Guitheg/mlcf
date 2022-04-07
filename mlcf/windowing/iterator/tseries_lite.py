"""Lite Windowed Time Series Module.
This module provides a data structure named "windowed time series (lite)". This data structure
allows us to handle a time series with a windowed index to simulate a windowed time series.
This data structure is lighter than the classic WTSeries.

    Example:

    .. code-block:: python

        # If we don't want to use the Data Interval Module. We can simple use a WTSeriesLite with
        # our data.
        from mlcf.datatools.windowing.tseries import WTSeriesLite
        # To create a WTSeriesLite from pandas.DataFrame
        wtseries_lite = WTSeriesLite.create_wtseries_lite(
            data=data,
            window_width=30,
            window_step=1,
            selected_columns=["close", "return", "adx"],
            window_filter=LabelBalanceFilter("label")
        )
        # Or from a wtseries_lite .h5 file:
        wtseries_lite = WTSeriesLite.read(Path("/tests/testdata/wtseries_lite.h5"))
        # We can save the wtseries_lite as a file.
        wtseries_lite.write(Path("/tests/testdata", "wtserieslite"))
        # we can iterate over the wtseries_lite lite:
        for window in wtseries_lite:
            pass  # Where window is a pd.Dataframe representing a window
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mlcf.windowing.iterator import WindowIterator, SUFFIX_FILE
from mlcf.windowing.filtering import WindowFilter


__all__ = [
    "TIME_INDEX_NAME",
    "DataEmptyException",
    "WTSeriesLite"
]


TIME_INDEX_NAME = "TimeIndex"


class DataEmptyException(Exception):
    pass


class WTSeriesLite(WindowIterator):
    """A Lite Windowed Time Series data structure.
    This class inherit of WindowIterator which allow us to iterate over a time series dataframe
    with a sliding window.
    The WTSeriesLite contains {data} which is the time series and {index_array} which is the
    windowed index.
    The data structure is seperate with the time series and the index array which simulate the
    windows.

    The memory complexity of the WTSeriesLite is:

    T*(W+F)

    with T the length of the time series, W the window width and F the number of feature.

    Attributes:
        data (pandas.DataFrame): The time series data frame.
        index_array (pandas.DataFrame): The index array which correspond to the windowed index.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        index_array: np.ndarray
    ):
        """Build a WTSeriesLite from a time series and an index_array.

        Args:
            data (pd.DataFrame): The time series data frame.

            index_array (np.ndarray): The index array which correspond to the windowed index.
        """

        self._data: pd.DataFrame = data
        self._index_array: pd.DataFrame = index_array

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def index_array(self) -> pd.DataFrame:
        return self._index_array

    @classmethod
    def create_wtseries_lite(
        self,
        data: pd.DataFrame,
        window_width: int,
        window_step: int,
        selected_columns: Optional[List[str]] = None,
        window_filter: Optional[WindowFilter] = None
    ) -> WTSeriesLite:
        """This function create a WTSeriesLite given a time series dataframe.

        Args:
            dataframe (pd.DataFrame): The time series dataframe

            window_width (int): The window width

            window_step (int): The step between each window

            selected_columns (Optional[List[str]], optional): The list of names of the selected
                features. If None is given, then all features will be kept. Defaults to None.

            window_filter (Optional[WindowFilter], optional): An object which inherit from
                :py:class:`WindowFilter <mlcf.windowing.filtering.filter.WindowFilter>`.
                This allows to filter windows according to some condition.
                For example, filter the window in order to uniformise the histogram on a feature.
                If None, any window filtering is performed. Defaults to None.

        Raises:
            DataEmptyException: Raise this exception if the time series dataframe
                is null or if the window width is to large.

        Returns:
            WTSeriesLite: The lite windowed time series dataframe.
        """

        dataframe = data.copy()

        data_columns = list(dataframe.columns)
        dataframe[TIME_INDEX_NAME] = np.arange(len(dataframe), dtype=int)
        if len(dataframe) == 0 or len(dataframe) < window_width:
            raise DataEmptyException(
                "The given dataframe is empty or smaller than the window width.")

        # Slid window on all dataframe
        index_data = sliding_window_view(
            dataframe[TIME_INDEX_NAME],
            window_shape=(window_width),
        ).reshape((-1, window_width))

        # filter and select windows
        index_data = index_data[::window_step]
        if window_filter:
            index_data = index_data[window_filter(dataframe, index_data)]

        # Select columns
        if selected_columns is None:
            selected_columns = data_columns

        # Make list of window (dataframe)
        return WTSeriesLite(
            data=dataframe[selected_columns],
            index_array=pd.DataFrame(index_data)
        )

    def filter(self, window_filter: WindowFilter) -> WTSeriesLite:
        """Filter the windows given a window filter and returns the filtered WTSeriesLite.

        Args:
            window_filter (WindowFilter): The window filter use to filter the windows.

        Returns:
            WTSeriesLite: The filterd WTSeriesLite
        """
        index_data = self.index_array.iloc[window_filter(self.data, self.index_array.values)]
        return WTSeriesLite(
            data=self.data.copy(),
            index_array=index_data
        )

    def __len__(self) -> int:
        """Return the number of window.

        Returns:
            int: The number of window.
        """

        return self.n_window

    @property
    def n_window(self) -> int:
        """The number of window.

        Returns:
            int: The number of window
        """

        return len(self.index_array)

    @property
    def width(self) -> int:
        """The window width.

        Returns:
            int: The window width
        """

        return len(self.index_array.columns)

    @property
    def ndim(self) -> int:
        """The number of features. ndim for number of dimension (for one row).

        Returns:
            int: The number of features.
        """

        return len(self.data.columns)

    @property
    def features(self) -> List[str]:
        """The list of features name (column name of the dataframe).

        Returns:
            List[str]: List of the features name.
        """

        return self.data.columns

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """Return the window corresponding to the index.

        Args:
            idx (int): the corresponding index of the wanted window

        Returns:
            pd.DataFrame: The corresponding window given an index
        """

        return self.data.iloc[self.index_array.iloc[idx]]

    def copy(self) -> WTSeriesLite:
        """Return a copy of this WTSeriesLite.

        Returns:
            WTSeriesLite: A copy of this WTSeriesLite
        """

        return WTSeriesLite(
            data=self.data.copy(),
            index_array=self.index_array.copy()
        )

    # TODO (enhancement): make it as a classmethod
    def merge(self, wtseries_lite: WTSeriesLite) -> WTSeriesLite:
        """Merge a wtseries lite to the current WTSeriesLite.
        It will add all the window of the given wtseries lite to the current WTSeriesLite.

        Args:
            wtseries_lite (WTSeriesLite): The WTSeriesLite we want to merge on the current
                WTSeriesLite.

        Returns:
            WTSeriesLite: The current and the given WTSeriesLite merged.
        """

        return WTSeriesLite(
            data=self.data,
            index_array=pd.concat([self.index_array, wtseries_lite.index_array])
        )

    def write(self, dirpath: Path, filename: str, group_file_key: str = None) -> Path:
        """Create a .h5 file dataset in the {dirpath}.

        Save the time series dataframe and the index array.

        Args:
            dirpath (Path): The direcotry path where the file will be created.

            filename (str): The file name of the dataset. It will be such as : {filename}.h5

            group_file_key (str, optional): If we want to save the current data in a
                namespace of the file. Defaults to None.

        Returns:
            Path: Return the path of the saved file.
        """

        filepath = dirpath.joinpath(filename).with_suffix(SUFFIX_FILE)
        filekey_data = self._get_dataset_namespace("data", group_file_key=group_file_key)
        filekey_index = self._get_dataset_namespace("index", group_file_key=group_file_key)
        self.data.to_hdf(filepath, key=filekey_data, mode="a")
        self.index_array.to_hdf(filepath, key=filekey_index, mode="a")
        return filepath

    @staticmethod
    def read(filepath: Path, group_file_key: str = None) -> WTSeriesLite:
        """Read a WTSeriesLite dataset .h5 file and return it.

        Args:
            filepath (Path): The file path of the WTSeriesLite dataset .h5 file.

            group_file_key (str, optional): The namespace which contains the wanted data.
                Defaults to None.

        Returns:
            WTSeriesLite: The corresponding WTSeriesLite
        """

        filekey_data = WTSeriesLite._get_dataset_namespace("data", group_file_key=group_file_key)
        filekey_index = WTSeriesLite._get_dataset_namespace("index", group_file_key=group_file_key)
        data = pd.read_hdf(filepath.with_suffix(SUFFIX_FILE), key=filekey_data, mode="r")
        index_array = pd.read_hdf(filepath.with_suffix(SUFFIX_FILE), key=filekey_index, mode="r")
        return WTSeriesLite(data=data, index_array=index_array)

    @classmethod
    def _get_dataset_namespace(self, data_type: str, group_file_key: str = None):
        return "{typedata}_{primary}{secondary}".format(
            typedata=data_type,
            primary=WTSeriesLite.__name__,
            secondary=(f"_{group_file_key}" if group_file_key else "")
        )
