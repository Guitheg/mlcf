""" Windowed Time Series Module
This module provides a data structure named "windowed time series (wtseries)". This data structure
allows us to handle a multi-indexed data frame that represents a windowed time series data frame.

    Example:

    .. code-block:: python

        # If we don't want to use the Data Interval Module. We can simple use a WTSeries with
        # our data.
        from mlcf.datatools.windowing.tseries import WTSeries
        # To create a WTSeries from pandas.DataFrame
        wtseries = WTSeries.create_wtseries(
            dataframe=data,
            window_width=30,
            window_step=1,
            selected_columns=["close", "return", "adx"],
            window_filter=LabelBalanceFilter("label"),
            std_by_feature=std_by_feautures
        )
        # Or from a wtseries .h5 file:
        wtseries = WTSeries.read(Path("/tests/testdata/wtseries.h5"))
        # We can save the wtseries as a file.
        wtseries.write(Path("/tests/testdata", "wtseries"))
        # we can iterate over the wtseries:
        for window in wtseries:
            pass  # Where window is a pd.Dataframe representing a window
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import numpy as np
from mlcf.datatools.standardisation import StandardisationModule
from mlcf.windowing.filtering.filter import WindowFilter
from mlcf.windowing.iterator.iterator import WindowIterator


__all__ = [
    "WINDOW_INDEX_NAME",
    "TIME_INDEX_NAME",
    "WTSeries"
]


WINDOW_INDEX_NAME = "WindowIndex"
TIME_INDEX_NAME = "TimeIndex"
SUFFIX_FILE = ".h5"


class DataEmptyException(Exception):
    pass


class IncompatibleDataException(Exception):
    pass


# TODO (doc) review
class WTSeries(WindowIterator):
    """A Windowed Time Series data structure.
    This class inherit of WindowIterator which allow us to iterate over a time series dataframe
    with a sliding window.
    The WTSeries contains {data} which is the Windowed Time Series.
    It is a multi-index pandas.DataFrame with a WindowIndex and a TimeIndex.
    WTSeries build the Windowed Time Series. So the iteration is performed on the WindowIndex
    of the dataframe.

    Attributes:
        data (pandas.DataFrame): A multi-index windowed pandas.DataFrame with a WindowIndex
            and a TimeIndex.
    """
    def __init__(self, data: pd.DataFrame):
        """Build the WTSeries from a windowed data.
        Use :py:func:`~create_wtseries` to build a WTSeries from a unwindowed data frame.

        Args:
            data (pandas.DataFrame): A multi-index windowed pandas.DataFrame with a
                WindowIndex and a TimeIndex.

        Raises:
            IncompatibleDataException: If the both WindowIndex and TimeIndex are
                not the index of the dataframe.
        """
        super(WTSeries, self).__init__()

        if not (
            data.index.names[0] == WINDOW_INDEX_NAME and
            data.index.names[1] == TIME_INDEX_NAME
        ):
            raise IncompatibleDataException(
                "To create a WTSeries please use by WTSeries.create_wtseries")
        self.data: pd.DataFrame = data

    @classmethod
    def create_wtseries(
        self,
        dataframe: pd.DataFrame,
        window_width: int,
        window_step: int,
        selected_columns: Optional[List[str]] = None,
        window_filter: Optional[WindowFilter] = None,
        std_by_feature: Optional[Dict[str, StandardisationModule]] = None
    ) -> WTSeries:
        """
        This function create a WTSeries given a time series dataframe.

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

        Raises:
            DataEmptyException: Raise this exception if the time series dataframe
                is null or if the window width is to large.

        Returns:
            WTSeries: The Windowed multi-indexed time series dataframe.
        """
        data = dataframe.copy()

        data_columns = list(data.columns)
        data[TIME_INDEX_NAME] = np.arange(len(data), dtype=int)
        if len(data) == 0 or len(data) < window_width:
            raise DataEmptyException("The given data is empty or smaller than the window width.")

        # Slid window on all data
        index_data = sliding_window_view(
            data[TIME_INDEX_NAME],
            window_shape=(window_width),
        ).reshape((-1, window_width))

        # filter and select windows
        index_data = index_data[::window_step]
        if window_filter:
            index_data = index_data[window_filter(data, index_data)]

        # Set the indexes
        window_index = np.mgrid[
            0: index_data.shape[0]: 1,
            0:window_width: 1
        ][0].reshape(-1, 1)

        # Select columns
        if selected_columns is None:
            selected_columns = data_columns

        windows = data.iloc[index_data.reshape(-1)][selected_columns]
        windows.rename_axis(TIME_INDEX_NAME, inplace=True)
        windows[WINDOW_INDEX_NAME] = window_index
        windows.set_index(WINDOW_INDEX_NAME, append=True, inplace=True)
        windows = windows.reorder_levels([WINDOW_INDEX_NAME, TIME_INDEX_NAME])

        # Standardization
        if std_by_feature:
            for feature, std_object in std_by_feature.items():
                datafit = windows[feature].values.reshape(-1, window_width)[:, :].T
                windows[feature] = std_object.fit_transform(datafit).T.reshape(-1)

        # Make list of window (dataframe)
        return WTSeries(data=windows)

    def __len__(self) -> int:
        """Return the number of window.

        Returns:
            int: The number of window.
        """
        return self.n_window

    @property
    def n_window(self) -> int:
        """The number of window

        Returns:
            int: The number of window
        """
        return len(self.data.groupby(level=WINDOW_INDEX_NAME).size())

    @property
    def width(self) -> int:
        """The window width

        Returns:
            int: The window width
        """
        return len(self.data.loc[0])

    @property
    def ndim(self) -> int:
        """
        The number of features. ndim for number of dimension (for one row)

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
        return self.data.loc[0].columns

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """Return the window corresponding to the index.

        Args:
            idx (int): the corresponding index of the wanted window

        Returns:
            pd.DataFrame: The corresponding window given an index
        """
        return self.data.loc[idx]

    def copy(self) -> WTSeries:
        """Return a copy of this WTSeries

        Returns:
            WTSeries: A copy of this WTSeries
        """
        return WTSeries(data=self.data.copy())

    # TODO (enhancement): make it as a classmethod
    def merge(self, wtseries: WTSeries) -> WTSeries:
        """Merge a wtseries to the current WTSeries.
        It will add all the window of the given wtseries to the current WTSeries.

        Args:
            wtseries (WTSeries): The WTSeries we want to merge on the current WTSeries.

        Returns:
            WTSeries: The current and the given WTSeries merged.
        """
        wtseries = wtseries.copy()
        wtseries.data.index = wtseries.data.index.set_levels([
            (
                wtseries.data.index.levels[0] + len(self)
            ).astype(int),
            wtseries.data.index.levels[1]
        ])
        return WTSeries(data=pd.concat([self.data, wtseries.data]))

    def write(self, dirpath: Path, filename: str, group_file_key: str = None) -> Path:
        """
        Create a .h5 file dataset in the {dirpath}.

        Args:
            dirpath (Path): The direcotry path where the file will be created.

            filename (str): The file name of the dataset. It will be such as : {filename}.h5

            group_file_key (str, optional): If we want to save the current data in a
                namespace of the file. Defaults to None.

        Returns:
            Path: Return the path of the saved file.
        """
        filepath = dirpath.joinpath(filename).with_suffix(SUFFIX_FILE)
        filekey = "{primary}{secondary}".format(
            primary=self.__class__.__name__,
            secondary=(f"_{group_file_key}" if group_file_key else "")
        )
        self.data.to_hdf(filepath, key=filekey, mode="w")
        return filepath

    @staticmethod
    def read(filepath: Path, group_file_key: str = None) -> WTSeries:
        """
        Read a WTSeries dataset .h5 file and return it.

        Args:
            filepath (Path): The file path of the WTSeries dataset .h5 file.

            group_file_key (str, optional): The namespace which contains the wanted data.
                Defaults to None.

        Returns:
            WTSeries: The corresponding WTSeries
        """
        filekey = "{primary}{secondary}".format(
            primary=WTSeries.__name__,
            secondary=(f"_{group_file_key}" if group_file_key else "")
        )
        return WTSeries(pd.read_hdf(filepath.with_suffix(SUFFIX_FILE), key=filekey, mode="r"))
