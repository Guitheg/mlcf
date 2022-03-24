"""Windowed Time Series Module
This module provides a data structure named "windowed time series (wtseries)". This data structure
allows us to handle a multi-indexed data frame that represents a windowed time series data frame.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import numpy as np
from mlcf.datatools.standardize_fct import StandardisationFct
from mlcf.datatools.windowing.filter import WindowFilter
from mlcf.datatools.windowing.iterator import WindowIterator

# TODO: (doc)

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


class WTSeries(WindowIterator):
    """

    Attributes:
        data (pandas.DataFrame): __description__
    """
    def __init__(self, data):
        super(WTSeries, self).__init__()

        if not (
            data.index.names[0] == WINDOW_INDEX_NAME and
            data.index.names[1] == TIME_INDEX_NAME
        ):
            raise IncompatibleDataException(
                "To create a WTSeries please use by WTSeries.create_wtseries")
        self.data: pd.DataFrame = data

    # TODO: (opti) parrallelize standardization ?
    @classmethod
    def create_wtseries(
        self,
        dataframe: pd.DataFrame,
        window_width: int,
        window_step: int,
        selected_columns: Optional[List[str]] = None,
        window_filter: Optional[WindowFilter] = None,
        std_by_feature: Optional[Dict[str, StandardisationFct]] = None
    ) -> WTSeries:
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
            window_filter(data, index_data)
            index_data = index_data[[window_filter[idx] for idx in index_data]]

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
        return self.n_window

    @property
    def n_window(self) -> int:
        return len(self.data.groupby(level=WINDOW_INDEX_NAME).size())

    @property
    def width(self) -> int:
        return len(self.data.loc[0])

    @property
    def ndim(self) -> int:
        return len(self.data.columns)

    @property
    def features(self) -> List[str]:
        return self.data.loc[0].columns

    def __getitem__(self, idx) -> pd.DataFrame:
        return self.data.loc[idx]

    def copy(self):
        return WTSeries(data=self.data.copy())

    def merge(self, wtseries) -> WTSeries:
        wtseries = wtseries.copy()
        wtseries.data.index = wtseries.data.index.set_levels([
            (
                wtseries.data.index.levels[0] + len(self)
            ).astype(int),
            wtseries.data.index.levels[1]
        ])
        return WTSeries(data=pd.concat([self.data, wtseries.data]))

    def write(self, dirpath: Path, filename: str, group_file_key: str = None):
        filepath = dirpath.joinpath(filename).with_suffix(SUFFIX_FILE)
        filekey = "{primary}{secondary}".format(
            primary=self.__class__.__name__,
            secondary=(f"_{group_file_key}" if group_file_key else "")
        )
        self.data.to_hdf(filepath, key=filekey, mode="w")
        return filepath

    @staticmethod
    def read(filepath: Path, group_file_key: str = None) -> WTSeries:
        filekey = "{primary}{secondary}".format(
            primary=WTSeries.__name__,
            secondary=(f"_{group_file_key}" if group_file_key else "")
        )
        return WTSeries(pd.read_hdf(filepath.with_suffix(SUFFIX_FILE), key=filekey, mode="r"))
