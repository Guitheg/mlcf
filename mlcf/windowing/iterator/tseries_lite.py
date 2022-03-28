
from __future__ import annotations
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mlcf.datatools.standardisation import StandardisationModule, copy_std_feature_dict
from mlcf.windowing.iterator import WindowIterator
from mlcf.windowing.filtering import WindowFilter
from mlcf.windowing.iterator.tseries import TIME_INDEX_NAME, DataEmptyException


class WTSeriesLite(WindowIterator):

    def __init__(
        self,
        data: pd.DataFrame,
        index_array: np.ndarray,
        std_by_feature: Optional[Dict[str, StandardisationModule]] = None
    ):
        self._data: pd.DataFrame = data
        self._index_array: np.ndarray = index_array
        self._std_by_feature: Optional[Dict[str, StandardisationModule]] = std_by_feature

    @property
    def data(self):
        return self._data

    @property
    def index_array(self):
        return self._index_array

    @property
    def std_by_feature(self):
        return self._std_by_feature

    @classmethod
    def create_wtseries_lite(
        self,
        data: pd.DataFrame,
        window_width: int,
        window_step: int,
        selected_columns: Optional[List[str]] = None,
        window_filter: Optional[WindowFilter] = None,
        std_by_feature: Optional[Dict[str, StandardisationModule]] = None
    ) -> WTSeriesLite:

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
            dataframe=dataframe[selected_columns],
            index_array=index_data,
            std_by_feature=std_by_feature
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
        return self.index_array.shape[0]

    @property
    def width(self) -> int:
        """The window width.

        Returns:
            int: The window width
        """
        return self.index_array.shape[1]

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
        window = self.data.iloc[self.index_array[idx]]

        if self.std_by_feature:
            for feature, std_obj in self.std_by_feature:
                window[feature].loc[:] = std_obj.fit_transform(window[feature])

        return window

    def copy(self) -> WTSeriesLite:
        """Return a copy of this WTSeries.

        Returns:
            WTSeries: A copy of this WTSeries
        """
        return WTSeriesLite(
            data=self.data.copy(),
            index_array=self.index_array.copy(),
            std_by_feature=copy_std_feature_dict(self.std_by_feature)
        )
