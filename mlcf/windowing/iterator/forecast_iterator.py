"""Window Forecast Iterator Module.

This module provide WindowForecastIterator which allows us to iterate over a WindowIterator with a
input window and a target window.

    Example:

    .. code-block:: python

        # This class allow us to iterate over a WTSeries but the iteration
        # (__getitem__) give us a tuple of 2

        from mlcf.datatools.windowing.forecast_iterator import WindowForecastIterator

        data_train = WindowForecastIterator(
            wtseries,
            input_width=29,
            target_width=1,  # The sum of the input_width and target_width must not exceed the
                             # window width
                             # of the wtseries
            input_features=["close", "adx"],
            target_features=["return"]
        )
        for window in data_train:
            window_input, window_target = window
            pass
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from mlcf.datatools.standardisation import StandardisationModule, standardize
from mlcf.windowing.iterator.iterator import WindowIterator

# TODO (doc) correct English

__all__ = [
    "WindowForecastIterator"
]


class WindowForecastIterator():
    """This class use a WindowIterator in order to returns an input and a target window instead of a
    single window.
    """
    def __init__(
        self,
        w_iterator: WindowIterator,
        input_width: int,
        target_width: int,
        input_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        std_by_feature: Optional[Dict[str, StandardisationModule]] = None
    ):
        """Create a WindowForecastIterator which give for each item an input window and
        a target window.
        The sum of the input window width and the target window width must not exceed
        the WindowIterator window width.
        If the sum of input window width and the target window width is not equal to the
        WindowIterator window width then the difference between the sum and the window with
        is considered as the offset.

        Args:
            w_iterator (WindowIterator): The window iterator used

            input_width (int): The width of the input window

            target_width (int): The width of the target window

            input_features (Optional[List[str]], optional): The list of the selected features
                for the input window. If None, then every feature of the WindowIterator are taken.
                Defaults to None.

            target_features (Optional[List[str]], optional): The list of selected features
                for the target window. If None, then every feature of the WindowIterator are taken.
                Defaults to None.

            std_by_feature (Optional[Dict[str, StandardisationModule]], optional):
                A dictionary prodiving the standardisation to be applied on each column.
                Here, the standardisation is done independently on each window.
                The fit is done only on the window input.
                The dictionary format must be as following:
                {string -> :py:class:`StandardisationModule
                <mlcf.datatools.standardisation.StandardisationModule>`}.
                The key must correspond to a column name (a feature) of the data frame.
                The value is any object inheriting from the
                :py:class:`StandardisationModule
                <mlcf.datatools.standardisation.StandardisationModule>` class.

        Raises:
            AttributeError: If one of the feature gived in the input or target feature doesn't
                belong to the features of the WindowIterator
            ValueError: If the sum of the input width and the target width is greater than the
                WindowIterator window width.
        """
        self.__input_width: int = input_width
        self.__target_width: int = target_width
        self.__input_features: Optional[List[str]] = input_features
        self.__target_features: Optional[List[str]] = target_features
        self.__data: WindowIterator = w_iterator
        self.__features: List[str] = w_iterator.features
        self.__window_index: int = 0
        self.__std_by_feature: Optional[Dict[str, StandardisationModule]] = std_by_feature

        feature_in = [feature in self.features for feature in self.input_features]
        if not np.all(feature_in):
            raise AttributeError(
                "The given input features are not in the feature's iterator:" +
                f"{[feat for feat, isin in zip(self.input_features, feature_in) if not isin]}")

        feature_in = [feature in self.features for feature in self.target_features]
        if not np.all(feature_in):
            raise AttributeError(
                "The given target features are not in the feature's iterator:" +
                f"{[feat for feat, isin in zip(self.target_features, feature_in) if not isin]}")

        if self.input_width + self.target_width > self.data.width:
            raise ValueError(
                "The input width and the target width doesn't match with the " +
                f"window width of the given WindowIterator. ({self.input_width} " +
                f"+ {self.target_width} > {self.data.width})")

    @property
    def std_by_feature(self) -> Optional[Dict[str, StandardisationModule]]:
        """A dictionary prodiving the standardisation to be applied on each column.
        Here, the standardisation is done independently on each window.
        The fit is done only on the window input.
        The dictionary format must be as following:
        {string -> :py:class:`StandardisationModule
        <mlcf.datatools.standardisation.StandardisationModule>`}.
        The key must correspond to a column name (a feature) of the data frame.
        The value is any object inheriting from the
        :py:class:`StandardisationModule
        <mlcf.datatools.standardisation.StandardisationModule>` class.
        """

        return self.__std_by_feature

    @property
    def data(self) -> WindowIterator:
        """ A WindowIterator on which the window will be taken from."""

        return self.__data

    @property
    def features(self) -> List[str]:
        """The list of features of the windows of {data}"""

        return self.__features

    @property
    def input_features(self) -> List[str]:
        """The list of the selected features for the input window"""

        if not self.__input_features:
            return self.features
        return self.__input_features

    @property
    def target_features(self) -> List[str]:
        """The list of the selected features for the target window"""

        if not self.__target_features:
            return self.features
        return self.__target_features

    @property
    def target_width(self) -> int:
        """The width of the target window"""

        return self.__target_width

    @property
    def input_width(self) -> int:
        """The width of the input window"""

        return self.__input_width

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Given an index return the corresponding pair of input window and target window.

        Args:
            idx (int): The index to selection the desired pair

        Returns:
            Tuple[np.ndarray, np.ndarray]: A pair of input window and target window
        """

        window: pd.DataFrame = self.data[idx]
        w_input = window.iloc[:self.input_width][self.input_features].copy()
        w_target = window.iloc[-self.target_width:][self.target_features].copy()
        if self.std_by_feature:
            standardize(w_input, [w_input, w_target], self.std_by_feature, std_fct_save=False)
        return w_input, w_target

    def __iter__(self) -> WindowForecastIterator:
        self.__window_index = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.__window_index < len(self):
            item = self[self.__window_index]
            self.__window_index += 1
            return item
        else:
            raise StopIteration
