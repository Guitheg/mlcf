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
from typing import Callable, Dict, List, Optional, Tuple, Union
from mlcf.datatools.standardisation import StandardisationModule, standardize
from mlcf.windowing.iterator.iterator import WindowIterator
from mlcf.datatools.utils import subset_selection

# TODO (doc) correct English

__all__ = [
    "WindowForecastIterator"
]


# TODO (doc) explain index_selection_mode
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
        std_by_feature: Optional[Dict[str, StandardisationModule]] = None,
        index_selection_mode: Optional[Union[str, List[int]]] = None,
        transform_input: Optional[Callable] = None,
        transform_target: Optional[Callable] = None
    ):
        """Create a WindowForecastIterator which give for each item an input window and
        a target window.
        The sum of the input window width and the target window width must not exceed
        the WindowIterator window width.
        If the sum of input window width and the target window width is not equal to the
        WindowIterator window width then the difference between the sum and the window with
        is the number of ignored rows/index.

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

            index_selection_mode (Union[str, List[int]], optional): The index selection mode can be
                a predefined index selection mode that is indicated by these keys: 'default', 'left'
                or 'right'. It can also be a custom index selection mode defined by a list of
                integers. The list of integers is called 'subset selection list' (see
                :py:func:`subset_selection <mlcf.datatools.utils.subset_selection>`
                for more information).
                A subset selection list is a list interpreted by the subset_selection function that
                takes only positive and negative integers and indicates whether or not we are
                selecting or ignoring a subset of elements in a list (positive integers indicate
                the number of elements we are selecting and negative integers indicate the number
                of elements we are ignoring - the subset selection list is order sensitive).
                Here, the list of elements corresponds to the index of a window itself.

                .. code-block:: python

                    # called in the __init__:
                    selected_index = subset_selection(
                        list(np.arange(self.data.width)),
                        self.__index_selection_mode
                    )

                Here some further information about pre-defined index selection mode:

                    - 'default':
                        correspond to the subset selection list
                        [input_width, 0, target_width]. Here, it correspond to an offset between
                        the input and the target window.
                    - 'left':
                        correspond to [0, input_width, target_width].  Here, it correspond to an
                        offset on the left before the input and target window.
                    - 'right':
                        correspond to [input_width, target_width, 0].  Here, it correspond to an
                        offset on the right after the input and target window.

                Defaults to 'default'.

            transform_input (Callable, optional): A function called in __getitem__ that takes an
                input window as input and returns its transformation (type, shape, etc.).
                Defaults to None.

            transform_target (Callable, optional): A function called in __getitem__ that takes a
                target window as input and returns its transformation (type, shape, etc.).
                Defaults to None.

        Raises:
            AttributeError: If one of the feature gived in the input or target feature doesn't
                belong to the features of the WindowIterator
            ValueError: If the sum of the input width and the target width is greater than the
                WindowIterator window width.
        """

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

        self.__index_selection_mode: List[int] = [0]
        if isinstance(index_selection_mode, str):
            if index_selection_mode == "default":
                self.__index_selection_mode = [input_width, 0, target_width]
            if index_selection_mode == "left":
                self.__index_selection_mode = [0, input_width, target_width]
            if index_selection_mode == "right":
                self.__index_selection_mode = [input_width, target_width, 0]
        elif isinstance(index_selection_mode, list) and np.all(
            [isinstance(i, int) for i in index_selection_mode]
        ):
            self.__index_selection_mode = index_selection_mode

        selected_window_width = input_width + target_width
        if selected_window_width > self.data.width:
            raise ValueError(
                "The sum of the input width and the target width is greater than the window width" +
                f" ({input_width} + {target_width} > {self.data.width}).")

        if np.sum(np.abs(self.index_selection_mode)) > self.data.width:
            raise ValueError(
                "The absolute sum of the index selection mode is greater than the window width.")

        selected_index = subset_selection(
            list(np.arange(self.data.width)),
            self.__index_selection_mode)

        if len(selected_index) != input_width + target_width:
            raise ValueError(
                "The number of selected index is differents than the window width."
            )

        self.__input_index: List[int] = selected_index[:input_width]
        self.__target_index: List[int] = selected_index[-target_width:]

        self.__transform_input = transform_input
        self.__transform_target = transform_target

    @property
    def index_selection_mode(self):
        return self.__index_selection_mode

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

        if self.__input_features is None:
            return self.features
        return self.__input_features

    @property
    def target_features(self) -> List[str]:
        """The list of the selected features for the target window"""

        if self.__target_features is None:
            return self.features
        return self.__target_features

    @property
    def target_index(self) -> List[int]:
        """Relative target index by window"""

        return self.__target_index

    @property
    def input_index(self) -> List[int]:
        """Relative input index by window"""

        return self.__input_index

    @property
    def transform_input(self) -> Callable:
        """This function is called in __getitem__ before returning the input window."""
        if self.__transform_input:
            return self.__transform_input
        else:
            return lambda x: x

    @property
    def transform_target(self) -> Callable:
        """This function is called in __getitem__ before returning the target window."""
        if self.__transform_target:
            return self.__transform_target
        else:
            return lambda y: y

    def set_transform_input(self, transform_input: Callable):
        if callable(transform_input):
            self.__transform_target = transform_input
        else:
            raise ValueError("The transform paramater must be a callable (a function)")

    def set_transform_target(self, transform_target: Callable):
        if callable(transform_target):
            self.__transform_target = transform_target
        else:
            raise ValueError("The transform paramater must be a callable (a function)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Given an index return the corresponding pair of input window and target window.

        Args:
            idx (int): The index to selection the desired pair

        Returns:
            Tuple[np.ndarray, np.ndarray]: A pair of input window and target window
        """

        window: pd.DataFrame = self.data[idx]
        w_input = window.iloc[self.input_index][self.input_features].copy()
        w_target = window.iloc[self.target_index][self.target_features].copy()
        if self.std_by_feature:
            standardize(w_input, [w_input, w_target], self.std_by_feature, std_fct_save=False)
        return self.transform_input(w_input), self.transform_target(w_target)

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
