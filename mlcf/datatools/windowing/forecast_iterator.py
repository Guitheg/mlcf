"""_summary_
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from mlcf.datatools.windowing.iterator import WindowIterator

from mlcf.datatools.windowing.tseries import WTSeries


__all__ = [
    "WindowForecastIterator"
]


class WindowForecastIterator():
    def __init__(
        self,
        w_iterator: WindowIterator,
        input_width: int,
        target_width: int,
        input_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
    ):
        self.__input_width: int = input_width
        self.__target_width: int = target_width
        self.__input_features: Optional[List[str]] = input_features
        self.__target_features: Optional[List[str]] = target_features
        self.__data: WindowIterator = w_iterator
        self.__features: List[str] = w_iterator.features
        self.__window_index: int = 0

        feature_in = [feature in self.features for feature in self.input_features]
        if not np.all(feature_in):
            raise AttributeError(
                "The given input features are not in the feature's iterator:" +
                f"{self.input_features[feature_in]}")

        feature_in = [feature in self.features for feature in self.target_features]
        if not np.all(feature_in):
            raise AttributeError(
                "The given target features are not in the feature's iterator:" +
                f"{self.target_features[feature_in]}")

        if self.input_width + self.target_width > self.data.width:
            raise ValueError(
                "The input width and the target width doesn't match with the " +
                f"window width of the given WindowIterator. ({self.input_width} " +
                f"+ {self.target_width} > {self.data.width})")

    @property
    def data(self):
        return self.__data

    @property
    def features(self):
        return self.__features

    @property
    def input_features(self):
        if self.__input_features is None:
            return self.features
        return self.__input_features

    @property
    def target_features(self):
        if self.__target_features is None:
            return self.features
        return self.__target_features

    @property
    def target_width(self):
        return self.__target_width

    @property
    def input_width(self):
        return self.__input_width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]
        w_input = window.iloc[:self.input_width]
        w_target = window.iloc[-self.target_features:]
        return w_input[self.input_features], w_target[self.target_features]

    def __iter__(self):
        return self

    def __next__(self):
        if self.__window_index == len(self):
            self.__window_index = 0
            raise StopIteration
        item = self[self.__window_index]
        self.__window_index += 1
        return item
