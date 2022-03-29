"""Window Filtering Module.

It provides classes that are used during the windowing process,
with the purpose of filtering windows according to certain conditions.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


__all__ = [
    "WindowFilter"
]


class WindowFilter(ABC):
    """ Window Filtering Class.
    It allows us to perform window filtering according to certain conditions
    during the windowing process.
    This class is compatible with
    :py:class:`DataIntevals <mlcf.datatools.data_intervals.DataIntervals>`
    and :py:class:`WTSeries <mlcf.windowing.iterator.tseries.WTSeries>`.
    Classes whose purpose is to filter windows must be inherited from this class.
    """

    @abstractmethod
    def __call__(
        self,
        data: pd.DataFrame,
        index_array: np.ndarray,
        *args, **kwargs
    ) -> pd.DataFrame:
        pass
