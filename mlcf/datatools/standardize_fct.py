"""_summary_
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

__all__ = [
    "StandardisationFct",
    "ClassicStd",
    "MinMaxStd",
    "standardize",
    "standardize_fit_transform"
]


class IdentityTransform():
    def transform(self, x: pd.Series):
        return x.loc[:]


class StandardisationFct():
    def __init__(self):
        self.std = IdentityTransform()

    def partial_fit(self, data: pd.Series):
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        series = data.copy()
        series.loc[:] = self.std.transform(series)
        return series


class ClassicStd(StandardisationFct):
    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True
    ):
        super(ClassicStd, self).__init__()
        self.std = StandardScaler(with_mean=with_mean, with_std=with_std)

    def partial_fit(self, data: pd.Series):
        self.std.partial_fit(data)


class MinMaxStd(StandardisationFct):
    def __init__(
        self,
        minmax: Tuple[float, float] = (0, 1),
        feature_range: Tuple[float, float] = (0, 1)
    ):
        super(MinMaxStd, self).__init__()
        self.std = MinMaxScaler(feature_range=feature_range)
        self.std.fit([[minmax[0]], [minmax[1]]])


# TODO: (refactoring) list comprehension for data_transformed or verify if it's inplace or not
def standardize(
    fit_data: Union[pd.DataFrame, List[pd.DataFrame]],
    transform_data: Union[pd.DataFrame, List[pd.DataFrame]],
    std_by_feature: Optional[Dict[str, StandardisationFct]],
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    if not std_by_feature:
        return transform_data
    if isinstance(fit_data, pd.DataFrame):
        if not fit_data.empty:
            for feature, std_obj in std_by_feature.items():
                std_obj.partial_fit(fit_data[feature])
    elif isinstance(fit_data, List[pd.DataFrame]):
        for data in fit_data:
            if not data.empty:
                for feature, std_obj in std_by_feature.items():
                    std_obj.partial_fit(data[feature])
    else:
        raise Exception(f"Bad fit_data ({type(fit_data)}) type in standardize")

    if isinstance(transform_data, pd.DataFrame):
        if not transform_data.empty:
            for feature, std_obj in std_by_feature.items():
                transform_data[feature] = std_obj.transform(transform_data[feature])
    elif isinstance(transform_data, List[pd.DataFrame]):
        for data in transform_data:
            if not data.empty:
                for feature, std_obj in std_by_feature.items():
                    data[feature] = std_obj.transform(data[feature])
    else:
        raise Exception(f"Bad transform_data ({type(transform_data)}) type in standardize")

    return transform_data


def standardize_fit_transform(
    fit_transform_data: Union[pd.DataFrame, List[pd.DataFrame]],
    std_by_feature: Optional[Dict[str, StandardisationFct]],
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    return standardize(fit_transform_data, fit_transform_data, std_by_feature)
