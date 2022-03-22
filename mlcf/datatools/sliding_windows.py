"""_summary_
"""

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from mlcf.datatools.standardize_fct import StandardisationFct


__all__ = [
    "predicate_windows_step",
    "predicate_balance_tag",
    "data_windowing"
]


def predicate_windows_step(
    data,
    idx: List[int],
    step_tag_name: str
) -> bool:
    return data.loc[data.index[idx[-1]], step_tag_name]


def predicate_balance_tag(
    data,
    idx: List[int],
    balance_tag_name: str,
    step_tag_name: str
) -> bool:
    return (
        data.loc[data.index[idx[-1]], balance_tag_name] and
        data.loc[data.index[idx[-1]], step_tag_name]
    )


# TODO: (opti) parrallelize standardization ?
def data_windowing(
    dataframe: pd.DataFrame,
    window_width: int,
    window_step: int,
    selected_columns: Optional[List[str]] = None,
    predicate_row_selection: Optional[Callable] = None,
    std_by_feature: Optional[Dict[str, StandardisationFct]] = None
) -> List[pd.DataFrame]:
    data = dataframe.copy()
    data_columns = list(data.columns)
    data["__index"] = np.arange(len(data))
    if len(data) == 0 or len(data) < window_width:
        return [pd.DataFrame(columns=data_columns).rename_axis(data.index.name)]

    # Slid window on all data
    windowed_data = sliding_window_view(
        data,
        window_shape=(window_width, len(data.columns))
    ).reshape((-1, window_width, len(data.columns)))
    if not windowed_data.flags["WRITEABLE"]:
        windowed_data = windowed_data.copy()

    # Reshape windowed data
    windowed_data_shape: Tuple[int, int, int] = (-1, window_width, len(data.columns))
    windowed_data = np.reshape(windowed_data, newshape=windowed_data_shape)

    index_data = windowed_data[:, :, list(data.columns).index("__index")]

    # Select rows and windows
    if predicate_row_selection is None:
        windowed_data = windowed_data[::window_step]
    else:
        predicate = partial(predicate_row_selection, data)
        with Pool(processes=cpu_count()) as pl:
            windowed_data = windowed_data[pl.map(predicate, [idx for idx in index_data])]
    index_data = windowed_data[:, :, list(data.columns).index("__index")]

    # Set the indexes
    window_index = np.mgrid[
        0: windowed_data.shape[0]: 1,
        0:window_width: 1
    ][0].reshape(-1, 1)

    # Select columns
    if selected_columns is None:
        selected_columns = data_columns
    idx_selected_columns = [data_columns.index(col) for col in selected_columns]

    # Standardization
    if std_by_feature:
        for feature, std_object in std_by_feature.items():
            datafit = windowed_data[:, :, data_columns.index(feature)].T
            std_object.fit(datafit)
            windowed_data[:, :, data_columns.index(feature)] = std_object.transform(datafit).T
    windowed_data = windowed_data[:, :, idx_selected_columns+[list(data.columns).index("__index")]]
    # Make list of window (dataframe)

    multi_dataframe_windows = pd.DataFrame(
        np.concatenate([windowed_data.reshape(-1, len(selected_columns)+1), window_index], axis=1),
        columns=selected_columns + ["TimeIndex", "WindowIndex"]
    ).set_index(["WindowIndex", "TimeIndex"])
    multi_dataframe_windows.index = multi_dataframe_windows.index.set_levels(
        [
            multi_dataframe_windows.index.levels[0],
            pd.to_datetime(data.iloc[multi_dataframe_windows.index.levels[1]].index)
        ]
    )
    return multi_dataframe_windows
