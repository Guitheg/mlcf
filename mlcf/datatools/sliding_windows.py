"""_summary_
"""

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from time import perf_counter
from mlcf.datatools.standardize_fct import StandardisationFct, standardize_fit_transform


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


def _build_standardize_window(window_idx, data, selected_columns, std_by_feature):
    window, idx = window_idx
    return standardize_fit_transform(
        fit_transform_data=pd.DataFrame(
            window,
            index=data.iloc[idx].index,
            columns=selected_columns
        ),
        std_by_feature=std_by_feature,
        std_fct_save=False
    )


# TODO: (opti) parrallelize ?
# TODO: (opti) optimization of the standardization: memory of the previous window
def data_windowing(
    dataframe: pd.DataFrame,
    window_width: int,
    window_step: int,
    selected_columns: Optional[List[str]] = None,
    predicate_row_selection: Optional[Callable] = None,
    std_by_feature: Optional[Dict[str, StandardisationFct]] = None
) -> List[pd.DataFrame]:
    # t0 = perf_counter()
    data = dataframe.copy()
    data_columns = list(data.columns)
    data["__index"] = np.arange(len(data))
    if len(data) == 0 or len(data) < window_width:
        return [pd.DataFrame(columns=data_columns).rename_axis(data.index.name)]
    # t1 = perf_counter()
    # Slid window on all data
    windowed_data: np.ndarray = sliding_window_view(
        data,
        window_shape=(window_width, len(data.columns))
    )
    # t2 = perf_counter()
    # Reshape windowed data
    windowed_data_shape: Tuple[int, int, int] = (-1, window_width, len(data.columns))
    windowed_data = np.reshape(windowed_data, newshape=windowed_data_shape)
    # t3 = perf_counter()
    index_data = windowed_data[:, :, list(data.columns).index("__index")]
    # t4 = perf_counter()
    # Select rows
    if predicate_row_selection is None:
        windowed_data = windowed_data[::window_step]
    else:
        # windowed_data = windowed_data[[
        #     predicate_row_selection(data, list(idx))
        #     for idx in index_data
        # ]]
        predicate = partial(predicate_row_selection, data)
        with Pool(processes=cpu_count()) as pl:
            windowed_data = windowed_data[pl.map(predicate, [idx for idx in index_data])]
    # t5 = perf_counter()
    # Select columns and rows
    if selected_columns is None:
        selected_columns = data_columns
    idx_selected_columns = [data_columns.index(col) for col in selected_columns]
    windowed_data = windowed_data[:, :, idx_selected_columns]
    # t6 = perf_counter()
    # Make list of window (dataframe)
    build_standardize_window = partial(
        _build_standardize_window,
        data=data,
        selected_columns=selected_columns,
        std_by_feature=std_by_feature)
    with Pool(processes=cpu_count()) as pl:
        list_windows: List[pd.DataFrame] = pl.map(
            build_standardize_window,
            zip(windowed_data, index_data)
        )

    # list_windows: List[pd.DataFrame] = [
    #     standardize_fit_transform(
    #         fit_transform_data=pd.DataFrame(
    #             window,
    #             index=data.iloc[idx].index,
    #             columns=selected_columns
    #         ),
    #         std_by_feature=std_by_feature,
    #         std_fct_save=False
    #     ) for window, idx in zip(windowed_data, index_data)
    # ]
    # t7 = perf_counter()
    # total = t7 - t0
    # init = t1 - t0
    # sliding = t2 - t1
    # reshape = t3 - t2
    # index = t4 - t3
    # rows = t5 - t4
    # col = t6 - t5
    # win = t7 - t6
    # print(init, sliding, reshape, index, rows, col, win, total)
    return list_windows
