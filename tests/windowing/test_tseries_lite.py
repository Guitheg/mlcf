from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from mlcf.datatools.data_intervals import DataIntervals
from mlcf.windowing.filtering import LabelBalanceFilter
from mlcf.windowing.iterator.tseries_lite import DataEmptyException
from mlcf.datatools.standardisation import ClassicStd
from mlcf.windowing.iterator.tseries_lite import WTSeriesLite


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ({"window_width": 20, "window_step": 1},
         {"length": 15328, "first_window": lambda data: data["return"].iloc[0:20].index}),
        ({"window_width": 30, "window_step": 1},
         {"length": 15318, "first_window": lambda data: data["return"].iloc[0:30].index}),
        ({"window_width": 30, "window_step": 2},
         {"length": 7659, "first_window": lambda data: data["return"].iloc[0:30].index}),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            },
            {
                "length": 7659,
                "first_window": lambda data: data["return"].iloc[0:30].index
            }
        ),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            },
            {
                "length": 7659,
                "first_window": lambda data: data["return"].iloc[0:30].index
            }
        ),
        (
            {
                "window_width": 300,
                "window_step": 2,
                "selected_columns": ["close", "return"],
                "window_filter": LabelBalanceFilter("label")
            },
            {
                "length": 2616,
                "first_window": lambda data: data["return"].iloc[68:368].index
            }
        )
    ]
)
def test_wtseries_lite(ohlcvrl_btc, test_input, expected):

    data = ohlcvrl_btc.copy()

    wtseries = WTSeriesLite.create_wtseries_lite(data, **test_input)
    assert len(wtseries) == expected["length"]
    assert np.all(wtseries[0]["return"].index == expected["first_window"](data))
    if "selected_columns" in test_input:
        assert list(wtseries.features) == test_input["selected_columns"]
        assert wtseries.ndim == len(test_input["selected_columns"])
    else:
        assert wtseries.ndim == len(list(data.columns))
    if "std_by_feature" in test_input:
        for feature in test_input["std_by_feature"]:
            assert np.round(wtseries[0][feature].mean(), 4) == 0.0
            assert np.round(wtseries[0][feature].std(), 1) == 1.0
    assert wtseries.n_window == expected["length"]
    assert wtseries.width == test_input["window_width"]


@pytest.mark.parametrize(
    "data_selection, test_input, expected_exception",
    [
        (
            "few",
            {"window_width": 101, "window_step": 1},
            DataEmptyException
        ),
        (
            "empty",
            {"window_width": 20, "window_step": 1},
            DataEmptyException
        )
    ]
)
def test_create_wtseries_lite_exception(
    ohlcvra_btc,
    data_selection,
    test_input,
    expected_exception
):
    data = {
        "few": ohlcvra_btc.iloc[:100],
        "empty": pd.DataFrame(columns=ohlcvra_btc.columns)
    }
    with pytest.raises(expected_exception):
        WTSeriesLite.create_wtseries_lite(data[data_selection], **test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        ({"window_width": 20, "window_step": 1}),
        ({"window_width": 30, "window_step": 10}),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            }
        ),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            }
        )
    ]
)
def test_merge(ohlcvra_btc, test_input):
    data_intervals = DataIntervals.create_list_interval(ohlcvra_btc, n_intervals=2)
    wtseries_1 = WTSeriesLite.create_wtseries_lite(data_intervals[0], **test_input)
    wtseries_2 = WTSeriesLite.create_wtseries_lite(data_intervals[0], **test_input)
    wtseries = wtseries_1.merge(wtseries_2)
    assert np.all(wtseries[len(wtseries_1)].values == wtseries_2[0].values)
    assert len(wtseries_1) + len(wtseries_2) == len(wtseries)


@pytest.mark.parametrize(
    "test_input, group_key",
    [
        ({"window_width": 20, "window_step": 1}, None),
        ({"window_width": 30, "window_step": 10}, None),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            },
            "train"
        ),
    ]
)
def test_write(ohlcvr_btc, test_input, group_key, tmp_path: Path):
    wtseries = WTSeriesLite.create_wtseries_lite(ohlcvr_btc, **test_input)
    userdir = tmp_path.joinpath("userdir")
    userdir.mkdir()
    file_name = "datasetwts"
    file_path = wtseries.write(userdir, file_name, group_key)
    assert file_path.is_file()
    assert np.all(
        pd.read_hdf(
            file_path, key=WTSeriesLite._get_dataset_namespace("data", group_key)).values ==
        wtseries.data.values
    )
    assert np.all(
        pd.read_hdf(
            file_path, key=WTSeriesLite._get_dataset_namespace("index", group_key)).values ==
        wtseries.index_array.values
    )


@pytest.mark.parametrize(
    "test_input, group_key",
    [
        ({"window_width": 20, "window_step": 1}, None),
        ({"window_width": 30, "window_step": 10}, None),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            },
            "train"
        ),
    ]
)
def test_read(ohlcvr_btc, test_input, group_key, tmp_path: Path):
    wtseries = WTSeriesLite.create_wtseries_lite(ohlcvr_btc, **test_input)
    userdir = tmp_path.joinpath("userdir")
    userdir.mkdir()
    file_name = "datasetwts"
    file_path = wtseries.write(userdir, file_name, group_key)
    assert file_path.is_file()
    wtseries_read = WTSeriesLite.read(file_path, group_key)
    assert np.all(wtseries.data.values == wtseries_read.data.values)
    assert np.all(wtseries.index_array.values == wtseries_read.index_array.values)
