
import pytest
import pandas as pd
from mlcf.datatools.utils import (
    labelize,
    binarize,
    split_pandas,
    split_train_val_test,
    subset_selection
)


@pytest.mark.parametrize(
    "test_input, expected_exception",
    [
        ({"column": "a", "labels": 10.1}, TypeError),
        ({"column": "a", "labels": (1, 2, 3)}, TypeError),
    ]
)
def test_binarize_exception(test_input, expected_exception):
    df = pd.DataFrame({"a": [-10, -3, -2, -1, -0.5, -0.01, 0, 0.01, 0.5, 1, 2, 3, 10]})

    with pytest.raises(expected_exception):
        binarize(data=df, **test_input)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {"column": "a"},
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
        (
            {"column": "a", "labels": [-1, 1]},
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1]),
        (
            {"column": "a", "labels": [-1, 1], "include_sep": False},
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]
        ),
        (
            {"column": "a", "labels": [-1, 1], "sep_value": 3},
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1]
        ),
    ]
)
def test_binarize(test_input, expected):
    df = pd.DataFrame({"a": [-10, -3, -2, -1, -0.5, -0.01, 0, 0.01, 0.5, 1, 2, 3, 10]})

    data_labelized = binarize(data=df, **test_input)
    assert list(data_labelized["a_label"]) == expected


@pytest.mark.parametrize(
    "test_input, expected_exception",
    [
        ({"column": "a", "labels": 0}, ValueError),
        ({"column": "a", "labels": 10.1}, TypeError),
        ({"column": "a", "labels": (1, 2, 5, 6), "bounds": (1, 2, 3)}, TypeError),
    ]
)
def test_labelize_exception(test_input, expected_exception):
    df = pd.DataFrame({"a": [-4, -3, -2, -1, -0.5, -0.01, 0, 0.01, 0.5, 1, 2, 3, 4]})
    with pytest.raises(expected_exception):
        labelize(data=df, **test_input)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {"column": "a", "labels": 2},
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
        (
            {"column": "a", "labels": [-1, 1]},
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1]),
        (
            {"column": "a", "labels": [-1, 1], "bounds": (-2, 2)},
            ["-inf", "-inf", -1, -1, -1, -1, 1, 1, 1, 1, "+inf", "+inf", "+inf"]
        ),
        (
            {"column": "a", "labels": [-1, 1, 2], "bounds": (-2, 2)},
            ['-inf', '-inf', -1, -1, 1, 1, 1, 1, 1, 2, '+inf', '+inf', '+inf']
        ),
        (
            {"column": "a", "labels": 4, "bounds": (-2, 2)},
            ['-inf', '-inf', 0, 1, 1, 1, 2, 2, 2, 3, '+inf', '+inf', '+inf']
        ),
        (
            {"column": "a", "labels": 5},
            [0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4]
        ),
    ]
)
def test_labelize(test_input, expected):
    df = pd.DataFrame({"a": [-4, -3, -2, -1, -0.5, -0.01, 0, 0.01, 0.5, 1, 2, 3, 4]})

    data_labelized = labelize(data=df, **test_input)
    assert list(data_labelized["a_label"]) == expected


@pytest.mark.parametrize(
    "test_input, expected_exception",
    [
        ({"prop_snd_elem": "a"}, TypeError),
        ({"prop_snd_elem": -0.1}, ValueError),
        ({"prop_snd_elem": 1.1}, ValueError),
    ]
)
def test_split_pandas_exception(test_input, expected_exception):
    df = pd.DataFrame({"a": [-4, -3, -2, -1, -0.5, -0.01, 0, 0.01, 0.5, 1, 2, 3, 4]})
    with pytest.raises(expected_exception):
        split_pandas(data=df, **test_input)


@pytest.mark.parametrize(
    "data_selection, test_input, expect_elem_a, expect_elem_b",
    [
        ("full", {"prop_snd_elem": 0.5}, 7674, 7674),
        ("full", {"prop_snd_elem": 0.0}, 15348, 0),
        ("full", {"prop_snd_elem": 1.0}, 0, 15348),
        ("full", {"prop_snd_elem": 0.3}, 10744, 4604),
        ("full", {"prop_snd_elem": 0.001}, 15333, 15),
        ("empty", {"prop_snd_elem": 0.5}, 0, 0)
    ]
)
def test_split_pandas(data_selection, test_input, expect_elem_a, expect_elem_b, ohlcv_btc):
    data = {
        "empty": pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]).rename_axis("date"),
        "full": ohlcv_btc
    }
    data_a, data_b = split_pandas(data=data[data_selection], **test_input)
    assert len(data_a) == expect_elem_a
    assert len(data_b) == expect_elem_b
    assert len(data_a) + len(data_b) == len(data[data_selection])
    assert list(data_a.columns) == list(data_b.columns)
    assert list(data[data_selection].columns) == list(data_a.columns)
    assert data_a.index.name == data_b.index.name
    assert data[data_selection].index.name == data_a.index.name


@pytest.mark.parametrize(
    "data_selection, test_input, expect_train, expect_val, expect_test",
    [
        ("full", {"prop_val_test": 0.5, "prop_val": 0.5}, 7674, 3837, 3837),
        ("full", {"prop_val_test": 0.5, "prop_val": 0.2}, 7674, 1535, 6139),
        ("full", {"prop_val_test": 0.0, "prop_val": 0.0}, 15348, 0, 0),
        ("full", {"prop_val_test": 0.0, "prop_val": 1.0}, 15348, 0, 0),
        ("full", {"prop_val_test": 1.0, "prop_val": 0.0}, 0, 0, 15348),
        ("full", {"prop_val_test": 1.0, "prop_val": 1.0}, 0, 15348, 0),
        ("full", {"prop_val_test": 0.001, "prop_val": 0.001}, 15333, 1, 14),
        ("empty", {"prop_val_test": 0.5, "prop_val": 0.0}, 0, 0, 0)
    ]
)
def test_split_train_val_test(
    data_selection,
    test_input,
    expect_train,
    expect_val,
    expect_test,
    ohlcv_btc
):
    data = {
        "empty": pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]).rename_axis("date"),
        "full": ohlcv_btc
    }
    train, val, test = split_train_val_test(data=data[data_selection], **test_input)
    assert len(train) == expect_train
    assert len(val) == expect_val
    assert len(test) == expect_test
    assert len(train) + len(val) + len(test) == len(data[data_selection])
    assert list(train.columns) == list(val.columns)
    assert list(train.columns) == list(test.columns)
    assert list(data[data_selection].columns) == list(train.columns)
    assert train.index.name == val.index.name
    assert train.index.name == test.index.name
    assert data[data_selection].index.name == train.index.name


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": []},
            []
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [-1]},
            []
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [0]},
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [1, 0]},
            [1]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [-1, 0]},
            [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [0, 1]},
            [10]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [0, -1]},
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [-1, 0, -1]},
            [2, 3, 4, 5, 6, 7, 8, 9]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [1, 0, 1]},
            [1, 10]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [5]},
            [1, 2, 3, 4, 5]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [2, -2, 2, -2, 2]},
            [1, 2, 5, 6, 9, 10]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [2, -2, 0, -2, 2]},
            [1, 2, 5, 6, 9, 10]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [2, -2, 2, 0, 2]},
            [1, 2, 5, 6, 9, 10]
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [2, -3, -3, 2]},
            [1, 2, 9, 10]
        ),
    ]
)
def test_subset_selection(test_input, expected):
    assert expected == subset_selection(**test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [-1, 0, 1]}
        ),
        (
            {"element_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "selection_list": [5, 0, 2, 0, 1]}
        ),
    ]
)
def test_subset_selection_exception(test_input):
    with pytest.raises(ValueError):
        subset_selection(**test_input)
