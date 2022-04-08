
import numpy as np
import pytest
from mlcf.datatools.standardisation import (
    ClassicStd,
    MinMaxStd,
    standardize,
    standardize_fit_transform)


def test_standardize_fit_transform(ohlcvra_btc):
    std_by_feature = {
        "close": ClassicStd(),
        "adx": MinMaxStd(minmax=(0, 100))
    }
    ret = standardize_fit_transform(ohlcvra_btc, std_by_feature)
    assert np.all(ret.values == ohlcvra_btc.values)
    assert np.round(ohlcvra_btc["close"].mean(), 4) == 0.0
    assert np.round(ohlcvra_btc["close"].std(), 4) == 1.0
    assert ohlcvra_btc["adx"].max() <= 1.0
    assert ohlcvra_btc["adx"].min() >= 0.0
    assert np.round(std_by_feature["close"].std.mean_[0], 4) == 8399.1056
    assert np.round(std_by_feature["close"].std.scale_[0], 4) == 955.8869


@pytest.mark.parametrize(
    "fit_data_col, transform_data_col, test_input, expected_exception",
    [
        ("close", ["close", "adx"], {}, TypeError),
        (["close", "adx"], "close", {}, TypeError),
        (["close", "adx"], ["close", "adx"], {"inplace": False}, NotImplementedError),
        (["close", "adx"], ["close", "adx"], {"std_fct_save": False}, AttributeError),
    ]
)
def test_standardize_exception(
    ohlcvra_btc,
    fit_data_col,
    transform_data_col,
    test_input,
    expected_exception
):
    fit_data = ohlcvra_btc[fit_data_col]
    transform_data = ohlcvra_btc[transform_data_col]
    std_by_feature = {
        "close": ClassicStd(),
        "adx": MinMaxStd(minmax=(0, 100))
    }
    with pytest.raises(expected_exception):
        standardize(fit_data, transform_data, std_by_feature, **test_input)
        std_by_feature.std.mean_


def test_standardize(ohlcvra_btc):
    std_by_feature = {
        "close": ClassicStd(),
        "adx": MinMaxStd(minmax=(0, 100)),
        "unknown": ClassicStd()
    }
    data = ohlcvra_btc.copy()
    train = data.iloc[:10000]
    test = data.iloc[10000:]
    data_fit = [train]
    data_transform = [test]
    standardize(data_fit, data_transform, std_by_feature)
    assert np.all(train.values == ohlcvra_btc.iloc[0:10000].values)
    assert np.round(test["close"].mean(), 4) == 1.0984
    assert np.round(test["close"].std(), 4) == 1.0253
    assert test["adx"].max() <= 1.0
    assert test["adx"].min() >= 0.0
    assert np.round(std_by_feature["close"].std.mean_[0], 4) == 8078.2446
    assert np.round(std_by_feature["close"].std.scale_[0], 4) == 841.2266
