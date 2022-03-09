import pytest
from torch import Tensor
import numpy as np

from mlcf.aitools.metrics import (
    mae,
    rmse,
    mape,
    smape,
    mase,
    rrse,
    rae,
    accu_neighborhood,
    oratio,
    uratio,
    stde,
    nrmse
)


@pytest.fixture
def y_values_one_dim():
    y_true = Tensor(np.array(
        [
            [2], [1]
        ]
    ))

    y_pred = Tensor(np.array(
        [
            [1], [2]
        ]
    ))
    return y_true, y_pred


@pytest.fixture
def y_values_very_bad():
    y_true = Tensor(np.array(
        [
            [10, 11], [12, 13]
        ]
    ))

    y_pred = Tensor(np.array(
        [
            [1, 0.5], [1.75, 0.2]
        ]
    ))
    return y_true, y_pred


@pytest.fixture
def y_values_perfect():
    y_true = Tensor(np.array(
        [
            [1, 2, 3], [4, 5, 6]
        ]
    ))

    y_pred = Tensor(np.array(
        [
            [1, 2, 3], [4, 5, 6]
        ]
    ))
    return y_true, y_pred


@pytest.fixture
def y_value_multi_dim():
    y_true = Tensor(np.array(
        [
            [1, 1], [1, 1]
        ]
    ))

    y_pred = Tensor(np.array(
        [
            [2, 1], [1, 2]
        ]
    ))
    return y_true, y_pred


def test_mae(y_values_one_dim, y_value_multi_dim):
    loss = mae(y_values_one_dim[0], y_values_one_dim[1])
    loss2 = mae(y_value_multi_dim[0], y_value_multi_dim[1])
    assert loss == 1.0
    assert loss2 == 0.5


def test_rmse(y_values_one_dim, y_value_multi_dim):
    loss = rmse(y_values_one_dim[0], y_values_one_dim[1])
    loss2 = rmse(y_value_multi_dim[0], y_value_multi_dim[1])
    assert loss == 1.0
    assert np.round(loss2, 6) == np.round(np.sqrt(0.5), 6)


def test_mape(y_values_one_dim, y_value_multi_dim):
    loss = mape(y_values_one_dim[0], y_values_one_dim[1])
    loss2 = mape(y_value_multi_dim[0], y_value_multi_dim[1])
    assert loss == 75.0
    assert np.round(loss2, 6) == 50.0


def test_smape(y_values_one_dim, y_value_multi_dim):
    loss = smape(y_values_one_dim[0], y_values_one_dim[1])
    loss2 = smape(y_value_multi_dim[0], y_value_multi_dim[1])
    assert np.round(loss, 6) == 66.666664
    assert np.round(loss2, 6) == 33.333332


def test_mase(y_values_one_dim, y_values_perfect, y_values_very_bad):
    loss = mase(y_values_perfect[0], y_values_perfect[1])
    loss2 = mase(y_values_very_bad[0], y_values_very_bad[1])
    loss3 = mase(y_values_one_dim[0], y_values_one_dim[1])
    assert np.round(loss, 6) == 0.0
    assert np.round(loss2, 6) == 2.659375
    assert np.round(loss3, 6) == 1.0


def test_rrse(y_values_one_dim, y_values_perfect, y_values_very_bad):
    loss = rrse(y_values_perfect[0], y_values_perfect[1])
    loss2 = rrse(y_values_very_bad[0], y_values_very_bad[1])
    loss3 = rrse(y_values_one_dim[0], y_values_one_dim[1])
    assert np.round(loss, 6) == 0.0
    assert np.round(loss2, 6) == 9.593253
    assert np.round(loss3, 6) == 2.0


def test_rae(y_values_one_dim, y_values_perfect, y_values_very_bad):
    loss = rae(y_values_perfect[0], y_values_perfect[1])
    loss2 = rae(y_values_very_bad[0], y_values_very_bad[1])
    loss3 = rae(y_values_one_dim[0], y_values_one_dim[1])
    assert np.round(loss, 6) == 0.0
    assert np.round(loss2, 6) == 92.030502
    assert np.round(loss3, 6) == 4.0


def test_accu_neighborhood(y_values_one_dim, y_values_perfect, y_values_very_bad):
    loss = accu_neighborhood(y_values_perfect[0], y_values_perfect[1])
    loss2 = accu_neighborhood(y_values_very_bad[0], y_values_very_bad[1])
    loss3 = accu_neighborhood(y_values_one_dim[0], y_values_one_dim[1])
    assert np.round(loss, 6) == 100.0
    assert np.round(loss2, 6) == 0.0
    assert np.round(loss3, 6) == 0.0


def test_oratio(y_values_one_dim, y_values_perfect, y_values_very_bad):
    loss = oratio(y_values_perfect[0], y_values_perfect[1])
    loss2 = oratio(y_values_very_bad[0], y_values_very_bad[1])
    loss3 = oratio(y_values_one_dim[0], y_values_one_dim[1])
    assert np.round(loss, 6) == 0.0
    assert np.round(loss2, 6) == 0.0
    assert np.round(loss3, 6) == 50.0


def test_uratio(y_values_one_dim, y_values_perfect, y_values_very_bad):
    loss = uratio(y_values_perfect[0], y_values_perfect[1])
    loss2 = uratio(y_values_very_bad[0], y_values_very_bad[1])
    loss3 = uratio(y_values_one_dim[0], y_values_one_dim[1])
    assert np.round(loss, 6) == 0.0
    assert np.round(loss2, 6) == 100.0
    assert np.round(loss3, 6) == 50.0


def test_stde(y_values_one_dim, y_values_perfect, y_values_very_bad):
    loss = stde(y_values_perfect[0], y_values_perfect[1])
    loss2 = stde(y_values_very_bad[0], y_values_very_bad[1])
    loss3 = stde(y_values_one_dim[0], y_values_one_dim[1])
    assert np.round(loss, 6) == 0.0
    assert np.round(loss2, 6) == 1.583969
    assert np.round(loss3, 6) == 0.0


def test_nrmse(y_values_one_dim, y_values_perfect, y_values_very_bad):
    loss = nrmse(y_values_perfect[0], y_values_perfect[1])
    loss2 = nrmse(y_values_very_bad[0], y_values_very_bad[1])
    loss3 = nrmse(y_values_one_dim[0], y_values_one_dim[1])
    assert np.round(loss, 6) == 0.0
    assert np.round(loss2, 6) == 0.932659
    assert np.round(loss3, 6) == 0.666667
