
import pandas as pd
import pytest
from mlcf.envtools.hometools import MlcfHome
from mlcf.aitools.super_module import SuperModule
from mlcf.datatools.wtst import WTSTraining
from mlcf.datatools.wtseries_tensor import WTSeriesTensor, Partition
from torch import nn, sigmoid
from pathlib import Path


@pytest.fixture
def testdatadir() -> Path:
    return (Path(__file__).parent / "testdata").resolve()


@pytest.fixture
def rawdata_btc_path(testdatadir):
    return Path(testdatadir / "user_data/data/binance/BTC_BUSD-1h.json")


@pytest.fixture
def rawdata_eth_path(testdatadir):
    return Path(testdatadir / "user_data/data/binance/ETH_BUSD-1h.json")


@pytest.fixture
def btc_ohlcv(rawdata_btc_path):
    data = pd.read_json(rawdata_btc_path)
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] = pd.to_datetime(data["date"], unit="ms")
    return data


@pytest.fixture
def eth_ohlcv(rawdata_eth_path):
    data = pd.read_json(rawdata_eth_path)
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] = pd.to_datetime(data["date"], unit="ms")
    return data


@pytest.fixture
def mlcf_home(tmp_path):
    userdir = tmp_path.joinpath("user_data")
    return MlcfHome(userdir, create_userdir=True)


@pytest.fixture
def ts_data(btc_ohlcv):
    ts_data = WTSTraining(20, index_column="date")
    ts_data.add_time_serie(btc_ohlcv[0:3000])
    return ts_data


@pytest.fixture
def ts_data_no_date(btc_ohlcv: pd.DataFrame):
    btc_ohlcv_copy = btc_ohlcv.drop('date', axis=1)
    ts_data = WTSTraining(20, features=btc_ohlcv_copy.columns)
    ts_data.add_time_serie(btc_ohlcv_copy[0:3000])
    return ts_data


@pytest.fixture
def eth_ts_data(eth_ohlcv):
    ts_data = WTSTraining(20, index_column="date")
    ts_data.add_time_serie(eth_ohlcv)
    return ts_data


@pytest.fixture
def mlp(ts_data):

    class MLP(SuperModule):

        def __init__(self, features, window_width, *args, **kwargs):
            super(MLP, self).__init__(*args, **kwargs)
            self.n_features = features*window_width

            self.layer = nn.Linear(self.n_features, 1)

        def forward(self, x):
            out = sigmoid(self.layer(x))
            return out

        def transform_x(self, x):
            x = x.reshape(self.n_features)
            return x

        def transform_y(self, y):
            return y[:, 0]

    return MLP(
        features=ts_data.ndim(),
        window_width=ts_data.input_width
    )


@pytest.fixture
def wts_tensor(ts_data_no_date):
    test_tensor = WTSeriesTensor(ts_data=ts_data_no_date, partition=Partition.TRAIN)
    return test_tensor
