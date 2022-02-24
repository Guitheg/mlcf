
import zipfile
from mlcf.datatools.datasetools import \
    merge_dict_of_dataframe, \
    read_ohlcv_json_rawdata, \
    select_raw_data_based_on_name, \
    write_wtstdataset_from_raw_data
import numpy as np
import pandas as pd
from pathlib import Path
import os
from mlcf.datatools.preprocessing import Identity
from mlcf.datatools.wtst_dataset import WTSTrainingDataset, \
    TS_DATA_ARCHDIR, is_dir_in_zipfile, iterdir_in_zipfile


def test_read_ohlcv_json_rawdata(btc_ohlcv, rawdata_btc_path):
    read_data = read_ohlcv_json_rawdata(rawdata_btc_path)
    assert np.all(read_data == btc_ohlcv)


def test_select_raw_data_based_on_name(rawdata_btc_path):
    assert select_raw_data_based_on_name(["BTC/BUSD"], ["1h"], rawdata_btc_path)
    assert not select_raw_data_based_on_name(["ETH/BUSD"], ["1h"], rawdata_btc_path)
    assert select_raw_data_based_on_name(["BTC/BUSD", "ETH/BUSD"], ["1h"], rawdata_btc_path)
    assert not select_raw_data_based_on_name(["BTC/BUSD", "ETH/BUSD"], ["1d"], rawdata_btc_path)


def test_merge_dict_of_dataframe():
    dict_a = {
        "t": [0.1, 0.2, 0.3],
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    }
    dict_b = {
        "t": [0.1, 0.2, 0.3],
        "a": [7, 8, 9],
        "b": [10, 11, 12]
    }

    dfa = pd.DataFrame(dict_a)
    dfb = pd.DataFrame(dict_b)

    dict_to_merge = {
        "A": dfa,
        "B": dfb
    }

    df = merge_dict_of_dataframe(dict_to_merge, index_column="t")
    assert df.columns[0] == "t"
    assert len(df.columns) == 5
    assert df.columns[3] == "a_B"
    assert len(df.values) == 3


def test_iterdir_in_zipfile(tmp_path):
    doss_a = tmp_path.joinpath("test/a/g")
    doss_b = tmp_path.joinpath("test/b")
    zippath = tmp_path.joinpath("myzip.zip")
    os.makedirs(doss_a)
    os.makedirs(doss_b)
    with open(doss_a.joinpath("text.txt"), "w") as f:
        f.write("blabla")
    with open(doss_b.joinpath("text.txt"), "w") as f:
        f.write("blabla")
    with zipfile.ZipFile(zippath, "w") as zipf:
        cwd = os.getcwd()
        os.chdir(tmp_path)
        zipf.write("test/a/g/text.txt")
        zipf.write("test/b/text.txt")
        assert is_dir_in_zipfile(zipf, Path("test/a"))
        assert not is_dir_in_zipfile(zipf, Path("test/c"))
        assert set(iterdir_in_zipfile(zipf, Path("test"))) == set(["a", "b"])
        os.chdir(cwd)


def test_write_wtstdataset_from_raw_data(mlcf_home, eth_ts_data, testdatadir):
    write_wtstdataset_from_raw_data(
        project=mlcf_home,
        rawdata_dir=Path(testdatadir / "user_data/data/binance"),
        dataset_name="TestDataSet",
        pairs=["ETH/BUSD"],
        timeframes=["1h"],
        input_width=20,
        target_width=1,
        offset=0,
        window_step=1,
        n_interval=1,
        index_column="date",
        prop_tv=0.2,
        prop_v=0.2,
        indices=[],
        preprocess=Identity,
        merge_pairs=True
    )

    dataset_path = mlcf_home.data_dir.joinpath("TestDataSet.wtst")
    assert dataset_path.is_file()
    with zipfile.ZipFile(dataset_path, "r") as zipf:
        assert is_dir_in_zipfile(zipf, Path(TS_DATA_ARCHDIR).joinpath("TRAIN"))
        assert is_dir_in_zipfile(zipf, Path(TS_DATA_ARCHDIR).joinpath("TEST"))
        assert is_dir_in_zipfile(zipf, Path(TS_DATA_ARCHDIR).joinpath("VALIDATION"))

    dataset = WTSTrainingDataset(
        mlcf_home.data_dir.joinpath("TestDataSet.wtst"),
        index_column="date"
    )
    assert np.all(np.array(dataset[0][0]) == np.array(eth_ts_data("train")[0][0]))
