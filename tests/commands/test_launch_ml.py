
from mlcf.datatools.wtst_dataset import WTSTrainingDataset
from pathlib import Path
from mlcf.envtools.importools import train_method_import


def test_launch_machine_learning(mlcf_home):
    ts_data = WTSTrainingDataset(Path("tests/testdata/user_data/mlcf_home/data/TestDataset.wtst"),
                                 index_column="date")
    trainer_path = Path("tests/testdata/user_data/mlcf_home/ml/trainers/lstm_trainer.py")
    train_method = train_method_import(trainer_path)
    train_method(mlcf_home, "test_train", ts_data)
