from torch.optim import SGD
from torch.nn import L1Loss
from mlcf.aitools.metrics import L2
from mlcf.aitools.training_manager import TrainingManager
from mlcf.envtools.hometools import MlcfHome


def test_training_manager(mlp, mlcf_home: MlcfHome):
    module = mlp
    module.init(
        loss=L1Loss(),
        optimizer=SGD(module.parameters(), lr=0.1),
        metrics=[L2],
        training_name="test_training",
        project=mlcf_home
    )
    tm: TrainingManager = module.manager
    assert str(tm.home_path) == str(mlcf_home.dir)
    assert tm.training_home.is_dir()
