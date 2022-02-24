
from mlcf.aitools.super_module import SuperModule, select_list_index_columns
from torch.optim import SGD
from torch.nn import L1Loss

from mlcf.aitools.metrics import L2


def test_list_index_columns():
    a = ["a", "b", "c", "d"]
    b = ["c", "d"]
    r = select_list_index_columns(b, a)
    assert r == [2, 3]


def test_super_module(mlp, mlcf_home, ts_data):
    module: SuperModule = mlp

    module.init(
        loss=L1Loss(),
        optimizer=SGD(module.parameters(), lr=0.1),
        metrics=[L2],
        training_name="test_training",
        project=mlcf_home
    )
    assert module.initialize
    txt = module.summary()
    assert str(module.parameters) == txt
    assert module.epoch == 0
    module.fit(ts_data, 2, 20, checkpoint=True)
    checkpoint_path = module.manager.checkpoint_path.joinpath(
        f"checkpoint_{module.training_name}_{module.manager.now}.pt"
    )
    assert checkpoint_path.is_file()

    module2 = mlp
    module2.init_load_checkpoint(
        loss=L1Loss(),
        training_name="test_training",
        project=mlcf_home,
        optimizer=SGD(module.parameters(), lr=0.1),
        resume_training=True
    )
    assert module2.epoch == 1
    assert module2.optimizer.state_dict() == module.optimizer.state_dict()
    module2.fit(ts_data, 2, 20, checkpoint=True)
    checkpoint_path = module.manager.checkpoint_path.joinpath(
        f"checkpoint_{module2.training_name}_{module2.manager.now}.pt"
    )
    assert checkpoint_path.is_file()
