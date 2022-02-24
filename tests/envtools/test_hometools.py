import pytest
from mlcf.envtools.hometools import MlcfHome
from pathlib import Path
from mlcf import __version__, __appname__, __fullappname__


def test_mlcf_home(tmp_path: Path):
    with pytest.raises(Exception):
        mlcf = MlcfHome(tmp_path.joinpath("user_data"))
    mlcf = MlcfHome(tmp_path.joinpath("user_data"), create_userdir=True)
    assert mlcf.data_dir == tmp_path.joinpath("user_data", "mlcf_home", "data")
    assert mlcf.ml_dir == tmp_path.joinpath("user_data", "mlcf_home", "ml")
    assert mlcf.trainer_dir == tmp_path.joinpath("user_data", "mlcf_home", "ml", "trainers")
    assert mlcf.models_dir == tmp_path.joinpath("user_data", "mlcf_home", "ml", "models")
    assert mlcf.dir == tmp_path.joinpath("user_data", "mlcf_home")
    assert tmp_path.joinpath("user_data", "mlcf_home", "parameters.ini").is_file()
    assert tmp_path.joinpath("user_data", "mlcf_home", "logs", "debugMessages").is_file()
    assert mlcf.id == f"{__appname__} - {__fullappname__} - version v{__version__}"
    assert mlcf.home_name == "mlcf_home"
