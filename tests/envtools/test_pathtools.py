from pathlib import Path
import pytest
from mlcf.envtools.pathtools import create_path, get_path


def test_create_path(tmp_path: Path):
    assert len(list(tmp_path.iterdir())) == 0
    test_path = create_path(str(tmp_path.joinpath("test_path")))
    assert test_path == tmp_path.joinpath("test_path")
    create_path(str(test_path))
    assert len(list(tmp_path.iterdir())) == 1


def test_get_path(tmp_path: Path):
    assert len(list(tmp_path.iterdir())) == 0
    with pytest.raises(Exception):
        get_path(str(tmp_path.joinpath("test_path")))
    test_path = get_path(str(tmp_path.joinpath("test_path")), create_dir=True)
    assert test_path == tmp_path.joinpath("test_path")
    assert len(list(tmp_path.iterdir())) == 1
