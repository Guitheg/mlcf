import importlib
from pathlib import Path
from pydantic import PathNotExistsError
from os.path import basename

def model_class_import(pyfile_path : Path):
    if not pyfile_path.is_file():
        raise PathNotExistsError(f"The file : {pyfile_path} doesn't exist")
    trainer_name = basename(str(pyfile_path.with_suffix("")))
    module_spec = importlib.util.spec_from_file_location(trainer_name, pyfile_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, str(trainer_name).capitalize())

def train_method_import(pyfile_path : Path):
    if not pyfile_path.is_file():
        raise PathNotExistsError(f"The file : {pyfile_path} doesn't exist")
    trainer_name = basename(str(pyfile_path.with_suffix("")))
    module_spec = importlib.util.spec_from_file_location(trainer_name, pyfile_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.train
    