import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
from pydantic import PathNotExistsError
from os.path import basename


def model_class_import(pyfile_path: Path):
    if not pyfile_path.is_file():
        raise PathNotExistsError(f"The file: {pyfile_path} doesn't exist")
    trainer_name = basename(str(pyfile_path.with_suffix("")))
    module_spec: ModuleSpec = importlib.util.spec_from_file_location(trainer_name, pyfile_path)
    if module_spec is not None:
        module = importlib.util.module_from_spec(module_spec)
        if module is None:
            module_spec.loader.exec_module(module)
        else:
            raise Exception("The module has not been found or correctly load...")
    else:
        raise Exception("The module has not been found or correctly load...")
    return getattr(module, str(trainer_name).capitalize())


def train_method_import(pyfile_path: Path):
    if not pyfile_path.is_file():
        raise PathNotExistsError(f"The file: {pyfile_path} doesn't exist")
    trainer_name = basename(str(pyfile_path.with_suffix("")))
    module_spec: ModuleSpec = importlib.util.spec_from_file_location(trainer_name, pyfile_path)
    if module_spec is not None:
        module = importlib.util.module_from_spec(module_spec)
        if module is None:
            module_spec.loader.exec_module(module)
        else:
            raise Exception("The module has not been found or correctly load...")
    else:
        raise Exception("The module has not been found or correctly load...")
    return module.train  # type: ignore
