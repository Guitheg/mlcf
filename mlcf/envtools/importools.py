from importlib import util
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from os.path import basename
from pathlib import Path
from types import ModuleType


def model_class_import(pyfile_path: Path):
    if not pyfile_path.is_file():
        raise Exception(f"The file: {pyfile_path} doesn't exist")
    trainer_name = basename(str(pyfile_path.with_suffix("")))
    module_spec = util.spec_from_file_location(
        trainer_name, pyfile_path
    )
    if isinstance(module_spec, ModuleSpec):
        module = util.module_from_spec(module_spec)
        loader = module_spec.loader
        if isinstance(module, ModuleType) and isinstance(loader, Loader):
            loader.exec_module(module)
        else:
            raise Exception("The module has not been found or correctly load...")
    else:
        raise Exception("The module has not been found or correctly load...")
    return getattr(module, str(trainer_name).capitalize())


def train_method_import(pyfile_path: Path):
    if not pyfile_path.is_file():
        raise Exception(f"The file: {pyfile_path} doesn't exist")
    trainer_name = basename(str(pyfile_path.with_suffix("")))
    module_spec = util.spec_from_file_location(
        trainer_name, pyfile_path
    )
    if isinstance(module_spec, ModuleSpec):
        module = util.module_from_spec(module_spec)
        loader = module_spec.loader
        if isinstance(module, ModuleType) and isinstance(loader, Loader):
            loader.exec_module(module)
        else:
            raise Exception("The module has not been found or correctly load...")
    else:
        raise Exception("The module has not been found or correctly load...")
    return module.train  # type: ignore
