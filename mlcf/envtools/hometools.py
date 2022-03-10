import platform
import shutil
from logging import Logger
from pathlib import Path
from typing import Tuple

# MLCF modules
from mlcf.envtools.logtools import init_logging
from mlcf.envtools.paramtools import FILE_PARAMETER_NAME, Parameters, get_config
from mlcf.envtools.pathtools import get_dir_prgm, get_path
from mlcf import __version__, __appname__, __fullappname__


class ProjectHome:
    def __init__(
        self, home_name: str, home_directory: Path, create_userdir: bool = False
    ):
        self.home_name = home_name
        self.dir_prgm: Path = get_dir_prgm()
        self.dir: Path = get_path(
            str(home_directory), home_name, create_dir=create_userdir
        )
        self.log, self.cfg, self.id = init_project(
            home_name=home_name, home_directory=self.dir
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()

        if exc_type is not None:
            raise Exception(exc_value, traceback)

        return True

    def get_dir(self):
        return self.dir

    def exit(self):
        close_project(self.log, self.id)


class MlcfHome(ProjectHome):
    HOME_NAME = "mlcf_home"
    DATA = "data"
    ML = "ml"
    TRAINERS = "trainers"
    MODELS = "models"

    def __init__(
        self, home_directory: Path, create_userdir: bool = False, *args, **kwargs
    ):
        super(MlcfHome, self).__init__(
            home_name=self.HOME_NAME,
            home_directory=home_directory,
            create_userdir=create_userdir,
        )
        self.data_dir: Path = get_path(
            str(self.dir), self.DATA, create_dir=create_userdir
        )
        self.ml_dir: Path = get_path(str(self.dir), self.ML, create_dir=create_userdir)
        self.trainer_dir: Path = get_path(
            str(self.ml_dir), self.TRAINERS, create_dir=create_userdir
        )
        self.models_dir: Path = get_path(
            str(self.ml_dir), self.MODELS, create_dir=create_userdir
        )

    def check_file(self, file_path: Path, dir: Path):
        if not file_path.is_file():
            list_file = [x.stem for x in dir.iterdir() if x.is_file()]
            raise Exception(
                f"{file_path} doesn't exist. Here the list of file detected by "
                + f"MlcfHome: {list_file}. All this file are in: {dir}"
            )


def init_project(
    home_name: str, home_directory: Path, talkative: bool = False
) -> Tuple[Logger, Parameters, str]:
    """
    #initialise le logger et le renvoie, avec des informations de chemin d'accès
    """
    dir_prgm: Path = get_dir_prgm()
    dir_pref: Path = get_path(str(home_directory))
    if talkative:
        print(f"Directory log of the program: {str(dir_pref)}")
    config = get_config(dir_prgm, dir_pref)

    logger = init_logging(home_name=home_name, dir_pref=dir_pref, config=config)

    id_prog = (
        f"{__appname__} - {__fullappname__} - version v{__version__}"
    )

    syst_info = f"{platform.system()} {platform.release()} - Python  {platform.python_version()}"

    logger.info("##########| **START** %s |##########", id_prog)
    logger.info(syst_info)

    """
    # Sauvegarde properties dans repertoire de home de l'utilisateur
    """
    try:
        with open(dir_pref.joinpath(FILE_PARAMETER_NAME), "w") as configFile:
            config.write(configFile)
        logger.info(
            "User preferences saved in: %s", dir_pref.joinpath(FILE_PARAMETER_NAME)
        )

    except EnvironmentError:
        try:
            shutil.copy(dir_pref.joinpath(FILE_PARAMETER_NAME), dir_pref)
            logger.info(
                "(Attempt 2) User preferences saved in: %s",
                dir_pref.joinpath(FILE_PARAMETER_NAME),
            )
        except EnvironmentError as exc:
            logger.warning("Impossible to save user preferences: %s", str(exc))

    return (logger, Parameters(config, dir_pref.joinpath(FILE_PARAMETER_NAME)), id_prog)


def close_project(logger: Logger, id_prog: str):
    logger.info("###########| **END** %s |###########", id_prog)
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
