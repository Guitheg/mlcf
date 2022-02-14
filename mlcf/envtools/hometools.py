
from enum import Enum, unique
import platform, os, sys, shutil
from pathlib import Path
from typing import Tuple
from logging import Logger, shutdown

### MLCF modules ###
from mlcf.envtools.logtools import init_logging
from mlcf.envtools.paramtools import get_config, Parameters, FILE_PARAMETER_NAME
from mlcf.envtools.pathtools import get_dir_prgm, create_path, get_path


class ProjectHome():
    def __init__(self, home_name : str,
                 home_directory : Path,
                 create_userdir : bool = False):
        self.home_name = home_name
        self.dir_prgm : Path = get_dir_prgm()
        self.dir : Path = get_path(home_directory, 
                                   home_name, 
                                   create_dir=create_userdir)
        self.log, self.cfg, self.id = init_project(home_name=home_name, 
                                                   home_directory=self.dir)
    def get_dir(self):
        return self.dir
    
    def exit(self):
        close_project(self.log, self.id)
          
class MlcfHome(ProjectHome):
    HOME_NAME = "mlcf"
    DATA = "data"
    ML = "ml"
    TRAINERS = "trainers"
    MODELS = "models"
    def __init__(self, home_directory : Path, 
                 create_userdir : bool = False, *args, **kwargs):
        super(MlcfHome, self).__init__(home_name=self.HOME_NAME, 
                                    home_directory=home_directory, 
                                    create_userdir=create_userdir,
                                    *args, **kwargs)
        self.data_dir : Path = get_path(self.dir, self.DATA, create_dir=create_userdir)
        self.ml_dir : Path = get_path(self.dir, self.ML, create_dir=create_userdir)
        self.trainer_dir : Path = get_path(self.ml_dir, self.TRAINERS, create_dir=create_userdir)
        self.models_dir : Path = get_path(self.ml_dir, self.MODELS, create_dir=create_userdir)

    def check_file(self, file_path : Path, dir : Path):
        if not file_path.is_file(): 
            list_file = [x.stem for x in dir.iterdir() if x.is_file()]
            raise Exception(f"{file_path} doesn't exist. Here the list of file detected by MlcfHome: "+
                        f"{list_file}. All this file are in : {dir}")

def init_project(home_name : str, 
                 home_directory : Path,
                 talkative : bool = False) -> Tuple[Logger, Parameters, str]:
    """
    #initialise le logger et le renvoie, avec des informations de chemin d'accès
    """
    dir_prgm = get_dir_prgm()
    dir_pref = get_path(home_directory)
    if talkative == True:
        print("Directory log of the program : "+dir_pref)
    config = get_config(dir_prgm, dir_pref)
    
    logger = init_logging(home_name=home_name, dir_pref=dir_pref, config=config)

    id_prog = config.get('Version', 'AppName') + ' - ' + config.get('Version', 'FullAppName') +\
    ' - version ' + config.get('Version', 'Number')
            
    syst_info = platform.system() + " " + platform.release() + \
                " - Python : " + platform.python_version()

    logger.info("##########| **START** %s |##########", id_prog)
    logger.info(syst_info)
    
    """
    # Sauvegarde properties dans repertoire de home de l'utilisateur
    """
    try:
        with open(os.path.join(dir_pref, FILE_PARAMETER_NAME), "w") as configFile:
            config.write(configFile)
        logger.info("User preferences saved in : %s", os.path.join(dir_pref, FILE_PARAMETER_NAME))

    except EnvironmentError as exc:
        try:
            shutil.copy(os.path.join(dir_pref, FILE_PARAMETER_NAME), dir_pref)
            logger.info("(Attempt 2) User preferences saved in : %s", 
                        os.path.join(dir_pref, FILE_PARAMETER_NAME))
        except EnvironmentError as exc:             
            logger.warning("Impossible to save user preferences : %s", str(exc))
            
    return (logger,
            Parameters(config, os.path.join(dir_pref, FILE_PARAMETER_NAME)),
            id_prog)
    
def close_project(logger : Logger, id_prog : str) :
    logger.info("###########| **END** %s |###########", id_prog)
    shutdown()
    sys.exit(0)