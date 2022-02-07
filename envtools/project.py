
import platform, os, sys, shutil
from pathlib import Path
from typing import Tuple
from logging import Logger, shutdown
from envtools.logging import init_logging
from envtools.parameters import get_config, Parameters, FILE_PARAMETER_NAME
from envtools.path import get_dir_prgm, create_path


class Project():
    def __init__(self, project_name : str,
                 project_directory):
        
        self.dir_prgm = get_dir_prgm()
        self.dir = project_directory
        self.log, self.cfg, self.id = init_project(project_name=project_name, 
                                                      project_directory=project_directory)
        
    def exit(self):
        close_project(self.log, self.id)

def init_project(project_name : str, 
             talkative : bool = False, 
             project_directory : Path = None) -> Tuple[Logger,
                                                       Parameters,
                                                       str]:
    """
    #initialise le logger et le renvoie, avec des informations de chemin d'accès
    """
    dir_prgm = get_dir_prgm()
    dir_pref = create_path(project_directory)
    if talkative == True:
        print("Répertoire log du programme : "+dir_pref)
    config = get_config(dir_prgm, dir_pref)
    
    logger = init_logging(project_name=project_name, dir_pref=dir_pref, config=config)

    id_prog = config.get('Version', 'AppName') + ' - version ' + \
             config.get('Version', 'Number')
            
    syst_info = platform.system() + " " + platform.release() + \
                " - Python : " + platform.python_version()

    logger.info("#####################################")
    logger.info("-DEBUT- de %s", id_prog)
    logger.info(syst_info)
    
    """
    # Sauvegarde properties dans repertoire de home de l'utilisateur
    """
    try:
        with open(os.path.join(dir_pref, FILE_PARAMETER_NAME), "w") as configFile:
            config.write(configFile)
        logger.info("Préférences utilisateur sauvées dans : %s", dir_pref)

    except EnvironmentError as exc:
        try:
            shutil.copy(os.path.join(dir_pref, FILE_PARAMETER_NAME), dir_pref)
            logger.info("(Essai 2) Préférences utilisateur sauvées dans : %s", dir_pref)
        except EnvironmentError as exc:             
            logger.warning("Impossible de sauver les préférences utilisateur : %s", str(exc))
            
    return (logger,
            Parameters(config, os.path.join(dir_pref, FILE_PARAMETER_NAME)),
            id_prog)
    
def close_project(logger : Logger, id_prog : str) :
    logger.info("- FIN - de %s", id_prog)
    logger.info("#####################################")
    shutdown()
    sys.exit(0)