
from pathlib import Path
import os.path
import configparser

FILE_PARAMETER_NAME = "parameters.ini"

class Parameters():
    def __init__(self, config : configparser.RawConfigParser, 
                 path : Path):
        self.config = config
        self.path = path
    
    def get(self, *args, **kwargs):
        return self.config.get(*args, **kwargs)
    
    def set(self, *args, **kwargs):
        self.config.set(*args, **kwargs)
        with open(self.path, "w") as f:
            self.config.write(f)

def get_config(dir_project : Path, dir_pref : Path) -> configparser.RawConfigParser:
    """
    #Renvoie la configuration de init.ini, se trouvant dans le home ou dans le repertoire projet
    Renvoi la configuration du programme si la version de la config projet et celle du programme est
    différente
    """
    dir_config_prgm = os.path.join(dir_project, FILE_PARAMETER_NAME)
    dir_config_pref = os.path.join(dir_pref, FILE_PARAMETER_NAME)    
    config = configparser.RawConfigParser()
    configproject = configparser.RawConfigParser()

    configproject.read(dir_config_prgm)
    config.read((dir_config_prgm, dir_config_pref))

    if config.get("Version","number") == configproject.get("Version","number"):
        return config
    else:
        return configproject

