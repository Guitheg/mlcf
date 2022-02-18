import configparser
from pathlib import Path
from typing import List

FILE_PARAMETER_NAME = "parameters.ini"


def get_param_dict_from_str(list_param: List[str]):
    param_dict = {}
    for param in list_param:
        name_value = param.split("=")
        name = name_value[0]
        value = [elem for elem in name_value[1].split(",") if elem != ""]
        param_dict[name] = value[0] if len(value) == 1 else value
    return param_dict


class Parameters:
    def __init__(self, config: configparser.RawConfigParser, path: Path):
        self.config = config
        self.path = path

    def get(self, *args, **kwargs):
        return self.config.get(*args, **kwargs)

    def set(self, *args, **kwargs):
        self.config.set(*args, **kwargs)
        with open(self.path, "w") as f:
            self.config.write(f)


def get_config(dir_project: Path, dir_pref: Path) -> configparser.RawConfigParser:
    """
    #Renvoie la configuration de init.ini, se trouvant dans le home ou dans le repertoire projet
    Renvoi la configuration du programme si la version de la config projet et celle du programme est
    différente
    """
    dir_config_prgm = dir_project.joinpath(FILE_PARAMETER_NAME)
    dir_config_pref = dir_pref.joinpath(FILE_PARAMETER_NAME)
    config = configparser.RawConfigParser()

    config.read((dir_config_prgm, dir_config_pref))

    return config
