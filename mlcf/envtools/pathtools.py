from os.path import join, isdir, abspath, dirname, expanduser
from os import makedirs
from pathlib import Path
import sys
from pydantic import PathNotExistsError


def create_path(*paths: str) -> Path:
    new_path = join(*paths)
    if not isdir(new_path):
        makedirs(new_path, exist_ok=True)
    return Path(new_path)


def get_path(*paths: str, create_dir: bool = False) -> Path:
    if create_dir:
        create_path(*paths)
    my_path = join(*paths)
    if not isdir(my_path):
        raise PathNotExistsError(f"The directory: '{my_path}' is not a directory")
    return Path(my_path)


def get_dir_prgm() -> Path:
    """
    #Renvoie la chaine de caractère correspondant au chemin d'accès aboslu de
    #l'endroit ou a été lancé le programme
    """
    return Path(dirname(abspath(sys.argv[0])))


def get_dir_home_user() -> Path:
    """
    #Renvoie le chemin d'accès correspondant au 'Home' de l'utilisateur
    """
    home_user = expanduser("~")
    return Path(home_user)