from os.path import join, isdir, abspath, dirname, expanduser
from os import makedirs
from pathlib import Path
import sys

def create_path(*paths : str) -> Path:
    new_path = join(*paths)
    if not isdir(new_path):
        makedirs(new_path, exist_ok=True)
    return Path(new_path)
  
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