
from pathlib import Path
from typing import List
from CGrbi.envtools.importools import train_method_import

### CG-RBI modules ###
from CGrbi.datatools.wtseries_training import read_wtseries_training
from CGrbi.envtools.project import CGrbi
from CGrbi.envtools.paramtools import get_param_dict_from_str

def launch_machine_learning(project : CGrbi, trainer_name : str, training_name : str,
                            dataset_name : str, *args, **kwargs):
    
    print(project.data_dir)
    data_path = project.data_dir.joinpath(dataset_name)
    project.check_file(data_path, project.data_dir)
    
    wtst_data = read_wtseries_training(data_path)
    
    pyfile_path = project.trainer_dir.joinpath(trainer_name.lower()+".py")
    project.check_file(pyfile_path, project.trainer_dir)
    
    train_method = train_method_import(pyfile_path)
    train_method(training_name, *args, **kwargs)
