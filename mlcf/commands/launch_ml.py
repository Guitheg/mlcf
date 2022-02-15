from mlcf.envtools.importools import train_method_import

# MLCF modules
from mlcf.datatools.wtseries_training import read_wtseries_training
from mlcf.envtools.hometools import MlcfHome


def launch_machine_learning(project: MlcfHome, trainer_name: str, training_name: str,
                            dataset_name: str, *args, **kwargs):

    data_path = project.data_dir.joinpath(dataset_name)
    project.check_file(data_path, project.data_dir)

    wtst_data = read_wtseries_training(data_path, project=project)

    pyfile_path = project.trainer_dir.joinpath(trainer_name.lower()+".py")
    project.check_file(pyfile_path, project.trainer_dir)

    train_method = train_method_import(pyfile_path)
    project.log.info(f"Trainer method used in: {pyfile_path}")
    train_method(project=project, training_name=training_name, wtst_data=wtst_data,
                 *args, **kwargs)