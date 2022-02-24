# MLCF modules
from mlcf.datatools.wtst_dataset import EXTENSION_FILE, WTSTrainingDataset
from mlcf.envtools.hometools import MlcfHome
from mlcf.envtools.importools import train_method_import


def launch_machine_learning(
    project: MlcfHome,
    trainer_name: str,
    training_name: str,
    dataset_name: str,
    *args,
    **kwargs,
):

    data_path = project.data_dir.joinpath(dataset_name).with_suffix(EXTENSION_FILE)
    project.check_file(data_path, project.data_dir)

    wtst_data = WTSTrainingDataset(dataset_path=data_path, project=project)

    pyfile_path = project.trainer_dir.joinpath(trainer_name.lower() + ".py")
    project.check_file(pyfile_path, project.trainer_dir)

    train_method = train_method_import(pyfile_path)
    project.log.info(f"Trainer method used in: {pyfile_path}")
    train_method(
        project=project,
        training_name=training_name,
        wtst_data=wtst_data,
        *args,
        **kwargs,
    )
