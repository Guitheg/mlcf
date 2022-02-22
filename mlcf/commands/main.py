from argparse import Namespace
from enum import Enum, unique

# MLCF modules
from mlcf.commands import build_dataset, launch_machine_learning
from mlcf.datatools.wtst import EXTENSION_FILE, read_wtseries_training
from mlcf.envtools.hometools import MlcfHome
from mlcf.datatools.preprocessing import PreProcessDict
from mlcf.datatools.indice import Indice


@unique
class Command(Enum):
    BUILD = "build-dataset"
    TRAIN = "train"
    VISUALIZE = "visualize"

    @classmethod
    def list_value(self):
        return [item.value for item in list(self)]


def run(mlcf: MlcfHome, args: Namespace):

    mlcf.log.info(f"Arguments: {args}")

    kwargs = vars(args).copy()
    kwargs.pop("create_userdir")
    if args.command:
        kwargs.pop("command")

        if args.command == Command.BUILD.value:

            kwargs["preprocess"] = PreProcessDict[args.preprocess]
            if args.indices:
                kwargs["indices"] = [Indice(indice) for indice in args.indices]
            build_dataset(project=mlcf, **kwargs)

        elif args.command == Command.VISUALIZE.value:
            dataset_filepath = mlcf.data_dir.joinpath(args.dataset_name).with_suffix(
                EXTENSION_FILE
            )
            mlcf.check_file(dataset_filepath, mlcf.data_dir)
            dataset = read_wtseries_training(dataset_filepath)
            print(dataset("train", "input")[0])

        elif args.command == Command.TRAIN.value:
            if args.training_name is None:
                kwargs["training_name"] = args.trainer_name

            launch_machine_learning(project=mlcf, **kwargs)
