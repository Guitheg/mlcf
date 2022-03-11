import argparse

import os
from pathlib import Path

# MLCF modules
from mlcf.datatools.indice import Indice
from mlcf.datatools.preprocessing import PreProcessDict
from mlcf.envtools.hometools import MlcfHome
from mlcf.commands.main import run, Command

PRGM_NAME = MlcfHome.HOME_NAME


def main():
    parser = argparse.ArgumentParser(prog=PRGM_NAME)
    # Generals arguments
    general_arguments_group = parser.add_argument_group(
        "Common arguments", "All this arguments are common with every commands"
    )
    general_arguments_group.add_argument(
        "-u",
        "--userdir",
        help="The user directory commonly called 'user_data'",
        type=Path,
        default=Path(os.curdir).joinpath("user_data"),
    )
    general_arguments_group.add_argument(
        "--create-userdir",
        help="If it is given then create the userdir"
        + "repositories (if userdir doesn't exist)."
        + "If userdir doesn't exist and if it's not given "
        + "then it raises an error.",
        action="store_true",
    )
    subcommands = parser.add_subparsers(
        dest="command",
        title="CG-RBI commands",
        description="",
        help="The list of commands you can use"
    )
    # Build arguments
    command_build = subcommands.add_parser(
        Command.BUILD.value, help="Dataset creation command"
    )
    command_build.add_argument(
        "--rawdata-dir",
        help="The directory of the raw data used to build the dataset. It will uses every file in" +
        " the given directory",
        type=Path,
        required=True
    )
    command_build.add_argument(
        "--dataset-name",
        help="The name of the dataset file which will be created",
        type=str,
        required=True,
    )
    command_build.add_argument(
        "--pairs",
        help="The list of pairs from which the dataset is build. They are space-separated. " +
             "(Default : BTC/BUSD)",
        type=str,
        nargs="+",
        default=["BTC/BUSD"]
    )
    command_build.add_argument(
        "--timeframes",
        help="The list of timeframes from which the dataset is build. They are space-separated. " +
             "(Default : 1d)",
        type=str,
        nargs="+",
        default=["1d"],
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h",
                 "12h", "1d", "3d", "1w", "2w", "1M", "1y"]
    )
    command_build.add_argument(
        "--input-width",
        help="The width of the input part in the sliding window. "
        + "Can also be seen as the sequence length of a neural network.",
        required=True,
        metavar="WIDTH",
        type=int,
    )
    command_build.add_argument(
        "--target-width",
        help="The width of the target part in the sliding window (Default: 1)",
        default=1,
        type=int,
        metavar="WIDTH",
    )
    command_build.add_argument(
        "--offset",
        help="The width of the offset part in the sliding window (Default: 0)",
        default=0,
        type=int,
        metavar="WIDTH",
    )
    command_build.add_argument(
        "--window-step",
        help="The step between each sliding window (Default: 1)",
        default=1,
        type=int,
        metavar="STEP",
    )
    command_build.add_argument(
        "--n-interval",
        help="The number of intervals by which the data will be divided. "
        + "It allows to not have test and validation part just at the end "
        + "(but at the end of each part) without having an overlap between"
        + " the train and the evaluations parts. (Default: 1)",
        default=1,
        type=int,
        metavar="NUMBER",
    )
    command_build.add_argument(
        "--index-column",
        help="Name of the index column (commonly the time) (Default: 'date')",
        default="date",
        metavar="NAME",
        type=str,
    )
    command_build.add_argument(
        "--prop-tv",
        help="The proportion of the test and validation part union "
        + "from the data (Default: 0.1)",
        default=0.1,
        type=float,
        metavar="PERCENTAGE",
    )
    command_build.add_argument(
        "--prop-v",
        help="The proportion of the validation part from the test and "
        + "the validation par union (Default: 0.3)",
        default=0.3,
        type=float,
        metavar="PERCENTAGE",
    )
    command_build.add_argument(
        "--indices",
        help="List of indicators we want to add in the data (Optionnal)",
        type=str,
        choices=Indice.list_value(),
        metavar="INDICE",
        nargs="+",
    )
    command_build.add_argument(
        "--preprocess",
        help="List of pre processing function we want to use "
        + "to pre process the data. Note: it's use independtly on each "
        + "window",
        type=str,
        choices=PreProcessDict.keys(),
        metavar="FUNCTION NAME",
    )
    command_build.add_argument(
        "--merge-pairs",
        help="Merge the pairs together in order to extend the number of features.",
        action="store_true"
    )
    command_build.add_argument(
        "--standardize",
        help="Standardize all the given dimension of the dataset",
        action="store_true"
    )
    command_build.add_argument(
        "--n-category",
        help="Give a number of category in order to balance number of returns category in the"
        + " training part of the dataset.",
        type=int,
        default=0
    )
    command_build.add_argument(
        "--selected-columns",
        help="List of the selected features (Default is all)",
        type=str,
        choices=["open", "low", "high", "close", "volume"],
        nargs="+",
        default=[],
    )
    command_build.add_argument(
        "--unselected-columns",
        help="List of unselected features. (such as 'low' price for example)",
        type=str,
        choices=["open", "low", "high", "close", "volume"],
        nargs="+",
        default=[]
    )

    # Train arguments
    command_train = subcommands.add_parser(
        Command.TRAIN.value, help="Neural Network training command"
    )
    command_train.add_argument(
        "--trainer-name",
        help="The name of the trainer file. IMPORTANT: "
        + "the command call the method: train() inside the "
        + "file given by the trainer file name.",
        type=str,
        metavar="NAME",
        required=True,
    )
    command_train.add_argument(
        "--training-name",
        help="The name of the training name, useful for logging, " + "checkpoint etc.",
        type=str,
        metavar="NAME",
    )
    command_train.add_argument(
        "--dataset-name",
        help="The dataset name use for the training",
        metavar="NAME",
        type=str,
        required=True,
    )
    command_train.add_argument(
        "--param",
        help="The list of arguments for the trainer. IMPORTANT: "
        + "The list must be in the form: key1=value1 key2=value2"
        + " key3=elem1,elem2,elem3",
        nargs="+",
        type=str,
    )
    # Visualize arguments
    command_visualize = subcommands.add_parser(
        Command.VISUALIZE.value, help="Dataset visualization command"
    )
    command_visualize.add_argument(
        "--dataset-name",
        help="The dataset name to visualize",
        type=str,
        metavar="PATH",
        required=True,
    )
    command_visualize.add_argument(
        "--type-visu",
        help="The type of visualization",
        choices=["console"],
        default="console",
        type=str,
    )
    args = parser.parse_args()

    with MlcfHome(home_directory=args.userdir, create_userdir=args.create_userdir) as mlcf:
        run(mlcf, args)


if __name__ == "__main__":
    main()
