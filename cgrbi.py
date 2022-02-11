from curses import meta
import argparse
from enum import Enum, unique
from pathlib import Path
from typing import List

from CGrbi.commands import build_dataset, launch_machine_learning

### CG-RBI modules ###
from CGrbi.datatools.indice import Indice
from CGrbi.datatools.preprocessing import PreProcessDict
from CGrbi.envtools.project import CGrbi, get_dir_prgm
from CGrbi.datatools.wtseries_training import EXTENSION_FILE, read_wtseries_training



@unique
class Command(Enum):
    BUILD = "build-dataset"
    TRAIN = "train"
    VISUALIZE = "visualize"

    @classmethod
    def list_value(self):
        return [item.value for item in list(self)]

def main():
    ################################################################################################
    ####################################### ARGUMENT PARSER ########################################
    ################################################################################################
    parser = argparse.ArgumentParser(prog="CG-RBI")
    ##### Generals arguments #####
    general_arguments_group = parser.add_argument_group(
        "Common arguments",
        "All this arguments are common with every commands")
    general_arguments_group.add_argument("-u", "--userdir", 
                                         help="The user directory commonly called 'user_data'",
                                         type=Path,
                                         default=Path(get_dir_prgm().joinpath("user_data")))
    general_arguments_group.add_argument("--create-userdir",
                                         help="If it is given then create the userdir" +\
                                             " repositories (if userdir doesn't exist)."+\
                                            " If userdir doesn't exist and if it's not given then"+\
                                            " it raises an error.",
                                         action="store_true")
    
    subcommands = parser.add_subparsers(
        dest="command",
        title="CG-RBI commands",
        description="",                            
        help='The list of commands you can use', required=True)
    
    ##### Build arguments #####
    command_build = subcommands.add_parser(Command.BUILD.value, 
                                           help="Dataset creation command")
    #- freqtrade
    group_freqtrade = command_build.add_argument_group(
        "Data download information", 
        "All info needed to download the wanted data. "+\
            "This will be passed to 'freqtrade download-data'")
    group_freqtrade.add_argument("--pairs",
                                 help="The list of pairs we want to download (Default : BTC/BUSD)",
                                 type=str,
                                 default=["BTC/BUSD"],
                                 nargs="+")
    group_freqtrade.add_argument("-t", "--timeframes", 
                                 help="The list of timeframes we want to download (Default : 1h)",
                                 type=str,
                                 default=["1h"],
                                 nargs="+")
    group_freqtrade.add_argument("--days", 
                                 help="Number of days of history data (Default : 30)", 
                                 metavar="NUMBER",
                                 default=30,
                                 type=int)
    group_freqtrade.add_argument("--exchange",
                                 help="The exchange market we will take the data from "+\
                                     "(default : binance)",
                                 type=str,
                                 default="binance"
                                 )
    #- wtst
    group_wtst = command_build.add_argument_group(
        "Windowed Time Series Training data information", 
        "All information about how to create the WTST data from the downloaded data")
    group_wtst.add_argument("--dataset-name",
                            help="The name of the dataset file which will be created",
                            type=str,
                            required=True)
    group_wtst.add_argument("-in", "--input-size",
                            help="The width of the input part in the sliding window. "+\
                                "Can also be seen as the sequence length of a neural network.",
                            required=True,
                            metavar="WIDTH",
                            type=int)
    group_wtst.add_argument("-tar", "--target-size", 
                            help="The width of the target part in the sliding window (Default : 1)",
                            default=1,
                            type=int,
                            metavar="WIDTH")
    group_wtst.add_argument("--offset",
                            help="The width of the offset part in the sliding window (Default : 0)",
                            default=0,
                            type=int,
                            metavar="WIDTH")
    group_wtst.add_argument("--window-step",
                            help="The step between each sliding window (Default : 1)",
                            default=1,
                            type=int,
                            metavar="STEP")
    group_wtst.add_argument("--n-interval",
                            help="The number of intervals by which the data will be divided. " +\
                                "It allows to not have test and validation part just at the end " +\
                                "(but at the end of each part) without having an overlap between"+\
                                " the train and the evaluations parts. (Default : 1)",
                            default=1,
                            type=int,
                            metavar="NUMBER")
    group_wtst.add_argument("--index-column",
                            help="Name of the index column (commonly the time) (Default : 'date')",
                            default="date",
                            metavar="NAME",
                            type=str)
    group_wtst.add_argument("--prop-tv",
                             help="The proportion of the test and validation part union "+\
                                 "from the data (Default : 0.1)",
                             default=0.1,
                             type=float,
                             metavar="PERCENTAGE")
    group_wtst.add_argument("--prop-v",
                             help="The proportion of the validation part from the test and "+\
                                 "the validation par union (Default : 0.3)",
                             default=0.3,
                             type=float,
                             metavar="PERCENTAGE")
    group_wtst.add_argument("--indices",
                            help="List of indicators we want to add in the data (Optionnal)",
                            type=str,
                            choices=Indice.list_value(),
                            metavar="INDICE",
                            nargs="+")
    
    group_wtst.add_argument("--preprocess",
                            help="List of pre processing function we want to use "+\
                                "to pre process the data. Note : it's use independtly on each "+\
                                "window",
                            type=str,
                            choices=PreProcessDict.keys(),
                            metavar="FUNCTION NAME")
    
    ##### Train arguments #####
    command_train = subcommands.add_parser(Command.TRAIN.value, 
                                           help="Neural Network training command")
    command_train.add_argument("--trainer-name",
                                help="The name of the trainer file. IMPORTANT : the command call "+\
                                "the method : train() inside the file given by the trainer "+\
                                "file name.",
                                type=str, metavar="NAME", required=True)
    command_train.add_argument("--training-name",
                                  help="The name of the training name, useful for logging, "+
                                  "checkpoint etc.", type = str, metavar="NAME")
    
    command_train.add_argument("--dataset-name", help="The dataset name use for the training",
                               metavar="NAME", type=str, required=True)
    
    command_train.add_argument("--param", help="The list of arguments for the trainer. IMPORTANT:"+\
                               "The list must be in the form : key1=value1 key2=value2"+\
                               " key3=elem1,elem2,elem3",
                               nargs="+", type=str)
    
    ##### Visualize arguments #####
    command_visualize = subcommands.add_parser(Command.VISUALIZE.value, 
                                           help="Dataset visualization command")
    command_visualize.add_argument("--dataset-name",
                                   help="The dataset name to visualize",
                                   type=str, metavar="PATH", required=True)
    command_visualize.add_argument("--type-visu", help="The type of visualization",
                                   choices=["console"], default="console", type=str)

    args = parser.parse_args()
    
    ################################################################################################
    ####################################### PROJECT ENV ############################################
    ################################################################################################
    userdir : Path = Path(args.userdir)
    try:
        cgrbi = CGrbi(project_directory=userdir, create_userdir=args.create_userdir)
    except:
        raise Exception(f"userdir : {userdir} doesn't exist yet. Add '--create-userdir' to create"+
                        " userdir repository or find a correct path.")
    cgrbi.log.info(f"Arguments passé : {args}")
    
    kwargs = vars(args).copy()
    kwargs.pop("command")
    kwargs.pop("create_userdir")
    ###################################  CGrbi Build Dataset #######################################
    if args.command == Command.BUILD.value:
        
        kwargs["preprocess"] = PreProcessDict[args.preprocess]
        if args.indices:
            kwargs["indices"] = [Indice(indice) for indice in args.indices]
        build_dataset(datadir=cgrbi.data_dir, **kwargs)
    
    ###############################  CGrbi Visualize Dataset #######################################
    elif args.command == Command.VISUALIZE.value:
        dataset_filepath = cgrbi.data_dir.joinpath(args.dataset_name).with_suffix(EXTENSION_FILE)
        cgrbi.check_file(dataset_filepath, cgrbi.data_dir)
        dataset = read_wtseries_training(dataset_filepath)
        print(dataset("train", "input")[5])

    ###############################  CGrbi Train Neural Network ####################################  
    elif args.command == Command.TRAIN.value:
        if args.training_name is None:
            kwargs["training_name"] = args.trainer_name
        
        launch_machine_learning(project=cgrbi, **kwargs)
    
    ########################################## EXIT ################################################
    cgrbi.exit()
    
if __name__ == "__main__":
    main()