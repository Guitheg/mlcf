import pandas as pd
from os.path import join, isfile

from pathlib import Path
from typing import Dict, List
from collections import OrderedDict
from mlcf.utils import ListEnum
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

# MLCF modules
from mlcf.envtools.hometools import ProjectHome
from mlcf.envtools.pathtools import create_path


class InfofileColumns(ListEnum):
    TRAINING: str = "Training name"
    MODEL: str = "Model name"
    OPTI: str = "Optimizer"
    PARAM_OPTI: str = "Param optimizer"
    LOSS_FCT: str = "Loss function"
    N_EPOCH: str = "Number of trained epochs"
    LAST_SCORE: str = "Last evaluation loss"
    PATH_LAST_CHECKPOINT = "Last checkpoint filepath"
    LOG_DIR = "Board log dir"


class TrainingHome(ListEnum):
    INFOFILE: str = "TrainingInfo.csv"
    HOME: str = "Training"
    CHECKPOINT: str = "checkpoints"
    TENSORBOARD_LOGS_NAME: str = "boards"


INFOFILE: str = TrainingHome.INFOFILE.value
HOME: str = TrainingHome.HOME.value
CHECKPOINT: str = TrainingHome.CHECKPOINT.value
TENSORBOARD_LOGS_NAME: str = TrainingHome.TENSORBOARD_LOGS_NAME.value


class CheckpointDict(ListEnum):
    MODEL_STATE: str = "model_state"
    EPOCH: str = "epoch"
    LOSS: str = "logs"
    OPTI_STATE: str = "optimizer_state"


MODEL_STATE = CheckpointDict.MODEL_STATE.value
EPOCH = CheckpointDict.EPOCH.value
LOSS = CheckpointDict.LOSS.value
OPTI_STATE = CheckpointDict.OPTI_STATE.value


def get_empty_infofile() -> pd.DataFrame:
    infofile = pd.DataFrame(columns=InfofileColumns.list_value())
    infofile.set_index(InfofileColumns.TRAINING.value, inplace=True)
    return infofile


class TrainingManager(object):
    def __init__(self,
                 model,
                 project: ProjectHome = None,
                 disable_warning: bool = True):
        self.has_training_manager = False
        self.disable_warning = disable_warning
        self.model = model
        if project is not None:
            self.project = project

            self.home_path = self.project.get_dir()
            self.training_home = self.home_path.joinpath(HOME)
            self.infofile_path = self.training_home.joinpath(INFOFILE)
            self.checkpoint_path = create_path(str(self.training_home), CHECKPOINT,
                                               f"checkpoints_{self.model.training_name}")
            self.board_path = create_path(str(self.training_home), TENSORBOARD_LOGS_NAME,
                                          f"boards_{self.model.training_name}")
            now = datetime.now()
            self.now = str(now.strftime("%Y-%d%b%Hh%Mm%S"))

            self.board_name = f"board_{self.now}_{self.model.training_name}"
            self.board = SummaryWriter(self.board_path.joinpath(self.board_name))
            self.has_training_manager = True
            self.load_infofile()
            print(self.infofile)

    def load_infofile(self) -> pd.DataFrame:
        if isfile(self.infofile_path):
            self.infofile: pd.DataFrame = pd.read_csv(
                self.infofile_path,
                index_col=InfofileColumns.TRAINING.value)
        else:
            self.infofile = get_empty_infofile()
        return self.infofile

    def get_last_checkpoint_path(self) -> Path:
        if len(self.infofile) == 0:
            raise Exception("Infofile doesn't exist yet")
        path = InfofileColumns.PATH_LAST_CHECKPOINT.value
        return self.infofile.loc[self.model.training_name][path]

    def load_checkpoint(self, resume_training: bool = False) -> None:
        """Reload a checkpoint (resume the training or not following {resume_training})

        Args:
            path (Path): Path to the checkpoint file
            resume_training (bool, optional): If we want to resume the training. Defaults to False.
        """
        if self.exist():
            checkpoint_path = self.get_last_checkpoint_path()
            self.checkpoint_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(self.checkpoint_dict[MODEL_STATE])
            self.model.optimizer.load_state_dict(self.checkpoint_dict[OPTI_STATE])
            self.model.epoch = self.checkpoint_dict[EPOCH]
            self.model.logs = self.checkpoint_dict[LOSS]
            self.info(f"Loading of the checkpoint [epoch:{self.model.epoch}]:{checkpoint_path}")
            if resume_training:
                self.debug("Resuming training...")
                self.model.train()
            else:
                if self.project:
                    self.debug("Evaluation...")
                self.model.eval()
        else:
            if not self.disable_warning:
                raise Warning("Trying to load a checkpoint but TrainingManager has no ProjectHome")

    def checkpoint(self,
                   logs: List[OrderedDict],
                   epoch: int,
                   model_state: Dict,
                   optimizer_state: Dict) -> None:
        """Save the checkpoint in a file

        Args:
            logs (List[OrderedDict]): Loss to save
            epoch (int): The current epoch to save
            model_state (Dict): the current state to save
            optimizer_state (Dict): the optimizer current state to save
        """
        if self.exist():
            self.model.epoch = epoch
            self.model.logs = logs
            checkpoint_path = join(self.checkpoint_path,
                                   f"checkpoint_{self.model.training_name}_{self.now}.pt")
            self.checkpoint_dict = {EPOCH: epoch,
                                    MODEL_STATE: model_state,
                                    OPTI_STATE: optimizer_state,
                                    LOSS: logs}
            torch.save(self.checkpoint_dict, checkpoint_path)
            self.info(f"Saving ckeckpoint [epoch:{epoch}]: {checkpoint_path}")
            self._update_infofile(str(epoch), str(logs[-1]["loss"]))
        else:
            if not self.disable_warning:
                raise Warning("Trying to save checkpoint but TrainingManager has no ProjectHome")

    def _update_infofile(self, n_epoch: str,
                         last_score: str):
        checkpoint_name = f"checkpoint_{self.model.training_name}_{self.now}.pt"
        last_checkpoint_path = join(self.checkpoint_path, checkpoint_name)

        list_keys = InfofileColumns.list_value()
        list_value = [
            self.model.training_name,
            self.model.model_name,
            self.model.optimizer.__class__.__name__,
            str(self.model.optimizer.state_dict()["param_groups"]),
            self.model.loss.__class__.__name__,
            n_epoch,
            last_score,
            last_checkpoint_path,
            self.board.log_dir
        ]

        update_dict = {k: v for k, v in zip(list_keys, list_value)}

        self.infofile.loc[self.model.training_name] = update_dict
        self.infofile.to_csv(self.infofile_path, index_label=InfofileColumns.TRAINING.value)

    def tensorboard_stream(self, log: OrderedDict, num_epoch: int) -> None:
        if self.exist():
            for key in log:
                self.board.add_scalar(key, log.get(key), num_epoch)
            self.board.close()
        else:
            if not self.disable_warning:
                raise Warning("Trying to save stream a tensorboard" +
                              " but TrainingManager has no ProjectHome")

    def info(self, msg: str) -> None:
        if self.exist():
            self.project.log.info(self._prefix_msg_log(msg))
        else:
            if not self.disable_warning:
                raise Warning("Trying to save stream log info" +
                              " but TrainingManager has no ProjectHome")

    def debug(self, msg: str) -> None:
        if self.exist():
            self.project.log.debug(self._prefix_msg_log(msg))
        else:
            if not self.disable_warning:
                raise Warning("Trying to save stream log debug " +
                              "but TrainingManager has no ProjectHome")

    def _prefix_msg_log(self, msg: str) -> str:
        """Just add a prefix to log messages

        Args:
            msg (str): The message

        Returns:
            str: prefix + message
        """
        return f"[{self.model.training_name}] - {msg}"

    def exist(self) -> bool:
        return self.has_training_manager
