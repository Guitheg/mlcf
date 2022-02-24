from __future__ import annotations
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor, no_grad, device, cuda, zeros

from collections import OrderedDict

from time import time_ns

# MLCF modules
from mlcf.datatools.wtseries_tensor import WTSeriesTensor
from mlcf.datatools.wtst import Partition, WTSTraining
from mlcf.aitools.log import ProgressBar, add_metrics_to_log, log_to_message
from mlcf.envtools.hometools import ProjectHome
from mlcf.aitools.training_manager import TrainingManager


def select_list_index_columns(list_to_select: List[str], list_all_collumns: List[str]):
    list_index = []
    for e in list_to_select:
        list_index.append(list_all_collumns.index(e))
    return list_index


class SuperModule(Module):
    def __init__(self, *args, **kwargs):
        """SuperModule mother class to instance supermodule modeles child. A SuperModule allows to
        handle models version, checkpoints, tensorboard stream, logging and configurations. Moreover
        it provide the function fit(), predict(), and init() to initialize and train a modele.

        Args:
            name (str, optional): The name of the modele. Defaults to None.
            project (ProjectHome, optional): A ProjectHome object which provide logging and config.
            Defaults to None.
        """
        super(SuperModule, self).__init__()
        self.model_name = self.__class__.__name__
        self.epoch: int = 0
        self.logs: List[OrderedDict] = []
        self.optimizer: torch.optim.Optimizer = None
        self.loss = None
        self.metrics: List[Callable] = None
        self.initialize: bool = False

    def summary(self) -> str:
        """Print the summary of the models
        """
        print(self.parameters)
        if self.manager:
            self.manager.info(f"\nModèle:\n{self.parameters}")
        return str(self.parameters)

    def load(self, path: Path) -> None:
        """Set the parameters of the model by loading parameters from a file

        Args:
            path (Path): Path to the file
        """
        self.load_state_dict(torch.load(path))
        self.eval()
        if self.manager:
            self.manager.info(f"Chargement du modèle: {path}")

    def save(self, path: str):
        """save the parameters of the model in a file

        Args:
            path (str): path to the file
        """
        save_path = path+".pt"
        path = path.split('.')[0]
        torch.save(self.state_dict(), save_path)
        if self.manager:
            self.manager.info(f"Sauvegarde du modèle: {save_path}")

    def set_device(self, device_str: str):
        """Set the device to CPU or CUDA

        Args:
            device_str (str): the name of the process unit wanted to be used
        """
        if device_str == "cpu":
            self.device = device("cpu")
        elif device_str == "cuda":
            self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.to(self.device)
        self.manager.info(f"  -Processeur utilisé: {self.device}")

    def init_load_checkpoint(
        self,
        training_name: str,
        loss: Callable,
        optimizer: torch.optim.Optimizer,
        device_str: str = "cuda",
        metrics: List[Callable] = None,
        project: ProjectHome = None,
        resume_training: bool = False
    ):
        self.init(
            loss,
            optimizer,
            device_str,
            metrics,
            training_name,
            project
        )
        self.initialize = False
        self.manager.load_checkpoint(resume_training=resume_training)
        self.initialize = True

    def init(self,
             loss: Callable,
             optimizer: torch.optim.Optimizer,
             device_str: str = "cuda",
             metrics: List[Callable] = None,
             training_name: str = None,
             project: ProjectHome = None,
             *args, **kwargs):
        """Initialize the model given a loss function, an optimizer,
        the device and a list of metrics

        Args:
            loss (Callable): A loss function to compute the loss
            optimizer (torch.optim.Optimizer): An optimizer to train the model
            device_str (str, optional): The process unit we want to use. Defaults to "cuda".
            metrics (List[Callable], optional): A list of metrics function. Defaults to None.
        """
        self.training_name = training_name if training_name is not None else self.model_name
        self.manager = TrainingManager(model=self, project=project)
        self.loss = loss
        self.optimizer = optimizer
        if metrics:
            self.metrics = metrics
        self.manager.info("Model initialisation:")
        self.manager.info(f"  -Loss: {self.loss.__class__.__name__}")
        self.manager.info(f"  -Optimizer: {self.optimizer.__class__.__name__}")
        if metrics:
            self.manager.info(f"  -Metrics: {[f.__name__ for f in self.metrics]}")
        else:
            self.manager.info("  -Metrics: Any")
        self.set_device(device_str)
        self.initialize = True

    def fit(self,
            dataset: WTSTraining,
            n_epochs: int,
            batchsize: int,
            shuffle: bool = False,
            talkative: bool = True,
            evaluate: bool = False,
            tensorboard: bool = False,
            checkpoint: bool = False,
            *args, **kwargs) -> Tuple[List[OrderedDict], Optional[OrderedDict]]:
        """Fit/train the model (need to be initialized first)

        Args:
            dataset (WTSTraining): The dataset used to train (and evaluate the model)
            n_epochs (int): The number of epochs
            batchsize (int): The batch size
            shuffle (bool, optional): If we want to shuffle the data. Defaults to True.
            talkative (bool, optional): To print the progress. Defaults to True.
            evaluate (bool, optional): If we want to evalueate at the end. Defaults to False.
            tensorboard (bool, optional): If we want to stream loss on tensorboard.
            Defaults to False.
            checkpoint (bool, optional): if we want to use checkpoints. Defaults to None.

        Raises:
            Exception: If The module has not been compiled

        Returns:
            Tuple[List[OrderedDict], OrderedDict]: the list of loss, and the loss of the evaluation
        """
        self.manager.info(f"Fit: Number of epochs: {n_epochs}, Batchsize: {batchsize}, " +
                          f"Shuffle: {shuffle}, Evaluation: {evaluate}")
        self.manager.info(f"Dataset: {dataset}")

        if not self.initialize:
            raise Exception("The module has not been compiled")

        train_data = WTSeriesTensor(
            ts_data=dataset,
            partition=Partition.TRAIN,
            transform_x=self.transform_x,
            transform_y=self.transform_y
        )
        validation_data = WTSeriesTensor(
            ts_data=dataset,
            partition=Partition.VALIDATION,
            transform_x=self.transform_x,
            transform_y=self.transform_y
        )
        test_data = WTSeriesTensor(
            ts_data=dataset,
            partition=Partition.TEST,
            transform_x=self.transform_x,
            transform_y=self.transform_y
        )

        self.manager.debug(
                f"\nSize train tensor (x,y): {train_data.x_size()}, {train_data.y_size()}" +
                f"\nSize validation tensor (x,y): {validation_data.x_size()}, "
                f"{validation_data.y_size()}" +
                f"\nSize test tensor (x,y): {test_data.x_size()}, {test_data.y_size()}")

        train_loader = train_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)
        validation_loader = validation_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)
        test_loader = test_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)

        logs: List[OrderedDict] = self.logs
        for num_epoch in range(self.epoch, n_epochs):
            log: OrderedDict = OrderedDict()
            pb = None
            self.manager.info(f"Epoch {num_epoch+1} / {n_epochs}")
            if talkative:
                print(f"Epoch {num_epoch+1} / {n_epochs}:")
                pb = ProgressBar(len(train_loader))

            log = self.fit_one_epoch(train_loader, log, talkative, pb)
            log = self.validate(validation_loader, log)

            logs.append(log)

            msg = log_to_message(log)

            if talkative and pb:
                pb.close(msg)

            self.manager.info(msg)

            if tensorboard:
                self.manager.tensorboard_stream(log, num_epoch=num_epoch)

            if checkpoint:
                self.manager.checkpoint(logs,
                                        num_epoch,
                                        self.state_dict(),
                                        self.optimizer.state_dict())

        log_eval = None
        if evaluate:
            log_eval = self.validate(test_loader, OrderedDict(), validation_type="test")
            self.manager.info(f"Evaluation loss results: {log_to_message(log_eval)}")

        return logs, log_eval

    def fit_one_epoch(self,
                      dataloader: DataLoader,
                      log: OrderedDict,
                      talkative: bool,
                      pb: ProgressBar = None) -> OrderedDict:
        """Fit one epoch given a dataloader

        Args:
            dataloader (DataLoader): The train dataloader
            log (OrderedDict): the log history used to store losses
            talkative (bool): If we want to print progress
            pb (ProgressBar, optional): Advanced printing with a loading bar. Defaults to None.

        Returns:
            [OrderedDict]: update log
        """
        if not self.initialize:
            raise Exception("The module has not been compiled")

        self.train()
        log, labels, predictions = self._run_model(dataloader, log, "train", talkative, pb)
        if self.metrics:
            add_metrics_to_log(log, self.metrics, labels, predictions)
        return log

    def validate(self,
                 dataloader: DataLoader,
                 log: OrderedDict,
                 validation_type: str = "validation") -> OrderedDict:
        """Run one epoch but in validation / evaluation mode

        Args:
            dataloader (DataLoader): the test/val dataloader
            log (OrderedDict): the log history used to store losses
            validation_type (str, optional): The type of evaluation ("test" or "validation").
            Defaults to "validation".

        Raises:
            Exception: Unknown validation type

        Returns:
            [OrderedDict]: update log
        """

        if validation_type == "test":
            prefix = "eval_"
        elif validation_type == "validation":
            prefix = "val_"
        else:
            raise Exception("Unknown validation type")

        self.eval()
        with no_grad():
            log, labels, predictions = self._run_model(dataloader, log, validation_type, False)

        if self.metrics:
            add_metrics_to_log(log, self.metrics, labels, predictions, prefix=prefix)

        return log

    def _run_model(self,
                   dataloader: DataLoader,
                   log: OrderedDict,
                   type_batchrun: str,
                   talkative: bool = False,
                   pb: ProgressBar = None) -> Tuple[OrderedDict, Tensor, Tensor]:
        """The core function to run the model

        Args:
            dataloader (DataLoader): A dataloader wich will give data to the model
            log (OrderedDict): the log history
            type_batchrun (str): the type of run "train", "validation", or "test"
            talkative (bool, optional): If we want to print progress
            pb (ProgressBar, optional): Advanced printing with a loading bar. Defaults to None.

        Raises:
            Exception: Unknown runbatch type

        Returns:
            Tuple[OrderedDict, Tensor, Tensor]: the update log, and the labels and prediction if
            metrics is used
        """
        loss_name = "loss"
        prefix = ""
        if type_batchrun == "train":
            pass
        elif type_batchrun == "validation":
            prefix = "val_"
        elif type_batchrun == "test":
            prefix = "test_"
        else:
            raise Exception("Unknown runbatch type")

        loss_name = prefix+loss_name
        loss = 0.0
        time_step = 0.0
        predictions: Tensor = Tensor()
        labels: Tensor = Tensor()
        N = len(dataloader.dataset)

        batch_index: int
        batch_inputs: Tensor
        batch_labels: Tensor
        for batch_index, (batch_inputs, batch_labels) in enumerate(dataloader):
            batch_size: int = len(batch_inputs)

            t0 = time_ns()  # start evaluation duration step --------------------------------START--
            if type_batchrun == 'train':
                self.optimizer.zero_grad()

            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Propagation avant
            batch_outputs: Tensor
            batch_loss: Tensor
            batch_outputs, batch_loss = self.feedforward(batch_inputs=batch_inputs,
                                                         batch_labels=batch_labels)

            if type_batchrun == 'train':
                # Propagation arrière
                self.feedbackward(batch_loss)

            t1 = time_ns()  # end evaluation duration step -----------------------------------STOP--
            loss += batch_loss.item()
            time_step += t1 - t0
            time_step_in_sec = time_step / 1_000_000_000  # nanosec: 10^-9
            log["sec/step"] = time_step_in_sec / (batch_index + 1)
            log[loss_name] = float(loss) / (batch_index + 1)

            # message log
            if type_batchrun == 'train' and talkative and pb is not None:
                pb.bar(batch_index, log_to_message(log))

            if self.metrics:
                if batch_index == 0:  # init if batch_index = 0
                    predictions = zeros((N,) + batch_outputs.shape[1:])
                    labels = zeros((N,) + batch_labels.shape[1:])
                index: int = batch_index*batch_size
                predictions[index: min(N, index + batch_size)] = batch_outputs
                labels[index: min(N, index + batch_size)] = batch_labels

        return log, labels, predictions

    def feedforward(self, batch_inputs, batch_labels):
        batch_outputs = self(batch_inputs)
        batch_loss = self.loss(batch_outputs, batch_labels)
        return batch_outputs, batch_loss

    def feedbackward(self, batch_loss):
        batch_loss.backward()
        self.optimizer.step()

    def transform_x(self, x: Tensor) -> Tensor:
        """Transform function of input data
        Is used by respecifying it in the child class

        Args:
            x (Tensor): input data

        Returns:
            [Tensor]: Modified input data
        """
        return x

    def transform_y(self, y: Tensor) -> Tensor:
        """Transform function of target data
        Is used by respecifying it in the child class

        Args:
            x (Tensor): target data

        Returns:
            [Tensor]: Modified target data
        """
        return y

    def predict(self, batch_inputs: Tensor) -> Tensor:
        """Give a prediction of the model given a input batch tensor

        Args:
            batch_inputs (Tensor): The input batch tensor

        Returns:
            Tensor: the output batch tensor prediction
        """
        x = self.transform_x(batch_inputs)
        x = x.to(self.device)
        return self(x).data
