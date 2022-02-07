from enum import Enum
from os.path import join
from pathlib import Path
from syslog import LOG_SYSLOG
from typing import Callable, Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor, no_grad, device, cuda, zeros
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from datatools.wtseries_tensor import WTSeriesTensor
from datatools.wtseries_training import WTSeriesTraining, TRAIN, VALIDATION, TEST
from ai.log import ProgressBar, add_metrics_to_log, log_to_message
from envtools.path import create_path
from datetime import datetime
from envtools.project import Project

class TrainingHome(Enum):
    HOME : str = "Training"
    DATA : str = "data"
    LOGS : str = "logs"
    MODELS : str = "models"
    CHECKPOINT : str = "checkpoints"
    TENSORBOARD_LOGS_NAME : str = "boards"

class CheckpointDict(Enum):
    MODEL_STATE : str = "model_state"
    EPOCH : str = "epoch"
    LOSS : str = "logs"
    OPTI_STATE : str = "optimizer_state"
    
HOME : str = TrainingHome.HOME.value
DATA : str = TrainingHome.DATA.value
LOGS : str = TrainingHome.LOGS.value
MODELS : str = TrainingHome.MODELS.value
CHECKPOINT : str = TrainingHome.CHECKPOINT.value
TENSORBOARD_LOGS_NAME : str = TrainingHome.TENSORBOARD_LOGS_NAME.value
MODEL_STATE = CheckpointDict.MODEL_STATE.value
EPOCH = CheckpointDict.EPOCH.value
LOSS = CheckpointDict.LOSS.value
OPTI_STATE = CheckpointDict.OPTI_STATE.value
    
class SuperModule(Module):    
    def __init__(self,
                 name : str = None,
                 project : Project = None,
                 use_tensorboard : bool = False,
                 use_checkpoint : bool = False,
                 *args, **kwargs):
        """SuperModule mother class to instance supermodule modeles child. A SuperModule allows to
        handle models version, checkpoints, tensorboard stream, logging and configurations. Moreover
        it provide the function fit(), predict(), and init() to initialize and train a modele.

        Args:
            name (str, optional): The name of the modele. Defaults to None.
            project (Project, optional): A Project object which provide logging and config. 
            Defaults to None.
            use_tensorboard (bool, optional): If we want to use tensorboard. Defaults to False.
            use_checkpoint (bool, optional): If we want to use a checkpoint. Defaults to False.
        """
        super(SuperModule, self).__init__()
        
        self.name : str = name if not name is None else self.__class__.__name__
        
        self.use_project : bool = not project is None
        self.project = project
        self.use_checkpoint : bool
        self.use_tensorboard : bool
        if self.use_project:
            self.use_checkpoint = self.project.cfg.get("Training", "Checkpoint") == "True"
            self.use_tensorboard = self.project.cfg.get("Training", "Tensorboard") == "True"
        else:
            self.use_checkpoint = use_checkpoint
            self.use_tensorboard = use_tensorboard
        
        if self.use_project:
            self.home_path = self.project.get_dir()
            self.training_home = join(self.home_path, HOME)
        else:
            self.training_home = HOME
        
        self.filehandling = self.use_checkpoint or self.use_tensorboard or self.use_project
        
        if self.filehandling :
            self.logs_path = create_path(self.training_home, LOGS)
            self.models_path = create_path(self.training_home, MODELS)
            self.checkpoint_path = create_path(self.models_path, CHECKPOINT, 
                                               f"checkpoints_{self.__class__.__name__}")
            self.data_path = create_path(self.training_home, DATA)
            self.board_path = create_path(self.logs_path, TENSORBOARD_LOGS_NAME, 
                                          f"boards_{self.__class__.__name__}")
            now = datetime.now()
            self.now = str(now.strftime("%Y-%d-%b_%Hh_%Mm_%S"))
        
        self.board = None
        if self.use_tensorboard:
            self.board_name = f"board_{self.now}_{self.name}"
            self.board = SummaryWriter(join(self.board_path, self.board_name))
        
        self.epoch : int = 0
        self.logs : List[OrderedDict] = []
        self.optimizer : torch.optim.Optimizer = None
        self.loss = None
        self.metrics : List[Callable] = None
        self.initialize : bool = False
        
    def _prefix_msg_log(self, msg : str) -> str:
        """Just add a prefix to log messages

        Args:
            msg (str): The message

        Returns:
            str: prefix + message
        """
        return f"[{self.name}] - {msg}"
    
    def load(self, path : Path) -> None:
        """Set the parameters of the model by loading parameters from a file

        Args:
            path (Path): Path to the file
        """
        self.load_state_dict(torch.load(path))
        self.eval()
        if self.use_project:
            self.project.log.info(self._prefix_msg_log(f"Chargement du modèle : {path}"))
    
    def save(self, path : str):
        """save the parameters of the model in a file

        Args:
            path (str): path to the file
        """
        if self.filehandling :
            save_path = path+".pt"
            if self.use_project:
                self.project.log.info(self._prefix_msg_log(f"Sauvegarde du modèle : {save_path}"))
            path = path.split('.')[0]
            torch.save(self.state_dict(), save_path)
    
    def load_checkpoint(self, path : Path, resume_training : bool = False):
        """Reload a checkpoint (resume the training or not following {resume_training})

        Args:
            path (Path): Path to the checkpoint file
            resume_training (bool, optional): If we want to resume the training. Defaults to False.
        """
        checkpoint_path = path
        self.checkpoint_dict = torch.load(checkpoint_path)
        self.load_state_dict(self.checkpoint_dict[MODEL_STATE])
        self.optimizer.load_state_dict(self.checkpoint_dict[OPTI_STATE])
        self.epoch = self.checkpoint_dict[EPOCH]
        self.logs = self.checkpoint_dict[LOSS]
        if self.use_project:
            self.project.log.info(
                self._prefix_msg_log(
                    f"Chargement du checkpoint à l'epoch [{self.epoch}] : {path}"))
        if resume_training:
            if self.use_project:
                self.project.log.debug(
                    self._prefix_msg_log("Reprise de l'entrainement..."))
            self.train()
        else:
            if self.use_project:
                self.project.log.debug(
                    self._prefix_msg_log("Evaluation en cours..."))
            self.eval()
        
    def checkpoint(self, 
                   logs : List[OrderedDict], 
                   epoch : int,
                   model_state : Dict,
                   optimizer_state : Dict):
        """Save the checkpoint in a file

        Args:
            logs (List[OrderedDict]): Loss to save
            epoch (int): The current epoch to save
            model_state (Dict): the current state to save
            optimizer_state (Dict): the optimizer current state to save
        """
        self.epoch = epoch
        self.logs = logs
        checkpoint_path = join(self.checkpoint_path, f"checkpoint_{self.name}_{self.now}")
        self.checkpoint_dict = {EPOCH : self.epoch,
                                MODEL_STATE : model_state,
                                OPTI_STATE : optimizer_state,
                                LOSS : self.logs}
        torch.save(self.checkpoint_dict, checkpoint_path)
        if self.use_project:
            self.project.log.info(
                self._prefix_msg_log(
                    f"Checkpoint sauvé à l'epoch [{self.epoch}]: {checkpoint_path}"))

    def set_device(self, device_str : str):
        """Set the device to CPU or CUDA

        Args:
            device_str (str): the name of the process unit wanted to be used
        """
        if device_str == "cpu":   
            self.device = device("cpu")
        elif device_str == "cuda":
            self.device = device("cuda:0" if cuda.is_available() else "cpu")
        if self.use_project:
            self.project.log.info(self._prefix_msg_log(f"  -Processeur utilisé : {self.device}"))
        self.to(self.device)
        
    def init(self, 
             loss : Callable, 
             optimizer : torch.optim.Optimizer, 
             device_str : str = "cuda", 
             metrics : List[Callable] = None):
        """Initialize the model given a loss function, an optimizer, 
        the device and a list of metrics

        Args:
            loss (Callable): A loss function to compute the loss
            optimizer (torch.optim.Optimizer): An optimizer to train the model
            device_str (str, optional): The process unit we want to use. Defaults to "cuda".
            metrics (List[Callable], optional): A list of metrics function. Defaults to None.
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        if self.use_project:
            self.project.log.info(self._prefix_msg_log(f"Initialisation du modèle :"))
            self.project.log.info(
                self._prefix_msg_log(f"  -Loss : {self.loss.__class__.__name__}"))
            self.project.log.info(
                self._prefix_msg_log(f"  -Optimizer : {self.optimizer.__class__.__name__}"))
            aff_metric = lambda met : [f.__name__ for f in met]
            self.project.log.info(self._prefix_msg_log(f"  -Metrics : {aff_metric(self.metrics)}"))
        self.set_device(device_str)
        self.initialize = True
        
    def summary(self) -> str:
        """Print the summary of the models
        """
        print(self.parameters)
        if self.use_project:
            self.project.log.info(self._prefix_msg_log(f"\nSommaire du modèle : {self.parameters}"))
        return str(self.parameters)
    
    def fit(self,
            dataset : WTSeriesTraining,
            n_epochs : int, 
            batchsize : int, 
            shuffle : bool = True, 
            talkative : bool = True,
            evaluate : bool = False,
            tensorboard : bool = False,
            checkpoint : bool = None) -> Tuple[List[OrderedDict], OrderedDict]:
        """Fit/train the model (need to be initialized first)

        Args:
            dataset (WTSeriesTraining): The dataset used to train (and evaluate the model)
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
        if self.use_project:
            self.project.log.info(
                self._prefix_msg_log(
                    f"Fit : Nombre d'epoch : {n_epochs}, Taille des batchs : {batchsize}, "+\
                    f"Mélange : {shuffle}, Evaluation : {evaluate}"))
            self.project.log.info(
                self._prefix_msg_log(
                    f"Dataset utilisée : {dataset}"))
        
        if not self.initialize:
            raise Exception("The module has not been compiled")
        
        train_data = WTSeriesTensor(TRAIN, ts_data=dataset, 
                                    transform_x=self.transform_x,
                                    transform_y=self.transform_y)
        validation_data = WTSeriesTensor(VALIDATION, ts_data=dataset, 
                                    transform_x=self.transform_x,
                                    transform_y=self.transform_y)
        test_data = WTSeriesTensor(TEST, ts_data=dataset, 
                                    transform_x=self.transform_x,
                                    transform_y=self.transform_y)
        if self.use_project:
            self.project.log.debug(
                self._prefix_msg_log(
                    f"Size train tensor (x,y) : {train_data.x_size()}, {train_data.y_size()}"+\
                    f"Size validation tensor (x,y): {validation_data.x_size()}, "
                                                f"{validation_data.y_size()}"+\
                    f"Size test tensor (x,y): {test_data.x_size()}, {test_data.y_size()}"))
        
        train_loader = train_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)
        validation_loader = validation_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)
        test_loader = test_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)
        
        logs : List[OrderedDict] = self.logs
        for num_epoch in range(self.epoch, n_epochs):
            log : OrderedDict = OrderedDict()
            pb = None
            
            if self.use_project:
                self.project.log.info( self._prefix_msg_log(f"Epoch {num_epoch+1} / {n_epochs}"))
                
            if talkative:
                print(f"Epoch {num_epoch+1} / {n_epochs} :")
                pb = ProgressBar(len(train_loader))  
                
            log = self.fit_one_epoch(train_loader, log, talkative, pb)
            log = self.validate(validation_loader, log)
            
            logs.append(log)
            
            msg = log_to_message(log)
            
            if talkative:
                pb.close(msg)
            
            if self.filehandling:
                if self.use_project:
                    self.project.log.info(self._prefix_msg_log(msg))
                
                if self.use_tensorboard and tensorboard:
                    for l in log:
                        self.board.add_scalar(l, log[l], num_epoch)
                    self.board.close()
                    
                if checkpoint is None:
                    checkpoint = self.use_checkpoint
                    
                if checkpoint:
                    self.checkpoint(logs, num_epoch, self.state_dict(), self.optimizer.state_dict())
               
        log_eval = None     
        if evaluate:
            log_eval = self.validate(test_loader, OrderedDict(), validation_type = "test")
            
            if self.use_project:
                self.project.log.info(
                    self._prefix_msg_log(
                        f"Log évaluation : {log_to_message(log_eval)}"))
          
        return logs, log_eval
        
    def fit_one_epoch(self, 
                      dataloader : DataLoader, 
                      log : OrderedDict, 
                      talkative : bool,
                      pb : ProgressBar = None) -> OrderedDict:
        """Fit one epoch given a dataloader

        Args:
            dataloader (DataLoader): The train dataloader
            log (OrderedDict): the log history used to store losses
            talkative (bool): If we want to print progress
            pb (ProgressBar, optional): Advanced printing with a loading bar. Defaults to None.

        Returns:
            [OrderedDict]: update log
        """
        self.train()
        log, labels, predictions = self._run_model(dataloader, log, "train", talkative, pb) 
        if self.metrics:
            add_metrics_to_log(log, self.metrics, labels, predictions)
        return log
    
    def validate(self,
                 dataloader : DataLoader, 
                 log : OrderedDict, 
                 validation_type : str = "validation") -> OrderedDict:
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
        else : 
            raise Exception("Unknown validation type")
    
        self.eval()
        with no_grad():
            log, labels, predictions = self._run_model(dataloader, log, validation_type, False)
            
        if self.metrics:    
            add_metrics_to_log(log, self.metrics, labels, predictions, prefix=prefix)
            
        return log
    
    def _run_model(self,  
                   dataloader : DataLoader, 
                   log : OrderedDict, 
                   type_batchrun : str, 
                   talkative : bool = False,
                   pb : ProgressBar = None) -> Tuple[OrderedDict, Tensor, Tensor]:
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
            Tuple[OrderedDict, Tensor, Tensor] : the update log, and the labels and prediction if
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
        else : 
            raise Exception("Unknown runbatch type")
        
        loss_name = prefix+loss_name
        loss = 0.0
        predictions = []
        labels = []
        N = len(dataloader.dataset)
        
        for batch_index, (batch_inputs, batch_labels) in enumerate(dataloader):
            if type_batchrun == 'train':
                self.optimizer.zero_grad()
            
            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Propagation avant
            batch_outputs = self(batch_inputs)
            batch_loss = self.loss(batch_outputs, batch_labels)
            
            loss += batch_loss.item()
            log[loss_name] = float(loss) / (batch_index + 1)
            
            if type_batchrun == 'train':
                # message log
                if talkative and not pb is None:
                    pb.bar(batch_index, log_to_message(log))
                    
                # Propagation arrière
                batch_loss.backward()
                self.optimizer.step()
            
            if self.metrics:
                if batch_index == 0:
                    predictions = zeros((N,) + batch_outputs.shape[1:])
                    labels = zeros((N,) + batch_labels.shape[1:])
                index = batch_index*dataloader.batch_size
                predictions[index : min(N, index + dataloader.batch_size)] = batch_outputs
                labels[index : min(N, index + dataloader.batch_size)] = batch_labels
                
        return log, labels, predictions

    def transform_x(self, x : Tensor) -> Tensor:
        """Transform function of input data
        Is used by respecifying it in the child class

        Args:
            x (Tensor): input data

        Returns:
            [Tensor]: Modified input data
        """
        return x
    
    def transform_y(self, y : Tensor) -> Tensor:
        """Transform function of target data
        Is used by respecifying it in the child class

        Args:
            x (Tensor): target data

        Returns:
            [Tensor]: Modified target data
        """
        return y

    def predict(self, batch_inputs : Tensor) -> Tensor:
        """Give a prediction of the model given a input batch tensor

        Args:
            batch_inputs (Tensor): The input batch tensor

        Returns:
            Tensor: the output batch tensor prediction
        """
        batch_inputs = batch_inputs.to(self.device)
        return self(batch_inputs).data