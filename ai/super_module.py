from os.path import join
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import no_grad, device, cuda, zeros
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter
from datatools.wtseries_tensor import WTSeriesTensor
from datatools.wtseries_training import WTSeriesTraining, TRAIN, VALIDATION, TEST
from ai.log import ProgressBar, add_metrics_to_log, log_to_message
from envtools.path import create_path
from datetime import datetime

from envtools.project import Project

TRAINING_HOME = "Training"
TRAINING_DATA = "data"
TRAINING_LOGS = "logs"
TRAINING_MODELS = "models"
TRAINING_CHECKPOINT = "checkpoints"
TENSORBOARD_LOGS_NAME = "boards"

class SuperModule(Module):
    
    def __init__(self,
                 name : str = None,
                 project : Project = None,
                 use_tensorboard : bool = False,
                 use_checkpoint : bool = False,
                 *args, **kwargs):
        super(SuperModule, self).__init__()
        
        self.name = name if not name is None else self.__class__.__name__
        self.use_checkpoint = use_checkpoint
        self.use_tensorboard = use_tensorboard
        if not project is None:
            self.home_path = self.project.get_dir()
            self.training_home = join(self.home_path, TRAINING_HOME)
        else:
            self.training_home = TRAINING_HOME
            
        self.logs_path = create_path(self.training_home, TRAINING_LOGS)
        self.models_path = create_path(self.training_home, TRAINING_MODELS)
        self.checkpoint_path = create_path(self.models_path, TRAINING_CHECKPOINT, f"checkpoints_{self.__class__.__name__}")
        self.data_path = create_path(self.training_home, TRAINING_DATA)
        self.board_path = create_path(self.logs_path, TENSORBOARD_LOGS_NAME, f"boards_{self.__class__.__name__}")
        now = datetime.now()
        self.now = str(now.strftime("%Y-%d-%b_%Hh_%Mm_%S"))
        
        self.board = None
        if self.use_tensorboard:
            self.board_name = f"board_{self.now}_{self.name}"
            self.board = SummaryWriter(join(self.board_path, self.board_name))
        
        self.epoch = 0
        self.logs = None
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.initialize = False
    
    def save(self, path : str):
        path = path.split('.')[0]
        torch.save(self, path+".pt")
        
    def checkpoint(self, logs : list, epoch : int = 0):
        self.epoch = epoch
        self.logs = logs
        checkpoint_path = join(self.checkpoint_path, f"checkpoint_{self.name}_{self.now}")
        self.save(checkpoint_path)
    
    def set_device(self, device_str : str):
        if device_str == "cpu":   
            self.device = device("cpu")
        elif device_str == "cuda":
            self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.to(self.device)
        
    def init(self, loss, optimizer, device_str = "cuda", metrics : list = None):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.set_device(device_str)
        self.initialize = True
    
    def summary(self, batchsize=1):
        print(self.parameters)     
    
    def fit(self,
            dataset : WTSeriesTraining,
            n_epochs : int, 
            batchsize : int, 
            shuffle : bool = True, 
            talkative : bool = True,
            evaluate : bool = False):
        
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
         
        train_loader = train_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)
        validation_loader = validation_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)
        test_loader = test_data.get_dataloader(batch_size=batchsize, shuffle=shuffle)
        
        logs = []
        for num_epoch in range(self.epoch, n_epochs):
            log = OrderedDict()
            pb = None
            if talkative:
                print(f"Epoch {num_epoch+1} / {n_epochs} :")
                pb = ProgressBar(len(train_loader))  
                
            log = self.fit_one_epoch(train_loader, log, talkative, pb)
            log = self.validate(validation_loader, log)
            
            if self.use_tensorboard:
                for l in log:
                    self.board.add_scalar(l, log[l], num_epoch)
                self.board.close()
            
            logs.append(log)
            
            if talkative:
                pb.close(log_to_message(log))

            if self.use_checkpoint:
                self.checkpoint(logs, num_epoch)
               
        log_eval = None     
        if evaluate:
            log_eval = self.validate(test_loader, OrderedDict(), validation_type = "test")
            
        return logs, log_eval
        
    def fit_one_epoch(self, 
                      dataloader : DataLoader, 
                      log : OrderedDict, 
                      talkative : bool,
                      pb : ProgressBar = None):
        self.train()
        log, labels, predictions = self._run_model(dataloader, log, "train", talkative)#, pb) 
        if self.metrics:
            add_metrics_to_log(log, self.metrics, labels, predictions)
        return log
    
    def validate(self,
                 dataloader : DataLoader, 
                 log : OrderedDict, 
                 validation_type : str = "validation"):
        
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
                   pb : ProgressBar = None):
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

    def transform_x(self, x):
        return x
    
    def transform_y(self, y):
        return y

    def predict(self, batch_inputs):
        batch_inputs = batch_inputs.to(self.device)
        return self(batch_inputs).data