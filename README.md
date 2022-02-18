# MLCF - Machine Learning Toolkit for Cryptocurrency Forecasting  

This library provides tools for cryptocurrency forecasting and trade decision making.  
Among which :

- data acquisition to download historical cryptocurrency data.

- shaping and preprocessing data for time series machine learning.

- directory and file management to help machine learning training (models versionning, checkpoint, logs, visualization, etc.).

This library doesn't provide models or an end-to-end trade bot. It is only providing tools to make easier the data acquisition, the data analysis, the forecasting and decision making training. However, MLCF provide a file management which allows to make an end-to-end model easier.

In addition to using modules' functions, we can use MLCF as a python module :

```bash
python -m mlcf <list_of_arguments>
```

---

## Installation

OS officially supported:  

- **Linux**  

Python version officially supported:  

- **3.7**  

- **3.8**  

- **3.9**

To succeed the installation, it needs to install some dependencies, which are:

- the TA-LIB C library
- PyTorch v1.10.2 (cuda 11.3)

---

### Installation for Linux (python v3.7, v3.8, v3.9)

- TA-LIB C library installation:  

*Note: the talib-install.sh file and the ta-lib-0.4.0-src.tar.gz archive will be downloaded on your PC. They can be manually deleted at the end of the installation.*  

```bash
wget https://raw.githubusercontent.com/Guitheg/mlcf/main/build_helper/talib-install.sh
sh talib-install.sh
```

- Pytorch with cuda 11.3 installation:  

```bash
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

- MLCF package

```bash
pip install mlcf
```

---

## MLCF module usage

In this part, we will introduce all the usage of MLCF module: the mlcf_home repository, the build-dataset command and the train command. The project is currently in development and new features will come soon.  

---

### mlcf_home repository  

Before all, we need to create an user directory where all of our project will be stored : logs, checkpoint, models, trainers, parameters file etc.  
The arborescence of a mlfc_home repository is as follow:  

```bash
mlfc_home
│   parameters.ini
├───data
│   │   dataset_a.wst
│   │   dataset_b.wst
│   └   ...
│
├───logs
│   ├─  debugMessages
│   └─  infoMessages
│
├───ml
│   ├───models
│   │   │   lstm.py
│   │   │   cnn_n_lstm.py
│   │   └─  ...
│   │
│   └───trainers
│       |   trainer_lstm_a.py
│       |   trainer_lstm_b.py
│       └─  ...
│
└───Training
    ├───TrainingInfo.csv
    ├───boards
    │       ├───boards_TrainingA
    │       ├───boards_TrainingB
    │       │   ├───board_17feb_17h_TrainingB
    │       │   ├───board_18feb_19h_TrainingB
    │       │   └─  ...
    │       └─  ...
    │   
    └───checkpoints
            ├───checkpoints_TrainingA
            ├───checkpoints_TrainingB
            │   ├───checkpoint_TrainingA_17feb_17h.pt
            │   ├───checkpoint_TrainingB_18feb_19h.pt
            │   └─  ...
            └─  ... 
        
```

To create the mlcf_home repository in a choosen repository, use:

```bash
mlfc --create-userdir --userdir <path_to_dir>
```

or to create it in the current repository:  

```bash
mlfc --create-userdir
```

The list of files in the mlcf_home:  

- paramters.ini : is a configuration file (not very useful for now). Later it will help the user to specify some wanted configurations.

- logs : this repository have all the logs file which relate what happened during a session

- ml : it contains models and trainers. models contains all your personnal neural network model you create with PyTorch and trainers contains all the scripts that can be executed by the train command. (It's kinda a main.py).

- Training : it is create when a train run for the first time. It is the repository that will contains the logs and checkpoint of all training.

- TrainingInfo.csv : It's a file which gives information about the past trainings. (The training name, the model used, the last lost, the current checkpoint, etc.)

- boards : it contains all the tensorboard logs. So they can be stream with tensorboard to visualize the evolution of the loss and other metrics.

- checkpoints : it contains all the checkpoints of your model during a training.

---

### build-dataset command

To build a dataset we need to download data. For now MLCF cannot do it alone.
To perform the downloading we commonly use [freqtrade](https://github.com/freqtrade/freqtrade/) library. To know more about it see [the freqtrade home documentation](https://www.freqtrade.io/en/stable/) and [how to download data with freqtrade](https://www.freqtrade.io/en/stable/data-download/).  
However you can use any OHLCV data download with your own way.

**Important: The data used by MLCF are OHLCV (incompatible with any other kind). The file format of the data MUST be a '.json'. The expected JSON string format is : [columns -> [index -> value]].**

build-dataset usage:

```
mlcf_home build-dataset [-h] --rawdata-dir RAWDATA_DIR --dataset-name DATASET_NAME -in WIDTH [-tar WIDTH] [--offset WIDTH] [--window-step STEP] [--n-interval NUMBER] [--index-column NAME] [--prop-tv PERCENTAGE] [--prop-v PERCENTAGE] [--indices INDICE [INDICE ...]] [--preprocess FUNCTION NAME]

-h, --help                      show this help message and exit

--rawdata-dir RAWDATA_DIR       The directory of the raw data used to 
                                build the dataset. It will uses every 
                                file in the given directory

--dataset-name DATASET_NAME     The name of the dataset file which will 
                                be created

-in WIDTH, --input-size WIDTH   The width of the input part in the 
                                sliding window. Can also be seen as the 
                                sequence length of a neural network.

-tar WIDTH, --target-size WIDTH The width of the target part in the 
                                sliding window (Default: 1)

--offset WIDTH                  The width of the offset part in the 
                                sliding window (Default: 0)

--window-step STEP              The step between each sliding window 
                                (Default: 1)

--n-interval NUMBER             The number of intervals by which the 
                                data will be divided. It allows to not 
                                have test and validation part just at 
                                the end (but at the end of each part) 
                                without having an overlap between
                                the train and the evaluations parts. 
                                (Default: 1)

--index-column NAME             Name of the index column (commonly the 
                                time) (Default: 'date')

--prop-tv PERCENTAGE            The proportion of the test and 
                                validation part union from the data 
                                (Default: 0.1)

--prop-v PERCENTAGE             The proportion of the validation part 
                                from the test and the validation par 
                                union (Default: 0.3)

--indices INDICE [INDICE ...]   List of indicators we want to add in the 
                                data (Optionnal)

--preprocess FUNCTION NAME      List of pre processing function we want  
                                to use to pre process the data. Note: 
                                it's use independtly on each 
                                window              
```

usage example:

Build a dataset named SimpleDataset from a raw_data which is in $HOME/Documents/data, with an input width of 55, and the default parameters:

```bash
mlcf build-dataset --rawdata-dir ~/Documents/data --dataset-name SimpleDataset --input-size 55
```

Build a dataset named DatasetRSI which have the STOCH and RSI indicators :  

```bash
mlcf build-dataset --rawdata-dir ~/Documents/data --dataset-name DatasetRSI --input-size 55 --indices STOCH_SLOW RSI :
```

Build a dataset named DatasetRSInorm which have the RSI indicator and a preprocessing apply on each windows named AutoNormalize :

```bash
mlcf build-dataset --rawdata-dir ~/Documents/data --dataset-name DatasetRSInorm --input-size 55 --indices RSI --preprocess AutoNormalize
```

---

### train command

```
usage: mlcf_home train [-h] --trainer-name NAME [--training-name NAME] --dataset-name NAME [--param PARAM [PARAM ...]]

optional arguments:
  -h, --help                show this help message and exit
  --trainer-name NAME       The name of the trainer file. IMPORTANT: 
                            the command call the method: train() inside 
                            the file given by the trainer file name.
  --training-name NAME      The name of the training name, useful for 
                            logging, checkpoint etc.
  --dataset-name NAME       The dataset name use for the training
  --param PARAM [PARAM ...] The list of arguments for the trainer.
                            IMPORTANT: The list must be in the form: key1=value1 key2=value2 key3=elem1,elem2,elem3
```
*Note: param argument doesn't work yet*

usage example:

Begin a training with a trainer script my_trainer.py, the training name is MyFirstTraining and the dataset DatasetRSInorm.wts:

```
mlcf train --trainer-name my_trainer --training-name MyFirstTraining --dataset-name DatasetRSInorm
```

Information about trainers script:  
A python trainer script is a script with the definition of a function called train which take 3 required arguments and 2 optionnals:  

- project : it is our project home class. It manage the logs, checkpoints, it allow to have a training manager.

- training_name : it is the name of this training.

- wtst_data : it is the WTSeriesTraining use for the training.

- args and kwargs : will work with the argument param in order to pass personnal parameters.

Here an example:  

```python
import torch

from ritl import add  
add(__file__, "..")  # ritl.add(__file__, "..") allows to import from 
                     # .../mlcf_home/ml so models.lstm can be imported in a 
                     # relative way.

from models.lstm import LSTM

def train(
    project,
    training_name,
    wtst_data,
    *args,
    **kwargs,
):
    list_columns = wtst_data.features
    lstm = LSTM(30, list_columns)
    lstm.init(torch.nn.L1Loss(),
              torch.optim.SGD(lstm.parameters(), lr=0.01, momentum=0.9),
              training_name=training_name,
              project=project)
    lstm.fit(dataset=wtst_data, n_epochs=5, batchsize=20)
```

## MLCF library

In this part is introduced all the current tools of MLCF.

### Datatools

The datatools library provides :

- WTSeries
- WTSeriesTraining
- WTSeriesTensor
- WTSeriesPreProcess
- Indice
- datasetools

### AiTools

The aitools library provides :

- SuperModule
- TrainingManager

### EnvTools

The envtools library provides :

- ProjectHome


*More details explanation are coming soon...*
