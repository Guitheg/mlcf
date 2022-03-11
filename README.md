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
mlcf_home
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
│       │   trainer_lstm_a.py
│       │   trainer_lstm_b.py
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
mlcf --create-userdir --userdir <path_to_dir>
```

or to create it in the current repository:  

```bash
mlcf --create-userdir
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
usage: mlcf_home build-dataset [-h] --rawdata-dir RAWDATA_DIR --dataset-name
                               DATASET_NAME [--pairs PAIRS [PAIRS ...]]
                               [--timeframes {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} ...]]
                               --input-width WIDTH [--target-width WIDTH]
                               [--offset WIDTH] [--window-step STEP]
                               [--n-interval NUMBER] [--index-column NAME]
                               [--prop-tv PERCENTAGE] [--prop-v PERCENTAGE]
                               [--indices INDICE [INDICE ...]]
                               [--preprocess FUNCTION NAME] [--merge-pairs]
                               [--standardize] [--n-category N_CATEGORY]
                               [--unselected-columns {open,low,high,close,volume} [{open,low,high,close,volume} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --rawdata-dir RAWDATA_DIR
                        The directory of the raw data used to build the
                        dataset. It will uses every file in the given
                        directory
  --dataset-name DATASET_NAME
                        The name of the dataset file which will be created
  --pairs PAIRS [PAIRS ...]
                        The list of pairs from which the dataset is build.
                        They are space-separated. (Default : BTC/BUSD)
  --timeframes {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} ...]
                        The list of timeframes from which the dataset is
                        build. They are space-separated. (Default : 1d)
  --input-width WIDTH   The width of the input part in the sliding window. Can
                        also be seen as the sequence length of a neural
                        network.
  --target-width WIDTH  The width of the target part in the sliding window
                        (Default: 1)
  --offset WIDTH        The width of the offset part in the sliding window
                        (Default: 0)
  --window-step STEP    The step between each sliding window (Default: 1)
  --n-interval NUMBER   The number of intervals by which the data will be
                        divided. It allows to not have test and validation
                        part just at the end (but at the end of each part)
                        without having an overlap between the train and the
                        evaluations parts. (Default: 1)
  --index-column NAME   Name of the index column (commonly the time) (Default:
                        'date')
  --prop-tv PERCENTAGE  The proportion of the test and validation part union
                        from the data (Default: 0.1)
  --prop-v PERCENTAGE   The proportion of the validation part from the test
                        and the validation par union (Default: 0.3)
  --indices INDICE [INDICE ...]
                        List of indicators we want to add in the data
                        (Optionnal)
  --preprocess FUNCTION NAME
                        List of pre processing function we want to use to pre
                        process the data. Note: it's use independtly on each
                        window
  --merge-pairs         Merge the pairs together in order to extend the number
                        of features.
  --standardize         Standardize all the given dimension of the dataset
  --n-category N_CATEGORY
                        Give a number of category in order to balance number
                        of returns category in the training part of the
                        dataset.
  --unselected-columns {open,low,high,close,volume} [{open,low,high,close,volume} ...]
                        List of unselected features. (such as 'low' price for
                        example)
```
usage example:

Build a dataset named SimpleDataset from a raw_data which is in $HOME/Documents/data, with an input width of 55, and the default parameters, with the BTC/BUSD pair and a timeframe of '1d':  

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

Build a dataset named DatasetETHBTC which have merged BTC and ETH features with a timeframe of one hour.

```bash
mlcf build-dataset --rawdata-dir ~/Documents/data --dataset-name DatasetETHBTC --input-size 55 --pairs BTC/BUSD ETH/BUSD --timeframes 1h --merge-pairs
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

- wtst_data : it is the WTSTraining use for the training.

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

- ### WTSeries  
    - ```WTSeries(window_width, raw_data (optional), window_step (optional) (default: 1))```  
    WTSeries for Windowed Time Series is a list of window (dataframe) extract from a time series data.
         - ```make_common_shuffle(other: WTSeries)```  
         perform a common shuffle between the self WTSeries and an other. (it's used to shuffle target and inputs together)
         - ```get_features() -> List[str]```  
         Return the list of features (name of columns)
         - ```width() -> int```  
         Return the width of the window
         - ```ndim() -> int```  
         Return the number of features. (the number of columns)
         - ```shape() -> (int, int)```  
         Return the shape of a window : width, ndim
         - ```size() -> (int, int, int)```  
         Return the full shape/size of the wtseries : n_windows, width, ndim
         - ```is_empty() -> bool```  
         Return True if the wtseries is empty
         - ```add_data(data: DataFrame)```  
         From a raw data, perform the sliding window operation in order to add the windowed data in the list of window of the wtseries.
         - ```add_one_window(window: DataFrame)```  
         Add a window to the list of window
         - ```add_window_data(window_data: WTSeries)```  
         Add all the window of the given window_data wtseries to the current wtseries.
- ### WTSTraining
    - ```WTSTraining(input_width: int, target_width: int, partition: str, features: List[str], index_column: str, project: MlcfHome)```  
    WTSTraining is used to divide the data into windows, to create input windows and target window, to create train part, validation part and test part. WTSTraining allow to handle time series data in a machine learning training. The component of the WTSTraining is the WTSeries which is a list of window extract from window sliding of a time series data.
        - ```set_partition(partition: str)```  
        Set the partition : 'train', 'validation' or 'test'.
        - ```add_time_serie(dataframe: DataFrame, prop_tv: float, prop_v: float, do_shuffle: bool, n_interval: int, offset: int, window_step: int, preprocess: WTSeriesPreProcess)```  
        This function perform the sliding window operation on the given dataframe to add train, validation and test data in the WTSTraining.
            - ```prop_tv```: is the percentage of the evaluation part (the union of test and validation part)
            - ```prop_v```: is the percentage of validation part amoung the evaluation part
            - ```do_shuffle```: if true then perform a shuffle of the wtseries
            - ```n_interval```: the number of interval from which the raw data will be divide. (to avoid to have test and val only at the end of the raw_data)
            - ```offset```: the width between the input window and the target window
            - ```window_step```: the step between each sliding window
        - ```__call__() -> WTSeries, WTSeries```  
        Return the input and target WTSeries of the current partition  
        Example of use :
        ```python
        myWtstraining = WTSTraining(...)
        myWtstraining.add_time_serie(...)
        inputs, target = myWtstraining() # call : __call__() function
        ```
        - ```width() -> int, int```  
        Return the width of the input and of the target
        - ```ndim() -> int```  
        Return the number of features
        - ```copy(filter: List[str | bool] (optional) -> WTSTraining```  
        Return the same WTSTraining and if filter is set then return the filtered WTSTraining
- ### WTSTrainingDataset
    - ```WTSTrainingDataset(dataset_path: Path, ...)```  
    WTSTrainingDataset specify WTSTraining. The new argument is dataset_path which lead to the dataset file. It work the same way as WTSTraining.
- ### WTSeriesPreProcess  
    Available WTSeriesPreprocess:  
    - Identity
    - AutoNormalize
- ### Indice  
    Available indices:  
    - ADX
    - Plus Directional Indicator / Movement (P_DIM)
    - Minus Directional Indicator / Movement (M_DIM)
    - Aroon, Aroon Oscillator (AROON)
    - Awesome Oscillator (AO)
    - Keltner Channel (KC)
    - Ultimate Oscillator (UO)
    - Commodity Channel Index (CCI)
    - RSI
    - Inverse Fisher transform on RSI (FISHER_RSI)
    - Stochastic RSI (STOCH_SLOW)
    - STOCH_FAST
    - STOCH_SLOW
    - MACD
    - MFI
    - ROC
    - Bollinger Bands (BBANDS)
    - Bollinger Bands - Weighted (W_BBANDS)
    - EMA - Exponential Moving Average (EMA)
    - SMA - Simple Moving Average (SMA)
    - Parabolic SAR (SAR)
    - TEMA - Triple Exponential Moving Average (TEMA)
    - Hilbert Transform Indicator - SineWave (HT)
    - Percent growth (return) (PERCENTGROWTH)
    - SMA1 (SMA1)
    - Log of SMA1 (LNSMA1)
    - Return (RETURN)
    - Candle direction (CANDLE_DIR)
    - Candle height (CANDLE_HEIGHT)
    - Stats : variance, standard deviation, median, mean, min, max, kurtosis, skewness (STATS)
    - Pattern Recognition Indicators (PATTERNS)

- ### datasetools  
    Available function to handle WTSTrainingDataset:  
    - 
    ```python
    write_wtstdataset_from_raw_data(
        project: MlcfHome,
        rawdata_dir: Path,
        dataset_name: str,
        pairs: List[str],
        timeframes: List[str],
        input_width: int,
        target_width: int,
        offset: int,
        window_step: int,
        n_interval: int,
        index_column: str,
        prop_tv: float,
        prop_v: float,
        indices: List[Indice],
        preprocess: WTSeriesPreProcess,
        merge_pairs: bool
       )
    ```
    This function create a .wtst WTSTrainingDataset file for a ```rawdata_dir``` directory.  

### AiTools

The aitools library provides :

- ### SuperModule  
    SuperModule is an abstract class which allow to have a files, logs, and checkpoints managements.  
    Here the function implemented by SuperModule:  
    - ```init(loss, optimizer, device_str, metrics, training_name, project)```  
    The init function allows to initialize a training
    - ```init_load_checkpoint(loss, optimizer, device_str, metrics, project, training_name, resume_training)```  
    This function allows to load the last checkpoint of the training having the same name given the project
    - ```fit(dataset: WTSTraining, n_epochs: int, batchsize: int, shuffle: bool, evaluate: bool, checkpoint: bool)```  
    The fit function allows to perform a training
    - ```predict(bacth_input: Tensor)```  
    Given a batch of input give the prediction batch in output
    - ```summary()```  
    Return the summary of the current module neural network
    
    
    To write a module from this super module you need to specify : 
    - ```__init__```  
    Here it allows to define the network and initialize the shape of the input of the network 
    - ```forward```  
    Here it allows to define the feed forwarding of the neural network
    - ```transform_x``` 
    Here it allows to define all shape transform the input data will go through
    - ```transform_y``` 
    Here it allows to define all shape transform the target data will go through  
    

### EnvTools

The envtools library provides :

- ### MlcfHome
    Attributes available:  
    - ```home_name```: the name of the directory name
    - ```dir_prgm```: the path of the path from where the programm has been executed
    - ```dir```: the path to the project directory
    - ```data_dir```: the path where the WTSTrainingDataset are stored
    - ```trainer_dir```: the path where trainers are stored
    - ```models_dir```: the path where models are stored
    - ```log```: logger of the project, will write every log in the files of the projects
    - ```cfg```: link to the parameters.ini file stored in the project directory
    - ```id```: information in text about the project


*More details explanation are coming soon...*
