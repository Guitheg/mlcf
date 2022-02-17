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

To create the project repository use:

*Note: the ```path_to_dir``` value is set by default to the current directory.*

```bash
mlfc --create-userdir --userdir <path_to_dir>
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

More details explanation are coming soon...
