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

## Installation

A simple pip install is require to install MLCF.

```bash
pip install mlcf
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
