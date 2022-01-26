SHELL:=/bin/bash
UNAME:=$(shell uname)
ENV_NAME = cgrbi
PYTHON_VERSION:=3.9

CONDA_HOME = $(HOME)/miniconda3
CONDA_BIN_DIR = $(CONDA_HOME)/bin
CONDA = $(CONDA_BIN_DIR)/conda

ENV_DIR = $(CONDA_HOME)/envs/$(ENV_NAME)
ENV_BIN_DIR = $(ENV_DIR)/bin
ENV_LIB_DIR = $(ENV_DIR)/lib
PYTHON = $(ENV_BIN_DIR)/python
PIP=$(ENV_BIN_DIR)/pip3

setup-run: setup run
setup: conda-install conda-env-create packages-install

####################################################################################################
################# Download and install conda (if it's not installed yet) ###########################
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
ifeq (,$(shell which conda)) # check if conda is installed
ifeq ($(OS),Windows_NT)
CONDASH:=Miniconda3-latest-Windows-x86_64.exe
else
ifeq ($(UNAME),Darwin)
CONDASH:=Miniconda3-latest-MacOSX-x86_64.sh
endif # MacOS
ifeq ($(UNAME),Linux)
CONDASH:=Miniconda3-latest-Linux-x86_64.sh
endif # Linux
endif # not Windows
CONDAURL:=https://repo.anaconda.com/miniconda/$(CONDASH)

conda-install: # If conda is not installed
	@echo ">>> Setting up miniconda3..."
	@echo ">>> Downloading $(CONDAURL) : ... "
	@wget "$(CONDAURL)" && \
	bash "$(CONDASH)" -b -p $(HOME)/miniconda3 && \
	rm -f "$(CONDASH)"

else # Else it is installed
conda-install:
	@echo ">>> miniconda3 is installed in $(CONDA_HOME)"
endif # not HAS_CONDA


####################################################################################################
############################## Check if the conda environment exist ################################
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
conda-env-create:
ifeq (,$(shell conda env list | grep $(ENV_NAME))) # check if the conda environment exist
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y
	conda install -n $(ENV_NAME) pip -y
else
	@echo ">>> $(ENV_NAME) conda environment exist"
endif

####################################################################################################
################### Check dependencies and update/install packages #################################
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

packages-install:
#conda update -n base -c defaults conda -y

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~REQUIREMENTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#---CONDA INSTALLATION
#	conda install -n $(ENV_NAME) pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
#	conda install -n $(ENV_NAME) -c conda-forge -c pytorch u8darts-all -y


#---PIP INSTALLATION
	$(PIP) install ritl

run:
	$(PYTHON) main.py